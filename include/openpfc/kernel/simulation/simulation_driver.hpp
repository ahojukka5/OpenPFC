// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file simulation_driver.hpp
 * @brief Thin time loop over `Time` plus a physics `step` (M10).
 *
 * @details
 * Same ordering as ETD session `run()`:
 *
 * 1. while not done
 * 2. if increment is 0: `on_start`, then `on_save` when `do_save()`
 * 3. `Time::next()`
 * 4. `apply_conditions`
 * 5. `step(current time)`
 * 6. `on_save` when `do_save()`
 *
 * This driver does not own writers, diagnostics, or checkpoints. Callers
 * pass those as hooks. `SnapshotSeries` and `DiagnosticsSeries` stay
 * usable without this driver.
 *
 * `run` always accepts the fixed `dt`. `run_attempts` is the adaptive
 * contract: a rejected attempt does not commit `Time`, so a save or
 * checkpoint hooked to `on_save` does not run for it. `apply` may stage
 * boundary data for the attempt; rolling the field back is the stepper's
 * job.
 */

#include <stdexcept>
#include <utility>

#include <openpfc/kernel/simulation/time.hpp>

namespace pfc {
class SimulationState;
}

namespace pfc::sim {

struct NoopHook {
  template <class... Args> constexpr void operator()(Args &&...) const noexcept {}
};

/// Result of one adaptive attempt. `accepted == false` leaves `Time` put.
struct StepDecision {
  bool accepted{true};
};

/**
 * Drive @p time to completion. @p step is `void(double t)` (accepted time
 * after `next()`). Optional hooks take `Time &` / `const Time &`.
 */
template <class Step, class OnStart = NoopHook, class Apply = NoopHook,
          class OnSave = NoopHook>
void run(Time &time, Step &&step, OnStart &&on_start = {}, Apply &&apply = {},
         OnSave &&on_save = {}) {
  while (!pfc::time::done(time)) {
    if (pfc::time::increment(time) == 0) {
      on_start(time);
      if (pfc::time::do_save(time)) {
        on_save(time);
      }
    }
    pfc::time::next(time);
    apply(time);
    step(pfc::time::current(time));
    if (pfc::time::do_save(time)) {
      on_save(time);
    }
  }
}

/**
 * @brief Adaptive driver. A rejected attempt does not commit time and does
 *        not call @p on_save. Consecutive rejections stop with an exception
 *        so a stepper that never accepts cannot spin.
 *
 * @p step is invoked with an active attempt. It returns whether that
 * attempt becomes the next accepted state.
 */
template <class Step, class OnStart = NoopHook, class Apply = NoopHook,
          class OnSave = NoopHook>
void run_attempts(Time &time, Step &&step, OnStart &&on_start = {},
                  Apply &&apply = {}, OnSave &&on_save = {},
                  int max_consecutive_rejections = 10000) {
  if (max_consecutive_rejections < 1) {
    throw std::invalid_argument(
        "run_attempts: max_consecutive_rejections must be positive");
  }
  if (pfc::time::done(time)) return;
  if (pfc::time::increment(time) == 0) {
    on_start(time);
    if (pfc::time::do_save(time)) on_save(time);
  }
  int streak = 0;
  while (!pfc::time::done(time)) {
    time.begin_attempt(time.get_dt());
    apply(time);
    const StepDecision decision = step(time);
    if (!decision.accepted) {
      time.reject_attempt();
      if (++streak >= max_consecutive_rejections) {
        throw std::runtime_error("run_attempts: the stepper rejected every attempt");
      }
      continue;
    }
    streak = 0;
    time.commit_attempt();
    if (pfc::time::do_save(time)) on_save(time);
  }
}

/**
 * Non-owning bundle of `Time` plus optional `SimulationState`. Call `run`
 * with the same hook pack as the free function.
 */
class SimulationDriver {
public:
  explicit SimulationDriver(Time &time, SimulationState *state = nullptr) noexcept
      : m_time(&time), m_state(state) {}

  template <class Step, class OnStart = NoopHook, class Apply = NoopHook,
            class OnSave = NoopHook>
  void run(Step &&step, OnStart &&on_start = {}, Apply &&apply = {},
           OnSave &&on_save = {}) {
    pfc::sim::run(*m_time, std::forward<Step>(step), std::forward<OnStart>(on_start),
                  std::forward<Apply>(apply), std::forward<OnSave>(on_save));
  }

  /**
   * @brief Adaptive loop. @p step is `StepDecision(Time &)` and is called
   *        while an attempt is active. `get_accepted_time()` is the start
   *        of the attempt and `get_attempted_dt()` is the clipped interval.
   */
  template <class Step, class OnStart = NoopHook, class Apply = NoopHook,
            class OnSave = NoopHook>
  void run_attempts(Step &&step, OnStart &&on_start = {}, Apply &&apply = {},
                    OnSave &&on_save = {}, int max_consecutive_rejections = 10000) {
    pfc::sim::run_attempts(*m_time, std::forward<Step>(step),
                           std::forward<OnStart>(on_start),
                           std::forward<Apply>(apply), std::forward<OnSave>(on_save),
                           max_consecutive_rejections);
  }

  [[nodiscard]] Time &time() noexcept { return *m_time; }
  [[nodiscard]] const Time &time() const noexcept { return *m_time; }
  [[nodiscard]] SimulationState *state() noexcept { return m_state; }
  [[nodiscard]] const SimulationState *state() const noexcept { return m_state; }

private:
  Time *m_time{};
  SimulationState *m_state{};
};

} // namespace pfc::sim

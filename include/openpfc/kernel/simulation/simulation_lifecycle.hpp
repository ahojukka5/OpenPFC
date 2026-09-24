// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file simulation_lifecycle.hpp
 * @brief Compose Time, named fields, and IC/BC modifiers around `run`.
 *
 * @details
 * A custom stepper does not need `SpectralETDSession`. This type owns the
 * clock and the modifier lists. The application owns the physics, the
 * fields, and any scientific observer.
 *
 * `SnapshotSeries` and `DiagnosticsSeries` stay independent. Call them
 * from the save observer, which runs only when `Time::do_save()` is true,
 * including the initial state at `t0`.
 *
 * `set_accepted_step_hook` runs after every accepted physical step and
 * not at `t0`. Checkpoint cadence belongs in that hook, for example
 * `CheckpointService::maybe_save`. A rejected attempt calls neither hook.
 * `prepare_stage` applies boundary modifiers at a substep time and does
 * not save, checkpoint, or advance `Time`.
 *
 * The whole-step `apply` hook also applies those boundary modifiers, once
 * per attempt, before `step`. A rejected attempt can stage boundary data
 * and still does not save or notify the accepted-step hook.
 *
 * Profiling is optional. A non-null `ProfilingSession` frames each call
 * of the step callback, including a rejected attempt, because that work
 * ran. It does not frame `prepare_stage` or the accepted-step hook.
 */

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/profiling/session.hpp>
#include <openpfc/kernel/simulation/apply_field_modifier.hpp>
#include <openpfc/kernel/simulation/simulation_context.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace pfc::sim {

class SimulationLifecycle {
public:
  SimulationLifecycle(const SimulationLifecycle &) = delete;
  SimulationLifecycle &operator=(const SimulationLifecycle &) = delete;
  SimulationLifecycle(SimulationLifecycle &&) = delete;
  SimulationLifecycle &operator=(SimulationLifecycle &&) = delete;

  /**
   * @brief Clock plus the communicator modifiers and checkpoints share.
   *
   * @p saveat in @ref schedule follows `Time`: a positive value is the
   * output spacing. A non-positive value uses `t1 - t0`, so the initial
   * state and the final state still save.
   */
  SimulationLifecycle(Time time, MPI_Comm comm)
      : m_time(std::move(time)), m_context(comm) {}

  [[nodiscard]] static Time schedule(double t0, double t1, double dt,
                                     double saveat) {
    const double span = t1 - t0;
    const double cadence = saveat > 0.0 ? saveat : span;
    return Time({t0, t1, dt}, cadence);
  }

  [[nodiscard]] Time &time() noexcept { return m_time; }
  [[nodiscard]] const Time &time() const noexcept { return m_time; }
  [[nodiscard]] const SimulationContext &context() const noexcept {
    return m_context;
  }

  /// Non-owning. @p field must outlive this lifecycle. Names are unique.
  template <class MemorySpace>
  void bind_field(std::string name, data::Field<double, MemorySpace> &field) {
    if (find_field(name) != nullptr) {
      throw std::invalid_argument("SimulationLifecycle: duplicate field " + name);
    }
    m_fields.push_back(BoundField{
        std::move(name), [&field, this](FieldModifier &modifier, double when) {
          apply_field_modifier(modifier, field, when, &m_context);
        }});
  }

  void add_initial_condition(std::string field,
                             std::unique_ptr<FieldModifier> modifier) {
    require_modifier(field, modifier.get());
    m_initial.push_back(NamedModifier{std::move(field), std::move(modifier)});
  }

  void add_boundary_condition(std::string field,
                              std::unique_ptr<FieldModifier> modifier) {
    require_modifier(field, modifier.get());
    m_boundary.push_back(NamedModifier{std::move(field), std::move(modifier)});
  }

  /// Apply initial modifiers once. A later `run` does not apply them again.
  void apply_initial_conditions() {
    if (m_initial_applied) return;
    apply_list(m_initial, 0.0);
    m_initial_applied = true;
  }

  /// Whole-step boundary application at @p time.
  void apply_boundary_conditions(double time) { apply_list(m_boundary, time); }

  /**
   * @brief Boundary application for one internal stage of the stepper.
   *
   * Does not call the save observer, the accepted-step hook, or the
   * profiler, and does not change `Time`.
   */
  void prepare_stage(double stage_time) { apply_boundary_conditions(stage_time); }

  /// Output cadence. Includes the initial state when `do_save()` is true.
  void set_save_observer(std::function<void(const Time &)> observer) {
    m_observer = std::move(observer);
  }

  /**
   * @brief Called once after each accepted physical step, not at `t0`.
   *
   * Independent of `do_save()`. Compose `CheckpointService::maybe_save`
   * here so checkpoint cadence is not the output cadence. A rejected
   * attempt does not call this hook.
   */
  void set_accepted_step_hook(std::function<void(const Time &)> hook) {
    m_accepted = std::move(hook);
  }

  /// @p session null disables framing. The session is not owned.
  void set_profiling(profiling::ProfilingSession *session) noexcept {
    m_profile = session;
  }

  template <class Step> void run(Step &&step) {
    auto on_start = [this](Time &) { apply_initial_conditions(); };
    auto apply = [this](Time &now) {
      apply_boundary_conditions(pfc::time::current(now));
    };
    auto on_save = [this](const Time &now) {
      if (m_observer) m_observer(now);
    };
    auto on_accepted = [this](const Time &now) {
      if (m_accepted) m_accepted(now);
    };
    if constexpr (std::is_invocable_v<Step &, double, double>) {
      pfc::sim::run(
          m_time,
          [&](double t_now, double interval) {
            ProfileFrame frame(m_profile);
            step(t_now, interval);
          },
          on_start, apply, on_save, on_accepted);
    } else {
      pfc::sim::run(
          m_time,
          [&](double t_now) {
            ProfileFrame frame(m_profile);
            step(t_now);
          },
          on_start, apply, on_save, on_accepted);
    }
  }

  template <class Step>
  void run_attempts(Step &&step, int max_consecutive_rejections = 10000) {
    auto on_start = [this](Time &) { apply_initial_conditions(); };
    auto apply = [this](Time &now) {
      apply_boundary_conditions(pfc::time::current(now));
    };
    auto on_save = [this](const Time &now) {
      if (m_observer) m_observer(now);
    };
    auto on_accepted = [this](const Time &now) {
      if (m_accepted) m_accepted(now);
    };
    pfc::sim::run_attempts(
        m_time,
        [&](Time &now) {
          ProfileFrame frame(m_profile);
          return step(now);
        },
        on_start, apply, on_save, on_accepted, max_consecutive_rejections);
  }

private:
  struct BoundField {
    std::string name;
    std::function<void(FieldModifier &, double)> apply;
  };

  struct NamedModifier {
    std::string field;
    std::unique_ptr<FieldModifier> modifier;
  };

  struct ProfileFrame {
    profiling::ProfilingSession *session;
    explicit ProfileFrame(profiling::ProfilingSession *session_in)
        : session(session_in) {
      if (session != nullptr) session->begin_frame();
    }
    ProfileFrame(const ProfileFrame &) = delete;
    ProfileFrame &operator=(const ProfileFrame &) = delete;
    ~ProfileFrame() {
      if (session != nullptr) session->end_frame();
    }
  };

  void require_modifier(const std::string &field, FieldModifier *modifier) const {
    if (modifier == nullptr) {
      throw std::invalid_argument("SimulationLifecycle: null modifier");
    }
    if (find_field(field) == nullptr) {
      throw std::invalid_argument("SimulationLifecycle: unknown field " + field);
    }
  }

  [[nodiscard]] const BoundField *find_field(std::string_view name) const {
    for (const auto &field : m_fields) {
      if (field.name == name) return &field;
    }
    return nullptr;
  }

  void apply_list(std::vector<NamedModifier> &list, double time) {
    for (auto &item : list) {
      BoundField *field = nullptr;
      for (auto &candidate : m_fields) {
        if (candidate.name == item.field) field = &candidate;
      }
      if (field == nullptr) {
        throw std::invalid_argument("SimulationLifecycle: unknown field " +
                                    item.field);
      }
      field->apply(*item.modifier, time);
    }
  }

  Time m_time;
  SimulationContext m_context;
  std::vector<BoundField> m_fields;
  std::vector<NamedModifier> m_initial;
  std::vector<NamedModifier> m_boundary;
  std::function<void(const Time &)> m_observer;
  std::function<void(const Time &)> m_accepted;
  profiling::ProfilingSession *m_profile{nullptr};
  bool m_initial_applied{false};
};

} // namespace pfc::sim

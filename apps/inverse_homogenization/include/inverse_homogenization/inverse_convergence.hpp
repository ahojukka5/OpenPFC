// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file inverse_convergence.hpp
 * @brief Frozen-parameter convergence protocol for inverse homogenization.
 *
 * Continuation interpolates SIMP / regularization for
 * `continuation_steps` iterates. After that the problem is held fixed.
 * Convergence requires simultaneous post-projection design RMS, relative
 * objective change, and relative \(\lVert C_H\rVert_F\) change for
 * `conv_window` consecutive frozen iterates, then a verification hold.
 * `--max-steps` is a ceiling, not success. Job 22162138 still had
 * pre-projection `step_rms≈0.019` at step 300, so a J-only stop would
 * have fired while the design was moving.
 */

#include <algorithm>
#include <cmath>

namespace pfc::apps::inverse {

enum class TerminationReason {
  Running = 0,
  Converged = 1,
  MaxSteps = 2,
  ElasticityFailure = 3,
};

[[nodiscard]] inline const char *termination_name(TerminationReason r) {
  switch (r) {
  case TerminationReason::Running: return "RUNNING";
  case TerminationReason::Converged: return "CONVERGED";
  case TerminationReason::MaxSteps: return "MAX_STEPS";
  case TerminationReason::ElasticityFailure: return "ELASTICITY_FAILURE";
  }
  return "UNKNOWN";
}

struct ConvergenceConfig {
  int continuation_steps{300};
  int max_steps{5000};
  int conv_window{20};
  int verify_steps{100};
  double tol_design{1e-4};
  double tol_objective{1e-6};
  double tol_tensor{1e-4};
  double tensor_eps{1e-30};
};

[[nodiscard]] inline double continuation_fraction(int step, int continuation_steps) {
  if (continuation_steps <= 1) return 1.0;
  if (step >= continuation_steps - 1) return 1.0;
  if (step <= 0) return 0.0;
  return static_cast<double>(step) / static_cast<double>(continuation_steps - 1);
}

/// True once the step just completed used frozen continuation parameters
/// for the whole step (the first frozen step is `continuation_steps`).
[[nodiscard]] inline bool params_frozen(int step, int continuation_steps) {
  if (continuation_steps <= 0) return true;
  return step >= continuation_steps;
}

[[nodiscard]] inline double relative_objective_change(double J_prev, double J) {
  return std::abs(J - J_prev) / std::max(1.0, std::abs(J_prev));
}

[[nodiscard]] inline double relative_norm_change(double diff_norm, double prev_norm,
                                                 double eps) {
  return diff_norm / std::max(prev_norm, eps);
}

[[nodiscard]] inline bool criteria_hold(double design_rms, double dJ_rel,
                                        double dC_rel,
                                        const ConvergenceConfig &cfg) {
  return design_rms < cfg.tol_design && dJ_rel < cfg.tol_objective &&
         dC_rel < cfg.tol_tensor;
}

struct ConvergenceMetrics {
  double design_rms{0.0};
  double dJ_rel{0.0};
  double dC_rel{0.0};
  double morph_frac{0.0};
  bool quiet{false};
};

[[nodiscard]] inline ConvergenceMetrics
make_metrics(double design_rms, double J, double J_prev, double dC_norm,
             double C_prev_norm, double morph_frac, const ConvergenceConfig &cfg) {
  ConvergenceMetrics m;
  m.design_rms = design_rms;
  m.dJ_rel = relative_objective_change(J_prev, J);
  m.dC_rel = relative_norm_change(dC_norm, C_prev_norm, cfg.tensor_eps);
  m.morph_frac = morph_frac;
  m.quiet = criteria_hold(m.design_rms, m.dJ_rel, m.dC_rel, cfg);
  return m;
}

struct ConvergenceTracker {
  ConvergenceConfig cfg{};
  int quiet_count{0};
  int verify_left{0};
  bool candidate{false};
  bool verified{false};
  bool rejected_hold{false};

  /// @param step 0-based index of the iterate that just finished.
  /// @param steps_done `step + 1`.
  TerminationReason after_step(int step, bool elasticity_ok,
                               const ConvergenceMetrics &m) {
    rejected_hold = false;
    if (!elasticity_ok) return TerminationReason::ElasticityFailure;
    const bool frozen = params_frozen(step, cfg.continuation_steps);
    if (!frozen) {
      quiet_count = 0;
      candidate = false;
      verified = false;
      verify_left = 0;
    } else if (!m.quiet) {
      if (candidate) rejected_hold = true;
      quiet_count = 0;
      candidate = false;
      verified = false;
      verify_left = 0;
    } else {
      ++quiet_count;
      if (candidate) {
        --verify_left;
        if (verify_left <= 0) {
          verified = true;
          return TerminationReason::Converged;
        }
      } else if (quiet_count >= cfg.conv_window) {
        candidate = true;
        verify_left = cfg.verify_steps;
        if (verify_left <= 0) {
          verified = true;
          return TerminationReason::Converged;
        }
      }
    }
    if (step + 1 >= cfg.max_steps) return TerminationReason::MaxSteps;
    return TerminationReason::Running;
  }
};

} // namespace pfc::apps::inverse

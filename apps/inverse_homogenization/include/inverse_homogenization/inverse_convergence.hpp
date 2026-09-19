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
 *
 * Logged row \(s\) compares consecutive **accepted** states: design RMS
 * is \(\lVert h_s-h_{s-1}\rVert_2/\sqrt{N}\) measured before the
 * Allen--Cahn update; \(J\) and \(C_H\) are evaluated on that same
 * \(h_s\). The update to \(h_{s+1}\) happens after the row is written.
 * The first iterate has no predecessor and is never quiet, so a zero
 * first-row design RMS cannot seed the window. The trailing one-step
 * update is not in the certified pair; the verification hold is the
 * buffer. Unpenalized final \(C_H\) is a `# FINAL_RECOMPUTE` comment,
 * never a truncated iterate row.
 */

#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

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

/// Iterate-row schema used by CPU and HIP inverse drivers.
inline constexpr std::string_view kInverseCsvHeader =
    "step,J,J_tensor,J_volume,J_reg,volume,grey,C11,C12,nu_eff,"
    "C_fro,design_rms,dJ_rel,dC_rel,morph_frac,step_rms,grad_rms,"
    "simp_p,lambda_reg,frozen,conv_window,candidate,verified,ms,"
    "elasticity,termination";

[[nodiscard]] inline std::size_t csv_field_count(std::string_view line) {
  if (line.empty()) return 0;
  std::size_t n = 1;
  for (char c : line)
    if (c == ',') ++n;
  return n;
}

[[nodiscard]] inline bool csv_is_comment_line(std::string_view line) {
  std::size_t i = 0;
  while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
  return i < line.size() && line[i] == '#';
}

/// Empty string if every non-empty, non-comment row has the documented
/// column count. Comment lines (`# …`) are not data rows.
[[nodiscard]] inline std::string validate_inverse_csv(std::istream &in) {
  const auto ncol = csv_field_count(kInverseCsvHeader);
  std::string line;
  int n = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || csv_is_comment_line(line)) continue;
    const auto c = csv_field_count(line);
    if (c != ncol) {
      return "row " + std::to_string(n) + " has " + std::to_string(c) +
             " fields, expected " + std::to_string(ncol);
    }
    ++n;
  }
  return {};
}

struct InverseCsvRow {
  int step{0};
  double J{0}, J_tensor{0}, J_volume{0}, J_reg{0}, volume{0}, grey{0};
  double C11{0}, C12{0}, nu_eff{0}, C_fro{0};
  double design_rms{0}, dJ_rel{0}, dC_rel{0}, morph_frac{0};
  double step_rms{0}, grad_rms{0}, simp_p{1}, lambda_reg{0};
  int frozen{0}, conv_window{0}, candidate{0}, verified{0};
  double ms{0};
  int elasticity{0};
  const char *termination{"RUNNING"};
};

inline void write_inverse_csv_row(std::ostream &os, const InverseCsvRow &r) {
  os << r.step << ',' << r.J << ',' << r.J_tensor << ',' << r.J_volume << ','
     << r.J_reg << ',' << r.volume << ',' << r.grey << ',' << r.C11 << ',' << r.C12
     << ',' << r.nu_eff << ',' << r.C_fro << ',' << r.design_rms << ',' << r.dJ_rel
     << ',' << r.dC_rel << ',' << r.morph_frac << ',' << r.step_rms << ','
     << r.grad_rms << ',' << r.simp_p << ',' << r.lambda_reg << ',' << r.frozen
     << ',' << r.conv_window << ',' << r.candidate << ',' << r.verified << ','
     << r.ms << ',' << r.elasticity << ',' << r.termination << '\n';
}

[[nodiscard]] inline std::string format_inverse_csv_row(const InverseCsvRow &r) {
  std::ostringstream os;
  write_inverse_csv_row(os, r);
  std::string s = os.str();
  if (!s.empty() && s.back() == '\n') s.pop_back();
  return s;
}

} // namespace pfc::apps::inverse

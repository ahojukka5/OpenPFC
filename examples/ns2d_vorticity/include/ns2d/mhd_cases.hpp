// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file mhd_cases.hpp
 * @brief Initial conditions for 2-D incompressible MHD (#23).
 *
 * Orszag–Tang is the incompressible periodic vortex of Orszag & Tang,
 * JFM 90 (1979), not a compressible shock-tube variant.
 */

#include <cmath>
#include <stdexcept>
#include <string_view>

#include <openpfc/kernel/data/constants.hpp>

namespace ns2d {

enum class MHDCase { orszag_tang, hydro_control, force_free, alfven };

[[nodiscard]] inline const char *mhd_case_name(MHDCase c) noexcept {
  switch (c) {
  case MHDCase::orszag_tang:
    return "orszag_tang";
  case MHDCase::hydro_control:
    return "hydro_control";
  case MHDCase::force_free:
    return "force_free";
  case MHDCase::alfven:
    return "alfven";
  }
  return "unknown";
}

[[nodiscard]] inline MHDCase parse_mhd_case(std::string_view s) {
  if (s == "orszag_tang" || s == "ot") return MHDCase::orszag_tang;
  if (s == "hydro_control" || s == "hydro") return MHDCase::hydro_control;
  if (s == "force_free" || s == "diffuse") return MHDCase::force_free;
  if (s == "alfven") return MHDCase::alfven;
  throw std::invalid_argument(
      "unknown mhd case (expected orszag_tang|hydro_control|force_free|alfven)");
}

/// φ = cos x + cos y  ⇒  u = (-sin y, sin x), ω = cos x + cos y.
[[nodiscard]] inline double ot_omega(double x, double y) noexcept {
  return std::cos(x) + std::cos(y);
}

/// a = (1/2) cos 2x + cos y  ⇒  B = (-sin y, sin 2x).
[[nodiscard]] inline double ot_a(double x, double y) noexcept {
  return 0.5 * std::cos(2.0 * x) + std::cos(y);
}

[[nodiscard]] inline double force_free_a(double x, double y) noexcept {
  return std::sin(x) * std::sin(y);
}

[[nodiscard]] inline double force_free_a_exact(double x, double y, double eta,
                                               double t) noexcept {
  return force_free_a(x, y) * std::exp(-2.0 * eta * t);
}

/// Two-mode streamfunction shared by ω and a for the Alfvénic state u=B.
[[nodiscard]] inline double alfven_phi(double x, double y) noexcept {
  return std::sin(x) * std::sin(y) + 0.25 * std::sin(2.0 * x) * std::sin(2.0 * y);
}

[[nodiscard]] inline double alfven_omega(double x, double y) noexcept {
  return 2.0 * std::sin(x) * std::sin(y) +
         2.0 * std::sin(2.0 * x) * std::sin(2.0 * y);
}

} // namespace ns2d

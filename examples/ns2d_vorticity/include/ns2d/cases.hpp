// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cases.hpp
 * @brief Initial conditions and exact solutions for the 2-D NS prototype.
 *
 * Taylor–Green is the quantitative verification case. The double shear
 * layer is the visual Kelvin–Helmholtz showcase only: a pretty roll-up
 * is not evidence that the solver is correct.
 */

#include <cmath>
#include <stdexcept>
#include <string_view>

#include <openpfc/kernel/data/constants.hpp>

namespace ns2d {

enum class Case { taylor_green, shear, two_mode };

[[nodiscard]] inline const char *case_name(Case c) noexcept {
  switch (c) {
  case Case::taylor_green:
    return "taylor_green";
  case Case::shear:
    return "shear";
  case Case::two_mode:
    return "two_mode";
  }
  return "unknown";
}

[[nodiscard]] inline Case parse_case(std::string_view s) {
  if (s == "taylor_green" || s == "tg") return Case::taylor_green;
  if (s == "shear" || s == "kh" || s == "kelvin_helmholtz") return Case::shear;
  if (s == "two_mode") return Case::two_mode;
  throw std::invalid_argument(
      "unknown ns2d case (expected taylor_green|shear|two_mode)");
}

/// Exact Taylor–Green streamfunction on \f$[0,2\pi]^2\f$:
/// \f$\psi=\sin x\sin y\,e^{-2\nu t}\f$.
[[nodiscard]] inline double taylor_green_psi(double x, double y, double nu,
                                             double t) noexcept {
  return std::sin(x) * std::sin(y) * std::exp(-2.0 * nu * t);
}

/// \f$u=\partial_y\psi=\sin x\cos y\,e^{-2\nu t}\f$.
[[nodiscard]] inline double taylor_green_u(double x, double y, double nu,
                                           double t) noexcept {
  return std::sin(x) * std::cos(y) * std::exp(-2.0 * nu * t);
}

/// \f$v=-\partial_x\psi=-\cos x\sin y\,e^{-2\nu t}\f$.
[[nodiscard]] inline double taylor_green_v(double x, double y, double nu,
                                           double t) noexcept {
  return -std::cos(x) * std::sin(y) * std::exp(-2.0 * nu * t);
}

/// \f$\omega=-\nabla^2\psi=2\sin x\sin y\,e^{-2\nu t}\f$.
[[nodiscard]] inline double taylor_green_omega(double x, double y, double nu,
                                               double t) noexcept {
  return 2.0 * std::sin(x) * std::sin(y) * std::exp(-2.0 * nu * t);
}

/// Mean kinetic-energy density \f$\frac12\langle u^2+v^2\rangle=e^{-4\nu t}/4\f$.
[[nodiscard]] inline double taylor_green_ke(double nu, double t) noexcept {
  return 0.25 * std::exp(-4.0 * nu * t);
}

/// Mean enstrophy density \f$\frac12\langle\omega^2\rangle=e^{-4\nu t}/2\f$.
[[nodiscard]] inline double taylor_green_enstrophy(double nu, double t) noexcept {
  return 0.5 * std::exp(-4.0 * nu * t);
}

/**
 * @brief Bell–Mei / Minion–Brown double shear layer on \f$[0,2\pi]^2\f$.
 *
 * \f$u=\tanh(\rho(y-\pi/2))\f$ for \f$y\le\pi\f$ and
 * \f$u=\tanh(\rho(3\pi/2-y))\f$ otherwise; \f$v=\varepsilon\sin x\f$.
 * Vorticity is \f$\omega=\partial_x v-\partial_y u\f$.
 */
[[nodiscard]] inline double double_shear_omega(double x, double y, double rho,
                                               double eps) noexcept {
  const double pi = pfc::pi;
  const auto sech2 = [](double z) {
    const double c = std::cosh(z);
    return 1.0 / (c * c);
  };
  const double dudy = (y <= pi) ? rho * sech2(rho * (y - 0.5 * pi))
                                : -rho * sech2(rho * (1.5 * pi - y));
  return eps * std::cos(x) - dudy;
}

/// Two Fourier modes: a nonlinear IC for timestep-refinement tests.
/// Taylor–Green is spectrally exact, so it cannot measure temporal order.
[[nodiscard]] inline double two_mode_omega(double x, double y) noexcept {
  return std::sin(x) * std::sin(y) + 0.5 * std::sin(2.0 * x) * std::sin(2.0 * y);
}

} // namespace ns2d

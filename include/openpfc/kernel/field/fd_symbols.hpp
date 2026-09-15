// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_symbols.hpp
 * @brief Fourier symbols of the shipped even central D2 stencils.
 *
 * @details
 * `EvenCentralD2<Order>` applied to \f$e^{i\kappa x}\f$ has symbol
 * \f$\lambda_p(\kappa)\,h^2 = \bigl(c_0 + 2\sum_j c_j\cos(j\theta)\bigr)/D_2\f$
 * with \f$\theta=\kappa h\f$. The exact second derivative is \f$-\theta^2\f$.
 * Operator-error analysis needs the **defect** \f$\theta^2+\lambda_p h^2\f$,
 * which at high order and small \f$\theta\f$ is many orders below the two
 * terms; a direct subtraction is round-off.
 *
 * With \f$\delta^2=2-2\cos\theta\f$,
 * \f$\theta^2=(2\arcsin(\delta/2))^2=\sum_{m\ge1}a_m\delta^{2m}\f$ and
 * \f$a_m=2/(m^2\binom{2m}{m})\f$. The order-\f$2M\f$ stencil is that series
 * truncated at \f$m=M\f$, so the defect is the positive-term tail. Near
 * \f$\theta\to\pi\f$ the tail is slow and the defect is a large fraction of
 * \f$\theta^2\f$; `d2_symbol_defect` then uses the direct difference.
 *
 * @see fd_stencils.hpp
 * @see periodic_spectra.hpp
 */

#include <cmath>
#include <stdexcept>
#include <string>

#include <openpfc/kernel/field/fd_stencils.hpp>

namespace pfc::field::fd {

/**
 * @brief Coefficient \f$a_m=2/(m^2\binom{2m}{m})\f$ of
 *        \f$(2\arcsin(\delta/2))^2=\sum_m a_m\delta^{2m}\f$.
 *
 * Recurrence \f$a_{m+1}=a_m m^2/((m+1)\,2(2m+1))\f$ from \f$a_1=1\f$, so the
 * binomial never overflows a `double`.
 */
[[nodiscard]] inline double arcsin_series_coefficient(int m) {
  if (m < 1) throw std::invalid_argument("arcsin_series_coefficient: m >= 1");
  double a = 1.0;
  for (int i = 1; i < m; ++i) {
    a *= static_cast<double>(i) * static_cast<double>(i) /
         (static_cast<double>(i + 1) * 2.0 * static_cast<double>(2 * i + 1));
  }
  return a;
}

/**
 * @brief Symbol \f$\lambda_p(\kappa)\,h^2\f$ of a tabulated central D2 stencil.
 *
 * @param order Even FD order with a tabulated D2 stencil (2..20).
 * @param theta \f$\kappa h\f$, normally in \f$[0,\pi]\f$.
 */
[[nodiscard]] inline double d2_symbol(int order, double theta) {
  EvenCentralD2View view{};
  if (!lookup_even_central_d2(order, &view)) {
    throw std::invalid_argument("d2_symbol: no tabulated D2 stencil for order " +
                                std::to_string(order));
  }
  double sum = static_cast<double>(view.coeffs[0]);
  for (int j = 1; j <= view.half_width; ++j) {
    sum += 2.0 * static_cast<double>(view.coeffs[j]) *
           std::cos(static_cast<double>(j) * theta);
  }
  return sum / static_cast<double>(view.denom);
}

/// Exact spectral second-derivative symbol \f$-\theta^2\f$.
[[nodiscard]] inline double spectral_d2_symbol(double theta) noexcept {
  return -theta * theta;
}

/**
 * @brief Dispersion defect \f$\theta^2+\lambda_p h^2\ge 0\f$.
 *
 * The stencil under-estimates \f$|\lambda|\f$. The heat-equation mode error
 * is this quantity combined across axes.
 */
[[nodiscard]] inline double d2_symbol_defect(int order, double theta) {
  const double delta_sq = 2.0 - 2.0 * std::cos(theta);
  constexpr double kSeriesLimit = 2.5;
  if (delta_sq <= kSeriesLimit) {
    const int half_width = order / 2;
    double a = arcsin_series_coefficient(half_width);
    double power = std::pow(delta_sq, static_cast<double>(half_width));
    double total = 0.0;
    for (int m = half_width; m < half_width + 600; ++m) {
      a *= static_cast<double>(m) * static_cast<double>(m) /
           (static_cast<double>(m + 1) * 2.0 * static_cast<double>(2 * m + 1));
      power *= delta_sq;
      const double term = a * power;
      total += term;
      if (term <= 1.0e-18 * total) return total;
    }
    return total;
  }
  return theta * theta + d2_symbol(order, theta);
}

} // namespace pfc::field::fd

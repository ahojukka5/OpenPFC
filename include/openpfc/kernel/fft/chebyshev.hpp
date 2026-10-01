// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file chebyshev.hpp
 * @brief FFT-backed Chebyshev transform on a Lobatto grid.
 *
 * Values on x_j = cos(pi j / n) are an even extension of a cosine series.
 * The transform and the first derivative use one ordinary complex FFT of
 * length 2n. This is the one-dimensional oracle. It does not impose a
 * boundary condition and it does not solve a mixed Fourier–Chebyshev
 * problem.
 *
 * The FFT plans are created on each call. Do not call these functions
 * concurrently.
 */

#include <span>
#include <vector>

namespace pfc::fft {

/// Lobatto nodes from +1 down to -1. Length is `degree + 1`.
[[nodiscard]] std::vector<double> chebyshev_lobatto(int degree);

/// Coefficients a_0..a_n of the interpolant through `values`.
[[nodiscard]] std::vector<double>
chebyshev_coefficients(std::span<const double> values);

/// Values on the Lobatto grid of coefficients a_0..a_n.
[[nodiscard]] std::vector<double>
chebyshev_values(std::span<const double> coefficients);

/// First derivative on the same Lobatto grid, including both endpoints.
[[nodiscard]] std::vector<double>
chebyshev_derivative(std::span<const double> values);

} // namespace pfc::fft

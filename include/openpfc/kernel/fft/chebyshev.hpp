// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file chebyshev.hpp
 * @brief FFT-backed Chebyshev transform on a Lobatto grid.
 *
 * Values on x_j = cos(pi j / n) are an even extension of a cosine series.
 * The transform and the first derivative use one ordinary complex FFT of
 * length 2n. The mixed Laplacian differentiates one periodic Fourier
 * direction and this Chebyshev axis on a single rank. Dirichlet
 * Poisson matches both endpoint values. Neumann Poisson matches both
 * endpoint slopes and sets the integral to zero. Dirichlet Helmholtz
 * solves u'' - lambda u = f with those endpoint values. Neumann
 * Helmholtz matches both endpoint slopes. Robin Poisson mixes the
 * value and the slope at each end. Robin Helmholtz uses the same
 * mix for u'' - lambda u = f. A zero lambda reuses the Robin
 * Poisson integral. A channel wall is still a separate step.
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

/// Laplacian d²/dz² + d²/dx² on one rank.
/// `nx` uniform nodes span a periodic interval of length `period`.
/// The other axis is Chebyshev–Lobatto, stored from +1 down to -1.
/// `values` is row-major with the periodic index contiguous.
[[nodiscard]] std::vector<double>
fourier_chebyshev_laplacian(std::span<const double> values, int nx,
                            double period);

/// Solve u'' = forcing on [-1, 1], with both endpoints prescribed.
/// `forcing` is sampled on the Lobatto grid from +1 down to -1.
/// The result uses that same grid.
[[nodiscard]] std::vector<double>
chebyshev_dirichlet_poisson(std::span<const double> forcing,
                            double value_at_plus, double value_at_minus);

/// Solve u'' = forcing on [-1, 1], with both endpoint slopes prescribed.
/// The additive constant is chosen so the integral over the interval is zero.
/// `forcing` is sampled on the Lobatto grid from +1 down to -1.
/// The result uses that same grid. Incompatible slopes are rejected.
[[nodiscard]] std::vector<double>
chebyshev_neumann_poisson(std::span<const double> forcing, double slope_at_plus,
                          double slope_at_minus);

/// Solve u'' - lambda u = forcing on [-1, 1], with both endpoints
/// prescribed. A zero lambda reuses the Dirichlet Poisson solve.
/// The two highest modes are fixed by the endpoint values.
/// `forcing` is sampled on the Lobatto grid from +1 down to -1.
/// The result uses that same grid.
[[nodiscard]] std::vector<double>
chebyshev_dirichlet_helmholtz(std::span<const double> forcing, double lambda,
                              double value_at_plus, double value_at_minus);

/// Solve u'' - lambda u = forcing on [-1, 1], with both endpoint
/// slopes prescribed. A zero lambda reuses the Neumann Poisson solve.
/// The two highest modes are set by the endpoint slopes.
/// `forcing` is sampled on the Lobatto grid from +1 down to -1.
/// The result uses that same grid.
[[nodiscard]] std::vector<double>
chebyshev_neumann_helmholtz(std::span<const double> forcing, double lambda,
                            double slope_at_plus, double slope_at_minus);

/// Solve u'' = forcing on [-1, 1].
/// Each end imposes weight_value * u + weight_slope * u' = data.
/// Pure value weights reuse the Dirichlet solve. Pure slope weights
/// reuse the Neumann solve and its zero-integral gauge.
/// `forcing` is sampled on the Lobatto grid from +1 down to -1.
/// The result uses that same grid.
[[nodiscard]] std::vector<double>
chebyshev_robin_poisson(std::span<const double> forcing, double value_weight_plus,
                        double slope_weight_plus, double data_plus,
                        double value_weight_minus, double slope_weight_minus,
                        double data_minus);

/// Solve u'' - lambda u = forcing on [-1, 1].
/// Each end imposes weight_value * u + weight_slope * u' = data.
/// A zero lambda reuses the Robin Poisson solve. Pure value weights
/// reuse the Dirichlet Helmholtz solve. Pure slope weights reuse the
/// Neumann Helmholtz solve.
/// `forcing` is sampled on the Lobatto grid from +1 down to -1.
/// The result uses that same grid.
[[nodiscard]] std::vector<double>
chebyshev_robin_helmholtz(std::span<const double> forcing, double lambda,
                          double value_weight_plus, double slope_weight_plus,
                          double data_plus, double value_weight_minus,
                          double slope_weight_minus, double data_minus);

} // namespace pfc::fft

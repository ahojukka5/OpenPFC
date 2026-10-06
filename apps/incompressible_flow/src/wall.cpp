// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file wall.cpp
 * @brief Compose the channel wall from the Chebyshev Neumann solve.
 *
 * Divergence and the pressure gradient use the mixed Fourier ×
 * Chebyshev derivative. The pressure itself is the Neumann Poisson
 * problem whose endpoint slopes are the wall-normal tendency.
 */

#include <flow/wall.hpp>

#include <limits>
#include <stdexcept>

#include <openpfc/kernel/fft/chebyshev.hpp>

namespace flow {
namespace {

[[nodiscard]] std::size_t wall_index(int iz, int iy, int ix, int ny, int nx) {
  return (static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny) +
          static_cast<std::size_t>(iy)) *
             static_cast<std::size_t>(nx) +
         static_cast<std::size_t>(ix);
}

} // namespace

WallAcceleration impermeable_acceleration(std::span<const double> tendency_x,
                                          std::span<const double> tendency_y,
                                          std::span<const double> tendency_z, int nx,
                                          int ny, double period_x, double period_y) {
  if (nx < 1 || ny < 1) {
    throw std::invalid_argument("chebyshev: periodic count must be positive");
  }
  if (!(period_x > 0.0) || !(period_y > 0.0)) {
    throw std::invalid_argument("chebyshev: period must be positive");
  }
  const auto plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  if (tendency_x.size() != tendency_y.size() ||
      tendency_x.size() != tendency_z.size() || tendency_x.empty() ||
      tendency_x.size() % plane != 0) {
    throw std::invalid_argument(
        "chebyshev: tendency does not match the periodic count");
  }
  const auto lines = tendency_x.size() / plane;
  if (lines > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  const int nline = static_cast<int>(lines);
  if (nline < 2) {
    throw std::invalid_argument("chebyshev: channel wall needs both endpoints");
  }

  const auto slope_x =
      pfc::fft::fourier_chebyshev_gradient(tendency_x, nx, ny, period_x, period_y);
  const auto slope_y =
      pfc::fft::fourier_chebyshev_gradient(tendency_y, nx, ny, period_x, period_y);
  const auto slope_z =
      pfc::fft::fourier_chebyshev_gradient(tendency_z, nx, ny, period_x, period_y);
  std::vector<double> divergence(tendency_x.size());
  for (std::size_t i = 0; i < divergence.size(); ++i) {
    divergence[i] = slope_x.x[i] + slope_y.y[i] + slope_z.z[i];
  }

  std::vector<double> slope_at_plus(plane);
  std::vector<double> slope_at_minus(plane);
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const auto slot = wall_index(0, iy, ix, ny, nx);
      slope_at_plus[slot] = tendency_z[slot];
      slope_at_minus[slot] = tendency_z[wall_index(nline - 1, iy, ix, ny, nx)];
    }
  }
  const auto pressure = pfc::fft::fourier_chebyshev_neumann_poisson(
      divergence, nx, ny, period_x, period_y, slope_at_plus, slope_at_minus);
  const auto gradient =
      pfc::fft::fourier_chebyshev_gradient(pressure, nx, ny, period_x, period_y);

  WallAcceleration acceleration;
  acceleration.x.resize(tendency_x.size());
  acceleration.y.resize(tendency_x.size());
  acceleration.z.resize(tendency_x.size());
  for (std::size_t i = 0; i < tendency_x.size(); ++i) {
    acceleration.x[i] = tendency_x[i] - gradient.x[i];
    acceleration.y[i] = tendency_y[i] - gradient.y[i];
    acceleration.z[i] = tendency_z[i] - gradient.z[i];
  }
  return acceleration;
}

} // namespace flow

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_channel.cpp
 * @brief Plane Couette and plane Poiseuille on the Chebyshev wall.
 *
 * Both profiles are polynomials of degree at most 2, so every
 * degree from 2 upward is an exact grid. The check is the steady
 * balance: wall values, flow rate, wall shear, momentum, and a
 * zero acceleration. A viscous step holds both profiles and
 * advances sin(π z) at −ν π². The same step advances the
 * spanwise mode sin(π z) cos(2x) at −ν (4 + π²). Turbulent
 * channel statistics are not here.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include <flow/channel.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("Plane Couette and Poiseuille stay fixed on a channel wall", "[channel]") {
  const double nu = 0.3;
  const std::array<std::pair<int, int>, 3> grids{{{8, 6}, {8, 1}, {1, 6}}};
  const std::array<int, 3> degrees{{2, 8, 16}};
  for (const int degree : degrees) {
    for (const auto &grid : grids) {
      for (const auto profile :
           {flow::LaminarProfile::couette, flow::LaminarProfile::poiseuille}) {
        const auto report =
            flow::laminar_balance(profile, grid.first, grid.second, degree, nu);
        const bool couette = profile == flow::LaminarProfile::couette;
        REQUIRE(report.profile_error < 1e-10);
        REQUIRE(report.wall_error < 1e-10);
        REQUIRE_THAT(report.flow_rate, WithinAbs(couette ? 0.0 : 4.0 / 3.0, 1e-10));
        REQUIRE(report.flow_spread < 1e-10);
        REQUIRE_THAT(report.shear_plus, WithinAbs(couette ? 1.0 : -2.0, 1e-10));
        REQUIRE_THAT(report.shear_minus, WithinAbs(couette ? 1.0 : 2.0, 1e-10));
        REQUIRE(report.shear_spread < 1e-10);
        REQUIRE(std::abs(report.momentum) < 1e-10);
        REQUIRE(report.max_acceleration < 1e-8);
        REQUIRE(report.max_normal < 1e-9);
        REQUIRE(report.velocity_change < 1e-8);
        REQUIRE(report.step_drift < 1e-9);
      }
    }
  }
}

TEST_CASE("A wall-normal mode decays at the viscous rate", "[channel]") {
  const double nu = 0.3;
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const double step = 0.1;
  const std::array<std::pair<int, int>, 4> grids{{{1, 1}, {8, 1}, {1, 6}, {8, 6}}};

  const auto decay_error = [&](int nx, int ny, int degree, bool spanwise) {
    const auto z = pfc::fft::chebyshev_lobatto(degree);
    const int nline = static_cast<int>(z.size());
    const std::size_t count = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nline);
    const double wave = spanwise ? 4.0 : 0.0;
    const double rate = -nu * (wave + pi * pi);
    std::vector<double> velocity_x(count, 0.0);
    std::vector<double> velocity_y(count, 0.0);
    std::vector<double> velocity_z(count, 0.0);
    std::vector<double> initial(count, 0.0);
    for (int iz = 0; iz < nline; ++iz) {
      const double sine = std::sin(pi * z[static_cast<std::size_t>(iz)]);
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
          const double x =
              period * static_cast<double>(ix) / static_cast<double>(nx);
          const double sample = sine * (spanwise ? std::cos(2.0 * x) : 1.0);
          const auto index = flow::channel_index(iz, iy, ix, ny, nx);
          initial[index] = sample;
          if (spanwise) {
            velocity_y[index] = sample;
          } else {
            velocity_x[index] = sample;
          }
        }
      }
    }
    flow::viscous_advance(velocity_x, velocity_y, velocity_z, nx, ny, period, period,
                          nu, 0.0, step, 0.0, 0.0);
    const auto &advanced = spanwise ? velocity_y : velocity_x;
    const auto &quiet = spanwise ? velocity_x : velocity_y;
    double error = 0.0;
    double moved = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      const double expected = initial[i] * (1.0 + step * rate);
      error = std::max(error, std::abs(advanced[i] - expected));
      moved = std::max(moved, std::abs(advanced[i] - initial[i]));
      error = std::max(error, std::abs(quiet[i]));
      error = std::max(error, std::abs(velocity_z[i]));
    }
    REQUIRE(moved > 0.2);
    return error;
  };

  for (const auto &grid : grids) {
    const double coarse = decay_error(grid.first, grid.second, 16, false);
    const double fine = decay_error(grid.first, grid.second, 32, false);
    REQUIRE(coarse < 1e-7);
    REQUIRE(fine < 1e-9);
    REQUIRE(fine < coarse);
    if (grid.first >= 8) {
      const double mode_coarse = decay_error(grid.first, grid.second, 16, true);
      const double mode_fine = decay_error(grid.first, grid.second, 32, true);
      REQUIRE(mode_coarse < 1e-7);
      REQUIRE(mode_fine < 1e-9);
      REQUIRE(mode_fine < mode_coarse);
    }
  }
}

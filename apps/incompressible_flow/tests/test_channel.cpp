// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_channel.cpp
 * @brief Plane Couette and plane Poiseuille on the Chebyshev wall.
 *
 * Both profiles are polynomials of degree at most 2, so every
 * degree from 2 upward is an exact grid. The check is the steady
 * balance: wall values, flow rate, wall shear, momentum, and a
 * zero acceleration. Turbulent channel statistics are not here.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <utility>

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

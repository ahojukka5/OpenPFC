// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_taylor_green.cpp
 * @brief Small-grid regression for the 3-D Taylor–Green driver.
 *
 * Thresholds are the analytic t = 0 cell averages and an FFT round trip.
 * They were set before the refinement ladders. N = 16 does not verify
 * the spatial or temporal order.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>

#include <tg3d/verify.hpp>

using Catch::Matchers::WithinAbs;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

} // namespace

TEST_CASE("Taylor-Green starts at the analytic energy and grows a third component",
          "[tg3d][taylor]") {
  if (world_size() != 1) SKIP("one rank owns every Fourier mode");
  constexpr double nu = 0.05;
  constexpr double dt = 0.01;
  constexpr int n = 16;
  auto flow = tg3d::make_flow(n, nu, dt, 0, 1);
  tg3d::initialize_taylor_green(flow);
  const auto initial = tg3d::diagnose(flow);

  REQUIRE(initial.modal_div_max < 1.0e-8);
  REQUIRE(initial.div_linf < 1.0e-10);
  REQUIRE(initial.div_l2 < 1.0e-10);
  REQUIRE(initial.w_l2 < 1.0e-12);
  REQUIRE(initial.max_abs_w < 1.0e-12);
  REQUIRE_THAT(initial.ke, WithinAbs(tg3d::kinetic_energy_0, 1.0e-12));
  REQUIRE_THAT(initial.enstrophy, WithinAbs(tg3d::enstrophy_0, 1.0e-10));
  REQUIRE(std::abs(initial.mean_u) < 1.0e-12);
  REQUIRE(std::abs(initial.mean_v) < 1.0e-12);
  REQUIRE(std::abs(initial.mean_w) < 1.0e-12);
  REQUIRE(initial.outer_ke_fraction < 1.0e-8);
  REQUIRE_THAT(initial.cfl,
               WithinAbs(dt * static_cast<double>(n) / pfc::two_pi, 1.0e-12));

  double shells = 0.0;
  for (const auto &shell : initial.spectrum) shells += shell.ke;
  REQUIRE(std::abs(shells - initial.ke) / initial.ke < 1.0e-8);

  tg3d::step(flow);
  REQUIRE(tg3d::state_finite(flow));
  const auto stepped = tg3d::diagnose(flow);
  REQUIRE(stepped.modal_div_max < 1.0e-8);
  REQUIRE(stepped.div_linf < 1.0e-8);
  REQUIRE(stepped.ke < tg3d::kinetic_energy_0 - 1.0e-8);
  REQUIRE(stepped.w_l2 > 1.0e-8);
}

TEST_CASE("common-band restriction reproduces a resolved Taylor-Green mode",
          "[tg3d][band]") {
  if (world_size() != 1) SKIP("one rank owns every Fourier mode");
  auto fine = tg3d::make_flow(32, 0.05, 0.01, 0, 1);
  auto coarse = tg3d::make_flow(16, 0.05, 0.01, 0, 1);
  tg3d::initialize_taylor_green(fine);
  tg3d::initialize_taylor_green(coarse);
  const auto hats = tg3d::copy_hats(fine);
  const auto err = tg3d::field_error(coarse, hats);
  REQUIRE(err.velocity_l2 < 1.0e-10);
  REQUIRE(err.velocity_l2_rel < 1.0e-10);
  REQUIRE(err.vorticity_l2 < 1.0e-10);
  REQUIRE(err.vorticity_linf < 1.0e-10);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}

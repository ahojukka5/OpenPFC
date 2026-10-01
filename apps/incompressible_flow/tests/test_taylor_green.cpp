// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_taylor_green.cpp
 * @brief Small-grid regression for the maintained Taylor–Green case.
 *
 * Thresholds match the analytic t = 0 cell averages. N = 16 does not
 * repeat the spatial or temporal ladder.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>

#include <flow/taylor_green.hpp>

using Catch::Matchers::WithinAbs;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

void world(int &rank, int &nproc) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
}

} // namespace

TEST_CASE("Taylor-Green starts at the analytic energy and grows a third component",
          "[flow][taylor]") {
  if (world_size() != 1) SKIP("one rank owns every Fourier mode");
  constexpr double nu = 0.05;
  constexpr double dt = 0.01;
  constexpr int n = 16;
  auto state = flow::make_state(n, nu, dt, 0, 1);
  flow::initialize_taylor_green(state);
  const auto initial = flow::diagnose(state);

  REQUIRE(initial.finite);
  REQUIRE(initial.modal_div_max < 1.0e-8);
  REQUIRE(initial.div_linf < 1.0e-10);
  REQUIRE(initial.div_l2 < 1.0e-10);
  REQUIRE(initial.w_l2 < 1.0e-12);
  REQUIRE(initial.max_abs_w < 1.0e-12);
  REQUIRE_THAT(initial.ke, WithinAbs(flow::kinetic_energy_0, 1.0e-12));
  REQUIRE_THAT(initial.enstrophy, WithinAbs(flow::enstrophy_0, 1.0e-10));
  REQUIRE(std::abs(initial.mean_u) < 1.0e-12);
  REQUIRE(std::abs(initial.mean_v) < 1.0e-12);
  REQUIRE(std::abs(initial.mean_w) < 1.0e-12);
  REQUIRE_THAT(initial.cfl,
               WithinAbs(dt * static_cast<double>(n) / pfc::two_pi, 1.0e-12));

  flow::step(state);
  const auto stepped = flow::diagnose(state);
  REQUIRE(stepped.finite);
  REQUIRE(stepped.modal_div_max < 1.0e-8);
  REQUIRE(stepped.div_linf < 1.0e-8);
  REQUIRE(stepped.ke < flow::kinetic_energy_0 - 1.0e-8);
  REQUIRE(stepped.w_l2 > 1.0e-8);
  REQUIRE(flow::advance(state, 0) == "ok");
}

TEST_CASE("Taylor-Green energy is the same on every pencil", "[flow][taylor][mpi]") {
  int rank = 0;
  int nproc = 1;
  world(rank, nproc);
  constexpr double nu = 0.05;
  constexpr double dt = 0.01;
  constexpr int n = 16;
  auto state = flow::make_state(n, nu, dt, rank, nproc);
  flow::initialize_taylor_green(state);
  const auto initial = flow::diagnose(state);
  REQUIRE(initial.finite);
  REQUIRE(initial.modal_div_max < 1.0e-8);
  REQUIRE(initial.div_l2 < 1.0e-10);
  REQUIRE(initial.w_l2 < 1.0e-12);
  REQUIRE_THAT(initial.ke, WithinAbs(flow::kinetic_energy_0, 1.0e-12));
  REQUIRE_THAT(initial.enstrophy, WithinAbs(flow::enstrophy_0, 1.0e-10));
  REQUIRE(std::abs(initial.mean_u) < 1.0e-12);
  REQUIRE(std::abs(initial.mean_v) < 1.0e-12);
  REQUIRE(std::abs(initial.mean_w) < 1.0e-12);

  flow::step(state);
  const auto stepped = flow::diagnose(state);
  REQUIRE(stepped.finite);
  REQUIRE(stepped.modal_div_max < 1.0e-8);
  REQUIRE(stepped.ke < flow::kinetic_energy_0 - 1.0e-8);
  REQUIRE(stepped.w_l2 > 1.0e-8);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}

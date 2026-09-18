// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_mhd.cpp
 * @brief Catch2 verification hierarchy for 2-D incompressible MHD (#23).
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <mpi.h>

#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

#include <ns2d/cases.hpp>
#include <ns2d/mhd.hpp>
#include <ns2d/mhd_cases.hpp>
#include <ns2d/vorticity_stream.hpp>

using Catch::Matchers::WithinAbs;

namespace {

int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}
int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

double reduce_max(double local) {
  double g = 0.0;
  MPI_Allreduce(&local, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

struct MHDStack {
  pfc::sim::stacks::SpectralCPUStack stack;
  ns2d::MHDSolver solver;
  MHDStack(int n, ns2d::MHDParams p)
      : stack(ns2d::make_twopi_slab(n), world_rank(), world_size(),
              MPI_COMM_WORLD),
        solver(stack, p) {}
};

} // namespace

TEST_CASE("a=0 MHD reproduces the NS solver on a two-mode field",
          "[mhd][hydro-reduction]") {
  const double nu = 0.05;
  const double dt = 0.01;
  const int steps = 20;
  pfc::sim::stacks::SpectralCPUStack ns_stack(
      ns2d::make_twopi_slab(32), world_rank(), world_size(), MPI_COMM_WORLD);
  ns2d::VorticityStreamCPU ns(ns_stack, ns2d::Params{nu, dt});
  ns.initialize_omega(
      [](double x, double y, double) { return ns2d::two_mode_omega(x, y); });

  MHDStack mhd(32, ns2d::MHDParams{nu, nu, dt, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::two_mode_omega(x, y); },
      [](double, double, double) { return 0.0; });

  for (int i = 0; i < steps; ++i) {
    ns.step();
    mhd.solver.step();
  }
  double local = 0.0;
  ns.omega().for_each_owned([&](int i, int j, int k) {
    local = std::max(local, std::abs(ns.omega()(i, j, k) -
                                     mhd.solver.omega()(i, j, k)));
  });
  REQUIRE(reduce_max(local) < 1.0e-11);
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE(d.me < 1.0e-20);
  REQUIRE(d.max_b < 1.0e-10);
}

TEST_CASE("force-free Fourier mode decays as exp(-eta k^2 t) with u=0",
          "[mhd][force-free]") {
  const double eta = 0.1;
  const double dt = 0.02;
  const int steps = 10;
  MHDStack mhd(32, ns2d::MHDParams{0.05, eta, dt, +1.0});
  mhd.solver.initialize([](double, double, double) { return 0.0; },
                        [](double x, double y, double) {
                          return ns2d::force_free_a(x, y);
                        });
  for (int i = 0; i < steps; ++i) mhd.solver.step();
  const double T = steps * dt;
  const double aerr = mhd.solver.linf_a_error(
      MPI_COMM_WORLD, [eta, T](double x, double y) {
        return ns2d::force_free_a_exact(x, y, eta, T);
      });
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(aerr, WithinAbs(0.0, 1.0e-11));
  REQUIRE(d.max_speed < 1.0e-12);
  REQUIRE(d.max_abs_omega < 1.0e-12);
  REQUIRE_THAT(d.me, WithinAbs(0.25 * std::exp(-4.0 * eta * T), 1.0e-12));
}

TEST_CASE("correct Lorentz sign cancels Alfvénic u=B; the wrong sign does not",
          "[mhd][lorentz-sign]") {
  auto nlinf = [](double sign) {
    MHDStack mhd(32, ns2d::MHDParams{0.0, 0.0, 0.01, sign});
    mhd.solver.initialize(
        [](double x, double y, double) { return ns2d::alfven_omega(x, y); },
        [](double x, double y, double) { return ns2d::alfven_phi(x, y); });
    mhd.solver.nonlinear_omega_from_current();
    return mhd.solver.n_omega_linf(MPI_COMM_WORLD);
  };
  const double good = nlinf(+1.0);
  const double bad = nlinf(-1.0);
  REQUIRE(good < 1.0e-10);
  REQUIRE(bad > 1.0);
  REQUIRE(bad > 1.0e6 * (good + 1.0e-18));
}

TEST_CASE("Alfvénic u=B with nu=eta decays as a viscous eigenmode",
          "[mhd][alfven]") {
  const double nu = 0.05;
  const double dt = 0.01;
  const int steps = 15;
  MHDStack mhd(32, ns2d::MHDParams{nu, nu, dt, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::alfven_omega(x, y); },
      [](double x, double y, double) { return ns2d::alfven_phi(x, y); });
  const auto d0 = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(d0.ke - d0.me, WithinAbs(0.0, 1.0e-12));
  for (int i = 0; i < steps; ++i) mhd.solver.step();
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(d.ke - d.me, WithinAbs(0.0, 1.0e-11));
  REQUIRE(d.div_u_linf < 1.0e-11);
  REQUIRE(d.div_b_linf < 1.0e-11);
}

TEST_CASE("ideal two-mode MHD conserves E, Hc and A2 over a short window",
          "[mhd][invariants]") {
  MHDStack mhd(32, ns2d::MHDParams{0.0, 0.0, 0.01, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::ot_omega(x, y); },
      [](double x, double y, double) { return ns2d::ot_a(x, y); });
  const auto d0 = mhd.solver.diagnostics(MPI_COMM_WORLD);
  for (int i = 0; i < 40; ++i) mhd.solver.step();
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE(std::isfinite(d.energy));
  REQUIRE(std::abs(d.energy - d0.energy) / d0.energy < 2.0e-3);
  REQUIRE(std::abs(d.a2 - d0.a2) / d0.a2 < 2.0e-3);
  REQUIRE(std::abs(d.cross_helicity - d0.cross_helicity) < 2.0e-3);
  REQUIRE(d.div_u_linf < 1.0e-10);
  REQUIRE(d.div_b_linf < 1.0e-10);
}

TEST_CASE("dissipative energy budget matches nu <w^2> + eta <j^2>",
          "[mhd][budget]") {
  MHDStack mhd(32, ns2d::MHDParams{0.05, 0.05, 0.005, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::ot_omega(x, y); },
      [](double x, double y, double) { return ns2d::ot_a(x, y); });
  (void)mhd.solver.diagnostics(MPI_COMM_WORLD);
  mhd.solver.step();
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE(std::abs(d.energy_budget_residual) < 5.0e-3);
  REQUIRE(d.dissipation > 0.0);
}

TEST_CASE("u and B remain solenoidal to transform roundoff",
          "[mhd][divergence]") {
  MHDStack mhd(32, ns2d::MHDParams{0.02, 0.02, 0.01, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::ot_omega(x, y); },
      [](double x, double y, double) { return ns2d::ot_a(x, y); });
  for (int i = 0; i < 10; ++i) mhd.solver.step();
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE(d.div_u_linf < 1.0e-11);
  REQUIRE(d.div_b_linf < 1.0e-11);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}

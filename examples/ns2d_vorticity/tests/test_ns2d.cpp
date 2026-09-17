// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_ns2d.cpp
 * @brief Catch2 tests for the 2-D vorticity–streamfunction prototype.
 *
 * Taylor–Green is spectrally exact: a small Linf is the spatial check.
 * Temporal order is measured on a two-mode nonlinear IC, not on TG.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

#include <ns2d/cases.hpp>
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

struct StackSolver {
  pfc::sim::stacks::SpectralCPUStack stack;
  ns2d::VorticityStreamCPU solver;
  StackSolver(int n, ns2d::Params p)
      : stack(ns2d::make_periodic_square(n), world_rank(), world_size(),
              MPI_COMM_WORLD),
        solver(stack, p) {}
};

} // namespace

TEST_CASE("Poisson invert recovers Taylor-Green streamfunction velocity",
          "[ns2d][poisson]") {
  StackSolver ss(32, ns2d::Params{0.1, 0.01});
  ss.solver.initialize_omega(
      [](double x, double y, double) { return ns2d::taylor_green_omega(x, y, 0.1, 0.0); });

  double u_err = 0.0, v_err = 0.0;
  ss.stack.u().for_each_owned([&](int i, int j, int k) {
    const auto c = ss.stack.u().coords(i, j, k);
    const std::size_t idx =
        static_cast<std::size_t>(i) +
        static_cast<std::size_t>(j) *
            static_cast<std::size_t>(ss.stack.fft().get_inbox_bounds().size[0]) +
        static_cast<std::size_t>(k) *
            static_cast<std::size_t>(ss.stack.fft().get_inbox_bounds().size[0]) *
            static_cast<std::size_t>(ss.stack.fft().get_inbox_bounds().size[1]);
    u_err = std::max(u_err, std::abs(ss.solver.u()[idx] -
                                     ns2d::taylor_green_u(c[0], c[1], 0.1, 0.0)));
    v_err = std::max(v_err, std::abs(ss.solver.v()[idx] -
                                     ns2d::taylor_green_v(c[0], c[1], 0.1, 0.0)));
  });
  double g_u = 0.0, g_v = 0.0;
  MPI_Allreduce(&u_err, &g_u, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&v_err, &g_v, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE_THAT(g_u, WithinAbs(0.0, 1.0e-12));
  REQUIRE_THAT(g_v, WithinAbs(0.0, 1.0e-12));
}

TEST_CASE("zero mode of the streamfunction is exactly zero", "[ns2d][zero-mode]") {
  StackSolver ss(16, ns2d::Params{0.05, 0.02});
  ss.solver.initialize_omega(
      [](double x, double y, double) { return ns2d::two_mode_omega(x, y); });
  const auto &inv = ss.solver.psi_inv_k2();
  if (ss.solver.rank_owns_zero_mode()) {
    REQUIRE_THAT(inv.front(), WithinAbs(0.0, 0.0));
  }
  pfc::fft::kspace::for_each_kpoint(
      ss.stack.fft().get_outbox_bounds(), ss.stack.u().global_size(),
      ss.stack.u().spacing(),
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        if (i == 0 && j == 0 && k == 0) {
          REQUIRE_THAT(inv[idx], WithinAbs(0.0, 0.0));
        }
      });
}

TEST_CASE("2/3 dealias mask zeros the high-k third", "[ns2d][dealias]") {
  StackSolver ss(24, ns2d::Params{0.1, 0.01});
  const auto &mask = ss.solver.dealias_mask();
  const auto spacing = ss.stack.u().spacing();
  int kept = 0, killed = 0;
  pfc::fft::kspace::for_each_kpoint(
      ss.stack.fft().get_outbox_bounds(), ss.stack.u().global_size(), spacing,
      [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
        const bool keep = pfc::fft::kspace::two_thirds_keep(kx, ky, kz, spacing);
        if (keep) {
          REQUIRE_THAT(mask[idx], WithinAbs(1.0, 0.0));
          ++kept;
        } else {
          REQUIRE_THAT(mask[idx], WithinAbs(0.0, 0.0));
          ++killed;
        }
      });
  REQUIRE(kept > 0);
  REQUIRE(killed > 0);
}

TEST_CASE("Taylor-Green decays to spectral accuracy", "[ns2d][taylor-green]") {
  const double nu = 0.1;
  const double dt = 0.05;
  const int steps = 8;
  StackSolver ss(32, ns2d::Params{nu, dt});
  ss.solver.initialize_omega(
      [nu](double x, double y, double) { return ns2d::taylor_green_omega(x, y, nu, 0.0); });
  for (int i = 0; i < steps; ++i) ss.solver.step();
  const auto d = ss.solver.diagnostics(MPI_COMM_WORLD);
  const double T = static_cast<double>(steps) * dt;
  const double linf = ss.solver.linf_omega_error(
      MPI_COMM_WORLD, [nu, T](double x, double y) {
        return ns2d::taylor_green_omega(x, y, nu, T);
      });
  REQUIRE_THAT(linf, WithinAbs(0.0, 1.0e-11));
  REQUIRE_THAT(d.ke, WithinAbs(ns2d::taylor_green_ke(nu, T), 1.0e-12));
  REQUIRE_THAT(d.enstrophy,
               WithinAbs(ns2d::taylor_green_enstrophy(nu, T), 1.0e-12));
  REQUIRE_THAT(d.div_linf, WithinAbs(0.0, 1.0e-11));
  REQUIRE_THAT(d.mean_omega, WithinAbs(0.0, 1.0e-14));
}

TEST_CASE("Taylor-Green error is already at roundoff on N=16 and N=32",
          "[ns2d][spatial]") {
  const double nu = 0.05;
  const double dt = 0.02;
  const int steps = 5;
  const double T = steps * dt;
  auto run = [&](int n) {
    StackSolver ss(n, ns2d::Params{nu, dt});
    ss.solver.initialize_omega(
        [nu](double x, double y, double) { return ns2d::taylor_green_omega(x, y, nu, 0.0); });
    for (int i = 0; i < steps; ++i) ss.solver.step();
    return ss.solver.linf_omega_error(MPI_COMM_WORLD, [nu, T](double x, double y) {
      return ns2d::taylor_green_omega(x, y, nu, T);
    });
  };
  const double e16 = run(16);
  const double e32 = run(32);
  REQUIRE(e16 < 1.0e-11);
  REQUIRE(e32 < 1.0e-11);
}

TEST_CASE("mean vorticity is conserved on a nonlinear two-mode field",
          "[ns2d][invariants]") {
  StackSolver ss(32, ns2d::Params{0.01, 0.005});
  ss.solver.initialize_omega(
      [](double x, double y, double) { return ns2d::two_mode_omega(x, y); });
  const double mean0 = ss.solver.diagnostics(MPI_COMM_WORLD).mean_omega;
  for (int i = 0; i < 40; ++i) ss.solver.step();
  const auto d = ss.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(d.mean_omega, WithinAbs(mean0, 1.0e-12));
  REQUIRE(d.div_linf < 1.0e-10);
  REQUIRE(d.ke > 0.0);
  REQUIRE(d.enstrophy > 0.0);
}

TEST_CASE("ETD1 is first-order on a two-mode nonlinear Jacobian",
          "[ns2d][timestep]") {
  const double nu = 0.05;
  const double T = 0.2;
  auto run = [&](double dt) {
    const int steps = static_cast<int>(std::llround(T / dt));
    StackSolver ss(32, ns2d::Params{nu, dt});
    ss.solver.initialize_omega(
        [](double x, double y, double) { return ns2d::two_mode_omega(x, y); });
    for (int i = 0; i < steps; ++i) ss.solver.step();
    std::vector<double> local(static_cast<std::size_t>(ss.stack.u().size()), 0.0);
    ss.stack.u().for_each_owned([&](int i, int j, int k) {
      local[ss.stack.u().idx(i, j, k)] = ss.stack.u()(i, j, k);
    });
    return local;
  };
  const auto fine = run(0.0025);
  auto err = [&](double dt) {
    auto w = run(dt);
    double local = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) {
      local = std::max(local, std::abs(w[i] - fine[i]));
    }
    double g = 0.0;
    MPI_Allreduce(&local, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return g;
  };
  const double e_coarse = err(0.02);
  const double e_mid = err(0.01);
  REQUIRE(e_coarse > 0.0);
  REQUIRE(e_mid > 0.0);
  REQUIRE(e_coarse / e_mid > 1.6);
  REQUIRE(e_coarse / e_mid < 2.6);
}

TEST_CASE("double shear layer has two opposite vorticity sheets at t=0",
          "[ns2d][shear]") {
  StackSolver ss(64, ns2d::Params{0.001, 0.002});
  constexpr double rho = 30.0;
  constexpr double eps = 0.05;
  ss.solver.initialize_omega(
      [](double x, double y, double) { return ns2d::double_shear_omega(x, y, rho, eps); });
  const auto d = ss.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE(d.max_abs_omega > 20.0);
  REQUIRE_THAT(d.mean_omega, WithinAbs(0.0, 1.0e-10));
  REQUIRE(d.div_linf < 1.0e-8);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}

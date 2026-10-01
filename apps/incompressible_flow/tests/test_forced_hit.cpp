// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_forced_hit.cpp
 * @brief Constant-power contract on the |k| = 1 shell.
 *
 * The stationary ladder is not this test.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <optional>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/incompressible.hpp>

#include <flow/forced_hit.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

std::optional<flow::Complex> coefficient(const std::vector<flow::Complex> &hat,
                                         const flow::State &state, int si, int sj,
                                         int sk) {
  const auto outbox = state.stack->fft().get_outbox_bounds();
  std::optional<flow::Complex> found;
  pfc::fft::kspace::for_each_kpoint(
      outbox, state.n, state.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        if (flow::signed_index(i, state.n[0]) != si) return;
        if (flow::signed_index(j, state.n[1]) != sj) return;
        if (flow::signed_index(k, state.n[2]) != sk) return;
        found = hat[idx];
      });
  return found;
}

void clear_lowest_shell(flow::State &state) {
  const auto outbox = state.stack->fft().get_outbox_bounds();
  pfc::fft::kspace::for_each_kpoint(
      outbox, state.n, state.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        const int si = flow::signed_index(i, state.n[0]);
        const int sj = flow::signed_index(j, state.n[1]);
        const int sk = flow::signed_index(k, state.n[2]);
        if (!flow::lowest_shell(si, sj, sk)) return;
        state.u[idx] = state.v[idx] = state.w[idx] = flow::Complex{};
      });
}

} // namespace

TEST_CASE("constant-power force stays on the lowest shell", "[flow][forced]") {
  if (world_size() != 1) SKIP("one rank owns every Fourier mode");
  constexpr int n = 16;
  constexpr double power = 0.25;
  auto state = flow::make_state(n, 0.02, 0.01, 0, 1);
  flow::initialize_decaying_hit(state, 1);
  const auto outbox = state.stack->fft().get_outbox_bounds();
  std::vector<flow::Complex> fu, fv, fw;
  const auto force = flow::write_band_force(outbox, state.n, state.spacing, state.u,
                                            state.v, state.w, fu, fv, fw, power);
  REQUIRE(force.applied);
  REQUIRE(force.band_ke > 0.0);
  REQUIRE_THAT(force.injection, WithinRel(power, 1.0e-12));

  auto &fft = state.stack->fft();
  std::vector<double> rx(fft.size_inbox()), ry(fft.size_inbox()),
      rz(fft.size_inbox());
  std::vector<double> fx(fft.size_inbox()), fy(fft.size_inbox()),
      fz(fft.size_inbox());
  std::vector<flow::Complex> u = state.u;
  std::vector<flow::Complex> v = state.v;
  std::vector<flow::Complex> w = state.w;
  fft.backward(u, rx);
  fft.backward(v, ry);
  fft.backward(w, rz);
  fft.backward(fu, fx);
  fft.backward(fv, fy);
  fft.backward(fw, fz);
  double dot = 0.0;
  double mean_f = 0.0;
  const double ncells = flow::cell_count(state.n);
  for (std::size_t i = 0; i < rx.size(); ++i) {
    dot += rx[i] * fx[i] + ry[i] * fy[i] + rz[i] * fz[i];
    mean_f += fx[i] + fy[i] + fz[i];
  }
  REQUIRE_THAT(dot / ncells, WithinRel(power, 1.0e-8));
  REQUIRE_THAT(mean_f / ncells, WithinAbs(0.0, 1.0e-12));

  const auto off = coefficient(fu, state, 2, 0, 0);
  const auto mean = coefficient(fu, state, 0, 0, 0);
  const auto on = coefficient(fv, state, 1, 0, 0);
  REQUIRE(off.has_value());
  REQUIRE(mean.has_value());
  REQUIRE(on.has_value());
  REQUIRE(*off == flow::Complex{});
  REQUIRE(*mean == flow::Complex{});
  REQUIRE(std::abs(*on) > 0.0);
  REQUIRE(pfc::field::max_modal_divergence(outbox, state.n, state.spacing, fu.data(),
                                           fv.data(), fw.data(),
                                           fu.size()) < 1.0e-8);

  std::vector<flow::Complex> gu, gv, gw;
  const auto repeat = flow::write_band_force(outbox, state.n, state.spacing, state.u,
                                             state.v, state.w, gu, gv, gw, power);
  REQUIRE(fu == gu);
  REQUIRE(fv == gv);
  REQUIRE(fw == gw);
  REQUIRE(repeat.applied);

  clear_lowest_shell(state);
  const auto idle = flow::write_band_force(outbox, state.n, state.spacing, state.u,
                                           state.v, state.w, fu, fv, fw, power);
  REQUIRE_FALSE(idle.applied);
  REQUIRE(idle.injection == 0.0);
  REQUIRE(fu == std::vector<flow::Complex>(fu.size()));

  flow::initialize_decaying_hit(state, 1);
  flow::step_forced(state, power);
  const auto stepped = flow::diagnose(state);
  REQUIRE(stepped.finite);
  REQUIRE(stepped.modal_div_max < 1.0e-8);
  const auto after = flow::write_band_force(outbox, state.n, state.spacing, state.u,
                                            state.v, state.w, fu, fv, fw, power);
  REQUIRE(after.applied);
  REQUIRE_THAT(after.injection, WithinRel(power, 1.0e-10));
}

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// FieldOperations: padded-Field vectors for gmres/cg/newton (#471).

#include <catch2/catch_all.hpp>

#include <cmath>
#include <vector>

#include <mpi.h>

#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/newton.hpp>
#include <openpfc/solvers/vector_operations.hpp>

#include "field_operations_problem.hpp"

using namespace field_ops_test;
using P = Problem<pfc::HostSpace>;

TEST_CASE("FieldOperations reduce owned cells only, optionally masked",
          "[solvers][field_operations]") {
  P p(16);
  auto x = p.ops.make(p.ops.size()), y = p.ops.make(p.ops.size());
  auto mask = p.ops.make(p.ops.size());
  for (int c = 0; c < 2; ++c) {
    for (std::size_t q = 0; q < x[c].size(); ++q) { // halos get junk
      x[c].data()[q] = 1e3;
      y[c].data()[q] = -7.0;
    }
    p.owned(x[c], [&](int i, int j, int gi, int gj) {
      x[c](i, j, 0) = gi + 1.0 + c;
      y[c](i, j, 0) = gj - 2.0;
      mask[c](i, j, 0) = (gi + gj) % 3 == 0 ? 0.0 : 1.0;
    });
  }
  double ref = 0, ref_masked = 0;
  for (int c = 0; c < 2; ++c)
    for (int gj = 0; gj < 16; ++gj)
      for (int gi = 0; gi < 16; ++gi) {
        const double v = (gi + 1.0 + c) * (gj - 2.0);
        ref += v;
        if ((gi + gj) % 3 != 0 || c == 0) ref_masked += v;
      }
  REQUIRE(p.ops.dot(x, y) == Catch::Approx(ref).epsilon(1e-14));
  p.ops.set_mask(1, &mask[1]);
  REQUIRE(p.ops.dot(x, y) == Catch::Approx(ref_masked).epsilon(1e-14));
  p.ops.set_mask(1, nullptr);
  auto z = p.ops.make(p.ops.size());
  p.ops.copy(x, z);
  p.ops.axpy(z, -2.0, x);
  p.ops.scale(z, -1.0);
  REQUIRE(p.ops.dot(z, y) == Catch::Approx(ref).epsilon(1e-14));
  REQUIRE_THROWS(p.ops.make(p.ops.size() + 1));
}

TEST_CASE("CG and GMRES on FieldVectors match the flat global path",
          "[solvers][field_operations]") {
  const int n = 32;
  P p(n);
  auto b = p.ops.make(p.ops.size());
  b = p.right_hand_side();
  pfc::solvers::KrylovControls kc;
  kc.relative_tolerance = 1e-10;
  kc.maximum_iterations = 2000;
  kc.restart = 400;

  // Flat reference on the global grid (every rank solves it whole).
  pfc::solvers::VectorOperations<> flat;
  std::vector<double> fb(std::size_t(2) * n * n), fx(fb.size(), 0.0);
  for (int c = 0; c < 2; ++c)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        fb[(std::size_t(c) * n + j) * n + i] = rhs(c, i, j, n);
  auto flat_cg = pfc::solvers::cg(
      flat, [&](const auto &u, auto &o) { flat_apply(n, false, u, o); }, fb, fx, kc);
  REQUIRE(flat_cg.converged());

  auto x = p.ops.make(p.ops.size());
  auto r_cg = pfc::solvers::cg(
      p.ops, [&](const auto &u, auto &o) { p.apply(u, o, false, false); }, b, x, kc);
  REQUIRE(r_cg.converged());
  const auto gx = p.gather(x);
  double e = 0, s = 0;
  for (std::size_t q = 0; q < gx.size(); ++q) {
    e += (gx[q] - fx[q]) * (gx[q] - fx[q]);
    s += fx[q] * fx[q];
  }
  INFO("CG iterations field " << r_cg.iterations << " flat " << flat_cg.iterations
                              << " rel diff " << std::sqrt(e / s));
  REQUIRE(std::sqrt(e / s) < 1e-8);
  REQUIRE(std::abs(r_cg.iterations - flat_cg.iterations) <= 2);

  auto xg = p.ops.make(p.ops.size());
  auto r_gm = pfc::solvers::gmres(
      p.ops, [&](const auto &u, auto &o) { p.apply(u, o, false, false); }, b, xg,
      kc);
  REQUIRE(r_gm.converged());
  const auto gg = p.gather(xg);
  e = 0;
  for (std::size_t q = 0; q < gg.size(); ++q) e += (gg[q] - fx[q]) * (gg[q] - fx[q]);
  REQUIRE(std::sqrt(e / s) < 1e-8);
}

TEST_CASE("GMRES on FieldVectors with a block FieldMultigrid preconditioner",
          "[solvers][field_operations]") {
  int its[2] = {0, 0}, plain = 0;
  const int ns[2] = {32, 64};
  for (int q = 0; q < 2; ++q) {
    P p(ns[q]);
    auto b = p.right_hand_side();
    auto x = p.ops.make(p.ops.size());
    pfc::solvers::KrylovControls kc;
    kc.relative_tolerance = 1e-8;
    auto res = pfc::solvers::gmres(
        p.ops, [&](const auto &u, auto &o) { p.apply(u, o, false, false); }, b, x,
        kc, [&](const auto &r, auto &z, int) { p.precondition(r, z); });
    REQUIRE(res.converged());
    its[q] = res.iterations;
    if (q == 1) {
      auto x2 = p.ops.make(p.ops.size());
      kc.maximum_iterations = 4000;
      auto un = pfc::solvers::gmres(
          p.ops, [&](const auto &u, auto &o) { p.apply(u, o, false, false); }, b, x2,
          kc);
      plain = un.iterations;
    }
  }
  INFO("MG-GMRES " << its[0] << " " << its[1] << ", unpreconditioned " << plain);
  REQUIRE(its[1] <= 15);
  REQUIRE(its[1] <= its[0] + 2);
  REQUIRE(plain > 5 * its[1]);
}

TEST_CASE("Newton-GMRES on FieldVectors with a refreshed multigrid preconditioner",
          "[solvers][field_operations]") {
  const int n = 32;
  P p(n);
  CubicModel<pfc::HostSpace> model(p);
  auto x = p.ops.make(p.ops.size());
  pfc::solvers::NewtonControls nc;
  nc.relative_tolerance = 1e-10;
  auto res = pfc::solvers::newton(p.ops, model, x, nc);
  int linear = 0;
  for (const auto &h : res.history) linear += h.linear.iterations;
  INFO("Newton " << res.history.size() << " steps, " << linear
                 << " GMRES iterations");
  REQUIRE(res.converged());
  REQUIRE(res.history.size() <= 10);
  REQUIRE(linear <= 8 * static_cast<int>(res.history.size()));
  // Same solution as the flat Newton residual: R(x) ~ 0 on the global grid.
  const auto gx = p.gather(x);
  std::vector<double> r;
  flat_apply(n, true, gx, r);
  double e = 0, s = 0;
  for (int c = 0; c < 2; ++c)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const double f = rhs(c, i, j, n),
                     v = r[(std::size_t(c) * n + j) * n + i] - f;
        e += v * v;
        s += f * f;
      }
  REQUIRE(std::sqrt(e / s) < 1e-9);
}

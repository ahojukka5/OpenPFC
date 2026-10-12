// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// FieldOperations on device Fields: the same Krylov/Newton runs as on the host
// (#471).

#include "mechanics_backend.hpp"

#include <cmath>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/newton.hpp>

#include "../../solvers/field_operations_problem.hpp"

using namespace field_ops_test;
using Dev = Problem<mechanics_test::Space>;
using Host = Problem<pfc::HostSpace>;

namespace {
double rel(const std::vector<double> &a, const std::vector<double> &b) {
  double e = 0, s = 0;
  for (std::size_t q = 0; q < a.size(); ++q) {
    e += (a[q] - b[q]) * (a[q] - b[q]);
    s += b[q] * b[q];
  }
  return std::sqrt(e / s);
}
} // namespace

TEST_CASE("FieldOperations on device match the host: CG, MG-GMRES, Newton",
          "[field_operations][device]") {
  if (!mechanics_test::available()) SKIP("GPU unavailable");
  const int n = 32;
  Dev d(n);
  Host h(n);
  pfc::solvers::KrylovControls kc;
  kc.relative_tolerance = 1e-10;
  kc.maximum_iterations = 2000;
  kc.restart = 400;

  auto bd = d.right_hand_side();
  auto bh = h.right_hand_side();
  REQUIRE(std::abs(d.ops.dot(bd, bd) - h.ops.dot(bh, bh)) <=
          1e-12 * h.ops.dot(bh, bh));

  auto xd = d.ops.make(d.ops.size());
  auto xh = h.ops.make(h.ops.size());
  auto cd = pfc::solvers::cg(
      d.ops, [&](const auto &u, auto &o) { d.apply(u, o, false, false); }, bd, xd,
      kc);
  auto ch = pfc::solvers::cg(
      h.ops, [&](const auto &u, auto &o) { h.apply(u, o, false, false); }, bh, xh,
      kc);
  INFO("CG device " << cd.iterations << " host " << ch.iterations);
  REQUIRE(cd.converged());
  REQUIRE(std::abs(cd.iterations - ch.iterations) <= 1);
  REQUIRE(rel(d.gather(xd), h.gather(xh)) < 1e-9);

  pfc::solvers::KrylovControls mc;
  mc.relative_tolerance = 1e-8;
  auto gd = d.ops.make(d.ops.size());
  auto gh = h.ops.make(h.ops.size());
  auto md = pfc::solvers::gmres(
      d.ops, [&](const auto &u, auto &o) { d.apply(u, o, false, false); }, bd, gd,
      mc, [&](const auto &r, auto &z, int) { d.precondition(r, z); });
  auto mh = pfc::solvers::gmres(
      h.ops, [&](const auto &u, auto &o) { h.apply(u, o, false, false); }, bh, gh,
      mc, [&](const auto &r, auto &z, int) { h.precondition(r, z); });
  INFO("MG-GMRES device " << md.iterations << " host " << mh.iterations);
  REQUIRE(md.converged());
  REQUIRE(md.iterations == mh.iterations);
  REQUIRE(rel(d.gather(gd), h.gather(gh)) < 1e-10);

  CubicModel<mechanics_test::Space> nd(d);
  CubicModel<pfc::HostSpace> nh(h);
  auto ud = d.ops.make(d.ops.size());
  auto uh = h.ops.make(h.ops.size());
  pfc::solvers::NewtonControls nc;
  nc.relative_tolerance = 1e-10;
  auto rd = pfc::solvers::newton(d.ops, nd, ud, nc);
  auto rh = pfc::solvers::newton(h.ops, nh, uh, nc);
  INFO("Newton device " << rd.history.size() << " host " << rh.history.size());
  REQUIRE(rd.converged());
  REQUIRE(rd.history.size() == rh.history.size());
  REQUIRE(rel(d.gather(ud), h.gather(uh)) < 1e-10);
  REQUIRE(d.ops.scalar_transfer_bytes > 0);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}

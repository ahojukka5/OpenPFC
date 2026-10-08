// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <openpfc/kernel/field/fd_gradient.hpp>

namespace {
struct Mixed {
  double xy{}, xz{}, yz{};
};
struct XY {
  double xy{};
};
constexpr auto full = pfc::comm::HaloConnectivity::Full;

// Analytic coordinate sampling fills every ghost independently of FD weights.
template <class F> void fill(pfc::Field<double> &u, F fn) {
  const auto n = u.local_size();
  const auto h = u.spacing();
  const auto low = u.box().low;
  const int w = u.storage_halo();
  for (int k = -w; k < n[2] + w; ++k)
    for (int j = -w; j < n[1] + w; ++j)
      for (int i = -w; i < n[0] + w; ++i)
        u(i, j, k) =
            fn((low[0] + i) * h[0], (low[1] + j) * h[1], (low[2] + k) * h[2]);
}
} // namespace

TEST_CASE("Host mixed FD recovers polynomial Hessian with unequal spacing",
          "[fd_gradient][fd_mixed][unit]") {
  const auto d = pfc::domain::with_spacing({7, 9, 11}, {0.13, 0.27, 0.41});
  for (int order : {2, 4, 6, 8, 10, 12, 14}) {
    const int w = order / 2;
    pfc::Field<double> u(d, pfc::Box3i::from_bounds({0, 0, 0}, {6, 8, 10}), w);
    fill(u, [](double x, double y, double z) {
      return 2 * x * y - 3 * x * z + 5 * y * z + x * x;
    });
    const auto g = pfc::field::create<Mixed>(u, order, full);
    u.for_each_owned([&](int i, int j, int k) {
      const auto v = g(i, j, k);
      REQUIRE(v.xy == Catch::Approx(2).margin(2e-11));
      REQUIRE(v.xz == Catch::Approx(-3).margin(2e-11));
      REQUIRE(v.yz == Catch::Approx(5).margin(2e-11));
    });
  }
}

TEST_CASE("Host mixed FD fails closed for invalid configurations",
          "[fd_gradient][fd_mixed][unit]") {
  const auto d = pfc::domain::with_spacing({8, 9, 10}, {0.2, 0.3, 0.4});
  pfc::Field<double> u(d, pfc::Box3i::from_bounds({0, 0, 0}, {7, 8, 9}), 3);
  REQUIRE_THROWS_AS(pfc::field::create<Mixed>(u, 6), std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::field::create<Mixed>(u, 16, full), std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::field::create<Mixed>(u, 5, full), std::invalid_argument);
  pfc::Field<double> narrow(d, u.box(), 2);
  REQUIRE_THROWS_AS(pfc::field::create<Mixed>(narrow, 6, full),
                    std::invalid_argument);
  REQUIRE_THROWS_AS((pfc::gradient::FDGradient<Mixed>(u.data(), 8, 9, 10, 0, 0.3,
                                                      0.4, 3, 6, {}, full)),
                    std::invalid_argument);
  REQUIRE_THROWS_AS((pfc::gradient::FDGradient<Mixed>(u.data(), 4, 9, 10, 0.2, 0.3,
                                                      0.4, 3, 6, {}, full)),
                    std::invalid_argument);
  pfc::Field<double> thin(d, pfc::Box3i::from_bounds({0, 0, 0}, {1, 8, 9}), 3);
  REQUIRE_THROWS_AS(pfc::field::create<Mixed>(thin, 6, full), std::invalid_argument);
  const auto slab = pfc::domain::with_spacing({8, 9, 1}, {0.2, 0.3, 1});
  pfc::Field<double> s(slab, pfc::Box3i::from_bounds({0, 0, 0}, {7, 8, 0}), 3);
  REQUIRE_THROWS_AS(pfc::field::create<Mixed>(s, 6, full), std::invalid_argument);
  // Poison every off-plane z ghost. XY must only read the owned z plane.
  std::fill(s.vec().begin(), s.vec().end(),
            std::numeric_limits<double>::quiet_NaN());
  for (int j = -3; j < 12; ++j)
    for (int i = -3; i < 11; ++i) s(i, j, 0) = 2 * (i * 0.2) * (j * 0.3);
  const auto g = pfc::field::create<XY>(s, 6, full);
  s.for_each_owned(
      [&](int i, int j, int k) { REQUIRE(g(i, j, k).xy == Catch::Approx(2)); });
  std::array<double, 15 * 15> raw{};
  const auto rawg =
      pfc::gradient::FDGradient<XY>(raw.data(), 15, 15, 1, 1, 1, 1, 3, 6, {}, full);
  REQUIRE(rawg.kmin() == 0);
  REQUIRE(rawg.kmax() == 1);
}

TEST_CASE("Host mixed FD trig convergence includes periodic MPI corners",
          "[MPI][fd_mixed]") {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  REQUIRE((size == 1 || size == 2 || size == 4 || size == 8));
  const pfc::Int3 grid{size >= 2 ? 2 : 1, size >= 4 ? 2 : 1, size >= 8 ? 2 : 1};
  for (int order : {2, 4, 6}) {
    std::array<double, 2> errors{};
    for (int refinement = 0; refinement < 2; ++refinement) {
      const int n = 16 << refinement;
      const pfc::Int3 dims{n, n + 8 * (1 << refinement), n + 16 * (1 << refinement)};
      const double pi = std::numbers::pi;
      const pfc::Real3 h{2 * pi / dims[0], 3 * pi / dims[1], 5 * pi / dims[2]};
      const auto d = pfc::domain::with_spacing(dims, h);
      const auto dec = pfc::decomposition::create(d, grid);
      pfc::Field<double> u(d, pfc::decomposition::local_box(dec, rank), order / 2);
      const auto analytic = [](double x, double y, double z) {
        return std::sin(x) * std::cos(2 * y / 3) * std::sin(2 * z / 5);
      };
      std::fill(u.vec().begin(), u.vec().end(),
                std::numeric_limits<double>::quiet_NaN());
      const auto low = u.box().low;
      u.for_each_owned([&](int i, int j, int k) {
        u(i, j, k) =
            analytic((low[0] + i) * h[0], (low[1] + j) * h[1], (low[2] + k) * h[2]);
      });
      pfc::comm::HaloExchangeOptions opt;
      opt.connectivity = full;
      pfc::comm::HaloExchange<pfc::HostSpace, double> halo(u, dec, rank,
                                                           MPI_COMM_WORLD, opt);
      const auto g =
          pfc::gradient::FDGradient<Mixed>(u, order, [&] { halo.exchange(); }, full);
      auto prepared = g;
      prepared.prepare();
      double err = 0;
      u.for_each_owned([&](int i, int j, int k) {
        const double x = (low[0] + i) * h[0], y = (low[1] + j) * h[1],
                     z = (low[2] + k) * h[2];
        const auto v = prepared(i, j, k);
        REQUIRE(std::isfinite(v.xy));
        REQUIRE(std::isfinite(v.xz));
        REQUIRE(std::isfinite(v.yz));
        err = std::max({err,
                        std::abs(v.xy + 2. / 3 * std::cos(x) * std::sin(2 * y / 3) *
                                            std::sin(2 * z / 5)),
                        std::abs(v.xz - 2. / 5 * std::cos(x) * std::cos(2 * y / 3) *
                                            std::cos(2 * z / 5)),
                        std::abs(v.yz + 4. / 15 * std::sin(x) * std::sin(2 * y / 3) *
                                            std::cos(2 * z / 5))});
      });
      MPI_Allreduce(&err, &errors[refinement], 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
    }
    const double rate = std::log2(errors[0] / errors[1]);
    if (rank == 0)
      std::cout << "mixed_fd ranks=" << size << " order=" << order
                << " errors=" << errors[0] << "," << errors[1] << " rate=" << rate
                << '\n';
    REQUIRE(rate > order - 0.15);
  }
}

TEST_CASE("Host FD6 mixed derivatives respect physical and decomposition ghosts",
          "[MPI][fd_mixed]") {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  REQUIRE((size == 1 || size == 2 || size == 4 || size == 8));
  const auto domain = pfc::domain::with_spacing({24, 32, 40}, {0.13, 0.27, 0.41},
                                                {false, false, false});
  const auto dec = pfc::decomposition::create(
      domain, pfc::Int3{size >= 2 ? 2 : 1, size >= 4 ? 2 : 1, size >= 8 ? 2 : 1});
  pfc::Field<double> u(domain, pfc::decomposition::local_box(dec, rank), 3);
  // Exact polynomial boundary condition on all physical ghosts. Exchange
  // supplies decomposition ghosts; it must leave physical values consistent.
  fill(u, [](double x, double y, double z) {
    return 2 * x * y - 3 * x * z + 5 * y * z + x * x;
  });
  pfc::comm::HaloExchangeOptions opt;
  opt.connectivity = full;
  pfc::comm::HaloExchange<pfc::HostSpace, double> halo(u, dec, rank, MPI_COMM_WORLD,
                                                       opt);
  halo.exchange();
  const auto grad = pfc::field::create<Mixed>(u, 6, full);
  u.for_each_owned([&](int i, int j, int k) {
    const auto g = grad(i, j, k);
    REQUIRE(g.xy == Catch::Approx(2).margin(2e-10));
    REQUIRE(g.xz == Catch::Approx(-3).margin(2e-10));
    REQUIRE(g.yz == Catch::Approx(5).margin(2e-10));
  });
}

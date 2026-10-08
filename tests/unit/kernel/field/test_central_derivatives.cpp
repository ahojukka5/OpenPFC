// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fixtures/central_derivatives_cases.hpp>
#include <numbers>
using namespace derivative_cases;
using pfc::field::fd::evaluate;
namespace {
template <int Order> void polynomial() {
  Polynomial s;
  auto got = entries(evaluate<Order>(s, s.hx, s.hy, s.hz)),
       want = entries(s.exact());
  for (int j = 0; j < 9; ++j)
    REQUIRE(got[j] == Catch::Approx(want[j]).margin(1e-11));
  auto constant = [](int, int, int) { return 123456789.; };
  for (double value : entries(evaluate<Order>(constant, .25, .5, 1.)))
    REQUIRE(value == 0);
}
template <int Order> void convergence() {
  for (bool boundary : {false, true}) {
    std::array<double, 9> previous{};
    for (int n : {16, 32, 64}) {
      double h = std::numbers::pi * 2 / n;
      Result result, exact;
      if (boundary) {
        Reflected s{n};
        h = std::numbers::pi / n;
        REQUIRE(s(-1, 0, 0) == s(0, 0, 0));
        REQUIRE(s(-2, 0, 0) == s(1, 0, 0));
        result = evaluate<Order>(s, h, h, h);
        exact = s.exact();
      } else {
        Periodic s;
        s.h = h;
        result = evaluate<Order>(s, h, h, h);
        exact = s.exact();
      }
      auto got = entries(result), want = entries(exact);
      for (int j = 0; j < 9; ++j) {
        const double error = std::abs(got[j] - want[j]);
        REQUIRE(std::isfinite(got[j]));
        if (n > 16) REQUIRE(error < previous[j] / (std::pow(2., Order) * .7));
        previous[j] = error;
      }
    }
  }
}
} // namespace
TEST_CASE("Central derivatives reproduce analytic polynomial Hessians",
          "[fd][derivatives]") {
  polynomial<2>();
  polynomial<4>();
  polynomial<6>();
}
TEST_CASE("Central derivatives converge in all mixed entries and reflected cells",
          "[fd][derivatives]") {
  convergence<2>();
  convergence<4>();
  convergence<6>();
}
TEST_CASE("Node and cell-centred reflection use distinct coordinate origins",
          "[fd][derivatives]") {
  const double h = .125;
  auto node = [h](int x, int y, int z) {
    return cos(std::abs(x) * h) + cos(y * h) + cos(z * h);
  };
  auto cell = [h](int x, int y, int z) {
    int j = x < 0 ? -x - 1 : x;
    return cos((j + .5) * h) + cos(y * h) + cos(z * h);
  };
  auto a = evaluate<6>(node, h, h, h), b = evaluate<6>(cell, h, h, h);
  REQUIRE(a.x == 0);
  REQUIRE(b.x == Catch::Approx(-sin(h / 2)).margin(1e-8));
  REQUIRE(a.xx == Catch::Approx(-1).margin(1e-8));
  REQUIRE(b.xx == Catch::Approx(-cos(h / 2)).margin(1e-8));
  // A clamped high-order accessor is a different extension, not FD6 Neumann.
  auto clamped = [h](int x, int y, int z) {
    return cos((std::max(0, x) + .5) * h) + cos(y * h) + cos(z * h);
  };
  REQUIRE(std::abs(evaluate<6>(clamped, h, h, h).xx - b.xx) > 1e-3);
}

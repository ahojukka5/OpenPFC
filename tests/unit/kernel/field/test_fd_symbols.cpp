// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fd_symbols.cpp
 * @brief Fourier symbols of the shipped EvenCentralD2 tables.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include <openpfc/kernel/field/fd_symbols.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
namespace fd = pfc::field::fd;

TEST_CASE("d2_symbol matches the order-2 identity and vanishes at theta=0",
          "[field][fd-symbols][unit]") {
  for (double theta : {0.1, 0.7, 1.5, 3.0}) {
    const double s = std::sin(0.5 * theta);
    REQUIRE_THAT(fd::d2_symbol(2, theta), WithinRel(-4.0 * s * s, 1e-12));
  }
  for (int order : {2, 4, 6, 8, 10, 12, 20}) {
    REQUIRE_THAT(fd::d2_symbol(order, 0.0), WithinAbs(0.0, 1e-12));
    const double theta = 1e-3;
    REQUIRE_THAT(fd::d2_symbol(order, theta), WithinRel(-theta * theta, 1e-6));
    REQUIRE_THAT(fd::spectral_d2_symbol(theta), WithinRel(-theta * theta, 1e-15));
  }
  REQUIRE_THROWS(fd::d2_symbol(3, 0.5));
}

TEST_CASE("EvenCentralD2 is the truncated arcsin series",
          "[field][fd-symbols][unit]") {
  for (int order : {2, 4, 6, 8, 10, 12, 20}) {
    const int m_max = order / 2;
    for (double theta : {0.2, 0.8, 1.6, 2.4, 3.0}) {
      const double delta_sq = 2.0 - 2.0 * std::cos(theta);
      double series = 0.0;
      double power = 1.0;
      for (int m = 1; m <= m_max; ++m) {
        power *= delta_sq;
        series += fd::arcsin_series_coefficient(m) * power;
      }
      INFO("order " << order << " theta " << theta);
      REQUIRE_THAT(fd::d2_symbol(order, theta), WithinAbs(-series, 1e-13));
    }
  }
  REQUIRE_THAT(fd::arcsin_series_coefficient(1), WithinRel(1.0, 1e-15));
  REQUIRE_THAT(fd::arcsin_series_coefficient(2), WithinRel(1.0 / 12.0, 1e-15));
  REQUIRE_THAT(fd::arcsin_series_coefficient(3), WithinRel(1.0 / 90.0, 1e-15));
  {
    const double theta = 1.0;
    const double delta_sq = 2.0 - 2.0 * std::cos(theta);
    double series = 0.0, power = 1.0;
    for (int m = 1; m <= 200; ++m) {
      power *= delta_sq;
      series += fd::arcsin_series_coefficient(m) * power;
    }
    REQUIRE_THAT(series, WithinRel(theta * theta, 1e-14));
  }
}

TEST_CASE("d2_symbol_defect is positive and matches the direct difference",
          "[field][fd-symbols][unit]") {
  for (int order : {2, 4, 6, 8, 12}) {
    for (double theta : {0.4, 0.9, 1.4, 2.0, 2.8}) {
      const double d = fd::d2_symbol_defect(order, theta);
      REQUIRE(d > 0.0);
      const double direct = theta * theta + fd::d2_symbol(order, theta);
      REQUIRE_THAT(direct, WithinRel(d, 1e-5));
    }
  }
  const double tiny = fd::d2_symbol_defect(12, 0.1);
  REQUIRE(tiny > 0.0);
  REQUIRE(tiny < 1e-18);
  REQUIRE(tiny > 1e-20);
  const double d1 = fd::d2_symbol_defect(4, 0.02);
  const double d2 = fd::d2_symbol_defect(4, 0.01);
  REQUIRE_THAT(d1 / d2, WithinRel(64.0, 1e-3));
}

TEST_CASE("Nyquist theta=pi is a well-defined even D2 symbol",
          "[field][fd-symbols][unit]") {
  const double pi = std::acos(-1.0);
  for (int order : {2, 4, 8, 12}) {
    const double lam = fd::d2_symbol(order, pi);
    REQUIRE(lam < 0.0);
    REQUIRE(fd::d2_symbol_defect(order, pi) > 0.0);
    REQUIRE_THAT(fd::spectral_d2_symbol(pi), WithinRel(-pi * pi, 1e-15));
  }
}

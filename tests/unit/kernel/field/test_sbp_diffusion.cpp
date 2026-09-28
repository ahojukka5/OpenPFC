// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <openpfc/kernel/field/sbp_diffusion.hpp>
using namespace pfc::field::fd;
using Catch::Approx;
TEST_CASE("SBP diffusion conserves physical flux with variable material",
          "[unit][field][sbp]") {
  DiffusionAxis axis(33, 1. / 32);
  auto u = [](int i) {
    double x = i / 32.;
    return x * x;
  };
  auto d = [](int i) { return 1. + i / 64.; };
  FaceCondition left{BoundaryQuantity::NormalDerivative, [](double) { return 0.; }};
  FaceCondition right{BoundaryQuantity::NormalDerivative, [](double) { return 2.; }};
  double mass = 0, energy = 0;
  for (int i = 0; i < 33; ++i) {
    double value = axis.apply(i, u, d, .3, &left, &right);
    REQUIRE(value == Approx(2. + 2. * i / 32.).margin(1e-8));
    mass += axis.weight(i) * value;
    energy += axis.weight(i) * u(i) * axis.apply(i, u, d, .3);
  }
  REQUIRE(mass == Approx(3.).margin(1e-11));
  REQUIRE(energy <= 1e-11);
  auto one = [](int) { return 1.; };
  FaceCondition outgoing{BoundaryQuantity::ConstitutiveFlux,
                         [](double t) { return 2 * t; }};
  mass = 0;
  for (int i = 0; i < 33; ++i)
    mass += axis.weight(i) * axis.apply(i, one, d, .3, &outgoing, &outgoing);
  REQUIRE(mass == Approx(-1.2).margin(1e-11));
}
TEST_CASE("SBP face semantics and ledger fail closed", "[unit][field][sbp]") {
  auto one = [](int) { return 1.; };
  auto zero = [](int) { return 0.; };
  FaceCondition flux{BoundaryQuantity::ConstitutiveFlux, [](double) { return 1.; }};
  DiffusionAxis axis(33, .1), periodic(33, .1, true);
  REQUIRE_THROWS(axis.apply(0, one, zero, 0, &flux));
  REQUIRE_THROWS(periodic.apply(4, one, one, 0, &flux));
  REQUIRE_THROWS(DiffusionAxis(16, .1));
  REQUIRE_THROWS(axis.apply(
      2, [](int) { return std::numeric_limits<double>::quiet_NaN(); }, one, 0));
  FluxLedger ledger;
  REQUIRE_THROWS(ledger.stage(1, 1, .1));
  ledger.begin();
  ledger.stage(2, 3, .25);
  ledger.reject();
  REQUIRE(ledger.accepted() == 0);
  ledger.begin();
  ledger.stage(2, 3, .25);
  ledger.stage(4, 3, .25);
  ledger.accept();
  REQUIRE(ledger.accepted() == Approx(4.5));
  ledger.begin();
  REQUIRE_THROWS(ledger.stage(1, -1, 1));
  REQUIRE_THROWS(ledger.accept());
  ledger.reject();
  REQUIRE(ledger.accepted() == Approx(4.5));
}

namespace {
std::pair<double, std::vector<double>> manufactured(int intervals, bool variable,
                                                    bool closed, int multiplier) {
  const int n = intervals + 1;
  const double h = 1. / intervals, pi = std::acos(-1.);
  DiffusionAxis axis(n, h);
  std::vector<double> u(n), d(n), force(n), initial(n);
  const double rate = closed ? pi * pi : 1.;
  for (int i = 0; i < n; ++i) {
    const double x = i * h;
    initial[i] = closed ? std::cos(pi * x) : 2 + std::cos(pi * x) + x;
    d[i] = variable ? 1 + x / 2 : 1;
    const double ux = 1 - pi * std::sin(pi * x), uxx = -pi * pi * std::cos(pi * x);
    force[i] = closed ? 0 : -initial[i] - d[i] * uxx - (variable ? .5 * ux : 0);
  }
  u = initial;
  int steps =
      static_cast<int>(std::ceil(.02 * intervals * intervals * 20)) * multiplier;
  const double dt = .02 / steps;
  FaceCondition left{BoundaryQuantity::ConstitutiveFlux,
                     [&](double t) { return closed ? 0 : d[0] * std::exp(-t); }};
  FaceCondition right{BoundaryQuantity::ConstitutiveFlux, [&](double t) {
                        return closed ? 0 : -d.back() * std::exp(-t);
                      }};
  auto rhs = [&](const std::vector<double> &state, double t,
                 std::vector<double> &out) {
    for (int i = 0; i < n; ++i)
      out[i] = axis.apply(
                   i, [&](int j) { return state[j]; }, [&](int j) { return d[j]; },
                   t, &left, &right) +
               force[i] * std::exp(-rate * t);
  };
  std::vector<double> a(n), b(n), c(n), e(n), v(n);
  for (int step = 0; step < steps; ++step) {
    double t = step * dt;
    rhs(u, t, a);
    for (int i = 0; i < n; ++i) v[i] = u[i] + dt * a[i] / 2;
    rhs(v, t + dt / 2, b);
    for (int i = 0; i < n; ++i) v[i] = u[i] + dt * b[i] / 2;
    rhs(v, t + dt / 2, c);
    for (int i = 0; i < n; ++i) v[i] = u[i] + dt * c[i];
    rhs(v, t + dt, e);
    for (int i = 0; i < n; ++i) u[i] += dt * (a[i] + 2 * b[i] + 2 * c[i] + e[i]) / 6;
  }
  double error = 0, mass = 0;
  for (int i = 0; i < n; ++i) {
    double diff = u[i] - initial[i] * std::exp(-rate * .02);
    error += axis.weight(i) * diff * diff;
    mass += axis.weight(i) * u[i];
  }
  return {std::sqrt(error), u};
}
} // namespace
TEST_CASE("SBP manufactured evolution converges independently",
          "[unit][field][sbp]") {
  for (int mode = 0; mode < 3; ++mode) {
    double previous = 0;
    for (int n : {32, 64, 128}) {
      auto full = manufactured(n, mode == 1, mode == 2, 1);
      auto half = manufactured(n, mode == 1, mode == 2, 2);
      DiffusionAxis axis(n + 1, 1. / n);
      double temporal = 0;
      for (int i = 0; i <= n; ++i) {
        double v = full.second[i] - half.second[i];
        temporal += axis.weight(i) * v * v;
      }
      REQUIRE(std::sqrt(temporal) < .01 * full.first);
      std::cout << std::setprecision(17) << "SBP_CONVERGENCE " << mode << " " << n
                << " " << full.first << " " << std::sqrt(temporal) << "\n";
      if (previous) {
        double order = std::log2(previous / full.first);
        INFO(mode << " " << n << " " << order);
        REQUIRE(order >= 3.5);
      }
      previous = full.first;
    }
  }
}
TEST_CASE("Tensor SBP weights conserve mixed periodic and physical faces",
          "[unit][field][sbp]") {
  const int n = 17;
  const double h = .1, pi = std::acos(-1.);
  DiffusionAxis xaxis(n, h, true), yaxis(n, h);
  auto u = [&](int i, int j) { return std::cos(2 * pi * i / n) + j * j * h * h; };
  auto d = [&](int i, int j) { return 2 + .1 * std::sin(2 * pi * i / n) + .02 * j; };
  double inventory = 0, flux = 0, scale = 1;
  for (int i = 0; i < n; ++i) {
    FaceCondition bottom{BoundaryQuantity::NormalDerivative,
                         [](double) { return 0.; }};
    FaceCondition top{BoundaryQuantity::NormalDerivative,
                      [&](double) { return 2 * (n - 1) * h; }};
    flux += xaxis.weight(i) * top.flux(0, d(i, n - 1));
    for (int j = 0; j < n; ++j) {
      double rhs =
          xaxis.apply(
              i, [&](int k) { return u(k, j); }, [&](int k) { return d(k, j); }, 0) +
          yaxis.apply(
              j, [&](int k) { return u(i, k); }, [&](int k) { return d(i, k); }, 0,
              &bottom, &top);
      const double term = xaxis.weight(i) * yaxis.weight(j) * rhs;
      inventory += term;
      scale += std::abs(term);
    }
  }
  REQUIRE(std::abs(inventory + flux) < 1e-11 * scale);
  FaceCondition left{BoundaryQuantity::NormalDerivative, [](double) { return -1.; }};
  FaceCondition right{BoundaryQuantity::NormalDerivative, [](double) { return 1.; }};
  for (int i = 0; i < n; ++i)
    REQUIRE(yaxis.apply(
                i, [&](int j) { return j * h; }, [](int) { return 2.; }, 0, &left,
                &right) == Approx(0).margin(1e-10));
}
TEST_CASE("Time dependent physical flux matches accepted inventory",
          "[unit][field][sbp]") {
  constexpr int n = 17;
  DiffusionAxis axis(n, 1. / 16);
  std::vector<double> u(n), a(n), b(n), c(n), d(n), v(n);
  FaceCondition left{BoundaryQuantity::ConstitutiveFlux,
                     [](double t) { return t * t; }};
  FluxLedger ledger;
  ledger.begin();
  ledger.stage(999, 1, 1);
  ledger.reject();
  auto rhs = [&](const std::vector<double> &state, double t,
                 std::vector<double> &out) {
    for (int i = 0; i < n; ++i)
      out[i] = axis.apply(
          i, [&](int j) { return state[j]; }, [](int) { return 1.; }, t, &left);
  };
  const double dt = 1e-5, start = .2;
  const int steps = 100;
  for (int k = 0; k < steps; ++k) {
    double t = start + k * dt;
    ledger.begin();
    rhs(u, t, a);
    for (int i = 0; i < n; ++i) v[i] = u[i] + dt * a[i] / 2;
    rhs(v, t + dt / 2, b);
    for (int i = 0; i < n; ++i) v[i] = u[i] + dt * b[i] / 2;
    rhs(v, t + dt / 2, c);
    for (int i = 0; i < n; ++i) v[i] = u[i] + dt * c[i];
    rhs(v, t + dt, d);
    for (int i = 0; i < n; ++i) u[i] += dt * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) / 6;
    ledger.stage(t * t, 1, dt / 6);
    ledger.stage((t + dt / 2) * (t + dt / 2), 1, 2 * dt / 3);
    ledger.stage((t + dt) * (t + dt), 1, dt / 6);
    ledger.accept();
  }
  double inventory = 0;
  for (int i = 0; i < n; ++i) inventory += axis.weight(i) * u[i];
  double end = start + steps * dt,
         exact = (end * end * end - start * start * start) / 3;
  REQUIRE(inventory == Approx(-exact).margin(1e-11));
  REQUIRE(ledger.accepted() == Approx(exact).margin(1e-11));
  REQUIRE(inventory + ledger.accepted() == Approx(0).margin(1e-11));
}

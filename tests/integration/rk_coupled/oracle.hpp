// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "model.hpp"
#include <algorithm>
#include <array>
#include <numbers>
#include <stdexcept>
#include <vector>
namespace coupled_rk {
inline void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
using Plane = std::vector<double>;
using Pair = std::array<Plane, 2>;
inline std::size_t index(int i, int j, int k, int n) {
  auto wrap = [n](int x) { return (x % n + n) % n; };
  return std::size_t(wrap(i)) + std::size_t(n) * (wrap(j) + n * wrap(k));
}
inline Pair initial(int n, bool uniform = false) {
  Pair q{Plane(std::size_t(n) * n * n), Plane(std::size_t(n) * n * n)};
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const double mode = std::cos(2 * std::numbers::pi * (i + 2 * j + 3 * k) / n);
        q[0][index(i, j, k, n)] = uniform ? 1. : mode;
        q[1][index(i, j, k, n)] = uniform ? 1. : .6 * mode;
      }
  return q;
}
// Independent periodic dense-array stencil, deliberately not the library FD
// helpers/evaluators. Explicit D1 and D2 weights implement sixth order.
inline Pair spatial_rhs(const Pair &q, int n, double t, const Model &m) {
  Pair out{Plane(q[0].size()), Plane(q[1].size())};
  const double h = 2 * std::numbers::pi / n;
  const double d1[7] = {-1. / 60, 3. / 20, -3. / 4, 0, 3. / 4, -3. / 20, 1. / 60};
  const double d2[7] = {1. / 90, -3. / 20, 3. / 2, -49. / 18,
                        3. / 2,  -3. / 20, 1. / 90};
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        auto at = index(i, j, k, n);
        double ux = 0, lap = 0, uxy = 0, uyz = 0, vxz = 0;
        for (int s = -3; s <= 3; ++s) {
          ux += d1[s + 3] * q[0][index(i + s, j, k, n)] / h;
          lap += d2[s + 3] *
                 (q[0][index(i + s, j, k, n)] + q[0][index(i, j + s, k, n)] +
                  q[0][index(i, j, k + s, n)]) /
                 (h * h);
          for (int r = -3; r <= 3; ++r) {
            const double w = d1[s + 3] * d1[r + 3] / (h * h);
            uxy += w * q[0][index(i + s, j + r, k, n)];
            uyz += w * q[0][index(i, j + s, k + r, n)];
            vxz += w * q[1][index(i + s, j, k + r, n)];
          }
        }
        const double u = q[0][at], v = q[1][at];
        // Written independently of Model::rhs and aggregate/scatter machinery.
        out[0][at] =
            v + m.a * uxy + m.first * ux + m.nonlinear * u * u + m.time_source * t;
        out[1][at] = m.b * lap + m.c * vxz + m.d * uyz - m.nonlinear * u * v +
                     m.time_source * std::sin(t);
        if (m.manufactured_uniform) {
          out[0][at] += std::exp(t) - std::exp(-t) - m.nonlinear * std::exp(2 * t) -
                        m.time_source * t;
          out[1][at] += -std::exp(-t) + m.nonlinear - m.time_source * std::sin(t);
        }
      }
  return out;
}
struct Oracle {
  std::array<Pair, 4> inputs, slopes;
  std::array<double, 4> times{};
  Pair result;
  int stages{};
  Oracle(const Pair &accepted, int n, double t, double dt, bool fourth,
         const Model &m, bool spectral_mode = false) {
    stages = fourth ? 4 : 2;
    // These are independent midpoint/classical RK coefficients, not reads of
    // the production ButcherTableau or calls to the production stepper.
    const double c[4] = {0, .5, .5, 1};
    const double a[4][4] = {
        {0, 0, 0, 0}, {.5, 0, 0, 0}, {0, .5, 0, 0}, {0, 0, 1, 0}};
    const double b4[4] = {1. / 6, 1. / 3, 1. / 3, 1. / 6};
    for (int s = 0; s < stages; ++s) {
      inputs[s] = accepted;
      times[s] = t + c[s] * dt;
      for (int f = 0; f < 2; ++f)
        for (std::size_t x = 0; x < accepted[f].size(); ++x)
          for (int r = 0; r < s; ++r)
            inputs[s][f][x] += dt * a[s][r] * slopes[r][f][x];
      if (spectral_mode) {
        slopes[s] = inputs[s];
        for (std::size_t x = 0; x < accepted[0].size(); ++x) {
          const double u = inputs[s][0][x], v = inputs[s][1][x];
          slopes[s][0][x] = v - 2 * m.a * u;
          slopes[s][1][x] = (-14 * m.b - 6 * m.d) * u - 3 * m.c * v;
        }
      } else
        slopes[s] = spatial_rhs(inputs[s], n, times[s], m);
    }
    result = accepted;
    for (int f = 0; f < 2; ++f)
      for (std::size_t x = 0; x < accepted[f].size(); ++x)
        for (int s = 0; s < stages; ++s)
          result[f][x] +=
              dt * (fourth ? b4[s] : (s == 1 ? 1. : 0)) * slopes[s][f][x];
  }
};
} // namespace coupled_rk

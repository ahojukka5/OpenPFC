// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fourier_vector.hpp
 * @brief Periodic Fourier Leray projection.
 *
 * Wave numbers are the odd multipliers from `k_component_odd`: the even-grid
 * Nyquist component is zero, so a purely real Nyquist coefficient is not
 * given an imaginary direction. A mode whose odd wavevector vanishes,
 * including the mean, is left unchanged by the projection (`P = I` there).
 *
 * These operators stay `pfc::field`. The rotational Navier–Stokes step
 * that calls them is `pfc::incompressible` in `solvers/incompressible/`.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/fft/box3i.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>

namespace pfc::field {

using Complex = std::complex<double>;

struct OddWave {
  double kx{0.0};
  double ky{0.0};
  double kz{0.0};
  double k2{0.0};
};

[[nodiscard]] inline OddWave odd_wave(int i, int j, int k, std::array<int, 3> n,
                                      std::array<double, 3> spacing) noexcept {
  const double fx = two_pi / (spacing[0] * static_cast<double>(n[0]));
  const double fy = two_pi / (spacing[1] * static_cast<double>(n[1]));
  const double fz = two_pi / (spacing[2] * static_cast<double>(n[2]));
  OddWave w;
  w.kx = fft::kspace::k_component_odd(i, n[0], fx);
  w.ky = fft::kspace::k_component_odd(j, n[1], fy);
  w.kz = fft::kspace::k_component_odd(k, n[2], fz);
  w.k2 = w.kx * w.kx + w.ky * w.ky + w.kz * w.kz;
  return w;
}

inline void require_hat(const fft::Box3i &outbox, std::size_t n) {
  const std::size_t expected = static_cast<std::size_t>(outbox.size[0]) *
                               static_cast<std::size_t>(outbox.size[1]) *
                               static_cast<std::size_t>(outbox.size[2]);
  if (n != expected) {
    throw std::invalid_argument(
        "incompressible: hat length must equal the outbox volume");
  }
}

/// `P = I - k kᵀ / |k|²` on one mode. A vanishing odd wavevector is unchanged.
inline void leray_mode(Complex &u, Complex &v, Complex &w, OddWave wave) noexcept {
  if (wave.k2 == 0.0) return;
  const Complex dot = wave.kx * u + wave.ky * v + wave.kz * w;
  const Complex scale = dot / wave.k2;
  u -= wave.kx * scale;
  v -= wave.ky * scale;
  w -= wave.kz * scale;
}

inline void leray_project(const fft::Box3i &outbox, std::array<int, 3> n,
                          std::array<double, 3> spacing, Complex *u, Complex *v,
                          Complex *w, std::size_t count) {
  require_hat(outbox, count);
  fft::kspace::for_each_kpoint(
      outbox, n, spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        leray_mode(u[idx], v[idx], w[idx], odd_wave(i, j, k, n, spacing));
      });
}

inline void apply_two_thirds(const fft::Box3i &outbox, std::array<int, 3> n,
                             std::array<double, 3> spacing, Complex *u, Complex *v,
                             Complex *w, std::size_t count) {
  require_hat(outbox, count);
  fft::kspace::for_each_kpoint(
      outbox, n, spacing,
      [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
        if (fft::kspace::two_thirds_keep(kx, ky, kz, spacing)) return;
        u[idx] = v[idx] = w[idx] = Complex{};
      });
}

[[nodiscard]] inline Complex modal_divergence(Complex u, Complex v, Complex w,
                                              OddWave wave) noexcept {
  return Complex(0.0, wave.kx) * u + Complex(0.0, wave.ky) * v +
         Complex(0.0, wave.kz) * w;
}

[[nodiscard]] inline double
max_modal_divergence(const fft::Box3i &outbox, std::array<int, 3> n,
                     std::array<double, 3> spacing, const Complex *u,
                     const Complex *v, const Complex *w, std::size_t count) {
  require_hat(outbox, count);
  double max_abs = 0.0;
  fft::kspace::for_each_kpoint(
      outbox, n, spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        const auto div =
            modal_divergence(u[idx], v[idx], w[idx], odd_wave(i, j, k, n, spacing));
        max_abs = std::max(max_abs, std::abs(div));
      });
  return max_abs;
}

inline void curl_hat(const fft::Box3i &outbox, std::array<int, 3> n,
                     std::array<double, 3> spacing, const Complex *u,
                     const Complex *v, const Complex *w, Complex *ox, Complex *oy,
                     Complex *oz, std::size_t count) {
  require_hat(outbox, count);
  fft::kspace::for_each_kpoint(
      outbox, n, spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        const auto wave = odd_wave(i, j, k, n, spacing);
        const Complex ikx(0.0, wave.kx);
        const Complex iky(0.0, wave.ky);
        const Complex ikz(0.0, wave.kz);
        ox[idx] = iky * w[idx] - ikz * v[idx];
        oy[idx] = ikz * u[idx] - ikx * w[idx];
        oz[idx] = ikx * v[idx] - iky * u[idx];
      });
}

inline void gradient_hat(const fft::Box3i &outbox, std::array<int, 3> n,
                         std::array<double, 3> spacing, const Complex *phi,
                         Complex *gx, Complex *gy, Complex *gz, std::size_t count) {
  require_hat(outbox, count);
  fft::kspace::for_each_kpoint(
      outbox, n, spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        const auto wave = odd_wave(i, j, k, n, spacing);
        gx[idx] = Complex(0.0, wave.kx) * phi[idx];
        gy[idx] = Complex(0.0, wave.ky) * phi[idx];
        gz[idx] = Complex(0.0, wave.kz) * phi[idx];
      });
}

/// Multiply by `-|k_odd|²`. The mean and the odd-Nyquist modes stay put.
inline void laplacian_hat(const fft::Box3i &outbox, std::array<int, 3> n,
                          std::array<double, 3> spacing, Complex *hat,
                          std::size_t count) {
  require_hat(outbox, count);
  fft::kspace::for_each_kpoint(
      outbox, n, spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        hat[idx] *= -odd_wave(i, j, k, n, spacing).k2;
      });
}

inline void zero_mean(const fft::Box3i &outbox, std::array<int, 3> n,
                      std::array<double, 3> spacing, Complex *u, Complex *v,
                      Complex *w, std::size_t count) {
  require_hat(outbox, count);
  fft::kspace::for_each_kpoint(
      outbox, n, spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        if (i == 0 && j == 0 && k == 0) u[idx] = v[idx] = w[idx] = Complex{};
      });
}

} // namespace pfc::field

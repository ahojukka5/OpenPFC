// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file incompressible.hpp
 * @brief Periodic Fourier Leray projection and a dealiased rotational term.
 *
 * Wave numbers are the odd multipliers from `k_component_odd`: the even-grid
 * Nyquist component is zero, so a purely real Nyquist coefficient is not
 * given an imaginary direction. A mode whose odd wavevector vanishes,
 * including the mean, is left unchanged by the projection (`P = I` there).
 *
 * The momentum equation adds `-P((u · ∇)u)`. On a divergence-free field that
 * equals `P(u × ω)`, because `(u · ∇)u = ∇(|u|²/2) - u × ω` and `P` removes
 * the gradient. The installed term is that rotational form. Inside the
 * resolved band it matches the projected advective and skew-symmetric forms;
 * the unit test is that comparison. A 3/2 padded product is not used.
 *
 * The term is evaluated after a state-level 2/3 truncation. The mean of the
 * tendency is set to zero: a periodic flux has no mean, and a leftover mean
 * is an alias rather than a force. The caller adds a mean force on top of
 * the tendency. Viscosity is the diagonal symbol `-ν |k_odd|²`, which is
 * zero on the same modes, so an unforced mean velocity is invariant.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/fft/box3i.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
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

inline void viscous_exponentials(const fft::Box3i &outbox, std::array<int, 3> n,
                                 std::array<double, 3> spacing, double nu, double dt,
                                 std::vector<double> &exp_dt,
                                 std::vector<double> &exp_half) {
  const std::size_t count = static_cast<std::size_t>(outbox.size[0]) *
                            static_cast<std::size_t>(outbox.size[1]) *
                            static_cast<std::size_t>(outbox.size[2]);
  exp_dt.assign(count, 1.0);
  exp_half.assign(count, 1.0);
  fft::kspace::for_each_kpoint(
      outbox, n, spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        const double decay = -nu * odd_wave(i, j, k, n, spacing).k2;
        exp_dt[idx] = std::exp(decay * dt);
        exp_half[idx] = std::exp(decay * 0.5 * dt);
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

/**
 * @brief `P(u × ω)` of the r2c velocity.
 *
 * When `dealias` is set, the state and the tendency are multiplied by the
 * 2/3 mask. The mask is the alias control for this quadratic term. The mean
 * of the tendency is zero either way. The caller's velocity is not modified.
 */
inline void rotational_tendency(fft::IHostFFT &fft, std::array<int, 3> n,
                                std::array<double, 3> spacing,
                                const std::vector<Complex> &u,
                                const std::vector<Complex> &v,
                                const std::vector<Complex> &w,
                                std::vector<Complex> &tu, std::vector<Complex> &tv,
                                std::vector<Complex> &tw, bool dealias = true) {
  const auto outbox = fft.get_outbox_bounds();
  const std::size_t nhat = fft.size_outbox();
  const std::size_t nreal = fft.size_inbox();
  if (u.size() != nhat || v.size() != nhat || w.size() != nhat) {
    throw std::invalid_argument("incompressible: velocity hat size mismatch");
  }
  tu.assign(nhat, Complex{});
  tv.assign(nhat, Complex{});
  tw.assign(nhat, Complex{});
  std::vector<Complex> uh = u;
  std::vector<Complex> vh = v;
  std::vector<Complex> wh = w;
  if (dealias) {
    apply_two_thirds(outbox, n, spacing, uh.data(), vh.data(), wh.data(), nhat);
  }
  std::vector<Complex> ox(nhat), oy(nhat), oz(nhat);
  curl_hat(outbox, n, spacing, uh.data(), vh.data(), wh.data(), ox.data(), oy.data(),
           oz.data(), nhat);
  std::vector<double> ur(nreal), vr(nreal), wr(nreal), oxr(nreal), oyr(nreal),
      ozr(nreal);
  fft.backward(uh, ur);
  fft.backward(vh, vr);
  fft.backward(wh, wr);
  fft.backward(ox, oxr);
  fft.backward(oy, oyr);
  fft.backward(oz, ozr);
  std::vector<double> tx(nreal), ty(nreal), tz(nreal);
  for (std::size_t i = 0; i < nreal; ++i) {
    tx[i] = vr[i] * ozr[i] - wr[i] * oyr[i];
    ty[i] = wr[i] * oxr[i] - ur[i] * ozr[i];
    tz[i] = ur[i] * oyr[i] - vr[i] * oxr[i];
  }
  fft.forward(tx, tu);
  fft.forward(ty, tv);
  fft.forward(tz, tw);
  leray_project(outbox, n, spacing, tu.data(), tv.data(), tw.data(), nhat);
  if (dealias) {
    apply_two_thirds(outbox, n, spacing, tu.data(), tv.data(), tw.data(), nhat);
  }
  zero_mean(outbox, n, spacing, tu.data(), tv.data(), tw.data(), nhat);
}

/// Integrating-factor RK4. `L = -ν |k_odd|²` is carried by `exp_dt` and `exp_half`.
template <class Tend>
void ifrk4_velocity(const fft::Box3i &outbox, std::array<int, 3> n,
                    std::array<double, 3> spacing, std::vector<Complex> &u,
                    std::vector<Complex> &v, std::vector<Complex> &w,
                    const std::vector<double> &exp_dt,
                    const std::vector<double> &exp_half, double dt, Tend &&tendency,
                    bool dealias = true) {
  const std::size_t count = u.size();
  require_hat(outbox, count);
  if (v.size() != count || w.size() != count || exp_dt.size() != count ||
      exp_half.size() != count) {
    throw std::invalid_argument("incompressible: RK4 buffer size mismatch");
  }
  auto mask = [&](std::vector<Complex> &a, std::vector<Complex> &b,
                  std::vector<Complex> &c) {
    leray_project(outbox, n, spacing, a.data(), b.data(), c.data(), count);
    if (dealias) {
      apply_two_thirds(outbox, n, spacing, a.data(), b.data(), c.data(), count);
    }
  };
  std::vector<Complex> n1u, n1v, n1w, n2u, n2v, n2w, n3u, n3v, n3w, n4u, n4v, n4w;
  std::vector<Complex> su(count), sv(count), sw(count);
  mask(u, v, w);
  tendency(u, v, w, n1u, n1v, n1w);
  for (std::size_t i = 0; i < count; ++i) {
    su[i] = exp_half[i] * (u[i] + (0.5 * dt) * n1u[i]);
    sv[i] = exp_half[i] * (v[i] + (0.5 * dt) * n1v[i]);
    sw[i] = exp_half[i] * (w[i] + (0.5 * dt) * n1w[i]);
  }
  mask(su, sv, sw);
  tendency(su, sv, sw, n2u, n2v, n2w);
  for (std::size_t i = 0; i < count; ++i) {
    su[i] = exp_half[i] * u[i] + (0.5 * dt) * n2u[i];
    sv[i] = exp_half[i] * v[i] + (0.5 * dt) * n2v[i];
    sw[i] = exp_half[i] * w[i] + (0.5 * dt) * n2w[i];
  }
  mask(su, sv, sw);
  tendency(su, sv, sw, n3u, n3v, n3w);
  for (std::size_t i = 0; i < count; ++i) {
    su[i] = exp_dt[i] * u[i] + dt * exp_half[i] * n3u[i];
    sv[i] = exp_dt[i] * v[i] + dt * exp_half[i] * n3v[i];
    sw[i] = exp_dt[i] * w[i] + dt * exp_half[i] * n3w[i];
  }
  mask(su, sv, sw);
  tendency(su, sv, sw, n4u, n4v, n4w);
  const double dt6 = dt / 6.0;
  for (std::size_t i = 0; i < count; ++i) {
    u[i] = exp_dt[i] * u[i] + dt6 * (exp_dt[i] * n1u[i] +
                                     2.0 * exp_half[i] * (n2u[i] + n3u[i]) + n4u[i]);
    v[i] = exp_dt[i] * v[i] + dt6 * (exp_dt[i] * n1v[i] +
                                     2.0 * exp_half[i] * (n2v[i] + n3v[i]) + n4v[i]);
    w[i] = exp_dt[i] * w[i] + dt6 * (exp_dt[i] * n1w[i] +
                                     2.0 * exp_half[i] * (n2w[i] + n3w[i]) + n4w[i]);
  }
  mask(u, v, w);
}

} // namespace pfc::field

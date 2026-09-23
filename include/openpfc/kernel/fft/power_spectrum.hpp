// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file power_spectrum.hpp
 * @brief Power of a stored real-to-complex spectrum.
 *
 * Backend-independent bookkeeping for an OpenPFC r2c outbox. The caller
 * owns what the real field means. This header owns which stored mode
 * stands for an unstored conjugate, how `|k|` shells are reduced, and
 * which axis dominates a wavevector.
 *
 * The forward transform is unnormalized: `hat[k] = sum_n u_n exp(-ik·x)`.
 * The backward transform divides by the global number of real samples.
 * `weighted_power` is therefore `N` times the real-space sum of squares
 * (Parseval), once every stored coefficient is given its conjugate weight.
 *
 * An even grid stores `kx = 0` and the `kx` Nyquist mode once; each is
 * its own conjugate (`weight = 1`). Every other `kx` is stored once and
 * stands for itself and `-kx` (`weight = 2`). Applying weight 2 to the
 * Nyquist mode, or weight 1 to an ordinary `kx`, mis-counts the full
 * spectrum. Mixed shells therefore differ from an unweighted average of
 * the stored coefficients. That difference is the correction.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>

namespace pfc::fft {

/**
 * @brief Weight of one stored r2c coefficient in the full spectrum.
 *
 * @param index  x index in the outbox, in `[0, extent/2]`
 * @param extent global number of real samples along x
 */
[[nodiscard]] constexpr double r2c_multiplicity(int index, int extent) noexcept {
  if (index == 0) return 1.0;
  if (extent % 2 == 0 && index == extent / 2) return 1.0;
  return 2.0;
}

/**
 * @brief `sum weight |hat|^2` over every stored mode, including `k = 0`.
 *
 * Reduced over @p comm. Divide by the global real-sample count to recover
 * `sum u^2`.
 */
[[nodiscard]] inline double weighted_power(const pfc::Box3i &outbox,
                                           const pfc::Domain &domain,
                                           const std::complex<double> *spectrum,
                                           MPI_Comm comm) {
  const auto size = pfc::domain::get_size(domain);
  double local = 0.0;
  pfc::fft::kspace::for_each_kpoint(
      outbox, domain, [&](std::size_t i, double, double, double, int ix, int, int) {
        local += r2c_multiplicity(ix, size[0]) * std::norm(spectrum[i]);
      });
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global;
}

/**
 * @brief One radial binning of a spectrum. The zero mode is omitted.
 *
 * Shells run from 0 to the Nyquist radius of the coarsest active axis,
 * `k_max = pi / dx`. Bin `b` covers `[b, b+1) * k_max / n_bins` and is
 * reported at its centre. A mode with `|k| >= k_max` (the corner of a
 * square or cubic Fourier box) is not in any shell.
 *
 * `mean_power` is the multiplicity-weighted mean of `|hat|^2` in the
 * shell. `integrated_power` is the sum of those weighted values, so
 * `mean_power = integrated_power / (weighted mode count)`.
 * `total_power` sums `integrated_power`. It is not a Parseval identity:
 * `k = 0` and modes outside the ball are absent. `first_moment` is the
 * first moment of the shell *means*, `sum k mean_power / sum mean_power`,
 * not of the integrated power.
 */
struct RadialSpectrum {
  std::vector<double> wavenumber;
  std::vector<double> mean_power;
  std::vector<double> integrated_power;
  double first_moment{0.0};
  double peak_wavenumber{0.0};
  /// Largest `mean_power`.
  double peak_power{0.0};
  /// Sum of `integrated_power` over the shells that were kept.
  double total_power{0.0};

  /// `2 pi / first_moment`. Zero when no shell has mean power.
  [[nodiscard]] double mean_wavelength() const {
    return (first_moment > 0.0) ? 2.0 * std::numbers::pi / first_moment : 0.0;
  }
  /// `2 pi / peak_wavenumber`.
  [[nodiscard]] double dominant_wavelength() const {
    return (peak_wavenumber > 0.0) ? 2.0 * std::numbers::pi / peak_wavenumber : 0.0;
  }
};

/**
 * @brief Bin `|hat|^2` into shells of `|k|`.
 *
 * @param n_bins shells from 0 to `k_max`. Empty shells are omitted.
 */
[[nodiscard]] inline RadialSpectrum
radial_average(const pfc::Box3i &outbox, const pfc::Domain &domain,
               const std::complex<double> *spectrum, MPI_Comm comm,
               int n_bins = 64) {
  const auto size = pfc::domain::get_size(domain);
  const auto dx = pfc::domain::get_spacing(domain);

  double k_max = 0.0;
  for (int d = 0; d < 3; ++d) {
    if (size[d] > 1) k_max = std::max(k_max, std::numbers::pi / dx[d]);
  }
  if (k_max <= 0.0 || n_bins < 1) return {};

  const double bin_width = k_max / static_cast<double>(n_bins);
  std::vector<double> power(static_cast<std::size_t>(n_bins), 0.0);
  std::vector<double> counts(static_cast<std::size_t>(n_bins), 0.0);

  pfc::fft::kspace::for_each_kpoint(
      outbox, domain,
      [&](std::size_t i, double kx, double ky, double kz, int ix, int, int) {
        const double kk = std::sqrt(kx * kx + ky * ky + kz * kz);
        if (kk <= 0.0) return;
        const int bin = static_cast<int>(kk / bin_width);
        if (bin < 0 || bin >= n_bins) return;
        const double weight = r2c_multiplicity(ix, size[0]);
        const auto b = static_cast<std::size_t>(bin);
        power[b] += weight * std::norm(spectrum[i]);
        counts[b] += weight;
      });

  std::vector<double> global_power(static_cast<std::size_t>(n_bins), 0.0);
  std::vector<double> global_counts(static_cast<std::size_t>(n_bins), 0.0);
  MPI_Allreduce(power.data(), global_power.data(), n_bins, MPI_DOUBLE, MPI_SUM,
                comm);
  MPI_Allreduce(counts.data(), global_counts.data(), n_bins, MPI_DOUBLE, MPI_SUM,
                comm);

  RadialSpectrum out;
  out.wavenumber.reserve(static_cast<std::size_t>(n_bins));
  out.mean_power.reserve(static_cast<std::size_t>(n_bins));
  out.integrated_power.reserve(static_cast<std::size_t>(n_bins));
  double moment = 0.0;
  double mean_sum = 0.0;
  for (int b = 0; b < n_bins; ++b) {
    const auto bi = static_cast<std::size_t>(b);
    if (global_counts[bi] <= 0.0) continue;
    const double kc = (static_cast<double>(b) + 0.5) * bin_width;
    const double mean = global_power[bi] / global_counts[bi];
    out.wavenumber.push_back(kc);
    out.mean_power.push_back(mean);
    out.integrated_power.push_back(global_power[bi]);
    out.total_power += global_power[bi];
    mean_sum += mean;
    moment += kc * mean;
    if (mean > out.peak_power) {
      out.peak_power = mean;
      out.peak_wavenumber = kc;
    }
  }
  if (mean_sum > 0.0) out.first_moment = moment / mean_sum;
  return out;
}

/// Integrated `|hat|^2`, split by the unique largest `|k|` component.
struct DirectionalPower {
  double along_x{0.0};
  double along_y{0.0};
  double along_z{0.0};
  /// Modes whose largest component is shared by two or more axes.
  double unresolved{0.0};

  [[nodiscard]] double sum() const {
    return along_x + along_y + along_z + unresolved;
  }
};

/**
 * @brief Sum multiplicity-weighted `|hat|^2` into axis bins.
 *
 * The zero mode is omitted. A grid with `Nz = 1` leaves `along_z` at
 * zero and sends `|kx| == |ky|` to `unresolved`. Unlike `radial_average`,
 * every nonzero mode is counted: there is no `|k|` ball.
 */
[[nodiscard]] inline DirectionalPower
directional_power(const pfc::Box3i &outbox, const pfc::Domain &domain,
                  const std::complex<double> *spectrum, MPI_Comm comm) {
  const auto size = pfc::domain::get_size(domain);
  DirectionalPower local;
  pfc::fft::kspace::for_each_kpoint(
      outbox, domain,
      [&](std::size_t i, double kx, double ky, double kz, int ix, int, int) {
        const double ax = std::abs(kx);
        const double ay = std::abs(ky);
        const double az = std::abs(kz);
        if (ax == 0.0 && ay == 0.0 && az == 0.0) return;
        const double weight = r2c_multiplicity(ix, size[0]);
        const double p = weight * std::norm(spectrum[i]);
        const double largest = std::max(ax, std::max(ay, az));
        const int winners = static_cast<int>(ax == largest) +
                            static_cast<int>(ay == largest) +
                            static_cast<int>(az == largest);
        if (winners != 1) {
          local.unresolved += p;
        } else if (ax == largest) {
          local.along_x += p;
        } else if (ay == largest) {
          local.along_y += p;
        } else {
          local.along_z += p;
        }
      });

  const double send[4] = {local.along_x, local.along_y, local.along_z,
                          local.unresolved};
  double recv[4] = {};
  MPI_Allreduce(send, recv, 4, MPI_DOUBLE, MPI_SUM, comm);
  return DirectionalPower{recv[0], recv[1], recv[2], recv[3]};
}

/**
 * @brief `mean_power` of the shell whose centre is nearest @p target.
 *
 * An exact tie goes to the shell with more mean power, so a target on a
 * bin boundary does not report an empty neighbour.
 */
[[nodiscard]] inline double power_near(const RadialSpectrum &spectrum,
                                       double target) {
  if (spectrum.wavenumber.empty()) return 0.0;
  std::size_t best = 0;
  double best_distance = std::abs(spectrum.wavenumber[0] - target);
  for (std::size_t i = 1; i < spectrum.wavenumber.size(); ++i) {
    const double distance = std::abs(spectrum.wavenumber[i] - target);
    if (distance < best_distance ||
        (distance == best_distance &&
         spectrum.mean_power[i] > spectrum.mean_power[best])) {
      best_distance = distance;
      best = i;
    }
  }
  return spectrum.mean_power[best];
}

} // namespace pfc::fft

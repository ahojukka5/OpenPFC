// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file power_spectrum.hpp
 * @brief Radial and directional power of a stored real-to-complex spectrum.
 *
 * The caller owns what the field means. This header owns the bookkeeping:
 * which stored mode stands for an unstored conjugate, how shells are
 * averaged, and which axis dominates a wavevector.
 *
 * Spectra are HeFFTe r2c outboxes: `kx` runs from 0 through the Nyquist
 * index, and `ky`/`kz` span their full signed range. A mode with `kx = 0`,
 * or with `kx` at the Nyquist index of an even grid, is its own conjugate
 * and is stored once. Every other `kx` has an unstored conjugate of equal
 * power. `r2c_multiplicity` is that weight.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>

namespace pfc::spectral {

/**
 * @brief Weight of one stored r2c coefficient in the full complex spectrum.
 *
 * @param index  x index in the outbox, in `[0, extent/2]`
 * @param extent global number of real samples along x
 */
[[nodiscard]] constexpr double r2c_multiplicity(int index, int extent) noexcept {
  if (index == 0) return 1.0;
  if (extent % 2 == 0 && index == extent / 2) return 1.0;
  return 2.0;
}

/// Shell-averaged power, with the zero mode omitted.
struct RadialSpectrum {
  std::vector<double> wavenumber; ///< shell-centre `|k|`
  std::vector<double> power;      ///< multiplicity-weighted shell mean
  double first_moment{0.0};       ///< `\sum k P / \sum P`
  double peak_wavenumber{0.0};
  double peak_power{0.0};
  double total_power{0.0};

  /// `2\pi` over `first_moment`. Zero when the spectrum has no power.
  [[nodiscard]] double mean_wavelength() const {
    return (first_moment > 0.0) ? 2.0 * std::numbers::pi / first_moment : 0.0;
  }
  /// `2\pi` over `peak_wavenumber`.
  [[nodiscard]] double dominant_wavelength() const {
    return (peak_wavenumber > 0.0) ? 2.0 * std::numbers::pi / peak_wavenumber : 0.0;
  }
};

/**
 * @brief Bin `|spectrum|^2` into shells of `|k|`.
 *
 * Each stored coefficient is weighted by `r2c_multiplicity` in both the
 * power and the count, so a shell of one weight class has the same mean as
 * an unweighted average of its stored values. The `k = 0` mode is dropped.
 * Histograms are summed over @p comm.
 *
 * @param n_bins shells from 0 to the Nyquist radius of the coarsest active axis
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
  out.power.reserve(static_cast<std::size_t>(n_bins));
  double moment = 0.0;
  for (int b = 0; b < n_bins; ++b) {
    const auto bi = static_cast<std::size_t>(b);
    if (global_counts[bi] <= 0.0) continue;
    const double kc = (static_cast<double>(b) + 0.5) * bin_width;
    const double s = global_power[bi] / global_counts[bi];
    out.wavenumber.push_back(kc);
    out.power.push_back(s);
    out.total_power += s;
    moment += kc * s;
    if (s > out.peak_power) {
      out.peak_power = s;
      out.peak_wavenumber = kc;
    }
  }
  if (out.total_power > 0.0) out.first_moment = moment / out.total_power;
  return out;
}

/// Power split by the axis with the strictly largest `|k|` component.
struct DirectionalPower {
  double along_x{0.0};
  double along_y{0.0};
  double along_z{0.0};
  /// Modes whose largest component is shared by two or more axes.
  double unresolved{0.0};
};

/**
 * @brief Sum multiplicity-weighted `|spectrum|^2` into axis bins.
 *
 * The zero mode is omitted. A 2-D grid (`Nz = 1`) leaves `along_z` at zero
 * and sends `|kx| == |ky|` to `unresolved`.
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
 * @brief Power of the shell whose centre is nearest @p target.
 *
 * An exact tie goes to the shell with more power, so a target that lands on
 * a bin boundary does not report an empty neighbour.
 */
[[nodiscard]] inline double power_near(const RadialSpectrum &spectrum,
                                       double target) {
  if (spectrum.wavenumber.empty()) return 0.0;
  std::size_t best = 0;
  double best_distance = std::abs(spectrum.wavenumber[0] - target);
  for (std::size_t i = 1; i < spectrum.wavenumber.size(); ++i) {
    const double distance = std::abs(spectrum.wavenumber[i] - target);
    if (distance < best_distance ||
        (distance == best_distance && spectrum.power[i] > spectrum.power[best])) {
      best_distance = distance;
      best = i;
    }
  }
  return spectrum.power[best];
}

} // namespace pfc::spectral

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file indexed_noise.hpp
 * @brief Deterministic grid noise keyed by the global cell index.
 *
 * The sample at `(i, j, k)` depends on the seed and on those indices, not
 * on which rank owns the cell, so every decomposition writes the same
 * number. `amplitude` scales the sample; it is not an RMS value. Exact
 * mean removal reduces the integer samples over @p comm and is the only
 * step that needs a communicator. Generation itself does not.
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/state_access.hpp>

namespace pfc::field {

/// SplitMix64 of the global index. Both projections below use this word.
[[nodiscard]] constexpr std::uint64_t
indexed_noise_mix(std::uint64_t seed, int i, int j, int k, int nx, int ny) noexcept {
  std::uint64_t x =
      seed + static_cast<std::uint64_t>(i) +
      static_cast<std::uint64_t>(nx) *
          (static_cast<std::uint64_t>(j) +
           static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(k));
  // Unsigned wraparound is the hash.
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

/// Integer sample in `[0, 65535]`. Mean removal sums these.
[[nodiscard]] constexpr std::uint64_t indexed_noise_sample(std::uint64_t seed, int i,
                                                           int j, int k, int nx,
                                                           int ny) noexcept {
  return indexed_noise_mix(seed, i, j, k, nx, ny) >> 48;
}

/// Sample in `[-1, 1]`, for a caller that applies its own amplitude.
[[nodiscard]] constexpr double indexed_noise_signed(std::uint64_t seed, int i, int j,
                                                    int k, int nx, int ny) noexcept {
  const auto word = indexed_noise_mix(seed, i, j, k, nx, ny) >> 11;
  return 2.0 * (static_cast<double>(word) / static_cast<double>(1ULL << 53)) - 1.0;
}

struct IndexedNoise {
  std::uint64_t seed{0};
  double amplitude{0.0};
  /// When true, subtract the global mean of the integer samples.
  bool remove_mean{true};
};

namespace detail {

inline void check_indexed_noise(const pfc::Domain &domain, double amplitude) {
  if (!std::isfinite(amplitude) || amplitude < 0.0) {
    throw std::invalid_argument("indexed noise: amplitude must be finite and >= 0");
  }
  const auto n = pfc::domain::get_size(domain);
  const double count = static_cast<double>(n[0]) * static_cast<double>(n[1]) *
                       static_cast<double>(n[2]);
  if (count >
      static_cast<double>(std::numeric_limits<std::uint64_t>::max() / 65535)) {
    throw std::invalid_argument("indexed noise: grid too large for exact centering");
  }
}

inline double indexed_noise_shift(const pfc::Domain &domain, const pfc::Box3i &box,
                                  const IndexedNoise &noise, MPI_Comm comm) {
  if (!noise.remove_mean) return 0.0;
  const auto n = pfc::domain::get_size(domain);
  std::uint64_t sum = 0;
  for (int k = box.low[2]; k <= box.high[2]; ++k) {
    for (int j = box.low[1]; j <= box.high[1]; ++j) {
      for (int i = box.low[0]; i <= box.high[0]; ++i) {
        sum += indexed_noise_sample(noise.seed, i, j, k, n[0], n[1]);
      }
    }
  }
  std::uint64_t global = 0;
  MPI_Allreduce(&sum, &global, 1, MPI_UINT64_T, MPI_SUM, comm);
  const double count = static_cast<double>(n[0]) * static_cast<double>(n[1]) *
                       static_cast<double>(n[2]);
  return static_cast<double>(global) / count;
}

template <typename Write>
inline void write_indexed_noise(FieldOutput<double> field, const pfc::Domain &domain,
                                const pfc::Box3i &box, const IndexedNoise &noise,
                                double shift, Write &&write) {
  const auto n = pfc::domain::get_size(domain);
  std::size_t at = 0;
  for (int k = box.low[2]; k <= box.high[2]; ++k) {
    for (int j = box.low[1]; j <= box.high[1]; ++j) {
      for (int i = box.low[0]; i <= box.high[0]; ++i) {
        const double sample = static_cast<double>(
            indexed_noise_sample(noise.seed, i, j, k, n[0], n[1]));
        const double delta = noise.amplitude * ((sample - shift) / 65535.0);
        write(field.data()[at], delta);
        ++at;
      }
    }
  }
}

} // namespace detail

/**
 * @brief Add the noise to the values already in @p field.
 *
 * @p comm is used only when `noise.remove_mean` is true. The boxes on that
 * communicator must partition the grid.
 */
inline void add_indexed_noise(FieldOutput<double> field, const pfc::Domain &domain,
                              const pfc::Box3i &box, const IndexedNoise &noise,
                              MPI_Comm comm) {
  detail::check_indexed_noise(domain, noise.amplitude);
  const double shift = detail::indexed_noise_shift(domain, box, noise, comm);
  detail::write_indexed_noise(field, domain, box, noise, shift,
                              [](double &cell, double delta) { cell += delta; });
}

/// Replace @p field with `offset + noise`.
inline void fill_indexed_noise(FieldOutput<double> field, const pfc::Domain &domain,
                               const pfc::Box3i &box, double offset,
                               const IndexedNoise &noise, MPI_Comm comm) {
  if (!std::isfinite(offset)) {
    throw std::invalid_argument("indexed noise: offset must be finite");
  }
  detail::check_indexed_noise(domain, noise.amplitude);
  const double shift = detail::indexed_noise_shift(domain, box, noise, comm);
  detail::write_indexed_noise(
      field, domain, box, noise, shift,
      [offset](double &cell, double delta) { cell = offset + delta; });
}

} // namespace pfc::field

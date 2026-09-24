// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file observable_reduce.hpp
 * @brief Rank-local field sums and MPI allreduce for free-energy/observables.
 *
 * @details
 * Aluminum (M9) needs a global integral of a density. This header sums
 * owned cells (halo excluded), multiplies by the cell volume, and
 * `MPI_Allreduce`s with `MPI_SUM`.
 *
 * Device fields (`CUDASpace` / `HIPSpace`) pull a current host mirror via
 * `with_host_view` and sum the owned interior of that mirror. Kernel-safe:
 * no runtime GPU headers.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>

namespace pfc::sim {

[[nodiscard]] inline double cell_volume(const Domain &domain) noexcept {
  const auto &s = domain::get_spacing(domain);
  return s[0] * s[1] * s[2];
}

/**
 * @brief Sum owned cells of @p field on this rank (halo excluded).
 *
 * Host fields iterate `for_each_owned`. Device fields refresh the host
 * mirror and sum the owned index box.
 */
template <class T, class MemorySpace = pfc::HostSpace>
[[nodiscard]] double sum_owned(pfc::data::Field<T, MemorySpace> &field) {
  double acc = 0.0;
  if constexpr (pfc::data::Field<T, MemorySpace>::is_host_space) {
    field.for_each_owned(
        [&](int i, int j, int k) { acc += static_cast<double>(field(i, j, k)); });
  } else {
    field.with_host_view([&](T *data, std::size_t) {
      const int nx = field.box().size[0];
      const int ny = field.box().size[1];
      const int nz = field.box().size[2];
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            acc += static_cast<double>(data[field.idx(i, j, k)]);
          }
        }
      }
    });
  }
  return acc;
}

[[nodiscard]] inline double allreduce_sum(double local, MPI_Comm comm) {
  double global = 0.0;
  const int err = MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Allreduce SUM in allreduce_sum");
  return global;
}

/**
 * @brief Global integral ∫ field dV ≈ (sum owned) × Δx Δy Δz, all ranks.
 */
template <class T, class MemorySpace = pfc::HostSpace>
[[nodiscard]] double integrate_owned(pfc::data::Field<T, MemorySpace> &field,
                                     MPI_Comm comm) {
  const double local = sum_owned(field) * cell_volume(field.domain());
  return allreduce_sum(local, comm);
}

/// Equal-weight moments of owned cells. Halo cells are not included.
struct FieldReduction {
  double sum{0.0};
  double min{0.0};
  double max{0.0};
  double mean{0.0};
  double l1{0.0};
  double l2{0.0};
  double rms{0.0};
  /// Second moment about the mean. Equal weight per owned cell.
  double variance{0.0};
  /// `sum * Δx Δy Δz`.
  double integral{0.0};
  std::uint64_t count{0};
};

namespace detail {

template <class T, class MemorySpace, class Map>
[[nodiscard]] FieldReduction local_moments(pfc::data::Field<T, MemorySpace> &field,
                                           Map &&map) {
  FieldReduction m;
  m.min = std::numeric_limits<double>::infinity();
  m.max = -m.min;
  auto consume = [&](double raw) {
    const double v = static_cast<double>(map(raw));
    m.sum += v;
    m.l1 += std::fabs(v);
    m.variance += v * v;
    m.min = std::min(m.min, v);
    m.max = std::max(m.max, v);
    ++m.count;
  };
  if constexpr (pfc::data::Field<T, MemorySpace>::is_host_space) {
    field.for_each_owned(
        [&](int i, int j, int k) { consume(static_cast<double>(field(i, j, k))); });
  } else {
    field.with_host_view([&](T *data, std::size_t) {
      const int nx = field.box().size[0];
      const int ny = field.box().size[1];
      const int nz = field.box().size[2];
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            consume(static_cast<double>(data[field.idx(i, j, k)]));
          }
        }
      }
    });
  }
  return m;
}

} // namespace detail

/**
 * @brief Global owned-cell reductions of `map(value)`.
 *
 * `map` is a local per-cell function. It does not see other cells, so a
 * quantity such as a gradient stays in the caller. Empty fields reduce to
 * a zero count and a NaN min/max.
 */
template <class T, class MemorySpace, class Map>
[[nodiscard]] FieldReduction reduce_owned(pfc::data::Field<T, MemorySpace> &field,
                                          MPI_Comm comm, Map &&map) {
  if (comm == MPI_COMM_NULL) {
    throw std::invalid_argument("reduce_owned: pass an explicit communicator");
  }
  const FieldReduction local = detail::local_moments(field, std::forward<Map>(map));
  double sums[3] = {local.sum, local.l1, local.variance};
  double gsums[3] = {};
  pfc::mpi::throw_on_mpi_error(
      MPI_Allreduce(sums, gsums, 3, MPI_DOUBLE, MPI_SUM, comm),
      "reduce_owned: MPI_Allreduce SUM");
  double gmin = 0.0;
  double gmax = 0.0;
  pfc::mpi::throw_on_mpi_error(
      MPI_Allreduce(&local.min, &gmin, 1, MPI_DOUBLE, MPI_MIN, comm),
      "reduce_owned: MPI_Allreduce MIN");
  pfc::mpi::throw_on_mpi_error(
      MPI_Allreduce(&local.max, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm),
      "reduce_owned: MPI_Allreduce MAX");
  unsigned long long lcount = local.count;
  unsigned long long gcount = 0;
  pfc::mpi::throw_on_mpi_error(
      MPI_Allreduce(&lcount, &gcount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm),
      "reduce_owned: MPI_Allreduce count");

  FieldReduction out;
  out.sum = gsums[0];
  out.l1 = gsums[1];
  out.count = static_cast<std::uint64_t>(gcount);
  const double sum_sq = gsums[2];
  if (out.count == 0) {
    out.min = std::numeric_limits<double>::quiet_NaN();
    out.max = out.min;
    return out;
  }
  out.min = gmin;
  out.max = gmax;
  out.mean = out.sum / static_cast<double>(out.count);
  out.l2 = std::sqrt(sum_sq);
  out.rms = std::sqrt(sum_sq / static_cast<double>(out.count));
  out.variance = sum_sq / static_cast<double>(out.count) - out.mean * out.mean;
  out.integral = out.sum * cell_volume(field.domain());
  return out;
}

template <class T, class MemorySpace = pfc::HostSpace>
[[nodiscard]] FieldReduction reduce_owned(pfc::data::Field<T, MemorySpace> &field,
                                          MPI_Comm comm) {
  return reduce_owned(field, comm, [](double v) { return v; });
}

} // namespace pfc::sim

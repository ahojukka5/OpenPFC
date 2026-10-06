// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file field_stats.hpp
 * @brief Global reductions of a local tensor field on an FFT inbox.
 *
 * Norms are the un-normalized discrete 2-norm over voxels. The weighted
 * checksum uses global indices so a permutation of the grid changes it.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/mechanics/tensor.hpp>

namespace pfc::finite_strain {

struct ComponentStats {
  double minimum = 0.0;
  double maximum = 0.0;
  double l2 = 0.0;
  double sum = 0.0;
  /// `sum (x+1)(y+1)(z+1) component`, global indices, x fastest in memory.
  double weighted_checksum = 0.0;
};

struct ScalarStats {
  double minimum = 0.0;
  double maximum = 0.0;
  double l2 = 0.0;
};

struct TensorFieldStats {
  ComponentStats component[3][3]{};
  double frobenius_l2 = 0.0;
  ScalarStats equivalent{};
};

[[nodiscard]] inline TensorFieldStats
tensor_field_stats(const std::vector<Tensor2> &field, const Box3i &inbox,
                   MPI_Comm comm, bool with_equivalent_stress) {
  if (static_cast<long long>(field.size()) != inbox.count()) {
    throw std::invalid_argument(
        "tensor_field_stats: field size does not match inbox");
  }

  double local_sum[9]{};
  double local_sumsq[9]{};
  double local_weighted[9]{};
  double local_min[9];
  double local_max[9];
  for (int c = 0; c < 9; ++c) {
    local_min[c] = std::numeric_limits<double>::infinity();
    local_max[c] = -std::numeric_limits<double>::infinity();
  }
  double eq_min = std::numeric_limits<double>::infinity();
  double eq_max = -std::numeric_limits<double>::infinity();
  double eq_sumsq = 0.0;
  double frob_sumsq = 0.0;

  std::size_t n = 0;
  for (int z = inbox.low[2]; z <= inbox.high[2]; ++z) {
    for (int y = inbox.low[1]; y <= inbox.high[1]; ++y) {
      for (int x = inbox.low[0]; x <= inbox.high[0]; ++x, ++n) {
        const double weight = static_cast<double>(x + 1) *
                              static_cast<double>(y + 1) *
                              static_cast<double>(z + 1);
        const Tensor2 &value = field[n];
        frob_sumsq += frobenius_dot(value, value);
        if (with_equivalent_stress) {
          const double eq = von_mises(value);
          eq_min = std::min(eq_min, eq);
          eq_max = std::max(eq_max, eq);
          eq_sumsq += eq * eq;
        }
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            const int c = a * 3 + b;
            const double component = value(a, b);
            local_sum[c] += component;
            local_sumsq[c] += component * component;
            local_weighted[c] += weight * component;
            local_min[c] = std::min(local_min[c], component);
            local_max[c] = std::max(local_max[c], component);
          }
        }
      }
    }
  }

  double sum[9]{};
  double sumsq[9]{};
  double weighted[9]{};
  double minimum[9]{};
  double maximum[9]{};
  MPI_Allreduce(local_sum, sum, 9, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(local_sumsq, sumsq, 9, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(local_weighted, weighted, 9, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(local_min, minimum, 9, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(local_max, maximum, 9, MPI_DOUBLE, MPI_MAX, comm);

  double frob_global = 0.0;
  MPI_Allreduce(&frob_sumsq, &frob_global, 1, MPI_DOUBLE, MPI_SUM, comm);
  double eq_sumsq_global = 0.0;
  double eq_min_global = 0.0;
  double eq_max_global = 0.0;
  if (with_equivalent_stress) {
    MPI_Allreduce(&eq_sumsq, &eq_sumsq_global, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&eq_min, &eq_min_global, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&eq_max, &eq_max_global, 1, MPI_DOUBLE, MPI_MAX, comm);
  }

  TensorFieldStats stats;
  stats.frobenius_l2 = std::sqrt(frob_global);
  if (with_equivalent_stress) {
    stats.equivalent.minimum = eq_min_global;
    stats.equivalent.maximum = eq_max_global;
    stats.equivalent.l2 = std::sqrt(eq_sumsq_global);
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const int c = a * 3 + b;
      ComponentStats &out = stats.component[a][b];
      out.minimum = minimum[c];
      out.maximum = maximum[c];
      out.l2 = std::sqrt(sumsq[c]);
      out.sum = sum[c];
      out.weighted_checksum = weighted[c];
    }
  }
  return stats;
}

} // namespace pfc::finite_strain

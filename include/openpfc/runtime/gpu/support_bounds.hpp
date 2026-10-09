// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#include <openpfc/kernel/grain/support_bounds.hpp>
#include <openpfc/runtime/gpu/grain_transfer.hpp>

namespace pfc::grain::distributed {
namespace detail {
__global__ void bounds_kernel(Grid3D grid, Grid3D global, std::size_t lo0,
                              std::size_t lo1, std::size_t lo2, Slot slots,
                              const Id *labels,
                              const pfc::grain::detail::Assignment *assignments,
                              std::size_t count, unsigned long long *bounds,
                              unsigned *failure) {
  auto cells = grid.nx * grid.ny * grid.nz;
  for (auto cell = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; cell < cells;
       cell += std::size_t(gridDim.x) * blockDim.x) {
    const unsigned long long x[3]{lo0 + cell % grid.nx,
                                  lo1 + (cell / grid.nx) % grid.ny,
                                  lo2 + cell / (grid.nx * grid.ny)};
    for (Slot s = 0; s < slots; ++s) {
      auto uid = labels[cells * s + cell];
      if (!uid) continue;
      auto a = pfc::grain::detail::find_assignment(assignments, count, uid);
      if (a == count || assignments[a].source != s) {
        atomicMax(failure, 1u);
        continue;
      }
      atomicAdd(bounds + 7 * a, 1ull);
      for (int d = 0; d < 3; ++d) {
        atomicMin(bounds + 7 * a + 1 + 2 * d, x[d]);
        atomicMax(bounds + 7 * a + 2 + 2 * d, x[d]);
      }
    }
  }
}
} // namespace detail
/// Reduce complete known-UID support directly on the device. Only seven
/// integers per registered UID and one validation flag return to the host.
/// No full arrays, boundary shells or sampled support thresholds are used.
template <class Backend>
BoundObservation observe_bounds(Partition p, Slot slots,
                                const pfc::core::DataBuffer<Backend, Id> &labels,
                                std::span<const Grain> grains,
                                pfc::gpuStream_t stream = nullptr) {
  validate(p);
  auto grid = p.local();
  auto cells = cell_count(grid);
  if (!slots || slots == unassigned || cells > SIZE_MAX / slots ||
      labels.size() != cells * slots)
    throw std::invalid_argument("invalid device bound layout");
  auto prepared = pfc::grain::detail::prepare_transfer(slots, grains, {});
  if (prepared.status != TransferStatus::Success ||
      prepared.assignments.size() > SIZE_MAX / 7)
    throw std::invalid_argument("invalid device bound registry");
  GPU_CHECK(pfc::gpuStreamSynchronize(stream));
  pfc::core::DataBuffer<Backend, pfc::grain::detail::Assignment> assignments(
      prepared.assignments.size());
  assignments.copy_from_host(prepared.assignments);
  std::vector<unsigned long long> initial(7 * assignments.size(), 0);
  for (std::size_t a = 0; a < assignments.size(); ++a)
    for (int d = 0; d < 3; ++d) initial[7 * a + 1 + 2 * d] = ULLONG_MAX;
  pfc::core::DataBuffer<Backend, unsigned long long> bounds(initial.size());
  bounds.copy_from_host(initial);
  pfc::core::DataBuffer<Backend, unsigned> failure(1);
  failure.copy_from_host(std::vector<unsigned>{0});
  GPU_CHECK(pfc::gpuStreamSynchronize(nullptr));
  unsigned blocks = unsigned(std::min<std::size_t>((cells - 1) / 256 + 1, 65535));
  GPU_LAUNCH_KERNEL(detail::bounds_kernel, blocks, 256,
                    (grid, p.global, p.lower[0], p.lower[1], p.lower[2], slots,
                     labels.data(), assignments.data(), assignments.size(),
                     bounds.data(), failure.data()),
                    stream);
  GPU_CHECK(pfc::gpuStreamSynchronize(stream));
  if (failure.to_host().front())
    throw std::invalid_argument("unknown UID/source slot in device bounds");
  auto compact = bounds.to_host();
  BoundObservation out{p, {}};
  for (std::size_t a = 0; a < assignments.size(); ++a)
    if (compact[7 * a]) {
      auto record = prepared.assignments[a];
      SupportBounds b{record.id, record.source, compact[7 * a], {}, {}};
      for (int d = 0; d < 3; ++d) {
        b.lower[d] = compact[7 * a + 1 + 2 * d];
        b.upper[d] = compact[7 * a + 2 + 2 * d];
      }
      out.bounds.push_back(b);
    }
  return out;
}
} // namespace pfc::grain::distributed
#endif

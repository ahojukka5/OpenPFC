// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

// Header-only device realization; instantiate in a CUDA/HIP translation unit.
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)

#include <type_traits>

#include <openpfc/runtime/gpu/gpu_api.hpp>

#include <openpfc/kernel/grain/transfer.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>

namespace pfc::grain {

namespace detail {

__global__ inline void
stage_transfer(std::size_t cells, Slot slots, const double *values, const Id *labels,
               const pfc::grain::detail::Assignment *assignments,
               std::size_t assignment_count, double background_value,
               double *staged_values, Id *staged_labels, unsigned *failure,
               unsigned *present) {
  for (auto cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < cells; cell += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto status = pfc::grain::detail::stage_cell(
        cell, cells, slots, values, labels, assignments, assignment_count,
        background_value, staged_values, staged_labels);
    if (status != TransferStatus::Success)
      atomicMax(failure, static_cast<unsigned>(status));
    for (Slot slot = 0; slot < slots; ++slot) {
      const auto index = pfc::grain::detail::find_assignment(
          assignments, assignment_count, labels[cells * slot + cell]);
      if (index != assignment_count) atomicExch(present + index, 1u);
    }
  }
}

} // namespace detail

/// Same support and transaction contract as pfc::grain::transfer. Original
/// device inputs are never mutated; caller publishes successful owning buffers
/// and grain registry together. Only status and O(grains) flags return to host.
/// Runtime/allocation exceptions also leave original inputs intact.
template <typename Backend>
TransferResult<pfc::core::DataBuffer<Backend, double>,
               pfc::core::DataBuffer<Backend, Id>>
transfer(const Grid2D &grid, Slot slots,
         const pfc::core::DataBuffer<Backend, double> &values,
         const pfc::core::DataBuffer<Backend, Id> &labels,
         std::span<const Grain> grains, std::span<const Transfer> moves,
         double background_value = 0.0, pfc::gpuStream_t stream = nullptr) {
#if defined(__HIPCC__) || defined(__HIP__)
  static_assert(std::is_same_v<Backend, pfc::backend::HIPTag>);
#else
  static_assert(std::is_same_v<Backend, pfc::backend::CUDATag>);
#endif
  const auto layout = pfc::grain::detail::transfer_layout(
      grid, slots, values.size(), labels.size(), background_value);
  if (layout != TransferStatus::Success) return {layout, {}, {}, {}};
  auto prepared = pfc::grain::detail::prepare_transfer(slots, grains, moves);
  if (prepared.status != TransferStatus::Success)
    return {prepared.status, {}, {}, {}};
  const auto cells = cell_count(grid);
  // Synchronize caller writes before default-stream staging allocations/copies.
  GPU_CHECK(pfc::gpuStreamSynchronize(stream));
  pfc::core::DataBuffer<Backend, pfc::grain::detail::Assignment> assignments(
      prepared.assignments.size());
  assignments.copy_from_host(prepared.assignments);
  pfc::core::DataBuffer<Backend, unsigned> present(prepared.assignments.size());
  present.copy_from_host(std::vector<unsigned>(prepared.assignments.size(), 0));
  pfc::core::DataBuffer<Backend, unsigned> failure(1);
  failure.copy_from_host(std::vector<unsigned>{0});
  TransferResult<pfc::core::DataBuffer<Backend, double>,
                 pfc::core::DataBuffer<Backend, Id>>
      result;
  result.values = pfc::core::DataBuffer<Backend, double>(values.size());
  result.labels = pfc::core::DataBuffer<Backend, Id>(labels.size());
  const unsigned blocks =
      static_cast<unsigned>(std::min<std::size_t>((cells - 1) / 256 + 1, 65535));
  GPU_LAUNCH_KERNEL(detail::stage_transfer, blocks, 256,
                    (cells, slots, values.data(), labels.data(), assignments.data(),
                     assignments.size(), background_value, result.values.data(),
                     result.labels.data(), failure.data(), present.data()),
                    stream);
  GPU_CHECK(pfc::gpuStreamSynchronize(stream));
  result.status = static_cast<TransferStatus>(failure.to_host().front());
  const auto presence = present.to_host();
  if (std::find(presence.begin(), presence.end(), 0) != presence.end())
    result.status =
        pfc::grain::detail::worst(result.status, TransferStatus::MissingSupport);
  if (result.status != TransferStatus::Success) return {result.status, {}, {}, {}};
  result.grains = std::move(prepared.grains);
  return result;
}

} // namespace pfc::grain

#endif

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

// Header-only device realization; instantiate in a CUDA/HIP translation unit.
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)

#include <type_traits>

#include <openpfc/runtime/gpu/gpu_api.hpp>

#include <openpfc/kernel/grain/transfer.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/grain_diagnostics.hpp>

namespace pfc::grain {

namespace detail {

template <bool Observe>
__global__ void
stage_transfer(std::size_t cells, Slot slots, const double *values, const Id *labels,
               const pfc::grain::detail::Assignment *assignments,
               std::size_t assignment_count, double background_value,
               double *staged_values, Id *staged_labels, unsigned *failure,
               unsigned *present, diagnostics::Accesses *totals) {
  std::conditional_t<Observe, diagnostics::Accesses, diagnostics::NoAccesses>
      local{};
  diagnostics::Accesses *counts = nullptr;
  if constexpr (Observe) counts = &local;
  for (auto cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < cells; cell += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto status = pfc::grain::detail::stage_cell<Observe>(
        cell, cells, slots, values, labels, assignments, assignment_count,
        background_value, staged_values, staged_labels, counts);
    if (status != TransferStatus::Success) {
      if constexpr (Observe) {
        diagnostics::read(counts, diagnostics::Field::Flags, sizeof(unsigned));
        diagnostics::write(counts, diagnostics::Field::Flags, sizeof(unsigned));
      }
      atomicMax(failure, static_cast<unsigned>(status));
    }
    for (Slot slot = 0; slot < slots; ++slot) {
      const auto index = pfc::grain::detail::find_assignment<Observe>(
          assignments, assignment_count,
          diagnostics::load<Observe>(labels, cells * slot + cell, counts,
                                     diagnostics::Field::Labels),
          counts);
      if (index != assignment_count) {
        if constexpr (Observe) {
          diagnostics::read(counts, diagnostics::Field::Counts, sizeof(unsigned));
          diagnostics::write(counts, diagnostics::Field::Counts, sizeof(unsigned));
        }
        atomicExch(present + index, 1u);
      }
    }
  }
  if constexpr (Observe) diagnostics::merge(totals, local);
}

} // namespace detail

/// Same support and transaction contract as pfc::grain::transfer. Original
/// device inputs are never mutated; caller publishes successful owning buffers
/// and grain registry together. Only status and O(grains) flags return to host.
/// Runtime/allocation exceptions also leave original inputs intact.
template <bool Observe, typename Backend>
TransferResult<pfc::core::DataBuffer<Backend, double>,
               pfc::core::DataBuffer<Backend, Id>>
transfer_impl(const Grid2D &grid, Slot slots,
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
  diagnostics::Observation<Backend> observation;
  diagnostics::Interval interval(diagnostics::Phase::Transfer, stream);
  diagnostics::scan(diagnostics::host_accesses(), diagnostics::Phase::Transfer,
                    slots);
  pfc::core::DataBuffer<Backend, pfc::grain::detail::Assignment> assignments(
      prepared.assignments.size());
  assignments.copy_from_host(prepared.assignments, diagnostics::Field::Assignments);
  pfc::core::DataBuffer<Backend, unsigned> present(prepared.assignments.size());
  present.copy_from_host(std::vector<unsigned>(prepared.assignments.size(), 0),
                         diagnostics::Field::Counts);
  pfc::core::DataBuffer<Backend, unsigned> failure(1);
  failure.copy_from_host(std::vector<unsigned>{0}, diagnostics::Field::Flags);
  TransferResult<pfc::core::DataBuffer<Backend, double>,
                 pfc::core::DataBuffer<Backend, Id>>
      result;
  result.values = pfc::core::DataBuffer<Backend, double>(values.size());
  result.labels = pfc::core::DataBuffer<Backend, Id>(labels.size());
  // Pageable default-stream uploads need completion before a nonblocking
  // caller stream can consume their device metadata.
  if (stream != nullptr) GPU_CHECK(pfc::gpuStreamSynchronize(nullptr));
  const unsigned blocks =
      static_cast<unsigned>(std::min<std::size_t>((cells - 1) / 256 + 1, 65535));
  GPU_LAUNCH_KERNEL(detail::stage_transfer<Observe>, blocks, 256,
                    (cells, slots, values.data(), labels.data(), assignments.data(),
                     assignments.size(), background_value, result.values.data(),
                     result.labels.data(), failure.data(), present.data(),
                     observation.data()),
                    stream);
  GPU_CHECK(pfc::gpuStreamSynchronize(stream));
  result.status = static_cast<TransferStatus>(
      failure.to_host(diagnostics::Field::Flags).front());
  const auto presence = present.to_host(diagnostics::Field::Counts);
  if (std::find(presence.begin(), presence.end(), 0) != presence.end())
    result.status =
        pfc::grain::detail::worst(result.status, TransferStatus::MissingSupport);
  if (result.status != TransferStatus::Success) return {result.status, {}, {}, {}};
  result.grains = std::move(prepared.grains);
  return result;
}

template <typename Backend>
TransferResult<pfc::core::DataBuffer<Backend, double>,
               pfc::core::DataBuffer<Backend, Id>>
transfer(const Grid2D &grid, Slot slots,
         const pfc::core::DataBuffer<Backend, double> &values,
         const pfc::core::DataBuffer<Backend, Id> &labels,
         std::span<const Grain> grains, std::span<const Transfer> moves,
         double background_value = 0.0, pfc::gpuStream_t stream = nullptr) {
  if (diagnostics::current) {
    try {
      diagnostics::current->covered();
      return transfer_impl<true>(grid, slots, values, labels, grains, moves,
                                 background_value, stream);
    } catch (...) {
      diagnostics::current->fail();
      throw;
    }
  }
  return transfer_impl<false>(grid, slots, values, labels, grains, moves,
                              background_value, stream);
}

} // namespace pfc::grain

#endif

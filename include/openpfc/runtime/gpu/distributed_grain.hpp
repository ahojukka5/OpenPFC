// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#include <openpfc/kernel/grain/distributed.hpp>
#include <openpfc/kernel/grain/signed_transfer.hpp>
#include <openpfc/runtime/gpu/grain_tracking.hpp>
#include <openpfc/runtime/gpu/grain_transfer.hpp>

namespace pfc::grain::distributed {
namespace detail {
__global__ void inspect(Grid3D grid, Grid3D global, std::size_t lo0, std::size_t lo1,
                        std::size_t lo2, Slot slots, const Id *labels,
                        const pfc::grain::detail::Assignment *assignments,
                        std::size_t count, std::size_t radius, BoundarySample *shell,
                        std::size_t capacity, unsigned long long *shell_count,
                        unsigned long long *counts, unsigned *failure) {
  const auto cells = grid.nx * grid.ny * grid.nz;
  for (auto cell = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; cell < cells;
       cell += std::size_t(gridDim.x) * blockDim.x) {
    auto x = cell % grid.nx, y = (cell / grid.nx) % grid.ny,
         z = cell / (grid.nx * grid.ny);
    bool edge = (grid.nx != global.nx && (x < radius || grid.nx - 1 - x < radius)) ||
                (grid.ny != global.ny && (y < radius || grid.ny - 1 - y < radius)) ||
                (grid.nz != global.nz && (z < radius || grid.nz - 1 - z < radius));
    for (Slot s = 0; s < slots; ++s) {
      auto uid = labels[cells * s + cell];
      if (!uid) continue;
      auto a = pfc::grain::detail::find_assignment(assignments, count, uid);
      if (a == count || assignments[a].source != s) {
        atomicMax(failure, 1u);
        continue;
      }
      atomicAdd(counts + a, 1ull);
      if (edge) {
        auto at = atomicAdd(shell_count, 1ull);
        if (at < capacity)
          shell[at] = {lo0 + x + global.nx * (lo1 + y + global.ny * (lo2 + z)), uid,
                       s};
        else
          atomicMax(failure, 2u);
      }
      for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if (!pfc::grain::detail::stencil(grid, dx, dy, dz, 1)) continue;
            std::size_t other;
            if (pfc::grain::detail::offset(grid, cell, dx, dy, dz, other)) {
              auto b = labels[cells * s + other];
              if (b && b != uid) atomicMax(failure, 3u);
            }
          }
    }
  }
}
__global__ void signed_stage(std::size_t cells, Slot slots, const double *values,
                             const Id *labels,
                             const pfc::grain::detail::Assignment *assignments,
                             std::size_t count, double background_value,
                             double *staged_values, Id *staged_labels,
                             unsigned *failure, unsigned *present) {
  for (auto cell = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; cell < cells;
       cell += std::size_t(gridDim.x) * blockDim.x) {
    auto status = pfc::grain::detail::stage_signed_cell(
        cell, cells, slots, values, labels, assignments, count, background_value,
        staged_values, staged_labels);
    if (status != TransferStatus::Success) atomicMax(failure, unsigned(status));
    for (Slot s = 0; s < slots; ++s) {
      auto a = pfc::grain::detail::find_assignment(assignments, count,
                                                   labels[cells * s + cell]);
      if (a != count) atomicExch(present + a, 1u);
    }
  }
}
} // namespace detail

/// Complete known-UID producer: every occupied support sample already carries
/// its persistent UID. No new/unseeded identities are invented. Only occupied
/// shell samples, compact support counts and contact edges return to host.
/// Put this producer inside collective prepare's callback to vote exceptions.
template <class Backend>
Observation
observe(Partition p, Slot slots, const pfc::core::DataBuffer<Backend, Id> &labels,
        std::span<const Grain> grains, std::size_t radius,
        std::size_t edge_capacity = 4096, pfc::gpuStream_t stream = nullptr,
        std::size_t shell_capacity = 4 * 1024 * 1024) {
  validate(p);
  auto grid = p.local();
  auto cells = cell_count(grid);
  if (!slots || slots == unassigned || cells > SIZE_MAX / slots ||
      labels.size() != cells * slots || !radius || radius > std::size_t(INT_MAX / 2))
    throw std::invalid_argument("device distributed layout");
  auto prepared = pfc::grain::detail::prepare_transfer(slots, grains, {});
  if (prepared.status != TransferStatus::Success)
    throw std::invalid_argument("device distributed registry");
  GPU_CHECK(pfc::gpuStreamSynchronize(stream));
  pfc::core::DataBuffer<Backend, pfc::grain::detail::Assignment> assignments(
      prepared.assignments.size());
  assignments.copy_from_host(prepared.assignments);
  pfc::core::DataBuffer<Backend, unsigned long long> counts(
      prepared.assignments.size()),
      used(1);
  counts.copy_from_host(
      std::vector<unsigned long long>(prepared.assignments.size(), 0));
  used.copy_from_host(std::vector<unsigned long long>{0});
  pfc::core::DataBuffer<Backend, unsigned> failure(1);
  failure.copy_from_host(std::vector<unsigned>{0});
  // Cap allocation at the geometrical shell bound and explicit caller capacity.
  std::size_t interior = 1;
  const std::size_t n[3]{grid.nx, grid.ny, grid.nz},
      g[3]{p.global.nx, p.global.ny, p.global.nz};
  for (int d = 0; d < 3; ++d)
    interior *= n[d] == g[d] ? n[d] : (n[d] > 2 * radius ? n[d] - 2 * radius : 0);
  auto capacity = std::min(shell_capacity, (cells - interior) * slots);
  pfc::core::DataBuffer<Backend, BoundarySample> shell(capacity);
  GPU_CHECK(pfc::gpuStreamSynchronize(nullptr));
  unsigned blocks = unsigned(std::min<std::size_t>((cells - 1) / 256 + 1, 65535));
  GPU_LAUNCH_KERNEL(detail::inspect, blocks, 256,
                    (grid, p.global, p.lower[0], p.lower[1], p.lower[2], slots,
                     labels.data(), assignments.data(), assignments.size(), radius,
                     shell.data(), capacity, used.data(), counts.data(),
                     failure.data()),
                    stream);
  GPU_CHECK(pfc::gpuStreamSynchronize(stream));
  auto bad = failure.to_host().front();
  if (bad == 3) throw std::domain_error("device same-slot touching grain UIDs");
  if (bad == 2) throw std::length_error("device boundary shell capacity");
  if (bad) throw std::invalid_argument("device unknown UID or source slot");
  Observation out;
  out.partition = p;
  auto support = counts.to_host();
  std::vector<Id> identities;
  for (std::size_t a = 0; a < prepared.assignments.size(); ++a) {
    auto assignment = prepared.assignments[a];
    identities.push_back(assignment.id);
    if (support[a])
      out.components.push_back(
          {assignment.id, assignment.source, assignment.id, support[a]});
  }
  auto size = std::size_t(used.to_host().front());
  out.boundary.resize(size);
  if (size)
    GPU_CHECK(pfc::gpuMemcpy(out.boundary.data(), shell.data(),
                             size * sizeof(BoundarySample),
                             pfc::gpuMemcpyDeviceToHost));
  pfc::core::DataBuffer<Backend, Contact> contacts(edge_capacity);
  auto adjacent = pfc::grain::tracking::adjacency(
      Backend{}, grid, slots, std::span<const Id>(labels.data(), labels.size()),
      identities, std::span<Contact>(contacts.data(), contacts.size()), radius);
  if (adjacent.status != pfc::grain::tracking::Status::Success)
    throw std::length_error("device contact capacity or validation failure");
  std::vector<Contact> edges(adjacent.edge_count);
  if (!edges.empty())
    GPU_CHECK(pfc::gpuMemcpy(edges.data(), contacts.data(),
                             edges.size() * sizeof(Contact),
                             pfc::gpuMemcpyDeviceToHost));
  for (auto e : edges) out.contacts.push_back({e.first, e.second});
  return out;
}
} // namespace pfc::grain::distributed

namespace pfc::grain {
template <class Backend, class Grid>
auto transfer_signed(Grid grid, Slot slots,
                     const pfc::core::DataBuffer<Backend, double> &values,
                     const pfc::core::DataBuffer<Backend, Id> &labels,
                     std::span<const Grain> grains, std::span<const Transfer> moves,
                     double background_value = 0,
                     pfc::gpuStream_t stream = nullptr) {
  using Result = TransferResult<pfc::core::DataBuffer<Backend, double>,
                                pfc::core::DataBuffer<Backend, Id>>;
  auto layout = detail::transfer_layout(grid, slots, values.size(), labels.size(),
                                        background_value);
  if (layout != TransferStatus::Success) return Result{layout, {}, {}, {}};
  auto prepared = detail::prepare_transfer(slots, grains, moves);
  if (prepared.status != TransferStatus::Success)
    return Result{prepared.status, {}, {}, {}};
  GPU_CHECK(pfc::gpuStreamSynchronize(stream));
  pfc::core::DataBuffer<Backend, detail::Assignment> assignments(
      prepared.assignments.size());
  assignments.copy_from_host(prepared.assignments);
  pfc::core::DataBuffer<Backend, unsigned> present(prepared.assignments.size()),
      failure(1);
  present.copy_from_host(std::vector<unsigned>(prepared.assignments.size(), 0));
  failure.copy_from_host(std::vector<unsigned>{0});
  Result result;
  result.values = pfc::core::DataBuffer<Backend, double>(values.size());
  result.labels = pfc::core::DataBuffer<Backend, Id>(labels.size());
  GPU_CHECK(pfc::gpuStreamSynchronize(nullptr));
  auto cells = cell_count(grid);
  unsigned blocks = unsigned(std::min<std::size_t>((cells - 1) / 256 + 1, 65535));
  GPU_LAUNCH_KERNEL(distributed::detail::signed_stage, blocks, 256,
                    (cells, slots, values.data(), labels.data(), assignments.data(),
                     assignments.size(), background_value, result.values.data(),
                     result.labels.data(), failure.data(), present.data()),
                    stream);
  GPU_CHECK(pfc::gpuStreamSynchronize(stream));
  result.status = TransferStatus(failure.to_host().front());
  auto support = present.to_host();
  if (std::find(support.begin(), support.end(), 0) != support.end())
    result.status = detail::worst(result.status, TransferStatus::MissingSupport);
  if (result.status != TransferStatus::Success)
    return Result{result.status, {}, {}, {}};
  result.grains = std::move(prepared.grains);
  return result;
}
} // namespace pfc::grain
#endif

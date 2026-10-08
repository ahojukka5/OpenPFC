// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

// Instantiate in a CUDA/HIP translation unit; field arrays stay device-owned.
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#include <openpfc/runtime/common/grain_remapping.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/grain_tracking.hpp>
#include <openpfc/runtime/gpu/grain_transfer.hpp>

namespace pfc::grain::remapping {
namespace detail {
__global__ inline void preflight(std::size_t cells, Slot slots, const double *values,
                                 const Id *seeds,
                                 const pfc::grain::detail::Assignment *assignments,
                                 std::size_t grains, std::uint8_t *occupied,
                                 unsigned *error) {
  for (auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < cells * slots; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto q = values[i];
    occupied[i] = q > 0;
    if (!(q >= 0 && q <= std::numeric_limits<double>::max())) {
      atomicMax(error, static_cast<unsigned>(Status::InvalidInput));
      continue;
    }
    if (!occupied[i] || !seeds[i]) continue;
    const auto j =
        pfc::grain::detail::find_assignment(assignments, grains, seeds[i]);
    if (j == grains)
      atomicMax(error, static_cast<unsigned>(Status::UnknownIdentity));
    else if (assignments[j].source != i / cells)
      atomicMax(error, static_cast<unsigned>(Status::InvalidInput));
  }
}

__global__ inline void inspect(Grid2D grid, Slot slots, const Id *labels,
                               const pfc::grain::detail::Assignment *assignments,
                               std::size_t grains, unsigned long long *counts,
                               unsigned *error) {
  const auto cells = grid.nx * grid.ny;
  for (auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < cells * slots; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (!labels[i]) continue;
    const auto index =
        pfc::grain::detail::find_assignment(assignments, grains, labels[i]);
    if (index == grains) {
      atomicMax(error, static_cast<unsigned>(Status::UnknownIdentity));
      continue;
    }
    atomicAdd(counts + index, 1ULL);
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        if ((!dx && !dy) || (grid.connectivity == Connectivity::Four && dx && dy))
          continue;
        std::size_t cell;
        if (!neighbor(grid, i % cells, dx, dy, cell)) continue;
        const auto other = labels[(i / cells) * cells + cell];
        if (other && other != labels[i])
          atomicMax(error, static_cast<unsigned>(Status::UnsafeCadence));
      }
  }
}
} // namespace detail

/** Same owning transaction and cadence contract as the CPU remap overload.
 * Synchronizes prior work on stream; all constituent staging completes before
 * returning. Only compact counts/graph/registry/status return to the host.
 * Fatal allocation/runtime exceptions preserve const originals where hardware
 * remains usable; this is not recovery from a poisoned device/runtime.
 */
template <typename Backend>
Result<pfc::core::DataBuffer<Backend, double>, pfc::core::DataBuffer<Backend, Id>>
remap(Grid2D grid, Slot slots, const pfc::core::DataBuffer<Backend, double> &values,
      const pfc::core::DataBuffer<Backend, Id> &seeds, std::span<const Grain> grains,
      const Options &options = {}, pfc::gpuStream_t stream = nullptr) {
  const auto operation_start = detail::Clock::now();
  const auto operation = [&] {
    using Values = pfc::core::DataBuffer<Backend, double>;
    using Labels = pfc::core::DataBuffer<Backend, Id>;
    Result<Values, Labels> result;
    const auto start = detail::Clock::now();
    auto stage_start = start;
    double *stage = nullptr;
    const auto finish = [&](Status status) {
      if (stage) *stage = detail::seconds(stage_start);
      result.status = status;
      result.statistics.total_seconds = detail::seconds(start);
      return std::move(result);
    };
    if (!options.check_now) return finish(Status::Deferred);
    if (!detail::layout(grid, slots, values.size(), seeds.size(), options))
      return finish(Status::InvalidInput);
    auto prepared = pfc::grain::detail::prepare_transfer(slots, grains, {});
    if (prepared.status != TransferStatus::Success) {
      result.transfer_status = prepared.status;
      return finish(Status::InvalidInput);
    }
    const auto synchronize = detail::Clock::now();
    GPU_CHECK(pfc::gpuStreamSynchronize(stream));
    result.statistics.synchronization_seconds = detail::seconds(synchronize);
    const auto detection = detail::Clock::now();
    stage_start = detection;
    stage = &result.statistics.detection_seconds;
    const auto cells = cell_count(grid), n = values.size();
    pfc::core::DataBuffer<Backend, pfc::grain::detail::Assignment> assignments(
        prepared.assignments.size());
    assignments.copy_from_host(prepared.assignments);
    pfc::core::DataBuffer<Backend, std::uint8_t> occupied(n);
    Labels propagated(n);
    pfc::core::DataBuffer<Backend, unsigned> error(1);
    error.copy_from_host(std::vector<unsigned>{0});
    const auto blocks =
        static_cast<unsigned>(std::min<std::size_t>((n - 1) / 256 + 1, 65535));
    GPU_LAUNCH_KERNEL(detail::preflight, blocks, 256,
                      (cells, slots, values.data(), seeds.data(), assignments.data(),
                       assignments.size(), occupied.data(), error.data()),
                      0);
    GPU_CHECK(pfc::gpuDeviceSynchronize());
    auto bad = static_cast<Status>(error.to_host().front());
    if (bad != Status::Success) {
      if (bad == Status::InvalidInput)
        result.transfer_status = TransferStatus::InvalidSupport;
      return finish(bad);
    }
    auto ownership = tracking::propagate(
        Backend{}, grid, slots, std::span<const std::uint8_t>(occupied.data(), n),
        std::span<const Id>(seeds.data(), n), std::span<Id>(propagated.data(), n),
        options.max_sweeps);
    result.statistics.propagation_sweeps = ownership.sweeps;
    if (ownership.status != tracking::Status::Success)
      return finish(ownership.status == tracking::Status::Unseeded
                        ? Status::Unseeded
                        : Status::IterationLimit);
    pfc::core::DataBuffer<Backend, unsigned long long> device_counts(
        assignments.size());
    device_counts.copy_from_host(
        std::vector<unsigned long long>(assignments.size(), 0));
    GPU_LAUNCH_KERNEL(detail::inspect, blocks, 256,
                      (grid, slots, propagated.data(), assignments.data(),
                       assignments.size(), device_counts.data(), error.data()),
                      0);
    GPU_CHECK(pfc::gpuDeviceSynchronize());
    bad = static_cast<Status>(error.to_host().front());
    if (bad != Status::Success) return finish(bad);
    const auto counted = device_counts.to_host();
    if (std::find(counted.begin(), counted.end(), 0) != counted.end()) {
      result.transfer_status = TransferStatus::MissingSupport;
      return finish(Status::MissingSupport);
    }
    const std::vector<std::uint64_t> counts(counted.begin(), counted.end());
    const std::vector<double> weights(counted.begin(), counted.end());
    const auto ids = detail::identities(prepared.assignments);
    result.statistics.detection_seconds = detail::seconds(detection);
    const auto adjacency = detail::Clock::now();
    stage_start = adjacency;
    stage = &result.statistics.adjacency_seconds;
    pfc::core::DataBuffer<Backend, Contact> device_edges(options.contact_capacity);
    auto topology = tracking::adjacency(
        Backend{}, grid, slots, std::span<const Id>(propagated.data(), n), ids,
        std::span<Contact>(device_edges.data(), device_edges.size()),
        options.contact_radius);
    result.statistics.edges = topology.edge_count;
    if (topology.status != tracking::Status::Success) {
      result.statistics.adjacency_seconds = detail::seconds(adjacency);
      return finish(topology.status == tracking::Status::CapacityOverflow
                        ? Status::CapacityOverflow
                        : Status::UnknownIdentity);
    }
    std::vector<Contact> contacts(topology.edge_count);
    if (!contacts.empty())
      GPU_CHECK(pfc::gpuMemcpy(contacts.data(), device_edges.data(),
                               contacts.size() * sizeof(Contact),
                               pfc::gpuMemcpyDeviceToHost));
    auto graph = make_contact_graph(std::move(topology.active), std::move(contacts));
    if (graph.vertices != ids) return finish(Status::MissingSupport);
    result.statistics.graph_device_to_host_bytes =
        topology.edge_count * sizeof(Contact);
    result.statistics.adjacency_seconds = detail::seconds(adjacency);
    const auto solving = detail::Clock::now();
    stage_start = solving;
    stage = &result.statistics.decision_seconds;
    auto decision = detail::decide(graph, grains, slots, weights, options);
    result.statistics.attempts = decision.attempts;
    result.statistics.conflict = decision.conflict;
    result.statistics.decision_seconds = detail::seconds(solving);
    if (decision.status != Status::Success) return finish(decision.status);
    detail::payload(result.statistics, ids, counts, decision.moves);
    const auto moving = detail::Clock::now();
    stage_start = moving;
    stage = &result.statistics.transfer_seconds;
    result.statistics.staged_storage_bytes = n * (sizeof(double) + sizeof(Id));
    auto transaction = pfc::grain::transfer(grid, slots, values, propagated, grains,
                                            decision.moves);
    result.transfer_status = transaction.status;
    result.statistics.transfer_seconds = detail::seconds(moving);
    stage = nullptr;
    if (transaction.status != TransferStatus::Success)
      return finish(Status::TransferFailure);
    result.values = std::move(transaction.values);
    result.labels = std::move(transaction.labels);
    result.snapshot = {options.epoch, std::move(transaction.grains),
                       std::move(graph)};
    pfc::grain::validate(result.snapshot, slots);
    result.statistics.published_storage_bytes = n * (sizeof(double) + sizeof(Id));
    result.statistics.wrapper_storage_bytes =
        n * (sizeof(Id) + sizeof(std::uint8_t)) +
        device_edges.size() * sizeof(Contact) +
        assignments.size() * sizeof(pfc::grain::detail::Assignment) +
        device_counts.size() * sizeof(unsigned long long) + sizeof(unsigned);
    const auto completed = detail::Clock::now();
    GPU_CHECK(pfc::gpuDeviceSynchronize());
    result.statistics.synchronization_seconds += detail::seconds(completed);
    return finish(Status::Success);
  };
  auto completed = operation(); // Includes destruction of all private scratch.
  completed.statistics.total_seconds = detail::seconds(operation_start);
  return completed;
}
} // namespace pfc::grain::remapping
#endif

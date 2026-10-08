// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/grain/topology.hpp>

#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace pfc::grain {

/// One persistent grain changes slot; source must describe the original state.
/// Duplicate identities are invalid. A validated self-move is a no-op.
struct Transfer {
  Id id = background;
  Slot source = unassigned;
  Slot destination = unassigned;
};

/// Failure leaves every input untouched and returns no staged state.
enum class TransferStatus : unsigned {
  Success = 0,
  InvalidLayout,
  InvalidRegistry,
  InvalidTransfer,
  StaleSource,
  MissingSupport,
  InvalidSupport,
  OccupiedDestination
};

/// All members are published together by the caller after Success.
template <typename Values = std::vector<double>, typename Labels = std::vector<Id>>
struct TransferResult {
  TransferStatus status = TransferStatus::Success;
  Values values;
  Labels labels;
  std::vector<Grain> grains;
};

namespace detail {

struct Assignment {
  Id id;
  Slot source;
  Slot destination;
};

struct PreparedTransfer {
  TransferStatus status = TransferStatus::Success;
  std::vector<Assignment> assignments;
  std::vector<Grain> grains;
};

inline TransferStatus transfer_layout(const Grid2D &grid, Slot slots,
                                      std::size_t values, std::size_t labels,
                                      double background_value) {
  if (!std::isfinite(background_value) || grid.nx == 0 || grid.ny == 0 ||
      grid.nx > std::numeric_limits<std::size_t>::max() / grid.ny)
    return TransferStatus::InvalidLayout;
  const auto cells = grid.nx * grid.ny;
  if (slots > std::numeric_limits<std::size_t>::max() / cells)
    return TransferStatus::InvalidLayout;
  const auto count = cells * slots;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(Id) ||
      values != count || labels != count)
    return TransferStatus::InvalidLayout;
  return TransferStatus::Success;
}

OPENPFC_INLINE_HD std::size_t find_assignment(const Assignment *assignments,
                                              std::size_t count, Id id) {
  std::size_t first = 0, last = count;
  while (first < last) {
    const auto middle = first + (last - first) / 2;
    if (assignments[middle].id < id)
      first = middle + 1;
    else
      last = middle;
  }
  return first < count && assignments[first].id == id ? first : count;
}

inline PreparedTransfer prepare_transfer(Slot slots, std::span<const Grain> grains,
                                         std::span<const Transfer> moves) {
  PreparedTransfer prepared;
  Id previous = background;
  for (const auto &grain : grains) {
    if (grain.id == background || grain.id <= previous ||
        (grain.active && (grain.slot == unassigned || grain.slot >= slots)) ||
        (!grain.active && grain.slot != unassigned)) {
      prepared.status = TransferStatus::InvalidRegistry;
      return prepared;
    }
    previous = grain.id;
    if (grain.active)
      prepared.assignments.push_back({grain.id, grain.slot, grain.slot});
  }
  prepared.grains.assign(grains.begin(), grains.end());
  std::vector<Id> moved;
  for (const auto &move : moves) {
    const auto index = find_assignment(prepared.assignments.data(),
                                       prepared.assignments.size(), move.id);
    if (move.id == background || move.destination >= slots ||
        index == prepared.assignments.size() ||
        std::find(moved.begin(), moved.end(), move.id) != moved.end()) {
      prepared.status = TransferStatus::InvalidTransfer;
      return prepared;
    }
    auto &assignment = prepared.assignments[index];
    if (move.source != assignment.source) {
      prepared.status = TransferStatus::StaleSource;
      return prepared;
    }
    moved.push_back(move.id);
    assignment.destination = move.destination;
  }
  for (auto &grain : prepared.grains) {
    if (!grain.active) continue;
    const auto index = find_assignment(prepared.assignments.data(),
                                       prepared.assignments.size(), grain.id);
    grain.slot = prepared.assignments[index].destination;
  }
  return prepared;
}

OPENPFC_INLINE_HD TransferStatus worst(TransferStatus first, TransferStatus second) {
  return static_cast<unsigned>(first) > static_cast<unsigned>(second) ? first
                                                                      : second;
}

/// One thread owns all slots at a cell. Proposals always read original buffers.
/// Returns a deterministic failure priority shared by host and device paths.
OPENPFC_INLINE_HD TransferStatus stage_cell(
    std::size_t cell, std::size_t cells, Slot slots, const double *values,
    const Id *labels, const Assignment *assignments, std::size_t assignment_count,
    double background_value, double *staged_values, Id *staged_labels) {
  auto status = TransferStatus::Success;
  for (Slot slot = 0; slot < slots; ++slot) {
    const auto i = cells * slot + cell;
    staged_values[i] = values[i];
    staged_labels[i] = labels[i];
    // Comparisons reject NaN and both infinities without vendor math intrinsics.
    if (!(values[i] >= background_value &&
          values[i] <= std::numeric_limits<double>::max()) ||
        ((labels[i] == background) != (values[i] == background_value)))
      status = worst(status, TransferStatus::InvalidSupport);
    if (labels[i] == background) continue;
    const auto assignment =
        find_assignment(assignments, assignment_count, labels[i]);
    if (assignment == assignment_count || assignments[assignment].source != slot) {
      status = worst(status, TransferStatus::InvalidSupport);
      continue;
    }
    // Two original grains must never publish to the same slot at this cell.
    for (Slot other = 0; other < slot; ++other) {
      const auto label = labels[cells * other + cell];
      if (label == background) continue;
      const auto neighbor = find_assignment(assignments, assignment_count, label);
      if (neighbor != assignment_count && assignments[neighbor].source == other &&
          assignments[neighbor].destination == assignments[assignment].destination)
        status = worst(status, TransferStatus::OccupiedDestination);
    }
  }
  if (status != TransferStatus::Success) return status;
  // Clear all vacating sources first; then populate from immutable originals.
  for (Slot slot = 0; slot < slots; ++slot) {
    const auto i = cells * slot + cell;
    if (labels[i] == background) continue;
    const auto assignment =
        find_assignment(assignments, assignment_count, labels[i]);
    if (assignments[assignment].destination == slot) continue;
    staged_values[i] = background_value;
    staged_labels[i] = background;
  }
  for (Slot slot = 0; slot < slots; ++slot) {
    const auto i = cells * slot + cell;
    if (labels[i] == background) continue;
    const auto assignment =
        find_assignment(assignments, assignment_count, labels[i]);
    const auto destination = assignments[assignment].destination;
    if (destination == slot) continue;
    staged_values[cells * destination + cell] = values[i];
    staged_labels[cells * destination + cell] = labels[i];
  }
  return status;
}

} // namespace detail

/**
 * Stage a local 2D transaction in slot-major arrays: slot*(nx*ny)+x+nx*y.
 *
 * Values must be finite and >= background_value. Every nonbackground value
 * has a nonzero persistent label, every background value has label zero, and
 * each active grain has support in its recorded source slot. Labels identify
 * the entire support, including diffuse tails; this API discards none.
 *
 * All moves read the original state. Valid cycles and noncontiguous supports
 * are supported. A destination may contain a simultaneously vacating grain,
 * but two supports cannot publish into the same cell/slot. Unrelated values
 * and labels are copied exactly. The multiset of nonbackground values per
 * cell is preserved, hence summed occupancy (value-background_value) is
 * conserved algebraically without tail loss. Floating summation order may
 * differ after a permutation; there is no transfer error or tolerance knob.
 *
 * No graph, tracker, halo, MPI, or snapshot epoch is modified. Publication is
 * caller-owned and must replace values, labels, and grain slots together.
 */
inline TransferResult<>
transfer(const Grid2D &grid, Slot slots, std::span<const double> values,
         std::span<const Id> labels, std::span<const Grain> grains,
         std::span<const Transfer> moves, double background_value = 0.0) {
  const auto layout = detail::transfer_layout(grid, slots, values.size(),
                                              labels.size(), background_value);
  if (layout != TransferStatus::Success) return {layout, {}, {}, {}};
  auto prepared = detail::prepare_transfer(slots, grains, moves);
  if (prepared.status != TransferStatus::Success)
    return {prepared.status, {}, {}, {}};
  const auto cells = cell_count(grid);
  TransferResult<> result;
  result.values.resize(values.size());
  result.labels.resize(labels.size());
  std::vector<unsigned> present(prepared.assignments.size(), 0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    result.status = detail::worst(
        result.status,
        detail::stage_cell(cell, cells, slots, values.data(), labels.data(),
                           prepared.assignments.data(), present.size(),
                           background_value, result.values.data(),
                           result.labels.data()));
    for (Slot slot = 0; slot < slots; ++slot) {
      const auto index = detail::find_assignment(
          prepared.assignments.data(), present.size(), labels[cells * slot + cell]);
      if (index != present.size()) present[index] = 1;
    }
  }
  if (std::find(present.begin(), present.end(), 0) != present.end())
    result.status = detail::worst(result.status, TransferStatus::MissingSupport);
  if (result.status != TransferStatus::Success) return {result.status, {}, {}, {}};
  result.grains = std::move(prepared.grains);
  return result;
}

} // namespace pfc::grain

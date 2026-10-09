// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/kernel/grain/transfer.hpp>

namespace pfc::grain {
namespace detail {
/// Explicit signed amplitude contract. UID labels define full support, not
/// amplitude sign or positivity. Zero-label cells equal declared background;
/// UID-owned cells may carry any finite signed value, including exact zero.
OPENPFC_INLINE_HD TransferStatus stage_signed_cell(
    std::size_t cell, std::size_t cells, Slot slots, const double *values,
    const Id *labels, const Assignment *assignments, std::size_t count,
    double background_value, double *staged_values, Id *staged_labels) {
  auto status = TransferStatus::Success;
  for (Slot s = 0; s < slots; ++s) {
    staged_values[cells * s + cell] = background_value;
    staged_labels[cells * s + cell] = background;
  }
  for (Slot s = 0; s < slots; ++s) {
    auto i = cells * s + cell;
    auto uid = labels[i];
    double value = values[i];
    if (!(value >= -std::numeric_limits<double>::max() &&
          value <= std::numeric_limits<double>::max()) ||
        (!uid && value != background_value)) {
      status = worst(status, TransferStatus::InvalidSupport);
      continue;
    }
    if (!uid) continue;
    auto a = find_assignment(assignments, count, uid);
    if (a == count || assignments[a].source != s) {
      status = worst(status, TransferStatus::InvalidSupport);
      continue;
    }
    auto j = cells * assignments[a].destination + cell;
    if (staged_labels[j]) {
      status = worst(status, TransferStatus::OccupiedDestination);
      continue;
    }
    staged_values[j] = value;
    staged_labels[j] = uid;
  }
  return status;
}
} // namespace detail

/// Owning transactional permutation without projection or tail trimming.
/// Existing positive-amplitude transfer contract is deliberately unchanged.
template <class Grid>
TransferResult<>
transfer_signed(Grid grid, Slot slots, std::span<const double> values,
                std::span<const Id> labels, std::span<const Grain> grains,
                std::span<const Transfer> moves, double background_value = 0) {
  auto layout = detail::transfer_layout(grid, slots, values.size(), labels.size(),
                                        background_value);
  if (layout != TransferStatus::Success) return {layout, {}, {}, {}};
  auto prepared = detail::prepare_transfer(slots, grains, moves);
  if (prepared.status != TransferStatus::Success)
    return {prepared.status, {}, {}, {}};
  TransferResult<> result;
  result.values.resize(values.size());
  result.labels.resize(labels.size());
  std::vector<unsigned> present(prepared.assignments.size(), 0);
  auto cells = cell_count(grid);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    result.status = detail::worst(
        result.status,
        detail::stage_signed_cell(cell, cells, slots, values.data(), labels.data(),
                                  prepared.assignments.data(),
                                  prepared.assignments.size(), background_value,
                                  result.values.data(), result.labels.data()));
    for (Slot s = 0; s < slots; ++s) {
      auto a = detail::find_assignment(prepared.assignments.data(), present.size(),
                                       labels[cells * s + cell]);
      if (a != present.size()) present[a] = 1;
    }
  }
  if (std::find(present.begin(), present.end(), 0) != present.end())
    result.status = detail::worst(result.status, TransferStatus::MissingSupport);
  if (result.status != TransferStatus::Success) return {result.status, {}, {}, {}};
  result.grains = std::move(prepared.grains);
  return result;
}
} // namespace pfc::grain

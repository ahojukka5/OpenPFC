// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/kernel/grain/distributed.hpp>
#include <openpfc/kernel/grain/transfer.hpp>

namespace pfc::grain::distributed {
/// Inclusive global bounds enclose all samples of one already known UID.
/// Disconnected islands and periodic wrapping may substantially enlarge bounds.
struct SupportBounds {
  Id uid = 0;
  Slot slot = unassigned;
  std::uint64_t samples = 0;
  std::array<std::size_t, 3> lower{}, upper{};
};
struct BoundObservation {
  Partition partition;
  std::vector<SupportBounds> bounds;
};
inline BoundObservation observe_bounds(Partition p, Slot slots,
                                       std::span<const Id> labels,
                                       std::span<const Grain> grains) {
  validate(p);
  const auto cells = cell_count(p.local());
  if (!slots || slots == unassigned || cells > SIZE_MAX / slots ||
      labels.size() != cells * slots)
    throw std::invalid_argument("invalid bound observation layout");
  auto prepared = pfc::grain::detail::prepare_transfer(slots, grains, {});
  if (prepared.status != TransferStatus::Success)
    throw std::invalid_argument("invalid bound registry");
  std::vector<SupportBounds> bounds;
  for (auto a : prepared.assignments)
    bounds.push_back({a.id, a.source, 0, {SIZE_MAX, SIZE_MAX, SIZE_MAX}, {0, 0, 0}});
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (!labels[i]) continue;
    auto a = pfc::grain::detail::find_assignment(
        prepared.assignments.data(), prepared.assignments.size(), labels[i]);
    if (a == bounds.size() || prepared.assignments[a].source != i / cells)
      throw std::invalid_argument("unknown UID or source slot in bounds");
    auto &b = bounds[a];
    ++b.samples;
    auto cell = global_cell(p, i % cells);
    const std::array<std::size_t, 3> x{cell % p.global.nx,
                                       (cell / p.global.nx) % p.global.ny,
                                       cell / (p.global.nx * p.global.ny)};
    for (int d = 0; d < 3; ++d) {
      b.lower[d] = std::min(b.lower[d], x[d]);
      b.upper[d] = std::max(b.upper[d], x[d]);
    }
  }
  BoundObservation out{p, {}};
  for (auto b : bounds)
    if (b.samples) out.bounds.push_back(b);
  return out;
}
inline bool within_radius(Grid3D grid, const SupportBounds &a,
                          const SupportBounds &b, std::size_t radius) {
  const std::size_t n[3]{grid.nx, grid.ny, grid.nz};
  const bool periodic[3]{grid.periodic_x, grid.periodic_y, grid.periodic_z};
  std::size_t manhattan = 0, chebyshev = 0;
  for (int d = 0; d < 3; ++d) {
    auto distance = std::numeric_limits<long long>::max();
    for (int shift = -1; shift <= 1; ++shift) {
      if (shift && !periodic[d]) continue;
      auto lo =
          static_cast<long long>(b.lower[d]) + static_cast<long long>(n[d]) * shift;
      auto hi =
          static_cast<long long>(b.upper[d]) + static_cast<long long>(n[d]) * shift;
      auto first = static_cast<long long>(a.lower[d]),
           last = static_cast<long long>(a.upper[d]);
      auto gap = first > hi ? first - hi : lo > last ? lo - last : 0;
      distance = std::min(distance, gap);
    }
    manhattan += std::size_t(distance);
    chebyshev = std::max(chebyshev, std::size_t(distance));
  }
  return pfc::grain::detail::axial(grid) ? manhattan <= radius : chebyshev <= radius;
}
} // namespace pfc::grain::distributed

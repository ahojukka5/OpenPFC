// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <array>
#include <climits>
#include <map>
#include <numeric>
#include <openpfc/kernel/grain/topology.hpp>
#include <queue>
#include <set>

namespace pfc::grain::distributed {
/// Disjoint owned Cartesian box; the global grid determines periodic seams.
struct Partition {
  Grid3D global;
  std::array<std::size_t, 3> lower{}, extent{};
  Grid3D local() const {
    return {extent[0],
            extent[1],
            extent[2],
            extent[0] == global.nx && global.periodic_x,
            extent[1] == global.ny && global.periodic_y,
            extent[2] == global.nz && global.periodic_z,
            global.connectivity};
  }
};
inline void validate(const Partition &p) {
  if (cell_count(p.global) > std::size_t(LLONG_MAX))
    throw std::invalid_argument("distributed global indexing exceeds signed range");
  cell_count(p.local());
  if (!pfc::grain::detail::connectivity_valid(p.global))
    throw std::invalid_argument("distributed grain connectivity must be 6 or 26");
  const std::array<std::size_t, 3> n{p.global.nx, p.global.ny, p.global.nz};
  for (int d = 0; d < 3; ++d)
    if (n[d] > std::size_t(INT_MAX) || p.lower[d] >= n[d] ||
        p.extent[d] > n[d] - p.lower[d])
      throw std::invalid_argument("distributed grain box outside global grid");
}
inline std::size_t global_cell(const Partition &p, std::size_t cell) {
  return p.lower[0] + cell % p.extent[0] +
         p.global.nx *
             (p.lower[1] + (cell / p.extent[0]) % p.extent[1] +
              p.global.ny * (p.lower[2] + cell / (p.extent[0] * p.extent[1])));
}
inline bool boundary(const Partition &p, std::size_t cell, std::size_t width) {
  const std::array<std::size_t, 3> x{cell % p.extent[0],
                                     (cell / p.extent[0]) % p.extent[1],
                                     cell / (p.extent[0] * p.extent[1])};
  for (int d = 0; d < 3; ++d)
    if (p.extent[d] != (d == 0   ? p.global.nx
                        : d == 1 ? p.global.ny
                                 : p.global.nz) &&
        (x[d] < width || p.extent[d] - 1 - x[d] < width))
      return true;
  return false;
}
struct Component {
  Id key = 0; // Rank-local nonzero key, not a newly allocated grain UID.
  Slot slot = unassigned;
  Id seed = 0; // Unique admitted persistent UID, or zero until reconciliation.
  std::uint64_t samples = 0;
};
struct BoundarySample {
  std::uint64_t cell = 0; // Global x-contiguous integer cell index.
  Id component = 0;
  Slot slot = unassigned;
};
struct ComponentContact {
  Id first = 0, second = 0;
};
struct Observation {
  Partition partition;
  std::vector<Component> components;
  std::vector<BoundarySample> boundary;
  std::vector<ComponentContact> contacts;
  // Optional caller-owned local dense component map. Never exchanged by MPI.
  std::vector<Id> local_components;
};

/// CPU component producer. Entire nonzero support is occupied; every global
/// component needs exactly one admitted seed UID. Two distinct seed UIDs in a
/// connected same-slot support are unsafe, not silently resolved by minimum ID.
inline Observation observe(Partition p, Slot slots,
                           std::span<const std::uint8_t> occupied,
                           std::span<const Id> seeds, std::size_t radius) {
  validate(p);
  const auto grid = p.local();
  const auto cells = cell_count(grid);
  if (!slots || slots == unassigned || cells > SIZE_MAX / slots ||
      occupied.size() != cells * slots || seeds.size() != occupied.size() ||
      !radius || radius > static_cast<std::size_t>(INT_MAX / 2))
    throw std::invalid_argument("invalid distributed observation layout");
  Observation out{p, {}, {}, {}, std::vector<Id>(seeds.size())};
  std::queue<std::size_t> pending;
  for (std::size_t i = 0; i < occupied.size(); ++i) {
    if (!occupied[i]) {
      if (seeds[i]) throw std::invalid_argument("UID seed outside occupied support");
      continue;
    }
    if (out.local_components[i]) continue;
    Component component{out.components.size() + 1, Slot(i / cells), 0, 0};
    out.local_components[i] = component.key;
    pending.push(i % cells);
    while (!pending.empty()) {
      const auto cell = pending.front();
      pending.pop();
      const auto at = cells * component.slot + cell;
      ++component.samples;
      if (seeds[at]) {
        if (component.seed && component.seed != seeds[at])
          throw std::domain_error(
              "same-slot component contains different grain UIDs");
        component.seed = seeds[at];
      }
      if (boundary(p, cell, radius))
        out.boundary.push_back(
            {global_cell(p, cell), component.key, component.slot});
      for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if (!pfc::grain::detail::stencil(grid, dx, dy, dz, 1)) continue;
            std::size_t other;
            if (!pfc::grain::detail::offset(grid, cell, dx, dy, dz, other)) continue;
            const auto j = cells * component.slot + other;
            if (occupied[j] && !out.local_components[j]) {
              out.local_components[j] = component.key;
              pending.push(other);
            }
          }
    }
    out.components.push_back(component);
  }
  std::set<std::pair<Id, Id>> contacts;
  const int r = static_cast<int>(radius);
  for (std::size_t cell = 0; cell < cells; ++cell)
    for (Slot slot = 0; slot < slots; ++slot) {
      const Id a = out.local_components[cells * slot + cell];
      if (!a) continue;
      for (int dz = -r; dz <= r; ++dz)
        for (int dy = -r; dy <= r; ++dy)
          for (int dx = -r; dx <= r; ++dx) {
            if (!pfc::grain::detail::stencil(grid, dx, dy, dz, r)) continue;
            std::size_t other;
            if (!pfc::grain::detail::offset(grid, cell, dx, dy, dz, other)) continue;
            for (Slot s = 0; s < slots; ++s) {
              const Id b = out.local_components[cells * s + other];
              if (b && a != b) contacts.emplace(std::min(a, b), std::max(a, b));
            }
          }
    }
  for (auto [a, b] : contacts) out.contacts.push_back({a, b});
  return out;
}
} // namespace pfc::grain::distributed

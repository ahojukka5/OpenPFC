// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <openpfc/kernel/data/host_device.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pfc::grain {

/// Persistent identity; zero labels background, never a grain.
using Id = std::uint64_t;
/// Recyclable, zero-based order-parameter field index, independent of Id.
using Slot = std::uint32_t;
inline constexpr Id background = 0;
inline constexpr Slot unassigned = std::numeric_limits<Slot>::max();
/// Largest finite double, usable in CUDA device code where the host-only
/// constexpr std::numeric_limits<double>::max() is not.
inline constexpr double largest_finite = std::numeric_limits<double>::max();

/// Applications retain this counter across observations/checkpoints.
struct IdentitySequence {
  Id next = 1;
};

/// Allocate without reuse or wraparound; zero marks an exhausted sequence.
inline Id allocate_identity(IdentitySequence &sequence) {
  if (sequence.next == background)
    throw std::overflow_error("grain identity sequence exhausted");
  const Id result = sequence.next;
  sequence.next = result == std::numeric_limits<Id>::max() ? background : result + 1;
  return result;
}

/// Inactive records are tombstones: their identity is retained, slot released.
struct Grain {
  Id id = background;
  Slot slot = unassigned;
  bool active = false;
};

/// Undirected contact, represented canonically as first < second.
struct Contact {
  Id first = background;
  Id second = background;
  bool operator==(const Contact &) const = default;
  bool operator<(const Contact &other) const noexcept {
    return first < other.first || (first == other.first && second < other.second);
  }
};

/// Sorted unique active identities and sorted unique undirected contacts.
/// Storage has no fixed maximum degree. No slot or field data are owned here.
struct ContactGraph {
  std::vector<Id> vertices;
  std::vector<Contact> edges;
};

/// Validate canonical structure; malformed public data never silently truncate.
inline void validate(const ContactGraph &graph) {
  if (!std::is_sorted(graph.vertices.begin(), graph.vertices.end()) ||
      std::adjacent_find(graph.vertices.begin(), graph.vertices.end()) !=
          graph.vertices.end() ||
      (!graph.vertices.empty() && graph.vertices.front() == background))
    throw std::invalid_argument(
        "grain vertices must be sorted unique nonzero identities");
  if (!std::is_sorted(graph.edges.begin(), graph.edges.end()) ||
      std::adjacent_find(graph.edges.begin(), graph.edges.end()) !=
          graph.edges.end())
    throw std::invalid_argument("grain contacts must be sorted and unique");
  for (const auto &edge : graph.edges) {
    if (edge.first >= edge.second ||
        !std::binary_search(graph.vertices.begin(), graph.vertices.end(),
                            edge.first) ||
        !std::binary_search(graph.vertices.begin(), graph.vertices.end(),
                            edge.second))
      throw std::invalid_argument("grain contact has invalid endpoints");
  }
}

/// Canonicalize insertion order, duplicate contacts, and endpoint direction.
/// Self contacts and contacts to background/unknown identities are rejected.
inline ContactGraph make_contact_graph(std::vector<Id> vertices,
                                       std::vector<Contact> edges) {
  std::sort(vertices.begin(), vertices.end());
  vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
  for (auto &edge : edges)
    if (edge.first > edge.second) std::swap(edge.first, edge.second);
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  ContactGraph graph{std::move(vertices), std::move(edges)};
  validate(graph);
  return graph;
}

/// One immutable logical observation. Consumers copy before changing slots.
/// Epoch belongs to the caller; it denotes observations, not numerical time.
struct Snapshot {
  std::uint64_t epoch = 0;
  std::vector<Grain> grains;
  ContactGraph graph;
};

/// Validate identity/slot bounds and exact active graph membership.
/// Slot conflicts on edges are permitted here: resolving them is a separate API.
inline void validate(const Snapshot &snapshot, Slot slot_count) {
  validate(snapshot.graph);
  std::vector<Id> active;
  Id previous = background;
  for (const auto &grain : snapshot.grains) {
    if (grain.id == background || grain.id <= previous)
      throw std::invalid_argument(
          "grain records must be sorted unique nonzero identities");
    previous = grain.id;
    if (grain.active) {
      if (grain.slot == unassigned || grain.slot >= slot_count)
        throw std::invalid_argument("active grain slot is out of range");
      active.push_back(grain.id);
    } else if (grain.slot != unassigned) {
      throw std::invalid_argument("inactive grain must release its slot");
    }
  }
  if (active != snapshot.graph.vertices)
    throw std::invalid_argument("contact graph must contain exactly active grains");
}

/// A retirement does not recycle identity, mutate the graph, or transfer data.
inline Grain retire(Grain grain) {
  if (grain.id == background)
    throw std::invalid_argument("cannot retire background");
  grain.active = false;
  grain.slot = unassigned;
  return grain;
}

/// Explicit topology delta; changed includes both endpoints of changed contacts
/// and all inserted/removed vertices, in sorted order, without duplicates.
struct TopologyChange {
  std::vector<Id> added;
  std::vector<Id> removed;
  std::vector<Contact> added_contacts;
  std::vector<Contact> removed_contacts;
  std::vector<Id> changed;
};

inline TopologyChange diff_topology(const ContactGraph &before,
                                    const ContactGraph &after) {
  validate(before);
  validate(after);
  TopologyChange change;
  std::set_difference(after.vertices.begin(), after.vertices.end(),
                      before.vertices.begin(), before.vertices.end(),
                      std::back_inserter(change.added));
  std::set_difference(before.vertices.begin(), before.vertices.end(),
                      after.vertices.begin(), after.vertices.end(),
                      std::back_inserter(change.removed));
  std::set_difference(after.edges.begin(), after.edges.end(), before.edges.begin(),
                      before.edges.end(), std::back_inserter(change.added_contacts));
  std::set_difference(before.edges.begin(), before.edges.end(), after.edges.begin(),
                      after.edges.end(),
                      std::back_inserter(change.removed_contacts));
  change.changed = change.added;
  change.changed.insert(change.changed.end(), change.removed.begin(),
                        change.removed.end());
  const auto add_endpoints = [&](const auto &edges) {
    for (const auto &edge : edges) {
      change.changed.push_back(edge.first);
      change.changed.push_back(edge.second);
    }
  };
  add_endpoints(change.added_contacts);
  add_endpoints(change.removed_contacts);
  std::sort(change.changed.begin(), change.changed.end());
  change.changed.erase(std::unique(change.changed.begin(), change.changed.end()),
                       change.changed.end());
  return change;
}

/// Explicit single-label cell-neighbor stencil.
enum class Connectivity { Four, Eight, Six, TwentySix };

/// Local single-domain 2D grid; x is contiguous: index = x + nx*y.
/// No ghost cells, decomposition, physical metric, or implicit z dimension.
struct Grid2D {
  std::size_t nx = 0;
  std::size_t ny = 0;
  bool periodic_x = false;
  bool periodic_y = false;
  Connectivity connectivity = Connectivity::Four;
};

/// Local 3D grid; x is contiguous: index = x + nx*(y + ny*z).
/// Six uses face neighbors; TwentySix includes faces, edges and corners.
struct Grid3D {
  std::size_t nx = 0;
  std::size_t ny = 0;
  std::size_t nz = 0;
  bool periodic_x = false;
  bool periodic_y = false;
  bool periodic_z = false;
  Connectivity connectivity = Connectivity::Six;
};

namespace detail {
OPENPFC_INLINE_HD std::size_t depth(Grid2D) { return 1; }
OPENPFC_INLINE_HD std::size_t depth(Grid3D grid) { return grid.nz; }
OPENPFC_INLINE_HD bool periodic_depth(Grid2D) { return false; }
OPENPFC_INLINE_HD bool periodic_depth(Grid3D grid) { return grid.periodic_z; }
OPENPFC_INLINE_HD int depth_radius(Grid2D, int) { return 0; }
OPENPFC_INLINE_HD int depth_radius(Grid3D, int radius) { return radius; }
OPENPFC_INLINE_HD bool connectivity_valid(Grid2D grid) {
  return grid.connectivity == Connectivity::Four ||
         grid.connectivity == Connectivity::Eight;
}
OPENPFC_INLINE_HD bool connectivity_valid(Grid3D grid) {
  return grid.connectivity == Connectivity::Six ||
         grid.connectivity == Connectivity::TwentySix;
}
template <class Grid> OPENPFC_INLINE_HD bool axial(Grid grid) {
  return grid.connectivity == Connectivity::Four ||
         grid.connectivity == Connectivity::Six;
}
template <class Grid>
OPENPFC_INLINE_HD bool stencil(Grid grid, int dx, int dy, int dz, int radius) {
  const auto distance = static_cast<long long>(dx < 0 ? -dx : dx) +
                        static_cast<long long>(dy < 0 ? -dy : dy) +
                        static_cast<long long>(dz < 0 ? -dz : dz);
  return !axial(grid) || distance <= radius;
}
/// Checked by public entry points before use; coordinates use wide signed math.
template <class Grid>
OPENPFC_INLINE_HD bool offset(Grid grid, std::size_t cell, int dx, int dy, int dz,
                              std::size_t &result) {
  const auto nx = static_cast<long long>(grid.nx),
             ny = static_cast<long long>(grid.ny),
             nz = static_cast<long long>(depth(grid));
  auto x = static_cast<long long>(cell % grid.nx) + dx;
  auto y = static_cast<long long>((cell / grid.nx) % grid.ny) + dy;
  auto z = static_cast<long long>(cell / (grid.nx * grid.ny)) + dz;
  if (grid.periodic_x) x = (x % nx + nx) % nx;
  if (grid.periodic_y) y = (y % ny + ny) % ny;
  if (periodic_depth(grid)) z = (z % nz + nz) % nz;
  if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return false;
  result = static_cast<std::size_t>(x) +
           grid.nx *
               (static_cast<std::size_t>(y) + grid.ny * static_cast<std::size_t>(z));
  return true;
}
} // namespace detail

inline std::size_t cell_count(const Grid2D &grid) {
  if (grid.nx == 0 || grid.ny == 0)
    throw std::invalid_argument("grain grid dimensions must be positive");
  if (grid.nx > std::numeric_limits<std::size_t>::max() / grid.ny)
    throw std::overflow_error("grain grid size overflow");
  return grid.nx * grid.ny;
}

template <class Grid>
  requires std::same_as<Grid, Grid3D>
inline std::size_t cell_count(const Grid &grid) {
  if (grid.nz == 0)
    throw std::invalid_argument("grain grid dimensions must be positive");
  const auto plane = cell_count(Grid2D{grid.nx, grid.ny});
  if (plane > std::numeric_limits<std::size_t>::max() / grid.nz)
    throw std::overflow_error("grain grid size overflow");
  return plane * grid.nz;
}

/// Deterministic four- or eight-neighbor oracle for already identified grains.
/// Labels are persistent identities, NOT slots or inferred connected components.
/// Background and same-identity contacts are omitted; stencil is explicit.
inline ContactGraph contact_graph(const Grid2D &grid, std::span<const Id> labels) {
  if (grid.connectivity != Connectivity::Four &&
      grid.connectivity != Connectivity::Eight)
    throw std::invalid_argument("unknown grain connectivity");
  if (labels.size() != cell_count(grid))
    throw std::invalid_argument("grain label count does not match grid");
  std::vector<Id> vertices;
  std::vector<Contact> edges;
  for (Id id : labels)
    if (id != background) vertices.push_back(id);
  const auto add = [&](std::size_t first, std::size_t second) {
    const Id a = labels[first], b = labels[second];
    if (a != background && b != background && a != b) edges.push_back({a, b});
  };
  for (std::size_t y = 0; y < grid.ny; ++y)
    for (std::size_t x = 0; x < grid.nx; ++x) {
      const auto i = x + grid.nx * y;
      if (x + 1 < grid.nx)
        add(i, i + 1);
      else if (grid.periodic_x)
        add(i, grid.nx * y);
      if (y + 1 < grid.ny)
        add(i, i + grid.nx);
      else if (grid.periodic_y)
        add(i, x);
      if (grid.connectivity == Connectivity::Eight &&
          (y + 1 < grid.ny || grid.periodic_y)) {
        const auto next_y = y + 1 < grid.ny ? y + 1 : 0;
        if (x + 1 < grid.nx || grid.periodic_x) {
          const auto next_x = x + 1 < grid.nx ? x + 1 : 0;
          add(i, next_x + grid.nx * next_y);
        }
        if (x > 0 || grid.periodic_x) {
          const auto previous_x = x > 0 ? x - 1 : grid.nx - 1;
          add(i, previous_x + grid.nx * next_y);
        }
      }
    }
  return make_contact_graph(std::move(vertices), std::move(edges));
}

/// Single-label 3D contact oracle at radius one. Cross-plane overlap requires
/// the tracking producer; this raster represents one identity per cell only.
template <class Grid>
  requires std::same_as<Grid, Grid3D>
inline ContactGraph contact_graph(const Grid &grid, std::span<const Id> labels) {
  if (!detail::connectivity_valid(grid))
    throw std::invalid_argument("unknown grain connectivity");
  const auto cells = cell_count(grid);
  if (labels.size() != cells)
    throw std::invalid_argument("grain label count does not match grid");
  if (grid.nx > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      grid.ny > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      grid.nz > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("grain axis exceeds signed coordinate limit");
  std::vector<Id> vertices;
  std::vector<Contact> edges;
  for (std::size_t i = 0; i < cells; ++i) {
    if (!labels[i]) continue;
    vertices.push_back(labels[i]);
    for (int dz = -1; dz <= 1; ++dz)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if ((!dx && !dy && !dz) || !detail::stencil(grid, dx, dy, dz, 1)) continue;
          std::size_t other;
          if (detail::offset(grid, i, dx, dy, dz, other) && labels[other] &&
              labels[other] != labels[i])
            edges.push_back({labels[i], labels[other]});
        }
  }
  return make_contact_graph(std::move(vertices), std::move(edges));
}

} // namespace pfc::grain

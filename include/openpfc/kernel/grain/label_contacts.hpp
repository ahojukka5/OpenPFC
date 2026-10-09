// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cmath>
#include <openpfc/kernel/grain/distributed.hpp>
#include <openpfc/kernel/grain/transfer.hpp>
#include <type_traits>

namespace pfc::grain::distributed {
/// Slot-major padded UID images. Caller publishes generation only after packing
/// the requested accepted state and completing a Full halo exchange. A midpoint
/// image is not the new accepted image. Generation tags cannot detect caller lies.
struct LabelHalo {
  Partition partition;
  std::size_t width = 0;
  std::uint64_t generation = 0;
  bool full = true;
  OPENPFC_INLINE_HD std::size_t field_size() const {
    return (partition.extent[0] + 2 * width) * (partition.extent[1] + 2 * width) *
           (partition.extent[2] + 2 * width);
  }
  OPENPFC_INLINE_HD std::size_t at(long long x, long long y, long long z) const {
    return std::size_t(x + static_cast<long long>(width)) +
           (partition.extent[0] + 2 * width) *
               (std::size_t(y + static_cast<long long>(width)) +
                (partition.extent[1] + 2 * width) *
                    std::size_t(z + static_cast<long long>(width)));
  }
};
struct ContactObservation {
  Partition partition;
  std::vector<Component> support; // key=seed=UID; owned samples only.
  std::vector<Contact> edges;     // Exact known UIDs, including halo neighbors.
  bool proximity = false;         // Distinct UIDs in one slot within radius.
  bool unsafe = false;            // Distinct UIDs in one slot within radius one.
  std::size_t radius = 0;
  Slot slots = 0;
  std::uint64_t generation = 0;
  bool complete_edges = false;
};
namespace label_detail {
inline void validate(LabelHalo halo, Slot slots, std::size_t length,
                     std::size_t radius, std::uint64_t generation) {
  distributed::validate(halo.partition);
  if (!halo.full || !halo.width || halo.width > std::size_t(INT_MAX / 4) ||
      radius > halo.width || !radius || generation != halo.generation || !slots ||
      slots == unassigned)
    throw std::invalid_argument("stale, incomplete or insufficient UID halo");
  std::size_t count = 1;
  for (auto n : halo.partition.extent) {
    if (n + 2 * halo.width > std::size_t(INT_MAX) ||
        count > SIZE_MAX / (n + 2 * halo.width))
      throw std::overflow_error("padded UID halo indexing overflow");
    count *= n + 2 * halo.width;
  }
  if (count > SIZE_MAX / slots || length != count * slots)
    throw std::invalid_argument("padded UID image length mismatch");
}
template <class T> OPENPFC_INLINE_HD Id decode(T value, unsigned &bad) {
  static_assert(std::is_same_v<T, Id> || std::is_same_v<T, double>);
  if constexpr (std::is_same_v<T, double>) {
    // Validate before conversion: NaN/out-of-range casts are not permitted.
    if (!std::isfinite(value) || value < 0 || value > 9007199254740992.0) {
      bad |= 1u;
      return 0;
    }
    const Id id = Id(value);
    if (double(id) != value) bad |= 1u;
    return id;
  } else {
    return value;
  }
}
OPENPFC_INLINE_HD bool neighbor(LabelHalo h, long long x, long long y, long long z,
                                int dx, int dy, int dz) {
  const long long q[3]{x + dx, y + dy, z + dz};
  const auto &g = h.partition.global;
  const long long n[3]{static_cast<long long>(g.nx), static_cast<long long>(g.ny),
                       static_cast<long long>(g.nz)};
  const bool periodic[3]{g.periodic_x, g.periodic_y, g.periodic_z};
  for (int d = 0; d < 3; ++d) {
    auto global = static_cast<long long>(h.partition.lower[d]) + q[d];
    if (!periodic[d] && (global < 0 || global >= n[d])) return false;
  }
  return true;
}
/// Shared pure stencil, not an identity propagation algorithm. Every occupied
/// center already owns an admitted persistent UID; no threshold/UID is invented.
template <class T, class Visitor>
OPENPFC_INLINE_HD unsigned visit(LabelHalo halo, Slot slots, const T *labels,
                                 const pfc::grain::detail::Assignment *assignments,
                                 std::size_t grains, std::size_t cell, int radius,
                                 bool all_slots, Visitor visitor) {
  const auto nx = halo.partition.extent[0], ny = halo.partition.extent[1];
  const long long x = cell % nx, y = (cell / nx) % ny, z = cell / (nx * ny);
  unsigned bad = 0;
  const bool axial = halo.partition.global.connectivity == Connectivity::Six;
  for (Slot s = 0; s < slots; ++s) {
    Id id = decode(labels[s * halo.field_size() + halo.at(x, y, z)], bad);
    if (!id) continue;
    auto a = pfc::grain::detail::find_assignment(assignments, grains, id);
    if (a == grains) {
      bad |= 2u;
      continue;
    }
    if (assignments[a].source != s) {
      bad |= 4u;
      continue;
    }
    visitor.present(a);
    for (int dz = -radius; dz <= radius; ++dz)
      for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
          const int distance = std::abs(dx) + std::abs(dy) + std::abs(dz);
          if ((axial && distance > radius) || !neighbor(halo, x, y, z, dx, dy, dz))
            continue;
          for (Slot t = all_slots ? 0 : s; t < (all_slots ? slots : s + 1); ++t) {
            Id other = decode(
                labels[t * halo.field_size() + halo.at(x + dx, y + dy, z + dz)],
                bad);
            if (!other || other == id) continue;
            auto b = pfc::grain::detail::find_assignment(assignments, grains, other);
            if (b == grains) {
              bad |= 2u;
              continue;
            }
            if (assignments[b].source != t) {
              bad |= 4u;
              continue;
            }
            if (s == t) {
              bad |= 8u;
              if ((axial && distance <= 1) ||
                  (!axial && std::abs(dx) <= 1 && std::abs(dy) <= 1 &&
                   std::abs(dz) <= 1))
                bad |= 16u;
            }
            if (all_slots) visitor.edge(a, b);
          }
        }
  }
  return bad;
}
inline void check(unsigned bad) {
  if (bad & 1u) throw std::invalid_argument("invalid numeric UID label");
  if (bad & 2u) throw std::out_of_range("unknown UID label");
  if (bad & 4u) throw std::invalid_argument("UID label source slot mismatch");
}
inline ContactObservation
finish(LabelHalo halo, std::span<const pfc::grain::detail::Assignment> a,
       std::span<const std::uint64_t> samples, unsigned bad, std::size_t radius,
       std::uint64_t generation, Slot slots, bool extract) {
  check(bad);
  ContactObservation out{halo.partition, {}, {}, bool(bad & 8u), bool(bad & 16u)};
  out.radius = radius;
  out.slots = slots;
  out.generation = generation;
  out.complete_edges = extract;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (samples[i])
      out.support.push_back({a[i].id, a[i].source, a[i].id, samples[i]});
  return out;
}
struct HostVisitor {
  std::uint64_t *samples;
  std::set<std::pair<Id, Id>> *edges;
  const pfc::grain::detail::Assignment *assignments;
  void present(std::size_t a) { ++samples[a]; }
  void edge(std::size_t a, std::size_t b) {
    auto first = assignments[a].id, second = assignments[b].id;
    edges->emplace(std::min(first, second), std::max(first, second));
  }
};
} // namespace label_detail

/// Host realization. Probe mode returns only flags/owned support counts.
/// Full extraction is explicit; overflow/invalid labels leave input untouched.
template <class T>
ContactObservation
label_contacts(LabelHalo halo, Slot slots, std::span<const T> labels,
               std::span<const Grain> grains, std::size_t radius,
               std::uint64_t required_generation, bool extract_edges = false,
               std::size_t edge_capacity = 4096) {
  label_detail::validate(halo, slots, labels.size(), radius, required_generation);
  auto prepared = pfc::grain::detail::prepare_transfer(slots, grains, {});
  if (prepared.status != TransferStatus::Success)
    throw std::invalid_argument("invalid UID contact registry");
  std::vector<std::uint64_t> samples(prepared.assignments.size());
  std::set<std::pair<Id, Id>> edges;
  label_detail::HostVisitor visitor{samples.data(), &edges,
                                    prepared.assignments.data()};
  unsigned bad = 0;
  const auto cells = cell_count(halo.partition.local());
  for (std::size_t c = 0; c < cells; ++c)
    bad |= label_detail::visit(
        halo, slots, labels.data(), prepared.assignments.data(),
        prepared.assignments.size(), c, int(radius), extract_edges, visitor);
  auto out = label_detail::finish(halo, prepared.assignments, samples, bad, radius,
                                  required_generation, slots, extract_edges);
  if (edges.size() > edge_capacity)
    throw std::length_error("UID contact edge capacity");
  for (auto [a, b] : edges) out.edges.push_back({a, b});
  return out;
}
} // namespace pfc::grain::distributed

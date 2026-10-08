// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/runtime/cpu/detail/grain_tracking.hpp>
#include <queue>

namespace grain_3d_test {
using namespace pfc::grain;
inline std::size_t index(Grid3D g, std::size_t x, std::size_t y, std::size_t z) {
  return x + g.nx * (y + g.ny * z);
}
// Independent coordinate-pair metric; no shared offset or stencil helper.
inline std::size_t distance(Grid3D g, std::size_t a, std::size_t b) {
  std::size_t aa[]{a % g.nx, (a / g.nx) % g.ny, a / (g.nx * g.ny)};
  std::size_t bb[]{b % g.nx, (b / g.nx) % g.ny, b / (g.nx * g.ny)};
  std::size_t dims[]{g.nx, g.ny, g.nz};
  bool periodic[]{g.periodic_x, g.periodic_y, g.periodic_z};
  std::size_t sum = 0, maximum = 0;
  for (int k = 0; k < 3; ++k) {
    auto d = aa[k] > bb[k] ? aa[k] - bb[k] : bb[k] - aa[k];
    if (periodic[k]) d = std::min(d, dims[k] - d);
    sum += d;
    maximum = std::max(maximum, d);
  }
  return g.connectivity == Connectivity::Six ? sum : maximum;
}
inline ContactGraph graph(Grid3D g, Slot slots, std::span<const Id> labels,
                          std::size_t radius) {
  const auto cells = cell_count(g);
  std::vector<Id> ids;
  std::vector<Contact> edges;
  for (auto id : labels)
    if (id) ids.push_back(id);
  for (std::size_t a = 0; a < labels.size(); ++a)
    for (std::size_t b = a + 1; b < labels.size(); ++b)
      if (labels[a] && labels[b] && labels[a] != labels[b] &&
          distance(g, a % cells, b % cells) <= radius)
        edges.push_back({labels[a], labels[b]});
  (void)slots;
  return make_contact_graph(ids, edges);
}
// One ordinary BFS per admitted seed; combine distances and identity order.
// Seed cells are barriers to other seeds, matching fixed-seed ownership.
inline std::vector<Id> ownership(Grid3D g, Slot slots,
                                 std::span<const std::uint8_t> occupied,
                                 std::span<const Id> seeds) {
  const auto cells = cell_count(g), absent = std::numeric_limits<std::size_t>::max();
  std::vector<Id> out(seeds.size());
  std::vector<std::size_t> best(seeds.size(), absent);
  for (Slot slot = 0; slot < slots; ++slot)
    for (std::size_t seed = 0; seed < cells; ++seed) {
      const auto si = slot * cells + seed;
      if (!occupied[si] || !seeds[si]) continue;
      std::vector<std::size_t> d(cells, absent);
      d[seed] = 0;
      std::queue<std::size_t> q;
      q.push(seed);
      while (!q.empty()) {
        const auto a = q.front();
        q.pop();
        const auto ai = slot * cells + a;
        if (d[a] < best[ai] || (d[a] == best[ai] && seeds[si] < out[ai])) {
          best[ai] = d[a];
          out[ai] = seeds[si];
        }
        for (std::size_t b = 0; b < cells; ++b) {
          const auto bi = slot * cells + b;
          if (d[b] == absent && occupied[bi] && !seeds[bi] &&
              distance(g, a, b) == 1) {
            d[b] = d[a] + 1;
            q.push(b);
          }
        }
      }
    }
  return out;
}
} // namespace grain_3d_test

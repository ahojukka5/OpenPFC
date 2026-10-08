// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <functional>
#include <openpfc/kernel/grain/diagnostics.hpp>
#include <openpfc/kernel/grain/topology.hpp>
#include <queue>
#include <tuple>

/// Internal independent, small-grid correctness reference; no device dependency.
namespace pfc::grain::tracking::reference {
struct PropagationResult {
  bool complete;
  std::vector<Id> labels; // Zero in active cells lacking any admitted seed path.
};
namespace detail {
inline std::size_t checked_size(Grid2D grid, Slot slots) {
  const auto cells = cell_count(grid);
  if (slots == 0 || slots == unassigned ||
      grid.nx > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      grid.ny > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      (grid.connectivity != Connectivity::Four &&
       grid.connectivity != Connectivity::Eight))
    throw std::invalid_argument("invalid reference tracking geometry");
  if (cells > std::numeric_limits<std::size_t>::max() / slots)
    throw std::overflow_error("reference tracking storage overflow");
  return cells * slots;
}
} // namespace detail

// Independent shortest-path oracle: a priority queue ordered by distance and Id,
// using signed coordinates, not the GPU ping-pong sweeps or device wrap helper.
template <bool Observe>
inline PropagationResult propagate_impl(Grid2D grid, Slot slots,
                                        std::span<const std::uint8_t> occupied,
                                        std::span<const Id> seeds) {
  const auto n = detail::checked_size(grid, slots);
  if (occupied.size() != n || seeds.size() != n)
    throw std::invalid_argument("reference ownership storage mismatch");
  const auto cells = grid.nx * grid.ny;
  auto *counts = diagnostics::host_accesses();
  if constexpr (Observe) {
    diagnostics::scan(counts, diagnostics::Phase::Propagation, slots);
    for (std::size_t i = 0; i < n; ++i)
      diagnostics::write(counts, diagnostics::Field::StagedLabels, sizeof(Id));
  }
  std::vector<Id> output(n, 0);
  std::vector<std::size_t> distance(n, std::numeric_limits<std::size_t>::max());
  using Node = std::tuple<std::size_t, Id, std::size_t>;
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> queue;
  for (std::size_t i = 0; i < n; ++i)
    if (diagnostics::load<Observe>(occupied.data(), i, counts,
                                   diagnostics::Field::Occupancy) &&
        diagnostics::load<Observe>(seeds.data(), i, counts,
                                   diagnostics::Field::Labels)) {
      diagnostics::store<Observe>(
          output.data(), i,
          diagnostics::load<Observe>(seeds.data(), i, counts,
                                     diagnostics::Field::Labels),
          counts, diagnostics::Field::StagedLabels);
      distance[i] = 0;
      queue.emplace(0,
                    diagnostics::load<Observe>(seeds.data(), i, counts,
                                               diagnostics::Field::Labels),
                    i);
    }
  while (!queue.empty()) {
    const auto [length, id, i] = queue.top();
    queue.pop();
    if (length != distance[i] ||
        id != diagnostics::load<Observe>(output.data(), i, counts,
                                         diagnostics::Field::StagedLabels))
      continue;
    const auto plane = (i / cells) * cells;
    const auto x = static_cast<long long>((i % cells) % grid.nx),
               y = static_cast<long long>((i % cells) / grid.nx);
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        if ((dx == 0 && dy == 0) ||
            (grid.connectivity == Connectivity::Four && dx && dy))
          continue;
        long long xx = x + dx, yy = y + dy;
        if (grid.periodic_x)
          xx = (xx + static_cast<long long>(grid.nx)) %
               static_cast<long long>(grid.nx);
        if (grid.periodic_y)
          yy = (yy + static_cast<long long>(grid.ny)) %
               static_cast<long long>(grid.ny);
        if (xx < 0 || yy < 0 || xx >= static_cast<long long>(grid.nx) ||
            yy >= static_cast<long long>(grid.ny))
          continue;
        const auto j = plane + static_cast<std::size_t>(xx) +
                       grid.nx * static_cast<std::size_t>(yy);
        if (!diagnostics::load<Observe>(occupied.data(), j, counts,
                                        diagnostics::Field::Occupancy) ||
            (diagnostics::load<Observe>(seeds.data(), j, counts,
                                        diagnostics::Field::Labels) != 0 &&
             diagnostics::load<Observe>(occupied.data(), j, counts,
                                        diagnostics::Field::Occupancy)))
          continue;
        if (length + 1 < distance[j] ||
            (length + 1 == distance[j] &&
             id < diagnostics::load<Observe>(output.data(), j, counts,
                                             diagnostics::Field::StagedLabels))) {
          distance[j] = length + 1;
          diagnostics::store<Observe>(output.data(), j, id, counts,
                                      diagnostics::Field::StagedLabels);
          queue.emplace(length + 1, id, j);
        }
      }
  }
  bool complete = true;
  if constexpr (Observe)
    diagnostics::scan(counts, diagnostics::Phase::Propagation, slots);
  for (std::size_t i = 0; i < n; ++i)
    if (diagnostics::load<Observe>(occupied.data(), i, counts,
                                   diagnostics::Field::Occupancy) &&
        diagnostics::load<Observe>(output.data(), i, counts,
                                   diagnostics::Field::StagedLabels) == background)
      complete = false;
  return {complete, std::move(output)};
}

// Brute-force coordinate-pair/slot-pair contact oracle independent of GPU
// registry matrix extraction. Includes same-cell cross-plane contact.
template <bool Observe>
inline ContactGraph contact_graph_impl(Grid2D grid, Slot slots,
                                       std::span<const Id> labels,
                                       std::size_t radius = 1) {
  const auto n = detail::checked_size(grid, slots);
  if (labels.size() != n)
    throw std::invalid_argument("reference contact storage mismatch");
  if (radius >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("reference contact radius exceeds coordinate limit");
  auto *counts = diagnostics::host_accesses();
  if constexpr (Observe)
    diagnostics::scan(counts, diagnostics::Phase::Adjacency, slots);
  std::vector<Id> vertices;
  std::vector<Contact> edges;
  const auto cells = grid.nx * grid.ny;
  for (std::size_t i = 0; i < n; ++i) {
    auto id = diagnostics::load<Observe>(labels.data(), i, counts,
                                         diagnostics::Field::Labels);
    if (id) vertices.push_back(id);
  }
  for (std::size_t a = 0; a < labels.size(); ++a)
    for (std::size_t b = a + 1; b < labels.size(); ++b) {
      if (!diagnostics::load<Observe>(labels.data(), a, counts,
                                      diagnostics::Field::Labels) ||
          !diagnostics::load<Observe>(labels.data(), b, counts,
                                      diagnostics::Field::Labels) ||
          diagnostics::load<Observe>(labels.data(), a, counts,
                                     diagnostics::Field::Labels) ==
              diagnostics::load<Observe>(labels.data(), b, counts,
                                         diagnostics::Field::Labels))
        continue;
      const auto ax = (a % cells) % grid.nx, ay = (a % cells) / grid.nx;
      const auto bx = (b % cells) % grid.nx, by = (b % cells) / grid.nx;
      auto dx = ax > bx ? ax - bx : bx - ax, dy = ay > by ? ay - by : by - ay;
      if (grid.periodic_x) dx = std::min(dx, grid.nx - dx);
      if (grid.periodic_y) dy = std::min(dy, grid.ny - dy);
      if ((grid.connectivity == Connectivity::Four && dx + dy <= radius) ||
          (grid.connectivity == Connectivity::Eight && dx <= radius && dy <= radius))
        edges.push_back({diagnostics::load<Observe>(labels.data(), a, counts,
                                                    diagnostics::Field::Labels),
                         diagnostics::load<Observe>(labels.data(), b, counts,
                                                    diagnostics::Field::Labels)});
    }
  return make_contact_graph(vertices, edges);
}

inline PropagationResult propagate(Grid2D grid, Slot slots,
                                   std::span<const std::uint8_t> occupied,
                                   std::span<const Id> seeds) {
  if (diagnostics::current) {
    try {
      diagnostics::current->covered();
      return propagate_impl<true>(grid, slots, occupied, seeds);
    } catch (...) {
      diagnostics::current->fail();
      throw;
    }
  }
  return propagate_impl<false>(grid, slots, occupied, seeds);
}
inline ContactGraph contact_graph(Grid2D grid, Slot slots,
                                  std::span<const Id> labels,
                                  std::size_t radius = 1) {
  if (diagnostics::current) {
    try {
      diagnostics::current->covered();
      return contact_graph_impl<true>(grid, slots, labels, radius);
    } catch (...) {
      diagnostics::current->fail();
      throw;
    }
  }
  return contact_graph_impl<false>(grid, slots, labels, radius);
}

} // namespace pfc::grain::tracking::reference

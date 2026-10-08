// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

/** @file grain_coloring.hpp
 *  @brief Internal deterministic reference coloring without field ownership.
 */
namespace pfc::grain::reference {
using Color = std::size_t;
inline constexpr Color uncolored = std::numeric_limits<Color>::max();
using Graph = std::vector<std::vector<std::size_t>>;

enum class Status { success, infeasible, search_limit };
struct Limits {
  std::size_t attempts = 1000000;
  std::size_t depth = 5; // Local repair only; zero permits direct moves.
};
struct Result {
  Status status;
  std::vector<Color> colors; // Empty on failure; inputs are never modified.
  std::size_t attempts = 0;
};
struct Migration {
  std::size_t changed_vertices = 0;
  double weight = 0; // Caller-supplied proxy, not actual transferred bytes.
};

inline Graph make_graph(std::size_t vertices,
                        std::span<const std::pair<std::size_t, std::size_t>> edges) {
  Graph graph(vertices);
  for (auto [u, v] : edges) {
    if (u >= vertices || v >= vertices || u == v)
      throw std::invalid_argument("coloring edge has invalid endpoints");
    graph[u].push_back(v);
    graph[v].push_back(u);
  }
  for (auto &neighbors : graph) {
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());
  }
  return graph;
}

inline void validate(const Graph &graph) {
  for (std::size_t v = 0; v < graph.size(); ++v) {
    auto sorted = graph[v];
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
      throw std::invalid_argument("coloring graph contains duplicate neighbors");
    for (auto u : sorted) {
      if (u >= graph.size() || u == v ||
          std::find(graph[u].begin(), graph[u].end(), v) == graph[u].end())
        throw std::invalid_argument("coloring graph must be simple and symmetric");
    }
  }
}

inline bool proper_coloring(const Graph &graph, std::span<const Color> colors,
                            std::size_t palette) {
  validate(graph);
  if (colors.size() != graph.size()) return false;
  for (std::size_t v = 0; v < graph.size(); ++v) {
    if (colors[v] >= palette) return false;
    for (auto u : graph[v])
      if (colors[u] == colors[v]) return false;
  }
  return true;
}

inline Migration migration(std::span<const Color> before,
                           std::span<const Color> after,
                           std::span<const double> weights = {}) {
  if (before.size() != after.size() ||
      (!weights.empty() && weights.size() != before.size()))
    throw std::invalid_argument("migration dimensions differ");
  Migration result;
  for (std::size_t v = 0; v < before.size(); ++v) {
    const double weight = weights.empty() ? 1.0 : weights[v];
    if (!std::isfinite(weight) || weight < 0)
      throw std::invalid_argument("migration weight must be finite and nonnegative");
    if (before[v] != after[v]) {
      ++result.changed_vertices;
      result.weight += weight;
      if (!std::isfinite(result.weight))
        throw std::overflow_error("migration proxy sum overflow");
    }
  }
  return result;
}

/** Align a new partition's numeric labels to old slots with minimum weighted
 * migration. Hungarian assignment changes only a color permutation, so any
 * proper coloring stays proper. Empty weights minimize changed vertices.
 */
inline std::vector<Color> align_colors(std::span<const Color> before,
                                       std::span<const Color> after,
                                       std::size_t palette,
                                       std::span<const double> weights = {}) {
  if (before.size() != after.size() ||
      (!weights.empty() && weights.size() != before.size()) || palette == uncolored)
    throw std::invalid_argument("alignment dimensions differ");
  if (palette == 0) {
    if (!before.empty()) throw std::invalid_argument("alignment palette is empty");
    return {};
  }
  // Negative retained weight is a minimum assignment cost. Long double
  // accumulation avoids losing ordinary double-scale migration differences.
  std::vector<std::vector<long double>> cost(palette,
                                             std::vector<long double>(palette, 0));
  for (std::size_t v = 0; v < before.size(); ++v) {
    double weight = weights.empty() ? 1.0 : weights[v];
    if (before[v] >= palette || after[v] >= palette || !std::isfinite(weight) ||
        weight < 0)
      throw std::invalid_argument("invalid alignment color or weight");
    cost[after[v]][before[v]] -= weight;
    if (!std::isfinite(cost[after[v]][before[v]]))
      throw std::overflow_error("alignment cost overflow");
  }
  std::vector<long double> row_potential(palette + 1), col_potential(palette + 1);
  std::vector<std::size_t> matched_row(palette + 1), previous(palette + 1);
  for (std::size_t row = 1; row <= palette; ++row) {
    matched_row[0] = row;
    std::size_t col = 0;
    std::vector<long double> distance(palette + 1,
                                      std::numeric_limits<long double>::infinity());
    std::vector<bool> used(palette + 1, false);
    do {
      used[col] = true;
      auto active_row = matched_row[col];
      long double delta = std::numeric_limits<long double>::infinity();
      std::size_t next = 0;
      for (std::size_t candidate = 1; candidate <= palette; ++candidate) {
        if (used[candidate]) continue;
        auto reduced = cost[active_row - 1][candidate - 1] -
                       row_potential[active_row] - col_potential[candidate];
        if (reduced < distance[candidate]) {
          distance[candidate] = reduced;
          previous[candidate] = col;
        }
        if (distance[candidate] < delta) {
          delta = distance[candidate];
          next = candidate;
        }
      }
      for (std::size_t candidate = 0; candidate <= palette; ++candidate) {
        if (used[candidate]) {
          row_potential[matched_row[candidate]] += delta;
          col_potential[candidate] -= delta;
        } else {
          distance[candidate] -= delta;
        }
      }
      col = next;
    } while (matched_row[col] != 0);
    do {
      auto predecessor = previous[col];
      matched_row[col] = matched_row[predecessor];
      col = predecessor;
    } while (col != 0);
  }
  std::vector<Color> mapping(palette), aligned(after.begin(), after.end());
  for (std::size_t col = 1; col <= palette; ++col)
    mapping[matched_row[col] - 1] = col - 1;
  for (auto &color : aligned) color = mapping[color];
  return aligned;
}

namespace detail {
inline bool available(const Graph &graph, const std::vector<Color> &colors,
                      std::size_t vertex, Color candidate) {
  return std::none_of(graph[vertex].begin(), graph[vertex].end(),
                      [&](auto u) { return colors[u] == candidate; });
}

// DSATUR: descending distinct colored neighbors, degree, ascending index.
inline std::size_t select_vertex(const Graph &graph,
                                 const std::vector<Color> &colors) {
  std::size_t best = graph.size(), best_saturation = 0, best_degree = 0;
  for (std::size_t v = 0; v < graph.size(); ++v) {
    if (colors[v] != uncolored) continue;
    std::vector<Color> neighbor_colors;
    for (auto u : graph[v])
      if (colors[u] != uncolored) neighbor_colors.push_back(colors[u]);
    std::sort(neighbor_colors.begin(), neighbor_colors.end());
    neighbor_colors.erase(
        std::unique(neighbor_colors.begin(), neighbor_colors.end()),
        neighbor_colors.end());
    auto saturation = neighbor_colors.size();
    if (best == graph.size() || saturation > best_saturation ||
        (saturation == best_saturation && graph[v].size() > best_degree)) {
      best = v;
      best_saturation = saturation;
      best_degree = graph[v].size();
    }
  }
  return best;
}

inline Status global_search(const Graph &graph, std::vector<Color> &colors,
                            std::size_t palette, std::size_t budget,
                            std::size_t &attempts) {
  auto v = select_vertex(graph, colors);
  if (v == graph.size()) return Status::success;
  for (Color c = 0; c < palette; ++c) {
    if (!available(graph, colors, v, c)) continue;
    if (attempts == budget) return Status::search_limit;
    ++attempts;
    colors[v] = c;
    auto status = global_search(graph, colors, palette, budget, attempts);
    if (status == Status::success) return status;
    colors[v] = uncolored;
    if (status == Status::search_limit) return status;
  }
  return Status::infeasible;
}

// Reserve a candidate before recursively moving its blockers. Every candidate
// is a transaction on a private assignment; ancestors cannot be displaced.
inline bool repair(const Graph &graph, std::vector<Color> &colors,
                   std::size_t palette, std::size_t vertex,
                   std::vector<bool> &locked, std::size_t depth,
                   const Limits &limits, std::size_t &attempts) {
  if (locked[vertex] || depth > limits.depth) return false;
  auto original = colors;
  std::vector<Color> candidates(palette);
  std::iota(candidates.begin(), candidates.end(), Color{0});
  std::vector<std::size_t> blockers(palette), usage(palette);
  for (auto c : colors) ++usage[c];
  for (auto u : graph[vertex]) ++blockers[colors[u]];
  std::sort(candidates.begin(), candidates.end(), [&](auto a, auto b) {
    if (blockers[a] != blockers[b]) return blockers[a] < blockers[b];
    if (usage[a] != usage[b]) return usage[a] < usage[b];
    return a < b;
  });
  locked[vertex] = true;
  for (auto candidate : candidates) {
    if (candidate == original[vertex]) continue;
    if (attempts == limits.attempts) break;
    ++attempts;
    colors = original;
    colors[vertex] = candidate;
    bool solved = true;
    // Canonical traversal regardless of adjacency storage order.
    auto neighbors = graph[vertex];
    std::sort(neighbors.begin(), neighbors.end());
    for (auto u : neighbors) {
      if (colors[u] != candidate) continue;
      if (!repair(graph, colors, palette, u, locked, depth + 1, limits, attempts)) {
        solved = false;
        break;
      }
    }
    if (solved && available(graph, colors, vertex, candidate)) {
      locked[vertex] = false;
      return true;
    }
  }
  colors = original;
  locked[vertex] = false;
  return false;
}
} // namespace detail

enum class GlobalOrder { saturation, largest_first };

/** Practical single-pass global heuristics, separated from exact fallback.
 * A greedy dead end is search_limit, not a proof of uncolorability.
 */
inline Result global_greedy_coloring(const Graph &graph, std::size_t palette,
                                     GlobalOrder order = GlobalOrder::saturation,
                                     Limits limits = {}) {
  validate(graph);
  if (palette == uncolored ||
      (order != GlobalOrder::saturation && order != GlobalOrder::largest_first))
    throw std::invalid_argument("invalid global coloring policy");
  std::vector<Color> colors(graph.size(), uncolored);
  std::vector<std::size_t> vertices(graph.size());
  std::iota(vertices.begin(), vertices.end(), std::size_t{0});
  std::sort(vertices.begin(), vertices.end(), [&](auto a, auto b) {
    if (graph[a].size() != graph[b].size()) return graph[a].size() > graph[b].size();
    return a < b;
  });
  std::size_t attempts = 0;
  for (std::size_t step = 0; step < graph.size(); ++step) {
    auto vertex = order == GlobalOrder::saturation
                      ? detail::select_vertex(graph, colors)
                      : vertices[step];
    bool assigned = false;
    for (Color color = 0; color < palette; ++color) {
      if (!detail::available(graph, colors, vertex, color)) continue;
      if (attempts == limits.attempts) return {Status::search_limit, {}, attempts};
      ++attempts;
      colors[vertex] = color;
      assigned = true;
      break;
    }
    if (!assigned) return {Status::search_limit, {}, attempts};
  }
  if (!proper_coloring(graph, colors, palette))
    throw std::logic_error("greedy coloring postcondition failed");
  return {Status::success, std::move(colors), attempts};
}

/** Global first-solution DSATUR with bounded complete backtracking.
 * Infeasible means the complete search finished. A budget limit proves nothing
 * about feasibility. This reference has exponential worst-case cost.
 */
inline Result global_coloring(const Graph &graph, std::size_t palette,
                              Limits limits = {}) {
  validate(graph);
  if (palette == uncolored)
    throw std::invalid_argument("palette collides with uncolored sentinel");
  std::vector<Color> colors(graph.size(), uncolored);
  std::size_t attempts = 0;
  auto status =
      detail::global_search(graph, colors, palette, limits.attempts, attempts);
  if (status == Status::success && !proper_coloring(graph, colors, palette))
    throw std::logic_error("global coloring postcondition failed");
  if (status != Status::success) colors.clear();
  return {status, std::move(colors), attempts};
}

/** Preserve legal assignments, then recursively free colors at conflict ends.
 * Search_limit includes depth-limited local dead ends, even on impossible
 * graphs: this heuristic never claims a proof of infeasibility or optimality.
 */
inline Result incremental_coloring(const Graph &graph,
                                   std::span<const Color> initial,
                                   std::size_t palette, Limits limits = {}) {
  validate(graph);
  if (palette == uncolored || initial.size() != graph.size() ||
      std::any_of(initial.begin(), initial.end(),
                  [&](auto c) { return c >= palette; }))
    throw std::invalid_argument("initial coloring does not fit graph/palette");
  std::vector<Color> colors(initial.begin(), initial.end());
  std::vector<bool> locked(graph.size(), false);
  std::size_t attempts = 0;
  while (true) {
    std::size_t first = graph.size(), second = graph.size();
    for (std::size_t v = 0; v < graph.size() && first == graph.size(); ++v) {
      for (auto u : graph[v]) {
        if (v < u && colors[v] == colors[u] &&
            (second == graph.size() || u < second)) {
          first = v;
          second = u;
        }
      }
    }
    if (first == graph.size()) {
      if (!proper_coloring(graph, colors, palette))
        throw std::logic_error("incremental coloring postcondition failed");
      return {Status::success, std::move(colors), attempts};
    }
    if (!detail::repair(graph, colors, palette, first, locked, 0, limits,
                        attempts) &&
        !detail::repair(graph, colors, palette, second, locked, 0, limits, attempts))
      return {Status::search_limit, {}, attempts};
  }
}
} // namespace pfc::grain::reference

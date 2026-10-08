// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <numeric>
#include <openpfc/runtime/cpu/detail/grain_coloring.hpp>

namespace ref = pfc::grain::reference;
namespace {
// Independent brute-force mixed-radix enumeration: no DSATUR or recursion.
bool legal(const ref::Graph &g, const std::vector<ref::Color> &c, std::size_t k) {
  if (c.size() != g.size()) return false;
  for (std::size_t v = 0; v < g.size(); ++v) {
    if (c[v] >= k) return false;
    for (auto u : g[v])
      if (c[u] == c[v]) return false;
  }
  return true;
}
struct Oracle {
  bool feasible;
  std::size_t changed;
  double weight;
};
Oracle enumerate(const ref::Graph &g, std::size_t k,
                 const std::vector<ref::Color> &initial,
                 const std::vector<double> &weights) {
  Oracle result{false, g.size() + 1, std::numeric_limits<double>::infinity()};
  std::size_t assignments = 1;
  for (std::size_t i = 0; i < g.size(); ++i) assignments *= k;
  for (std::size_t code = 0; code < assignments; ++code) {
    std::vector<ref::Color> c(g.size());
    auto remaining = code;
    for (auto &color : c) {
      color = remaining % k;
      remaining /= k;
    }
    if (!legal(g, c, k)) continue;
    result.feasible = true;
    std::size_t changed = 0;
    double weight = 0;
    for (std::size_t v = 0; v < c.size(); ++v)
      if (c[v] != initial[v]) {
        ++changed;
        weight += weights[v];
      }
    result.changed = std::min(result.changed, changed);
    result.weight = std::min(result.weight, weight);
  }
  return result;
}
ref::Graph graph(std::size_t n,
                 std::initializer_list<std::pair<std::size_t, std::size_t>> edges) {
  return ref::make_graph(n, std::span(edges.begin(), edges.size()));
}
} // namespace

TEST_CASE("Coloring agrees with exhaustive tiny-graph feasibility",
          "[grain][coloring]") {
  // Every labeled simple graph through five vertices, palettes 1..3.
  for (std::size_t n = 0; n <= 5; ++n) {
    auto pairs = n * (n == 0 ? 0 : n - 1) / 2;
    for (std::size_t mask = 0; mask < (std::size_t{1} << pairs); ++mask) {
      ref::Graph g(n);
      std::size_t bit = 0;
      for (std::size_t u = 0; u < n; ++u)
        for (std::size_t v = u + 1; v < n; ++v, ++bit)
          if (mask & (std::size_t{1} << bit)) {
            g[u].push_back(v);
            g[v].push_back(u);
          }
      for (std::size_t k = 1; k <= 3; ++k) {
        CAPTURE(n, mask, k);
        std::vector<ref::Color> initial(n, 0);
        std::vector<double> weights(n);
        std::iota(weights.begin(), weights.end(), 1.0);
        auto oracle = enumerate(g, k, initial, weights);
        auto global = ref::global_coloring(g, k);
        REQUIRE(global.status ==
                (oracle.feasible ? ref::Status::success : ref::Status::infeasible));
        if (oracle.feasible)
          REQUIRE(legal(g, global.colors, k));
        else
          REQUIRE(global.colors.empty());
        auto local = ref::incremental_coloring(g, initial, k);
        if (local.status == ref::Status::success) {
          REQUIRE(oracle.feasible);
          REQUIRE(legal(g, local.colors, k));
          auto cost = ref::migration(initial, local.colors, weights);
          REQUIRE(cost.changed_vertices >= oracle.changed);
          REQUIRE(cost.weight >= oracle.weight);
          auto again = ref::incremental_coloring(g, local.colors, k, {0, 0});
          REQUIRE(again.colors == local.colors);
          REQUIRE(again.attempts == 0);
        } else {
          REQUIRE(local.status == ref::Status::search_limit);
          REQUIRE(local.colors.empty());
        }
        REQUIRE(initial == std::vector<ref::Color>(n, 0));
      }
    }
  }
}

TEST_CASE("Local coloring frees saturated neighborhoods transactionally",
          "[grain][coloring]") {
  // Vertex 0 sees all three colors; every conflict endpoint needs a chain.
  auto g = graph(6, {{0, 1}, {0, 2}, {0, 3}, {1, 4}, {1, 5}});
  std::vector<ref::Color> initial{0, 0, 1, 2, 1, 2};
  auto limited = ref::incremental_coloring(g, initial, 3, {100, 0});
  REQUIRE(limited.status == ref::Status::search_limit);
  REQUIRE(limited.colors.empty());
  auto result = ref::incremental_coloring(g, initial, 3, {100, 2});
  REQUIRE(result.status == ref::Status::success);
  REQUIRE(legal(g, result.colors, 3));
  REQUIRE(ref::migration(initial, result.colors).changed_vertices == 2);
  REQUIRE(initial == std::vector<ref::Color>{0, 0, 1, 2, 1, 2});
  REQUIRE(ref::incremental_coloring(g, initial, 3, {100, 2}).colors ==
          result.colors);
  for (auto &neighbors : g) std::reverse(neighbors.begin(), neighbors.end());
  REQUIRE(ref::incremental_coloring(g, initial, 3, {100, 2}).colors ==
          result.colors);
}

TEST_CASE("Global coloring distinguishes limits from impossibility",
          "[grain][coloring]") {
  auto triangle = graph(3, {{0, 1}, {1, 2}, {0, 2}});
  REQUIRE(ref::global_coloring(triangle, 2).status == ref::Status::infeasible);
  REQUIRE(ref::global_coloring(triangle, 3, {0, 0}).status ==
          ref::Status::search_limit);
  REQUIRE(ref::global_coloring(triangle, 3, {3, 0}).status == ref::Status::success);
  REQUIRE(ref::global_coloring({}, 0).status == ref::Status::success);
  REQUIRE(ref::global_coloring(ref::Graph(1), 0).status == ref::Status::infeasible);
  REQUIRE(ref::incremental_coloring({}, {}, 0).status == ref::Status::success);
  auto cycle = graph(5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}});
  REQUIRE(ref::global_coloring(cycle, 2).status == ref::Status::infeasible);
  REQUIRE(legal(cycle, ref::global_coloring(cycle, 3).colors, 3));
  std::vector<std::size_t> permutation{3, 0, 4, 1, 2};
  ref::Graph permuted(5);
  for (std::size_t v = 0; v < 5; ++v)
    for (auto u : cycle[v]) permuted[permutation[v]].push_back(permutation[u]);
  REQUIRE(legal(permuted, ref::global_coloring(permuted, 3).colors, 3));
}

TEST_CASE("Coloring validates graph and migration inputs", "[grain][coloring]") {
  REQUIRE_THROWS_AS(ref::global_coloring({{1}, {}}, 2), std::invalid_argument);
  REQUIRE_THROWS_AS(ref::global_coloring({{0}}, 2), std::invalid_argument);
  REQUIRE_THROWS_AS(ref::global_coloring({{1, 1}, {0}}, 2), std::invalid_argument);
  REQUIRE_THROWS_AS(ref::global_coloring({{2}, {}}, 2), std::invalid_argument);
  REQUIRE_THROWS_AS(ref::incremental_coloring(ref::Graph(1), {}, 2),
                    std::invalid_argument);
  std::vector<ref::Color> before{0, 1, 2}, after{2, 1, 0};
  std::vector<double> weights{2, 100, 3};
  auto cost = ref::migration(before, after, weights);
  REQUIRE(cost.changed_vertices == 2);
  REQUIRE(cost.weight == 5);
  weights[1] = -1;
  REQUIRE_THROWS_AS(ref::migration(before, after, weights), std::invalid_argument);
  weights[1] = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(ref::migration(before, after, weights), std::invalid_argument);
}

TEST_CASE("Global labels align at minimum weighted migration", "[grain][coloring]") {
  auto g = graph(4, {{0, 1}, {1, 2}, {2, 3}, {3, 0}});
  std::vector<ref::Color> before{1, 0, 1, 0};
  auto global = ref::global_coloring(g, 2);
  REQUIRE(ref::migration(before, global.colors).changed_vertices == 4);
  auto aligned = ref::align_colors(before, global.colors, 2);
  REQUIRE(aligned == before);
  REQUIRE(legal(g, aligned, 2));
  // Every old/new assignment of three vertices, compare every permutation.
  for (std::size_t a = 0; a < 27; ++a)
    for (std::size_t b = 0; b < 27; ++b) {
      std::vector<ref::Color> old_colors(3), new_colors(3);
      auto aa = a, bb = b;
      for (std::size_t v = 0; v < 3; ++v) {
        old_colors[v] = aa % 3;
        aa /= 3;
        new_colors[v] = bb % 3;
        bb /= 3;
      }
      std::vector<double> weights{0.5, 3, 1};
      double best = std::numeric_limits<double>::infinity();
      std::vector<ref::Color> permutation{0, 1, 2};
      do {
        auto candidate = new_colors;
        for (auto &c : candidate) c = permutation[c];
        best = std::min(best, ref::migration(old_colors, candidate, weights).weight);
      } while (std::next_permutation(permutation.begin(), permutation.end()));
      auto result = ref::align_colors(old_colors, new_colors, 3, weights);
      REQUIRE(ref::migration(old_colors, result, weights).weight == best);
    }
  REQUIRE(ref::align_colors({}, {}, 0).empty());
  REQUIRE_THROWS_AS(ref::align_colors(before, global.colors, 0),
                    std::invalid_argument);
}

TEST_CASE("Local successes remain legal for arbitrary initial assignments",
          "[grain][coloring]") {
  // All graphs and all 3-slot assignments on four vertices.
  for (std::size_t mask = 0; mask < 64; ++mask) {
    ref::Graph g(4);
    std::size_t bit = 0;
    for (std::size_t u = 0; u < 4; ++u)
      for (std::size_t v = u + 1; v < 4; ++v, ++bit)
        if (mask & (std::size_t{1} << bit)) {
          g[u].push_back(v);
          g[v].push_back(u);
        }
    for (std::size_t code = 0; code < 81; ++code) {
      auto remaining = code;
      std::vector<ref::Color> initial(4);
      for (auto &c : initial) {
        c = remaining % 3;
        remaining /= 3;
      }
      auto original = initial;
      auto result = ref::incremental_coloring(g, initial, 3, {10000, 4});
      CAPTURE(mask, code);
      REQUIRE(initial == original);
      if (result.status == ref::Status::success)
        REQUIRE(legal(g, result.colors, 3));
      else
        REQUIRE(result.colors.empty());
    }
  }
}

TEST_CASE("Practical global heuristics expose dead ends without false proofs",
          "[grain][coloring]") {
  auto triangle = graph(3, {{0, 1}, {0, 2}, {1, 2}});
  for (auto order :
       {ref::GlobalOrder::saturation, ref::GlobalOrder::largest_first}) {
    auto good = ref::global_greedy_coloring(triangle, 3, order);
    REQUIRE(good.status == ref::Status::success);
    REQUIRE(legal(triangle, good.colors, 3));
    REQUIRE(good.attempts == 3);
    auto impossible = ref::global_greedy_coloring(triangle, 2, order);
    REQUIRE(impossible.status == ref::Status::search_limit);
    REQUIRE(impossible.colors.empty());
    REQUIRE(ref::global_greedy_coloring(triangle, 3, order, {2, 0}).status ==
            ref::Status::search_limit);
    REQUIRE(ref::global_greedy_coloring({}, 0, order).status ==
            ref::Status::success);
  }
  // Bipartite path ordered so largest-first greedily needs a third color.
  // Exact DSATUR remains a separate oracle and finds the legal two-coloring.
  auto path = graph(6, {{0, 2}, {2, 4}, {4, 5}, {5, 3}, {3, 1}});
  REQUIRE(
      ref::global_greedy_coloring(path, 2, ref::GlobalOrder::largest_first).status ==
      ref::Status::search_limit);
  auto global = ref::global_coloring(path, 2);
  REQUIRE(global.status == ref::Status::success);
  REQUIRE(legal(path, global.colors, 2));
}

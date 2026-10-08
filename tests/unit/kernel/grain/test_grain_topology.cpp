// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <openpfc/kernel/grain/topology.hpp>
#include <type_traits>

namespace gr = pfc::grain;

TEST_CASE("grain identities do not recycle or wrap", "[grain][contract][unit]") {
  gr::IdentitySequence ids;
  REQUIRE(gr::allocate_identity(ids) == 1);
  REQUIRE(gr::allocate_identity(ids) == 2);
  auto grain = gr::retire({1, 0, true});
  REQUIRE(grain.id == 1);
  REQUIRE_FALSE(grain.active);
  REQUIRE(grain.slot == gr::unassigned);
  REQUIRE(gr::allocate_identity(ids) == 3);
  gr::IdentitySequence last{std::numeric_limits<gr::Id>::max()};
  REQUIRE(gr::allocate_identity(last) == std::numeric_limits<gr::Id>::max());
  REQUIRE_THROWS_AS(gr::allocate_identity(last), std::overflow_error);
  REQUIRE(last.next == gr::background);
  REQUIRE_THROWS_AS(gr::retire({}), std::invalid_argument);
  static_assert(std::is_trivially_copyable_v<gr::Grain>);
}

TEST_CASE("grain contact graphs canonicalize and reject malformed edges",
          "[grain][contract][unit]") {
  const auto graph = gr::make_contact_graph({7, 2, 9, 7}, {{7, 2}, {2, 7}});
  REQUIRE(graph.vertices == std::vector<gr::Id>{2, 7, 9});
  REQUIRE(graph.edges == std::vector<gr::Contact>{{2, 7}});
  REQUIRE_THROWS_AS(gr::make_contact_graph({0, 1}, {}), std::invalid_argument);
  REQUIRE_THROWS_AS(gr::make_contact_graph({1}, {{1, 1}}), std::invalid_argument);
  REQUIRE_THROWS_AS(gr::make_contact_graph({1}, {{1, 2}}), std::invalid_argument);
  REQUIRE_THROWS_AS(gr::validate(gr::ContactGraph{{2, 1}, {}}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(gr::validate(gr::ContactGraph{{1, 2}, {{1, 2}, {1, 2}}}),
                    std::invalid_argument);
}

TEST_CASE("periodic grain edges are explicit and degenerate axes are safe",
          "[grain][contract][unit]") {
  const std::vector<gr::Id> labels{1, 0, 2, 3, 0, 4};
  const auto open = gr::contact_graph({3, 2}, labels);
  REQUIRE(open.edges == std::vector<gr::Contact>{{1, 3}, {2, 4}});
  const auto periodic = gr::contact_graph({3, 2, true, true}, labels);
  REQUIRE(periodic.edges ==
          std::vector<gr::Contact>{{1, 2}, {1, 3}, {2, 4}, {3, 4}});
  const std::vector<gr::Id> one{8};
  REQUIRE(gr::contact_graph({1, 1, true, true}, one).edges.empty());
  const std::vector<gr::Id> line{1, 2, 3};
  REQUIRE(gr::contact_graph({1, 3, true, true}, line).edges ==
          std::vector<gr::Contact>{{1, 2}, {1, 3}, {2, 3}});
  const std::vector<gr::Id> diagonal{1, 0, 0, 2};
  REQUIRE(gr::contact_graph({2, 2}, diagonal).edges.empty());
  REQUIRE(gr::contact_graph({2, 2, false, false, gr::Connectivity::Eight}, diagonal)
              .edges == std::vector<gr::Contact>{{1, 2}});
  const std::vector<gr::Id> wrap_diagonal{1, 0, 0, 0, 0, 2};
  REQUIRE(
      gr::contact_graph({3, 2, false, false, gr::Connectivity::Eight}, wrap_diagonal)
          .edges.empty());
  REQUIRE(
      gr::contact_graph({3, 2, true, false, gr::Connectivity::Eight}, wrap_diagonal)
          .edges == std::vector<gr::Contact>{{1, 2}});
}

TEST_CASE("grain grid rejects dimensions and label storage errors",
          "[grain][contract][unit]") {
  REQUIRE_THROWS_AS(gr::cell_count({0, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(gr::cell_count({1, 0}), std::invalid_argument);
  REQUIRE_THROWS_AS(gr::cell_count({std::numeric_limits<std::size_t>::max(), 2}),
                    std::overflow_error);
  const std::vector<gr::Id> labels{1, 2};
  REQUIRE_THROWS_AS(gr::contact_graph({2, 2}, labels), std::invalid_argument);
  REQUIRE(gr::contact_graph({2, 1}, std::vector<gr::Id>{0, 0}).vertices.empty());
}

TEST_CASE("separate grain components may share a slot without losing identity",
          "[grain][contract][unit]") {
  const std::vector<gr::Id> labels{41, 0, 77};
  gr::Snapshot observation{
      5, {{41, 0, true}, {77, 0, true}}, gr::contact_graph({3, 1}, labels)};
  REQUIRE_NOTHROW(gr::validate(observation, 1));
  REQUIRE(observation.graph.vertices == std::vector<gr::Id>{41, 77});
  // Conflicts are valid topology observations; a resolver must detect them.
  observation.graph = gr::make_contact_graph({41, 77}, {{41, 77}});
  REQUIRE_NOTHROW(gr::validate(observation, 1));
  REQUIRE_THROWS_AS(gr::validate(observation, 0), std::invalid_argument);
  observation.grains[0].slot = gr::unassigned;
  REQUIRE_THROWS_AS(gr::validate(observation, 1), std::invalid_argument);
}

TEST_CASE("annihilation is a stable tombstone and explicit topology delta",
          "[grain][contract][unit]") {
  gr::Snapshot before{9,
                      {{1, 0, true}, {2, 1, true}, {3, 0, true}},
                      gr::make_contact_graph({1, 2, 3}, {{1, 2}, {2, 3}})};
  const auto original = before;
  gr::Snapshot after = before;
  ++after.epoch;
  after.grains[1] = gr::retire(after.grains[1]);
  after.graph = gr::make_contact_graph({1, 3}, {{1, 3}});
  REQUIRE_NOTHROW(gr::validate(after, 2));
  const auto change = gr::diff_topology(before.graph, after.graph);
  REQUIRE(change.added.empty());
  REQUIRE(change.removed == std::vector<gr::Id>{2});
  REQUIRE(change.added_contacts == std::vector<gr::Contact>{{1, 3}});
  REQUIRE(change.removed_contacts == std::vector<gr::Contact>{{1, 2}, {2, 3}});
  REQUIRE(change.changed == std::vector<gr::Id>{1, 2, 3});
  REQUIRE(before.graph.edges == original.graph.edges);
  REQUIRE(before.grains[1].active);
  const auto stable = gr::diff_topology(after.graph, after.graph);
  REQUIRE(stable.changed.empty());
  after.grains[1].slot = 0;
  REQUIRE_THROWS_AS(gr::validate(after, 2), std::invalid_argument);
  after.grains[1] = gr::retire(after.grains[1]);
  after.graph.vertices.push_back(99);
  REQUIRE_THROWS_AS(gr::validate(after, 2), std::invalid_argument);
}

TEST_CASE("grain graph stores pathological degree without fixed-capacity loss",
          "[grain][contract][unit]") {
  std::vector<gr::Id> vertices{1};
  std::vector<gr::Contact> edges;
  for (gr::Id id = 2; id <= 257; ++id) {
    vertices.push_back(id);
    edges.push_back({id, 1});
  }
  const auto graph = gr::make_contact_graph(vertices, edges);
  REQUIRE(graph.edges.size() == 256);
  REQUIRE(graph.edges.front() == gr::Contact{1, 2});
  REQUIRE(graph.edges.back() == gr::Contact{1, 257});
  REQUIRE_NOTHROW(gr::validate(graph));
}

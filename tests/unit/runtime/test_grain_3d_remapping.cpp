// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "fixtures/grain_3d_cases.hpp"
#include <array>
#include <catch2/catch_test_macros.hpp>
using namespace grain_3d_test;

TEST_CASE("3D complete support cycles preserve every value and reject atomically",
          "[grain][3d]") {
  Grid3D g{2, 2, 2};
  const auto n = cell_count(g);
  std::vector<double> q(3 * n);
  std::vector<Id> labels(3 * n);
  std::vector<Grain> grains{{11, 0, true}, {22, 1, true}, {33, 2, true}};
  for (std::size_t slot = 0; slot < 3; ++slot)
    for (std::size_t cell = 0; cell < n; ++cell) {
      q[slot * n + cell] = 0.125 * (slot + 1) + 0.001 * cell;
      labels[slot * n + cell] = grains[slot].id;
    }
  const auto original = q;
  const auto original_labels = labels;
  std::vector<Transfer> moves{{11, 0, 1}, {22, 1, 2}, {33, 2, 0}};
  auto result = pfc::grain::transfer(g, 3, q, labels, grains, moves);
  REQUIRE(result.status == TransferStatus::Success);
  for (std::size_t slot = 0; slot < 3; ++slot)
    for (std::size_t cell = 0; cell < n; ++cell) {
      REQUIRE(result.values[((slot + 1) % 3) * n + cell] == q[slot * n + cell]);
      REQUIRE(result.labels[((slot + 1) % 3) * n + cell] == labels[slot * n + cell]);
    }
  REQUIRE(q == original);
  REQUIRE(labels == original_labels);
  REQUIRE(pfc::grain::transfer(g, 3, q, labels, grains, {}).values == q);
  moves = {{11, 0, 1}};
  auto rejected = pfc::grain::transfer(g, 3, q, labels, grains, moves);
  REQUIRE(rejected.status == TransferStatus::OccupiedDestination);
  REQUIRE(rejected.values.empty());
  REQUIRE(q == original);
  labels[0] = 0;
  rejected = pfc::grain::transfer(g, 3, q, labels, grains, {});
  REQUIRE(rejected.status == TransferStatus::InvalidSupport);
  REQUIRE(rejected.labels.empty());
}
TEST_CASE("3D remapping preserves full support and reports every expected failure",
          "[grain][3d]") {
  using S = remapping::Status;
  const std::vector<S> expected{
      S::Success,     S::Success,        S::Success,          S::Success,
      S::Success,     S::Deferred,       S::CapacityOverflow, S::IterationLimit,
      S::Unseeded,    S::MissingSupport, S::UnsafeCadence,    S::Infeasible,
      S::SearchLimit, S::InvalidInput,   S::UnknownIdentity,  S::InvalidInput,
      S::Success,     S::Success,        S::UnsafeCadence};
  auto fixtures = cases();
  REQUIRE(fixtures.size() == expected.size());
  for (std::size_t i = 0; i < fixtures.size(); ++i)
    for (bool observed : {false, true}) {
      auto &c = fixtures[i];
      const auto q = c.values;
      const auto seeds = c.seeds;
      Diagnostics diagnostic;
      c.options.diagnostics = observed ? &diagnostic : nullptr;
      auto actual = run(c);
      INFO("fixture " << i << " observed " << observed);
      if (observed) {
        REQUIRE(diagnostic.source_accesses_complete);
        REQUIRE_FALSE(diagnostic.observation_failed);
      }
      REQUIRE(diagnostic.allocations(diagnostics::Space::Device).requests == 0);
      REQUIRE(actual.status == expected[i]);
      REQUIRE(c.seeds == seeds);
      for (std::size_t j = 0; j < q.size(); ++j)
        REQUIRE((c.values[j] == q[j] || std::isnan(q[j])));
      if (actual.status != S::Success) {
        REQUIRE(actual.values.empty());
        REQUIRE(actual.labels.empty());
        continue;
      }
      REQUIRE(actual.snapshot.epoch == 43);
      validate(actual.snapshot, c.slots);
      std::vector<std::uint8_t> occupied(q.size());
      for (std::size_t j = 0; j < q.size(); ++j) occupied[j] = q[j] > 0;
      const auto owners = ownership(c.grid, c.slots, occupied, c.seeds);
      const auto n = cell_count(c.grid);
      for (std::size_t j = 0; j < q.size(); ++j)
        if (owners[j]) {
          auto grain = std::find_if(actual.snapshot.grains.begin(),
                                    actual.snapshot.grains.end(),
                                    [&](auto g) { return g.id == owners[j]; });
          REQUIRE(grain != actual.snapshot.grains.end());
          REQUIRE(actual.values[grain->slot * n + j % n] == q[j]);
          REQUIRE(actual.labels[grain->slot * n + j % n] == owners[j]);
        }
      auto graph_now =
          graph(c.grid, c.slots, actual.labels, c.options.contact_radius);
      REQUIRE(actual.snapshot.graph.edges == graph_now.edges);
      for (auto edge : graph_now.edges) {
        auto a = std::find_if(actual.snapshot.grains.begin(),
                              actual.snapshot.grains.end(),
                              [&](auto g) { return g.id == edge.first; });
        auto b = std::find_if(actual.snapshot.grains.begin(),
                              actual.snapshot.grains.end(),
                              [&](auto g) { return g.id == edge.second; });
        REQUIRE(a->slot != b->slot);
      }
    }
}

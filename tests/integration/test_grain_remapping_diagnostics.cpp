// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/grain/transfer.hpp>
using namespace pfc::grain;
namespace dg = pfc::grain::diagnostics;
#include "grain_remapping_cases.hpp"
TEST_CASE("all remapping policies observe identical CPU transactions",
          "[grain][diagnostics]") {
  for (auto method :
       {remapping::Method::Incremental, remapping::Method::GlobalSaturation,
        remapping::Method::GlobalLargestFirst, remapping::Method::CompleteOracle})
    for (bool componentwise : {false, true}) {
      auto fixture = remapping_test::pair();
      fixture.options.method = method;
      fixture.options.componentwise = componentwise;
      auto plain = remapping_test::run(fixture);
      Diagnostics d;
      d.host_observer_connected = true;
      fixture.options.diagnostics = &d;
      auto observed = remapping_test::run(fixture);
      REQUIRE(observed.status == plain.status);
      REQUIRE(observed.values == plain.values);
      REQUIRE(observed.labels == plain.labels);
      REQUIRE(observed.snapshot.graph.edges == plain.snapshot.graph.edges);
      REQUIRE(d.source_accesses_complete);
      REQUIRE_FALSE(d.observation_failed);
      REQUIRE(d.allocations(dg::Space::Host).requests > 0);
      REQUIRE(d.allocations(dg::Space::Device).requests == 0);
      for (auto event : d.events) REQUIRE_FALSE(event.available);
      const auto values = static_cast<std::size_t>(dg::Field::Values);
      REQUIRE(d.accesses.reads[values] > 0);
      fixture.options.diagnostics = nullptr;
      {
        dg::Scope outer(&d);
        d.reset_interval();
        auto off = remapping_test::run(fixture);
        dg::PauseObservation pause;
        REQUIRE(off.labels == plain.labels);
      }
      REQUIRE(d.allocations(dg::Space::Host).requests == 0);
      REQUIRE(d.accesses.reads[values] == 0);
    }
}

TEST_CASE("standalone nested producer exception keeps diagnostics incomplete",
          "[grain][diagnostics]") {
  Diagnostics d;
  bool threw = false;
  {
    dg::Scope scope(&d);
    try {
      tracking::reference::propagate({1, 1, false, false, Connectivity::Eight}, 1,
                                     {}, {});
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    auto accepted = transfer({1, 1, false, false, Connectivity::Eight}, 1,
                             std::vector<double>{.5}, std::vector<Id>{7},
                             std::vector<Grain>{{7, 0, true}}, {});
  }
  REQUIRE(threw);
  REQUIRE(d.observation_failed);
  REQUIRE_FALSE(d.source_accesses_complete);
  d.reset_interval();
  REQUIRE_FALSE(d.observation_failed);
}

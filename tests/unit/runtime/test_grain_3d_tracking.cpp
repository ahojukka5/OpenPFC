// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "fixtures/grain_3d_geometry.hpp"
#include <array>
#include <catch2/catch_test_macros.hpp>
using namespace grain_3d_test;

TEST_CASE("3D contacts match every ternary tiny raster and periodic axis",
          "[grain][3d]") {
  for (auto stencil : {Connectivity::Six, Connectivity::TwentySix})
    for (unsigned periodic = 0; periodic < 8; ++periodic) {
      Grid3D g{
          2,      2, 2, bool(periodic & 1), bool(periodic & 2), bool(periodic & 4),
          stencil};
      for (unsigned code = 0; code < 6561; ++code) {
        auto digits = code;
        std::vector<Id> labels(8);
        for (auto &id : labels) {
          id = digits % 3;
          digits /= 3;
        }
        const auto expected = graph(g, 1, labels, 1);
        const auto actual = pfc::grain::contact_graph(g, labels);
        REQUIRE(actual.vertices == expected.vertices);
        REQUIRE(actual.edges == expected.edges);
      }
    }
  REQUIRE_THROWS_AS(cell_count(Grid3D{1, 1, 0}), std::invalid_argument);
  REQUIRE_THROWS_AS(
      cell_count(Grid3D{std::numeric_limits<std::size_t>::max(), 2, 2}),
      std::overflow_error);
  REQUIRE_THROWS_AS(pfc::grain::contact_graph(
                        Grid3D{1, 1, 1, false, false, false, Connectivity::Eight},
                        std::vector<Id>{1}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::grain::contact_graph(Grid2D{1, 1, false, false, Connectivity::Six},
                                std::vector<Id>{1}),
      std::invalid_argument);
  REQUIRE(pfc::grain::contact_graph({1, 1}, std::vector<Id>{1}).edges.empty());
}
TEST_CASE("3D face edge corner seams and radius are explicit", "[grain][3d]") {
  for (auto stencil : {Connectivity::Six, Connectivity::TwentySix})
    for (unsigned periodic = 0; periodic < 8; ++periodic) {
      Grid3D g{
          5,      5, 5, bool(periodic & 1), bool(periodic & 2), bool(periodic & 4),
          stencil};
      std::vector<Id> labels(250);
      labels[0] = 1;
      for (auto xyz :
           {std::array<std::size_t, 3>{4, 0, 0}, {4, 4, 0}, {4, 4, 4}, {2, 2, 2}}) {
        std::fill(labels.begin() + 125, labels.end(), 0);
        labels[125 + index(g, xyz[0], xyz[1], xyz[2])] = 2;
        for (std::size_t radius = 0; radius <= 3; ++radius) {
          const auto expected = graph(g, 2, labels, radius);
          const auto actual =
              tracking::reference::contact_graph(g, 2, labels, radius);
          REQUIRE(actual.edges == expected.edges);
        }
      }
    }
  Grid3D g{5, 5, 5, true, true, true, Connectivity::Six};
  std::vector<Id> labels(125);
  labels[0] = 1;
  labels[index(g, 4, 4, 4)] = 2;
  REQUIRE(tracking::reference::contact_graph(g, 1, labels, 1).edges.empty());
  REQUIRE(tracking::reference::contact_graph(g, 1, labels, 3).edges ==
          std::vector<Contact>{{1, 2}});
  g.connectivity = Connectivity::TwentySix;
  REQUIRE(tracking::reference::contact_graph(g, 1, labels, 1).edges ==
          std::vector<Contact>{{1, 2}});
}
TEST_CASE("3D ownership matches independent seed BFS on every tiny occupancy",
          "[grain][3d]") {
  for (auto stencil : {Connectivity::Six, Connectivity::TwentySix}) {
    Grid3D g{2, 2, 2, true, true, true, stencil};
    for (unsigned mask = 0; mask < 256; ++mask) {
      std::vector<std::uint8_t> occupied(8);
      std::vector<Id> seeds(8);
      for (unsigned i = 0; i < 8; ++i) occupied[i] = (mask >> i) & 1;
      if (occupied[0]) seeds[0] = 9;
      if (occupied[7]) seeds[7] = 3;
      auto expected = ownership(g, 1, occupied, seeds);
      auto actual = tracking::reference::propagate(g, 1, occupied, seeds);
      REQUIRE(actual.labels == expected);
      bool complete = true;
      for (unsigned i = 0; i < 8; ++i)
        if (occupied[i] && !expected[i]) complete = false;
      REQUIRE(actual.complete == complete);
    }
  }
  Grid3D g{3, 1, 1, false, false, false, Connectivity::Six};
  const auto tied = tracking::reference::propagate(
      g, 1, std::vector<std::uint8_t>{1, 1, 1}, std::vector<Id>{9, 0, 3});
  REQUIRE(tied.labels == std::vector<Id>{9, 3, 3});
  const auto disconnected = tracking::reference::propagate(
      g, 1, std::vector<std::uint8_t>{1, 0, 1}, std::vector<Id>{9, 0, 0});
  REQUIRE_FALSE(disconnected.complete);
  REQUIRE(disconnected.labels == std::vector<Id>{9, 0, 0});
}

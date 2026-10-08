// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "grain_transfer_cases.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <limits>

using namespace transfer_test;

namespace {
void require_unchanged(const Case &before, const Case &after) {
  REQUIRE(before.labels == after.labels);
  REQUIRE(before.values.size() == after.values.size());
  REQUIRE(std::memcmp(before.values.data(), after.values.data(),
                      before.values.size() * sizeof(double)) == 0);
  REQUIRE(before.grains.size() == after.grains.size());
  for (std::size_t i = 0; i < before.grains.size(); ++i) {
    REQUIRE(before.grains[i].id == after.grains[i].id);
    REQUIRE(before.grains[i].slot == after.grains[i].slot);
    REQUIRE(before.grains[i].active == after.grains[i].active);
  }
}

void require_conserved(const Case &before, const TransferResult<> &after) {
  const auto cells = cell_count(before.grid);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    std::vector<double> original, staged;
    double old_occupancy = 0, new_occupancy = 0;
    for (Slot slot = 0; slot < before.slots; ++slot) {
      original.push_back(before.values[cells * slot + cell]);
      staged.push_back(after.values[cells * slot + cell]);
      old_occupancy += original.back() - before.background_value;
      new_occupancy += staged.back() - before.background_value;
    }
    std::sort(original.begin(), original.end());
    std::sort(staged.begin(), staged.end());
    REQUIRE(original == staged);
    // Binary-exact test amplitudes make rounding irrelevant for this check.
    REQUIRE(old_occupancy == new_occupancy);
  }
  REQUIRE(after.grains.size() == before.grains.size());
  for (const auto &grain : before.grains) {
    const auto record =
        std::find_if(after.grains.begin(), after.grains.end(),
                     [&](const auto &g) { return g.id == grain.id; });
    REQUIRE(record != after.grains.end());
    REQUIRE(record->active == grain.active);
    if (!grain.active) {
      REQUIRE(record->slot == unassigned);
      continue;
    }
    for (std::size_t cell = 0; cell < cells; ++cell)
      if (before.labels[cells * grain.slot + cell] == grain.id) {
        const auto staged = cells * record->slot + cell;
        REQUIRE(after.labels[staged] == grain.id);
        REQUIRE(after.values[staged] == before.values[cells * grain.slot + cell]);
      }
  }
}

void require_rejected(const Case &fixture, TransferStatus expected) {
  const auto before = fixture;
  const auto result = run(fixture);
  REQUIRE(result.status == expected);
  REQUIRE(result.values.empty());
  REQUIRE(result.labels.empty());
  REQUIRE(result.grains.empty());
  require_unchanged(before, fixture);
}
} // namespace

TEST_CASE("grain transfer preserves shared fields and noncontiguous seam support",
          "[runtime][grain][transfer]") {
  const auto fixture = shared_destination();
  const auto before = fixture;
  const auto result = run(fixture);
  REQUIRE(result.status == TransferStatus::Success);
  require_conserved(fixture, result);
  require_unchanged(before, fixture);
  for (const auto cell : {2u, 6u}) {
    REQUIRE(result.values[8 + cell] == fixture.values[8 + cell]);
    REQUIRE(result.labels[8 + cell] == 22);
  }
  REQUIRE(std::equal(fixture.values.begin() + 16, fixture.values.end(),
                     result.values.begin() + 16));
  REQUIRE(std::equal(fixture.labels.begin() + 16, fixture.labels.end(),
                     result.labels.begin() + 16));
  for (const auto cell : {0u, 3u, 7u}) {
    REQUIRE(result.values[cell] == 0);
    REQUIRE(result.labels[cell] == background);
  }
}

TEST_CASE("grain transfer stages complete cycles from original values",
          "[runtime][grain][transfer]") {
  for (const auto &fixture : {cycle(), signed_cycle()}) {
    const auto result = run(fixture);
    REQUIRE(result.status == TransferStatus::Success);
    require_conserved(fixture, result);
    REQUIRE(result.values[0] == fixture.values[16]);
    REQUIRE(result.values[8] == fixture.values[0]);
    REQUIRE(result.values[16] == fixture.values[8]);
  }
}

TEST_CASE("grain transfer makes empty batches and self moves exact no ops",
          "[runtime][grain][transfer]") {
  auto fixture = shared_destination();
  SECTION("empty") { fixture.moves.clear(); }
  SECTION("self") {
    fixture.moves.front().destination = fixture.moves.front().source;
  }
  const auto result = run(fixture);
  REQUIRE(result.status == TransferStatus::Success);
  REQUIRE(result.values == fixture.values);
  REQUIRE(result.labels == fixture.labels);
  for (std::size_t i = 0; i < fixture.grains.size(); ++i)
    REQUIRE(result.grains[i].slot == fixture.grains[i].slot);
}

TEST_CASE("grain transfer rejects unsafe instructions before publication",
          "[runtime][grain][transfer]") {
  auto fixture = shared_destination();
  auto expected = TransferStatus::InvalidTransfer;
  SECTION("duplicate UID") { fixture.moves.push_back(fixture.moves.front()); }
  SECTION("unknown UID") { fixture.moves.front().id = 999; }
  SECTION("inactive UID") { fixture.moves.front().id = 77; }
  SECTION("background UID") { fixture.moves.front().id = background; }
  SECTION("destination out of range") { fixture.moves.front().destination = 3; }
  SECTION("stale source") {
    fixture.moves.front().source = 1;
    expected = TransferStatus::StaleSource;
  }
  SECTION("nonvacating destination") {
    put(fixture, 1, 0, 22, .5);
    expected = TransferStatus::OccupiedDestination;
  }
  SECTION("two final arrivals") {
    fixture = cycle();
    fixture.moves = {{11, 0, 2}, {22, 1, 2}};
    expected = TransferStatus::OccupiedDestination;
  }
  require_rejected(fixture, expected);
}

TEST_CASE("grain transfer rejects partial labels and missing support",
          "[runtime][grain][transfer]") {
  auto fixture = shared_destination();
  auto expected = TransferStatus::InvalidSupport;
  SECTION("unlabeled diffuse tail") { fixture.values[5] = 1.e-12; }
  SECTION("label on background") { fixture.labels[5] = 22; }
  SECTION("unknown field label") { fixture.labels[0] = 999; }
  SECTION("label in stale slot") { put(fixture, 2, 5, 22, .25); }
  SECTION("nonfinite field") {
    fixture.values[0] = std::numeric_limits<double>::infinity();
  }
  SECTION("NaN field") {
    fixture.values[0] = std::numeric_limits<double>::quiet_NaN();
  }
  SECTION("below background") { fixture.values[0] = -.125; }
  SECTION("empty source grain") {
    for (const auto cell : {0u, 3u, 7u}) {
      fixture.values[cell] = 0;
      fixture.labels[cell] = 0;
    }
    expected = TransferStatus::MissingSupport;
  }
  require_rejected(fixture, expected);
}

TEST_CASE("grain transfer validates extents and original registry",
          "[runtime][grain][transfer]") {
  auto fixture = shared_destination();
  auto expected = TransferStatus::InvalidLayout;
  SECTION("value size mismatch") { fixture.values.pop_back(); }
  SECTION("label size mismatch") { fixture.labels.pop_back(); }
  SECTION("zero dimension") { fixture.grid.nx = 0; }
  SECTION("extent overflow") {
    fixture.grid.nx = std::numeric_limits<std::size_t>::max();
    fixture.grid.ny = 2;
  }
  SECTION("invalid background") {
    fixture.background_value = std::numeric_limits<double>::infinity();
  }
  SECTION("unsorted registry") {
    std::reverse(fixture.grains.begin(), fixture.grains.end());
    expected = TransferStatus::InvalidRegistry;
  }
  SECTION("active slot outside layout") {
    fixture.grains.front().slot = 3;
    expected = TransferStatus::InvalidRegistry;
  }
  SECTION("retired grain retained slot") {
    fixture.grains[2].slot = 0;
    expected = TransferStatus::InvalidRegistry;
  }
  require_rejected(fixture, expected);
}

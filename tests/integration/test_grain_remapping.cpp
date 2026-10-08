// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "grain_remapping.hpp"
#include "grain_remapping_cases.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

namespace {
using namespace remapping_test;
void immutable(const Case &fixture, const std::vector<double> &values,
               const std::vector<Id> &seeds) {
  REQUIRE(fixture.values.size() == values.size());
  REQUIRE(std::memcmp(fixture.values.data(), values.data(),
                      values.size() * sizeof(double)) == 0);
  REQUIRE(fixture.seeds == seeds);
}
void empty(const remapping::Result<> &result) {
  REQUIRE(result.values.empty());
  REQUIRE(result.labels.empty());
  REQUIRE(result.snapshot.grains.empty());
  REQUIRE(result.snapshot.graph.vertices.empty());
}
} // namespace

TEST_CASE("2D remapping publishes legal identity fields and complete current graph",
          "[grain][remapping]") {
  for (auto method :
       {remapping::Method::Incremental, remapping::Method::GlobalSaturation,
        remapping::Method::GlobalLargestFirst, remapping::Method::CompleteOracle}) {
    auto fixture = pair();
    fixture.options.method = method;
    const auto before = fixture.values;
    const auto seeds = fixture.seeds;
    auto result = run(fixture);
    REQUIRE(result.status == remapping::Status::Success);
    validate(result.snapshot, fixture.slots);
    REQUIRE(result.snapshot.epoch == 17);
    REQUIRE(result.snapshot.graph.edges == std::vector<Contact>{{11, 42}});
    REQUIRE(result.snapshot.grains[0].slot != result.snapshot.grains[1].slot);
    REQUIRE(result.snapshot.grains[2].id == 80);
    REQUIRE_FALSE(result.snapshot.grains[2].active);
    REQUIRE(result.statistics.changed_grains == 1);
    REQUIRE(result.statistics.moved_samples == 1);
    REQUIRE(result.statistics.moved_value_bytes == sizeof(double));
    REQUIRE(result.statistics.moved_label_bytes == sizeof(Id));
    REQUIRE(result.statistics.conflict);
    const auto cells = cell_count(fixture.grid);
    for (std::size_t i = 0; i < 2; ++i) {
      const auto x = i == 0 ? 2 : 5;
      const auto cell = x + 2 * fixture.grid.nx;
      const auto index = cells * result.snapshot.grains[i].slot + cell;
      REQUIRE(result.labels[index] == fixture.grains[i].id);
      REQUIRE(result.values[index] == (i == 0 ? .75 : .25));
    }
    immutable(fixture, before, seeds);
    auto repeated = run(fixture);
    REQUIRE(repeated.values == result.values);
    REQUIRE(repeated.labels == result.labels);
    fixture.values = result.values;
    fixture.seeds = result.labels;
    fixture.grains = result.snapshot.grains;
    auto noop = run(fixture);
    REQUIRE(noop.status == remapping::Status::Success);
    REQUIRE(noop.values == fixture.values);
    REQUIRE(noop.labels == fixture.seeds);
    REQUIRE(noop.snapshot.graph.edges == result.snapshot.graph.edges);
    REQUIRE(noop.statistics.changed_grains == 0);
    REQUIRE_FALSE(noop.statistics.conflict);
    REQUIRE(noop.statistics.staged_storage_bytes ==
            fixture.values.size() * (sizeof(double) + sizeof(Id)));
  }
}

TEST_CASE("global label alignment weights actual unequal support and wide UIDs",
          "[grain][remapping]") {
  for (bool higher_heavier : {false, true}) {
    auto fixture = pair();
    constexpr Id wide = Id{1} << 40;
    fixture.grains = {{11, 0, true},
                      {81, 2, true},
                      {wide, 0, true},
                      {std::numeric_limits<Id>::max(), unassigned, false}};
    fixture.seeds[37] = wide;
    if (higher_heavier) {
      put(fixture, 0, 5, 1, wide, .25);
      put(fixture, 0, 5, 3, wide, .25);
    } else
      put(fixture, 0, 1, 2, 11, .25);
    put(fixture, 2, 11, 2, 81, .125);
    fixture.options.method = remapping::Method::GlobalSaturation;
    auto result = run(fixture);
    REQUIRE(result.status == remapping::Status::Success);
    REQUIRE(result.snapshot.graph.edges == std::vector<Contact>{{11, wide}});
    REQUIRE((result.snapshot.grains[0].slot == 0) == !higher_heavier);
    REQUIRE(result.snapshot.grains[1].slot == 2);
    REQUIRE((result.snapshot.grains[2].slot == 0) == higher_heavier);
    REQUIRE(result.values[203] == .125);
    REQUIRE(result.labels[203] == 81);
    REQUIRE(result.statistics.moved_samples == 1);
    REQUIRE(result.snapshot.grains.back().id == std::numeric_limits<Id>::max());
  }
}

TEST_CASE("2D remapping distinguishes impossible bounded deferred and unsafe calls",
          "[grain][remapping]") {
  auto fixture = pair();
  SECTION("proven impossible K5") {
    fixture = impossible();
    auto result = run(fixture);
    REQUIRE(result.statistics.edges == 10);
    REQUIRE(result.status == remapping::Status::Infeasible);
    empty(result);
  }
  SECTION("every solver honors zero attempt budget") {
    for (auto method :
         {remapping::Method::Incremental, remapping::Method::GlobalSaturation,
          remapping::Method::GlobalLargestFirst,
          remapping::Method::CompleteOracle}) {
      fixture.options.method = method;
      fixture.options.limits.attempts = 0;
      auto result = run(fixture);
      REQUIRE(result.status == remapping::Status::SearchLimit);
      empty(result);
    }
  }
  SECTION("global components share one invocation budget") {
    put(fixture, 0, 10, 2, 0, 0);
    put(fixture, 0, 13, 2, 0, 0);
    fixture.grains.insert(fixture.grains.begin() + 2,
                          {{51, 0, true}, {61, 0, true}});
    put(fixture, 0, 10, 2, 51, .4);
    put(fixture, 0, 13, 2, 61, .6);
    fixture.options.method = remapping::Method::GlobalSaturation;
    fixture.options.limits.attempts = 3;
    auto result = run(fixture);
    REQUIRE(result.status == remapping::Status::SearchLimit);
    REQUIRE(result.statistics.attempts == 3);
    empty(result);
  }
  SECTION("complete known labels still refuse already touching supports") {
    put(fixture, 0, 5, 2, 0, 0);
    put(fixture, 0, 3, 2, 42, .25);
    auto result = run(fixture);
    REQUIRE(result.status == remapping::Status::UnsafeCadence);
    empty(result);
  }
  SECTION("skip is explicitly unverified") {
    fixture.options.check_now = false;
    auto result = run(fixture);
    REQUIRE(result.status == remapping::Status::Deferred);
    empty(result);
  }
  SECTION("overflow cannot publish truncated graph") {
    fixture.options.contact_capacity = 0;
    auto result = run(fixture);
    REQUIRE(result.status == remapping::Status::CapacityOverflow);
    REQUIRE(result.statistics.edges == 1);
    empty(result);
  }
}

TEST_CASE("2D remapping validates registry seeds full support and retirement",
          "[grain][remapping]") {
  auto fixture = pair();
  remapping::Status expected = remapping::Status::InvalidInput;
  SECTION("unknown positive seed") {
    fixture.seeds[34] = 999;
    expected = remapping::Status::UnknownIdentity;
  }
  SECTION("unknown beats earlier nonfinite input deterministically") {
    fixture.values[0] = std::numeric_limits<double>::quiet_NaN();
    fixture.seeds[34] = 999;
    expected = remapping::Status::UnknownIdentity;
  }
  SECTION("negative field") { fixture.values[0] = -.1; }
  SECTION("wrong seed slot") { fixture.grains[0].slot = 1; }
  SECTION("invalid registry") { fixture.grains[0].id = 42; }
  SECTION("invalid layout") { fixture.seeds.pop_back(); }
  SECTION("new unseeded positive component") {
    fixture.values[0] = 1.e-100;
    expected = remapping::Status::Unseeded;
  }
  SECTION("active disappearance is not silently retired") {
    put(fixture, 0, 5, 2, 0, 0);
    expected = remapping::Status::MissingSupport;
  }
  SECTION("explicit retirement preserves tombstone UID") {
    put(fixture, 0, 5, 2, 0, 0);
    fixture.grains[1] = retire(fixture.grains[1]);
    auto result = run(fixture);
    REQUIRE(result.status == remapping::Status::Success);
    REQUIRE(result.snapshot.grains[1].id == 42);
    REQUIRE_FALSE(result.snapshot.grains[1].active);
    REQUIRE(result.snapshot.grains[1].slot == unassigned);
    return;
  }
  SECTION("propagation iteration budget is a distinct failure") {
    fixture.options.max_sweeps = 0;
    expected = remapping::Status::IterationLimit;
  }
  SECTION("partial seeds honor the exact multi-sweep horizon") {
    put(fixture, 0, 1, 2, 0, .5);
    put(fixture, 0, 0, 2, 0, .5);
    fixture.options.max_sweeps = 1;
    auto limited = run(fixture);
    REQUIRE(limited.status == remapping::Status::IterationLimit);
    empty(limited);
    fixture.options.max_sweeps = 2;
    auto complete = run(fixture);
    REQUIRE(complete.status == remapping::Status::Success);
    REQUIRE(complete.statistics.propagation_sweeps == 2);
    return;
  }
  SECTION("empty active set retains explicit tombstones") {
    fixture.values.assign(fixture.values.size(), 0);
    for (auto &grain : fixture.grains) grain = retire(grain);
    auto result = run(fixture);
    REQUIRE(result.status == remapping::Status::Success);
    REQUIRE(result.snapshot.graph.vertices.empty());
    REQUIRE(result.snapshot.graph.edges.empty());
    REQUIRE(result.snapshot.grains.size() == 3);
    REQUIRE(std::all_of(result.labels.begin(), result.labels.end(),
                        [](Id id) { return id == background; }));
    return;
  }
  const auto before = fixture.values;
  const auto seeds = fixture.seeds;
  auto result = run(fixture);
  REQUIRE(result.status == expected);
  empty(result);
  immutable(fixture, before, seeds);
  REQUIRE(result.statistics.total_seconds >= result.statistics.detection_seconds);
}

TEST_CASE(
    "buffered remapping preserves per-UID Allen-Cahn-style collision evolution",
    "[grain][remapping][evolution]") {
  for (bool seam : {false, true}) {
    for (bool one : {false, true}) {
      auto state = grain_example::initial(seam, one);
      const auto cells = cell_count(state.grid);
      std::vector<double> reference(cells * state.grains.size(), 0);
      for (std::size_t i = 0; i < state.labels.size(); ++i)
        if (state.labels[i]) {
          const auto index = state.labels[i] == 11 ? 0 : 1;
          reference[index * cells + i % cells] = state.values[i];
        }
      for (int step = 0; step < 6; ++step) {
        reference = grain_example::advance_values(
            state.grid, static_cast<Slot>(state.grains.size()), reference);
        REQUIRE(grain_example::advance(state) == remapping::Status::Success);
        for (std::size_t g = 0; g < state.grains.size(); ++g) {
          double expected_area = 0, actual_area = 0;
          for (std::size_t cell = 0; cell < cells; ++cell) {
            const auto expected = reference[g * cells + cell];
            const auto index = state.grains[g].slot * cells + cell;
            if (expected > 0) {
              REQUIRE(state.labels[index] == state.grains[g].id);
              REQUIRE(std::abs(state.values[index] - expected) <= 1.e-13);
              actual_area += state.values[index];
            } else
              REQUIRE(state.labels[index] != state.grains[g].id);
            expected_area += expected;
          }
          REQUIRE(std::abs(actual_area - expected_area) <= 1.e-12);
        }
      }
      if (!one)
        REQUIRE(state.grains[0].slot != state.grains[1].slot);
      else
        REQUIRE(state.grains[0].slot == 0);
    }
  }
}

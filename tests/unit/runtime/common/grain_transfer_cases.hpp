// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <openpfc/kernel/grain/transfer.hpp>

namespace transfer_test {
using namespace pfc::grain;

struct Case {
  Grid2D grid{4, 2, true, true, Connectivity::Eight};
  Slot slots = 3;
  double background_value = 0;
  std::vector<double> values = std::vector<double>(24, 0);
  std::vector<Id> labels = std::vector<Id>(24, 0);
  std::vector<Grain> grains;
  std::vector<Transfer> moves;
};

inline void put(Case &fixture, Slot slot, std::size_t cell, Id id, double value) {
  const auto i = slot * cell_count(fixture.grid) + cell;
  fixture.values[i] = value;
  fixture.labels[i] = id;
}

inline Case shared_destination() {
  Case fixture;
  // UID exceeds both 24-bit legacy packing and a 32-bit identity range.
  constexpr Id a = (Id{1} << 40) + 11;
  fixture.grains = {
      {22, 1, true}, {33, 2, true}, {77, unassigned, false}, {a, 0, true}};
  fixture.moves = {{a, 0, 1}};
  // Noncontiguous support crosses both periodic edges of the 4x2 grid.
  put(fixture, 0, 0, a, .25);
  put(fixture, 0, 3, a, .5);
  put(fixture, 0, 7, a, .125);
  put(fixture, 1, 2, 22, .75);
  put(fixture, 1, 6, 22, .5);
  put(fixture, 2, 1, 33, .125);
  return fixture;
}

inline Case cycle() {
  Case fixture;
  fixture.grains = {{11, 0, true}, {22, 1, true}, {33, 2, true}};
  fixture.moves = {{33, 2, 0}, {11, 0, 1}, {22, 1, 2}};
  // Three grain supports overlap across distinct fields and rotate safely.
  put(fixture, 0, 0, 11, .125);
  put(fixture, 1, 0, 22, .25);
  put(fixture, 2, 0, 33, .5);
  put(fixture, 0, 7, 11, .75);
  put(fixture, 2, 3, 33, .25);
  return fixture;
}

inline Case signed_cycle() {
  auto fixture = cycle();
  fixture.background_value = -1;
  for (auto &value : fixture.values) value = 2 * value - 1;
  return fixture;
}

inline TransferResult<> run(const Case &fixture) {
  return transfer(fixture.grid, fixture.slots, fixture.values, fixture.labels,
                  fixture.grains, fixture.moves, fixture.background_value);
}

} // namespace transfer_test

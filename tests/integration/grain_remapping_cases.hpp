// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/runtime/cpu/grain_remapping.hpp>

namespace remapping_test {
using namespace pfc::grain;
struct Case {
  Grid2D grid{16, 5, true, true, Connectivity::Eight};
  Slot slots = 3;
  std::vector<double> values = std::vector<double>(240, 0);
  std::vector<Id> seeds = std::vector<Id>(240, 0);
  std::vector<Grain> grains{{11, 0, true}, {42, 0, true}, {80, unassigned, false}};
  remapping::Options options;
};
inline void put(Case &fixture, Slot slot, std::size_t x, std::size_t y, Id id,
                double q) {
  const auto i = cell_count(fixture.grid) * slot + x + fixture.grid.nx * y;
  fixture.values[i] = q;
  fixture.seeds[i] = id;
}
inline Case pair() {
  Case fixture;
  fixture.options.epoch = 17;
  put(fixture, 0, 2, 2, 11, .75);
  put(fixture, 0, 5, 2, 42, .25);
  return fixture;
}
inline Case impossible() {
  Case fixture;
  fixture.grid = {8, 8, false, false, Connectivity::Eight};
  fixture.slots = 4;
  fixture.values.assign(256, 0);
  fixture.seeds.assign(256, 0);
  fixture.grains = {
      {1, 0, true}, {2, 1, true}, {3, 0, true}, {4, 2, true}, {5, 3, true}};
  put(fixture, 0, 2, 2, 1, .5);
  put(fixture, 1, 3, 2, 2, .5);
  put(fixture, 0, 4, 2, 3, .5);
  put(fixture, 2, 2, 3, 4, .5);
  put(fixture, 3, 3, 3, 5, .5);
  fixture.options.method = remapping::Method::CompleteOracle;
  return fixture;
}
inline remapping::Result<> run(const Case &fixture) {
  return remapping::remap(fixture.grid, fixture.slots, fixture.values, fixture.seeds,
                          fixture.grains, fixture.options);
}
} // namespace remapping_test

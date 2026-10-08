// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "grain_3d_geometry.hpp"
#include <openpfc/runtime/cpu/grain_remapping.hpp>
namespace grain_3d_test {
struct Case {
  Grid3D grid{7, 5, 5, true, true, true, Connectivity::TwentySix};
  Slot slots = 2;
  std::vector<double> values;
  std::vector<Id> seeds;
  std::vector<Grain> grains{{11, 0, true}, {22, 0, true}};
  remapping::Options options{};
};
inline Case conflict() {
  Case c;
  const auto cells = cell_count(c.grid);
  c.values.resize(cells * c.slots);
  c.seeds.resize(cells * c.slots);
  for (std::size_t z = 1; z <= 3; ++z) c.values[index(c.grid, 1, 2, z)] = 0.1 * z;
  c.seeds[index(c.grid, 1, 2, 2)] = 11;
  c.values[index(c.grid, 4, 2, 2)] = 0.75;
  c.seeds[index(c.grid, 4, 2, 2)] = 22;
  c.options.epoch = 43;
  return c;
}
inline remapping::Result<> run(const Case &c) {
  return remapping::remap(c.grid, c.slots, c.values, c.seeds, c.grains, c.options);
}
inline std::vector<Case> cases() {
  std::vector<Case> out;
  auto add = [&](Case c) { out.push_back(std::move(c)); };
  for (auto method :
       {remapping::Method::Incremental, remapping::Method::GlobalSaturation,
        remapping::Method::GlobalLargestFirst, remapping::Method::CompleteOracle}) {
    auto c = conflict();
    c.options.method = method;
    add(c);
  }
  auto c = conflict();
  const auto cells = cell_count(c.grid);
  const auto at = index(c.grid, 4, 2, 2);
  c.values[cells + at] = c.values[at];
  c.values[at] = 0;
  c.seeds[cells + at] = 22;
  c.seeds[at] = 0;
  c.grains[1].slot = 1;
  add(c);
  c = conflict();
  c.options.check_now = false;
  add(c);
  c = conflict();
  c.options.contact_capacity = 0;
  add(c);
  c = conflict();
  c.options.max_sweeps = 0;
  add(c);
  c = conflict();
  c.seeds[at] = 0;
  add(c);
  c = conflict();
  c.grains.push_back({33, 1, true});
  add(c);
  c = conflict();
  c.values[at] = 0;
  c.seeds[at] = 0;
  c.values[index(c.grid, 2, 2, 2)] = 0.75;
  c.seeds[index(c.grid, 2, 2, 2)] = 22;
  add(c);
  c = conflict();
  c.slots = 1;
  c.options.method = remapping::Method::CompleteOracle;
  c.values.resize(cells);
  c.seeds.resize(cells);
  add(c);
  c = conflict();
  c.options.limits.attempts = 0;
  add(c);
  c = conflict();
  c.values[at] = std::numeric_limits<double>::quiet_NaN();
  add(c);
  c = conflict();
  c.seeds[at] = 99;
  add(c);
  c = conflict();
  c.grid.connectivity = Connectivity::Eight;
  add(c);
  c = conflict();
  c.grid.connectivity = Connectivity::Six;
  add(c);
  // A periodic corner pair is a radius-three Six contact, but is already
  // immediate same-slot contact for TwentySix and must refuse certification.
  c = conflict();
  c.grid = {5, 5, 5, true, true, true, Connectivity::Six};
  c.values.assign(250, 0);
  c.seeds.assign(250, 0);
  c.values[0] = 0.25;
  c.seeds[0] = 11;
  c.values[124] = 0.75;
  c.seeds[124] = 22;
  add(c);
  c.grid.connectivity = Connectivity::TwentySix;
  add(c);
  return out;
}
} // namespace grain_3d_test

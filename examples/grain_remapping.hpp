// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/runtime/cpu/grain_remapping.hpp>

// Small teaching consumer: explicit 2D Allen-Cahn-style reaction/diffusion.
// No configuration frontend, trajectories, benchmarks, or physical validation.
namespace grain_example {
using namespace pfc::grain;
struct State {
  Grid2D grid{16, 5, true, true, Connectivity::Eight};
  Slot slots = 2;
  std::vector<double> values = std::vector<double>(160, 0);
  std::vector<Id> labels = std::vector<Id>(160, 0);
  std::vector<Grain> grains{{11, 0, true}, {42, 0, true}};
  std::uint64_t epoch = 0;
};
inline State initial(bool seam = false, bool one_grain = false) {
  State state;
  state.values[34] = .8;
  state.labels[34] = 11;
  if (one_grain) {
    state.grains.pop_back();
    return state;
  }
  const auto second = (seam ? 12 : 9) + 2 * state.grid.nx;
  state.values[second] = .8;
  state.labels[second] = 42;
  return state;
}

// Independent fields remain separate in the one-field-per-grain oracle.
// All coefficients are identical; changing a numeric slot changes storage only.
inline std::vector<double> advance_values(Grid2D grid, Slot slots,
                                          std::span<const double> values) {
  const auto cells = cell_count(grid);
  std::vector<double> next(values.begin(), values.end());
  for (Slot slot = 0; slot < slots; ++slot)
    for (std::size_t y = 0; y < grid.ny; ++y)
      for (std::size_t x = 0; x < grid.nx; ++x) {
        const auto cell = x + grid.nx * y, index = slot * cells + cell;
        const auto q = values[index];
        const auto left = (x + grid.nx - 1) % grid.nx + grid.nx * y;
        const auto right = (x + 1) % grid.nx + grid.nx * y;
        const auto down = x + grid.nx * ((y + grid.ny - 1) % grid.ny);
        const auto up = x + grid.nx * ((y + 1) % grid.ny);
        const auto lap = values[slot * cells + left] + values[slot * cells + right] +
                         values[slot * cells + down] + values[slot * cells + up] -
                         4 * q;
        double other = 0;
        for (Slot neighbor = 0; neighbor < slots; ++neighbor)
          if (neighbor != slot)
            other +=
                values[neighbor * cells + cell] * values[neighbor * cells + cell];
        next[index] = q + .01 * (.2 * lap - q * (q - 1) * (q - .5) - 2 * q * other);
      }
  return next;
}

// Stage the proposed PDE step as well: post-step failure leaves State unchanged.
inline remapping::Status advance(State &state) {
  remapping::Options options;
  options.epoch = state.epoch + 1;
  auto before = remapping::remap(state.grid, state.slots, state.values, state.labels,
                                 state.grains, options);
  if (before.status != remapping::Status::Success) return before.status;
  auto proposed = advance_values(state.grid, state.slots, before.values);
  options.epoch = state.epoch + 2;
  auto after = remapping::remap(state.grid, state.slots, proposed, before.labels,
                                before.snapshot.grains, options);
  if (after.status != remapping::Status::Success) return after.status;
  State next{state.grid,
             state.slots,
             std::move(after.values),
             std::move(after.labels),
             std::move(after.snapshot.grains),
             after.snapshot.epoch};
  state = std::move(next);
  return remapping::Status::Success;
}
} // namespace grain_example

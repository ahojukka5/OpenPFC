// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_field_modifiers.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/fourier_series.hpp>
#include <openpfc/kernel/field/indexed_noise.hpp>
#include <openpfc/kernel/simulation/initial_conditions/constant.hpp>
#include <openpfc/kernel/simulation/initial_conditions/fourier_modes.hpp>
#include <openpfc/kernel/simulation/initial_conditions/indexed_noise.hpp>
#include <openpfc/kernel/simulation/simulation_context.hpp>

using Catch::Matchers::WithinAbs;
using nlohmann::json;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

int world_rank() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

pfc::Domain grid(int nx, int ny, int nz) {
  return pfc::domain::create(pfc::GridSize({nx, ny, nz}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
}

pfc::Box3i full_box(int nx, int ny, int nz) {
  return pfc::Box3i::from_bounds({0, 0, 0}, {nx - 1, ny - 1, nz - 1});
}

double mean_of(const std::vector<double> &u) {
  double sum = 0.0;
  for (double v : u) sum += v;
  return sum / static_cast<double>(u.size());
}

} // namespace

TEST_CASE("A Fourier mode matches the cosine at grid coordinates",
          "[field][fourier]") {
  const auto domain = grid(8, 4, 1);
  const pfc::field::FourierMode mode{{1, 0, 0}, 2.0, 0.0};
  const double at_origin = pfc::field::fourier_series(
      domain, pfc::domain::to_coords(domain, {0, 0, 0}), std::span{&mode, 1});
  const double at_half = pfc::field::fourier_series(
      domain, pfc::domain::to_coords(domain, {4, 0, 0}), std::span{&mode, 1});
  REQUIRE_THAT(at_origin, WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(at_half, WithinAbs(-2.0, 1e-12));
}

TEST_CASE("An integer mode agrees at x and x + L", "[field][fourier]") {
  const auto domain = grid(8, 8, 8);
  const pfc::field::FourierMode mode{{2, -1, 3}, 0.4, 0.7};
  const pfc::Real3 x{0.3, 1.1, 2.5};
  pfc::Real3 shifted = x;
  const auto size = pfc::domain::get_size(domain);
  const auto spacing = pfc::domain::get_spacing(domain);
  shifted[0] += spacing[0] * static_cast<double>(size[0]);
  shifted[1] += spacing[1] * static_cast<double>(size[1]);
  shifted[2] += spacing[2] * static_cast<double>(size[2]);
  const double a = pfc::field::fourier_series(domain, x, std::span{&mode, 1});
  const double b = pfc::field::fourier_series(domain, shifted, std::span{&mode, 1});
  REQUIRE_THAT(a, WithinAbs(b, 1e-12));
}

TEST_CASE("Phase and superposition are the sum of the terms", "[field][fourier]") {
  const auto domain = grid(16, 8, 4);
  const pfc::field::FourierMode phased{{0, 0, 0}, 1.5, std::numbers::pi / 2.0};
  const pfc::Real3 origin = pfc::domain::to_coords(domain, {0, 0, 0});
  REQUIRE_THAT(pfc::field::fourier_series(domain, origin, std::span{&phased, 1}),
               WithinAbs(0.0, 1e-12));

  const pfc::field::FourierMode modes[] = {
      pfc::field::FourierMode{{1, 0, 0}, 0.2, 0.3},
      pfc::field::FourierMode{{0, 2, 1}, -0.4, -0.5},
  };
  const pfc::Real3 x = pfc::domain::to_coords(domain, {3, 1, 2});
  const double expected =
      pfc::field::fourier_series(domain, x, std::span{&modes[0], 1}) +
      pfc::field::fourier_series(domain, x, std::span{&modes[1], 1});
  REQUIRE_THAT(pfc::field::fourier_series(domain, x, modes),
               WithinAbs(expected, 1e-12));
}

TEST_CASE("Fourier modes add, and a fill writes the offset", "[field][fourier]") {
  const auto domain = grid(8, 8, 1);
  const auto box = full_box(8, 8, 1);
  std::vector<double> u(64, 0.0);
  pfc::Constant constant(0.25);
  constant.apply(u, domain, box, 0.0);
  pfc::FourierModes added;
  added.terms.push_back(pfc::field::FourierMode{{1, 0, 0}, 0.1, 0.0});
  added.apply(u, domain, box, 0.0);
  const pfc::Real3 origin = pfc::domain::to_coords(domain, {0, 0, 0});
  REQUIRE_THAT(u[0], WithinAbs(0.25 + 0.1, 1e-12));
  REQUIRE_THAT(
      u[0], WithinAbs(0.25 + pfc::field::fourier_series(domain, origin, added.terms),
                      1e-12));

  std::vector<double> filled(64, 7.0);
  pfc::FourierSeriesFill fill;
  fill.offset = -0.5;
  fill.terms.push_back(pfc::field::FourierMode{{0, 1, 0}, 0.2, 0.0});
  fill.apply(filled, domain, box, 0.0);
  REQUIRE_THAT(filled[0], WithinAbs(-0.5 + 0.2, 1e-12));
}

TEST_CASE("The same mode on two boxes matches the series", "[field][fourier]") {
  const auto domain = grid(8, 4, 2);
  const pfc::field::FourierMode mode{{1, 2, 1}, 0.3, 0.4};
  const auto left = pfc::Box3i::from_bounds({0, 0, 0}, {3, 3, 1});
  const auto right = pfc::Box3i::from_bounds({4, 0, 0}, {7, 3, 1});
  std::vector<double> a(static_cast<std::size_t>(left.count()), 0.0);
  std::vector<double> b(static_cast<std::size_t>(right.count()), 0.0);
  pfc::field::add_fourier_series(a, domain, left, std::span{&mode, 1});
  pfc::field::add_fourier_series(b, domain, right, std::span{&mode, 1});
  std::size_t at = 0;
  for (int k = left.low[2]; k <= left.high[2]; ++k) {
    for (int j = left.low[1]; j <= left.high[1]; ++j) {
      for (int i = left.low[0]; i <= left.high[0]; ++i) {
        const double expected = pfc::field::fourier_series(
            domain, pfc::domain::to_coords(domain, {i, j, k}), std::span{&mode, 1});
        REQUIRE_THAT(a[at], WithinAbs(expected, 1e-12));
        ++at;
      }
    }
  }
  at = 0;
  for (int k = right.low[2]; k <= right.high[2]; ++k) {
    for (int j = right.low[1]; j <= right.high[1]; ++j) {
      for (int i = right.low[0]; i <= right.high[0]; ++i) {
        const double expected = pfc::field::fourier_series(
            domain, pfc::domain::to_coords(domain, {i, j, k}), std::span{&mode, 1});
        REQUIRE_THAT(b[at], WithinAbs(expected, 1e-12));
        ++at;
      }
    }
  }
}

TEST_CASE("Indexed noise is a pure function of seed and global index",
          "[field][noise]") {
  const auto a = pfc::field::indexed_noise_sample(4, 1, 2, 3, 8, 8);
  REQUIRE(a == pfc::field::indexed_noise_sample(4, 1, 2, 3, 8, 8));
  REQUIRE(a != pfc::field::indexed_noise_sample(5, 1, 2, 3, 8, 8));
  REQUIRE(a <= 65535);
}

TEST_CASE("Filled indexed noise keeps the offset and changes with the seed",
          "[field][noise]") {
  if (world_size() != 1) SKIP("single-rank noise check");
  const auto domain = grid(8, 4, 1);
  const auto box = full_box(8, 4, 1);
  pfc::IndexedNoiseFill fill;
  fill.offset = 0.32;
  fill.noise = pfc::field::IndexedNoise{42, 0.02, true};
  std::vector<double> u(32, 0.0);
  fill.apply(u, domain, box, 0.0);
  REQUIRE_THAT(mean_of(u), WithinAbs(0.32, 1e-12));
  const auto first = u;
  fill.apply(u, domain, box, 0.0);
  REQUIRE(u == first);
  fill.noise.seed = 43;
  fill.apply(u, domain, box, 0.0);
  REQUIRE(u != first);

  fill.noise.amplitude = 0.0;
  fill.noise.seed = 42;
  fill.apply(u, domain, box, 0.0);
  for (double v : u) REQUIRE_THAT(v, WithinAbs(0.32, 1e-15));
}

TEST_CASE("Noise without mean removal is the scaled sample", "[field][noise]") {
  if (world_size() != 1) SKIP("single-rank noise check");
  const auto domain = grid(4, 2, 1);
  const auto box = full_box(4, 2, 1);
  pfc::IndexedNoiseFill fill;
  fill.offset = 1.0;
  fill.noise = pfc::field::IndexedNoise{9, 0.5, false};
  std::vector<double> u(8, 0.0);
  fill.apply(u, domain, box, 0.0);
  const auto sample = pfc::field::indexed_noise_sample(9, 0, 0, 0, 4, 2);
  const double centered =
      (static_cast<double>(sample) - pfc::field::indexed_noise_midpoint) / 65535.0;
  REQUIRE_THAT(u[0], WithinAbs(1.0 + 0.5 * centered, 1e-15));
}

TEST_CASE("Constant, Fourier modes, and noise compose", "[field][fourier][noise]") {
  if (world_size() != 1) SKIP("single-rank composition check");
  const auto domain = grid(8, 8, 1);
  const auto box = full_box(8, 8, 1);
  std::vector<double> u(64, 0.0);
  pfc::Constant constant(0.4);
  constant.apply(u, domain, box, 0.0);
  pfc::FourierModes modes;
  modes.terms.push_back(pfc::field::FourierMode{{1, 0, 0}, 0.05, 0.2});
  modes.apply(u, domain, box, 0.0);
  pfc::IndexedNoiseModifier noise;
  noise.noise = pfc::field::IndexedNoise{3, 0.01, true};
  noise.apply(u, domain, box, 0.0);

  std::vector<double> series(64, 0.0);
  pfc::field::fill_fourier_series(series, domain, box, 0.0, modes.terms);
  std::vector<double> fluctuation(64, 0.0);
  pfc::field::fill_indexed_noise(fluctuation, domain, box, 0.0, noise.noise,
                                 MPI_COMM_SELF);
  for (std::size_t i = 0; i < u.size(); ++i) {
    REQUIRE_THAT(u[i], WithinAbs(0.4 + series[i] + fluctuation[i], 1e-12));
  }
  REQUIRE_THAT(mean_of(u), WithinAbs(0.4, 1e-12));
}

TEST_CASE("Split ranks write the same indexed noise as one rank",
          "[field][noise][MPI]") {
  if (world_size() != 2) SKIP("two-rank noise check");
  constexpr int nx = 8;
  constexpr int ny = 4;
  const auto domain = grid(nx, ny, 1);
  const int rank = world_rank();
  const auto local = pfc::Box3i::from_bounds(
      {rank * (nx / 2), 0, 0}, {rank * (nx / 2) + nx / 2 - 1, ny - 1, 0});
  pfc::IndexedNoiseFill fill;
  fill.offset = -0.15;
  fill.noise = pfc::field::IndexedNoise{11, 0.01, true};
  std::vector<double> slice(static_cast<std::size_t>(local.count()), 0.0);
  const pfc::SimulationContext context(MPI_COMM_WORLD);
  fill.apply(context, slice, domain, local, 0.0);

  std::vector<double> full(static_cast<std::size_t>(nx * ny), 0.0);
  fill.apply(full, domain, full_box(nx, ny, 1), 0.0);
  std::size_t at = 0;
  for (int j = local.low[1]; j <= local.high[1]; ++j) {
    for (int i = local.low[0]; i <= local.high[0]; ++i) {
      const std::size_t global = static_cast<std::size_t>(i + nx * j);
      REQUIRE(slice[at] == full[global]);
      ++at;
    }
  }
}

TEST_CASE("cosine_mode JSON fills one mode and fourier_modes only adds",
          "[field][fourier]") {
  pfc::FourierSeriesFill legacy;
  pfc::ui::from_json(json{{"type", "cosine_mode"},
                          {"offset", 0.32},
                          {"amplitude", 0.02},
                          {"nx", 3},
                          {"ny", 1},
                          {"nz", 0}},
                     legacy);
  REQUIRE_THAT(legacy.offset, WithinAbs(0.32, 1e-15));
  REQUIRE(legacy.terms.size() == 1);
  REQUIRE(legacy.terms[0].index == pfc::Int3{3, 1, 0});
  REQUIRE_THAT(legacy.terms[0].amplitude, WithinAbs(0.02, 1e-15));

  pfc::FourierSeriesFill several;
  pfc::ui::from_json(
      json{{"type", "cosine_mode"},
           {"offset", 0.0},
           {"modes", json::array({{{"nx", 2}, {"ny", 1}, {"amplitude", 0.08}},
                                  {{"nx", 8}, {"ny", 4}, {"amplitude", 0.05}}})}},
      several);
  REQUIRE(several.terms.size() == 2);
  REQUIRE(several.terms[1].index == pfc::Int3{8, 4, 0});

  pfc::FourierModes added;
  pfc::ui::from_json(json{{"type", "fourier_modes"},
                          {"modes", json::array({{{"n", json::array({1, 0, 0})},
                                                  {"amplitude", 0.1},
                                                  {"phase", 0.2}}})}},
                     added);
  REQUIRE(added.terms.size() == 1);
  REQUIRE_THAT(added.terms[0].phase, WithinAbs(0.2, 1e-15));
  REQUIRE_THROWS(pfc::ui::from_json(
      json{
          {"type", "fourier_modes"}, {"offset", 1.0}, {"amplitude", 0.1}, {"nx", 1}},
      added));
}

TEST_CASE("seeded_noise JSON reads offset and rejects a negative seed",
          "[field][noise]") {
  pfc::IndexedNoiseFill fill;
  pfc::ui::from_json(json{{"type", "seeded_noise"},
                          {"offset", -0.05},
                          {"amplitude", 0.01},
                          {"seed", 7}},
                     fill);
  REQUIRE_THAT(fill.offset, WithinAbs(-0.05, 1e-15));
  REQUIRE(fill.noise.seed == 7);
  REQUIRE(fill.noise.remove_mean);
  REQUIRE_THROWS(pfc::ui::from_json(json{{"type", "seeded_noise"},
                                         {"offset", 0.32},
                                         {"amplitude", 0.02},
                                         {"seed", -1}},
                                    fill));

  pfc::IndexedNoiseModifier added;
  pfc::ui::from_json(json{{"type", "indexed_noise"},
                          {"seed", 4},
                          {"amplitude", 0.02},
                          {"remove_mean", false}},
                     added);
  REQUIRE(added.noise.remove_mean == false);
  REQUIRE_THROWS(pfc::ui::from_json(json{{"type", "indexed_noise"},
                                         {"seed", 4},
                                         {"amplitude", 0.02},
                                         {"offset", 1.0}},
                                    added));
}

TEST_CASE("A nonzero Fourier index requires a periodic axis longer than one",
          "[field][fourier]") {
  const pfc::field::FourierMode wave{{1, 0, 0}, 1.0, 0.0};
  const pfc::field::FourierMode flat{{0, 0, 0}, 1.5, 0.0};
  const auto bounded = pfc::domain::create(
      pfc::GridSize({8, 4, 1}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}), pfc::Bool3{false, true, true});
  const auto thin = grid(1, 4, 1);
  REQUIRE_THROWS_AS(
      pfc::field::fourier_series(bounded, pfc::domain::to_coords(bounded, {0, 0, 0}),
                                 std::span{&wave, 1}),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::field::fourier_series(thin, pfc::domain::to_coords(thin, {0, 0, 0}),
                                 std::span{&wave, 1}),
      std::invalid_argument);
  REQUIRE_THAT(pfc::field::fourier_series(bounded,
                                          pfc::domain::to_coords(bounded, {0, 0, 0}),
                                          std::span{&flat, 1}),
               WithinAbs(1.5, 1e-12));
  REQUIRE_THAT(pfc::field::fourier_series(thin,
                                          pfc::domain::to_coords(thin, {0, 0, 0}),
                                          std::span{&flat, 1}),
               WithinAbs(1.5, 1e-12));
}

TEST_CASE("Fractional Fourier indices are rejected", "[field][fourier]") {
  pfc::FourierSeriesFill fill;
  REQUIRE_THROWS_AS(pfc::ui::from_json(json{{"type", "cosine_mode"},
                                            {"offset", 0.0},
                                            {"amplitude", 1.0},
                                            {"nx", 1.5}},
                                       fill),
                    std::invalid_argument);
  pfc::FourierModes modes;
  REQUIRE_THROWS_AS(pfc::ui::from_json(
                        json{{"type", "fourier_modes"},
                             {"modes", json::array({{{"n", json::array({1.5, 0, 0})},
                                                     {"amplitude", 1.0}}})}},
                        modes),
                    std::invalid_argument);
}

TEST_CASE("Uncentered noise on a local box needs no collective", "[field][noise]") {
  const auto domain = grid(8, 4, 1);
  const auto left = pfc::Box3i::from_bounds({0, 0, 0}, {3, 3, 0});
  pfc::IndexedNoiseFill fill;
  fill.offset = 0.0;
  fill.noise = pfc::field::IndexedNoise{5, 1.0, false};
  std::vector<double> u(static_cast<std::size_t>(left.count()), 0.0);
  fill.apply(u, domain, left, 0.0);
  const auto sample = pfc::field::indexed_noise_sample(5, 0, 0, 0, 8, 4);
  const double centered =
      (static_cast<double>(sample) - pfc::field::indexed_noise_midpoint) / 65535.0;
  REQUIRE_THAT(u[0], WithinAbs(centered, 1e-15));

  pfc::IndexedNoiseModifier add;
  add.noise = pfc::field::IndexedNoise{5, 1.0, true};
  REQUIRE_THROWS_AS(add.apply(u, domain, left, 0.0), std::invalid_argument);
  fill.noise.remove_mean = true;
  REQUIRE_THROWS_AS(fill.apply(u, domain, left, 0.0), std::invalid_argument);
}

TEST_CASE("An application-supplied alias becomes offset", "[field][fourier]") {
  const json raw{{"initial_conditions", json::array({{{"type", "cosine_mode"},
                                                      {"level", 0.2},
                                                      {"amplitude", 0.1},
                                                      {"nx", 1}}})}};
  const auto rewritten = pfc::ui::copy_initial_offset_alias(raw, "level");
  pfc::FourierSeriesFill fill;
  pfc::ui::from_json(rewritten["initial_conditions"][0], fill);
  REQUIRE_THAT(fill.offset, WithinAbs(0.2, 1e-15));

  pfc::FourierSeriesFill ignored;
  pfc::ui::from_json(raw["initial_conditions"][0], ignored);
  REQUIRE_THAT(ignored.offset, WithinAbs(0.0, 1e-15));
}

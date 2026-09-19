// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_finite_strain_grid.cpp
 * @brief Drive the shipped 2-D periodic finite-strain homogenizer.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <inverse_homogenization/finite_strain_grid.hpp>
#include <inverse_homogenization/finite_strain_inverse.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using pfc::apps::inverse::fs::assign_lame_two_phase;
using pfc::apps::inverse::fs::evaluate_design;
using pfc::apps::inverse::fs::fill_rotating_squares_vec;
using pfc::apps::inverse::fs::homogenize_periodic_2d;
using pfc::apps::inverse::fs::init_cell;
using pfc::apps::inverse::fs::lame_from_young_poisson;
using pfc::apps::inverse::fs::Model;
using pfc::apps::inverse::fs::nu_t_at;
using pfc::apps::inverse::fs::PixelCell;
using pfc::apps::inverse::fs::relax_transverse;

TEST_CASE("uniform 8x8 grid recovers shipped homogeneous F22",
          "[finite-strain][484]") {
  const auto lame = lame_from_young_poisson(1.0, 0.3);
  const auto hom = relax_transverse(Model::NeoHookean, lame, 1.05);
  PixelCell c;
  init_cell(c, 8, 8, Model::NeoHookean);
  for (auto &L : c.lame) L = lame;
  const auto g = homogenize_periodic_2d(c, 1.05);
  REQUIRE(g.converged);
  REQUIRE(hom.converged);
  REQUIRE_THAT(g.F22, WithinAbs(hom.F22, 1e-7));
  REQUIRE_THAT(g.P22, WithinAbs(0.0, 1e-8));
}

TEST_CASE("16x16 rotating-square is auxetic at F11=1.05", "[finite-strain][484]") {
  const auto stiff = lame_from_young_poisson(1.0, 0.3);
  const auto voided = lame_from_young_poisson(1e-3, 0.3);
  std::vector<double> h;
  fill_rotating_squares_vec(h, 16, 16, 0.20, 0.45);
  PixelCell c;
  init_cell(c, 16, 16, Model::NeoHookean);
  assign_lame_two_phase(c, h, stiff, voided);
  const auto g = homogenize_periodic_2d(c, 1.05);
  REQUIRE(g.converged);
  REQUIRE(g.F22 > 1.0);
  REQUIRE_THAT(g.P22, WithinAbs(0.0, 1e-8));
  const double nu = nu_t_at(c, 1.05);
  REQUIRE(nu < 0.0);
}

TEST_CASE("single-material rotating-square path converges", "[finite-strain][484]") {
  const auto d = evaluate_design("single-material+void", 0.20, false);
  REQUIRE(d.all_converged);
  REQUIRE(d.points.size() == 5);
  for (const auto &p : d.points) {
    REQUIRE(p.converged);
    REQUIRE(p.F22 > 0.0);
  }
}

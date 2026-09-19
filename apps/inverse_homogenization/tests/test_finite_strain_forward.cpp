// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_finite_strain_forward.cpp
 * @brief Drive the shipped 2-D finite-strain forward ladder (OpenPFC #55).
 *
 * Does not call PeriodicHomogenizer or reuse small-strain C_H.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include <inverse_homogenization/finite_strain_forward.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using pfc::apps::inverse::fs::first_pk;
using pfc::apps::inverse::fs::lame_from_young_poisson;
using pfc::apps::inverse::fs::laminate_uniaxial;
using pfc::apps::inverse::fs::Model;
using pfc::apps::inverse::fs::plane_strain_small_nu;
using pfc::apps::inverse::fs::relax_transverse;
using pfc::apps::inverse::fs::run_declared_ladder;
using pfc::apps::inverse::fs::tangent_poisson_at;
using pfc::apps::inverse::fs::tangent_poisson_log;

TEST_CASE("Lamé conversion matches isotropic identities", "[finite-strain][55]") {
  const auto lame = lame_from_young_poisson(1.0, 0.3);
  REQUIRE_THAT(lame.mu, WithinRel(1.0 / (2.0 * 1.3), 1e-12));
  REQUIRE_THAT(lame.lambda,
               WithinRel(1.0 * 0.3 / ((1.0 + 0.3) * (1.0 - 2.0 * 0.3)), 1e-12));
  REQUIRE_THAT(plane_strain_small_nu(lame),
               WithinRel(lame.lambda / (lame.lambda + 2.0 * lame.mu), 1e-15));
}

TEST_CASE("homogeneous neo-Hookean uniaxial meets P22=0", "[finite-strain][55]") {
  const auto lame = lame_from_young_poisson(1.0, 0.3);
  const auto s = relax_transverse(Model::NeoHookean, lame, 1.10);
  REQUIRE(s.converged);
  REQUIRE(s.stable);
  REQUIRE(s.J > 0.0);
  const auto pk =
      first_pk(Model::NeoHookean, lame, pfc::apps::inverse::fs::diag2(s.F11, s.F22));
  REQUIRE_THAT(pk.P.a22, WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(pk.P.a12, WithinAbs(0.0, 1e-14));
  REQUIRE_THAT(s.P22, WithinAbs(pk.P.a22, 1e-16));
}

TEST_CASE("small-strain tangent Poisson matches plane-strain Lamé",
          "[finite-strain][55]") {
  const auto lame = lame_from_young_poisson(1.0, 0.3);
  const double nu_t = tangent_poisson_at(Model::NeoHookean, lame, 1.0001, 1e-6);
  REQUIRE_THAT(nu_t, WithinAbs(plane_strain_small_nu(lame), 2e-4));
  const double nu_stvk =
      tangent_poisson_at(Model::StVenantKirchhoff, lame, 1.0001, 1e-6);
  REQUIRE_THAT(nu_stvk, WithinAbs(plane_strain_small_nu(lame), 2e-4));
}

TEST_CASE("reported tangent Poisson matches a second shipped FD pair",
          "[finite-strain][55]") {
  const auto lame = lame_from_young_poisson(1.0, 0.3);
  const double F11 = 1.10;
  const double nu_t = tangent_poisson_at(Model::NeoHookean, lame, F11, 1e-5);
  const auto sm = relax_transverse(Model::NeoHookean, lame, F11 * 0.999);
  const auto sp = relax_transverse(Model::NeoHookean, lame, F11 * 1.001);
  REQUIRE(sm.converged);
  REQUIRE(sp.converged);
  const double nu_fd = tangent_poisson_log(sm.F11, sm.F22, sp.F11, sp.F22);
  REQUIRE_THAT(nu_t, WithinAbs(nu_fd, 5e-4));
}

TEST_CASE("equal-phase laminate recovers homogeneous F22", "[finite-strain][55]") {
  const auto lame = lame_from_young_poisson(1.0, 0.3);
  const double F11 = 1.10;
  const auto hom = relax_transverse(Model::NeoHookean, lame, F11);
  const auto lam = laminate_uniaxial(Model::NeoHookean, lame, lame, 0.5, F11);
  REQUIRE(lam.converged);
  REQUIRE_THAT(lam.F22, WithinRel(hom.F22, 1e-12));
  REQUIRE_THAT(lam.P11, WithinRel(hom.P11, 1e-12));
  REQUIRE_THAT(lam.P22, WithinAbs(0.0, 1e-10));
}

TEST_CASE("stiff/compliant laminate averages shipped transverse stretches",
          "[finite-strain][55]") {
  const auto stiff = lame_from_young_poisson(1.0, 0.3);
  const auto soft = lame_from_young_poisson(0.1, 0.1);
  const double F11 = 1.10;
  const double vf = 0.4;
  const auto lam = laminate_uniaxial(Model::NeoHookean, stiff, soft, vf, F11);
  REQUIRE(lam.converged);
  REQUIRE(lam.stable);
  const auto a =
      first_pk(Model::NeoHookean, stiff,
               pfc::apps::inverse::fs::diag2(lam.phase_a.F11, lam.phase_a.F22));
  const auto b =
      first_pk(Model::NeoHookean, soft,
               pfc::apps::inverse::fs::diag2(lam.phase_b.F11, lam.phase_b.F22));
  REQUIRE_THAT(a.P.a22, WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(b.P.a22, WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(
      lam.F22,
      WithinRel(vf * lam.phase_a.F22 + (1.0 - vf) * lam.phase_b.F22, 1e-15));
  REQUIRE(lam.phase_a.F22 != lam.phase_b.F22);
}

TEST_CASE("StVK uniaxial is stable and relaxed", "[finite-strain][55]") {
  const auto lame = lame_from_young_poisson(1.0, 0.3);
  const auto s = relax_transverse(Model::StVenantKirchhoff, lame, 1.20);
  REQUIRE(s.converged);
  REQUIRE(s.stable);
  REQUIRE(s.J > 0.0);
  REQUIRE(s.newton_iters <= 50);
  const auto pk = first_pk(Model::StVenantKirchhoff, lame,
                           pfc::apps::inverse::fs::diag2(s.F11, s.F22));
  REQUIRE_THAT(pk.P.a22, WithinAbs(0.0, 1e-10));
}

TEST_CASE("declared ladder rungs converge", "[finite-strain][55]") {
  const auto report = run_declared_ladder();
  REQUIRE(report.rungs.size() == 16);
  REQUIRE(report.all_converged);
  REQUIRE(report.all_stable);
  REQUIRE(report.small_strain_nu > 0.0);
  for (const auto &r : report.rungs) {
    REQUIRE(r.converged);
    REQUIRE(r.stable);
    REQUIRE(r.residual < 1e-10);
    REQUIRE(r.F22 > 0.0);
    REQUIRE(std::isfinite(r.nu_t));
  }
}

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_inverse_convergence.cpp
 * @brief Drive the shipped inverse-homogenization stopping protocol.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <string>

#include <inverse_homogenization/inverse_convergence.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using pfc::apps::inverse::continuation_fraction;
using pfc::apps::inverse::ConvergenceConfig;
using pfc::apps::inverse::ConvergenceMetrics;
using pfc::apps::inverse::ConvergenceTracker;
using pfc::apps::inverse::criteria_hold;
using pfc::apps::inverse::make_metrics;
using pfc::apps::inverse::params_frozen;
using pfc::apps::inverse::relative_norm_change;
using pfc::apps::inverse::relative_objective_change;
using pfc::apps::inverse::termination_name;
using pfc::apps::inverse::TerminationReason;

TEST_CASE("continuation fraction freezes at the last continuation step",
          "[inverse-conv][59]") {
  REQUIRE_THAT(continuation_fraction(0, 300), WithinAbs(0.0, 0.0));
  REQUIRE_THAT(continuation_fraction(299, 300), WithinAbs(1.0, 0.0));
  REQUIRE_THAT(continuation_fraction(500, 300), WithinAbs(1.0, 0.0));
  REQUIRE_THAT(continuation_fraction(0, 0), WithinAbs(1.0, 0.0));
  REQUIRE(!params_frozen(299, 300));
  REQUIRE(params_frozen(300, 300));
  REQUIRE(params_frozen(0, 0));
}

TEST_CASE("convergence cannot trigger during continuation", "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 5;
  tr.cfg.max_steps = 100;
  tr.cfg.conv_window = 2;
  tr.cfg.verify_steps = 0;
  ConvergenceMetrics quiet{};
  quiet.quiet = true;
  for (int s = 0; s < 5; ++s) {
    const auto r = tr.after_step(s, true, quiet);
    REQUIRE(r == TerminationReason::Running);
    REQUIRE_FALSE(tr.candidate);
  }
}

TEST_CASE("one quiet frozen step is not enough for window 20",
          "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 100;
  tr.cfg.conv_window = 20;
  tr.cfg.verify_steps = 0;
  ConvergenceMetrics quiet{};
  quiet.quiet = true;
  REQUIRE(tr.after_step(0, true, quiet) == TerminationReason::Running);
  REQUIRE_FALSE(tr.candidate);
  REQUIRE(tr.quiet_count == 1);
}

TEST_CASE("quiet counter resets when a frozen criterion fails",
          "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 100;
  tr.cfg.conv_window = 3;
  tr.cfg.verify_steps = 0;
  ConvergenceMetrics quiet{};
  quiet.quiet = true;
  ConvergenceMetrics loud{};
  loud.quiet = false;
  REQUIRE(tr.after_step(0, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.after_step(1, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.quiet_count == 2);
  REQUIRE(tr.after_step(2, true, loud) == TerminationReason::Running);
  REQUIRE(tr.quiet_count == 0);
  REQUIRE_FALSE(tr.candidate);
}

TEST_CASE("objective relative change uses max(1, |J_prev|)", "[inverse-conv][59]") {
  REQUIRE_THAT(relative_objective_change(0.0, 1e-9), WithinAbs(1e-9, 1e-20));
  REQUIRE_THAT(relative_objective_change(2.0, 2.002), WithinAbs(0.001, 1e-15));
  REQUIRE(relative_objective_change(-0.5, -0.5) == 0.0);
}

TEST_CASE("tensor relative change floors the previous norm", "[inverse-conv][59]") {
  REQUIRE_THAT(relative_norm_change(1e-8, 0.0, 1e-30), WithinRel(1e22, 1e-6));
  REQUIRE_THAT(relative_norm_change(1e-6, 2.0, 1e-30), WithinRel(5e-7, 1e-12));
}

TEST_CASE("make_metrics uses post-projection design RMS not step_rms",
          "[inverse-conv][59]") {
  ConvergenceConfig cfg;
  cfg.tol_design = 1e-4;
  cfg.tol_objective = 1e-6;
  cfg.tol_tensor = 1e-4;
  const auto quiet = make_metrics(1e-5, 1.0, 1.0, 1e-8, 1.0, 0.0, cfg);
  REQUIRE(quiet.quiet);
  REQUIRE_THAT(quiet.design_rms, WithinAbs(1e-5, 0.0));
  const auto loud_design = make_metrics(0.018, 1.0, 1.0, 1e-8, 1.0, 0.0, cfg);
  REQUIRE_FALSE(loud_design.quiet);
}

TEST_CASE("max-step termination is distinct from convergence",
          "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 3;
  tr.cfg.conv_window = 20;
  tr.cfg.verify_steps = 100;
  ConvergenceMetrics loud{};
  loud.quiet = false;
  REQUIRE(tr.after_step(0, true, loud) == TerminationReason::Running);
  REQUIRE(tr.after_step(1, true, loud) == TerminationReason::Running);
  REQUIRE(tr.after_step(2, true, loud) == TerminationReason::MaxSteps);
  REQUIRE(std::string(termination_name(TerminationReason::MaxSteps)) == "MAX_STEPS");
  REQUIRE(std::string(termination_name(TerminationReason::Converged)) ==
          "CONVERGED");
}

TEST_CASE("elasticity failure stops immediately", "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 5000;
  ConvergenceMetrics quiet{};
  quiet.quiet = true;
  REQUIRE(tr.after_step(0, false, quiet) == TerminationReason::ElasticityFailure);
}

TEST_CASE("verification hold rejects a false candidate", "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 5000;
  tr.cfg.conv_window = 2;
  tr.cfg.verify_steps = 2;
  ConvergenceMetrics quiet{};
  quiet.quiet = true;
  ConvergenceMetrics loud{};
  loud.quiet = false;
  REQUIRE(tr.after_step(0, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.after_step(1, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.candidate);
  REQUIRE(tr.after_step(2, true, loud) == TerminationReason::Running);
  REQUIRE_FALSE(tr.candidate);
  REQUIRE(tr.rejected_hold);
  REQUIRE(tr.quiet_count == 0);
}

TEST_CASE("verified hold after window reports CONVERGED", "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 5000;
  tr.cfg.conv_window = 2;
  tr.cfg.verify_steps = 2;
  ConvergenceMetrics quiet{};
  quiet.quiet = true;
  REQUIRE(tr.after_step(0, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.after_step(1, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.after_step(2, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.after_step(3, true, quiet) == TerminationReason::Converged);
  REQUIRE(tr.verified);
}

TEST_CASE("criteria_hold requires all three tolerances", "[inverse-conv][59]") {
  ConvergenceConfig cfg;
  REQUIRE(criteria_hold(1e-5, 1e-7, 1e-5, cfg));
  REQUIRE_FALSE(criteria_hold(1e-3, 1e-7, 1e-5, cfg));
  REQUIRE_FALSE(criteria_hold(1e-5, 1e-4, 1e-5, cfg));
  REQUIRE_FALSE(criteria_hold(1e-5, 1e-7, 1e-3, cfg));
}

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/simulation/simulation_driver.hpp>

TEST_CASE("a rejected attempt does not save or advance time", "[driver][unit]") {
  pfc::Time time({0.0, 0.3, 0.1}, 0.3);
  std::vector<int> saved;
  int attempts = 0;
  pfc::sim::run_attempts(
      time,
      [&](pfc::Time &clock) {
        ++attempts;
        REQUIRE(clock.attempt_active());
        if (attempts == 1) return pfc::sim::StepDecision{false};
        return pfc::sim::StepDecision{true};
      },
      pfc::sim::NoopHook{}, pfc::sim::NoopHook{},
      [&](const pfc::Time &clock) { saved.push_back(pfc::time::increment(clock)); });
  REQUIRE(pfc::time::done(time));
  REQUIRE(time.get_increment() == 3);
  REQUIRE(attempts == 4);
  REQUIRE(saved.size() == 2);
  REQUIRE(saved.front() == 0);
  REQUIRE(saved.back() == 3);
}

TEST_CASE("endless rejection stops", "[driver][unit]") {
  pfc::Time time({0.0, 1.0, 0.5}, 1.0);
  REQUIRE_THROWS_AS(
      pfc::sim::run_attempts(
          time, [](pfc::Time &) { return pfc::sim::StepDecision{false}; },
          pfc::sim::NoopHook{}, pfc::sim::NoopHook{}, pfc::sim::NoopHook{}, 2),
      std::runtime_error);
  REQUIRE(time.get_increment() == 0);
}

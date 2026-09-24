// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/simulation/simulation_driver.hpp>

namespace {

struct ProbeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

} // namespace

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
  REQUIRE(time.get_accepted_steps() == 3);
  REQUIRE(time.get_rejected_steps() == 1);
  REQUIRE(saved.size() == 2);
  REQUIRE(saved.front() == 0);
  REQUIRE(saved.back() == 3);
  REQUIRE_FALSE(time.attempt_active());
}

TEST_CASE("endless rejection stops", "[driver][unit]") {
  pfc::Time time({0.0, 1.0, 0.5}, 1.0);
  REQUIRE_THROWS_AS(pfc::sim::run_attempts(
                        time,
                        [](pfc::Time &) { return pfc::sim::StepDecision{false}; },
                        pfc::sim::NoopHook{}, pfc::sim::NoopHook{},
                        pfc::sim::NoopHook{}, pfc::sim::NoopHook{}, 2),
                    std::runtime_error);
  REQUIRE(time.get_increment() == 0);
  REQUIRE(time.get_accepted_time() == Catch::Approx(0.0));
  REQUIRE(time.get_accepted_steps() == 0);
  REQUIRE(time.get_rejected_steps() == 2);
  REQUIRE_FALSE(time.attempt_active());
}

TEST_CASE("apply throw closes the attempt without saving", "[driver][unit]") {
  pfc::Time time({0.0, 1.0, 0.5}, 1.0);
  int saves = 0;
  REQUIRE_THROWS_AS(
      pfc::sim::run_attempts(
          time, [](pfc::Time &) { return pfc::sim::StepDecision{true}; },
          pfc::sim::NoopHook{}, [](const pfc::Time &) { throw ProbeError("apply"); },
          [&](const pfc::Time &) { ++saves; }),
      ProbeError);
  REQUIRE_FALSE(time.attempt_active());
  REQUIRE(time.get_accepted_time() == Catch::Approx(0.0));
  REQUIRE(time.get_increment() == 0);
  REQUIRE(time.get_accepted_steps() == 0);
  REQUIRE(time.get_rejected_steps() == 0);
  REQUIRE(saves == 1);
}

TEST_CASE("step throw closes the attempt without saving", "[driver][unit]") {
  pfc::Time time({0.0, 1.0, 0.5}, 1.0);
  int saves = 0;
  REQUIRE_THROWS_AS(
      pfc::sim::run_attempts(
          time,
          [](pfc::Time &) -> pfc::sim::StepDecision { throw ProbeError("step"); },
          pfc::sim::NoopHook{}, pfc::sim::NoopHook{},
          [&](const pfc::Time &) { ++saves; }),
      ProbeError);
  REQUIRE_FALSE(time.attempt_active());
  REQUIRE(time.get_accepted_time() == Catch::Approx(0.0));
  REQUIRE(time.get_increment() == 0);
  REQUIRE(time.get_accepted_steps() == 0);
  REQUIRE(time.get_rejected_steps() == 0);
  REQUIRE(saves == 1);
}

TEST_CASE("fixed run saves initial, aligned, and final once", "[driver][unit]") {
  pfc::Time time({0.0, 1.0, 0.5}, 0.5);
  std::vector<double> saved;
  pfc::sim::run(
      time, [](double) {}, pfc::sim::NoopHook{}, pfc::sim::NoopHook{},
      [&](const pfc::Time &now) { saved.push_back(pfc::time::current(now)); });
  REQUIRE(saved.size() == 3);
  REQUIRE(saved[0] == Catch::Approx(0.0));
  REQUIRE(saved[1] == Catch::Approx(0.5));
  REQUIRE(saved[2] == Catch::Approx(1.0));
}

TEST_CASE("adaptive run saves only accepted states", "[driver][unit]") {
  pfc::Time time({0.0, 1.0, 0.5}, 1.0);
  std::vector<double> saved;
  int rejects_left = 1;
  pfc::sim::run_attempts(
      time,
      [&](pfc::Time &) {
        if (rejects_left > 0) {
          --rejects_left;
          return pfc::sim::StepDecision{false};
        }
        return pfc::sim::StepDecision{true};
      },
      pfc::sim::NoopHook{}, pfc::sim::NoopHook{},
      [&](const pfc::Time &now) { saved.push_back(pfc::time::current(now)); });
  REQUIRE(saved.size() == 2);
  REQUIRE(saved[0] == Catch::Approx(0.0));
  REQUIRE(saved[1] == Catch::Approx(1.0));
  REQUIRE(time.get_accepted_steps() == 2);
  REQUIRE(time.get_rejected_steps() == 1);
  REQUIRE_FALSE(time.attempt_active());
}

TEST_CASE("fixed run exposes the clipped final interval", "[driver][unit]") {
  pfc::Time time({0.0, 1.0, 0.3}, 1.0);
  std::vector<double> intervals;
  int one_arg_calls = 0;
  pfc::sim::run(time,
                [&](double, double interval) { intervals.push_back(interval); });
  REQUIRE(intervals.size() == 4);
  REQUIRE(intervals[0] == Catch::Approx(0.3));
  REQUIRE(intervals[1] == Catch::Approx(0.3));
  REQUIRE(intervals[2] == Catch::Approx(0.3));
  REQUIRE(intervals[3] == Catch::Approx(0.1));
  REQUIRE(time.get_current() == Catch::Approx(1.0));

  pfc::Time again({0.0, 1.0, 0.3}, 1.0);
  pfc::sim::run(again, [&](double) { ++one_arg_calls; });
  REQUIRE(one_arg_calls == 4);
  REQUIRE(again.get_current() == Catch::Approx(1.0));
}

TEST_CASE("adaptive attempts clip the final interval", "[driver][unit]") {
  pfc::Time time({0.0, 1.0, 0.3}, 1.0);
  std::vector<double> intervals;
  std::vector<double> saved;
  pfc::sim::run_attempts(
      time,
      [&](pfc::Time &clock) {
        intervals.push_back(clock.get_attempted_dt());
        return pfc::sim::StepDecision{true};
      },
      pfc::sim::NoopHook{}, pfc::sim::NoopHook{},
      [&](const pfc::Time &now) { saved.push_back(pfc::time::current(now)); });
  REQUIRE(intervals.size() == 4);
  REQUIRE(intervals.back() == Catch::Approx(0.1));
  REQUIRE(time.get_current() == Catch::Approx(1.0));
  REQUIRE(time.get_accepted_steps() == 4);
  REQUIRE(time.get_rejected_steps() == 0);
  REQUIRE(saved.size() == 2);
  REQUIRE(saved.front() == Catch::Approx(0.0));
  REQUIRE(saved.back() == Catch::Approx(1.0));
}

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mpi.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/profiling/session.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/initial_conditions/constant.hpp>
#include <openpfc/kernel/simulation/simulation_lifecycle.hpp>

namespace {

pfc::Box3i whole_box(int n) {
  return pfc::Box3i::from_bounds({0, 0, 0}, {n - 1, n - 1, 0});
}

void ensure_mpi() {
  int ready = 0;
  MPI_Initialized(&ready);
  if (ready == 0) MPI_Init(nullptr, nullptr);
}

class AddOne : public pfc::FieldModifier {
public:
  void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &,
             const pfc::Box3i &, double) override {
    for (std::size_t i = 0; i < field.size(); ++i) field[i] += 1.0;
  }
};

class RecordTime : public pfc::FieldModifier {
public:
  std::vector<double> *times{nullptr};

  void apply(pfc::field::FieldOutput<double>, const pfc::Domain &,
             const pfc::Box3i &, double time) override {
    times->push_back(time);
  }
};

} // namespace

TEST_CASE("lifecycle applies initial conditions to named fields once",
          "[lifecycle][unit]") {
  ensure_mpi();
  const auto domain = pfc::domain::create({4, 4, 1});
  const auto box = whole_box(4);
  pfc::data::Field<double> density(domain, box, 0);
  pfc::data::Field<double> solute(domain, box, 0);
  pfc::sim::SimulationLifecycle life(
      pfc::sim::SimulationLifecycle::schedule(0.0, 0.2, 0.1, 0.2), MPI_COMM_WORLD);
  life.bind_field("density", density);
  life.bind_field("solute", solute);
  life.add_initial_condition("density", std::make_unique<pfc::Constant>(1.5));
  life.add_initial_condition("solute", std::make_unique<pfc::Constant>(-0.25));
  life.add_initial_condition("density", std::make_unique<AddOne>());

  bool saw_initial = false;
  life.set_save_observer([&](const pfc::Time &now) {
    if (pfc::time::increment(now) != 0) return;
    saw_initial = true;
    REQUIRE(density(0, 0, 0) == Catch::Approx(2.5));
    REQUIRE(solute(0, 0, 0) == Catch::Approx(-0.25));
  });
  life.run([](double) {});
  REQUIRE(saw_initial);
  REQUIRE(density(0, 0, 0) == Catch::Approx(2.5));
}

TEST_CASE("stage preparation does not save", "[lifecycle][unit]") {
  ensure_mpi();
  const auto domain = pfc::domain::create({2, 2, 1});
  const auto box = whole_box(2);
  pfc::data::Field<double> density(domain, box, 0);
  pfc::sim::SimulationLifecycle life(
      pfc::sim::SimulationLifecycle::schedule(0.0, 0.2, 0.1, 0.2), MPI_COMM_WORLD);
  life.bind_field("density", density);
  std::vector<double> boundary_times;
  auto recorder = std::make_unique<RecordTime>();
  recorder->times = &boundary_times;
  life.add_boundary_condition("density", std::move(recorder));

  std::vector<double> saved;
  life.set_save_observer(
      [&](const pfc::Time &now) { saved.push_back(pfc::time::current(now)); });
  int stage_calls = 0;
  life.run([&](double) {
    life.prepare_stage(0.25);
    life.prepare_stage(0.5);
    stage_calls += 2;
  });
  REQUIRE(stage_calls == 4);
  REQUIRE(saved.size() == 2);
  REQUIRE(saved.front() == Catch::Approx(0.0));
  REQUIRE(saved.back() == Catch::Approx(0.2));
  REQUIRE(boundary_times.size() == 6);
  int stage_marks = 0;
  for (double t : boundary_times) {
    if (t == Catch::Approx(0.25) || t == Catch::Approx(0.5)) ++stage_marks;
  }
  REQUIRE(stage_marks == 4);
  for (double t : saved) {
    REQUIRE(t != Catch::Approx(0.25));
    REQUIRE(t != Catch::Approx(0.5));
  }
}

TEST_CASE("rejected attempt retries with a new dt and does not save",
          "[lifecycle][unit]") {
  ensure_mpi();
  const auto domain = pfc::domain::create({2, 2, 1});
  pfc::data::Field<double> density(domain, whole_box(2), 0);
  pfc::sim::SimulationLifecycle life(
      pfc::sim::SimulationLifecycle::schedule(0.0, 1.0, 0.3, 1.0), MPI_COMM_WORLD);
  life.bind_field("density", density);
  std::vector<double> saved;
  int checkpoints = 0;
  life.set_save_observer(
      [&](const pfc::Time &now) { saved.push_back(pfc::time::current(now)); });
  life.set_checkpoint_hook([&](const pfc::Time &) { ++checkpoints; });

  pfc::profiling::ProfilingSession profile(pfc::profiling::ProfilingMetricCatalog{},
                                           {"step"});
  life.set_profiling(&profile);

  std::vector<double> intervals;
  bool rejected = false;
  life.run_attempts([&](pfc::Time &clock) {
    intervals.push_back(clock.get_attempted_dt());
    if (!rejected) {
      rejected = true;
      clock.set_dt(0.2);
      return pfc::sim::StepDecision{false};
    }
    return pfc::sim::StepDecision{true};
  });

  REQUIRE(intervals.front() == Catch::Approx(0.3));
  REQUIRE(intervals[1] == Catch::Approx(0.2));
  REQUIRE(intervals.back() == Catch::Approx(0.2));
  REQUIRE(life.time().get_current() == Catch::Approx(1.0));
  REQUIRE(life.time().get_accepted_steps() == 5);
  REQUIRE(life.time().get_rejected_steps() == 1);
  REQUIRE(saved.size() == 2);
  REQUIRE(saved.front() == Catch::Approx(0.0));
  REQUIRE(saved.back() == Catch::Approx(1.0));
  REQUIRE(checkpoints == 2);
  REQUIRE(profile.num_frames() == intervals.size());
  REQUIRE_FALSE(life.time().attempt_active());
}

TEST_CASE("an unknown field name is rejected before the run", "[lifecycle][unit]") {
  ensure_mpi();
  const auto domain = pfc::domain::create({2, 2, 1});
  pfc::data::Field<double> density(domain, whole_box(2), 0);
  pfc::sim::SimulationLifecycle life(
      pfc::sim::SimulationLifecycle::schedule(0.0, 0.1, 0.1, 0.1), MPI_COMM_WORLD);
  life.bind_field("density", density);
  REQUIRE_THROWS_AS(life.bind_field("density", density), std::invalid_argument);
  REQUIRE_THROWS_AS(
      life.add_initial_condition("pressure", std::make_unique<pfc::Constant>(0.0)),
      std::invalid_argument);
}

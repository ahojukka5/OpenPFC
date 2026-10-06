// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_history.cpp
 * @brief Rollback and commit of a constitutive history, with no material law.
 */

#include <catch2/catch_test_macros.hpp>

#include <openpfc/mechanics/constitutive/history.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

using pfc::finite_strain::TransactionalHistory;

namespace {

struct Marker {
  std::uint64_t tag = 0;
  double value = 0.0;
};

} // namespace

TEST_CASE("a rejected trial restores the committed marker exactly",
          "[finite_strain][history]") {
  TransactionalHistory<Marker> history(std::vector<Marker>{{7, 1.25}, {8, -2.5}});
  history.begin();
  history.trial(0).tag = 99;
  history.trial(0).value = 4.0;
  history.trial(1).value = 3.0;
  history.reject();

  const TransactionalHistory<Marker> &closed = history;
  REQUIRE(history.committed(0).tag == 7);
  REQUIRE(history.committed(0).value == 1.25);
  REQUIRE(history.committed(1).tag == 8);
  REQUIRE(history.committed(1).value == -2.5);
  REQUIRE(closed.trial(0).tag == 7);
  REQUIRE(closed.trial(1).value == -2.5);
  REQUIRE_FALSE(history.trial_open());
}

TEST_CASE("accept publishes the trial and a later reject stops there",
          "[finite_strain][history]") {
  TransactionalHistory<Marker> history(std::vector<Marker>{{1, 0.0}, {2, 0.0}});
  history.begin();
  history.trial(1).tag = 20;
  history.trial(1).value = 0.5;
  history.accept();
  REQUIRE(history.committed(1).tag == 20);
  REQUIRE(history.committed(1).value == 0.5);
  REQUIRE(history.committed(0).tag == 1);

  history.begin();
  history.trial(1).tag = 100;
  history.reject();
  REQUIRE(history.committed(1).tag == 20);
  REQUIRE(history.committed(0).tag == 1);
}

TEST_CASE("accept without an open trial is refused", "[finite_strain][history]") {
  TransactionalHistory<Marker> history(std::vector<Marker>{{1, 0.0}});
  REQUIRE_THROWS_AS(history.accept(), std::logic_error);
  REQUIRE_THROWS_AS(history.trial(0).tag = 4, std::logic_error);
  REQUIRE(history.committed(0).tag == 1);
}

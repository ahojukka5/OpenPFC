// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/decomposition/halo_overlap.hpp>

TEST_CASE("halo_overlap_mode_from_int maps 0/1/2", "[halo][overlap]") {
  using pfc::comm::HaloOverlapMode;
  REQUIRE(pfc::comm::halo_overlap_mode_from_int(0) == HaloOverlapMode::Blocking);
  REQUIRE(pfc::comm::halo_overlap_mode_from_int(1) == HaloOverlapMode::Waitall);
  REQUIRE(pfc::comm::halo_overlap_mode_from_int(2) == HaloOverlapMode::Testall);
}

TEST_CASE("halo_overlap_mode_from_int rejects out of range", "[halo][overlap]") {
  try {
    (void)pfc::comm::halo_overlap_mode_from_int(-1);
    FAIL("expected invalid_argument");
  } catch (const std::invalid_argument &e) {
    REQUIRE(std::string(e.what()).find("HaloOverlapMode") != std::string::npos);
  }
  REQUIRE_THROWS_AS(pfc::comm::halo_overlap_mode_from_int(3), std::invalid_argument);
}

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/grain/transfer.hpp>
using namespace pfc::grain;
namespace dg = pfc::grain::diagnostics;
TEST_CASE("single-cell no-op and move have independent source operation counts",
          "[grain][diagnostics]") {
  dg::Accesses counts{};
  const double q[]{.5};
  const Id uid[]{7};
  const detail::Assignment assignment[]{{7, 0, 0}};
  double staged[1];
  Id labels[1];
  REQUIRE(detail::stage_cell<true>(0, 1, 1, q, uid, assignment, 1, 0, staged, labels,
                                   &counts) == TransferStatus::Success);
  const auto f = [](dg::Field field) { return static_cast<std::size_t>(field); };
  REQUIRE(counts.reads[f(dg::Field::Values)] == 4);
  REQUIRE(counts.reads[f(dg::Field::Labels)] == 8);
  REQUIRE(counts.reads[f(dg::Field::Assignments)] == 9);
  REQUIRE(counts.writes[f(dg::Field::StagedValues)] == 1);
  REQUIRE(counts.writes[f(dg::Field::StagedLabels)] == 1);
  REQUIRE(counts.read_bytes[f(dg::Field::Assignments)] ==
          9 * sizeof(detail::Assignment));
  counts = {};
  const double two[]{.5, 0};
  const Id ids[]{7, 0};
  const detail::Assignment move[]{{7, 0, 1}};
  double out[2];
  Id out_ids[2];
  REQUIRE(detail::stage_cell<true>(0, 1, 2, two, ids, move, 1, 0, out, out_ids,
                                   &counts) == TransferStatus::Success);
  REQUIRE(out[0] == 0);
  REQUIRE(out[1] == .5);
  REQUIRE(out_ids[0] == 0);
  REQUIRE(out_ids[1] == 7);
  REQUIRE(counts.reads[f(dg::Field::Values)] == 9);
  REQUIRE(counts.reads[f(dg::Field::Labels)] == 14);
  REQUIRE(counts.writes[f(dg::Field::StagedValues)] == 4);
  REQUIRE(counts.writes[f(dg::Field::StagedLabels)] == 4);
}

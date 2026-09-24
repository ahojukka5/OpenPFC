// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/frontend/io/diagnostics_series.hpp>

using Catch::Matchers::ContainsSubstring;

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

std::filesystem::path temp_dir(const char *name) {
  const auto dir =
      std::filesystem::temp_directory_path() / (std::string("pfc-diag-") + name);
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

} // namespace

TEST_CASE("A diagnostics series writes a deterministic header and rows",
          "[diagnostics]") {
  if (world_size() != 1) SKIP("single-rank diagnostics");
  const auto dir = temp_dir("rows");
  const auto path = dir / "run.csv";
  {
    pfc::io::DiagnosticsSeries series(
        {"mean", "rms"},
        pfc::io::DiagnosticsSeriesOptions{.path = path, .comm = MPI_COMM_SELF});
    series.write(0, 0.5, {1.0, 0.25});
    series.write(2, 1.5, {3.0, 0.5});
    series.close();
    series.close();
  }
  std::ifstream in(path);
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  REQUIRE(text == "step,time,mean,rms\n0,0.5,1,0.25\n2,1.5,3,0.5\n");
  REQUIRE_THROWS_WITH(
      pfc::io::DiagnosticsSeries(
          {"mean"},
          pfc::io::DiagnosticsSeriesOptions{.path = path, .comm = MPI_COMM_SELF}),
      ContainsSubstring("already exists"));
  {
    pfc::io::DiagnosticsSeries series(
        {"mean"}, pfc::io::DiagnosticsSeriesOptions{
                      .path = path, .comm = MPI_COMM_SELF, .overwrite = true});
    series.write(1, 2.0, {4.0});
    series.close();
  }
  std::ifstream again(path);
  std::string replaced((std::istreambuf_iterator<char>(again)),
                       std::istreambuf_iterator<char>());
  REQUIRE(replaced == "step,time,mean\n1,2,4\n");
  std::filesystem::remove_all(dir);
}

TEST_CASE("A diagnostics series rejects a short row and a bad column name",
          "[diagnostics]") {
  if (world_size() != 1) SKIP("single-rank diagnostics");
  const auto dir = temp_dir("reject");
  REQUIRE_THROWS_WITH(
      pfc::io::DiagnosticsSeries(
          {"a,b"}, pfc::io::DiagnosticsSeriesOptions{.path = dir / "x.csv",
                                                     .comm = MPI_COMM_SELF}),
      ContainsSubstring("column name"));
  pfc::io::DiagnosticsSeries series(
      {"mean"}, pfc::io::DiagnosticsSeriesOptions{.path = dir / "ok.csv",
                                                  .comm = MPI_COMM_SELF});
  REQUIRE_THROWS_WITH(series.write(0, 0.0, {}), ContainsSubstring("expected 1"));
  series.close();
  std::filesystem::remove_all(dir);
}

TEST_CASE("Opening a diagnostics file that is a directory fails", "[diagnostics]") {
  if (world_size() != 1) SKIP("single-rank diagnostics");
  const auto dir = temp_dir("dir");
  REQUIRE_THROWS_WITH(
      pfc::io::DiagnosticsSeries(
          {"mean"}, pfc::io::DiagnosticsSeriesOptions{.path = dir,
                                                      .comm = MPI_COMM_SELF,
                                                      .overwrite = true}),
      ContainsSubstring("cannot open"));
  std::filesystem::remove_all(dir);
}

TEST_CASE("A bad column on one rank fails before any writer", "[diagnostics][MPI]") {
  if (world_size() != 2) SKIP("two-rank diagnostics");
  const int rank = world_rank();
  const auto path =
      std::filesystem::temp_directory_path() / "pfc-diag-mpi-column.csv";
  const std::vector<std::string> columns =
      rank == 0 ? std::vector<std::string>{"mean"} : std::vector<std::string>{"a,b"};
  if (rank == 0) {
    REQUIRE_THROWS_WITH(
        pfc::io::DiagnosticsSeries(
            columns,
            pfc::io::DiagnosticsSeriesOptions{.path = path, .comm = MPI_COMM_WORLD}),
        ContainsSubstring("a peer rejected the configuration"));
  } else {
    REQUIRE_THROWS_WITH(
        pfc::io::DiagnosticsSeries(
            columns,
            pfc::io::DiagnosticsSeriesOptions{.path = path, .comm = MPI_COMM_WORLD}),
        ContainsSubstring("column name"));
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("CSV open failure is reported on every rank", "[diagnostics][MPI]") {
  if (world_size() != 2) SKIP("two-rank diagnostics");
  const int rank = world_rank();
  const auto dir = std::filesystem::temp_directory_path() / "pfc-diag-mpi-open-dir";
  if (rank == 0) {
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    REQUIRE_THROWS_WITH(
        pfc::io::DiagnosticsSeries(
            {"mean"}, pfc::io::DiagnosticsSeriesOptions{.path = dir,
                                                        .comm = MPI_COMM_WORLD,
                                                        .overwrite = true}),
        ContainsSubstring("cannot open"));
  } else {
    REQUIRE_THROWS_WITH(
        pfc::io::DiagnosticsSeries(
            {"mean"}, pfc::io::DiagnosticsSeriesOptions{.path = dir,
                                                        .comm = MPI_COMM_WORLD,
                                                        .overwrite = true}),
        ContainsSubstring("another rank"));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::filesystem::remove_all(dir);
  MPI_Barrier(MPI_COMM_WORLD);
}

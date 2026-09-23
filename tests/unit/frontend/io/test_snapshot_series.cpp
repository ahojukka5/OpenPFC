// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

#include <openpfc/frontend/io/snapshot_series.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/state_access.hpp>

using Catch::Matchers::WithinAbs;
using nlohmann::json;

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

std::filesystem::path shared_temp(const char *tag) {
  std::string name;
  if (world_rank() == 0) {
    name = std::string("openpfc-snap-") + tag + "-" + std::to_string(::getpid());
  }
  int len = static_cast<int>(name.size());
  MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
  name.resize(static_cast<std::size_t>(len));
  MPI_Bcast(name.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
  return std::filesystem::temp_directory_path() / name;
}

std::vector<double> read_doubles(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  in.seekg(0, std::ios::end);
  const auto bytes = static_cast<std::size_t>(in.tellg());
  REQUIRE(bytes % sizeof(double) == 0);
  in.seekg(0);
  std::vector<double> out(bytes / sizeof(double));
  in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(bytes));
  REQUIRE(in.good());
  return out;
}

pfc::io::SnapshotSeriesOptions opts(const std::filesystem::path &dir,
                                    std::string prefix, MPI_Comm comm) {
  return pfc::io::SnapshotSeriesOptions{dir, std::move(prefix), comm};
}

} // namespace

TEST_CASE("A writer that omits complex fields rejects them before use",
          "[snapshot][writer]") {
  class RealOnly : public pfc::ResultsWriter {
  public:
    void set_domain(const std::array<int, 3> &, const std::array<int, 3> &,
                    const std::array<int, 3> &) override {}
    [[nodiscard]] bool writes_real() const override { return true; }
    MPI_Status write(int, pfc::field::FieldView<double>) override {
      return MPI_Status{};
    }
  };
  RealOnly writer;
  REQUIRE(writer.writes_real());
  REQUIRE_FALSE(writer.writes_complex());
  pfc::ResultsWriter &sink = writer;
  std::vector<std::complex<double>> z(2);
  REQUIRE_THROWS_AS(sink.write(0, z), std::invalid_argument);
}

TEST_CASE("One scalar field is written with step and time metadata", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("one");
  const auto domain = pfc::domain::create({4, 2, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> phi(domain, box, 0);
  phi.apply([](const pfc::Real3 &x) { return 10.0 * x[0] + x[1]; });
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("phi", phi);
    REQUIRE(std::filesystem::is_directory(dir));
    series.write(3, 1.5);
    series.write(6, 3.0);
    series.close();
    REQUIRE(series.steps() == std::vector<int>{3, 6});
    REQUIRE_THAT(series.times()[0], WithinAbs(1.5, 1e-15));
    REQUIRE_THAT(series.times()[1], WithinAbs(3.0, 1e-15));
  }
  const auto values = read_doubles(dir / "run_phi_0000.bin");
  REQUIRE(values.size() == 8);
  REQUIRE_THAT(values[0], WithinAbs(0.0, 1e-12));
  const auto manifest = json::parse(std::ifstream(dir / "run_manifest.json"));
  REQUIRE(manifest.at("fields") == json::array({"phi"}));
  REQUIRE(manifest.at("steps") == json::array({3, 6}));
  REQUIRE(manifest.at("nx") == 4);
  REQUIRE(std::filesystem::exists(dir / "run.xdmf"));
  std::filesystem::remove_all(dir);
}

TEST_CASE("Several scalar fields share one series", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("multi");
  const auto domain = pfc::domain::create({2, 2, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> phi(domain, box, 0);
  pfc::data::Field<double> temperature(domain, box, 0);
  phi.apply([](const pfc::Real3 &) { return 1.0; });
  temperature.apply([](const pfc::Real3 &) { return 2.0; });
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("phi", phi);
    series.add_field("temperature", temperature);
    series.write(0, 0.0);
    series.close();
  }
  const auto a = read_doubles(dir / "run_phi_0000.bin");
  const auto b = read_doubles(dir / "run_temperature_0000.bin");
  REQUIRE(a.size() == 4);
  REQUIRE(b.size() == 4);
  for (double v : a) REQUIRE_THAT(v, WithinAbs(1.0, 1e-15));
  for (double v : b) REQUIRE_THAT(v, WithinAbs(2.0, 1e-15));
  const auto manifest = json::parse(std::ifstream(dir / "run_manifest.json"));
  REQUIRE(manifest.at("fields").size() == 2);
  std::filesystem::remove_all(dir);
}

TEST_CASE("A padded field writes owned cells only", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("pad");
  const auto domain = pfc::domain::create({3, 2, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> field(domain, box, 1);
  field.with_host_view([&](double *data, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) data[i] = -12345.0;
  });
  field.apply([](double, double, double) { return 7.0; });
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("u", field);
    series.write(0, 0.0);
    series.close();
  }
  const auto values = read_doubles(dir / "run_u_0000.bin");
  REQUIRE(values.size() == 6);
  for (double v : values) REQUIRE_THAT(v, WithinAbs(7.0, 1e-15));
  std::filesystem::remove_all(dir);
}

TEST_CASE("An unpadded face-halo field writes its owned brick", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("face");
  const auto domain = pfc::domain::create({2, 2, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> field(domain, box, 0, 1);
  REQUIRE(field.storage_halo() == 0);
  REQUIRE(field.halo_width() == 1);
  field.apply([](double, double, double) { return 4.0; });
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("u", field);
    series.write(1, 0.25);
    series.close();
  }
  const auto values = read_doubles(dir / "run_u_0000.bin");
  REQUIRE(values.size() == 4);
  for (double v : values) REQUIRE_THAT(v, WithinAbs(4.0, 1e-15));
  std::filesystem::remove_all(dir);
}

TEST_CASE("JSON fields keep their names and VTK payload", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("vtk");
  const auto pattern = (dir / "h_%04d.vti").string();
  const auto domain = pfc::domain::create({2, 1, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> h(domain, box, 0);
  h.apply([](double, double, double) { return 1.25; });
  const json cfg{{"fields", json::array({{{"name", "h"}, {"data", pattern}}})}};
  {
    pfc::io::SnapshotSeries series(
        domain, box, pfc::io::SnapshotSeriesOptions{.comm = MPI_COMM_SELF});
    series.bind_json_field(cfg, "h", h);
    series.finish_json_fields(cfg);
    series.write(0, 0.0);
    series.close();
  }
  const auto path = dir / "h_0000.vti";
  REQUIRE(std::filesystem::exists(path));
  std::ifstream in(path, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  REQUIRE(text.find("Name=\"h\"") != std::string::npos);
  const auto marker = text.find("\n_");
  REQUIRE(marker != std::string::npos);
  const auto *raw =
      reinterpret_cast<const unsigned char *>(text.data() + marker + 2);
  std::uint64_t bytes = 0;
  for (int i = 0; i < 8; ++i) bytes |= static_cast<std::uint64_t>(raw[i]) << (8 * i);
  REQUIRE(bytes == 2 * sizeof(double));
  double value = 0.0;
  std::memcpy(&value, raw + 8, sizeof(double));
  REQUIRE_THAT(value, WithinAbs(1.25, 1e-15));
  const json missing_name{
      {"fields", json::array({{{"name", "c"}, {"data", pattern}}})}};
  pfc::io::SnapshotSeries missing(
      domain, box, pfc::io::SnapshotSeriesOptions{.comm = MPI_COMM_SELF});
  missing.bind_json_field(missing_name, "h", h);
  REQUIRE_THROWS_AS(missing.finish_json_fields(missing_name), std::invalid_argument);
  std::filesystem::remove_all(dir);
}

TEST_CASE("Output directory creation fails closed", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("fail");
  std::filesystem::create_directories(dir);
  const auto blocker = dir / "not-a-directory";
  {
    std::ofstream touch(blocker);
    REQUIRE(touch.good());
  }
  const auto domain = pfc::domain::create({2, 1, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> u(domain, box, 0);
  pfc::io::SnapshotSeries series(
      domain, box, pfc::io::SnapshotSeriesOptions{.comm = MPI_COMM_SELF});
  REQUIRE_THROWS_AS(series.add_field("u", u, (blocker / "u_%04d.bin").string(),
                                     pfc::io::SnapshotFormat::Binary),
                    std::runtime_error);
  std::filesystem::remove_all(dir);
}

TEST_CASE("Two ranks write one owned brick", "[snapshot][MPI]") {
  if (world_size() != 2) SKIP("two-rank snapshot");
  const auto dir = shared_temp("mpi");
  const int rank = world_rank();
  const auto domain = pfc::domain::create({4, 2, 1});
  const auto local = pfc::Box3i::from_bounds({rank * 2, 0, 0}, {rank * 2 + 1, 1, 0});
  pfc::data::Field<double> phi(domain, local, 1);
  phi.with_host_view([&](double *data, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) data[i] = -99.0;
  });
  for (int k = 0; k < local.size[2]; ++k) {
    for (int j = 0; j < local.size[1]; ++j) {
      for (int i = 0; i < local.size[0]; ++i) {
        phi(i, j, k) = static_cast<double>(local.low[0] + i + 10 * j);
      }
    }
  }
  {
    pfc::io::SnapshotSeries series(domain, local, opts(dir, "run", MPI_COMM_WORLD));
    series.add_field("phi", phi);
    series.write(4, 0.5);
    series.close();
  }
  if (rank == 0) {
    const auto values = read_doubles(dir / "run_phi_0000.bin");
    REQUIRE(values.size() == 8);
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 4; ++i) {
        REQUIRE_THAT(values[static_cast<std::size_t>(i + 4 * j)],
                     WithinAbs(static_cast<double>(i + 10 * j), 1e-12));
      }
    }
    const auto manifest = json::parse(std::ifstream(dir / "run_manifest.json"));
    REQUIRE(manifest.at("steps") == json::array({4}));
    REQUIRE(std::filesystem::exists(dir / "run.xdmf"));
    std::filesystem::remove_all(dir);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file snapshot_series.hpp
 * @brief Solver-independent scalar-field snapshots.
 *
 * A driver registers fields and calls `write(step, time)`. The series owns
 * the output directory, the owned-cell pack, the writer, and the time-series
 * manifest. It does not know which stepper produced the field.
 *
 * The registered field is not owned. It must outlive every `write`. Device
 * fields are read through `with_host_read`, so a device backend is not
 * special-cased here. `BinaryWriter::~BinaryWriter` can still throw; that
 * cleanup belongs to issue #171.
 */

#include <cstdio>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/frontend/io/xdmf_binary_series.hpp>
#include <openpfc/frontend/utils/utils.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/simulation/results_writer_domain.hpp>

namespace pfc::io {

enum class SnapshotFormat { Binary, Vtk };

struct SnapshotSeriesOptions {
  std::filesystem::path directory;
  std::string prefix;
  MPI_Comm comm{MPI_COMM_NULL};
};

class SnapshotSeries {
public:
  SnapshotSeries(const Domain &domain, const Box3i &owned,
                 SnapshotSeriesOptions options)
      : m_domain(domain), m_owned(owned), m_options(std::move(options)) {
    if (m_options.comm == MPI_COMM_NULL) {
      throw std::invalid_argument("snapshot series: pass an explicit communicator");
    }
    MPI_Comm_rank(m_options.comm, &m_rank);
  }

  SnapshotSeries(const SnapshotSeries &) = delete;
  SnapshotSeries &operator=(const SnapshotSeries &) = delete;

  ~SnapshotSeries() {
    try {
      close();
    } catch (...) {
    }
  }

  /// Pack @p field's owned cells. Halos are not written.
  template <typename Space>
  void add_field(std::string name, data::Field<double, Space> &field,
                 std::string pattern = {},
                 SnapshotFormat format = SnapshotFormat::Binary) {
    if (field.box() != m_owned) {
      throw std::invalid_argument("snapshot series: '" + name +
                                  "' is not on the series owned box");
    }
    if (pattern.empty()) {
      if (m_options.directory.empty() || m_options.prefix.empty()) {
        throw std::invalid_argument(
            "snapshot series: a field needs a path or a directory and prefix");
      }
      pattern = (m_options.directory / (m_options.prefix + "_" + name + "_%04d.bin"))
                    .string();
      format = SnapshotFormat::Binary;
    }
    add_packed(std::move(name), std::move(pattern), format,
               [&field](std::vector<double> &out) { pack_owned(field, out); });
  }

  /// @p pack writes the owned brick, x-fastest, with no halo cells.
  void add_field(std::string name, std::string pattern, SnapshotFormat format,
                 std::function<void(std::vector<double> &)> pack) {
    if (!pack) {
      throw std::invalid_argument("snapshot series: '" + name + "' has no source");
    }
    add_packed(std::move(name), std::move(pattern), format, std::move(pack));
  }

  /**
   * @brief Register @p field when `fields[]` asks for @p name.
   *
   * `.vti` selects VTK and `.bin` selects raw binary. Call
   * `finish_json_fields` afterwards so a listed name with no source fails.
   */
  template <typename Space>
  void bind_json_field(const nlohmann::json &cfg, std::string_view name,
                       data::Field<double, Space> &field) {
    if (!cfg.contains("fields")) return;
    for (const auto &entry : json_field_entries(cfg)) {
      if (entry.at("name").get<std::string>() != name) continue;
      const std::string path = entry.at("data").get<std::string>();
      const auto ext = std::filesystem::path(path).extension().string();
      const std::string owned(name);
      if (ext == ".vti") {
        add_field(owned, field, path, SnapshotFormat::Vtk);
      } else if (ext == ".bin") {
        add_field(owned, field, path, SnapshotFormat::Binary);
      } else {
        throw std::invalid_argument("fields: '" + path +
                                    "' must be a .vti or .bin pattern");
      }
      m_bound.push_back(owned);
    }
  }

  /// Fail if `fields[]` names a source that was not bound.
  void finish_json_fields(const nlohmann::json &cfg) const {
    if (!cfg.contains("fields")) return;
    for (const auto &entry : json_field_entries(cfg)) {
      const std::string name = entry.at("name").get<std::string>();
      bool found = false;
      for (const auto &bound : m_bound) {
        if (bound == name) found = true;
      }
      if (!found) {
        throw std::invalid_argument("fields: no registered source named '" + name +
                                    "'");
      }
    }
  }

  [[nodiscard]] bool empty() const noexcept { return m_slots.empty(); }
  [[nodiscard]] int frames() const noexcept { return m_index; }
  [[nodiscard]] const std::vector<int> &steps() const noexcept { return m_steps; }
  [[nodiscard]] const std::vector<double> &times() const noexcept { return m_times; }

  /// File ordinal is the number of `write` calls. @p step and @p time go in the
  /// manifest, not into the filename.
  void write(int step, double time) {
    if (m_closed) {
      throw std::invalid_argument("snapshot series: write after close");
    }
    if (m_slots.empty()) return;
    for (Slot &slot : m_slots) {
      slot.pack(m_packed);
      if (!slot.writer->writes_real()) {
        throw std::invalid_argument("snapshot series: '" + slot.name +
                                    "' writer does not support real fields");
      }
      slot.writer->write(m_index, pfc::field::FieldView<double>(m_packed));
    }
    m_steps.push_back(step);
    m_times.push_back(time);
    ++m_index;
  }

  /// Rank 0 writes the manifest and, for binary frames, one XDMF series.
  void close() {
    if (m_closed) return;
    m_closed = true;
    if (m_slots.empty()) return;
    int failed = 0;
    if (m_rank == 0) {
      try {
        write_manifest_();
      } catch (...) {
        failed = 1;
      }
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, m_options.comm);
    if (failed != 0) {
      throw std::runtime_error("snapshot series: manifest write failed");
    }
  }

private:
  static std::vector<nlohmann::json> json_field_entries(const nlohmann::json &cfg) {
    const auto &fields = cfg.at("fields");
    if (!fields.is_array()) {
      throw std::invalid_argument(
          "fields: must be an array of {name, data} objects");
    }
    std::vector<nlohmann::json> entries;
    entries.reserve(fields.size());
    for (const auto &entry : fields) {
      if (!entry.is_object() || !entry.contains("name") || !entry.contains("data") ||
          !entry.at("name").is_string() || !entry.at("data").is_string()) {
        throw std::invalid_argument(
            "fields: each entry needs string 'name' and 'data'");
      }
      entries.push_back(entry);
    }
    return entries;
  }

  struct Slot {
    std::string name;
    SnapshotFormat format{SnapshotFormat::Binary};
    std::string pattern;
    std::unique_ptr<ResultsWriter> writer;
    std::function<void(std::vector<double> &)> pack;
  };

  template <typename Space>
  static void pack_owned(data::Field<double, Space> &field,
                         std::vector<double> &out) {
    const auto n = field.box().size;
    const int nx = n[0];
    const int ny = n[1];
    const int nz = n[2];
    const int halo = field.storage_halo();
    const auto npx = static_cast<std::size_t>(nx + 2 * halo);
    const auto npy = static_cast<std::size_t>(ny + 2 * halo);
    const auto hw = static_cast<std::size_t>(halo);
    out.resize(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
               static_cast<std::size_t>(nz));
    field.with_host_read([&](const double *data, std::size_t) {
      std::size_t q = 0;
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            const std::size_t src = (static_cast<std::size_t>(i) + hw) +
                                    (static_cast<std::size_t>(j) + hw) * npx +
                                    (static_cast<std::size_t>(k) + hw) * npx * npy;
            out[q++] = data[src];
          }
        }
      }
    });
  }

  void add_packed(std::string name, std::string pattern, SnapshotFormat format,
                  std::function<void(std::vector<double> &)> pack) {
    if (pattern.empty()) {
      throw std::invalid_argument("snapshot series: '" + name + "' has no path");
    }
    ensure_parent_(std::filesystem::path(pattern));
    std::unique_ptr<ResultsWriter> writer;
    if (format == SnapshotFormat::Vtk) {
      auto vtk = std::make_unique<VTKWriter>(pattern, m_options.comm);
      vtk->set_field_name(name);
      writer = std::move(vtk);
    } else {
      writer = std::make_unique<BinaryWriter>(pattern, m_options.comm);
    }
    if (!writer->writes_real()) {
      throw std::invalid_argument("snapshot series: writer for '" + name +
                                  "' does not support real fields");
    }
    apply_writer_domain(*writer, m_domain, m_owned);
    m_slots.push_back(Slot{std::move(name), format, std::move(pattern),
                           std::move(writer), std::move(pack)});
  }

  void ensure_parent_(const std::filesystem::path &pattern) {
    const std::filesystem::path parent = pattern.parent_path();
    if (parent.empty()) return;
    int failed = 0;
    if (m_rank == 0) {
      std::error_code ec;
      std::filesystem::create_directories(parent, ec);
      if (ec) failed = 1;
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, m_options.comm);
    MPI_Barrier(m_options.comm);
    if (failed != 0) {
      throw std::runtime_error("snapshot series: cannot create output directory '" +
                               parent.string() + "'");
    }
  }

  void write_manifest_() const {
    const std::filesystem::path path = manifest_path_();
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::FILE *fp = std::fopen(path.string().c_str(), "w");
    if (fp == nullptr) {
      throw std::runtime_error("snapshot series: cannot open '" + path.string() +
                               "'");
    }
    const auto n = domain::get_size(m_domain);
    const auto spacing = domain::get_spacing(m_domain);
    const auto origin = domain::get_origin(m_domain);
    std::fprintf(fp, "{\n  \"prefix\": \"%s\",\n", m_options.prefix.c_str());
    std::fprintf(fp, "  \"nx\": %d,\n  \"ny\": %d,\n  \"nz\": %d,\n", n[0], n[1],
                 n[2]);
    std::fprintf(fp, "  \"dx\": %.17g,\n  \"dy\": %.17g,\n  \"dz\": %.17g,\n",
                 spacing[0], spacing[1], spacing[2]);
    std::fprintf(fp, "  \"origin\": [%.17g, %.17g, %.17g],\n", origin[0], origin[1],
                 origin[2]);
    std::fprintf(fp, "  \"order\": \"fortran\",\n  \"dtype\": \"float64\",\n");
    std::fprintf(fp, "  \"fields\": [");
    for (std::size_t i = 0; i < m_slots.size(); ++i) {
      std::fprintf(fp, "%s\"%s\"", i ? ", " : "", m_slots[i].name.c_str());
    }
    std::fprintf(fp, "],\n  \"steps\": [");
    for (std::size_t i = 0; i < m_steps.size(); ++i) {
      std::fprintf(fp, "%s%d", i ? ", " : "", m_steps[i]);
    }
    std::fprintf(fp, "],\n  \"times\": [");
    for (std::size_t i = 0; i < m_times.size(); ++i) {
      std::fprintf(fp, "%s%.17g", i ? ", " : "", m_times[i]);
    }
    std::fprintf(fp, "],\n  \"patterns\": [");
    for (std::size_t i = 0; i < m_slots.size(); ++i) {
      std::fprintf(fp, "%s\"%s\"", i ? ", " : "", m_slots[i].pattern.c_str());
    }
    std::fprintf(fp, "]\n}\n");
    std::fclose(fp);
    write_xdmf_();
  }

  void write_xdmf_() const {
    if (m_times.empty()) return;
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> rels;
    for (const Slot &slot : m_slots) {
      if (slot.format != SnapshotFormat::Binary) continue;
      names.push_back(slot.name);
      std::vector<std::string> frames;
      frames.reserve(m_times.size());
      const auto parent = std::filesystem::path(slot.pattern).parent_path();
      for (int frame = 0; frame < m_index; ++frame) {
        const auto file =
            std::filesystem::path(utils::format_with_number(slot.pattern, frame));
        frames.push_back(parent.empty() ? file.string()
                                        : file.lexically_relative(parent).string());
      }
      rels.push_back(std::move(frames));
    }
    if (names.empty()) return;
    const auto n = domain::get_size(m_domain);
    const auto spacing = domain::get_spacing(m_domain);
    const double dz = n[2] > 1 ? spacing[2] : 1.0;
    pfc::io::write_xdmf_binary_series(xdmf_path_().string(), n[0], n[1], n[2],
                                      spacing[0], spacing[1], dz, names, rels,
                                      m_times);
  }

  [[nodiscard]] std::filesystem::path manifest_path_() const {
    if (!m_options.directory.empty() && !m_options.prefix.empty()) {
      return m_options.directory / (m_options.prefix + "_manifest.json");
    }
    const auto parent = std::filesystem::path(m_slots.front().pattern).parent_path();
    return (parent.empty() ? std::filesystem::path(".") : parent) /
           "snapshots_manifest.json";
  }

  [[nodiscard]] std::filesystem::path xdmf_path_() const {
    if (!m_options.directory.empty() && !m_options.prefix.empty()) {
      return m_options.directory / (m_options.prefix + ".xdmf");
    }
    const auto parent = std::filesystem::path(m_slots.front().pattern).parent_path();
    return (parent.empty() ? std::filesystem::path(".") : parent) / "snapshots.xdmf";
  }

  Domain m_domain;
  Box3i m_owned;
  SnapshotSeriesOptions m_options;
  int m_rank{0};
  int m_index{0};
  bool m_closed{false};
  std::vector<Slot> m_slots;
  std::vector<int> m_steps;
  std::vector<double> m_times;
  std::vector<double> m_packed;
  std::vector<std::string> m_bound;
};

} // namespace pfc::io

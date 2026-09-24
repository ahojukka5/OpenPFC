// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file field_output.hpp
 * @brief MPI-IO bricks of the design field `h` plus a JSON manifest.
 *
 * Text `--dump-h` is single-rank and too large for a 1024² showcase. These
 * bricks are Fortran-ordered doubles. Rank 0 also writes an XDMF sidecar
 * so ParaView can File → Open the series (`scripts/xdmfgen.py`).
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/frontend/io/xdmf_binary_series.hpp>
#include <openpfc/kernel/field/state_access.hpp>

namespace pfc::apps::inverse {

struct FieldOutputConfig {
  std::string dir;
  int every{1};
};

class FieldSnapshotWriter {
public:
  FieldSnapshotWriter(FieldOutputConfig cfg, std::string run_id,
                      const std::array<int, 3> &global,
                      const std::array<int, 3> &local,
                      const std::array<int, 3> &offset, double dx, int rank,
                      MPI_Comm comm)
      : m_cfg(std::move(cfg)), m_run(std::move(run_id)), m_global(global),
        m_local(local), m_offset(offset), m_dx(dx), m_rank(rank), m_comm(comm),
        m_count(static_cast<std::size_t>(local[0]) *
                static_cast<std::size_t>(local[1]) *
                static_cast<std::size_t>(local[2])) {
    m_buf.resize(m_count);
  }

  [[nodiscard]] bool active() const noexcept { return !m_cfg.dir.empty(); }

  void set_directory(std::string dir) { m_cfg.dir = std::move(dir); }

  void restore_steps(std::vector<int> steps) { m_steps = std::move(steps); }

  [[nodiscard]] const std::vector<int> &steps() const noexcept { return m_steps; }

  [[nodiscard]] bool due(int sample) const noexcept {
    return active() && (sample % std::max(1, m_cfg.every)) == 0;
  }

  template <typename FieldT>
  void write(const std::string &name, int idx, const FieldT &f) {
    if (!active()) return;
    fill_owned_(f);
    write_buffer_(name, idx);
  }

  /// MPI-IO brick with an exact filename (`h_final.bin`, `h_thresh.bin`).
  template <typename FieldT>
  void write_named(const std::string &filename, const FieldT &f) {
    if (!active()) return;
    fill_owned_(f);
    pfc::BinaryWriter w(m_cfg.dir + "/" + filename, m_comm);
    w.set_domain(m_global, m_local, m_offset);
    w.write(0, pfc::field::FieldView<double>(m_buf));
  }

  void note_step(int step) { m_steps.push_back(step); }

  void write_manifest(const std::vector<std::string> &fields) const {
    if (!active() || m_rank != 0) return;
    const std::string path = m_cfg.dir + "/" + m_run + "_manifest.json";
    std::FILE *fp = std::fopen(path.c_str(), "w");
    if (fp == nullptr) return;
    std::fprintf(fp, "{\n  \"run_id\": \"%s\",\n", m_run.c_str());
    std::fprintf(fp, "  \"nx\": %d,\n  \"ny\": %d,\n  \"nz\": %d,\n", m_global[0],
                 m_global[1], m_global[2]);
    std::fprintf(fp, "  \"dx\": %.17g,\n", m_dx);
    std::fprintf(fp, "  \"order\": \"fortran\",\n  \"dtype\": \"float64\",\n");
    std::fprintf(fp, "  \"fields\": [");
    for (std::size_t i = 0; i < fields.size(); ++i) {
      std::fprintf(fp, "%s\"%s\"", i ? ", " : "", fields[i].c_str());
    }
    std::fprintf(fp, "],\n  \"steps\": [");
    for (std::size_t i = 0; i < m_steps.size(); ++i) {
      std::fprintf(fp, "%s%d", i ? ", " : "", m_steps[i]);
    }
    std::fprintf(fp, "],\n  \"pattern\": \"%s_{field}_{index:04d}.bin\"\n}\n",
                 m_run.c_str());
    std::fclose(fp);
    write_xdmf_(fields);
  }

  void write_xdmf_brick(const std::string &filename, const std::string &bin_name,
                        const char *attr) const {
    if (!active() || m_rank != 0) return;
    const std::string path = m_cfg.dir + "/" + filename;
    std::FILE *fp = std::fopen(path.c_str(), "w");
    if (fp == nullptr) return;
    const int nx = m_global[0], ny = m_global[1], nz = m_global[2];
    std::fprintf(
        fp,
        "<?xml version=\"1.0\"?>\n<Xdmf Version=\"2.0\"><Domain>"
        "<Grid Name=\"g\" GridType=\"Uniform\">\n"
        "<Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"%d %d %d\"/>\n"
        "<Geometry GeometryType=\"ORIGIN_DXDYDZ\">"
        "<DataItem Dimensions=\"3\" Format=\"XML\">0 0 0</DataItem>"
        "<DataItem Dimensions=\"3\" Format=\"XML\">%.17g %.17g %.17g"
        "</DataItem></Geometry>\n"
        "<Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Node\">"
        "<DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
        "Precision=\"8\" Format=\"Binary\" Endian=\"Little\">%s"
        "</DataItem></Attribute>\n</Grid></Domain></Xdmf>\n",
        nz, ny, nx, m_dx, m_dx, (nz > 1 ? m_dx : 1.0), attr, nz, ny, nx, bin_name);
    std::fclose(fp);
  }

private:
  template <typename FieldT> void fill_owned_(const FieldT &f) {
    std::size_t n = 0;
    const_cast<FieldT &>(f).for_each_owned(
        [&](int i, int j, int k) { m_buf[n++] = f(i, j, k); });
  }

  void write_xdmf_(const std::vector<std::string> &fields) const {
    if (m_steps.empty() || fields.empty()) return;
    std::vector<double> times;
    times.reserve(m_steps.size());
    for (int s : m_steps) times.push_back(static_cast<double>(s));
    std::vector<std::vector<std::string>> rels(fields.size());
    for (std::size_t f = 0; f < fields.size(); ++f) {
      rels[f].resize(m_steps.size());
      for (std::size_t i = 0; i < m_steps.size(); ++i) {
        char tail[96];
        std::snprintf(tail, sizeof(tail), "%s_%s_%04zu.bin", m_run.c_str(),
                      fields[f].c_str(), i);
        rels[f][i] = tail;
      }
    }
    const double dz = m_global[2] > 1 ? m_dx : 1.0;
    // This writer does not store a physical origin. Pass zero until it
    // uses SnapshotSeries.
    pfc::io::write_xdmf_binary_series(
        m_cfg.dir + "/" + m_run + ".xdmf",
        pfc::io::BinarySeriesGeometry{.nx = m_global[0],
                                      .ny = m_global[1],
                                      .nz = m_global[2],
                                      .x0 = 0.0,
                                      .y0 = 0.0,
                                      .z0 = 0.0,
                                      .dx = m_dx,
                                      .dy = m_dx,
                                      .dz = dz},
        fields, rels, times);
  }

  void write_buffer_(const std::string &name, int idx) {
    char tail[64];
    std::snprintf(tail, sizeof(tail), "_%s_%04d.bin", name.c_str(), idx);
    pfc::BinaryWriter w(m_cfg.dir + "/" + m_run + tail, m_comm);
    w.set_domain(m_global, m_local, m_offset);
    w.write(0, pfc::field::FieldView<double>(m_buf));
  }

  FieldOutputConfig m_cfg;
  std::string m_run;
  std::array<int, 3> m_global{};
  std::array<int, 3> m_local{};
  std::array<int, 3> m_offset{};
  double m_dx{1.0};
  int m_rank{0};
  MPI_Comm m_comm{MPI_COMM_WORLD};
  std::size_t m_count{0};
  std::vector<double> m_buf;
  std::vector<int> m_steps;
};

} // namespace pfc::apps::inverse

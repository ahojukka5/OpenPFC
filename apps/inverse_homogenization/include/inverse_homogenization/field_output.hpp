// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file field_output.hpp
 * @brief MPI-IO bricks of the design field `h` plus a JSON manifest.
 *
 * Text `--dump-h` is single-rank and too large for a 1024² showcase. These
 * bricks are the same Fortran-ordered doubles `docs/report/figures/field_io.py`
 * already reads.
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/frontend/io/binary_writer.hpp>
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

  [[nodiscard]] bool due(int sample) const noexcept {
    return active() && (sample % std::max(1, m_cfg.every)) == 0;
  }

  template <typename FieldT>
  void write(const std::string &name, int idx, const FieldT &f) {
    std::size_t n = 0;
    const_cast<FieldT &>(f).for_each_owned(
        [&](int i, int j, int k) { m_buf[n++] = f(i, j, k); });
    write_buffer_(name, idx);
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
  }

private:
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

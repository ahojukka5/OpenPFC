// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file reduced_output.hpp
 * @brief Presentation-rate projections of 1D2V phase space.
 *
 * A full `f(x,vx,vy)` brick at animation cadence is the dominant I/O of a
 * Weibel showcase. The panels only need `int f dvy`, `int f dvx`, the 1-D
 * fields, and the ledger already in CSV. These helpers integrate on the
 * owned slab and complete the `v_y` sum with `MPI_Allreduce`.
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/frontend/io/xdmf_binary_series.hpp>
#include <openpfc/kernel/field/state_access.hpp>

#include <vlasov_maxwell/parameters.hpp>
#include <vlasov_maxwell/phase_space.hpp>
#include <vlasov_maxwell/step.hpp>

namespace vlasov {

struct ReducedOutputConfig {
  std::string dir;
  int every{1};
};

/// `f(x, v_x) = int f dv_y` on this rank (partial `v_y` slab).
inline void accumulate_f_xvx(const PhaseSpace &ps, const PhaseField &f,
                              std::vector<double> &local) {
  const int nx = ps.nx();
  const int nvx = ps.nvx();
  const int nvy = ps.nvy_local();
  const double dvy = ps.params().dvy();
  local.assign(static_cast<std::size_t>(nx) * static_cast<std::size_t>(nvx), 0.0);
  for (int k = 0; k < nvy; ++k)
    for (int j = 0; j < nvx; ++j)
      for (int i = 0; i < nx; ++i)
        local[static_cast<std::size_t>(i) +
              static_cast<std::size_t>(j) * static_cast<std::size_t>(nx)] +=
            f(i, j, k) * dvy;
}

/// `f(x, v_y) = int f dv_x` on this rank's `v_y` slab (complete in `x`).
inline void accumulate_f_xvy(const PhaseSpace &ps, const PhaseField &f,
                              std::vector<double> &local) {
  const int nx = ps.nx();
  const int nvx = ps.nvx();
  const int nvy = ps.nvy_local();
  const double dvx = ps.params().dvx();
  local.assign(static_cast<std::size_t>(nx) * static_cast<std::size_t>(nvy), 0.0);
  for (int k = 0; k < nvy; ++k)
    for (int j = 0; j < nvx; ++j)
      for (int i = 0; i < nx; ++i)
        local[static_cast<std::size_t>(i) +
              static_cast<std::size_t>(k) * static_cast<std::size_t>(nx)] +=
            f(i, j, k) * dvx;
}

class ReducedSnapshotWriter {
public:
  ReducedSnapshotWriter(ReducedOutputConfig cfg, std::string run_id,
                        const PhaseSpace &ps, int rank)
      : m_cfg(std::move(cfg)), m_run(std::move(run_id)), m_rank(rank),
        m_comm(ps.comm()), m_nx(ps.nx()), m_nvx(ps.nvx()), m_nvy(ps.nvy_global()),
        m_nvy_local(ps.nvy_local()), m_vy_off(ps.vy_offset()),
        m_dx(ps.params().dx()), m_dvx(ps.params().dvx()),
        m_dvy(ps.params().dvy()) {}

  [[nodiscard]] bool active() const noexcept { return !m_cfg.dir.empty(); }

  [[nodiscard]] bool due(int sample) const noexcept {
    return active() && (sample % std::max(1, m_cfg.every)) == 0;
  }

  void write(int idx, double t, const PhaseSpace &ps, const Stepper &st) {
    if (!active()) return;
    if (m_rank == 0) std::filesystem::create_directories(m_cfg.dir);
    MPI_Barrier(m_comm);

    std::vector<double> loc, glo;
    accumulate_f_xvx(ps, ps.f(0), loc);
    glo.assign(loc.size(), 0.0);
    MPI_Allreduce(loc.data(), glo.data(), static_cast<int>(loc.size()), MPI_DOUBLE,
                  MPI_SUM, m_comm);
    write_rank0_(name_("f_xvx", idx), glo);

    accumulate_f_xvy(ps, ps.f(0), loc);
    pfc::BinaryWriter w(m_cfg.dir + "/" + m_run + suffix_("f_xvy", idx), m_comm);
    w.set_domain({m_nx, m_nvy, 1}, {m_nx, m_nvy_local, 1}, {0, m_vy_off, 0});
    w.write(0, pfc::field::FieldView<double>(loc));

    write_rank0_(name_("Bz", idx), st.fields.Bz);
    write_rank0_(name_("Ey", idx), st.fields.Ey);
    write_rank0_(name_("Jy", idx), st.sources.Jy);
    m_times.push_back(t);
  }

  void write_manifest() const {
    if (!active() || m_rank != 0) return;
    const std::string path = m_cfg.dir + "/" + m_run + "_reduced_manifest.json";
    std::FILE *fp = std::fopen(path.c_str(), "w");
    if (fp == nullptr) return;
    std::fprintf(fp, "{\n  \"run_id\": \"%s\",\n", m_run.c_str());
    std::fprintf(fp, "  \"nx\": %d,\n  \"nvx\": %d,\n  \"nvy\": %d,\n", m_nx, m_nvx,
                 m_nvy);
    std::fprintf(fp, "  \"order\": \"fortran\",\n  \"dtype\": \"float64\",\n");
    std::fprintf(fp,
                 "  \"fields\": [\"f_xvx\", \"f_xvy\", \"Bz\", \"Ey\", \"Jy\"],\n");
    std::fprintf(fp, "  \"times\": [");
    for (std::size_t i = 0; i < m_times.size(); ++i)
      std::fprintf(fp, "%s%.10g", i ? ", " : "", m_times[i]);
    std::fprintf(fp, "],\n  \"pattern\": \"%s_{field}_{index:04d}.bin\"\n}\n",
                 m_run.c_str());
    std::fclose(fp);
    write_xdmf_();
  }

private:
  void write_xdmf_field_(const std::string &field, int nx, int ny, int nz,
                         double dx, double dy, double dz) const {
    if (m_times.empty()) return;
    std::vector<std::string> rels;
    rels.reserve(m_times.size());
    for (std::size_t i = 0; i < m_times.size(); ++i) {
      char tail[96];
      std::snprintf(tail, sizeof(tail), "%s_%s_%04zu.bin", m_run.c_str(),
                    field.c_str(), i);
      rels.push_back(tail);
    }
    pfc::io::write_xdmf_binary_series(
        m_cfg.dir + "/" + m_run + "_" + field + ".xdmf", nx, ny, nz, dx, dy, dz,
        {field}, {rels}, m_times);
  }

  void write_xdmf_() const {
    write_xdmf_field_("f_xvx", m_nx, m_nvx, 1, m_dx, m_dvx, 1.0);
    write_xdmf_field_("f_xvy", m_nx, m_nvy, 1, m_dx, m_dvy, 1.0);
    write_xdmf_field_("Bz", m_nx, 1, 1, m_dx, 1.0, 1.0);
    write_xdmf_field_("Ey", m_nx, 1, 1, m_dx, 1.0, 1.0);
    write_xdmf_field_("Jy", m_nx, 1, 1, m_dx, 1.0, 1.0);
  }

  std::string suffix_(const std::string &field, int idx) const {
    char tail[80];
    std::snprintf(tail, sizeof(tail), "_%s_%04d.bin", field.c_str(), idx);
    return tail;
  }
  std::string name_(const std::string &field, int idx) const {
    return m_cfg.dir + "/" + m_run + suffix_(field, idx);
  }
  void write_rank0_(const std::string &path, const std::vector<double> &v) const {
    if (m_rank != 0) return;
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(v.data()),
                static_cast<std::streamsize>(v.size() * sizeof(double)));
  }

  ReducedOutputConfig m_cfg;
  std::string m_run;
  int m_rank{0};
  MPI_Comm m_comm{MPI_COMM_WORLD};
  int m_nx{0}, m_nvx{0}, m_nvy{0}, m_nvy_local{0}, m_vy_off{0};
  double m_dx{1.0}, m_dvx{1.0}, m_dvy{1.0};
  std::vector<double> m_times;
};

} // namespace vlasov

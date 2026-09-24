// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file kawahara_session.hpp
 * @brief Kawahara = `KawaharaPhysics` on `SpectralETDSession`.
 *
 * Primary field is `u`. Orszag 2/3-rule dealiasing is on because \(u^2\)
 * is a quadratic product. Optional `diagnostics.csv` (CPU session only)
 * samples the wave-packet envelope centroid/width, carrier phase, and an
 * edge/no-wrap sentinel at every `saveat`; see `wave_packet_diagnostics.hpp`.
 */

#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include <openpfc/frontend/io/diagnostics_series.hpp>

#include <kawahara/gaussian_pulse.hpp>
#include <kawahara/kawahara_physics.hpp>
#include <kawahara/kdv_soliton.hpp>
#include <kawahara/wave_packet.hpp>
#include <kawahara/wave_packet_diagnostics.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/json_spectral_etd_session.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#endif

namespace kawahara {

inline void register_catalog() {
  pfc::ui::register_field_modifier<GaussianPulse>("gaussian_pulse");
  pfc::ui::register_field_modifier<KdVSoliton>("kdv_soliton");
  pfc::ui::register_field_modifier<WavePacket>("wave_packet");
}

inline pfc::sim::SpectralETDOptions etd_options() {
  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "u";
  opt.dealias = true;
  return opt;
}

/// CPU session with an optional collective wave-packet diagnostics CSV.
class KawaharaSession
    : public pfc::ui::SpectralETDSession<KawaharaPhysics<double, pfc::HostSpace>,
                                         pfc::sim::stacks::SpectralCPUStack> {
public:
  using Base = pfc::ui::SpectralETDSession<KawaharaPhysics<double, pfc::HostSpace>,
                                           pfc::sim::stacks::SpectralCPUStack>;

  KawaharaSession(const nlohmann::json &settings, int rank, int nproc,
                  MPI_Comm comm = MPI_COMM_WORLD)
      : Base(pfc::ui::copy_initial_offset_alias(settings, "u0"), rank, nproc, comm,
             etd_options()),
        m_comm(comm) {}

  void run() {
    if (!this->settings().contains("diagnostics")) {
      Base::run();
      return;
    }
    const auto &cfg = this->settings().at("diagnostics");
    const auto path = cfg.at("csv").template get<std::string>();
    if (path.empty())
      throw std::invalid_argument("diagnostics.csv must not be empty");
    const std::string kind = cfg.value("kind", std::string("wave_packet"));
    if (kind == "wave_packet") {
      run_with_wave_packet_diagnostics(cfg, path);
    } else if (kind == "pulse") {
      run_with_pulse_diagnostics(cfg, path);
    } else {
      throw std::invalid_argument(
          "diagnostics.kind must be 'wave_packet' or 'pulse'");
    }
  }

private:
  void run_with_wave_packet_diagnostics(const nlohmann::json &cfg,
                                        const std::string &path) {
    const double k0 = cfg.at("k0").template get<double>();
    const double cutoff = cfg.value("cutoff_factor", 0.5);
    pfc::io::DiagnosticsSeries csv(
        {"mean", "mode_amplitude", "mode_phase", "centroid", "width",
         "peak_envelope", "edge_fraction"},
        pfc::io::DiagnosticsSeriesOptions{.path = path, .comm = m_comm});
    WavePacketDiagnostics diagnostics(this->domain(), this->fft(), m_comm, k0,
                                      cutoff);
    auto save = [&] {
      const auto s = diagnostics.sample(this->psi());
      csv.write(pfc::time::increment(this->time()), pfc::time::current(this->time()),
                {s.mean, s.mode_amplitude, s.mode_phase, s.centroid,
                 std::sqrt(std::max(0.0, s.width2)), s.peak_envelope,
                 s.edge_fraction});
    };
    if (pfc::time::increment(this->time()) != 0 || pfc::time::done(this->time()))
      save();
    Base::run(save);
    csv.close();
  }

  void run_with_pulse_diagnostics(const nlohmann::json &cfg,
                                  const std::string &path) {
    const double window = cfg.at("window").template get<double>();
    pfc::io::DiagnosticsSeries csv(
        {"mean", "peak_amplitude", "peak_x", "tail_rms"},
        pfc::io::DiagnosticsSeriesOptions{.path = path, .comm = m_comm});
    PulseDiagnostics diagnostics(m_comm, window);
    auto save = [&] {
      const auto s = diagnostics.sample(this->psi());
      csv.write(pfc::time::increment(this->time()), pfc::time::current(this->time()),
                {s.mean, s.peak_amplitude, s.peak_x, s.tail_rms});
    };
    if (pfc::time::increment(this->time()) != 0 || pfc::time::done(this->time()))
      save();
    Base::run(save);
    csv.close();
  }

private:
  MPI_Comm m_comm;
};

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
class KawaharaHIPSession : public pfc::ui::SpectralETDSession<
                               KawaharaPhysics<double, pfc::HIPSpace>,
                               pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>> {
public:
  using Base =
      pfc::ui::SpectralETDSession<KawaharaPhysics<double, pfc::HIPSpace>,
                                  pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>;

  KawaharaHIPSession(const nlohmann::json &settings, int rank, int nproc,
                     MPI_Comm comm = MPI_COMM_WORLD)
      : Base(pfc::ui::copy_initial_offset_alias(settings, "u0"), rank, nproc, comm,
             etd_options()) {}
};
static_assert(pfc::sim::SpectralETDPhysics<KawaharaPhysics<double, pfc::HIPSpace>>);
#endif

} // namespace kawahara

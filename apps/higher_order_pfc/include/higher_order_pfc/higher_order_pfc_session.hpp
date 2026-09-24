// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file higher_order_pfc_session.hpp
 * @brief Higher-order PFC = `HigherOrderPFCPhysics` on `SpectralETDSession`.
 *
 * @details
 * Primary field is `psi`. Orszag 2/3-rule dealiasing is on because the local
 * term is cubic: \f$\psi^3\f$ triples the highest wave number present, so
 * without dealiasing the aliased content folds straight back onto the
 * \f$k\approx1\f$ band the kernel is trying to select.
 *
 * `register_catalog()` adds `lattice_seed` (the controlled single-crystal
 * seed used by the crystal-selection benchmark, `lattice_seed.hpp`).
 * `cosine_mode` and `seeded_noise` are built-in modifiers. Optional
 * `diagnostics.csv` samples free-
 * energy density, the reciprocal-space peak report and, single-rank only, the
 * real-space bond-orientational order parameter (`diagnostics.hpp`) -- the
 * `#118` crystal-selection observables. HIP builds also define
 * `HigherOrderPFCHIPSession` on `GPUSpectralStack<HIPSpace>` (no diagnostics;
 * that session is exercised by the HIP smoke test only).
 */

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <vector>

#include <openpfc/frontend/io/diagnostics_series.hpp>

#include <higher_order_pfc/diagnostics.hpp>
#include <higher_order_pfc/higher_order_pfc_physics.hpp>
#include <higher_order_pfc/lattice_seed.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/json_spectral_etd_session.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#endif

namespace higher_order_pfc {

inline void register_catalog() {
  pfc::ui::register_field_modifier<LatticeSeed>("lattice_seed");
}

inline pfc::sim::SpectralETDOptions etd_options() {
  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "psi";
  opt.dealias = true;
  return opt;
}

/// Adds the optional `diagnostics.csv` sampling hook to `SpectralETDSession`,
/// the same shape as `cahn_hilliard::DiagnosticSession`.
class HigherOrderPFCSession : public pfc::ui::SpectralETDSession<
                                  HigherOrderPFCPhysics<double, pfc::HostSpace>,
                                  pfc::sim::stacks::SpectralCPUStack> {
public:
  using Base =
      pfc::ui::SpectralETDSession<HigherOrderPFCPhysics<double, pfc::HostSpace>,
                                  pfc::sim::stacks::SpectralCPUStack>;

  HigherOrderPFCSession(const nlohmann::json &settings, int rank, int nproc,
                        MPI_Comm comm = MPI_COMM_WORLD)
      : Base(pfc::ui::copy_initial_offset_alias(settings, "psi0"), rank, nproc, comm,
             etd_options()),
        m_comm(comm) {}

  void run() {
    if (!settings().contains("diagnostics")) {
      Base::run();
      return;
    }
    const auto path =
        settings().at("diagnostics").at("csv").template get<std::string>();
    if (path.empty())
      throw std::invalid_argument("diagnostics.csv must not be empty");
    const int sf_bins = settings()["diagnostics"].value("sf_bins", 64);
    const double cutoff =
        settings()["diagnostics"].value("neighbour_cutoff_factor", 1.3);
    pfc::io::DiagnosticsSeries csv(
        {"mean_psi", "free_energy_density", "k1", "domain_length", "k_peak",
         "dominant_wavelength", "S_at_1", "S_at_q1", "psi4_global", "psi4_local",
         "psi6_global", "psi6_local", "n_peaks", "mean_neighbours"},
        pfc::io::DiagnosticsSeriesOptions{.path = path, .comm = m_comm});
    Diagnostics<pfc::HostSpace> diagnostics(domain(), fft(), m_comm);
    auto save = [&] {
      const auto s =
          diagnostics.sample(psi(), system().physics().params, sf_bins, cutoff);
      csv.write(pfc::time::increment(time()), pfc::time::current(time()),
                {s.mean_psi, s.free_energy_density, s.k1, s.domain_length, s.k_peak,
                 s.dominant_wavelength, s.S_at_1, s.S_at_q1, s.psi4_global,
                 s.psi4_local, s.psi6_global, s.psi6_local, s.n_peaks,
                 s.mean_neighbours});
    };
    // Restarts do not invoke the driver's initial-state callback.
    if (pfc::time::increment(time()) != 0 || pfc::time::done(time())) save();
    Base::run(save);
    csv.close();
  }

private:
  MPI_Comm m_comm;
};

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
class HigherOrderPFCHIPSession
    : public pfc::ui::SpectralETDSession<
          HigherOrderPFCPhysics<double, pfc::HIPSpace>,
          pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>> {
public:
  using Base =
      pfc::ui::SpectralETDSession<HigherOrderPFCPhysics<double, pfc::HIPSpace>,
                                  pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>;

  HigherOrderPFCHIPSession(const nlohmann::json &settings, int rank, int nproc,
                           MPI_Comm comm = MPI_COMM_WORLD)
      : Base(pfc::ui::copy_initial_offset_alias(settings, "psi0"), rank, nproc, comm,
             etd_options()) {}
};
static_assert(
    pfc::sim::SpectralETDPhysics<HigherOrderPFCPhysics<double, pfc::HIPSpace>>);
#endif

} // namespace higher_order_pfc

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file wave2d_fd_manual.cpp
 * @brief 2D wave equation — manual 5-point Laplacian on `pfc::data::Field` +
 * periodic halos in x,z and physical y-boundary ghosts (Dirichlet or Neumann).
 */

#include <cmath>
#include <cstdlib>
#include <memory>
#include <mpi.h>

#include <openpfc/frontend/io/snapshot_series.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/common/mpi_timer.hpp>
#include <vector>

#include <wave2d/cli.hpp>
#include <wave2d/reporting.hpp>
#include <wave2d/wave_boundary.hpp>
#include <wave2d/wave_model.hpp>

using namespace pfc;
using wave2d::RunConfig;
using wave2d::WaveModel;
using wave2d::YBoundaryKind;

namespace {

int run_fd_manual(const RunConfig &cfg, int rank, int nproc) {
  WaveModel model;
  model.inv_dx2 = 1.0;
  model.inv_dy2 = 1.0;

  const auto global_domain = pfc::domain::create(GridSize({cfg.Nx, cfg.Ny, 1}),
                                                 PhysicalOrigin({0.0, 0.0, 0.0}),
                                                 GridSpacing({1.0, 1.0, 1.0}));
  const auto decomp = decomposition::create(global_domain, nproc);

  constexpr int hw = 1;
  auto u = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  auto v = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  auto lap = pfc::data::field_from_subdomain<double>(decomp, rank, hw);

  comm::HaloExchange<HostSpace, double> halo_u(u, decomp, rank, MPI_COMM_WORLD);

  const double xc = 0.5 * static_cast<double>(cfg.Nx - 1);
  const double yc = 0.5 * static_cast<double>(cfg.Ny - 1);
  const double sigma = 0.12 * static_cast<double>(std::min(cfg.Nx, cfg.Ny));

  u.apply([&](double x, double y, double /*z*/) {
    const double dx = x - xc;
    const double dy = y - yc;
    return std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  });
  v.apply([](double, double, double) { return 0.0; });

  halo_u.exchange();
  wave2d::fill_y_physical_ghosts_padded(u, cfg.y_bc, cfg.Ny,
                                        static_cast<double>(cfg.u_wall));
  if (cfg.y_bc == YBoundaryKind::Dirichlet) {
    wave2d::enforce_dirichlet_y_walls_owned(u, v, cfg.Ny,
                                            static_cast<double>(cfg.u_wall));
  }

  pfc::io::SnapshotSeries snapshots(
      u.domain(), u.box(), pfc::io::SnapshotSeriesOptions{.comm = MPI_COMM_WORLD});
  if (!cfg.vtk_pattern.empty()) {
    snapshots.add_field("u", u, cfg.vtk_pattern, pfc::io::SnapshotFormat::Vtk);
    snapshots.set_cadence(pfc::io::SnapshotCadence(cfg.vtk_every));
    snapshots.write(0, 0.0);
  }

  auto stencil_lap = [&](int i, int j, int k) {
    const double lxx = u(i + 1, j, k) - 2.0 * u(i, j, k) + u(i - 1, j, k);
    const double lyy = u(i, j + 1, k) - 2.0 * u(i, j, k) + u(i, j - 1, k);
    lap(i, j, k) = model.inv_dx2 * lxx + model.inv_dy2 * lyy;
  };

  runtime::MPITimer timer{MPI_COMM_WORLD};
  runtime::tic(timer);
  for (int step = 0; step < cfg.n_steps; ++step) {
    halo_u.exchange();
    wave2d::fill_y_physical_ghosts_padded(u, cfg.y_bc, cfg.Ny,
                                          static_cast<double>(cfg.u_wall));
    u.for_each_owned([&](int i, int j, int k) { stencil_lap(i, j, k); });

    u.for_each_owned([&](int i, int j, int k) {
      const double v0 = v(i, j, k);
      const double l = lap(i, j, k);
      u(i, j, k) += cfg.dt * v0;
      v(i, j, k) += cfg.dt * wave2d::kC * wave2d::kC * l;
    });

    if (cfg.y_bc == YBoundaryKind::Dirichlet) {
      wave2d::enforce_dirichlet_y_walls_owned(u, v, cfg.Ny,
                                              static_cast<double>(cfg.u_wall));
    }

    if (!snapshots.empty()) {
      snapshots.write_if_due(step + 1, step + 1,
                             cfg.dt * static_cast<double>(step + 1));
    }
  }
  snapshots.close();
  const double max_elapsed = runtime::toc(timer);

  const bool observable_ok = wave2d::report(
      rank, nproc, cfg, "fd_manual", "manual 5-point + padded halos", max_elapsed,
      "(y physical BC; interior RMS u)", wave2d::interior_stats(u, hw));
  return observable_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg =
            wave2d::parse_manual_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) return EXIT_FAILURE;
        return run_fd_manual(*cfg, rank, nproc);
      });
}

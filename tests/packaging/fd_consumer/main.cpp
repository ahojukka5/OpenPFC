// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cstdio>

#include <mpi.h>

#include <openpfc/kernel/field/face_flux.hpp>
#include <openpfc/kernel/simulation/simulation_lifecycle.hpp>

int main() {
  int ready = 0;
  MPI_Initialized(&ready);
  if (ready == 0) {
    MPI_Init(nullptr, nullptr);
  }
  using pfc::field::fd::average_face;
  using pfc::field::fd::FaceAverage;
  const double arithmetic = average_face(FaceAverage::Arithmetic, 1.0, 2.0);
  const double harmonic = average_face(FaceAverage::Harmonic, 1.0, 1.0);
  pfc::sim::SimulationLifecycle life(
      pfc::sim::SimulationLifecycle::schedule(0.0, 0.2, 0.1, 0.2), MPI_COMM_WORLD);
  int accepted = 0;
  life.set_accepted_step_hook([&](const pfc::Time &) { ++accepted; });
  life.run([](double, double) {});
  const bool ok = arithmetic == 1.5 && harmonic == 1.0 && accepted == 2;
  if (ok) {
    std::printf("OpenPFC FD consumer OK: accepted %d\n", accepted);
  }
  int done = 0;
  MPI_Finalized(&done);
  if (ready == 0 && done == 0) {
    MPI_Finalize();
  }
  return ok ? 0 : 1;
}

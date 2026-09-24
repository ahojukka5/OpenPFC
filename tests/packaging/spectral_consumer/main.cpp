// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cstdio>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

int main() {
  int ready = 0;
  MPI_Initialized(&ready);
  if (ready == 0) {
    MPI_Init(nullptr, nullptr);
  }
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  auto domain = pfc::domain::create({8, 8, 8});
  pfc::sim::stacks::SpectralCPUStack stack(domain, rank, nproc, MPI_COMM_WORLD);
  const auto size = stack.geometry().size;
  const bool ok = size[0] == 8 && size[1] == 8 && size[2] == 8;
  if (ok && rank == 0) {
    std::printf("OpenPFC spectral consumer OK: domain %dx%dx%d\n", size[0], size[1],
                size[2]);
  }
  int done = 0;
  MPI_Finalized(&done);
  if (ready == 0 && done == 0) {
    MPI_Finalize();
  }
  return ok ? 0 : 1;
}

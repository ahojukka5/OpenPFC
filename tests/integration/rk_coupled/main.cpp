// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "host.hpp"
#include <cstdlib>
#include <iostream>
#include <new>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "allocation.hpp"
using allocation_observer::allocation_requests;
using allocation_observer::observe;
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int result = 0;
#if defined(_OPENMP)
  omp_set_num_threads(1);
#endif
  try {
    int ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    coupled_rk::require((ranks == 1 || ranks == 2 || ranks == 4 || ranks == 8),
                        "coupled RK regression supports1/2/4/8 ranks");
    coupled_rk::require(allocation_observer::verify_provider(),
                        "allocation provider did not observe known requests");
    for (bool fourth : {false, true}) {
      const double e = coupled_rk::verify_fd_stages(fourth, observe);
      coupled_rk::require(allocation_requests == 0,
                          "C++ allocation inside warmed production RK step");
      coupled_rk::verify_fd_time(fourth);
      std::cout << "matched bare FD6 stage error="
                << coupled_rk::verify_fd_stages(fourth, nullptr, true) << '\n';
      std::cout << (fourth ? "RK4" : "RK2") << " FD6 stage/state error=" << e
                << " successful C++ allocation_requests=" << allocation_requests
                << '\n';
#if defined(OpenPFC_ENABLE_HEFFTE)
      std::cout << (fourth ? "RK4" : "RK2") << " spectral stage/state error="
                << coupled_rk::verify_spectral(fourth, observe) << '\n';
#endif
      coupled_rk::require(allocation_requests == 0,
                          "C++ allocation inside warmed spectral RK");
    }
    const double coarse = coupled_rk::spatial_error(32),
                 fine = coupled_rk::spatial_error(64);
    coupled_rk::require(coarse / fine > 30, "FD6 spatial refinement order");
    std::cout << "matched bare FD6 spatial coarse=" << coarse << " fine=" << fine
              << " error ratio=" << coarse / fine << '\n';
  } catch (const std::exception &e) {
    observe(false);
    std::cerr << e.what() << '\n';
    result = 1;
  }
  MPI_Finalize();
  return result;
}

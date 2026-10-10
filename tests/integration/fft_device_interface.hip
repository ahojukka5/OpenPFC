// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_device_interface.hip
 * @brief Device FFT round trip from a translation unit compiled as HIP.
 *
 * @details
 * Every other HIP spectral test reaches `FFT_HIP` from a host `.cpp` compiled
 * by the C++ compiler. This one includes `gpu_spectral_stack.hpp` (and so
 * `fft_heffte_backend.hpp`) from a `.hip` file, which the HIP compiler (Clang)
 * compiles. Clang rejected the header before issue #373 was fixed, so building
 * this target is the regression; running it checks that the `IDeviceFFT`
 * overriders still dispatch to the HeFFTe transforms.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int failed = 0;
  {
    using namespace pfc;
    auto domain =
        domain::create(GridSize({16, 8, 4}), PhysicalOrigin({0.0, 0.0, 0.0}),
                       GridSpacing({1.0, 1.0, 1.0}));
    auto decomposition = decomposition::create(domain, size);
    fft::FFT_HIP fft = fft::create_hip(decomposition, rank, MPI_COMM_WORLD);
    fft::IDeviceFFT<HIPSpace> &iface = fft;

    std::vector<double> x(iface.size_inbox());
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double t = static_cast<double>(i);
      x[i] = 0.25 + 0.5 * std::cos(0.3 * t) + 0.125 * std::sin(0.7 * t);
    }
    fft::RealDataBufferHIP real_in(iface.size_inbox());
    fft::RealDataBufferHIP real_out(iface.size_inbox());
    fft::ComplexDataBufferHIP spectrum(iface.size_outbox());
    real_in.copy_from_host(x);

    iface.forward(real_in, spectrum);
    iface.backward(spectrum, real_out);

    const std::vector<double> back = real_out.to_host();
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (!(std::abs(back[i] - x[i]) < 1e-10)) {
        std::fprintf(stderr, "rank %d: index %zu: %.17g != %.17g\n", rank, i,
                     back[i], x[i]);
        failed = 1;
        break;
      }
    }
  }

  int any_failed = 0;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("HIP FFT round trip through IDeviceFFT: %s\n",
                any_failed ? "FAILED" : "passed");
  }
  MPI_Finalize();
  return any_failed;
}

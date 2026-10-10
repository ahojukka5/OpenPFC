// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fft_host_interface.cpp
 * @brief Host FFT overloads are reached through the `IHostFFT` interface.
 *
 * @details
 * `FFT_Impl` implements `IHostFFT::forward` and `IHostFFT::backward` as
 * virtual overriders. This translation unit includes the HeFFTe backend
 * header and calls the transform through `IHostFFT&`, so the overriders are
 * instantiated and dispatched virtually. It is also the Clang compile
 * regression for issue #373: Clang rejects a virtual function that carries a
 * trailing requires-clause, while GCC accepts it.
 *
 * The oracle is the round trip itself. `backward` scales by `1/N`, so
 * `backward(forward(x))` must reproduce `x`, and the interface result must
 * match the concrete-type result exactly.
 */

#include <complex>
#include <cstddef>
#include <vector>

#include <mpi.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/detail/fft_heffte_backend.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>

using namespace pfc;

TEST_CASE("host FFT round trip through IHostFFT matches the concrete type",
          "[fft][unit][interface]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    SKIP("single-rank interface check");
  }

  const Int3 n{16, 8, 4};
  auto domain = domain::create(GridSize(n), PhysicalOrigin({0.0, 0.0, 0.0}),
                               GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(domain, 1);
  auto plan = fft::create(decomposition);
  fft::IHostFFT &iface = plan;

  const std::size_t n_real = iface.size_inbox();
  const std::size_t n_complex = iface.size_outbox();
  REQUIRE(n_real == static_cast<std::size_t>(n[0] * n[1] * n[2]));

  std::vector<double> x(n_real);
  for (std::size_t i = 0; i < n_real; ++i) {
    const double t = static_cast<double>(i);
    x[i] = 0.4 * std::cos(0.23 * t) + 0.1 * std::sin(0.61 * t) + 0.25;
  }

  std::vector<std::complex<double>> spectrum_iface(n_complex);
  std::vector<std::complex<double>> spectrum_concrete(n_complex);
  iface.forward(x, spectrum_iface);
  plan.forward(x, spectrum_concrete);

  for (std::size_t i = 0; i < n_complex; ++i) {
    INFO("index " << i);
    REQUIRE(spectrum_iface[i].real() == spectrum_concrete[i].real());
    REQUIRE(spectrum_iface[i].imag() == spectrum_concrete[i].imag());
  }

  std::vector<double> back(n_real, 0.0);
  iface.backward(spectrum_iface, back);
  for (std::size_t i = 0; i < n_real; ++i) {
    INFO("index " << i);
    REQUIRE_THAT(back[i], Catch::Matchers::WithinAbs(x[i], 1e-12));
  }
}

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fft_interface_overriders.cpp
 * @brief `FFT_Impl` overrides the `IHostFFT` and `IDeviceFFT` transforms.
 *
 * @details
 * Calls the transforms through the interface and compares them with the
 * same calls on the concrete type. This also keeps a translation unit that
 * instantiates the host `FFT_Impl` in every compiler's test build, so a
 * declaration that only GCC accepts (issue #373: a trailing requires-clause on
 * a virtual overrider) fails the Clang build.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <heffte.h>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>

using namespace pfc;

static_assert(std::is_base_of_v<fft::IHostFFT, fft::CPUFFT>);
static_assert(!std::is_abstract_v<fft::CPUFFT>);

TEST_CASE("FFT_Impl host transforms dispatch through IHostFFT",
          "[fft][unit][interface]") {
  auto domain = domain::create(GridSize({8, 4, 2}), PhysicalOrigin({0.0, 0.0, 0.0}),
                               GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(domain, 1);
  auto fft = fft::create(decomposition);
  fft::IHostFFT &iface = fft;

  const std::size_t n_real = iface.size_inbox();
  const std::size_t n_complex = iface.size_outbox();
  REQUIRE(n_real == 8 * 4 * 2);

  std::vector<double> x(n_real);
  for (std::size_t i = 0; i < n_real; ++i) {
    const double t = static_cast<double>(i);
    x[i] = 0.25 + 0.5 * std::cos(0.3 * t) + 0.125 * std::sin(0.7 * t);
  }

  std::vector<std::complex<double>> spectrum(n_complex);
  std::vector<std::complex<double>> spectrum_concrete(n_complex);
  iface.forward(x, spectrum);
  fft.forward(x, spectrum_concrete);
  REQUIRE(spectrum == spectrum_concrete);

  std::vector<double> back(n_real);
  iface.backward(spectrum, back);
  for (std::size_t i = 0; i < n_real; ++i) {
    REQUIRE(std::abs(back[i] - x[i]) < 1e-12);
  }

  std::vector<double> short_real(n_real - 1);
  REQUIRE_THROWS_AS(iface.forward(short_real, spectrum), std::invalid_argument);
  REQUIRE_THROWS_AS(iface.backward(spectrum, short_real), std::invalid_argument);
}

// The `IDeviceFFT` overriders on the host: HeFFTe's built-in `stock` backend
// is not FFTW, so `FFT_Impl` takes its device (`HeapBackend`) path, and
// `IDeviceFFT<HostSpace>` gives it host `DataBuffer`s. That keeps the device
// specialization of `FFTInterfaceTransforms` in every compiler's test build;
// before #373 was fixed Clang rejected it here too. FFTW on the same input is
// an independent transform to compare against.
using HostDeviceFFT =
    fft::FFT_Impl<heffte::backend::stock, fft::IDeviceFFT<HostSpace>>;
static_assert(std::is_base_of_v<fft::IDeviceFFT<HostSpace>, HostDeviceFFT>);
static_assert(!std::is_base_of_v<fft::IHostFFT, HostDeviceFFT>);
static_assert(!std::is_abstract_v<HostDeviceFFT>);

TEST_CASE("FFT_Impl device transforms dispatch through IDeviceFFT",
          "[fft][unit][interface]") {
  const heffte::box3d<> inbox({0, 0, 0}, {7, 3, 1});
  const heffte::box3d<> outbox({0, 0, 0}, {4, 3, 1});
  HostDeviceFFT device(
      heffte::fft3d_r2c<heffte::backend::stock>(inbox, outbox, 0, MPI_COMM_SELF));
  fft::CPUFFT host(
      heffte::fft3d_r2c<heffte::backend::fftw>(inbox, outbox, 0, MPI_COMM_SELF));
  fft::IDeviceFFT<HostSpace> &iface = device;
  using RealBuffer = fft::IDeviceFFT<HostSpace>::RealBuffer;
  using ComplexBuffer = fft::IDeviceFFT<HostSpace>::ComplexBuffer;

  const std::size_t n_real = iface.size_inbox();
  const std::size_t n_complex = iface.size_outbox();
  REQUIRE(n_real == 8 * 4 * 2);
  REQUIRE(n_complex == host.size_outbox());

  std::vector<double> x(n_real);
  RealBuffer in(n_real);
  for (std::size_t i = 0; i < n_real; ++i) {
    const double t = static_cast<double>(i);
    x[i] = 0.25 + 0.5 * std::cos(0.3 * t) + 0.125 * std::sin(0.7 * t);
    in.data()[i] = x[i];
  }

  ComplexBuffer spectrum(n_complex);
  ComplexBuffer spectrum_concrete(n_complex);
  iface.forward(in, spectrum);
  device.forward(in, spectrum_concrete);
  std::vector<std::complex<double>> spectrum_fftw(n_complex);
  host.forward(x, spectrum_fftw);
  for (std::size_t i = 0; i < n_complex; ++i) {
    REQUIRE(spectrum.data()[i] == spectrum_concrete.data()[i]);
    REQUIRE(std::abs(spectrum.data()[i] - spectrum_fftw[i]) < 1e-12);
  }

  RealBuffer back(n_real);
  iface.backward(spectrum, back);
  for (std::size_t i = 0; i < n_real; ++i) {
    REQUIRE(in.data()[i] == x[i]);
    REQUIRE(std::abs(back.data()[i] - x[i]) < 1e-12);
  }

  RealBuffer short_real(n_real - 1);
  REQUIRE_THROWS_AS(iface.forward(short_real, spectrum), std::invalid_argument);
  REQUIRE_THROWS_AS(iface.backward(spectrum, short_real), std::invalid_argument);
}

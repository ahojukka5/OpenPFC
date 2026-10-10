// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fft_interface_overriders.cpp
 * @brief `FFT_Impl` overrides the `IHostFFT` transforms.
 *
 * @details
 * Calls the host transforms through `IHostFFT&` and compares them with the
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

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
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

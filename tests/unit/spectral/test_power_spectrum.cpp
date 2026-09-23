// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>

#include <openpfc/spectral/power_spectrum.hpp>

#ifdef OpenPFC_ENABLE_HEFFTE
#include <complex>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#endif

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("r2c multiplicity is 1 on the real axis and the even Nyquist mode",
          "[spectral][power]") {
  static_assert(pfc::spectral::r2c_multiplicity(0, 64) == 1.0);
  static_assert(pfc::spectral::r2c_multiplicity(32, 64) == 1.0);
  static_assert(pfc::spectral::r2c_multiplicity(1, 64) == 2.0);
  static_assert(pfc::spectral::r2c_multiplicity(31, 63) == 2.0);
}

TEST_CASE("power_near breaks an exact-distance tie towards the occupied shell",
          "[spectral][power]") {
  pfc::spectral::RadialSpectrum spectrum;
  spectrum.wavenumber = {0.5, 1.5};
  spectrum.power = {0.0, 42.0};
  REQUIRE_THAT(pfc::spectral::power_near(spectrum, 1.0), WithinAbs(42.0, 1e-15));
  spectrum.power = {42.0, 0.0};
  REQUIRE_THAT(pfc::spectral::power_near(spectrum, 1.0), WithinAbs(42.0, 1e-15));
}

#ifdef OpenPFC_ENABLE_HEFFTE
namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

} // namespace

TEST_CASE("A single cosine lands in the shell of its wave number",
          "[spectral][power]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral check");
  }
  constexpr int n = 64;
  constexpr int nx = 6;
  const auto domain = pfc::domain::create(pfc::GridSize({n, n, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto &field = stack.u();
  const double twopi = 2.0 * std::numbers::pi;
  const double length = static_cast<double>(n);
  field.apply([&](double x, double, double) {
    return 0.32 + 0.01 * std::cos(twopi * nx * x / length);
  });

  pfc::data::Field<std::complex<double>> hat(domain, stack.fft().get_outbox_bounds(),
                                             0);
  pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), field, hat);
  pfc::spectral::RadialSpectrum spectrum;
  hat.with_host_view([&](std::complex<double> *values, std::size_t) {
    spectrum = pfc::spectral::radial_average(stack.fft().get_outbox_bounds(), domain,
                                             values, MPI_COMM_WORLD, 64);
  });

  const double k_expected = twopi * nx / length;
  const double bin = std::numbers::pi / 64.0;
  REQUIRE(spectrum.total_power > 0.0);
  REQUIRE_THAT(spectrum.peak_wavenumber, WithinAbs(k_expected, bin));
  REQUIRE_THAT(spectrum.dominant_wavelength(), WithinRel(length / nx, 0.1));
  REQUIRE_THAT(spectrum.first_moment, WithinAbs(k_expected, 2 * bin));
  REQUIRE(spectrum.mean_wavelength() > 0.0);
}

TEST_CASE("Equal x and y ridges carry equal directional power",
          "[spectral][power]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral check");
  }
  constexpr int n = 64;
  constexpr int mode = 4;
  const auto domain = pfc::domain::create(pfc::GridSize({n, n, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  const double twopi = 2.0 * std::numbers::pi;
  const double length = static_cast<double>(n);

  auto transform = [&](auto &&fill) {
    auto &field = stack.u();
    field.apply(fill);
    pfc::data::Field<std::complex<double>> hat(domain,
                                               stack.fft().get_outbox_bounds(), 0);
    pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), field, hat);
    pfc::spectral::DirectionalPower bins;
    hat.with_host_view([&](std::complex<double> *values, std::size_t) {
      bins = pfc::spectral::directional_power(stack.fft().get_outbox_bounds(),
                                              domain, values, MPI_COMM_WORLD);
    });
    return bins;
  };

  const auto along_x = transform(
      [&](double x, double, double) { return std::cos(twopi * mode * x / length); });
  const auto along_y = transform(
      [&](double, double y, double) { return std::cos(twopi * mode * y / length); });
  REQUIRE(along_x.along_x > 0.0);
  REQUIRE(along_y.along_y > 0.0);
  REQUIRE_THAT(along_x.along_x, WithinRel(along_y.along_y, 1e-12));
  REQUIRE_THAT(along_x.along_y, WithinAbs(0.0, 1e-18));
  REQUIRE_THAT(along_y.along_x, WithinAbs(0.0, 1e-18));
}

TEST_CASE("A z cosine is reported on the z axis", "[spectral][power]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral check");
  }
  constexpr int n = 32;
  constexpr int mode = 3;
  const auto domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto &field = stack.u();
  const double twopi = 2.0 * std::numbers::pi;
  const double length = static_cast<double>(n);
  field.apply(
      [&](double, double, double z) { return std::cos(twopi * mode * z / length); });
  pfc::data::Field<std::complex<double>> hat(domain, stack.fft().get_outbox_bounds(),
                                             0);
  pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), field, hat);
  pfc::spectral::DirectionalPower bins;
  pfc::spectral::RadialSpectrum spectrum;
  hat.with_host_view([&](std::complex<double> *values, std::size_t) {
    bins = pfc::spectral::directional_power(stack.fft().get_outbox_bounds(), domain,
                                            values, MPI_COMM_WORLD);
    spectrum = pfc::spectral::radial_average(stack.fft().get_outbox_bounds(), domain,
                                             values, MPI_COMM_WORLD, 32);
  });
  const double k_expected = twopi * mode / length;
  const double bin = std::numbers::pi / 32.0;
  REQUIRE(bins.along_z > bins.along_x);
  REQUIRE(bins.along_z > bins.along_y);
  REQUIRE_THAT(spectrum.peak_wavenumber, WithinAbs(k_expected, bin));
}
#endif

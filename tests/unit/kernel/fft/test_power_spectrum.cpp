// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>

#include <openpfc/kernel/fft/power_spectrum.hpp>

#ifdef OpenPFC_ENABLE_HEFFTE
#include <complex>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#endif

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("r2c multiplicity is 1 on the real axis and the even Nyquist mode",
          "[fft][power_spectrum]") {
  static_assert(pfc::fft::r2c_multiplicity(0, 64) == 1.0);
  static_assert(pfc::fft::r2c_multiplicity(32, 64) == 1.0);
  static_assert(pfc::fft::r2c_multiplicity(1, 64) == 2.0);
  static_assert(pfc::fft::r2c_multiplicity(31, 64) == 2.0);
  static_assert(pfc::fft::r2c_multiplicity(31, 63) == 2.0);
}

TEST_CASE("power_near breaks an exact-distance tie towards the occupied shell",
          "[fft][power_spectrum]") {
  pfc::fft::RadialSpectrum spectrum;
  spectrum.wavenumber = {0.5, 1.5};
  spectrum.mean_power = {0.0, 42.0};
  REQUIRE_THAT(pfc::fft::power_near(spectrum, 1.0), WithinAbs(42.0, 1e-15));
  spectrum.mean_power = {42.0, 0.0};
  REQUIRE_THAT(pfc::fft::power_near(spectrum, 1.0), WithinAbs(42.0, 1e-15));
}

#ifdef OpenPFC_ENABLE_HEFFTE
namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

double real_sum_squares(pfc::data::Field<double> &field, MPI_Comm comm) {
  double local = 0.0;
  field.with_host_view([&](double *values, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) local += values[i] * values[i];
  });
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global;
}

/// `|hat|^2` with the Nyquist mode counted twice. That is the error
/// `r2c_multiplicity` refuses.
double power_treating_nyquist_as_a_pair(const pfc::Box3i &outbox,
                                        const pfc::Domain &domain,
                                        const std::complex<double> *spectrum,
                                        MPI_Comm comm) {
  double local = 0.0;
  pfc::fft::kspace::for_each_kpoint(
      outbox, domain, [&](std::size_t i, double, double, double, int ix, int, int) {
        const double weight = (ix == 0) ? 1.0 : 2.0;
        local += weight * std::norm(spectrum[i]);
      });
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global;
}

template <class Fill, class Use>
void with_spectrum(const pfc::Domain &domain, int rank, int nproc, Fill &&fill,
                   Use &&use) {
  pfc::sim::stacks::SpectralCPUStack stack(domain, rank, nproc, MPI_COMM_WORLD);
  pfc::data::Field<std::complex<double>> hat(domain, stack.fft().get_outbox_bounds(),
                                             0);
  stack.u().apply(fill);
  pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), stack.u(), hat);
  hat.with_host_view(
      [&](std::complex<double> *values, std::size_t) { use(stack, values); });
}

} // namespace

TEST_CASE("Weighted power matches the real-space sum of squares",
          "[fft][power_spectrum]") {
  if (world_size() != 1) SKIP("single-rank spectral check");
  constexpr int n = 32;
  const auto domain = pfc::domain::create(pfc::GridSize({n, n, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  const double twopi = 2.0 * std::numbers::pi;
  const double length = static_cast<double>(n);
  with_spectrum(
      domain, 0, 1,
      [&](double x, double y, double) {
        return 0.3 + 0.5 * std::cos(twopi * 3.0 * x / length) +
               0.2 * std::cos(std::numbers::pi * x) +
               0.4 * std::cos(twopi * 2.0 * y / length);
      },
      [&](auto &stack, std::complex<double> *hat) {
        const double spectral = pfc::fft::weighted_power(
            stack.fft().get_outbox_bounds(), domain, hat, MPI_COMM_WORLD);
        const double samples = static_cast<double>(n) * n;
        const double real_power = real_sum_squares(stack.u(), MPI_COMM_WORLD);
        REQUIRE_THAT(spectral / samples, WithinRel(real_power, 1.0e-12));
      });
}

TEST_CASE("A Nyquist cosine is its own conjugate", "[fft][power_spectrum]") {
  if (world_size() != 1) SKIP("single-rank spectral check");
  constexpr int n = 16;
  const auto domain = pfc::domain::create(pfc::GridSize({n, 8, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  with_spectrum(
      domain, 0, 1,
      [](double x, double, double) { return std::cos(std::numbers::pi * x); },
      [&](auto &stack, std::complex<double> *hat) {
        const auto outbox = stack.fft().get_outbox_bounds();
        const double spectral =
            pfc::fft::weighted_power(outbox, domain, hat, MPI_COMM_WORLD);
        const double doubled =
            power_treating_nyquist_as_a_pair(outbox, domain, hat, MPI_COMM_WORLD);
        const double samples = static_cast<double>(n) * 8.0;
        const double real_power = real_sum_squares(stack.u(), MPI_COMM_WORLD);
        REQUIRE_THAT(spectral / samples, WithinRel(real_power, 1.0e-12));
        REQUIRE(std::abs(doubled / samples - real_power) > 0.1 * real_power);
      });
}

TEST_CASE("A constant field sits entirely in the zero mode",
          "[fft][power_spectrum]") {
  if (world_size() != 1) SKIP("single-rank spectral check");
  constexpr int n = 8;
  const auto domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  with_spectrum(
      domain, 0, 1, [](double, double, double) { return 1.25; },
      [&](auto &stack, std::complex<double> *hat) {
        const auto outbox = stack.fft().get_outbox_bounds();
        const double spectral =
            pfc::fft::weighted_power(outbox, domain, hat, MPI_COMM_WORLD);
        const auto axes =
            pfc::fft::directional_power(outbox, domain, hat, MPI_COMM_WORLD);
        const auto spectrum =
            pfc::fft::radial_average(outbox, domain, hat, MPI_COMM_WORLD, 8);
        const double samples = static_cast<double>(n) * n * n;
        const double real_power = real_sum_squares(stack.u(), MPI_COMM_WORLD);
        REQUIRE_THAT(spectral / samples, WithinRel(real_power, 1.0e-12));
        REQUIRE_THAT(axes.sum(), WithinAbs(0.0, 1.0e-8 * spectral));
        // k = 0 is omitted. Roundoff may still open a shell; its integrated
        // power has to be negligible next to the zero mode.
        REQUIRE_THAT(spectrum.total_power, WithinAbs(0.0, 1.0e-8 * spectral));
      });
}

TEST_CASE("Directional bins omit only the zero mode", "[fft][power_spectrum]") {
  if (world_size() != 1) SKIP("single-rank spectral check");
  constexpr int n = 16;
  const auto domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  const double twopi = 2.0 * std::numbers::pi;
  const double length = static_cast<double>(n);
  with_spectrum(
      domain, 0, 1,
      [&](double x, double y, double z) {
        return 0.1 + std::cos(twopi * 2.0 * x / length) +
               0.5 * std::cos(twopi * 3.0 * y / length) +
               0.25 * std::cos(twopi * z / length);
      },
      [&](auto &stack, std::complex<double> *hat) {
        const auto outbox = stack.fft().get_outbox_bounds();
        const double spectral =
            pfc::fft::weighted_power(outbox, domain, hat, MPI_COMM_WORLD);
        const auto axes =
            pfc::fft::directional_power(outbox, domain, hat, MPI_COMM_WORLD);
        const double samples = length * length * length;
        const double real_power = real_sum_squares(stack.u(), MPI_COMM_WORLD);
        REQUIRE_THAT(spectral / samples, WithinRel(real_power, 1.0e-12));
        REQUIRE(axes.along_x > 0.0);
        REQUIRE(axes.along_y > 0.0);
        REQUIRE(axes.along_z > 0.0);
        REQUIRE(spectral > axes.sum());
      });
}

TEST_CASE("A single cosine lands in the shell of its wave number",
          "[fft][power_spectrum]") {
  if (world_size() != 1) SKIP("single-rank spectral check");
  constexpr int n = 64;
  constexpr int nx = 6;
  const auto domain = pfc::domain::create(pfc::GridSize({n, n, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  const double twopi = 2.0 * std::numbers::pi;
  const double length = static_cast<double>(n);
  with_spectrum(
      domain, 0, 1,
      [&](double x, double, double) {
        return 0.32 + 0.01 * std::cos(twopi * nx * x / length);
      },
      [&](auto &stack, std::complex<double> *hat) {
        const auto spectrum = pfc::fft::radial_average(
            stack.fft().get_outbox_bounds(), domain, hat, MPI_COMM_WORLD, 64);
        const double k_expected = twopi * nx / length;
        const double bin = std::numbers::pi / 64.0;
        REQUIRE(spectrum.total_power > 0.0);
        REQUIRE(spectrum.integrated_power.size() == spectrum.mean_power.size());
        REQUIRE_THAT(spectrum.peak_wavenumber, WithinAbs(k_expected, bin));
        REQUIRE_THAT(spectrum.dominant_wavelength(), WithinRel(length / nx, 0.1));
        REQUIRE_THAT(spectrum.first_moment, WithinAbs(k_expected, 2 * bin));
        REQUIRE(spectrum.mean_wavelength() > 0.0);
      });
}

TEST_CASE("Equal x and y ridges carry equal directional power",
          "[fft][power_spectrum]") {
  if (world_size() != 1) SKIP("single-rank spectral check");
  constexpr int n = 64;
  constexpr int mode = 4;
  const auto domain = pfc::domain::create(pfc::GridSize({n, n, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  const double twopi = 2.0 * std::numbers::pi;
  const double length = static_cast<double>(n);
  auto ridge = [&](bool horizontal) {
    pfc::fft::DirectionalPower bins;
    with_spectrum(
        domain, 0, 1,
        [&](double x, double y, double) {
          const double s = horizontal ? x : y;
          return std::cos(twopi * mode * s / length);
        },
        [&](auto &stack, std::complex<double> *hat) {
          bins = pfc::fft::directional_power(stack.fft().get_outbox_bounds(), domain,
                                             hat, MPI_COMM_WORLD);
        });
    return bins;
  };
  const auto x_ridge = ridge(true);
  const auto y_ridge = ridge(false);
  REQUIRE(x_ridge.along_x > 0.0);
  REQUIRE(y_ridge.along_y > 0.0);
  REQUIRE_THAT(x_ridge.along_x, WithinRel(y_ridge.along_y, 1e-12));
  REQUIRE_THAT(x_ridge.along_y, WithinAbs(0.0, 1e-18));
  REQUIRE_THAT(y_ridge.along_x, WithinAbs(0.0, 1e-18));
}

TEST_CASE("A z cosine is reported on the z axis", "[fft][power_spectrum]") {
  if (world_size() != 1) SKIP("single-rank spectral check");
  constexpr int n = 32;
  constexpr int mode = 3;
  const auto domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  const double twopi = 2.0 * std::numbers::pi;
  const double length = static_cast<double>(n);
  with_spectrum(
      domain, 0, 1,
      [&](double, double, double z) { return std::cos(twopi * mode * z / length); },
      [&](auto &stack, std::complex<double> *hat) {
        const auto outbox = stack.fft().get_outbox_bounds();
        const auto bins =
            pfc::fft::directional_power(outbox, domain, hat, MPI_COMM_WORLD);
        const auto spectrum =
            pfc::fft::radial_average(outbox, domain, hat, MPI_COMM_WORLD, 32);
        const double k_expected = twopi * mode / length;
        const double bin = std::numbers::pi / 32.0;
        REQUIRE(bins.along_z > bins.along_x);
        REQUIRE(bins.along_z > bins.along_y);
        REQUIRE_THAT(spectrum.peak_wavenumber, WithinAbs(k_expected, bin));
      });
}

TEST_CASE("Weighted power is the same sum on a two-rank split",
          "[fft][power_spectrum][MPI]") {
  if (world_size() != 2) SKIP("two-rank reduction");
  constexpr int n = 32;
  const auto domain = pfc::domain::create(pfc::GridSize({n, n, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  const double twopi = 2.0 * std::numbers::pi;
  const double length = static_cast<double>(n);
  with_spectrum(
      domain, world_rank(), 2,
      [&](double x, double y, double) {
        return 0.2 + 0.7 * std::cos(twopi * 5.0 * x / length) +
               0.3 * std::cos(std::numbers::pi * x) * std::cos(twopi * y / length);
      },
      [&](auto &stack, std::complex<double> *hat) {
        const double spectral = pfc::fft::weighted_power(
            stack.fft().get_outbox_bounds(), domain, hat, MPI_COMM_WORLD);
        const double real_power = real_sum_squares(stack.u(), MPI_COMM_WORLD);
        REQUIRE_THAT(spectral / (length * length), WithinRel(real_power, 1.0e-12));
      });
}
#endif

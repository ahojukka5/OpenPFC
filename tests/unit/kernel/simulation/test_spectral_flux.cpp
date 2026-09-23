// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Host checks for `pfc::sim::SpectralFlux`: a constant coefficient reduces
 * to the Laplacian symbol, a variable coefficient matches the product rule
 * on resolved modes, and the zero Fourier mode of a divergence stays zero.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>
#include <openpfc/kernel/simulation/spectral_flux.hpp>

using pfc::data::Field;
using pfc::sim::AsStored;
using pfc::sim::FluxETD;
using pfc::sim::SpectralETDOps;
using pfc::sim::SpectralFlux;

namespace {

constexpr int N = 32;

struct ScaleMobility {
  double value{1.0};
  double operator()(double) const { return value; }
};

pfc::Domain film_domain() {
  return pfc::domain::create(pfc::GridSize({N, N, 1}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
}

double max_abs_against(const Field<double> &got,
                       const std::vector<double> &expected) {
  REQUIRE(got.size() == expected.size());
  double m = 0.0;
  const double *data = got.data();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    m = std::max(m, std::abs(data[i] - expected[i]));
  }
  return m;
}

std::vector<double> sample(const Field<double> &grid, auto &&fn) {
  std::vector<double> out(grid.size());
  const auto box = grid.box();
  for (int k = 0; k < box.size[2]; ++k) {
    for (int j = 0; j < box.size[1]; ++j) {
      for (int i = 0; i < box.size[0]; ++i) {
        const auto c = grid.coords(i, j, k);
        out[static_cast<std::size_t>(grid.idx(i, j, k))] = fn(c[0], c[1]);
      }
    }
  }
  return out;
}

double zero_mode_abs(pfc::fft::IHostFFT &fft, const pfc::Domain &domain,
                     const Field<std::complex<double>> &hat) {
  std::size_t iz = 0;
  bool found = false;
  pfc::fft::kspace::for_each_kpoint(
      fft.get_outbox_bounds(), domain,
      [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
        if (kx == 0.0 && ky == 0.0 && kz == 0.0) {
          iz = i;
          found = true;
        }
      });
  REQUIRE(found);
  return std::abs(hat.data()[iz]);
}

} // namespace

TEST_CASE("constant mobility reduces to the Laplacian symbol", "[spectral_flux]") {
  auto domain = film_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  const double lx = static_cast<double>(N);
  const double k = 2.0 * pfc::pi / lx;

  Field<double> potential(domain, fft.get_inbox_bounds(), 0);
  potential.apply([k](double x, double, double) { return std::cos(k * x); });
  Field<double> state(domain, fft.get_inbox_bounds(), 0);
  state.apply([](double, double, double) { return 1.0; });

  Field<std::complex<double>> p_hat(domain, fft.get_outbox_bounds(), 0);
  Field<std::complex<double>> out_hat(domain, fft.get_outbox_bounds(), 0);
  SpectralETDOps<pfc::HostSpace>::forward(fft, potential, p_hat);

  SpectralFlux<> flux(domain, fft);
  constexpr double m = 4.0;
  flux.divergence(p_hat, state, ScaleMobility{m}, out_hat);

  Field<double> got(domain, fft.get_inbox_bounds(), 0);
  SpectralETDOps<pfc::HostSpace>::backward(fft, out_hat, got);
  const auto expected = sample(potential, [k, m](double x, double) {
    return m * (-k * k) * std::cos(k * x);
  });
  REQUIRE(max_abs_against(got, expected) < 1.0e-8);
  REQUIRE(zero_mode_abs(fft, domain, out_hat) < 1.0e-8);
}

TEST_CASE("variable coefficient matches the product rule on resolved modes",
          "[spectral_flux]") {
  auto domain = film_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  const double lx = static_cast<double>(N);
  const double k = 2.0 * pfc::pi / lx;
  constexpr double a = 0.25;

  Field<double> potential(domain, fft.get_inbox_bounds(), 0);
  potential.apply([k](double x, double, double) { return std::cos(k * x); });
  Field<double> coefficient(domain, fft.get_inbox_bounds(), 0);
  coefficient.apply(
      [k, a](double x, double, double) { return 1.0 + a * std::cos(k * x); });

  Field<std::complex<double>> p_hat(domain, fft.get_outbox_bounds(), 0);
  Field<std::complex<double>> out_hat(domain, fft.get_outbox_bounds(), 0);
  SpectralETDOps<pfc::HostSpace>::forward(fft, potential, p_hat);

  SpectralFlux<> flux(domain, fft);
  flux.divergence(p_hat, coefficient, out_hat);

  Field<std::complex<double>> via_mobility(domain, fft.get_outbox_bounds(), 0);
  flux.divergence(p_hat, coefficient, AsStored{}, via_mobility);

  Field<double> got(domain, fft.get_inbox_bounds(), 0);
  SpectralETDOps<pfc::HostSpace>::backward(fft, out_hat, got);
  const auto expected = sample(potential, [k, a](double x, double) {
    return -k * k * std::cos(k * x) - a * k * k * std::cos(2.0 * k * x);
  });
  REQUIRE(max_abs_against(got, expected) < 1.0e-8);
  REQUIRE(zero_mode_abs(fft, domain, out_hat) < 1.0e-8);

  Field<double> got_mobility(domain, fft.get_inbox_bounds(), 0);
  SpectralETDOps<pfc::HostSpace>::backward(fft, via_mobility, got_mobility);
  REQUIRE(got.size() == got_mobility.size());
  double path_gap = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    path_gap = std::max(path_gap, std::abs(got.data()[i] - got_mobility.data()[i]));
  }
  REQUIRE(path_gap < 1.0e-12);
}

TEST_CASE("zero mobility ETD step leaves the field unchanged", "[spectral_flux]") {
  auto domain = film_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  Field<double> u(domain, fft.get_inbox_bounds(), 0);
  const double k = 2.0 * pfc::pi / static_cast<double>(N);
  u.apply([k](double x, double, double) { return 1.0 + 0.1 * std::cos(k * x); });
  std::vector<double> before(u.data(), u.data() + u.size());

  FluxETD<> stepper(domain, fft, 0.05, [](double) { return 0.0; });
  stepper.step(
      0.0, u,
      [](auto &, auto &, auto &p_hat) {
        p_hat.with_host_view([](std::complex<double> *p, std::size_t n) {
          for (std::size_t i = 0; i < n; ++i) p[i] = {0.0, 0.0};
        });
      },
      ScaleMobility{0.0});
  REQUIRE(max_abs_against(u, before) < 1.0e-12);
}

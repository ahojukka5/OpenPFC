// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Installed-header consumer for `SimulationState` and `SpectralETDSystem`.
// `simulation_state.hpp` includes `simulation_state.ipp`, so this translation
// unit fails to compile against a prefix that does not ship the `.ipp`
// (0.4.0 shipped without it). It then takes one ETD1 step of pure diffusion,
// `d psi / dt = lap psi`, from `mean + A cos(k x)`. ETD1 is exact for a linear
// equation, so the oracle is the closed form `mean + A exp(-k^2 dt) cos(k x)`.
#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

#include <cmath>
#include <cstdio>

namespace {

struct NoNonlinearity {
  double nonlinearity(const pfc::sim::SpectralCell &) const { return 0.0; }
};

struct Diffusion {
  pfc::Domain domain;
  pfc::Box3i box;

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<double>(state, "psi", domain, box, 0);
  }
  // `k_laplacian` is -|k|^2, so the diffusion symbol is the value itself.
  double linear_symbol(double k_laplacian) const { return k_laplacian; }
  NoNonlinearity pointwise() const { return {}; }
};

static_assert(pfc::sim::SpectralETDPhysics<Diffusion>);

} // namespace

// Returns 0 when the installed headers give the closed-form step.
int run_spectral_state_consumer() {
  constexpr int n = 8;
  constexpr double mean = 0.5;
  constexpr double amp = 0.1;
  constexpr double dt = 0.05;
  auto domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomposition);

  Diffusion physics{domain, fft.get_inbox_bounds()};
  pfc::SimulationState state;
  physics.declare_fields(state);
  const auto handle = state.get_field_handle<double>("psi");
  if (!handle.is_valid() ||
      &state.get_field_by_handle(handle) != &state.get_field<double>("psi")) {
    std::printf("SimulationState: psi was not declared\n");
    return 1;
  }

  const double k = 2.0 * pfc::pi / static_cast<double>(n);
  auto &psi = state.get_field<double>("psi");
  psi.apply([&](double x, double, double) { return mean + amp * std::cos(k * x); });

  pfc::sim::SpectralETDSystem<Diffusion> system(physics, fft, state, dt);
  const double t1 = system.step(0.0);

  const double decay = std::exp(-k * k * dt);
  double worst = 0.0;
  psi.for_each_owned([&](double x, double, double, double value) {
    worst =
        std::fmax(worst, std::fabs(value - (mean + amp * decay * std::cos(k * x))));
  });
  const bool ok = std::fabs(t1 - dt) < 1e-15 && worst < 1e-12;
  std::printf("OpenPFC SpectralETDSystem consumer %s: max error %.3e\n",
              ok ? "OK" : "FAILED", worst);
  return ok ? 0 : 1;
}

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Smallest possible use of the installed OpenPFC public API: construct a Domain
// and query it. If this configures, links, and runs, find_package(OpenPFC) and
// the exported targets/transitive deps are wired correctly.
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/face_flux.hpp>
#include <openpfc/kernel/simulation/spectral_flux.hpp>
#include <openpfc/solvers/microelasticity/microelasticity.hpp>
#include <openpfc/kernel/fft/power_spectrum.hpp>

#include <cstdio>
#include <type_traits>

int main() {
  static_assert(std::is_trivially_copyable_v<pfc::sim::AsStored>);
  static_assert(static_cast<int>(pfc::solvers::MicroelasticityScheme::EyreMilton) !=
                static_cast<int>(pfc::solvers::MicroelasticityScheme::Basic));
  static_assert(pfc::field::fd::average_face(pfc::field::fd::FaceAverage::Harmonic,
                                             0.0, 1.0) == 0.0);
  static_assert(pfc::fft::r2c_multiplicity(0, 64) == 1.0);
  static_assert(pfc::fft::r2c_multiplicity(1, 64) == 2.0);
  static_assert(pfc::fft::r2c_multiplicity(32, 64) == 1.0);
  pfc::fft::RadialSpectrum spectrum;
  spectrum.wavenumber = {1.0};
  spectrum.mean_power = {1.0};
  if (pfc::fft::power_near(spectrum, 1.0) != 1.0) {
    return 1;
  }
  pfc::solvers::MicroelasticityParams elastic;
  elastic.stiffness_at_one = pfc::solvers::Stiffness::isotropic(1.0, 0.3);
  elastic.stiffness_at_zero = elastic.stiffness_at_one;
  elastic.relative_tolerance = 1.0e-6;
  if (elastic.max_iterations < 1) {
    return 1;
  }
  auto domain = pfc::domain::create({8, 8, 8});
  const auto size = pfc::domain::get_size(domain);
  std::printf("OpenPFC consumer OK: domain %dx%dx%d\n", size[0], size[1], size[2]);
  return (size[0] == 8 && size[1] == 8 && size[2] == 8) ? 0 : 1;
}

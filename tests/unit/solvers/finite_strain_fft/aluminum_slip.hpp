// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file aluminum_slip.hpp
 * @brief DAMASK 3.0 aluminum coefficients for the FCC slip tests.
 *
 * Not library API. The only callers are the FCC slip unit tests.
 */

#pragma once

#include <openpfc/mechanics/constitutive/crystal_plasticity.hpp>

namespace pfc::finite_strain {

/// Aluminum coefficients from the DAMASK 3.0 grid example `material.yaml`.
[[nodiscard]] inline FccSlipParameters aluminum_slip_parameters() noexcept {
  FccSlipParameters parameters;
  parameters.c11 = 106.75e9;
  parameters.c12 = 60.41e9;
  parameters.c44 = 28.34e9;
  parameters.reference_rate = 0.001;
  parameters.rate_exponent = 20.0;
  parameters.hardening_exponent = 2.25;
  parameters.hardening_coefficient = 75.0e6;
  parameters.initial_resistance = 31.0e6;
  parameters.saturation_resistance = 63.0e6;
  parameters.time_step = 1.0;
  parameters.coplanar = 1.0;
  parameters.latent = 1.4;
  return parameters;
}

} // namespace pfc::finite_strain

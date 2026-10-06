// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_saint_venant_kirchhoff.cpp
 * @brief Local checks for the first Piola stress and its tangent action.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/mechanics/constitutive/saint_venant_kirchhoff.hpp>

#include <cmath>
#include <cstdint>

using Catch::Matchers::WithinAbs;
using pfc::finite_strain::constitutive_response;
using pfc::finite_strain::frobenius_norm;
using pfc::finite_strain::identity2;
using pfc::finite_strain::IsotropicModuli;
using pfc::finite_strain::tangent_action;
using pfc::finite_strain::Tensor2;

namespace {

Tensor2 random_tensor(std::uint32_t &state) {
  Tensor2 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      state = state * 1664525u + 1013904223u;
      const double unit = static_cast<double>(state >> 8) / 16777216.0;
      out(i, j) = 0.35 * (2.0 * unit - 1.0);
    }
  }
  return out;
}

} // namespace

TEST_CASE("identity deformation has zero Saint-Venant stress",
          "[finite_strain][constitutive]") {
  const IsotropicModuli moduli{0.833, 0.386};
  const auto response = constitutive_response(identity2(), moduli);
  REQUIRE_THAT(frobenius_norm(response.piola), WithinAbs(0.0, 1e-14));
  REQUIRE_THAT(frobenius_norm(response.second_pk), WithinAbs(0.0, 1e-14));
}

TEST_CASE("tangent action matches a central difference of the stress",
          "[finite_strain][constitutive]") {
  const IsotropicModuli moduli{0.833, 0.386};
  std::uint32_t state = 17u;
  const double step = 1e-7;
  for (int sample = 0; sample < 6; ++sample) {
    Tensor2 deformation = identity2();
    const Tensor2 offset = random_tensor(state);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        deformation(i, j) += offset(i, j);
      }
    }
    const Tensor2 direction = random_tensor(state);
    Tensor2 forward = deformation;
    Tensor2 backward = deformation;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        forward(i, j) += step * direction(i, j);
        backward(i, j) -= step * direction(i, j);
      }
    }
    const Tensor2 analytic = tangent_action(deformation, direction, moduli);
    const auto stress_forward = constitutive_response(forward, moduli).piola;
    const auto stress_backward = constitutive_response(backward, moduli).piola;
    double max_error = 0.0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const double finite =
            (stress_forward(i, j) - stress_backward(i, j)) / (2.0 * step);
        max_error = std::max(max_error, std::abs(finite - analytic(i, j)));
      }
    }
    REQUIRE(max_error < 1e-8);
  }
}

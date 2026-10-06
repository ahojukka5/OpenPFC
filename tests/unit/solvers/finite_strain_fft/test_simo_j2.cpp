// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_simo_j2.cpp
 * @brief Material-point ladder for the finite-strain Simo J2 update.
 *
 * The locked stress values are the one-point reduction of GooseFFT
 * `finite-strain/elasto-plasticity.py` at blob
 * f4aa52bc7eecbd1cffc2d0be5e524c7da1e5f08b, with the soft moduli of that
 * program. The tangent is also checked by a central difference.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/mechanics/constitutive/simo_j2.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

using Catch::Matchers::WithinAbs;
using pfc::finite_strain::frobenius_norm;
using pfc::finite_strain::identity2;
using pfc::finite_strain::identity_j2_state;
using pfc::finite_strain::integrate_j2;
using pfc::finite_strain::j2_tangent_action;
using pfc::finite_strain::J2Moduli;
using pfc::finite_strain::J2State;
using pfc::finite_strain::SimoJ2Material;
using pfc::finite_strain::Tensor2;

namespace {

const J2Moduli kSoft{0.833, 0.386, 0.004, 0.003};

Tensor2 pure_shear(double stretch) {
  Tensor2 deformation = identity2();
  deformation(0, 0) = stretch;
  deformation(1, 1) = 1.0 / stretch;
  return deformation;
}

Tensor2 random_direction(std::uint32_t &state) {
  Tensor2 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      state = state * 1664525u + 1013904223u;
      const double unit = static_cast<double>(state >> 8) / 16777216.0;
      out(i, j) = 2.0 * unit - 1.0;
    }
  }
  return out;
}

bool same_state(const J2State &left, const J2State &right) {
  if (left.plastic != right.plastic) {
    return false;
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (left.finger(i, j) != right.finger(i, j) ||
          left.deformation(i, j) != right.deformation(i, j)) {
        return false;
      }
    }
  }
  return true;
}

double max_abs(const Tensor2 &tensor) {
  double largest = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      largest = std::max(largest, std::abs(tensor(i, j)));
    }
  }
  return largest;
}

void require_close(const Tensor2 &value, const double expected[3][3], double tol) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      REQUIRE_THAT(value(i, j), WithinAbs(expected[i][j], tol));
    }
  }
}

} // namespace

TEST_CASE("identity deformation stores zero J2 stress", "[finite_strain][j2]") {
  const auto response = integrate_j2(kSoft, identity_j2_state(), identity2());
  REQUIRE_THAT(frobenius_norm(response.piola), WithinAbs(0.0, 1e-14));
  REQUIRE_THAT(response.plastic, WithinAbs(0.0, 0.0));
  REQUIRE_THAT(response.equivalent, WithinAbs(0.0, 1e-14));
}

TEST_CASE("a small pure shear stays elastic and matches the reference stress",
          "[finite_strain][j2]") {
  const Tensor2 deformation = pure_shear(1.002);
  const auto response = integrate_j2(kSoft, identity_j2_state(), deformation);
  REQUIRE_THAT(response.plastic, WithinAbs(0.0, 0.0));
  REQUIRE(response.equivalent < kSoft.initial_yield);
  const double expected[3][3] = {
      {1.5393792969895072e-03, 0.0, 0.0},
      {0.0, -1.5455429716948982e-03, 0.0},
      {0.0, 0.0, 0.0},
  };
  require_close(response.piola, expected, 1e-12);
}

TEST_CASE("the first plastic step hardens at the declared slope",
          "[finite_strain][j2]") {
  const double yield_stretch =
      std::exp(kSoft.initial_yield / (2.0 * std::sqrt(3.0) * kSoft.shear));
  const auto below =
      integrate_j2(kSoft, identity_j2_state(), pure_shear(yield_stretch * 0.999));
  const auto above =
      integrate_j2(kSoft, identity_j2_state(), pure_shear(yield_stretch * 1.001));
  REQUIRE_THAT(below.plastic, WithinAbs(0.0, 0.0));
  REQUIRE(above.plastic > 0.0);
  REQUIRE_THAT(
      above.equivalent,
      WithinAbs(kSoft.initial_yield + kSoft.hardening * above.plastic, 1e-12));

  const auto locked = integrate_j2(kSoft, identity_j2_state(), pure_shear(1.05));
  REQUIRE_THAT(locked.plastic, WithinAbs(5.3562338545469837e-02, 1e-12));
  const double expected[3][3] = {
      {1.7673787903613972e-03, 0.0, 0.0},
      {0.0, -1.9485351163734844e-03, 0.0},
      {0.0, 0.0, 0.0},
  };
  require_close(locked.piola, expected, 1e-12);
  REQUIRE_THAT(
      locked.equivalent,
      WithinAbs(kSoft.initial_yield + kSoft.hardening * locked.plastic, 1e-12));
}

TEST_CASE("monotonic commits increase plastic strain and unload does not",
          "[finite_strain][j2]") {
  const std::vector<std::uint8_t> uniform;
  SimoJ2Material material(uniform, 1, kSoft, kSoft);
  double previous = 0.0;
  Tensor2 last = identity2();
  for (int step = 1; step <= 8; ++step) {
    const Tensor2 deformation = pure_shear(1.0 + 0.01 * static_cast<double>(step));
    material.begin_trial();
    material.stage(0, deformation);
    const double trial = material.history().trial(0).plastic;
    REQUIRE(trial >= previous);
    material.accept();
    REQUIRE(material.accumulated(0) == trial);
    previous = trial;
    last = deformation;
  }
  REQUIRE(material.accumulated(0) > 0.0);

  Tensor2 unload = last;
  unload(0, 0) -= 0.002;
  unload(1, 1) = 1.0 / unload(0, 0);
  const double frozen = material.accumulated(0);
  const J2State committed = material.history().committed(0);
  material.begin_trial();
  material.stage(0, unload);
  REQUIRE(material.history().trial(0).plastic == frozen);
  material.reject();
  REQUIRE(same_state(material.history().committed(0), committed));
  REQUIRE(material.accumulated(0) == frozen);

  material.begin_trial();
  material.stage(0, pure_shear(1.09));
  REQUIRE(material.history().trial(0).plastic > frozen);
  material.accept();
  REQUIRE(material.accumulated(0) > frozen);
}

TEST_CASE("a rejected J2 trial leaves the committed state bitwise",
          "[finite_strain][j2]") {
  const std::vector<std::uint8_t> uniform;
  SimoJ2Material material(uniform, 1, kSoft, kSoft);
  material.begin_trial();
  material.stage(0, pure_shear(1.04));
  material.accept();
  const J2State committed = material.history().committed(0);
  REQUIRE(committed.plastic > 0.0);

  material.begin_trial();
  material.stage(0, pure_shear(1.08));
  REQUIRE(material.history().trial(0).plastic > committed.plastic);
  material.reject();
  REQUIRE(same_state(material.history().committed(0), committed));
  REQUIRE(material.history().trial(0).plastic == committed.plastic);

  const auto after =
      integrate_j2(kSoft, material.history().committed(0), pure_shear(1.04));
  // The spectral logarithm is not an exact fixed point on every libm.
  // Catch's ten-digit print of the two plastic strains already agrees.
  // 1e-9 sits above that print resolution and far below a return-map step.
  REQUIRE_THAT(after.plastic, WithinAbs(committed.plastic, 1e-9));
}

TEST_CASE("the algorithmic tangent matches a central difference",
          "[finite_strain][j2]") {
  const double stretches[2] = {1.002, 1.05};
  std::uint32_t state = 23u;
  const double step = 1e-7;
  for (const double stretch : stretches) {
    const Tensor2 deformation = pure_shear(stretch);
    const Tensor2 direction = random_direction(state);
    Tensor2 forward = deformation;
    Tensor2 backward = deformation;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        forward(i, j) += step * direction(i, j);
        backward(i, j) -= step * direction(i, j);
      }
    }
    const Tensor2 analytic =
        j2_tangent_action(kSoft, identity_j2_state(), deformation, direction);
    const Tensor2 stress_forward =
        integrate_j2(kSoft, identity_j2_state(), forward).piola;
    const Tensor2 stress_backward =
        integrate_j2(kSoft, identity_j2_state(), backward).piola;
    double largest = 0.0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const double finite =
            (stress_forward(i, j) - stress_backward(i, j)) / (2.0 * step);
        largest = std::max(largest, std::abs(finite - analytic(i, j)));
      }
    }
    REQUIRE(largest < 1e-7);
    REQUIRE(max_abs(analytic) > 1e-3);
  }
}

TEST_CASE("the locked plastic tangent matches the reference program",
          "[finite_strain][j2]") {
  Tensor2 direction;
  direction(0, 0) = 0.2;
  direction(0, 1) = -0.1;
  direction(0, 2) = 0.05;
  direction(1, 0) = 0.04;
  direction(1, 1) = 0.3;
  direction(1, 2) = -0.02;
  direction(2, 0) = 0.0;
  direction(2, 1) = 0.07;
  direction(2, 2) = -0.1;
  const Tensor2 action =
      j2_tangent_action(kSoft, identity_j2_state(), pure_shear(1.05), direction);
  const double expected[3][3] = {
      {0.3254427573274623, -0.0011356308537668, 0.0009052427950632},
      {-0.0011482967181776, 0.3601330444983381, 0.0010678922942661},
      {0.0008621359952983, 0.0009780695779260, 0.3288173376798252},
  };
  require_close(action, expected, 1e-10);
}

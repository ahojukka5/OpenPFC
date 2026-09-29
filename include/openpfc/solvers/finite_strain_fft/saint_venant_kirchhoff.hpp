// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file saint_venant_kirchhoff.hpp
 * @brief Local Saint-Venant–Kirchhoff map `F -> (P, dP)`.
 *
 * The strain energy uses the Green–Lagrange strain
 * `E = (F^T F - I) / 2` and the isotropic stiffness
 * `S = λ tr(E) I + 2 μ E`, with `P = F S`. Lamé's first parameter is
 * stored through the bulk modulus, `λ = κ - 2μ/3`, which is the splitting
 * used by the published GooseFFT hyperelastic program.
 *
 * `tangent_action(F, dF)` is the directional derivative `dP`. It is the
 * action that the Newton step writes as `K^{LT} : dF^T` in de Geus et al.,
 * Eq. (23). The fourth-order tensor is not stored.
 */

#pragma once

#include <openpfc/solvers/finite_strain_fft/tensor.hpp>

namespace pfc::finite_strain {

/// Isotropic moduli at one material point. `bulk` is `κ`, not Young's modulus.
struct IsotropicModuli {
  double bulk = 0.0;
  double shear = 0.0;
};

struct StressResponse {
  Tensor2 piola{};     ///< First Piola–Kirchhoff stress `P`.
  Tensor2 second_pk{}; ///< Second Piola–Kirchhoff stress `S`.
};

[[nodiscard]] inline Tensor2 green_lagrange(const Tensor2 &deformation) noexcept {
  const Tensor2 ft_f = matmul(transposed(deformation), deformation);
  Tensor2 strain = scaled(ft_f, 0.5);
  strain(0, 0) -= 0.5;
  strain(1, 1) -= 0.5;
  strain(2, 2) -= 0.5;
  return strain;
}

/// `S = λ tr(E) I + 2 μ E` with `λ = κ - 2μ/3`. `E` must be symmetric.
[[nodiscard]] inline Tensor2 second_piola(const Tensor2 &strain,
                                          IsotropicModuli moduli) noexcept {
  const double lambda = moduli.bulk - 2.0 * moduli.shear / 3.0;
  const double volumetric = lambda * trace(strain);
  Tensor2 stress = scaled(strain, 2.0 * moduli.shear);
  stress(0, 0) += volumetric;
  stress(1, 1) += volumetric;
  stress(2, 2) += volumetric;
  return stress;
}

[[nodiscard]] inline StressResponse
constitutive_response(const Tensor2 &deformation, IsotropicModuli moduli) noexcept {
  StressResponse out;
  out.second_pk = second_piola(green_lagrange(deformation), moduli);
  out.piola = matmul(deformation, out.second_pk);
  return out;
}

/**
 * Directional derivative `dP` at `deformation` in the direction `increment`.
 *
 * `dE = sym(F^T dF)`, `dS = λ tr(dE) I + 2 μ dE`, `dP = dF S + F dS`.
 */
[[nodiscard]] inline Tensor2 tangent_action(const Tensor2 &deformation,
                                            const Tensor2 &increment,
                                            IsotropicModuli moduli) noexcept {
  const Tensor2 second_pk = second_piola(green_lagrange(deformation), moduli);
  const Tensor2 strain_inc =
      symmetric_part(matmul(transposed(deformation), increment));
  const Tensor2 stress_inc = second_piola(strain_inc, moduli);
  return add(matmul(increment, second_pk), matmul(deformation, stress_inc));
}

/// Von Mises equivalent of a symmetric stress, `sqrt(3/2) ||dev σ||_F`.
[[nodiscard]] inline double von_mises(const Tensor2 &stress) noexcept {
  const double mean = trace(stress) / 3.0;
  Tensor2 deviator = stress;
  deviator(0, 0) -= mean;
  deviator(1, 1) -= mean;
  deviator(2, 2) -= mean;
  return std::sqrt(1.5 * frobenius_dot(deviator, deviator));
}

} // namespace pfc::finite_strain

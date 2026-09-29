// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file constitutive.hpp
 * @brief Local contract consumed by the finite-strain FFT Newton solver.
 *
 * A law maps one deformation gradient to the first Piola-Kirchhoff stress
 * and applies its tangent to an increment. The global solver never names a
 * material. History, if a later law has any, stays inside the law.
 */

#pragma once

#include <concepts>
#include <cstddef>

#include <openpfc/solvers/finite_strain_fft/tensor.hpp>

namespace pfc::finite_strain {

template <typename Law>
concept LocalConstitutiveLaw =
    requires(const Law &law, std::size_t index, const Tensor2 &deformation,
             const Tensor2 &increment) {
      { law.stress(index, deformation) } -> std::convertible_to<Tensor2>;
      {
        law.tangent_action(index, deformation, increment)
      } -> std::convertible_to<Tensor2>;
    };

} // namespace pfc::finite_strain

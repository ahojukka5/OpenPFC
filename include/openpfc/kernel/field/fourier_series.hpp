// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fourier_series.hpp
 * @brief A scalar field written as a sum of periodic cosines.
 *
 * \f$ u(\mathbf x) = \mathrm{offset} + \sum_m A_m \cos(\mathbf k_m\cdot
 * \mathbf x + \phi_m) \f$, with \f$ k_{m,d} = 2\pi n_{m,d} / L_d \f$ and
 * \f$ L_d \f$ the periodic length of that axis. Integer \f$ n \f$ makes each
 * term close on the domain. The header does not know which scalar it is
 * writing.
 */

#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/field/state_access.hpp>

namespace pfc::field {

/// One term. `index` is the integer wave count along each periodic axis.
struct FourierMode {
  Int3 index{0, 0, 0};
  double amplitude{0.0};
  double phase{0.0};
};

/// A nonzero index is a periodic wave count. A zero index is a constant on that
/// axis.
inline void require_periodic_mode(const pfc::Domain &domain,
                                  const FourierMode &mode) {
  const auto size = pfc::domain::get_size(domain);
  const auto periodic = pfc::domain::get_periodic(domain);
  for (int d = 0; d < 3; ++d) {
    if (mode.index[d] == 0) continue;
    if (periodic[d] && size[d] > 1) continue;
    throw std::invalid_argument(
        "fourier mode: a nonzero index requires a periodic axis with more than "
        "one point (axis " +
        std::to_string(d) + ")");
  }
}

/// \f$ 2\pi n / L \f$. Zero when that axis has no length.
[[nodiscard]] inline double periodic_wavenumber(int index, double spacing,
                                                int count) noexcept {
  const double length = spacing * static_cast<double>(count);
  if (!(length > 0.0)) return 0.0;
  return (2.0 * std::numbers::pi) * static_cast<double>(index) / length;
}

/// The sum, without the offset.
[[nodiscard]] inline double fourier_series(const pfc::Domain &domain,
                                           const pfc::Real3 &x,
                                           std::span<const FourierMode> modes) {
  const auto size = pfc::domain::get_size(domain);
  const auto spacing = pfc::domain::get_spacing(domain);
  double sum = 0.0;
  for (const FourierMode &mode : modes) {
    require_periodic_mode(domain, mode);
    const double kx = periodic_wavenumber(mode.index[0], spacing[0], size[0]);
    const double ky = periodic_wavenumber(mode.index[1], spacing[1], size[1]);
    const double kz = periodic_wavenumber(mode.index[2], spacing[2], size[2]);
    sum += mode.amplitude * std::cos(kx * x[0] + ky * x[1] + kz * x[2] + mode.phase);
  }
  return sum;
}

/// `field += series`. Leaves every other contribution in place.
inline void add_fourier_series(FieldOutput<double> field, const pfc::Domain &domain,
                               const pfc::Box3i &box,
                               std::span<const FourierMode> modes) {
  apply_inplace(field, domain, box, [&](const pfc::Real3 &x, double current) {
    return current + fourier_series(domain, x, modes);
  });
}

/// `field = offset + series`.
inline void fill_fourier_series(FieldOutput<double> field, const pfc::Domain &domain,
                                const pfc::Box3i &box, double offset,
                                std::span<const FourierMode> modes) {
  apply(field, domain, box, [&](const pfc::Real3 &x) {
    return offset + fourier_series(domain, x, modes);
  });
}

} // namespace pfc::field

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/kernel/field/sbp_diffusion.hpp>
#include <span>

namespace pfc::field::fd {
struct ReactionPoint {
  double state;
  std::span<const double> other_fields;
  std::array<double, 3> position;
  std::array<double, 3> outward_normal;
  double stage_time;
};
struct ReactionRate {
  double outward_flux;
  bool converged = true;
};

/// Evaluate once on the physical owner after preparing stage fields. Neither
/// law nor admissible may mutate accepted state or perform global collectives.
/// The returned face and its flux use precisely the normal-flux quadrature.
/// Caller records that same flux once with the actual stage weight and area.
template <class Params, class Law, class Admissible>
FaceCondition reaction_face(const ReactionPoint &point, const Params &params,
                            Law law, Admissible admissible, FluxLedger &ledger) {
  if (!ledger.active()) throw std::logic_error("reaction needs active flux trial");
  try {
    if (!std::isfinite(point.state) || !std::isfinite(point.stage_time))
      throw std::domain_error("nonfinite reaction state or time");
    for (double v : point.other_fields)
      if (!std::isfinite(v)) throw std::domain_error("nonfinite reaction field");
    int normal_axes = 0;
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(point.position[axis]))
        throw std::domain_error("nonfinite reaction position");
      const double n = point.outward_normal[axis];
      if (n == 1 || n == -1) ++normal_axes;
      else if (n != 0) throw std::domain_error("non-Cartesian face normal");
    }
    if (normal_axes != 1 || !admissible(point, params))
      throw std::domain_error("inadmissible reaction point");
    const ReactionRate result = law(point, params);
    if (!result.converged || !std::isfinite(result.outward_flux))
      throw std::domain_error("reaction solve failed or rate is nonfinite");
    return {BoundaryQuantity::ConstitutiveFlux,
            [flux = result.outward_flux](double) { return flux; }};
  } catch (...) {
    ledger.invalidate();
    throw;
  }
}
} // namespace pfc::field::fd

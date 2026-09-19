// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file simp_penalty.hpp
 * @brief SIMP density \(h^p\) and chain-rule \(p h^{p-1}\) used by CPU and HIP.
 *
 * \(p=1\) is linear interpolation and must not call \(\mathrm{pow}(h,0)\).
 * \(h\) is clamped to \([0,1]\) so \(h=0\) never produces NaN for \(p\ge 1\).
 */

#include <algorithm>
#include <cmath>

namespace pfc::apps::inverse {

[[nodiscard]] inline double clamp01(double h) noexcept {
  return std::min(1.0, std::max(0.0, h));
}

[[nodiscard]] inline double simp_density(double h, double p) noexcept {
  h = clamp01(h);
  if (p == 1.0) return h;
  return std::pow(h, p);
}

[[nodiscard]] inline double simp_chain(double h, double p) noexcept {
  h = clamp01(h);
  if (p == 1.0) return 1.0;
  return p * std::pow(h, p - 1.0);
}

} // namespace pfc::apps::inverse

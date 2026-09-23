// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file concentration_seed.hpp
 * @brief Keep a Cahn–Hilliard noise seed inside (0, 1).
 *
 * The generic indexed-noise modifier does not know that `c` is a mole
 * fraction. This check is the application bound: the perturbation must not
 * be able to leave that interval when it is written around `offset`.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_field_modifiers.hpp>

namespace cahn_hilliard {

inline void require_concentration_noise(double offset, double amplitude) {
  if (!std::isfinite(offset) || !std::isfinite(amplitude) || !(offset > 0.0) ||
      !(offset < 1.0) || amplitude < 0.0 ||
      !(amplitude < std::min(offset, 1.0 - offset))) {
    throw std::invalid_argument(
        "seeded_noise: require 0<offset<1 and 0<=amplitude<min(offset,1-offset)");
  }
}

inline void require_concentration_noise(const nlohmann::json &ic) {
  if (ic.value("type", "") != "seeded_noise") return;
  nlohmann::json copy = ic;
  pfc::ui::copy_json_alias(copy, "c0", "offset");
  require_concentration_noise(copy.at("offset").get<double>(),
                              copy.at("amplitude").get<double>());
}

} // namespace cahn_hilliard

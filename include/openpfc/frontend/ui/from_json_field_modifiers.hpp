// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file from_json_field_modifiers.hpp
 * @brief JSON hooks for built-in IC/BC types
 */

#ifndef PFC_UI_FROM_JSON_FIELD_MODIFIERS_HPP
#define PFC_UI_FROM_JSON_FIELD_MODIFIERS_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <openpfc/frontend/ui/from_json_fwd.hpp>
#include <openpfc/kernel/simulation/initial_conditions/constant.hpp>
#include <openpfc/kernel/simulation/initial_conditions/file_reader.hpp>
#include <openpfc/kernel/simulation/initial_conditions/fourier_modes.hpp>
#include <openpfc/kernel/simulation/initial_conditions/indexed_noise.hpp>
#include <openpfc/kernel/simulation/initial_conditions/random_seeds.hpp>
#include <openpfc/kernel/simulation/initial_conditions/seed_grid.hpp>
#include <openpfc/kernel/simulation/initial_conditions/single_seed.hpp>

namespace pfc::ui {
namespace detail {

inline void throw_unless_json_modifier_type(const json &j, const char *expected,
                                            std::string_view message) {
  if (!j.contains("type") || j["type"] != expected) {
    throw std::invalid_argument(std::string(message));
  }
}

} // namespace detail

inline void from_json(const json &j, Constant &ic) {
  detail::throw_unless_json_modifier_type(
      j, "constant", "Invalid JSON input: missing or incorrect 'type' field.");
  // Check that the JSON input has the required 'n0' field
  if (!j.contains("n0") || !j["n0"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'n0' field.");
  }
  ic.set_density(j["n0"]);
}

inline void from_json(const json &j, SingleSeed &seed) {
  detail::throw_unless_json_modifier_type(
      j, "single_seed", "JSON object does not contain a 'single_seed' type.");

  if (!j.contains("amp_eq")) {
    throw std::invalid_argument("JSON object does not contain an 'amp_eq' key.");
  }

  if (!j.contains("rho_seed")) {
    throw std::invalid_argument("JSON object does not contain an 'rho_seed' key.");
  }

  seed.set_amplitude(j["amp_eq"]);
  seed.set_density(j["rho_seed"]);
}

inline void from_json(const json &j, RandomSeeds &ic) {
  detail::throw_unless_json_modifier_type(
      j, "random_seeds", "Invalid JSON input: missing or incorrect 'type' field.");

  // Check that the JSON input has the required 'amplitude' field
  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'amplitude' field.");
  }

  // Check that the JSON input has the required 'rho' field
  if (!j.contains("rho") || !j["rho"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho' field.");
  }

  ic.set_amplitude(j["amplitude"]);
  ic.set_density(j["rho"]);
}

inline void from_json(const json &j, SeedGrid &ic) {
  detail::throw_unless_json_modifier_type(
      j, "seed_grid", "Invalid JSON input: missing or incorrect 'type' field.");

  if (!j.contains("Ny") || !j["Ny"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'Ny' field.");
  }

  if (!j.contains("Nz") || !j["Nz"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'Nz' field.");
  }

  if (!j.contains("X0") || !j["X0"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'X0' field.");
  }

  if (!j.contains("radius") || !j["radius"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'radius' field.");
  }

  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'amplitude' field.");
  }

  if (!j.contains("rho") || !j["rho"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho' field.");
  }

  if (j.contains("Nx")) {
    if (!j["Nx"].is_number()) {
      throw std::invalid_argument(
          "Invalid JSON input: missing or invalid 'Nx' field.");
    }
    ic.set_Nx(j["Nx"]);
  }
  ic.set_Ny(j["Ny"]);
  ic.set_Nz(j["Nz"]);
  ic.set_X0(j["X0"]);
  ic.set_radius(j["radius"]);
  ic.set_amplitude(j["amplitude"]);
  ic.set_density(j["rho"]);
}

inline void from_json(const json &j, FileReader &ic) {
  detail::throw_unless_json_modifier_type(
      j, "from_file", "Invalid JSON input: missing or incorrect 'type' field.");

  if (!j.contains("filename") || !j["filename"].is_string()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'filename' field.");
  }

  ic.set_filename(j["filename"]);
}

namespace detail {

inline int json_mode_index(const json &j, const char *key, int fallback) {
  if (!j.contains(key)) return fallback;
  if (!j[key].is_number()) {
    throw std::invalid_argument(std::string("fourier mode: '") + key +
                                "' must be an integer.");
  }
  return j[key].get<int>();
}

inline pfc::field::FourierMode one_fourier_mode(const json &m, int default_nx) {
  pfc::field::FourierMode mode;
  if (m.contains("n")) {
    if (!m["n"].is_array() || m["n"].size() != 3) {
      throw std::invalid_argument("fourier mode: 'n' must be [nx, ny, nz].");
    }
    mode.index = {m["n"].at(0).get<int>(), m["n"].at(1).get<int>(),
                  m["n"].at(2).get<int>()};
  } else {
    mode.index = {json_mode_index(m, "nx", default_nx), json_mode_index(m, "ny", 0),
                  json_mode_index(m, "nz", 0)};
  }
  if (!m.contains("amplitude") || !m["amplitude"].is_number()) {
    throw std::invalid_argument("fourier mode: missing or invalid 'amplitude'.");
  }
  mode.amplitude = m["amplitude"].get<double>();
  if (m.contains("phase")) {
    if (!m["phase"].is_number()) {
      throw std::invalid_argument("fourier mode: 'phase' must be numeric.");
    }
    mode.phase = m["phase"].get<double>();
  }
  return mode;
}

inline std::vector<pfc::field::FourierMode> fourier_terms_from_json(const json &j) {
  std::vector<pfc::field::FourierMode> terms;
  if (j.contains("modes")) {
    if (!j["modes"].is_array() || j["modes"].empty()) {
      throw std::invalid_argument(
          "fourier modes: 'modes' must be a non-empty array.");
    }
    terms.reserve(j["modes"].size());
    for (const auto &mode : j["modes"]) {
      if (!mode.contains("n") && !mode.contains("nx")) {
        throw std::invalid_argument("fourier mode: each mode needs 'n' or 'nx'.");
      }
      terms.push_back(one_fourier_mode(mode, 0));
    }
    return terms;
  }
  terms.push_back(one_fourier_mode(j, 1));
  return terms;
}

inline bool has_legacy_offset(const json &j) {
  for (const char *key : {"offset", "c0", "h0", "psi0", "u0", "g0"}) {
    if (j.contains(key)) return true;
  }
  return false;
}

inline double legacy_offset(const json &j, bool required) {
  const char *found = nullptr;
  double value = 0.0;
  for (const char *key : {"offset", "c0", "h0", "psi0", "u0", "g0"}) {
    if (!j.contains(key)) continue;
    if (!j[key].is_number()) {
      throw std::invalid_argument(std::string("offset: '") + key +
                                  "' must be numeric.");
    }
    if (found != nullptr) {
      throw std::invalid_argument(
          "offset: give one of offset, c0, h0, psi0, u0, g0.");
    }
    found = key;
    value = j[key].get<double>();
  }
  if (found == nullptr && required) {
    throw std::invalid_argument("offset: missing offset (or c0, h0, psi0, u0, g0).");
  }
  return value;
}

inline std::uint64_t json_seed(const json &j) {
  if (!j.contains("seed")) {
    throw std::invalid_argument("indexed noise: missing 'seed'.");
  }
  const auto &seed = j["seed"];
  if (!seed.is_number_integer() ||
      (!seed.is_number_unsigned() && seed.get<std::int64_t>() < 0)) {
    throw std::invalid_argument(
        "indexed noise: seed must be a nonnegative integer.");
  }
  return seed.get<std::uint64_t>();
}

inline double json_amplitude(const json &j) {
  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument("indexed noise: missing or invalid 'amplitude'.");
  }
  return j["amplitude"].get<double>();
}

inline void reject_offset_on_additive(const json &j, const char *type) {
  if (has_legacy_offset(j)) {
    throw std::invalid_argument(
        std::string(type) +
        " adds to the field. Set the offset with constant or with the fill form.");
  }
}

} // namespace detail

inline void from_json(const json &j, FourierModes &ic) {
  detail::throw_unless_json_modifier_type(
      j, "fourier_modes", "Invalid JSON input: missing or incorrect 'type' field.");
  detail::reject_offset_on_additive(j, "fourier_modes");
  ic.terms = detail::fourier_terms_from_json(j);
}

inline void from_json(const json &j, FourierSeriesFill &ic) {
  detail::throw_unless_json_modifier_type(
      j, "cosine_mode", "Invalid JSON input: missing or incorrect 'type' field.");
  ic.offset = detail::legacy_offset(j, false);
  ic.terms = detail::fourier_terms_from_json(j);
}

inline void from_json(const json &j, IndexedNoiseModifier &ic) {
  detail::throw_unless_json_modifier_type(
      j, "indexed_noise", "Invalid JSON input: missing or incorrect 'type' field.");
  detail::reject_offset_on_additive(j, "indexed_noise");
  ic.noise.seed = detail::json_seed(j);
  ic.noise.amplitude = detail::json_amplitude(j);
  if (j.contains("remove_mean")) {
    if (!j["remove_mean"].is_boolean()) {
      throw std::invalid_argument("indexed noise: 'remove_mean' must be a boolean.");
    }
    ic.noise.remove_mean = j["remove_mean"].get<bool>();
  }
}

inline void from_json(const json &j, IndexedNoiseFill &ic) {
  detail::throw_unless_json_modifier_type(
      j, "seeded_noise", "Invalid JSON input: missing or incorrect 'type' field.");
  ic.offset = detail::legacy_offset(j, true);
  ic.noise.seed = detail::json_seed(j);
  ic.noise.amplitude = detail::json_amplitude(j);
  ic.noise.remove_mean = true;
  if (j.contains("remove_mean")) {
    if (!j["remove_mean"].is_boolean()) {
      throw std::invalid_argument("indexed noise: 'remove_mean' must be a boolean.");
    }
    ic.noise.remove_mean = j["remove_mean"].get<bool>();
  }
}

} // namespace pfc::ui

#endif // PFC_UI_FROM_JSON_FIELD_MODIFIERS_HPP

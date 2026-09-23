// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fourier_modes.hpp
 * @brief Field modifiers that add, or replace with, a Fourier series.
 *
 * `FourierModes` adds the series to whatever is already in the field, so it
 * composes after a constant fill. `FourierSeriesFill` writes
 * `offset + series` and is the one-shot form used by existing
 * `cosine_mode` JSON. Neither knows which scalar it is writing.
 */

#include <string>
#include <vector>

#include <openpfc/kernel/field/fourier_series.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace pfc {

class FourierModes : public FieldModifier {
public:
  std::vector<field::FourierMode> terms;

  const std::string &get_modifier_name() const override {
    static const std::string name{"FourierModes"};
    return name;
  }

  void apply(field::FieldOutput<double> field, const Domain &domain,
             const Box3i &box, double /*time*/) override {
    pfc::field::add_fourier_series(field, domain, box, terms);
  }
};

class FourierSeriesFill : public FieldModifier {
public:
  double offset{0.0};
  std::vector<field::FourierMode> terms;

  const std::string &get_modifier_name() const override {
    static const std::string name{"FourierSeriesFill"};
    return name;
  }

  void apply(field::FieldOutput<double> field, const Domain &domain,
             const Box3i &box, double /*time*/) override {
    pfc::field::fill_fourier_series(field, domain, box, offset, terms);
  }
};

} // namespace pfc

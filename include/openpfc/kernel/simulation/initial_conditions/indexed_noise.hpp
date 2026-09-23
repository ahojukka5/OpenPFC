// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file indexed_noise.hpp
 * @brief Field modifiers for decomposition-independent indexed noise.
 *
 * `IndexedNoiseModifier` adds the noise, so a constant fill in front of it
 * keeps the offset. `IndexedNoiseFill` writes `offset + noise` and is the
 * one-shot form used by existing `seeded_noise` JSON. A communicator is
 * required only for exact mean removal, and only the one the caller passes.
 */

#include <stdexcept>
#include <string>

#include <openpfc/kernel/field/indexed_noise.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace pfc {

namespace detail {

inline void require_full_box_for_local_noise(const Domain &domain,
                                             const Box3i &box) {
  const auto n = domain::get_size(domain);
  if (box.low != Int3{0, 0, 0} || box.size != n) {
    throw std::invalid_argument(
        "indexed noise: distributed use needs SimulationContext");
  }
}

} // namespace detail

class IndexedNoiseModifier : public FieldModifier {
public:
  field::IndexedNoise noise{};

  const std::string &get_modifier_name() const override {
    static const std::string name{"IndexedNoise"};
    return name;
  }

  void apply(field::FieldOutput<double> field, const Domain &domain,
             const Box3i &box, double time) override {
    if (noise.remove_mean) {
      detail::require_full_box_for_local_noise(domain, box);
      apply(SimulationContext(MPI_COMM_SELF), field, domain, box, time);
      return;
    }
    pfc::field::add_indexed_noise(field, domain, box, noise, MPI_COMM_SELF);
  }

  void apply(const SimulationContext &ctx, field::FieldOutput<double> field,
             const Domain &domain, const Box3i &box, double) override {
    pfc::field::add_indexed_noise(field, domain, box, noise, ctx.mpi_comm());
  }
};

class IndexedNoiseFill : public FieldModifier {
public:
  double offset{0.0};
  field::IndexedNoise noise{};

  const std::string &get_modifier_name() const override {
    static const std::string name{"IndexedNoiseFill"};
    return name;
  }

  void apply(field::FieldOutput<double> field, const Domain &domain,
             const Box3i &box, double time) override {
    detail::require_full_box_for_local_noise(domain, box);
    apply(SimulationContext(MPI_COMM_SELF), field, domain, box, time);
  }

  void apply(const SimulationContext &ctx, field::FieldOutput<double> field,
             const Domain &domain, const Box3i &box, double) override {
    pfc::field::fill_indexed_noise(field, domain, box, offset, noise,
                                   ctx.mpi_comm());
  }
};

} // namespace pfc

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/kernel/grain/label_contacts.hpp>
#include <openpfc/runtime/common/distributed_grain.hpp>

namespace pfc::grain::distributed {
namespace label_detail {
inline Status exception_status() {
  try {
    throw;
  } catch (const std::out_of_range &) {
    return Status::UnknownIdentity;
  } catch (const std::bad_alloc &) {
    return Status::CapacityOverflow;
  } catch (const std::length_error &) {
    return Status::CapacityOverflow;
  } catch (...) {
    return Status::InvalidInput;
  }
}
} // namespace label_detail
struct ContactTrigger {
  Status status = Status::Success;
  bool proximity = false;
  bool unsafe = false;
};
/// Collective cheap flag only; not a recoloring/identity-admission plan. Current
/// registry coherence/retirement remains caller-owned. Local producer failures
/// and mismatched executed radius/generation/slot count reject before any false
/// no-op.
template <class Producer>
ContactTrigger contact_trigger(Producer producer, MPI_Comm comm) {
  ContactTrigger result;
  ContactObservation local;
  try {
    local = producer();
    if (!local.slots || local.slots == unassigned || !local.radius)
      throw std::invalid_argument("invalid scanned UID image layout");
  } catch (...) {
    result.status = label_detail::exception_status();
  }
  result.status = detail::vote(result.status, comm);
  if (result.status != Status::Success) return result;
  const Id metadata[3]{local.radius, local.generation, local.slots};
  Id lower[3]{}, upper[3]{};
  detail::mpi(MPI_Allreduce(metadata, lower, 3, MPI_UINT64_T, MPI_MIN, comm));
  detail::mpi(MPI_Allreduce(metadata, upper, 3, MPI_UINT64_T, MPI_MAX, comm));
  if (lower[0] != upper[0] || lower[1] != upper[1] || lower[2] != upper[2]) {
    result.status = Status::InvalidInput;
    return result;
  }
  unsigned flags = (local.proximity ? 1u : 0u) | (local.unsafe ? 2u : 0u);
  detail::mpi(MPI_Allreduce(MPI_IN_PLACE, &flags, 1, MPI_UNSIGNED, MPI_BOR, comm));
  result.proximity = flags & 1u;
  result.unsafe = flags & 2u;
  if (result.unsafe) result.status = Status::UnsafeCadence;
  return result;
}

/// Exact known-label route. Ghosts supply boundary contacts on device, so only
/// owned UID counts and compact edges are gathered; no shell/component union.
/// Reuses the existing resolver and stage/transfer contract. Complete known UID
/// support and a freshly exchanged Full halo are mandatory producer contracts.
template <class Producer>
Plan prepare_contacts(Producer producer, std::span<const Grain> grains, Slot slots,
                      const remapping::Options &options, MPI_Comm comm,
                      std::size_t max_words = 16 * 1024 * 1024) {
  ContactObservation local;
  Plan plan;
  std::vector<Id> encoded;
  try {
    local = producer();
    if (!slots || slots == unassigned || !local.complete_edges ||
        local.slots != slots || local.radius != options.contact_radius ||
        !detail::supported(options) || !local.radius ||
        pfc::grain::detail::prepare_transfer(slots, grains, {}).status !=
            TransferStatus::Success)
      throw std::invalid_argument("incomplete UID contact production/configuration");
    Observation summary{local.partition, local.support, {}, {}, {}};
    for (auto e : local.edges) summary.contacts.push_back({e.first, e.second});
    encoded = detail::encode(summary, slots, grains, options);
    encoded.push_back(local.unsafe);
    encoded.push_back(local.generation);
    plan.status = Status::Success;
  } catch (...) {
    plan.status = label_detail::exception_status();
  }
  plan.status = detail::vote(plan.status, comm);
  if (plan.status != Status::Success) return plan;
  std::vector<std::vector<Id>> packets;
  try {
    packets = detail::gather(encoded, comm, max_words);
  } catch (const std::length_error &) {
    plan.status = Status::CapacityOverflow;
    return plan;
  }
  try {
    [&] {
      const auto header = 23 + grains.size() * 3;
      std::vector<Partition> partitions;
      std::map<Id, std::uint64_t> counts;
      std::vector<Contact> edges;
      for (auto w : packets) {
        if (w.size() < header + 5 || w.back() != local.generation)
          throw std::invalid_argument("truncated/stale contact metadata");
        w.pop_back();
        const auto unsafe = w.back();
        w.pop_back();
        if (unsafe > 1) throw std::invalid_argument("invalid unsafe flag");
        if (unsafe) {
          plan.status = Status::UnsafeCadence;
          return;
        }
        for (std::size_t i = 0; i < header; ++i)
          if ((i < 7 || i >= 13) && w[i] != encoded[i])
            throw std::invalid_argument("inconsistent global contact configuration");
        plan.exchanged_bytes += (w.size() + 2) * sizeof(Id);
        auto observed = detail::decode(w);
        if (!observed.boundary.empty())
          throw std::invalid_argument("unexpected dense shell");
        partitions.push_back(observed.partition);
        std::set<Id> seen;
        for (auto c : observed.components) {
          auto g = std::lower_bound(grains.begin(), grains.end(), c.key,
                                    [](Grain g, Id id) { return g.id < id; });
          if (!c.key || c.seed != c.key || !c.samples ||
              c.samples > cell_count(observed.partition.local()) ||
              !seen.insert(c.key).second)
            throw std::invalid_argument("invalid owned UID support count");
          if (g == grains.end() || g->id != c.key || !g->active) {
            plan.status = Status::UnknownIdentity;
            return;
          }
          if (g->slot != c.slot || counts[c.key] > UINT64_MAX - c.samples)
            throw std::invalid_argument("UID slot/count mismatch");
          counts[c.key] += c.samples;
        }
        for (auto e : observed.contacts) edges.push_back({e.first, e.second});
      }
      std::size_t volume = 0;
      for (std::size_t r = 0; r < partitions.size(); ++r) {
        auto cells = cell_count(partitions[r].local());
        if (volume > SIZE_MAX - cells)
          throw std::overflow_error("partition volume overflow");
        volume += cells;
        for (std::size_t s = 0; s < r; ++s) {
          bool overlap = true;
          for (int d = 0; d < 3; ++d)
            overlap &= partitions[r].lower[d] <
                           partitions[s].lower[d] + partitions[s].extent[d] &&
                       partitions[s].lower[d] <
                           partitions[r].lower[d] + partitions[r].extent[d];
          if (overlap)
            throw std::invalid_argument("overlapping UID contact partitions");
        }
      }
      if (volume != cell_count(local.partition.global))
        throw std::invalid_argument("UID contact partitions do not cover domain");
      if (!options.check_now) {
        plan.status = Status::Deferred;
        return;
      }
      std::vector<Id> vertices;
      std::vector<double> weights;
      for (auto g : grains)
        if (g.active) {
          if (!counts.contains(g.id)) {
            plan.status = Status::MissingSupport;
            return;
          }
          vertices.push_back(g.id);
          weights.push_back(double(counts.at(g.id)));
        }
      auto graph = make_contact_graph(vertices, edges);
      if (graph.edges.size() > options.contact_capacity) {
        plan.status = Status::CapacityOverflow;
        return;
      }
      auto decision =
          remapping::detail::decide(graph, grains, slots, weights, options);
      if (decision.status != remapping::Status::Success) {
        plan.status = decision.status == remapping::Status::Infeasible
                          ? Status::Infeasible
                          : Status::SearchLimit;
        return;
      }
      auto prepared =
          pfc::grain::detail::prepare_transfer(slots, grains, decision.moves);
      if (prepared.status != TransferStatus::Success)
        throw std::invalid_argument("resolver produced invalid transfer");
      plan.snapshot = {options.epoch, std::move(prepared.grains), std::move(graph)};
      plan.slots = slots;
      plan.global_moves = decision.moves;
      std::set<Id> present;
      for (auto c : local.support) {
        present.insert(c.key);
        plan.bindings.push_back({c.key, c.key});
      }
      for (auto g : grains)
        if (present.contains(g.id)) plan.local_grains.push_back(g);
      for (auto m : decision.moves)
        if (present.contains(m.id)) plan.local_moves.push_back(m);
    }();
  } catch (...) {
    plan.status = label_detail::exception_status();
  }
  plan.status = detail::vote(plan.status, comm);
  if (plan.status != Status::Success) {
    plan.snapshot = {};
    plan.local_grains.clear();
    plan.local_moves.clear();
    plan.bindings.clear();
  }
  return plan;
}
} // namespace pfc::grain::distributed

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/kernel/grain/support_bounds.hpp>
#include <openpfc/runtime/common/distributed_grain.hpp>

namespace pfc::grain::distributed {
/// Explicit conservative alternative to exact occupied-shell contacts.
/// Bounds are globally combined first; buffered box contacts are a superset
/// of exact contacts. False contacts can reject an otherwise feasible palette.
/// Same-slot bounds at distance <=1 reject conservatively before any transfer.
/// Complete admitted UID support is a precondition; no identities are born.
template <class Producer>
Plan prepare_bounds(Producer producer, std::span<const Grain> grains, Slot slots,
                    const remapping::Options &options, MPI_Comm comm,
                    std::size_t max_words = 16 * 1024 * 1024) {
  BoundObservation local;
  Plan plan;
  std::vector<Id> encoded;
  try {
    local = producer();
    validate(local.partition);
    if (!detail::supported(options) || !slots || slots == unassigned ||
        !options.contact_radius ||
        options.contact_radius > std::size_t(INT_MAX / 2) ||
        pfc::grain::detail::prepare_transfer(slots, grains, {}).status !=
            TransferStatus::Success)
      throw std::invalid_argument("invalid bound remapping configuration");
    encoded = detail::encode(Observation{local.partition}, slots, grains, options);
    encoded.resize(23 + grains.size() * 3);
    encoded.push_back(local.bounds.size());
    for (auto b : local.bounds) {
      encoded.insert(encoded.end(), {b.uid, b.slot, b.samples});
      for (int d = 0; d < 3; ++d)
        encoded.insert(encoded.end(), {b.lower[d], b.upper[d]});
    }
    plan.status = Status::Success;
  } catch (const std::bad_alloc &) {
    plan.status = Status::CapacityOverflow;
  } catch (...) {
    plan.status = Status::InvalidInput;
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
      std::map<Id, SupportBounds> combined;
      for (const auto &w : packets) {
        if (w.size() < header + 1)
          throw std::invalid_argument("truncated bound packet");
        for (std::size_t i = 0; i < header; ++i)
          if ((i < 7 || i >= 13) && w[i] != encoded[i])
            throw std::invalid_argument("inconsistent global bound configuration");
        plan.exchanged_bytes += w.size() * sizeof(Id);
        std::vector<Id> base(w.begin(), w.begin() + header);
        base.insert(base.end(), {0, 0, 0});
        auto p = detail::decode(base).partition;
        partitions.push_back(p);
        detail::Reader rd{w, header};
        auto count = rd.next();
        std::set<Id> seen;
        for (std::size_t i = 0; i < count; ++i) {
          SupportBounds b;
          b.uid = rd.next();
          b.slot = Slot(rd.next());
          b.samples = rd.next();
          for (int d = 0; d < 3; ++d) {
            b.lower[d] = rd.next();
            b.upper[d] = rd.next();
          }
          auto g = std::lower_bound(grains.begin(), grains.end(), b.uid,
                                    [](Grain g, Id id) { return g.id < id; });
          if (!b.uid || !seen.insert(b.uid).second || !b.samples ||
              b.samples > cell_count(p.local()))
            throw std::invalid_argument("invalid UID support bound");
          if (g == grains.end() || g->id != b.uid || !g->active) {
            plan.status = Status::UnknownIdentity;
            return;
          }
          if (g->slot != b.slot)
            throw std::invalid_argument("bound UID slot mismatch");
          for (int d = 0; d < 3; ++d)
            if (b.lower[d] > b.upper[d] || b.lower[d] < p.lower[d] ||
                b.upper[d] >= p.lower[d] + p.extent[d])
              throw std::invalid_argument("bound outside owned volume");
          auto [it, inserted] = combined.emplace(b.uid, b);
          if (!inserted) {
            auto &c = it->second;
            if (c.samples > UINT64_MAX - b.samples)
              throw std::overflow_error("bound sample count overflow");
            c.samples += b.samples;
            for (int d = 0; d < 3; ++d) {
              c.lower[d] = std::min(c.lower[d], b.lower[d]);
              c.upper[d] = std::max(c.upper[d], b.upper[d]);
            }
          }
        }
        if (rd.cursor != w.size())
          throw std::invalid_argument("trailing bound metadata");
      }
      std::size_t volume = 0;
      for (std::size_t r = 0; r < partitions.size(); ++r) {
        volume += cell_count(partitions[r].local());
        for (std::size_t s = 0; s < r; ++s) {
          bool overlap = true;
          for (int d = 0; d < 3; ++d)
            overlap &= partitions[r].lower[d] <
                           partitions[s].lower[d] + partitions[s].extent[d] &&
                       partitions[s].lower[d] <
                           partitions[r].lower[d] + partitions[r].extent[d];
          if (overlap) throw std::invalid_argument("overlapping bound partitions");
        }
      }
      if (volume != cell_count(local.partition.global))
        throw std::invalid_argument("bound partitions do not cover global domain");
      if (!options.check_now) {
        plan.status = Status::Deferred;
        return;
      }
      std::vector<Id> vertices;
      std::vector<double> weights;
      std::vector<Contact> contacts;
      for (auto g : grains)
        if (g.active) {
          if (!combined.contains(g.id)) {
            plan.status = Status::MissingSupport;
            return;
          }
          vertices.push_back(g.id);
          weights.push_back(double(combined.at(g.id).samples));
        }
      for (auto a = combined.begin(); a != combined.end(); ++a)
        for (auto b = std::next(a); b != combined.end(); ++b) {
          if (a->second.slot == b->second.slot &&
              within_radius(local.partition.global, a->second, b->second, 1)) {
            plan.status = Status::UnsafeCadence;
            return;
          }
          if (within_radius(local.partition.global, a->second, b->second,
                            options.contact_radius))
            contacts.push_back({a->first, b->first});
        }
      auto graph = make_contact_graph(vertices, contacts);
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
      plan.snapshot = {options.epoch, std::move(prepared.grains), std::move(graph)};
      plan.slots = slots;
      plan.global_moves = decision.moves;
      std::set<Id> present;
      for (auto b : local.bounds) {
        present.insert(b.uid);
        plan.bindings.push_back({b.uid, b.uid});
      }
      for (auto g : grains)
        if (present.contains(g.id)) plan.local_grains.push_back(g);
      for (auto m : decision.moves)
        if (present.contains(m.id)) plan.local_moves.push_back(m);
    }();
  } catch (const std::bad_alloc &) {
    plan.status = Status::CapacityOverflow;
  } catch (...) {
    plan.status = Status::InvalidInput;
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

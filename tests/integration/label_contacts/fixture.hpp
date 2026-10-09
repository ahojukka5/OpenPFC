// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <iostream>
#include <numeric>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/grain/signed_transfer.hpp>
#include <openpfc/runtime/common/label_contacts.hpp>

namespace fixture {
namespace dg = pfc::grain::distributed;
using namespace pfc::grain;
inline void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
inline pfc::Int3 processes(int ranks) {
  if (ranks == 1) return {1, 1, 1};
  if (ranks == 2) return {2, 1, 1};
  if (ranks == 4) return {2, 2, 1};
  if (ranks == 8) return {2, 2, 2};
  throw std::invalid_argument("test supports 1/2/4/8 ranks");
}
struct Seed {
  Id uid;
  Slot slot;
  int x, y, z;
};
struct Case {
  Grid3D global{32, 16, 16, true, true, true, Connectivity::TwentySix};
  Slot slots = 3;
  std::vector<Grain> registry{{101, 0, true}, {211, 0, true}, {307, 1, true}};
  std::vector<Seed> seeds;
  Case(int scenario, Connectivity connectivity, bool periodic) {
    global.connectivity = connectivity;
    global.periodic_x = global.periodic_y = global.periodic_z = periodic;
    seeds = {{101, 0, 15, 7, 7}, {211, 0, 17, 7, 7}, {307, 1, 15, 7, 7}};
    if (scenario == 0) seeds[1] = {211, 0, 25, 12, 12};
    if (scenario == 2) seeds[1] = {211, 0, 17, 9, 9};
    if (scenario == 3) seeds[1] = {211, 0, 16, 7, 7};
    if (scenario == 4) {
      seeds[0] = {101, 0, 0, 0, 0};
      seeds[1] = {211, 0, 31, 0, 0};
      seeds[2] = {307, 1, 0, 0, 0};
    }
    if (scenario == 5) seeds[1] = {211, 0, 17, 9, 7}; // decomposition edge
    if (scenario == 6) { // Safe distance-two periodic corner.
      seeds[0] = {101, 0, 0, 0, 0};
      seeds[1] = {211, 0, 30, 14, 14};
      seeds[2] = {307, 1, 0, 0, 0};
    }
    // One UID also owns a disconnected island; known-label support preserves it.
    seeds.push_back({101, 0, 4, 4, 4});
  }
  Id label(int x, int y, int z, Slot slot) const {
    for (auto p : seeds)
      if (p.x == x && p.y == y && p.z == z && p.slot == slot) return p.uid;
    return 0;
  }
  // Independent global coordinate-pair oracle; no stencil/halo/offset helper.
  std::pair<ContactGraph, bool> oracle(int radius) const {
    std::vector<Contact> edges;
    bool flag = false;
    for (std::size_t a = 0; a < seeds.size(); ++a)
      for (std::size_t b = a + 1; b < seeds.size(); ++b) {
        auto p = seeds[a], q = seeds[b];
        if (p.uid == q.uid) continue;
        int d[3]{std::abs(p.x - q.x), std::abs(p.y - q.y), std::abs(p.z - q.z)};
        const int n[3]{32, 16, 16};
        const bool periodic[3]{global.periodic_x, global.periodic_y,
                               global.periodic_z};
        for (int axis = 0; axis < 3; ++axis)
          if (periodic[axis]) d[axis] = std::min(d[axis], n[axis] - d[axis]);
        bool near = global.connectivity == Connectivity::Six
                        ? d[0] + d[1] + d[2] <= radius
                        : std::max({d[0], d[1], d[2]}) <= radius;
        if (near) {
          edges.push_back({p.uid, q.uid});
          flag |= p.slot == q.slot;
        }
      }
    return {make_contact_graph({101, 211, 307}, edges), flag};
  }
  std::vector<Id> owned(dg::Partition p) const {
    auto n = cell_count(p.local());
    std::vector<Id> out(n * slots);
    for (std::size_t c = 0; c < n; ++c) {
      auto global_cell = dg::global_cell(p, c);
      for (Slot s = 0; s < slots; ++s)
        out[s * n + c] = label(global_cell % 32, (global_cell / 32) % 16,
                               global_cell / (32 * 16), s);
    }
    return out;
  }
};
inline dg::LabelHalo layout(const pfc::decomposition::Decomposition &d, int rank,
                            Grid3D global) {
  auto b = pfc::decomposition::local_box(d, rank);
  return {{global,
           {std::size_t(b.low[0]), std::size_t(b.low[1]), std::size_t(b.low[2])},
           {std::size_t(b.size[0]), std::size_t(b.size[1]), std::size_t(b.size[2])}},
          3,
          7,
          true};
}
inline std::vector<double> padded(const Case &c, dg::LabelHalo h) {
  std::vector<double> out(h.field_size() * c.slots, -123.5);
  auto owned = c.owned(h.partition);
  auto n = cell_count(h.partition.local());
  for (std::size_t i = 0; i < n; ++i)
    for (Slot s = 0; s < c.slots; ++s)
      out[s * h.field_size() +
          h.at(i % h.partition.extent[0],
               (i / h.partition.extent[0]) % h.partition.extent[1],
               i / (h.partition.extent[0] * h.partition.extent[1]))] =
          double(owned[s * n + i]);
  return out;
}
template <class Scan>
void verify(const Case &c, dg::LabelHalo h, Scan scan, int rank, int ranks) {
  for (int radius : {1, 2}) {
    auto oracle = c.oracle(radius);
    auto trigger = dg::contact_trigger([&] { return scan(radius, false, 4096, h); },
                                       MPI_COMM_WORLD);
    require(trigger.proximity == oracle.second,
            "global image proximity differs from independent oracle");
    require(trigger.unsafe == c.oracle(1).second,
            "unsafe adjacency differs from independent oracle");
    require(trigger.status ==
                (trigger.unsafe ? dg::Status::UnsafeCadence : dg::Status::Success),
            "probe status");
    pfc::grain::remapping::Options options;
    options.epoch = 7;
    options.contact_radius = radius;
    auto plan = dg::prepare_contacts([&] { return scan(radius, true, 4096, h); },
                                     c.registry, c.slots, options, MPI_COMM_WORLD);
    if (trigger.unsafe) {
      require(plan.status == dg::Status::UnsafeCadence, "unsafe state accepted");
      continue;
    }
    if (c.slots == 2 && oracle.first.edges.size() == 3) {
      require(plan.status == dg::Status::SearchLimit,
              "incremental palette failure mislabeled as complete proof");
      auto complete = options;
      complete.method = pfc::grain::remapping::Method::CompleteOracle;
      require(dg::prepare_contacts([&] { return scan(radius, true, 4096, h); },
                                   c.registry, c.slots, complete, MPI_COMM_WORLD)
                      .status == dg::Status::Infeasible,
              "genuine two-slot palette accepted a triangle");
      continue;
    }
    require(plan.status == dg::Status::Success,
            "exact contact preparation rejected");
    require(plan.snapshot.graph.edges == oracle.first.edges,
            "global graph differs from independent oracle");
    auto disabled = options;
    disabled.check_now = false;
    require(dg::prepare_contacts([&] { return scan(radius, true, 4096, h); },
                                 c.registry, c.slots, disabled, MPI_COMM_WORLD)
                    .status == dg::Status::Deferred,
            "disabled detection control ignored");
    require(dg::prepare_contacts([&] { return scan(radius, true, 4096, h); },
                                 c.registry, c.slots + 1, options, MPI_COMM_WORLD)
                    .status == dg::Status::InvalidInput,
            "extra-empty-slot producer/global layout mismatch accepted");
    auto absent = c.registry;
    absent.push_back({449, 1, true});
    require(dg::prepare_contacts([&] { return scan(radius, true, 4096, h); }, absent,
                                 c.slots, options, MPI_COMM_WORLD)
                    .status == dg::Status::MissingSupport,
            "globally absent UID silently retained");
    auto unordered = c.registry;
    std::reverse(unordered.begin(), unordered.end());
    require(dg::prepare_contacts([&] { return scan(radius, true, 4096, h); },
                                 unordered, c.slots, options, MPI_COMM_WORLD)
                    .status == dg::Status::InvalidInput,
            "noncanonical UID registry accepted");
    require(plan.global_moves.empty() == !trigger.proximity,
            "recolor only on actual same-slot contacts");
    auto labels = c.owned(h.partition);
    std::vector<double> values(labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i)
      if (labels[i]) values[i] = labels[i] == 307 ? 0 : -.25;
    auto original = values;
    auto apply = [&](auto grains, auto moves) {
      return transfer_signed(h.partition.local(), c.slots,
                             std::span<const double>(values),
                             std::span<const Id>(labels), grains, moves);
    };
    auto staged = dg::stage(plan, MPI_COMM_WORLD, apply);
    require(staged.status == dg::Status::Success,
            "signed collective publication rejected");
    require(values == original, "accepted input mutated by contact path");
    auto expected = transfer_signed(
        h.partition.local(), c.slots, std::span<const double>(values),
        std::span<const Id>(labels), plan.local_grains, plan.local_moves);
    require(staged.transaction->values == expected.values &&
                staged.transaction->labels == expected.labels,
            "signed stage mismatch");
    auto capacity =
        dg::prepare_contacts([&] { return scan(radius, true, 0, h); }, c.registry,
                             c.slots, options, MPI_COMM_WORLD);
    require(capacity.status == dg::Status::CapacityOverflow,
            "edge capacity silently truncates");
    auto stale = h;
    stale.generation = 6;
    auto rejected = dg::contact_trigger(
        [&] { return scan(radius, false, 4096, rank == 0 ? stale : h); },
        MPI_COMM_WORLD);
    require(rejected.status == dg::Status::InvalidInput,
            "one-rank stale accepted halo accepted");
    auto thin = h;
    thin.width = 1;
    rejected = dg::contact_trigger([&] { return scan(2, false, 4096, thin); },
                                   MPI_COMM_WORLD);
    require(rejected.status == dg::Status::InvalidInput,
            "insufficient halo accepted");
    auto faces = h;
    faces.full = false;
    rejected = dg::contact_trigger([&] { return scan(radius, false, 4096, faces); },
                                   MPI_COMM_WORLD);
    require(rejected.status == dg::Status::InvalidInput,
            "Faces halo silently accepted for true3D contacts");
    if (ranks > 1) {
      rejected = dg::contact_trigger(
          [&] {
            auto observed = scan(radius, false, 4096, h);
            if (rank == 0) ++observed.slots;
            return observed;
          },
          MPI_COMM_WORLD);
      require(rejected.status == dg::Status::InvalidInput,
              "rank-dependent scanned OP count accepted");
      rejected = dg::contact_trigger(
          [&] { return scan(rank == 0 ? 1 : 2, false, 4096, h); }, MPI_COMM_WORLD);
      require(rejected.status == dg::Status::InvalidInput,
              "rank-dependent executed contact radius accepted");
    }
    require(dg::prepare_contacts([&] { return scan(radius, false, 4096, h); },
                                 c.registry, c.slots, options, MPI_COMM_WORLD)
                    .status == dg::Status::InvalidInput,
            "flag-only observation accepted as full graph");
    require(dg::prepare_contacts([&] { return scan(radius, true, 4096, h); },
                                 c.registry, c.slots, options, MPI_COMM_WORLD, 1)
                    .status == dg::Status::CapacityOverflow,
            "collective metadata capacity ignored");
    if (rank == 0)
      std::cout << "label contacts ranks=" << ranks << " radius=" << radius
                << " edges=" << oracle.first.edges.size()
                << " moves=" << plan.global_moves.size() << " verified\n";
  }
}
} // namespace fixture

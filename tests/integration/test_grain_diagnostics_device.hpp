// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "grain_remapping_cases.hpp"
#include "test_grain_operations_diagnostics_device.hpp"
#include <openpfc/runtime/gpu/grain_remapping.hpp>
TEST_CASE("all device policies and expected failures preserve observer parity",
          "[grain][diagnostics]") {
  available();
  for (auto method :
       {remapping::Method::Incremental, remapping::Method::GlobalSaturation,
        remapping::Method::GlobalLargestFirst, remapping::Method::CompleteOracle})
    for (bool componentwise : {false, true})
      for (unsigned failure = 0; failure < 10; ++failure) {
        auto fixture = remapping_test::pair();
        fixture.options.method = method;
        fixture.options.componentwise = componentwise;
        if (failure == 1) fixture.options.contact_capacity = 0;
        if (failure == 2) fixture.options.limits.attempts = 0;
        if (failure == 3) fixture.options.check_now = false;
        if (failure == 4) fixture.seeds[34] = 0;
        if (failure == 5) fixture.values[34] = -1;
        if (failure == 6) fixture.seeds[34] = 99;
        if (failure == 7) fixture.options.max_sweeps = 0;
        if (failure == 8)
          fixture.grains.insert(fixture.grains.begin() + 2, {70, 2, true});
        if (failure == 9) {
          fixture.values[35] = .125;
          fixture.seeds[35] = 42;
        }
        pfc::core::DataBuffer<Backend, double> values(fixture.values.size());
        values.copy_from_host(fixture.values);
        pfc::core::DataBuffer<Backend, Id> seeds(fixture.seeds.size());
        seeds.copy_from_host(fixture.seeds);
        auto plain = remapping::remap(fixture.grid, fixture.slots, values, seeds,
                                      fixture.grains, fixture.options);
        Diagnostics d;
        d.host_observer_connected = true;
        fixture.options.diagnostics = &d;
        auto observed = remapping::remap(fixture.grid, fixture.slots, values, seeds,
                                         fixture.grains, fixture.options);
        REQUIRE(observed.status == plain.status);
        REQUIRE(observed.transfer_status == plain.transfer_status);
        REQUIRE(observed.values.to_host() == plain.values.to_host());
        REQUIRE(observed.labels.to_host() == plain.labels.to_host());
        REQUIRE(observed.snapshot.graph.edges == plain.snapshot.graph.edges);
        REQUIRE(observed.snapshot.graph.vertices == plain.snapshot.graph.vertices);
        REQUIRE(observed.snapshot.grains.size() == plain.snapshot.grains.size());
        for (std::size_t i = 0; i < observed.snapshot.grains.size(); ++i)
          REQUIRE(observed.snapshot.grains[i].slot == plain.snapshot.grains[i].slot);
        REQUIRE(d.source_accesses_complete);
        REQUIRE_FALSE(d.observation_failed);
        if (failure == 3) {
          REQUIRE(d.allocations(dg::Space::Device).requests == 0);
          REQUIRE_FALSE(d.events[0].available);
        } else {
          REQUIRE(d.allocations(dg::Space::Device).requests > 0);
          REQUIRE(d.events[0].available);
        }
        if (observed.status != remapping::Status::Success)
          REQUIRE(d.allocations(dg::Space::Device).live_bytes == 0);
        REQUIRE(values.to_host() == fixture.values);
        REQUIRE(seeds.to_host() == fixture.seeds);
      }
}

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "grain_remapping.hpp"
#include "grain_remapping_cases.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <openpfc/runtime/gpu/grain_remapping.hpp>

#if defined(__HIPCC__) || defined(__HIP__)
using Backend = pfc::backend::HIPTag;
#else
using Backend = pfc::backend::CUDATag;
#endif

namespace {
using namespace remapping_test;
void device_available() {
  int count = 0;
#if defined(__HIPCC__) || defined(__HIP__)
  REQUIRE(hipGetDeviceCount(&count) == hipSuccess);
#else
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
#endif
  REQUIRE(count > 0); // Host-only skips never masquerade as parity.
}
__global__ void produce(double *values, std::size_t index, double q) {
  if (blockIdx.x == 0 && threadIdx.x == 0) values[index] = q;
}
void parity(const Case &fixture, bool asynchronous_producer = false) {
  const auto expected = run(fixture);
  pfc::core::DataBuffer<Backend, double> values(fixture.values.size());
  pfc::core::DataBuffer<Backend, Id> labels(fixture.seeds.size());
  auto uploaded = fixture.values;
  if (asynchronous_producer) uploaded[34] = 0;
  values.copy_from_host(uploaded);
  labels.copy_from_host(fixture.seeds);
  pfc::gpuStream_t stream;
  GPU_CHECK(pfc::gpuStreamCreateWithFlags(&stream, pfc::gpuStreamNonBlocking));
  if (asynchronous_producer)
    GPU_LAUNCH_KERNEL(produce, 1, 1,
                      (values.data(), std::size_t{34}, fixture.values[34]), stream);
  auto actual = remapping::remap(fixture.grid, fixture.slots, values, labels,
                                 fixture.grains, fixture.options, stream);
  GPU_CHECK(pfc::gpuStreamDestroy(stream));
  REQUIRE(actual.status == expected.status);
  REQUIRE(actual.transfer_status == expected.transfer_status);
  REQUIRE(actual.values.to_host() == expected.values);
  REQUIRE(actual.labels.to_host() == expected.labels);
  REQUIRE(actual.snapshot.epoch == expected.snapshot.epoch);
  REQUIRE(actual.snapshot.graph.vertices == expected.snapshot.graph.vertices);
  REQUIRE(actual.snapshot.graph.edges == expected.snapshot.graph.edges);
  REQUIRE(actual.snapshot.grains.size() == expected.snapshot.grains.size());
  for (std::size_t i = 0; i < expected.snapshot.grains.size(); ++i) {
    REQUIRE(actual.snapshot.grains[i].id == expected.snapshot.grains[i].id);
    REQUIRE(actual.snapshot.grains[i].slot == expected.snapshot.grains[i].slot);
    REQUIRE(actual.snapshot.grains[i].active == expected.snapshot.grains[i].active);
  }
  REQUIRE(actual.statistics.propagation_sweeps ==
          expected.statistics.propagation_sweeps);
  REQUIRE(actual.statistics.edges == expected.statistics.edges);
  REQUIRE(actual.statistics.moved_samples == expected.statistics.moved_samples);
  REQUIRE(actual.statistics.changed_grains == expected.statistics.changed_grains);
  REQUIRE(actual.statistics.attempts == expected.statistics.attempts);
  REQUIRE(actual.statistics.conflict == expected.statistics.conflict);
  REQUIRE(actual.statistics.moved_value_bytes ==
          expected.statistics.moved_value_bytes);
  REQUIRE(actual.statistics.moved_label_bytes ==
          expected.statistics.moved_label_bytes);
  REQUIRE(actual.statistics.published_storage_bytes ==
          expected.statistics.published_storage_bytes);
  REQUIRE(actual.statistics.staged_storage_bytes ==
          expected.statistics.staged_storage_bytes);
  for (auto duration :
       {actual.statistics.total_seconds, actual.statistics.detection_seconds,
        actual.statistics.adjacency_seconds, actual.statistics.decision_seconds,
        actual.statistics.transfer_seconds,
        actual.statistics.synchronization_seconds}) {
    REQUIRE(std::isfinite(duration));
    REQUIRE(duration >= 0);
    REQUIRE(actual.statistics.total_seconds >= duration);
  }
  if (actual.status == remapping::Status::Success) {
    REQUIRE(actual.statistics.wrapper_storage_bytes >=
            fixture.values.size() * (sizeof(Id) + 1));
    REQUIRE(actual.statistics.graph_device_to_host_bytes ==
            actual.statistics.edges * sizeof(Contact));
  }
  const auto original = values.to_host();
  REQUIRE(std::memcmp(original.data(), fixture.values.data(),
                      original.size() * sizeof(double)) == 0);
  REQUIRE(labels.to_host() == fixture.seeds);
}
} // namespace

TEST_CASE("device remapping publishes or rejects with deterministic CPU parity",
          "[grain][remapping][parity]") {
  device_available();
  auto fixture = pair();
  parity(fixture, true); // Real producer on a nonblocking stream, no pre-call wait.
  for (auto method :
       {remapping::Method::Incremental, remapping::Method::GlobalSaturation,
        remapping::Method::GlobalLargestFirst, remapping::Method::CompleteOracle}) {
    fixture.options.method = method;
    parity(fixture);
    fixture.options.limits.attempts = 0;
    parity(fixture);
    fixture.options.limits.attempts = 1000000;
  }
  fixture = impossible();
  parity(fixture);
  fixture = pair();
  fixture.options.contact_capacity = 0;
  parity(fixture);
  fixture = pair();
  fixture.options.check_now = false;
  parity(fixture);
  fixture = pair();
  fixture.options.max_sweeps = 0;
  parity(fixture);
  fixture = pair();
  put(fixture, 0, 5, 2, 0, 0);
  put(fixture, 0, 3, 2, 42, .25);
  parity(fixture);
  fixture = pair();
  fixture.values[0] = 1.e-100;
  parity(fixture);
  fixture = pair();
  fixture.values[0] = std::numeric_limits<double>::quiet_NaN();
  fixture.seeds[34] = 999;
  parity(fixture);
  fixture = pair();
  fixture.values[0] = -.1;
  parity(fixture);
  fixture = pair();
  fixture.grains[0].slot = 1;
  parity(fixture);
  fixture = pair();
  put(fixture, 0, 5, 2, 0, 0);
  parity(fixture);
  fixture.grains[1] = retire(fixture.grains[1]);
  parity(fixture);
  fixture = pair();
  put(fixture, 0, 1, 2, 0, .5);
  put(fixture, 0, 0, 2, 0, .5);
  fixture.options.max_sweeps = 1;
  parity(fixture);
  fixture.options.max_sweeps = 2;
  parity(fixture);
  fixture = pair();
  fixture.values.assign(fixture.values.size(), 0);
  for (auto &grain : fixture.grains) grain = retire(grain);
  parity(fixture);
}

TEST_CASE("device remapping matches every CPU collision observation",
          "[grain][remapping][parity][evolution]") {
  device_available();
  for (bool seam : {false, true}) {
    auto state = grain_example::initial(seam);
    for (int step = 0; step < 6; ++step) {
      Case fixture;
      fixture.grid = state.grid;
      fixture.slots = state.slots;
      fixture.values = state.values;
      fixture.seeds = state.labels;
      fixture.grains = state.grains;
      fixture.options.epoch = state.epoch + 1;
      parity(fixture);
      auto before = run(fixture);
      fixture.values =
          grain_example::advance_values(state.grid, state.slots, before.values);
      fixture.seeds = before.labels;
      fixture.grains = before.snapshot.grains;
      fixture.options.epoch = state.epoch + 2;
      parity(fixture);
      REQUIRE(grain_example::advance(state) == remapping::Status::Success);
    }
  }
}

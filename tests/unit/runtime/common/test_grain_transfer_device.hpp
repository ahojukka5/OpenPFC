// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "grain_transfer_cases.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <limits>
#include <openpfc/runtime/gpu/grain_transfer.hpp>

#if defined(__HIPCC__) || defined(__HIP__)
using Backend = pfc::backend::HIPTag;
#else
using Backend = pfc::backend::CUDATag;
#endif

namespace {
using namespace transfer_test;

__global__ void produce_transfer_value(double *values, double value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) values[0] = value;
}

void check_parity(const Case &fixture, bool asynchronous_producer = false) {
  const auto expected = run(fixture);
  pfc::core::DataBuffer<Backend, double> values(fixture.values.size());
  pfc::core::DataBuffer<Backend, Id> labels(fixture.labels.size());
  auto uploaded = fixture.values;
  if (asynchronous_producer) uploaded[0] = fixture.background_value;
  values.copy_from_host(uploaded);
  labels.copy_from_host(fixture.labels);
  pfc::gpuStream_t stream;
  GPU_CHECK(pfc::gpuStreamCreateWithFlags(&stream, pfc::gpuStreamNonBlocking));
  if (asynchronous_producer)
    GPU_LAUNCH_KERNEL(produce_transfer_value, 1, 1,
                      (values.data(), fixture.values[0]), stream);
  const auto result = pfc::grain::transfer(fixture.grid, fixture.slots, values,
                                           labels, fixture.grains, fixture.moves,
                                           fixture.background_value, stream);
  GPU_CHECK(pfc::gpuStreamDestroy(stream));
  REQUIRE(result.status == expected.status);
  REQUIRE(result.values.to_host() == expected.values);
  REQUIRE(result.labels.to_host() == expected.labels);
  REQUIRE(result.grains.size() == expected.grains.size());
  for (std::size_t i = 0; i < expected.grains.size(); ++i) {
    REQUIRE(result.grains[i].id == expected.grains[i].id);
    REQUIRE(result.grains[i].slot == expected.grains[i].slot);
    REQUIRE(result.grains[i].active == expected.grains[i].active);
  }
  // Device originals are unchanged even when preflight rejects the batch.
  const auto original = values.to_host();
  REQUIRE(original.size() == fixture.values.size());
  REQUIRE(std::memcmp(original.data(), fixture.values.data(),
                      original.size() * sizeof(double)) == 0);
  REQUIRE(labels.to_host() == fixture.labels);
}
} // namespace

TEST_CASE("device grain transfer matches CPU cycles and shared destinations",
          "[grain][transfer][parity]") {
  // No device means this test fails; a host-only pass is not device parity.
  int count = 0;
#if defined(__HIPCC__) || defined(__HIP__)
  REQUIRE(hipGetDeviceCount(&count) == hipSuccess);
#else
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
#endif
  REQUIRE(count > 0);
  check_parity(shared_destination());
  check_parity(cycle());
  check_parity(signed_cycle());
  auto fixture = shared_destination();
  fixture.moves.clear();
  check_parity(fixture);
  fixture.moves = {{fixture.grains.back().id, 0, 0}};
  check_parity(fixture);
}

TEST_CASE("device grain transfer rejects the same unsafe batches as CPU",
          "[grain][transfer][parity]") {
  auto fixture = shared_destination();
  SECTION("occupied destination") { put(fixture, 1, 0, 22, .5); }
  SECTION("two final arrivals") {
    fixture = cycle();
    fixture.moves = {{11, 0, 2}, {22, 1, 2}};
  }
  SECTION("duplicate UID") { fixture.moves.push_back(fixture.moves.front()); }
  SECTION("unknown UID") { fixture.moves.front().id = 999; }
  SECTION("stale source") { fixture.moves.front().source = 1; }
  SECTION("unlabeled tail") { fixture.values[5] = 1.e-12; }
  SECTION("label on background") { fixture.labels[5] = 22; }
  SECTION("unknown label") { fixture.labels[0] = 999; }
  SECTION("misplaced support") { put(fixture, 2, 5, 22, .25); }
  SECTION("nonfinite field") {
    fixture.values[0] = std::numeric_limits<double>::infinity();
  }
  SECTION("NaN field") {
    fixture.values[0] = std::numeric_limits<double>::quiet_NaN();
  }
  SECTION("empty grain") {
    for (const auto cell : {0u, 3u, 7u}) {
      fixture.values[cell] = 0;
      fixture.labels[cell] = 0;
    }
  }
  SECTION("invalid layout") { fixture.labels.pop_back(); }
  SECTION("invalid registry") { fixture.grains.front().slot = 3; }
  REQUIRE(run(fixture).status != TransferStatus::Success);
  check_parity(fixture);
}

TEST_CASE("nonblocking transfer orders caller production and default metadata",
          "[grain][transfer][stream]") {
  check_parity(shared_destination(), true);
  auto rejected = shared_destination();
  put(rejected, 1, 0, 22, .5);
  REQUIRE(run(rejected).status == TransferStatus::OccupiedDestination);
  check_parity(rejected, true);
}

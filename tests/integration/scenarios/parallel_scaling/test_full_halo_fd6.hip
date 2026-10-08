// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "fixtures/full_halo_fd6.hpp"
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <string>

namespace {
__global__ void publish(double *data, int nx, int ny, int nz, int ox, int oy, int oz,
                        int field, int stage) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= nx * ny * nz) return;
  const int i = q % nx, j = (q / nx) % ny, k = q / (nx * ny);
  const int idx = (i + 3) + (nx + 6) * ((j + 3) + (ny + 6) * (k + 3));
  data[idx] = 1.0 + field * 0.25 + stage * 16777216.0 + (i + ox) + 128.0 * (j + oy) +
              16384.0 * (k + oz);
}
struct Env {
  std::string value;
  bool existed;
  Env() : existed(std::getenv("OPENPFC_HIP_FORCE_PACKED_HALO") != nullptr) {
    if (existed) value = std::getenv("OPENPFC_HIP_FORCE_PACKED_HALO");
  }
  ~Env() {
    if (existed)
      setenv("OPENPFC_HIP_FORCE_PACKED_HALO", value.c_str(), 1);
    else
      unsetenv("OPENPFC_HIP_FORCE_PACKED_HALO");
  }
};
} // namespace
TEST_CASE("HIP Full FD6 exact halos after asynchronous producer",
          "[MPI][full_halo_fd6][hip]") {
  int devices = 0, rank = 0;
  REQUIRE(hipGetDeviceCount(&devices) == hipSuccess);
  REQUIRE(devices > 0);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  REQUIRE(hipSetDevice(rank % devices) == hipSuccess);
  hipDeviceProp_t properties{};
  REQUIRE(hipGetDeviceProperties(&properties, rank % devices) == hipSuccess);
  char pci_bus[32]{};
  REQUIRE(hipDeviceGetPCIBusId(pci_bus, sizeof(pci_bus), rank % devices) ==
          hipSuccess);
  // MPI_Worker mutes std::cout on non-root ranks; retain every rank's device.
  std::fprintf(
      stderr,
      "Full FD6 native HIP rank=%d visible_devices=%d selected=%d name=%s pci=%s\n",
      rank, devices, rank % devices, properties.name, pci_bus);
  auto producer = [](auto &u, int field, int stage) {
    // Upload different owned values so a stale producer/read is observable.
    full_halo_fd6::initialize(u, field, stage + 9);
    u.sync_to_device();
    auto n = u.size3();
    auto g = u.global(0, 0, 0);
    publish<<<(n[0] * n[1] * n[2] + 127) / 128, 128>>>(
        u.data(), n[0], n[1], n[2], g[0], g[1], g[2], field, stage);
    REQUIRE(hipGetLastError() == hipSuccess);
    // Deliberately no device synchronization: exchange must wait for producer.
    u.note_device_write();
  };
  Env restore;
  for (const char *packed : {"0", "1"}) {
    setenv("OPENPFC_HIP_FORCE_PACKED_HALO", packed, 1);
    for (const auto periodic :
         {pfc::Bool3{true, true, true}, pfc::Bool3{false, false, false},
          pfc::Bool3{false, true, false}})
      full_halo_fd6::run<pfc::HIPSpace>(periodic, false, producer);
    full_halo_fd6::run<pfc::HIPSpace>({true, true, false}, true, producer);
  }
  full_halo_fd6::thin_rejection<pfc::HIPSpace>();
}

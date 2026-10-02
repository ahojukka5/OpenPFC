// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file device_advance.cpp
 * @brief Host-side owner of the HIP HeFFTe velocity step.
 *
 * Compiled into `incompressible_flow_hip` only. The device kernels live
 * in `openpfc_incompressible_hip`.
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "incompressible HIP advance requires the HIP spectral backend"
#endif

#include <flow/device_session.hpp>
#include <flow/taylor_green.hpp>

#include <iostream>
#include <optional>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/incompressible_device.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>

namespace flow {

struct DeviceSession {
  // The stack is not movable. Construct it in place, and destroy the
  // velocity buffers before the FFT they were planned against.
  std::optional<pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>> stack;
  pfc::field::DeviceVelocity<pfc::HIPSpace> velocity;
};

void destroy_device_session(DeviceSession *session) { delete session; }

DeviceSessionPtr start_device_session(State &state, int rank, int nproc) {
  auto session = DeviceSessionPtr(new DeviceSession());
  auto domain = pfc::domain::create(
      pfc::GridSize({state.n[0], state.n[1], state.n[2]}),
      pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({state.spacing[0], state.spacing[1], state.spacing[2]}));
  session->stack.emplace(std::move(domain), rank, nproc, MPI_COMM_WORLD);
  pfc::field::prepare_device_velocity(session->velocity, session->stack->fft(),
                                      state.n, state.spacing, state.nu, state.dt);
  pfc::field::upload_device_velocity(session->velocity, state.u, state.v, state.w);
  GPU_CHECK(pfc::gpuDeviceSynchronize());

  unsigned long long local = static_cast<unsigned long long>(
      pfc::field::device_velocity_bytes(session->velocity));
  unsigned long long total = 0;
  MPI_Reduce(&local, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "incompressible_flow device_bytes=" << total << std::endl;
  }
  return session;
}

void step_device_session(DeviceSession &session) {
  pfc::field::step_device_velocity(session.velocity, session.stack->fft(), true);
}

void finish_device_session(DeviceSession &session, State &state) {
  GPU_CHECK(pfc::gpuDeviceSynchronize());
  pfc::field::download_device_velocity(session.velocity, state.u, state.v, state.w);
}

} // namespace flow

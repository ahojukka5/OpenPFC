// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file device_session.hpp
 * @brief HIP HeFFTe session for one periodic velocity.
 *
 * The host state still owns initialization and diagnostics. The session
 * owns the device integrating-factor step between those samples.
 */

#include <memory>

namespace flow {

struct State;

struct DeviceSession;

void destroy_device_session(DeviceSession *session);

struct DeviceSessionDeleter {
  void operator()(DeviceSession *session) const noexcept {
    destroy_device_session(session);
  }
};

using DeviceSessionPtr = std::unique_ptr<DeviceSession, DeviceSessionDeleter>;

/// Upload the current hats and build the rocFFT pencil for this state.
[[nodiscard]] DeviceSessionPtr start_device_session(State &state, int rank,
                                                    int nproc);

/// One integrating-factor step. The hats stay on the device.
void step_device_session(DeviceSession &session);

/// Copy the hats back into `state` for a host diagnostic sample.
void finish_device_session(DeviceSession &session, State &state);

} // namespace flow

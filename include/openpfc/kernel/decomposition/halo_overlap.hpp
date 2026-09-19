// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file halo_overlap.hpp
 * @brief Overlap policy for split-phase Faces halo + interior stencil.
 *
 * @details
 * Device Faces already exposes `start()` / `finish()` / `progress()`.
 * Apps still copied the same 0/1/2 integer policy (blocking Waitall vs
 * Testall pump). This enum is the shared name; GPU stream ownership
 * lives in `runtime/gpu/halo_overlap_gpu.hpp`.
 *
 * Integers stay 0/1/2 so existing env knobs (`HEAT3D_HALO_OVERLAP`) map
 * without a silent shift. The library does not read those env names.
 *
 * @see runtime/gpu/halo_overlap_gpu.hpp
 */

#include <stdexcept>
#include <string>

namespace pfc::comm {

/// How a GPU Faces step waits for halo MPI relative to interior work.
enum class HaloOverlapMode : int {
  /// Blocking `exchange()`; no interior/border split.
  Blocking = 0,
  /// Interior on a non-blocking stream, then `start()` / `finish()`.
  Waitall = 1,
  /// Same as Waitall, pumping `progress()` until the interior event and
  /// MPI both complete.
  Testall = 2
};

/// Map a 0/1/2 integer to @ref HaloOverlapMode. Other values fail closed.
inline HaloOverlapMode halo_overlap_mode_from_int(int value) {
  if (value < 0 || value > 2) {
    throw std::invalid_argument(
        "HaloOverlapMode must be 0 (Blocking), 1 (Waitall), or 2 (Testall); got " +
        std::to_string(value));
  }
  return static_cast<HaloOverlapMode>(value);
}

} // namespace pfc::comm

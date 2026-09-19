// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file halo_overlap_gpu.hpp
 * @brief Overlap a device interior stencil with Faces pack/MPI.
 *
 * @details
 * Owns a non-blocking compute stream so default-stream pack/MPI does not
 * drain the interior kernel. Waitall posts Faces and `finish()`-waits.
 * Testall records an interior-done event and pumps `halo.progress()`
 * until that event and MPI both complete (H-progress-2).
 *
 * `rhs(halo, inner, border)` is the production sequence:
 * `inner(stream)` → `start()` → optional Testall pump → `finish()` →
 * `border()`. Blocking mode is rejected; call `halo.exchange()` instead.
 *
 * The helper does not read env knobs. Apps keep names such as
 * `HEAT3D_HALO_OVERLAP` and any single-rank default policy.
 *
 * @see kernel/decomposition/halo_overlap.hpp
 * @see runtime/gpu/comm_halo_exchange_gpu.hpp
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <stdexcept>
#include <string>

#include <openpfc/kernel/decomposition/halo_overlap.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>

namespace pfc::comm {

/// Two-stream Faces overlap: interior on a non-blocking compute stream.
class GpuHaloOverlap {
public:
  /// @throws std::invalid_argument if @p mode is Blocking.
  explicit GpuHaloOverlap(HaloOverlapMode mode) : m_mode(mode) {
    if (mode == HaloOverlapMode::Blocking) {
      throw std::invalid_argument("GpuHaloOverlap requires Waitall or Testall");
    }
    GPU_CHECK(gpuStreamCreateWithFlags(&m_compute, gpuStreamNonBlocking));
    if (mode == HaloOverlapMode::Testall) {
      GPU_CHECK(gpuEventCreate(&m_inner_done));
    }
  }

  ~GpuHaloOverlap() { destroy_(); }

  GpuHaloOverlap(const GpuHaloOverlap &) = delete;
  GpuHaloOverlap &operator=(const GpuHaloOverlap &) = delete;

  GpuHaloOverlap(GpuHaloOverlap &&other) noexcept
      : m_mode(other.m_mode), m_compute(other.m_compute),
        m_inner_done(other.m_inner_done) {
    other.m_compute = gpuStream_t{};
    other.m_inner_done = gpuEvent_t{};
  }

  GpuHaloOverlap &operator=(GpuHaloOverlap &&other) noexcept {
    if (this != &other) {
      destroy_();
      m_mode = other.m_mode;
      m_compute = other.m_compute;
      m_inner_done = other.m_inner_done;
      other.m_compute = gpuStream_t{};
      other.m_inner_done = gpuEvent_t{};
    }
    return *this;
  }

  [[nodiscard]] HaloOverlapMode mode() const noexcept { return m_mode; }

  /// Non-blocking stream for the halo-independent interior kernel.
  [[nodiscard]] gpuStream_t compute_stream() const noexcept { return m_compute; }

  /// Record the Testall interior-done event on @ref compute_stream.
  void record_inner_done() {
    require_testall_("record_inner_done");
    GPU_CHECK(gpuEventRecord(m_inner_done, m_compute));
  }

  /// Pump `halo.progress()` until the interior event and MPI complete.
  template <class Halo> void pump(Halo &halo) {
    require_testall_("pump");
    bool gpu_done = false;
    for (;;) {
      if (!gpu_done) {
        const gpuError_t q = gpuEventQuery(m_inner_done);
        if (q == gpuSuccess) {
          gpu_done = true;
        } else if (q != gpuErrorNotReady) {
          GPU_CHECK(q);
        }
      }
      const bool mpi_done = halo.progress();
      if (gpu_done && mpi_done) {
        break;
      }
    }
  }

  /// Interior → Faces start → optional Testall pump → finish → border.
  template <class Halo, class Inner, class Border>
  void rhs(Halo &halo, Inner &&inner, Border &&border) {
    inner(m_compute);
    if (m_mode == HaloOverlapMode::Testall) {
      record_inner_done();
    }
    halo.start();
    if (m_mode == HaloOverlapMode::Testall) {
      pump(halo);
    }
    halo.finish();
    border();
  }

private:
  void require_testall_(const char *what) const {
    if (m_mode != HaloOverlapMode::Testall) {
      throw std::logic_error(std::string("GpuHaloOverlap::") + what +
                             " requires HaloOverlapMode::Testall");
    }
  }

  void destroy_() noexcept {
    if (m_inner_done) {
      (void)gpuEventDestroy(m_inner_done);
      m_inner_done = gpuEvent_t{};
    }
    if (m_compute) {
      (void)gpuStreamDestroy(m_compute);
      m_compute = gpuStream_t{};
    }
  }

  HaloOverlapMode m_mode;
  gpuStream_t m_compute{};
  gpuEvent_t m_inner_done{};
};

} // namespace pfc::comm

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>

namespace pfc::grain::diagnostics {
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
// Kernel contracts accumulate into thread-private POD storage; the runtime
// supplies the concurrent device reduction, without vendor code in kernel.
__device__ inline void merge(Accesses *total, const Accesses &local) {
  if (!total) return;
  for (std::size_t f = 0; f < field_count; ++f) {
    if (local.reads[f]) atomicAdd(total->reads + f, local.reads[f]);
    if (local.writes[f]) atomicAdd(total->writes + f, local.writes[f]);
    if (local.read_bytes[f]) atomicAdd(total->read_bytes + f, local.read_bytes[f]);
    if (local.write_bytes[f])
      atomicAdd(total->write_bytes + f, local.write_bytes[f]);
  }
  for (std::size_t p = 0; p < phase_count; ++p)
    if (local.full_plane_scans[p])
      atomicAdd(total->full_plane_scans + p, local.full_plane_scans[p]);
}
#endif

/// Private counter storage is itself an observed allocation/copy. Counter
/// updates are excluded from declared logical production accesses.
template <class Backend> class Observation {
  using Counters = pfc::core::DataBuffer<Backend, Accesses>;
  Diagnostics *owner_ = current;
  Counters counters_ = allocate(owner_);

  // nvcc 13.1 cicc aborts on a function-try-block in this constructor, so a
  // failed counter allocation is recorded here instead.
  static Counters allocate(Diagnostics *owner) {
    try {
      return Counters{owner ? 1u : 0u};
    } catch (...) {
      if (owner) owner->fail();
      throw;
    }
  }

public:
  Observation() {
    if (owner_) {
      const Accesses zero{};
      try {
        counters_.copy_from_host(&zero, 1, Field::Instrumentation);
      } catch (...) {
        owner_->fail();
        throw;
      }
    }
  }
  Accesses *data() { return counters_.data(); }
  ~Observation() {
    if (!owner_) return;
    Accesses counts{};
    // A destructor must not throw while unwinding a fatal runtime error.
    const auto status = pfc::gpuMemcpy(&counts, counters_.data(), sizeof(counts),
                                       pfc::gpuMemcpyDeviceToHost);
    if (status != pfc::gpuSuccess) {
      owner_->fail();
      return;
    }
    copy(Direction::DeviceToHost, Field::Instrumentation, sizeof(counts));
    for (std::size_t f = 0; f < field_count; ++f) {
      owner_->accesses.reads[f] += counts.reads[f];
      owner_->accesses.writes[f] += counts.writes[f];
      owner_->accesses.read_bytes[f] += counts.read_bytes[f];
      owner_->accesses.write_bytes[f] += counts.write_bytes[f];
    }
    for (std::size_t p = 0; p < phase_count; ++p)
      owner_->accesses.full_plane_scans[p] += counts.full_plane_scans[p];
  }
};

/// Events delimit elapsed stream intervals. Synchronous metadata roundtrips
/// and host gaps between records are included; this is not active kernel time.
class Interval {
  Diagnostics *owner_ = current;
  pfc::gpuEvent_t first_{}, last_{};
  Phase phase_;
  pfc::gpuStream_t stream_ = nullptr;
  bool begun_ = false;
  bool valid_ = false;

public:
  explicit Interval(Phase phase, pfc::gpuStream_t stream = nullptr)
      : phase_(phase), stream_(stream) {
    if (!owner_) return;
    try {
      GPU_CHECK(pfc::gpuEventCreate(&first_));
      GPU_CHECK(pfc::gpuEventCreate(&last_));
      GPU_CHECK(pfc::gpuEventRecord(first_, stream_));
      begun_ = true;
    } catch (...) {
      if (last_) (void)pfc::gpuEventDestroy(last_);
      if (first_) (void)pfc::gpuEventDestroy(first_);
      owner_->fail();
      throw;
    }
  }
  void finish() noexcept {
    if (!owner_ || !begun_) return;
    float ms = 0;
    const bool valid =
        pfc::gpuEventRecord(last_, stream_) == pfc::gpuSuccess &&
        pfc::gpuEventSynchronize(last_) == pfc::gpuSuccess &&
        pfc::gpuEventElapsedTime(&ms, first_, last_) == pfc::gpuSuccess;
    auto &event = owner_->events[static_cast<std::size_t>(phase_)];
    ++event.intervals;
    if (!valid) ++event.failed_intervals;
    valid_ = valid;
    event.available = event.failed_intervals == 0;
    if (valid) event.seconds += ms / 1000.0;
    if (!valid) owner_->fail();
    begun_ = false;
  }
  ~Interval() {
    if (!owner_) return;
    finish();
    const auto last_status = pfc::gpuEventDestroy(last_);
    const auto first_status = pfc::gpuEventDestroy(first_);
    if (last_status != pfc::gpuSuccess || first_status != pfc::gpuSuccess) {
      auto &event = owner_->events[static_cast<std::size_t>(phase_)];
      if (valid_) ++event.failed_intervals;
      event.available = false;
      owner_->fail();
    }
  }
  Interval(const Interval &) = delete;
  Interval &operator=(const Interval &) = delete;
};
} // namespace pfc::grain::diagnostics

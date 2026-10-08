// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#ifdef COUPLED_NATIVE_ALLOCATIONS
#include <execinfo.h>
#include <pthread.h>
#include <unistd.h>
#endif
namespace allocation_observer {
enum class Phase : std::size_t {
  stepper,
  rhs,
  upload,
  halo,
  audit,
  dispatch,
  download,
  external_thread,
  instrumentation,
  count
};
constexpr std::size_t phase_count = std::size_t(Phase::count);
constexpr const char *phase_names[] = {
    "stepper",  "rhs",      "upload",          "halo",           "audit",
    "dispatch", "download", "external_thread", "instrumentation"};
static_assert(sizeof(phase_names) / sizeof(*phase_names) == phase_count);
std::atomic<bool> observing{false};
std::atomic<std::size_t> allocation_requests{0};
std::array<std::atomic<std::size_t>, phase_count> requests{};
#ifdef COUPLED_NATIVE_ALLOCATIONS
// 0 unclaimed, 1 writer active, 2 immutable trace published.
std::array<std::atomic<unsigned>, phase_count> capture_state{};
struct Trace {
  std::size_t bytes{}, alignment{};
  std::array<void *, 24> frames{};
  int length{};
};
std::array<Trace, phase_count> traces{};
thread_local bool tracing = false;
pthread_t owner_thread{};
#endif
thread_local Phase phase = Phase::stepper;
struct Scope {
  Phase prior;
  explicit Scope(Phase next) : prior(phase) { phase = next; }
  ~Scope() { phase = prior; }
};
void observe(bool enabled) { observing.store(enabled); }
void count(std::size_t bytes, std::size_t alignment = 0) {
  if (!observing.load()) return;
#ifdef COUPLED_NATIVE_ALLOCATIONS
  allocation_requests.fetch_add(1);
  if (tracing) {
    requests[std::size_t(Phase::instrumentation)].fetch_add(1);
    return;
  }
  const auto key = std::size_t(
      pthread_equal(pthread_self(), owner_thread) ? phase : Phase::external_thread);
#else
  allocation_requests.fetch_add(1);
  (void)bytes;
  (void)alignment;
  const auto key = std::size_t(phase);
#endif
  requests[key].fetch_add(1);
#ifdef COUPLED_NATIVE_ALLOCATIONS
  unsigned expected = 0;
  if (capture_state[key].compare_exchange_strong(expected, 1)) {
    auto &trace = traces[key];
    trace.bytes = bytes;
    trace.alignment = alignment;
    tracing = true;
    trace.length = backtrace(trace.frames.data(), int(trace.frames.size()));
    tracing = false;
    capture_state[key].store(2, std::memory_order_release);
  }
#endif
}
#ifdef COUPLED_NATIVE_ALLOCATIONS
struct Snapshot {
  std::size_t total;
  std::array<std::size_t, phase_count> phases;
};
Snapshot snapshot() {
  Snapshot result{};
  result.total = allocation_requests.load();
  for (std::size_t key = 0; key < phase_count; ++key)
    result.phases[key] = requests[key].load();
  return result;
}
void report(const Snapshot &before, const Snapshot &after) {
  std::fprintf(stderr, "warmed total C++ allocation requests=%zu\n",
               after.total - before.total);
  for (std::size_t key = 0; key < phase_count; ++key) {
    std::fprintf(stderr, "warmed C++ allocation phase=%s requests=%zu\n",
                 phase_names[key], after.phases[key] - before.phases[key]);
    if (after.phases[key] > before.phases[key] &&
        capture_state[key].load(std::memory_order_acquire) == 2) {
      const auto &trace = traces[key];
      std::fprintf(stderr, "first process trace phase=%s bytes=%zu alignment=%zu\n",
                   phase_names[key], trace.bytes, trace.alignment);
      backtrace_symbols_fd(trace.frames.data(), trace.length, STDERR_FILENO);
    }
  }
}
#endif
} // namespace allocation_observer
// Teaching-test observer: successful C++ new requests, not malloc/runtime pools.
void *operator new(std::size_t n) {
  if (void *p = std::malloc(n ? n : 1)) {
    allocation_observer::count(n);
    return p;
  }
  throw std::bad_alloc();
}
void *operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
void *operator new(std::size_t n, std::align_val_t align) {
  void *p = nullptr;
  // POSIX rejects alignments below sizeof(void*); C++ permits requests with
  // smaller fundamental alignment. Returning the stronger alignment is valid.
  const std::size_t requested = std::size_t(align);
  const std::size_t alignment = std::max(requested, sizeof(void *));
  const int result = posix_memalign(&p, alignment, n ? n : 1);
  if (result) {
    std::fprintf(stderr,
                 "aligned allocation failed: size=%zu alignment=%zu POSIX=%d\n", n,
                 requested, result);
    throw std::bad_alloc();
  }
  allocation_observer::count(n, requested);
  return p;
}
void *operator new[](std::size_t n, std::align_val_t a) {
  return ::operator new(n, a);
}
void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void *p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete[](void *p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}

namespace allocation_observer {
// Direct calls cannot be elided as an unused new expression. Check that the
// provider sees successful ordinary/aligned requests before accepting zero.
bool verify_provider() {
  // Load the unwinder before observation; nested observer-induced requests
  // are counted in a separate instrumentation bucket without recursive traces.
#ifdef COUPLED_NATIVE_ALLOCATIONS
  owner_thread = pthread_self();
  std::array<void *, 24> warm{};
  backtrace(warm.data(), int(warm.size()));
#endif
  allocation_requests.store(0);
  observe(true);
#ifdef COUPLED_NATIVE_ALLOCATIONS
  // The provider probe must not claim diagnostic traces: it is independently
  // counted but its stack is not part of warmed application observation.
  for (auto &state : capture_state) state.store(1);
#endif
  void *a = ::operator new(16);
  void *b = ::operator new(64, std::align_val_t{64});
  void *c = ::operator new(16, std::align_val_t{4});
  void *rhs_request = nullptr;
  {
    Scope scope(Phase::rhs);
    rhs_request = ::operator new(24);
  }
  void *restored_request = ::operator new(8);
  observe(false);
  ::operator delete(rhs_request);
  ::operator delete(restored_request);
  ::operator delete(a);
  ::operator delete(b, std::align_val_t{64});
  const bool aligned = reinterpret_cast<std::uintptr_t>(c) % 4 == 0;
  ::operator delete(c, std::align_val_t{4});
  const bool ok = allocation_requests.load() == 5 && aligned &&
                  requests[std::size_t(Phase::stepper)].load() == 4 &&
                  requests[std::size_t(Phase::rhs)].load() == 1 &&
                  phase == Phase::stepper;
  allocation_requests.store(0);
  for (auto &n : requests) n.store(0);
#ifdef COUPLED_NATIVE_ALLOCATIONS
  for (auto &state : capture_state) state.store(0);
#endif
  return ok;
}
} // namespace allocation_observer

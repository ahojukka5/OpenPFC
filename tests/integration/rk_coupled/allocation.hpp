// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
namespace allocation_observer {
std::atomic<bool> observing{false};
std::atomic<std::size_t> allocation_requests{0};
void observe(bool enabled) { observing.store(enabled); }
void count() {
  if (observing.load()) allocation_requests.fetch_add(1);
}
} // namespace allocation_observer
// Teaching-test observer: successful C++ new requests, not malloc/runtime pools.
void *operator new(std::size_t n) {
  if (void *p = std::malloc(n ? n : 1)) {
    allocation_observer::count();
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
  allocation_observer::count();
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
  allocation_requests.store(0);
  observe(true);
  void *a = ::operator new(16);
  void *b = ::operator new(64, std::align_val_t{64});
  void *c = ::operator new(16, std::align_val_t{4});
  observe(false);
  ::operator delete(a);
  ::operator delete(b, std::align_val_t{64});
  const bool aligned = reinterpret_cast<std::uintptr_t>(c) % 4 == 0;
  ::operator delete(c, std::align_val_t{4});
  const bool ok = allocation_requests.load() == 3 && aligned;
  allocation_requests.store(0);
  return ok;
}
} // namespace allocation_observer

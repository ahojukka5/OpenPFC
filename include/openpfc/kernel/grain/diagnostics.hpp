// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <openpfc/kernel/data/host_device.hpp>

namespace pfc::grain {
class Diagnostics;
namespace diagnostics {
enum class Space { Host, Device };
enum class Direction {
  HostToHost,
  HostToDevice,
  DeviceToHost,
  DeviceToDevice,
  Count
};
enum class Field {
  Values,
  Labels,
  StagedValues,
  StagedLabels,
  Occupancy,
  Assignments,
  Registry,
  Contacts,
  AdjacencyMatrix,
  Counts,
  Flags,
  Instrumentation,
  Other,
  Count
};
enum class Phase {
  Preflight,
  Propagation,
  Inspection,
  Adjacency,
  Decision,
  Transfer,
  Count
};
inline constexpr auto field_count = static_cast<std::size_t>(Field::Count);
inline constexpr auto phase_count = static_cast<std::size_t>(Phase::Count);

/// Source element operations; includes neither compiler transactions nor the
/// instrumentation's own counter updates. Array elements, not cache lines.
struct Accesses {
  unsigned long long reads[field_count]{};
  unsigned long long writes[field_count]{};
  unsigned long long read_bytes[field_count]{};
  unsigned long long write_bytes[field_count]{};
  unsigned long long full_plane_scans[phase_count]{};
};
struct AllocationTotals {
  std::uint64_t requests = 0, frees = 0;
  std::uint64_t requested_bytes = 0, freed_bytes = 0;
  std::uint64_t initial_live_bytes = 0, live_bytes = 0, peak_live_bytes = 0;
  bool complete = true; ///< False after an observed runtime free failure.
};
struct AllocationLedger {
  struct Counters {
    std::atomic<std::uint64_t> requests{0}, frees{0}, requested{0}, freed{0};
    std::atomic<std::uint64_t> initial{0}, live{0}, peak{0};
    std::atomic<bool> complete{true};
    void allocate(std::size_t bytes) noexcept {
      ++requests;
      requested += bytes;
      const auto now = live.fetch_add(bytes) + bytes;
      auto old = peak.load();
      while (old < now && !peak.compare_exchange_weak(old, now)) {
      }
    }
    void release(std::size_t bytes) noexcept {
      ++frees;
      freed += bytes;
      live -= bytes;
    }
    AllocationTotals snapshot() const noexcept {
      return {requests.load(), frees.load(), requested.load(), freed.load(),
              initial.load(),  live.load(),  peak.load(),      complete.load()};
    }
    // Only at a quiescent observation boundary; never erase live ownership.
    void reset_interval() noexcept {
      requests = 0;
      frees = 0;
      requested = 0;
      freed = 0;
      const auto baseline = live.load();
      initial = baseline;
      peak = baseline;
    }
  };
  Counters host, device;
};

/// A successful allocation captures stable ledger ownership. This token may
/// outlive Diagnostics; it never stores a pointer into a stack observation.
/// Call release() only after the matching actual free succeeds.
class AllocationToken {
  std::shared_ptr<AllocationLedger> ledger_;
  Space space_ = Space::Host;
  std::size_t bytes_ = 0;

public:
  AllocationToken() = default;
  AllocationToken(std::shared_ptr<AllocationLedger> ledger, Space space,
                  std::size_t bytes)
      : ledger_(std::move(ledger)), space_(space), bytes_(bytes) {
    if (ledger_)
      (space_ == Space::Host ? ledger_->host : ledger_->device).allocate(bytes_);
  }
  AllocationToken(const AllocationToken &) = delete;
  AllocationToken &operator=(const AllocationToken &) = delete;
  AllocationToken(AllocationToken &&) noexcept = default;
  AllocationToken &operator=(AllocationToken &&) = delete;
  void failed_release() noexcept {
    if (ledger_)
      (space_ == Space::Host ? ledger_->host : ledger_->device).complete = false;
  }
  void release() noexcept {
    if (!ledger_) return;
    (space_ == Space::Host ? ledger_->host : ledger_->device).release(bytes_);
    ledger_.reset();
    bytes_ = 0;
  }
};

inline thread_local Diagnostics *current = nullptr;
class PauseObservation {
  Diagnostics *previous_;

public:
  PauseObservation() : previous_(current) { current = nullptr; }
  ~PauseObservation() { current = previous_; }
  PauseObservation(const PauseObservation &) = delete;
  PauseObservation &operator=(const PauseObservation &) = delete;
};
class Scope {
  Diagnostics *previous_;

public:
  /// Null explicitly disables this scope, including an enclosing observation.
  explicit Scope(Diagnostics *observation) : previous_(current) {
    current = observation;
  }
  ~Scope() { current = previous_; }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
};
struct Copies {
  std::uint64_t calls[static_cast<std::size_t>(Direction::Count)][field_count]{};
  std::uint64_t bytes[static_cast<std::size_t>(Direction::Count)][field_count]{};
};
struct EventInterval {
  bool available = false;
  std::uint64_t intervals = 0, failed_intervals = 0;
  double seconds = 0; ///< Elapsed device-event interval, including host gaps.
};
} // namespace diagnostics

/// Optional caller-owned observation. A full host allocation claim requires
/// a consumer new/delete provider using successful_allocation() and tokens.
/// Begin the scope before persistent input preparation and keep it through
/// publication/release to include that lifecycle. reset_interval() retains
/// its exact live baseline. Bookkeeping and runtime-reserved memory excluded.
class Diagnostics {
  std::shared_ptr<diagnostics::AllocationLedger> ledger_;

public:
  bool host_observer_connected = false;
  bool source_accesses_complete =
      false;                       ///< Defined field/label/producer scope only.
  bool observation_failed = false; ///< Sticky within an interval.
  void fail() noexcept {
    observation_failed = true;
    source_accesses_complete = false;
  }
  void covered() noexcept { source_accesses_complete = !observation_failed; }
  diagnostics::Accesses accesses{};
  diagnostics::Copies copies{};
  std::array<diagnostics::EventInterval, diagnostics::phase_count> events{};
  Diagnostics() {
    diagnostics::PauseObservation pause;
    ledger_ = std::make_shared<diagnostics::AllocationLedger>();
  }
  diagnostics::AllocationTotals
  allocations(diagnostics::Space space) const noexcept {
    return (space == diagnostics::Space::Host ? ledger_->host : ledger_->device)
        .snapshot();
  }
  diagnostics::AllocationToken allocation(diagnostics::Space space,
                                          std::size_t bytes) {
    return {ledger_, space, bytes};
  }
  void reset_interval() noexcept {
    ledger_->host.reset_interval();
    ledger_->device.reset_interval();
    accesses = {};
    copies = {};
    events = {};
    source_accesses_complete = false;
    observation_failed = false;
  }
};

namespace diagnostics {
/// Provider ABI: call only after an actual successful allocation; retain the
/// move-only token with that allocation and release it after successful free.
inline AllocationToken successful_allocation(Space space, std::size_t bytes) {
  return current ? current->allocation(space, bytes) : AllocationToken{};
}
inline void copy(Direction direction, Field field, std::size_t bytes) noexcept {
  if (!current || !bytes) return;
  const auto d = static_cast<std::size_t>(direction),
             f = static_cast<std::size_t>(field);
  ++current->copies.calls[d][f];
  current->copies.bytes[d][f] += bytes;
}
OPENPFC_INLINE_HD void add(unsigned long long *counter, unsigned long long value) {
  *counter += value;
}
OPENPFC_INLINE_HD void read(Accesses *counts, Field field, std::size_t bytes) {
  if (!counts) return;
  const auto f = static_cast<std::size_t>(field);
  add(counts->reads + f, 1);
  add(counts->read_bytes + f, bytes);
}
OPENPFC_INLINE_HD void write(Accesses *counts, Field field, std::size_t bytes) {
  if (!counts) return;
  const auto f = static_cast<std::size_t>(field);
  add(counts->writes + f, 1);
  add(counts->write_bytes + f, bytes);
}
OPENPFC_INLINE_HD void scan(Accesses *counts, Phase phase, std::size_t planes) {
  if (counts)
    add(counts->full_plane_scans + static_cast<std::size_t>(phase), planes);
}
template <class T>
OPENPFC_INLINE_HD T load(const T *data, std::size_t i, Accesses *counts,
                         Field field) {
  read(counts, field, sizeof(T));
  return data[i];
}
template <class T, class U>
OPENPFC_INLINE_HD void store(T *data, std::size_t i, const U &value,
                             Accesses *counts, Field field) {
  write(counts, field, sizeof(T));
  data[i] = value;
}
template <bool Observe, class T>
OPENPFC_INLINE_HD T load(const T *data, std::size_t i, Accesses *counts,
                         Field field) {
  if constexpr (Observe) read(counts, field, sizeof(T));
  return data[i];
}
template <bool Observe, class T, class U>
OPENPFC_INLINE_HD void store(T *data, std::size_t i, const U &value,
                             Accesses *counts, Field field) {
  if constexpr (Observe) write(counts, field, sizeof(T));
  data[i] = value;
}
struct NoAccesses {};
inline Accesses *host_accesses() noexcept {
  return current ? &current->accesses : nullptr;
}
} // namespace diagnostics
} // namespace pfc::grain

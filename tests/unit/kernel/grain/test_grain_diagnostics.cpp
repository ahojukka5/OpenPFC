// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "fixtures/allocation_provider.hpp"
#include <catch2/catch_test_macros.hpp>
#include <optional>
#include <type_traits>
using namespace pfc::grain;
namespace dg = pfc::grain::diagnostics;
static_assert(!std::is_move_assignable_v<dg::AllocationToken>);

TEST_CASE("allocation intervals retain exact live ownership and overlap",
          "[grain][diagnostics]") {
  Diagnostics d;
  auto first = d.allocation(dg::Space::Device, 5);
  auto second = d.allocation(dg::Space::Device, 7);
  auto totals = d.allocations(dg::Space::Device);
  REQUIRE(totals.requests == 2);
  REQUIRE(totals.live_bytes == 12);
  REQUIRE(totals.peak_live_bytes == 12);
  first.release();
  d.reset_interval();
  totals = d.allocations(dg::Space::Device);
  REQUIRE(totals.requests == 0);
  REQUIRE(totals.initial_live_bytes == 7);
  REQUIRE(totals.peak_live_bytes == 7);
  auto replacement = d.allocation(dg::Space::Device, 11);
  second.release();
  replacement.release();
  totals = d.allocations(dg::Space::Device);
  REQUIRE(totals.requests == 1);
  REQUIRE(totals.frees == 2);
  REQUIRE(totals.peak_live_bytes == 18);
  REQUIRE(totals.live_bytes == 0);
  std::optional<dg::AllocationToken> survivor;
  {
    Diagnostics temporary;
    survivor.emplace(temporary.allocation(dg::Space::Host, 13));
  }
  survivor->release(); // Stable ledger remains owned after Diagnostics dies.
}

TEST_CASE("consumer ordinary aligned sized and failed new payload is exact",
          "[grain][diagnostics]") {
  Diagnostics d;
  d.host_observer_connected = true;
  dg::AllocationTotals allocated, released;
  void *first = nullptr, *second = nullptr;
  {
    dg::Scope scope(&d);
    first = ::operator new(5);
    second = ::operator new(7, std::align_val_t{64});
    allocated = d.allocations(dg::Space::Host);
    ::operator delete(first, std::size_t{5});
    ::operator delete(second, std::size_t{7}, std::align_val_t{64});
    released = d.allocations(dg::Space::Host);
  }
  REQUIRE(allocated.requests == 2);
  REQUIRE(allocated.requested_bytes == 12);
  REQUIRE(allocated.peak_live_bytes == 12);
  REQUIRE(reinterpret_cast<std::uintptr_t>(second) % 64 == 0);
  REQUIRE(released.frees == 2);
  REQUIRE(released.freed_bytes == 12);
  REQUIRE(released.live_bytes == 0);
  d.reset_interval();
  void *failed = nullptr;
  {
    dg::Scope scope(&d);
    failed = ::operator new(std::numeric_limits<std::size_t>::max(), std::nothrow);
  }
  REQUIRE(failed == nullptr);
  REQUIRE(d.allocations(dg::Space::Host).requests == 0);
  {
    dg::Scope outer(&d);
    {
      dg::Scope off(nullptr);
      REQUIRE(dg::current == nullptr);
    }
    REQUIRE(dg::current == &d);
    {
      dg::PauseObservation paused;
      REQUIRE(dg::current == nullptr);
    }
    REQUIRE(dg::current == &d);
  }
  REQUIRE(dg::current == nullptr);
  d.reset_interval();
  void *persistent = nullptr;
  {
    dg::Scope scope(&d);
    persistent = ::operator new(17);
  }
  d.reset_interval();
  REQUIRE(d.allocations(dg::Space::Host).initial_live_bytes == 17);
  ::operator delete(persistent);
  REQUIRE(d.allocations(dg::Space::Host).frees == 1);
  REQUIRE(d.allocations(dg::Space::Host).live_bytes == 0);
}

TEST_CASE("failed free keeps live ownership and completeness sticky",
          "[grain][diagnostics]") {
  Diagnostics d;
  auto token = d.allocation(dg::Space::Device, 19);
  token.failed_release();
  REQUIRE_FALSE(d.allocations(dg::Space::Device).complete);
  REQUIRE(d.allocations(dg::Space::Device).live_bytes == 19);
  d.reset_interval();
  REQUIRE_FALSE(d.allocations(dg::Space::Device).complete);
  REQUIRE(d.allocations(dg::Space::Device).initial_live_bytes == 19);
  token.release();
  REQUIRE(d.allocations(dg::Space::Device).live_bytes == 0);
  REQUIRE_FALSE(d.allocations(dg::Space::Device).complete);
}

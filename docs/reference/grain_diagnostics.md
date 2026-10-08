<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Optional grain invocation diagnostics

`pfc::grain::Diagnostics` supplies caller-owned allocation observation through
`diagnostics::Scope(&observation)`. `Scope(nullptr)` and `PauseObservation`
disable and then restore the prior thread-local scope.

The observation must outlive the scope and the synchronous operation. Owning
allocation tokens retain a shared ledger independently of that observation;
returned arrays can safely outlive it. Tokens are move-constructible, cannot
be move-assigned, and record release only after an actual successful free.
`reset_interval()` requires a quiescent boundary, clears interval requests,
frees, copies, accesses and events, and retains all live allocations as the
new initial/peak baseline. It does not erase ownership.

## Allocation scope

`allocations(Host)` and `allocations(Device)` report successful request/free
counts, requested/freed bytes, initial live bytes, current live bytes and
maximum simultaneously live requested payload bytes. Resize records the new
allocation before freeing the old, including their overlap. A failed request
is not an allocation. A runtime free failure leaves its payload live and
sets that ledger's `complete=false`, retained across interval resets.

Device coverage includes `DataBuffer` allocations. To include persistent
proposal/previously accepted buffers, begin observation before creating them
and keep the ledger through publication and old-state release. Buffers made
outside observation do not acquire tokens retroactively.

Host coverage requires a consumer-owned provider intercepting ordinary,
array, aligned, sized and nothrow global `new`/`delete`. Call
`diagnostics::successful_allocation(Host, requested_bytes)` after successful
allocation, store its token with the allocation, move-construct that token
before destroying its storage, and call `release()` after actual free. The
[unit-test provider](../../tests/fixtures/allocation_provider.hpp) demonstrates
this ABI; it is linked only into isolated consumer test executables. The
library never replaces global allocation functions. Set
`host_observer_connected=true` only after validating the complete provider.

This scope includes every intercepted current-thread C++ allocation during
the scope, including called-library allocations. It excludes `malloc`,
allocator headers/padding, device-runtime pools, runtime-reserved memory,
other threads without propagated observation, RSS and physical memory usage.
The observation ledger's own allocation and provider token metadata are
excluded. Disable observation around consumer logging, fingerprints and
other explicitly excluded bookkeeping; that choice is part of the consumer
contract. Peaks/counts aggregate instrumentation allocation payload too;
there is no separate instrumentation allocation peak.


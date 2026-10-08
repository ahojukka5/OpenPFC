<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Optional grain invocation diagnostics

`remapping::Options::diagnostics` optionally points at caller-owned
`pfc::grain::Diagnostics`. All remapping policies use the same observation
path, including no-op plans and expected failures. Null explicitly disables
an enclosing observation for that invocation. Nested tracking and transfer
inherit the current observation; standalone calls can use
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

Device coverage includes `DataBuffer` and the producer's nested private
buffers, including observation counter storage. To include persistent
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

## Copies and source operations

`copies.calls[direction][field]` and `copies.bytes[direction][field]` record
successful explicit transport calls. GPU buffer upload/download accepts an
optional `Field` tag (default `Other`). The remapper tags every nested
registry, assignment, count, flag, contact, label-publication and diagnostic
counter transfer. Zero-length copies have no call. Capacity rejection does
not invent a contact/publication transfer. This metric is distinct from
selected moved-grain payload and from ordinary element assignments.

`accesses` reports executed source-defined typed array reads/writes and
logical byte totals. It covers values, labels, staged arrays, occupancy and
the device producer's registry, assignment, matrix, count, flag and contact
arrays. Each explicit typed `load<T>` or `store<T>` contributes `sizeof(T)`;
an assignment-record load contributes the whole `sizeof(Assignment)`, even
when its consumer reads only `.id`. Atomic production updates contribute a
logical read and write. Branches and early exits determine actual counts.
CPU vector zero initialization of field staging contributes its exact
initialized-element writes. Counter atomics themselves are excluded from
production source accesses and counter transport is tagged `Instrumentation`.

This is **not** compiler loads/stores, cache transactions or HBM traffic.
Opaque host STL/queue/graph/coloring/alignment element operations and host
registry preparation are outside source-access coverage, although their
successful C++ allocations are covered by the provider. Host metadata
container copies/conversions are not transport calls. No full host memory
traffic claim is available. `source_accesses_complete` means the declared
array scope was observed; `observation_failed` makes failure sticky across
nested observers until `reset_interval()`.

`full_plane_scans[phase]` counts completed outer domain traversals, each
covering all cells of the reported number of slots. It does not multiply
by stencil neighbors, internal slot loops or reads per cell. Propagation
counts initialization and completed GPU sweeps; CPU counts seed admission
and final completeness traversal. CPU horizon initialization is Inspection;
CPU support-count traversal is another Inspection traversal. Adjacency counts
one domain traversal (CPU vertex admission/GPU contacts), and transfer counts
one cell-domain staging traversal. CPU STL zero initialization contributes
writes but is not another algorithm-domain traversal. The CPU unsafe check
can return early and contributes no full-plane traversal. Reached incomplete
passes still contribute their executed element accesses, never a completed
scan count.

## Event intervals and disabled mode

`events[phase]` exposes executed interval count, failed interval count,
availability and accumulated seconds. CPU calls leave device events
unavailable. GPU events delimit Preflight, Propagation, Inspection, Adjacency,
Decision and Transfer in the actual operation stream. Adjacency has producer
and subsequent compact graph-download intervals; these do not overlap.
Decision measures elapsed stream time across the host decision gap. Nested
synchronizations, copies, allocation/host gaps between event records are
included. These intervals are not pure active-kernel time and exclude
external publication and old-state release.

Unreached phases are unavailable, rather than measured zeros. Any failed
interval keeps that phase unavailable, even after later valid intervals.
Counter readback or event failure marks observation incomplete; fatal runtime
errors may throw while preserving original const inputs where hardware remains
usable. Diagnostics do not recover a poisoned device.

The observer-disabled GPU realization selects compile-time kernels with no
counter aggregate, counter atomics, counter buffers or events. Generic buffer
and consumer allocation-provider observation checks still exist in an
installed provider realization. Compare disabled and enabled numerical
outputs/status/snapshots before using separate diagnostic replay. Do not use
enabled replay elapsed intervals as disabled-operation timings or subtract
observation overhead. External lifecycle timing must bracket synchronized
remapping, publication and release itself; internal statistics are not that
external lifecycle.

For `Grid3D`, a completed `full_plane_scans` traversal covers one entire
slot volume, not one z slice. All other observation and coverage semantics
are unchanged; see [local 3D remapping](grain_3d.md).

<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Local 3D grain remapping

`Grid3D` extends the existing identity, contact, transfer and remapping
contracts to one local structured volume. It owns no fields, decomposition,
ghost cells or physical model. Persistent `Id` and recyclable `Slot` remain
independent. The existing coloring implementations consume the resulting
contact graph without dimensional changes.

```cpp
using namespace pfc::grain;
Grid3D grid{nx, ny, nz, true, true, true, Connectivity::TwentySix};
// Cell index: x + nx*(y + ny*z); field index: slot*cell_count(grid) + cell.
remapping::Options options;
options.contact_radius = 3;
auto staged = remapping::remap(grid, slots, values, seeds, grains, options);
// Publish all owning members together only after Status::Success.
```

The CPU overload is in `<openpfc/runtime/cpu/grain_remapping.hpp>`.
Device consumers use `<openpfc/runtime/gpu/grain_remapping.hpp>` in a
CUDA/HIP translation unit with the existing backend-tagged `DataBuffer`
arguments and optional stream. Explicit `Grid3D` objects select constrained
3D overloads; the concrete 2D overloads retain existing braced call syntax.
`Grid2D` accepts Four/Eight, and `Grid3D` accepts Six/TwentySix. A mismatched
connectivity fails tracking/contact/remapping validation instead of changing
dimensional meaning. Flat transfer uses only the volume layout and has no
stencil-dependent operation.

## Ownership and contacts

Six admits the six immediate face neighbors. TwentySix additionally admits
edge and corner neighbors. Occupancy and prior seeds are slot-major arrays.
Surviving seeds are fixed. Newly occupied cells inherit the nearest admitted
seed in the chosen stencil, with smaller UID resolving equal-distance ties.
Disconnected unseeded support fails; no UID is created. CPU ownership uses
an independent shortest-path reference rather than device ping-pong sweeps.

Contacts include same-cell cross-slot overlaps and all slots within the
explicit radius. Six uses Manhattan distance; TwentySix uses Chebyshev
distance. Each periodic axis uses its shortest wrapped distance, including
face, edge and corner seams. Radius zero in the standalone contact primitive
includes only same-cell overlaps. Composed remapping requires radius at
least one. The single-label `contact_graph(Grid3D, labels)` raster oracle
has radius one and cannot represent multiple occupants of a cell.

The conservative `UnsafeCadence` check examines distinct UIDs in immediate
same-slot neighbors under the selected 6/26 stencil. It does not prove that
amplitudes have mixed, but refuses to certify their separation after such
an approach. The caller must choose buffered contacts and check intervals
before its evolution can mix support. The default radius three is a cell
buffer, not a guarantee for an arbitrary solver or time step.

## Transactions, limits and observation

The [transfer contract](grain_transfer.md) applies unchanged to every 3D
cell. All finite nonbackground support is labeled and remains in its
registered source slot; no threshold discards tails. Complete batches,
including cyclic moves, read the original state and publish only after all
checks pass. Even a no-op plan stages whole arrays. Original inputs stay
unchanged on expected failures and usable-runtime exceptions. The caller
owns publication of values, labels, registry and epoch as one state.

Iteration limits, unseeded support, missing support, invalid input,
insufficient contact capacity, unsafe cadence, exhausted coloring budgets
and infeasible palettes return the same explicit remapping statuses as 2D.
A deferred check returns no state and certifies no cached graph.

Sizes and axis coordinates are checked before allocation. GPU contact
construction retains its dense O(registry-size squared) internal matrix;
its storage and radius-cubed enumeration limit practical local problem
sizes. CPU tracking/contact construction is an independent tiny-grid
correctness reference with quadratic coordinate-pair work, not a scalable
production detector. This extension supplies no MPI decomposition or
optimization claim.

Optional [diagnostics](grain_diagnostics.md) use the same scopes, source
array accounting, requested allocation payload, tagged copies and phase
event intervals. A completed full-plane scan now traverses a complete
slot volume: the inherited field name does not mean a separate z slice.
Diagnostics-disabled device kernels retain compile-time observer removal.
Instrumented event intervals may include host gaps and are not hardware
traffic or active-kernel-only measurements.

The tests use independent coordinate-pair distances and per-seed BFS,
exhaustive tiny label/occupancy states, seam and radius controls, exact
support permutations, nonblocking producer streams, failure rollback and
observed/unobserved parity. Backend execution evidence is recorded with
its source revision; compiling a CUDA realization is not CUDA execution.

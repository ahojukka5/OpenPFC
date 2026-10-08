<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Grain identity and topology

Include `<openpfc/kernel/grain/topology.hpp>` for the independent contract in
`pfc::grain`. It describes identities, slot assignments and contact topology;
it owns no numerical field, physics, execution backend or application.

An `Id` is a persistent 64-bit grain identity. Zero denotes background.
A `Slot` is a recyclable, zero-based 32-bit order-parameter field index.
`unassigned` is reserved and cannot index a field. Several disconnected grains
may occupy the same slot. A slot index is never evidence that two cells belong
to the same grain. These identities describe multiphase fields, not PFC lattice
orientation.

`IdentitySequence` allocates monotonically without recycling; the caller must
retain its counter in persistent state. The largest identity can be allocated
once, after which allocation throws `std::overflow_error` without wrapping.
An application reconstructing a sequence from restored identities is responsible
for choosing an unused next identity; the sequence does not own a registry.

`Grain` records retain their identity when `retire` releases their slot and
marks them inactive. Retirement is terminal in the tracking contract; the caller
must not reactivate a retired identity. `Snapshot` stores a caller-assigned epoch,
sorted grain records, and a contact graph. A snapshot is a value: an observation
held by a consumer is stable when the producer changes a separate copy. Consumers
must not modify shared public storage concurrently. The contract does not infer
merge/split lineage: a tracker must supply that policy, allocate identities for
new grains, and retain inactive records where historical observations need them.

`validate(snapshot, slot_count)` checks nonzero unique identities, slot bounds,
inactive tombstones and exact agreement between active records and graph
vertices. It deliberately accepts same-slot contact conflicts: observations of
such conflicts are the input to recoloring. Checking a valid coloring and moving
numerical data are separate operations. A failed validation never edits its input.
It does not check epoch history or detect reuse across separate snapshots.

`ContactGraph` contains sorted unique identities and canonical undirected
contacts `(first, second)` with `first < second`. It includes isolated active
grains and excludes background and retired grains. `make_contact_graph`
canonicalizes input order, reversed endpoints and duplicates; self contacts and
unknown endpoints are errors. `validate(graph)` checks already canonical data.
There is no fixed neighbor capacity. Allocation failures propagate rather than
silently dropping edges. `diff_topology` compares canonical graphs and returns
added/removed identities and contacts, with sorted affected identities. Slot
changes alone do not change topology.

## Local 2D contact oracle

`Grid2D` and `contact_graph(grid, labels)` provide a deterministic four- or
eight-neighbor contact oracle for one local structured grid. Storage is `x + nx*y`; labels are
already identified grains. Dimensions must be positive, and label length must
match the overflow-checked grid size. Periodicity is explicit on each axis;
wrapping a length-one axis never creates a self edge. Background interfaces
and equal-identity contacts do not create edges. `Connectivity::Four` is the
default; `Connectivity::Eight` also admits corner contacts.

A single identity per cell cannot represent overlapping or diffuse occupancy
across order-parameter planes. Cross-plane adjacency must explicitly inspect
those planes and the model's contact buffer/stencil.

The label oracle does not identify connected components, associate observations
over time, or construct buffered contacts from overlapping order-parameter
fields. Those operations can supply their own canonical `ContactGraph` under an
explicit contact policy. In particular, four-face contact is not a guarantee of
safe shared-slot reuse for a model with a wider stencil or diffuse interfaces.
The contract currently promises neither ghost-cell interpretation nor 3D or MPI
tracking. Existing field/decomposition types remain the owners of their layouts;
an adapter must explicitly describe how local labels are extracted.

Malformed data throw `std::invalid_argument`; identity and grid-size overflow
throw `std::overflow_error`. Graph sizes and degrees use dynamic host storage;
backend representations must report exhaustion explicitly and preserve every
admitted contact. Public records are intentionally simple values, so callers
validate them at API boundaries rather than assuming unchecked mutations preserve
invariants.

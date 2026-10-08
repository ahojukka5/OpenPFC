<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Transactional grain field transfer

Include `<openpfc/kernel/grain/transfer.hpp>` for
`pfc::grain::transfer`. It stages a batch of persistent-identity slot moves
without mutating the original values, labels, or grain registry. Publish all
three returned members together only when status is `TransferStatus::Success`.
A failure returns its explicit status and empty staged arrays and registry.
Allocation failure also leaves the originals intact.

The primitive uses local slot-major arrays: 2D index
`slot*(nx*ny) + x + nx*y`, or 3D index
`slot*(nx*ny*nz) + x + nx*(y + ny*z)` with an explicit `Grid3D`. It does not update topology, observation epochs,
halos, or distributed state. A `Transfer` holds the grain `id`, original
`source` slot, and `destination` slot. Slots are zero-based and independent
of persistent identity. Each grain must occur once in the sorted registry;
inactive records retain their identities and have `unassigned` slots.

All field values must be finite and at least the declared background value,
which defaults to zero for nonnegative occupancy fields. Every value unequal
to background must have its grain identity label; every background value
must have label zero. All support belonging to an identity must be in its
registered source slot. Every active grain must have support. No threshold
is applied, and no diffuse tail is silently discarded. An explicit background
of -1 supports signed fields under the same complete-label contract.

The transaction reads the original state for every instruction. It supports
noncontiguous grains, several nonoverlapping grains sharing a destination,
and cycles, including cycles whose original supports overlap in different
slots. A destination is safe when its support is empty or its occupying
grain simultaneously vacates that cell and slot. Two grains cannot finish
in the same cell and slot. Unknown or duplicate move identities, stale source
slots, invalid support, and occupied final destinations reject the entire
batch. A valid self-move or empty batch preserves the state exactly.

Staging preserves each cell's multiset of nonbackground values, and copies
unaffected data and identity labels unchanged. Consequently the sum of
`value-background_value` is conserved algebraically. For signed fields,
`(value+1)/2` is conserved by the same permutation. Floating summation order
can change after slot permutation, so a computed sum may differ by its
ordinary rounding error; the transfer itself incurs no amplitude error or
tail loss. The strict API offers no background tolerance or loss budget.
A thresholded contact graph by itself does not establish these transfer
preconditions.

```cpp
using namespace pfc::grain;
std::vector<Transfer> moves{{grain_id, old_slot, new_slot}};
auto staged = transfer(grid, slot_count, values, labels, grains, moves);
if (staged.status == TransferStatus::Success) {
  // Application publishes the complete state as one object.
  // No changes to values, labels, or grain slots precede this decision.
}
```

Include `<openpfc/runtime/gpu/grain_transfer.hpp>` in a CUDA or HIP translation
unit to use the same overload with `DataBuffer<Backend, double>` values and
`DataBuffer<Backend, Id>` labels. The returned `TransferResult<Values, Labels>`
owns staging buffers of those types. A per-cell device kernel validates support
and final destinations, then copies and clears from original buffers. Only one
status word and one presence flag per active grain are copied back to the host;
fields and identity labels remain on the device. An optional stream orders the
transaction after caller writes; nondefault streams explicitly wait for
default-stream metadata uploads before staging. The default-stream path adds
no extra stream synchronization. The operation completes before returning.
Runtime or allocation exceptions preserve the original buffers.

Staging requires one full values buffer and one full labels buffer, plus a
registry-sized assignment buffer, presence flags, and one status word. The
reference validation examines pairs of occupied slots at each cell; this is a
correctness primitive with no performance claim. It does not infer support
ownership, produce remapping instructions, or repair the contact graph.

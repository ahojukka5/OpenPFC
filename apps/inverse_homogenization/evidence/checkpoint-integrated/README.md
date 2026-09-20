<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Current-master checkpoint integration

The isolated source pinned in `provenance.json` rebases the checkpoint stack
onto master42a9ceec. The parent owner's branch is unchanged. The only code
difference from metadata-fault sourcec2a70238 in the HIP inverse driver is
the already-merged endpoint material reporter. Rebase conflicts were confined
to README, changelog and license annotations.

Canonical scripts/build.sh22182769 passes all139CTest batches plus Python.
The actual GPU replay22182878 passes all hold/moving comparisons and injected
fault checks in3:13 on one node/eight GCDs, accountproject_462001245. Complete
uninterrupted/restarted endpoint material JSON records are also identical for
both physical and thresholded fields in both cases. All123/11field comparisons,
CSV/tracker comparisons and rejected faults remain in `verified.json`.

Build the pinned source through scripts/build.sh and replay its
`apps/inverse_homogenization/scripts/replay_checkpoint_state.py` with the
pinned target, eight ranks and a fresh output directory. Exact commands,
module/scheduler records, checkpoint metadata and stdout/stderr are preserved
in `raw.json.gz`. This eliminates the old-base finite-strain-grid failure
from integration qualification. It does not establish different-rank restart,
arbitrary hardware-error recovery or power-loss durability. The stack still
requires intentional reviewable history before merge; this archive is not a
claim that the parent was merged or changed.

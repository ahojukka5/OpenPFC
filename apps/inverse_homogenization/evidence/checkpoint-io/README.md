<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Checkpoint metadata fault qualification

Job 22182481 completed successfully on account project_462001245, one node,
eight GCDs in 3:20. Source, binary, driver and target identities are pinned in
`provenance.json`. `raw.json.gz` includes every command, log, CSV, checkpoint
tracker/ledger, CURRENT pointer, scheduler record and environment identity.

Actual HIP restarts reject missing, truncated and malformed snapshot ledgers
in both the active verification hold and changing-design controls. Injected
`CURRENT.tmp -> /dev/full` causes buffered close/publication failure; the
job fails before final material export and leaves the previous checkpoint
bytes unchanged. A subsequent valid restart still reproduces all 123 hold
fields and 11 moving fields byte-for-byte, every overlapping non-timing CSV
value and complete terminal metadata. The original hold converges at accepted
120; the moving control terminates MAX_STEPS at 8. Exhausted budgets and
terminal checkpoint restarts remain rejected. All comparisons are recorded
in `verified.json`; field hashes permit exact replay checks.

Build the pinned source through `scripts/build.sh`, then run on eight GCDs:

```sh
python3.11 apps/inverse_homogenization/scripts/replay_checkpoint_state.py \
  --binary /path/to/openpfc_inverse_homogenize_hip \
  --target /path/to/target.txt --output /new/scratch/output --ranks 8
```

The exact target is available at the immutable source/path in provenance;
all commands and module versions are in the raw archive. Fields are
regenerated rather than duplicated in Git. Canonical build job 22182432
passes 138/139 CTest batches plus Python. Its sole failure is the known
finite-strain-grid defect on the parent branch, not a checkpoint failure.
Current-master integration still requires separate full qualification.

This checks deterministic metadata failures and continued trajectory identity.
It does not establish fsync/power-loss durability, hardware-error detection,
or equivalence across different rank decompositions. No optimizer, physical
target, material coefficients or convergence tolerances changed.

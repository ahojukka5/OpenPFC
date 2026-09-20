<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Reject exhausted restart budgets

Actual GPU entrypoint qualification, job22182245, accountproject_462001245,
one node/eight GCDs, 2:12. Source and binary/driver identities are pinned in
`provenance.json`; `raw.json.gz` preserves commands, logs, tracker records,
CSV histories, module stack and scheduler metadata.

For a saved next step50 in the active verification hold, requested budgets49
and50 both reject. For a changing design saved at next step4, budgets3 and4
both reject. All four attempts emit no material field and leave every file
in the saved checkpoint byte-identical. A subsequent valid restart still
matches the uninterrupted run: all123 hold-case fields, all11 moving-case
fields, all overlapping non-timing CSV values, and complete terminal tracker
metadata. The hold converges at accepted120 under original20+100 rules;
the moving control remains MAX_STEPS at8. Terminal checkpoints still reject.

Replay by building the pinned source through `scripts/build.sh` and running
its `apps/inverse_homogenization/scripts/replay_checkpoint_state.py` in the
recorded one-node/eight-GCD environment with the archived target identity:

```sh
python3.11 apps/inverse_homogenization/scripts/replay_checkpoint_state.py \
  --binary /path/to/openpfc_inverse_homogenize_hip \
  --target /path/to/target.txt --output /new/scratch/output --ranks 8
```

Fields are regenerated, not duplicated in Git. `verified.json` retains all
field hashes and explicit rejected-budget values. Full canonical build22182213
passes the guard and138/139 CTest batches plus Python; the sole failure is the
parent's known finite-strain-grid #76 defect. Its log resides under account519
storage, but scheduler billing is account245. This bounded success does not
qualify atomic checkpoint publication, failed writes or production walltime
recovery; those remain separate gates on the parent checkpoint implementation.

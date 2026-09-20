<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Qualified CPU and HIP checkpoint replay

Canonical `scripts/build.sh` job 22183109 builds source
`43d0d4224340532594f2ae6f618a7e661cbd0ff9` and passes all 139 CTest batches
plus Python. Replay job 22183155 uses those exact CPU/HIP binaries on account
`project_462001245`, one node, eight HIP ranks followed by two CPU ranks.

Both moving and active-verification-hold controls reproduce uninterrupted
fields and every non-timing history value exactly after restart. The hold
finishes CONVERGED at accepted step 120; the moving control intentionally
finishes MAX_STEPS at step 8. HIP compares 123/11 fields and CPU 124/12,
respectively. HIP final continuous and thresholded material JSON reports also
match byte-for-byte. Exhausted budgets and terminal checkpoints are rejected.
Missing/truncated/malformed HIP ledgers are rejected; the CPU has no such
ledger. A /dev/full metadata publication fault leaves prior CURRENT bytes
unchanged and emits no final field.

`verified.json` preserves all checks and field hashes. `raw.json.gz` contains
exact text records indexed by relative path: commands, runtime modules,
scheduler, histories, endpoint reports, checkpoint metadata, stdout/stderr,
and canonical build output. `provenance.json` pins source, binary, driver,
target and raw archive identities. Build the pinned source through
`scripts/build.sh`, then submit its
`apps/inverse_homogenization/slurm/inverse3d_restart_equiv_72.sbatch` with
INVERSE_SRC, INVERSE_BINARY, INVERSE_BINARY_SOURCE and INVERSE_CPU_BINARY.
The batch preserves both exact replay commands and accepts a fresh INVERSE_OUT.

This qualifies same-rank continuation of the tested frozen moving and hold
states. It does not claim power-loss durability, different-rank bitwise replay,
or lossless append-only CSV under arbitrary preemption. Use a separate CSV
for each allocation and use the checkpoint's absolute next_step to choose
retained history. The checkpoint is the restart authority.

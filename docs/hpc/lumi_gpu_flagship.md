<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# LUMI-G flagship scaling (issue #13)

Presentation-grade **weak scaling** of `tungsten_hip` on LUMI-G: node
counts 1, 2, 4, 8, 16, 32, then one gated **60-node / 480-GCD** point.
This is not a multi-application campaign. Heavy artifacts stay on
`/scratch/project_462001519/juaho/openpfc-scaling/`. Commit only the
compact CSV, recipes, and plots.

The earlier one-node / four-node **strong** curve on 768³ is
[lumi_gpu_scaling.md](lumi_gpu_scaling.md). Do not mix those job ids into
this table.

## Frozen protocol

| Knob | Value |
|------|--------|
| App | `tungsten_hip` |
| Input | [`tungsten_hip_scaling.toml`](../lumi_slurm/tungsten_hip_scaling.toml) |
| I/O | off (`saveat = -1`, no `[[fields]]`) |
| `dt` | 1 (accepted step count = `t1`) |
| Timed steps | 20 (10 is allowed at 32/60 nodes if documented) |
| Warm-up | drop step 1 (FFT plan / first-touch) |
| Metric | median `wall_step` from schema-v4 `timing_profile.json` |
| Decomposition | 8 ranks/node; `OPENPFC_FFT_NODE_GRID=1` off-node |
| Memory | host RSS from `memory_samples`; HBM is a `HIP_MEM` line |
| Scratch | `/scratch/project_462001519/juaho/openpfc-scaling/` |

Do not change physics parameters per node count.

## Candidate ladder

Cells/GCD stay near 54–57 M. Sizes are 5-smooth. **1200³ at 32 GCDs is
not divisible by 32**, so a 1-D 32-rank slab is illegal; the flagship
recipe forces the measured 1×8×nnodes layout instead.

| Nodes | GCDs | N | cells / GCD | 1×8×N grid |
|------:|-----:|--:|------------:|------------|
| 1 | 8 | 768 | 56.6 M | min-surface `2×2×2` (8 ranks) |
| 2 | 16 | 960 | 55.3 M | 1×8×2 |
| 4 | 32 | 1200 | 54.0 M | 1×8×4 |
| 8 | 64 | 1536 | 56.6 M | 1×8×8 |
| 16 | 128 | 1920 | 55.3 M | 1×8×16 |
| 32 | 256 | 2400 | 54.0 M | 1×8×32 |
| 60 | 480 | 3000 | 56.3 M | 1×8×60 |

3000³ is 27 billion cells. Confirm HBM on the 32-node point before
submitting 60 nodes.

## How to submit (LUMI login)

```bash
python3 apps/tungsten/scripts/flagship_ladder.py --check   # no binary
python3 apps/tungsten/scripts/flagship_ladder.py --grids 960 16
export TUNGSTEN_HIP_BIN=/flash/project_462001519/juaho/build/<tree>/apps/tungsten/tungsten_hip
./docs/lumi_slurm/submit_tungsten_hip_flagship.sh pilot   # 1 node first
./docs/lumi_slurm/submit_tungsten_hip_flagship.sh control # matched 1x8x1 + 2-node grids
# only after HBM/timing health and a justified layout:
./docs/lumi_slurm/submit_tungsten_hip_flagship.sh weak     # 1..32, not 60
./docs/lumi_slurm/submit_tungsten_hip_flagship.sh strong    # optional 768³
FLAGSHIP_ALLOW_60=1 ./docs/lumi_slurm/submit_tungsten_hip_flagship.sh max
./docs/lumi_slurm/submit_tungsten_hip_flagship.sh collect   # CSV from scratch
```

`control` submits the issue #13 diagnosis set (same frozen 768³ / 960³
physics): one-node min-surface vs explicit `1×8×1` (pencils off and on),
then two-node `1×8×2`, `1×1×16` slab, `2×2×4`, and `1×4×4`. Do not treat
the 1–32-node `1×8×N` curve as interpretable until the one-node matched
control exists.

Each run directory records revision, dirty count, `sha256` of
`tungsten_hip`, `module list`, `OPENPFC_FFT_NODE_GRID`,
`OPENPFC_FFT_PROC_GRID`, resolved proc grid, local brick, pencils flag,
and a copy of `input.toml`. Revision is captured on the login node with
`git -C` (worktrees included) because compute nodes have no `git` after
`module purge`. Fill
[`tungsten_lumi_g_flagship.csv`](../report/data/tungsten_lumi_g_flagship.csv)
with `--collect`; do not transcribe `wall_step` by hand.

Account `project_462001519`, partition `standard-g`. Do not run these
jobs on the login node.

## Results

Machine-readable table:
[`docs/report/data/tungsten_lumi_g_flagship.csv`](../report/data/tungsten_lumi_g_flagship.csv).
It ships **header only** until production jobs are admitted. Plot:

```bash
python3 apps/tungsten/scripts/plot_flagship_scaling.py
```

No interpolation. If 60 nodes hits a memory or decomposition limit,
record that row as a failure rather than shrinking N to decorate
efficiency.

## See also

- [LUMI GPU scaling](lumi_gpu_scaling.md) (strong 768³)
- [LUMI Slurm](../lumi_slurm/README.md)

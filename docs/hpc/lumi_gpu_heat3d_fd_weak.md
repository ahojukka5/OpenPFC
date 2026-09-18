<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# LUMI-G Heat3D FD weak scaling (issue #25)

Constant **256³ owned interior cells per GCD** for `heat3d_fd_hip` on LUMI-G.
This is not the fixed-global-grid strong curve in
[lumi_gpu_scaling.md](lumi_gpu_scaling.md).

## Frozen protocol

| Knob | Value |
|------|--------|
| App | `heat3d_fd_hip` |
| FD order | 2 (halo width 1) |
| I/O | off |
| Timed steps | 105 total, `HEAT3D_WARMUP=5` (100 admitted) |
| `dt` | 0.01 |
| Decomposition | `OPENPFC_FD_PROC_GRID` |
| Local gate | `HEAT3D_REQUIRE_INTERIOR=256x256x256` |
| Metric | median barriered `wall_step` |
| Scratch | `/scratch/project_462001519/juaho/openpfc-scaling/heat3d-fd-weak/` |

Do not set `HEAT3D_DIAG_TIMING` on admitted clean runs.

| Nodes | GCDs | proc grid | global grid |
|------:|-----:|-----------|-------------|
| 1 | 8 | `2x2x2` | `512x512x512` |
| 2 | 16 | `2x2x4` | `512x512x1024` |
| 4 | 32 | `2x4x4` | `512x1024x1024` |
| 8 | 64 | `4x4x4` | `1024x1024x1024` |
| 16 | 128 | `4x4x8` | `1024x1024x2048` |

## Submit (LUMI login)

```bash
python3 apps/heat3d/scripts/fd_weak_ladder.py --check
export HEAT3D_HIP_BIN=/flash/project_462001519/juaho/build/<tree>/apps/heat3d/heat3d_fd_hip
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh clean
# after the clean series, optional attribution:
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh diag
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh collect
```

Halo-only control (same grids, `OPENPFC_FD_PROC_GRID` set):

```bash
srun ... ./examples/23_halo_microtiming --hip \
  --nx 512 --ny 512 --nz 1024 --halo 1 --iters 50 \
  --output /scratch/project_462001519/juaho/openpfc-scaling/heat3d-fd-weak/halo.json
```

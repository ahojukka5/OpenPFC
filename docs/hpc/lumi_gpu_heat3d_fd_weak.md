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
# after the clean series, attribution (max rank-local median):
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh diag
export HEAT3D_HALO_BIN=/flash/project_462001519/juaho/build/<tree>/examples/23_halo_microtiming
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh halo
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh collect
```

`HEAT3D_DIAG` headline times are the **max across rank-local medians**,
matching the barriered clean `wall_step`. `halo_minmax` / `rhs_minmax` /
`update_minmax` are the spread.

## Admitted FD-2 result

Every admitted rank owns exactly `256^3`. `gpu_aware=1`, `contiguous=1`,
no field I/O. Clean production path: blocking halo → FD RHS → Euler →
device sync.

Clean barriered `wall_step` (100 admitted samples, SHA `d5b7e856`, dirty=0,
binary `d7a9338bdc431b8456c173da9d0f550ff3902933a315389b945bb0769d3362e5`):

| nodes | GCDs | proc grid | wall/step | weak efficiency | job |
| ----: | ---: | --------- | --------: | --------------: | ----: |
|     1 |    8 | 2x2x2     |  0.914 ms |           1.000 | 22151601 |
|     2 |   16 | 2x2x4     |  0.908 ms |           1.007 | 22151602 |
|     4 |   32 | 2x4x4     |  0.941 ms |           0.971 | 22151603 |
|     8 |   64 | 4x4x4     |  1.060 ms |           0.863 | 22151604 |
|    16 |  128 | 4x4x8     |  1.024 ms |           0.892 | 22151605 |

HIP-event attribution (`HEAT3D_DIAG_TIMING=1`, not admitted wall time;
`reduce=max_rank_median`; SHA `d12eeb4e`, binary
`b09d9b59be6234e25683b63fd7c2d55093522cc5392ce9b669f769c339d94913`)
and halo-only `23_halo_microtiming --hip` (binary
`636090f527f9707509c5ab9a2aa5c2292132918fd6518a104479c9e506e8c299`):

| nodes | clean | halo | RHS | update | halo-only | off-node/rank | jobs |
| ----: | ----: | ---: | --: | -----: | --------: | ------------: | ---- |
|     1 | 0.914 | 0.206 | 0.382 | 0.329 |     0.211 |             0 | 22151983 / 22151991 |
|     2 | 0.908 | 0.199 | 0.382 | 0.330 |     0.204 |             1 | 22151984 / 22151992 |
|     4 | 0.941 | 0.221 | 0.382 | 0.330 |     0.294 |             2 | 22151986 / 22151993 |
|     8 | 1.060 | 0.284 | 0.384 | 0.330 |     0.350 |             3 | 22151987 / 22151994 |
|    16 | 1.024 | 0.313 | 0.383 | 0.331 |     0.330 |             3 | 22151989 / 22151995 |

Times in milliseconds. Placement is uniform: every rank at a given node
count has the same off-node-face count. RHS and update stay flat
($+0.2\%$ and $+0.5\%$ from 1 to 16 nodes). Halo grows $+52\%$
($0.206\to 0.313$ ms) and accounts for the $+0.110$ ms clean-step
increase. The two-node Slingshot transition (one off-node face per rank)
does not increase wall time. Component sum / clean wall is 1.00, 1.00,
0.99, 0.94, 1.00; the diagnostic path is not the admitted timer.

Issue #25 H1 (halo/network) is supported. H2 is not primary: halo-only
time is not flat. H3 is not supported.

Issue #48 tests whether that blocking halo can be hidden. Keep admitted
weak-scaling runs on `HEAT3D_HALO_OVERLAP=0` until a clean production
A/B on this protocol justifies changing the default. Overlap modes `1`
(interior during `Waitall`) and `2` (`MPI_Testall` progress) must not
replace the clean barriered `wall_step`.

### Overlap A/B (not admitted; six-launch border)

Same protocol, SHA `2174c960`, binary
`cef9b6099bf910707f76b73c4afb72bd7eaac846b2b79d1cedc668aeb49209a2`.
`HEAT3D_HIP_CHECKSUM_HEX` bitwise identical on all jobs
(`sum_u=0x1.7785970621d7cp+3`, `sumsq_u=0x1.4ac44f14c882ep-1`).

Clean median `wall_step`:

| nodes | ov=0 | ov=1 | ov=2 | jobs |
| ----: | ---: | ---: | ---: | ---- |
|     2 | 0.907 | 0.945 | 0.960 | 22161584 / 86 / 87 |
|     4 | 0.937 | 0.950 | 0.961 | 22161588 / 89 / 90 |

Diagnostic `HEAT3D_OVERLAP` (ms, max rank-local median; jobs
22161679–84). Blocking halo/RHS/update match the admitted #25 split.
Mode 1 exposed wait ≈ inner kernel (H-progress-2: GPU-aware `Waitall`
does not complete while the default-stream interior kernel runs).
Mode 2 `MPI_Testall` drops exposed wait to ~0.060 ms, but six thin
border launches cost ~0.127 ms versus ~0.381 ms for the full RHS, so
clean wall stays worse than blocking.

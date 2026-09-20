<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Finite-difference order × halo-overlap scaling (issue #108)

Mechanism study: as Heat3D FD order increases, do wider halos hurt
large-scale weak scaling more than extra stencil work helps two-stream
overlap, or vice versa?

This is not an accuracy comparison. Do not change production FD physics
from these jobs.

Issue: [#108](https://github.com/ahojukka5/OpenPFC/issues/108).

## Frozen weak-scaling protocol

Keep exactly `256³` owned interior cells / GCD. Production two-stream
halo overlap. One rank/GCD. I/O off. `dt=0.01`, 105 steps,
`HEAT3D_WARMUP=5`. Process grids follow the admitted FD-2 construction
(double the smallest axis, z then y then x on ties):

| nodes | ranks | proc grid | global |
|------:|------:|-----------|--------|
| 1 | 8 | `2x2x2` | `512³` |
| 8 | 64 | `4x4x4` | `1024³` |
| 32 | 256 | `4x8x8` | `1024×2048×2048` |
| 128 | 1024 | `8x8x16` | `2048×2048×4096` |
| 512 | 4096 | `16x16x16` | `4096³` |
| 1024 | 8192 | `16x16x32` | `4096×4096×8192` |

Grids are exported as `OPENPFC_FD_PROC_GRID=gx x gy x gz` because
`sbatch --export` splits on commas.

## Orders

`2, 4, 8, 12, 20`. Halo width is `order/2`. Geometric packed-face bytes
per rank are `6 × 256² × width × 8`. If an order fails for a real
numerical or software reason, record the failure rather than substituting
another order.

If a higher order is unstable at `dt=0.01`, keep wall/step on the frozen
timestep and treat time-to-solution as a separate question.

## Account and scratch

New jobs bill **`project_462001245`**.

```text
/scratch/project_462001245/juaho/openpfc-scaling/fd-order-108/
```

Do not mix with the #25 FD-2 tree or with issue #106.

## First wave

Clean production matrix: every declared order × the node ladder.
Independent repeats (3) at 128, 512, and 1024 nodes. Component
diagnostics (`HEAT3D_DIAG_TIMING=1`) only at 8, 128, and 1024 nodes.

## Submit

```bash
export HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip
./docs/lumi_slurm/submit_fd_order_scaling.sh check
./docs/lumi_slurm/submit_fd_order_scaling.sh clean
./docs/lumi_slurm/submit_fd_order_scaling.sh diag
./docs/lumi_slurm/submit_fd_order_scaling.sh collect
./docs/lumi_slurm/submit_fd_order_scaling.sh analyze
```

Optional: `NODES=8,128`, `ORDERS=2,8`, `REPEATS=1`, `DRY_RUN=1`.

Clean jobs unset overlap/diagnostic knobs so the binary keeps the
production two-stream default. Diagnostic jobs set
`HEAT3D_KEEP_OVERRIDES=1` and `HEAT3D_DIAG_TIMING=1` only.

## Analysis

Compare clean wall/step and weak efficiency across orders at the same
node count. Attribute with halo bytes/rank, exposed MPI wait, interior
and boundary kernels, and off-node faces. Look for monotone degradation,
improved scaling at moderate/high order, or a non-monotone optimum.

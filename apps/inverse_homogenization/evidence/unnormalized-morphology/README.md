<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Native morphology of the converged small hold

The frozen protocol compares states 0, 300 and 1693 of the same 32x32x61,
dx=1 hold. This is **not** the earlier 64-grid fixed-300-step experiment.
Job22181820 runs the already-qualified native CPU forward executable at eight
MPI ranks; all six continuous/threshold evaluations converge. It does not
optimize or modify these fields. The tensors are unpenalized forward material
responses, distinct from the SIMP-penalized tensors in the optimization history.

At state300, the continuous tensor differs from the certified final tensor by
14.85% in relative Frobenius norm (denominator: final tensor). The corresponding
thresholded difference is 33.47%. CPU terminal matrices reproduce the archived
GPU endpoint matrices to relative errors 1.63e-11 and 1.60e-12, respectively.
These differences quantify continued evolution of this small hold, not
resolution convergence or physical validation.

Native periodic six-neighbor voxel diagnostics on h>0.5 report:

| Accepted state | Solid components | Void components | Solid percolation xyz | Grey fraction | Opening loss r=1 | Opening loss r=2 |
|---|---:|---:|---|---:|---:|---:|
| 0 | 4 | 3 | 111 | 0.497551 | 0.121621 | 0.353467 |
| 300 | 1 | 1 | 101 | 0.683978 | 0.008118 | 0.038965 |
| 1693 | 1 | 1 | 001 | 0.522957 | 0.002203 | 0.007965 |

Opening radii are one and two cells (dx=1) under the native lattice operation;
these diagnostics do not certify a manufacturing process. Solid percolation
is not void percolation. Grey fraction applies to the continuous field; it is
zero after thresholding. Both modes have the same h>0.5 topology. The final
thresholded volume fraction is 0.1889248207, below the continuous constraint
0.2575865186. All six final compliance-derived axial Poisson ratios are positive.

`raw.json.gz` retains the frozen protocol, scheduler/module records, batch
recipe, exact commands, complete native logs and exit codes. `inputs.tar.gz`
contains exact state0/state300 fields; state1693 is the existing neighboring
`../unnormalized-converged/h_final.bin.gz`. `provenance.json` binds source,
input and binary identities. Native measurements regenerate by building the
recorded source with `scripts/build.sh`, unpacking inputs, decompressing the
terminal input, and executing the six recorded commands with relocated input
paths under the recorded module stack. Every new Slurm run must explicitly
use account `project_462001245`.

The committed reduction is reproducible without MPI or scratch:

```sh
python3.11 apps/inverse_homogenization/scripts/summarize_hold_morphology.py \
  apps/inverse_homogenization/evidence/unnormalized-morphology/raw.json.gz \
  apps/inverse_homogenization/evidence/unnormalized-converged/material-reports.json \
  summary.json
```

NumPy is required (measured with 2.4.6). The entrypoint checks all native exit
codes, solve convergence, matrix/compliance consistency, Poisson ratios and
CPU/GPU terminal agreement before writing the full machine-readable summary.
The table above is rounded from that summary; do not treat it as an independent
source. No target, threshold or geometry was retuned after seeing results.

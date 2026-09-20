<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Historical 300-budget material re-evaluation

The original normalized 300-step arm A stored a post-budget `h_final.bin`
without the later accepted-state convergence semantics. It must not be called
a certified accepted step 300. Here the qualified native forward consumer
re-evaluates those exact archived bytes and their strict >0.5 threshold.

Against the new unnormalized converged 64-grid material, relative full-tensor
Frobenius differences are 0.3528737235 continuous and 0.2970208741 thresholded,
normalized by each corresponding new final tensor norm. Historical compliance
nu_xy is -0.0759245341 continuous and -0.3831037407 thresholded; both historical
variants have 111 threshold-defined solid components. The new converged
material has positive axial Poisson ratios and two solid components.

This comparison changes force normalization as well as optimization duration.
It cannot isolate the effect of running longer. The same-algorithm accepted
step-300 comparison remains separately preserved in `converged-resolution`.
No old topology or auxeticity is promoted into a converged result.

`historical-field.bin.gz` preserves the exact consumer input. `raw.json.gz`
contains original provenance/history plus native consumer commands, logs,
environment, scheduler and frozen protocol. Both native consumer solves pass.
`provenance.json` identifies all executable and input identities; `summary.json`
is generated from raw output by the actual reducer entry point:

```bash
python3.11 apps/inverse_homogenization/scripts/audit_converged_resolution.py \
  --archive apps/inverse_homogenization/evidence/converged-resolution/raw.json.gz \
  --historical apps/inverse_homogenization/evidence/historical300/raw.json.gz \
  --output results/historical300
```

Use Python 3.11 and NumPy 2.4.6. To repeat native evaluation, build the pinned
consumer source via scripts/build.sh, decompress the field, and replay the
archived continuous/threshold commands with local paths on eight MPI ranks.
All new compute uses account project_462001245; historical producer provenance
retains its original account without relabeling it.

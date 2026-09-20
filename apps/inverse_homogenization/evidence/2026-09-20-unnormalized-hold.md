<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

Follow-up to #74 / #77: qualify the existing unnormalized update over a longer
frozen trajectory, retaining the completed 160-state comparison unchanged.

The completed normalization=0 control (22179446) matches the baseline binary,
target and initial h190 byte for byte. All 160 elasticity solves pass; final
field equals the last accepted snapshot and thresholding matches. Total J has
no increases at CSV precision. Tail50 design RMS=0.000389786, tensor change
0.000333087, objective change0.00000381539; lag1/lag2 field RMS are
0.000391139/0.000781959. It is MAX_STEPS, not converged. The normalized baseline
has tail design RMS0.01490253 and lag2 much smaller than lag1. The local slope
test independently finds positive normalized and negative unnormalized slopes.

Scientific question: does the unnormalized update actually meet the original
convergence certificate, or merely replace oscillation with slow drift?

Freeze a separate 2000-accepted-state run from the same initial h190, dt=.04,
normalize=0, cap=.04, SIMP2, lambda_reg=.2, epsilon2, projected volume
.2575865186, continuation0 and original tolerances/window20/verification100.
Do not call the extended budget a matched-budget win; the matched 160-state
comparison is retained. Save every small-grid field for lag/morphology checks.
Stop with the reported CONVERGED/MAX_STEPS/ELASTICITY_FAILURE and preserve final
accepted state. A CONVERGED result requires independent history/hold checks and
final recomputation before admission. No target, tolerance, regularization or
production-default change. The normalized 64/128 jobs continue as exploratory
controls; do not cancel them. Use project_462001245, 8 GCD, four-hour allocation.

Executable work is a small driver extension stacked on #77, isolated from its
currently dirty synthesis files. This does not claim checkpoint equivalence or
qualify high-resolution normalization=0 by itself.

Admitted 64×64×121 and 128×128×242 `normalize=0` jobs **22180900** and
**22180901** (`6deeb76c`, `project_462001245`) both **CONVERGED** (steps
1374 / 1371, verified hold). SIMP and thresh `C_{12}>0`; grey ~0.358;
two solid components, xyz percolation. Certificate true; auxetic
production false. Compact record:
`2026-09-20-n0fixed-64-128.json`. Do not retarget Yang A3.

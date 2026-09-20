<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Converged unnormalized frozen hold

Job **22179764** satisfies the declared stopping certificate at accepted state
**1693**. The first candidate is 1593, followed by the original 100-state
verification hold. Tolerances remain design RMS 1e-4, objective change 1e-6 and
full tensor change 1e-4, with a 20-state consecutive window. All elasticity
solves pass. This is not proof of an exact stationary point or global optimum.

The frozen target, volume, SIMP=2, regularization=0.2, epsilon=2, dt=0.04,
projection and update cap are unchanged. This is the longer continuation of
[the 160-state numerical control](../unnormalized-control/README.md), restarted
from the same original input rather than checkpoint-resumed. Disabling the
elastic RMS normalization changes the iteration's force scaling; it does not
remove or retune the declared tensor objective. The original amplifying update
is retained as a distinct comparator, not claimed convergent by this result.

The field audit reads all 1694 accepted snapshots and independently reproduces
all design RMS and threshold-change diagnostics. The terminal field is exactly
the accepted state, SHA256
`c488298cc7647949f3e7d11c3bd0d689181271135d48c8695ccad1835ee0fffa`.
The saved threshold is exactly `h > 0.5`. The last 120 states have maximum
design RMS 9.99881e-5, objective change 2.4991e-7 and tensor change 7.56796e-5.
The scalar CSV cannot independently reconstruct the full 6x6 tensor increments;
those are evaluated by the qualified source and its regression tests.

## Numerical and material interpretation

Final objective components, at CSV precision, are total 0.00306072,
tensor 0.000255947, volume 1.14188e-26 and regularization 0.00280477.
The reported total objective decreases at every recorded step. Its tensor term
is larger than the initial 2.62272e-5: lower total energy did not improve every
component. No coefficient was retuned to hide this tradeoff.

Final mean density is 0.2575865185998933 and grey fraction is
0.5229572233606558. A separate full material evaluation (job **22180840**)
reloads the exact final field and verifies its byte identity after one accepted
state with positive dt. Its local label `MAX_STEPS`, step zero, describes this
one-state diagnostic evaluation; it does not replace the original optimization
certificate. All six elasticity loads pass for the continuous and thresholded
materials. `material-reports.json` preserves both complete unpenalized tensors,
compliances and solve reports. These differ from the SIMP-penalized convergence
tensor by design.

All six measured axial compliance-derived Poisson ratios are positive. The
continuous material ranges from 0.168916 to 0.309662; the thresholded material
ranges from 0.014301 to 0.380500. No conclusion about arbitrary rotated directions
is implied. Relative target Frobenius errors are 2.583278 and 0.618423,
respectively. Thresholding changes the volume fraction to 0.1889248207, so it
is not a volume-matched design satisfying the original optimization problem.
These results do not establish manufacturability or resolution independence.

## Replay and scope

`provenance.json` pins source, binary, initial and target identities, modules,
commands and output hashes. Exact terminal fields and history are compressed
with deterministic gzip. The full trajectory is omitted, but its input is
committed in the earlier control and `command.json` preserves every runtime
option. Build through `scripts/build.sh`, use account `project_462001245`,
and remap only artifact paths. Do not silently replace the input or target.

After regenerating the trajectory:

```sh
python3.11 apps/inverse_homogenization/scripts/audit_converged_hold.py \
  RUN_DIRECTORY VERIFIED.json
```

The audit uses NumPy 2.4.6. Replay may differ in parallel roundoff across runtime
stacks; bitwise historical outputs remain available here. The material
re-evaluation recipe and full reports preserve the independent endpoint check.
This small frozen hold is a numerical qualification, not a completed resolution
campaign. The next matched physical-domain 64/128 runs retain the objective,
original initial interpolant and continuation, changing only normalization.

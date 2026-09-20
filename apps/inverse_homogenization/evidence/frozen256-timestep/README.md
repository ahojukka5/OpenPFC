<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Frozen 256-grid timestep comparison

Both arms start from the same periodic refinement of accepted 128-grid state
600, with fixed target and final coefficients. Both reach pseudo-time 1.6 and
MAX_STEPS; neither reaches a verified convergence candidate. All elasticity
solves pass, including endpoint re-evaluation.

| Quantity | dt 0.04 | dt 0.01 |
| --- | ---: | ---: |
| Accepted terminal index | 40 | 160 |
| Initial objective | 0.00304835 | 0.00304835 |
| Final objective | 0.0285778 | 0.00302162 |
| Positive printed objective increments | 25 | 0 |
| Final design RMS | 0.0399536 | 0.0000394521 |
| Median reported step time, ms | 27197.2 | 26258.3 |

The larger step excites substantial regularization energy and nearly saturates
the update cap. The parabolically scaled step avoids this observed pathology
in the frozen interval. This supports a full-continuation experiment, not a
proof of nonlinear stability or convergence. Recorded objective monotonicity
is limited by CSV precision; clipping/projection contributions are not isolated.

Exact final accepted snapshots match h_final bytes; strict >0.5 thresholds
match independently. Continuous volume remains 0.2575865186. Grey fractions
are 0.587691 and 0.550888, threshold fractions 0.177066 and 0.178200. These
nonconverged endpoints are not resolution-comparison materials. HIP reports
maximum used memory 0.390625 GiB per GCD and total 47.3438 GiB for the scaled
arm. This single 128-GCD allocation does not establish a scaling optimum.

The checked-in raw archive preserves both commands, histories, endpoint
material JSON, scheduler/module records, failed/warning logs and the frozen
protocol. The deterministic reducer reproduces summary.json from that archive;
field-checks.json comes from its live-field mode. Source, binaries, input
hashes and exact seed regeneration are in provenance.json. The large fields
are regenerable from the committed 64-grid seed and pinned 128 trajectory;
scratch paths and job IDs alone are not the archive contract.

```bash
python3.11 apps/inverse_homogenization/scripts/audit_frozen256_timestep.py \
  --archive apps/inverse_homogenization/evidence/frozen256-timestep/raw.json.gz \
  --output results/frozen256-audit
```

Use Python 3.11 and NumPy 2.4.6. The next separately frozen experiment is #103:
a fresh 128/256 pair at dt .01, retaining stricter per-step tolerances, longer
verification time and checkpointed continuation. The warm-start diagnostic
must not replace that fresh-start campaign.

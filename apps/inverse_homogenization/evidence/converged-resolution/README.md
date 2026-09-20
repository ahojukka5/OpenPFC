<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Converged fixed-domain 64/128 controls

The frozen normalization=0 full-continuation controls both meet the original
20-state window plus 100-state hold and tolerances 1e-4/1e-6/1e-4. Exact
accepted final fields equal the manifest-indexed terminal snapshots and the
native consumers' inputs. Strict threshold fields equal h>0.5 byte-for-byte.
Every recorded elasticity solve passes. This is numerical stopping evidence,
not a global minimum or physical validation.

| Quantity | 64 x 64 x 121, dx=1 | 128 x 128 x 242, dx=0.5 |
| --- | ---: | ---: |
| Accepted converged step | 1374 | 1371 |
| Final total objective | 0.00267427 | 0.00267448 |
| Final tensor objective (SIMP) | 0.000320413 | 0.000321184 |
| Final grey fraction (0.1<h<0.9) | 0.3583036060 | 0.3583555616 |
| Continuous volume fraction | 0.2575865186 | 0.2575865186 |
| Strict threshold volume fraction | 0.2103281573 | 0.2103432900 |
| State300 continuous C error / final norm | 0.0975295151 | 0.0974385065 |
| State300 threshold C error / final norm | 0.5678801793 | 0.5654669398 |
| Solid components / periodic percolation | 2 / xyz | 2 / xyz |
| Opening loss at physical radius 1 | 0.0017075651 | 0.0011486866 |
| GCDs | 8 | 16 |
| Producer allocation elapsed | 2:17:46 | 2:08:35 |

Native unpenalized forward tensors differ by 0.151265% (continuous) and
3.37012% (threshold), relative to the 128 tensor Frobenius norm. CPU tensors
agree with independently exported GPU endpoint tensors to relative errors
below 1.6e-11. The six compliance-derived axial Poisson ratios are all positive:
continuous ranges about 0.1693--0.2721 and threshold about 0.0345--0.0932.
These are distinct from the penalized optimization tensor and its isotropic
shortcut. Thresholding changes volume and does not preserve the prescribed
continuous volume constraint.

At coincident physical grid nodes, final design RMS difference is
0.0019596496443, maximum difference 0.0242715107758, and strict-threshold
classification disagrees on 0.0007243511105 of nodes. No registration or
smoothing is applied. Two similar tensor/field results do not establish an
asymptotic resolution limit. Threshold C is noticeably more resolution-sensitive
than continuous C. Both native topologies contain an island solid fraction
about 0.178: percolation does not imply all material belongs to one component.
Opening radii are cells, so radius1 on the coarse grid is compared with radius2
on the fine grid. These voxel diagnostics do not certify a manufacturing process.

## Evidence and replay

`summary.json` contains all full matrices, compliances, directional ratios,
solver diagnostics, morphology values and independently reconstructed history
holds. `field-audit.json` preserves every dumped-state hash, exact terminal
identity and the field comparison. `raw.json.gz` is a JSON mapping from
`64|128/producer|material/relative-path` to exact text: histories, commands,
protocols frozen before execution, modules, scheduler records, endpoint JSON,
XDMF and native consumer logs. Large trajectories/final fields are regenerated.
The 3.7 MB compressed `initial64.bin.gz` preserves the exact seed rather than
requiring a still-existing old scheduler output.

`provenance.json` pins executable and interpolation sources, binary identities,
inputs, job/account/GCD information and replay commands. Decompress the seed,
regenerate initial128 with the pinned periodic trilinear refiner, build each
pinned executable through `scripts/build.sh`, and replay the exact archived
producer and consumer commands with only filesystem paths substituted. Keep
the same target, phases, physical cell, all continuation/step/stopping settings
and rank counts. Reuse of the recorded software/GPU stack is required for a
bitwise claim; other environments may be scientifically equivalent only.

The native protocol is a full-continuation comparison with its own state300.
It does not replace the older normalized fixed-300 experiment or the smaller
frozen hold. Preserve those as separately labelled historical evidence.

Reduce the committed text without claiming to re-audit omitted fields:

```sh
python3.11 apps/inverse_homogenization/scripts/audit_converged_resolution.py \
  --archive apps/inverse_homogenization/evidence/converged-resolution/raw.json.gz \
  --output /new/output
```

The actual live-field audit used:

```sh
python3.11 apps/inverse_homogenization/scripts/audit_converged_resolution.py \
  --live-root /scratch/project_462001245/juaho/inverse-homogenization \
  --output /new/output
```

The entrypoint regression also removes the terminal state and requires
rejection: successful elasticity on a late state cannot replace the hold.
Metric reconstruction uses printed CSV precision; intermediate accepted fields
were saved every20 states, so independent per-step field/tensor recomputation
is not claimed. The already-qualified driver supplies those metrics.

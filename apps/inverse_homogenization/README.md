<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Inverse homogenization (`apps/inverse_homogenization`)

**Issue [#161](https://github.com/VTT-ProperTune/OpenPFC/issues/161).** Explicit
catalog exception: this is the sixteenth application because it is
PDE-constrained inverse design on the existing FFT/phase-field stack, not
another PDE demo. Do not treat it as a licence to add a seventeenth app.

The central problem is **microstructure inverse design**: given a target
homogenized elasticity tensor \(C_{\mathrm{target}}\), find a periodic
microstructure whose effective tensor \(C_H\) matches it. This is not a
minimum-compliance structural topology-optimization demo.

The inverse map is **non-unique**. Many microstructures can realize nearly
the same \(C_H\).

## Status

Stages 1–8 of #161 are in tree. Remaining gaps are listed under
Limitations in the report chapter, not as missing stages.

| Stage | What | Where |
|-------|------|--------|
| 1 | Forward periodic \(C_H\) | `openpfc_apps/homogenization.hpp`, `openpfc_homogenize` |
| 2–3 | Allen–Cahn descent on the tensor-mismatch objective, volume penalty, perimeter | `phase_field_inverse.hpp`, `openpfc_inverse_homogenize` |
| 4 | Discrete mutual-energy \(\delta J/\delta h\) + finite-difference check | homogenization.hpp; Catch2 `apps-common-homogenization` |
| 5 | Free-topology campaigns (isotropic / auxetic / orthotropic; two seeds) | `openpfc_inverse_homogenize`, `auxetic_geometry.hpp`; jobs in the table below |
| 6 | Cahn–Hilliard process-parameter family vs rotating-square auxetic | `spinodal_generator.hpp`; job 21958468 |
| 7 | Percolation, islands, opening-loss metrics | `manufacturability.hpp`; job 21959215 |
| 8 | Genuine 3-D cells and HIP Green | `stage8_3d.sbatch`, `openpfc_homogenize_hip` / `openpfc_inverse_homogenize_hip` |

What is **not** claimed: grey linear \(C(h)\) cannot produce \(\nu<0\);
SIMP from noise did not enter the rotating-square basin; coupled dendrite
device Green is issue #157, not this app. Catalog stays closed.

## Convergence (issues #59 / #63)

`--steps` / `--max-steps` is a **ceiling**, not success. SIMP and
`lambda-reg` interpolate over `--continuation-steps` (default 300) and
then freeze. Convergence is impossible during continuation. After freeze,
three metrics must hold together for `--conv-window` (20) consecutive
iterates, then a `--verify-convergence-steps` (100) hold with the same
frozen problem. A broken hold rejects the candidate immediately.
Morphology-change fraction is a diagnostic, not a stop.

Logged row \(s\) compares consecutive **accepted** post-projection
states, not a post-update design RMS paired with a pre-update objective:

* design RMS \(\lVert h_s-h_{s-1}\rVert_2/\sqrt{N}<10^{-4}\) is measured
  **before** the Allen–Cahn update (not the pre-projection `step_rms`);
* \(\lvert J(h_s)-J(h_{s-1})\rvert/\max(1,\lvert J(h_{s-1})\rvert)<10^{-6}\);
* \(\lVert C_H(h_s)-C_H(h_{s-1})\rVert_F /
  \max(\lVert C_H(h_{s-1})\rVert_F,\varepsilon)<10^{-4}\).

\(J\) and \(C_H\) are the homogenization of that same accepted \(h_s\).
The in-place Allen–Cahn update then produces a trailing candidate
\(h_{s+1}\). On `CONVERGED`, `MAX_STEPS`, or `ELASTICITY_FAILURE` the
driver restores \(h_s\) before writing `h_final.bin` / `h_thresh.bin`
and before the unpenalized `FINAL_RECOMPUTE`. The declared material is
exactly the certified accepted field. The first iterate has no
predecessor and is never quiet. CSV volume is the accepted-state mean,
not the post-update candidate.

CPU and HIP write the same 26-column iterate schema
(`kInverseCsvHeader`). The unpenalized final \(C_H\) is a
`# FINAL_RECOMPUTE` comment, not a data row. A `# CERTIFIED_STEP`
comment records the iterate index. Comment lines are not iterate
records. The final accepted field is always snapshotted, even
between `--dump-every` points; `MAX_STEPS` is not labelled `CONVERGED`.

HIP SIMP uses the same `simp_density` / `simp_chain` helpers as CPU:
elasticity on \(h^p\), sensitivity chain rule \(p h^{p-1}\). \(p=1\) is
identity and does not call \(\mathrm{pow}(h,0)\). Before #60 the HIP
path set `spec.simp_p` but did not apply the transformation, so earlier
HIP SIMP-continuation claims need qualification.

Job 22162138 (300 fixed steps) still had `step_rms≈0.019` at the last
iterate, so those animations were cut while the design was moving.

## Checkpoint / restart (issue #72)

`--checkpoint-dir` publishes a complete generation (`gen_<next_step>/`
with `h.bin`, `h_prev.bin`, `state.txt`, and HIP `dump_steps.txt`) and
then atomically retargets `CURRENT`. A walltime kill during the write
leaves the previous published generation loadable. `--restart=DIR` reads
`CURRENT` (or an explicit generation directory). Schema 3 fingerprints
the target \(C\), phase moduli, SIMP / regularization endpoints,
step/projection, window, and tolerances, and records whether the bundle
is a running or terminal state. `--max-steps` is a run budget and may
change; any other mismatch is rejected. `CONVERGED` / `MAX_STEPS` /
`ELASTICITY_FAILURE` bundles are terminal snapshots and are
not continuation restarts. Schema 1 and schema 2 files do not load.
The replay entrypoint checks changing fields and interruption inside an active
verification hold; `--cpu` selects the CPU executable's output contract.
Keep each allocation's CSV under a distinct name. The checkpoint's next-step
index defines the retained prefix of an interrupted allocation; append-only
CSV files can contain a repeated or incomplete trailing row after a kill.
The `CURRENT` generation is the restart authority, not the newest CSV row.
No fsync/power-loss or different-rank equivalence guarantee is claimed.

Checkpoint loading validates tracker counters before restoration. A candidate
must have exactly the remaining verification hold implied by its consecutive
quiet-state count, and a `CONVERGED` snapshot must have completed that hold.
Impossible flags, counters and non-finite accepted-state values are rejected;
restarting cannot turn a negative hold counter into immediate convergence.
This validation does not by itself qualify file publication or full-field
restart equivalence.
The `scripts/replay_checkpoint_state.py` field and hold replay compares actual
GPU fields through an interrupted verification hold and a changing trajectory.

A restart requires `--max-steps` strictly greater than the saved `next_step`.
An exhausted requested budget is rejected before field restoration, so an
unevaluated next field cannot be exported as the previous accepted material.

Checkpoint metadata writes are checked through close before publishing
`CURRENT`. A staging or publication failure exits nonzero on every rank and
leaves the preceding published generation available. GPU restart rejects a
missing, malformed or inconsistent snapshot-index ledger instead of resetting
its frame identities. Field sizes are checked before publication; this is not
a guarantee against storage hardware corruption or power loss.

## Finite-strain forward ladder (issue #55)

Research #484 needs a
finite-strain Poisson along a loading path. That is **not** the
small-strain \(C_H\) from `openpfc_homogenize`.

`openpfc_finite_strain_forward` runs the frozen 2-D plane-strain ladder:

1. compressible neo-Hookean and St.\ Venant--Kirchhoff
   \(P=\partial W/\partial F\);
2. homogeneous uniaxial \(F_{11}\) with transverse relaxation \(P_{22}=0\);
3. two-material \(y\)-laminate (stiff \(E=1,\nu=0.3\);
   compliant \(E=0.1,\nu=0.1\));
4. tangent \(\nu_t=-\mathrm{d}\ln F_{22}/\mathrm{d}\ln F_{11}\) by finite
   difference of the same shipped \(F_{22}(F_{11})\);
5. Newton residual / \(J>0\) stability.

Catch2 `inverse-homogenization-finite-strain-tests` calls those shipped
functions.

`openpfc_finite_strain_inverse` then runs the frozen research #484
16² experiment: rotating-square half sweep, single-material+void versus
hinge-painted stiff/compliant/void, neo-Hookean, \(F_{11}\) path
\(1.02\)–\(1.20\). Catch2
`inverse-homogenization-finite-strain-grid-tests` drives the periodic
homogenizer (uniform cell recovers the homogeneous \(P_{22}=0\) path).
The frozen family does **not** produce \(\nu_t>0\) then \(\nu_t<0\).

```bash
./apps/inverse_homogenization/openpfc_finite_strain_forward \
  --evidence apps/inverse_homogenization/evidence/2026-09-19-finite-strain-forward-55.json
./apps/inverse_homogenization/openpfc_finite_strain_inverse \
  --evidence apps/inverse_homogenization/evidence/2026-09-19-finite-strain-inverse-484.json
```

## Reuse

The elliptic solve is `EigenstrainMicroelasticity` in
[`apps/common/include/openpfc_apps/microelasticity.hpp`](../common/include/openpfc_apps/microelasticity.hpp).
Homogenization is the same Green-operator problem with **zero eigenstrain**
and an imposed macroscopic strain (`applied_strain`). There is no second
elasticity implementation, no FEM, and no unstructured mesh.

## Forward driver

```bash
mpirun -n 1 ./apps/inverse_homogenization/openpfc_homogenize \
  --shape=homogeneous --nx=16 --ny=16 --nz=16 --volume=1
mpirun -n 2 ./apps/inverse_homogenization/openpfc_homogenize \
  --shape=laminate-z --E-solid=1 --E-void=0.25
```

`--shape` is `homogeneous`, `laminate-z`, `sphere`, `rotating-cubes`
(OpenPFC #31), `rotating-squares` (OpenPFC #33 extrusion consistency
oracle, not a 3-D metamaterial), `reentrant-3d` (OpenPFC #36 Yang /
Evans 3-D re-entrant honeycomb), or `yang-a3` (OpenPFC #43 Table 1
design A3; density-qualified, not auxetic in \(\nu_{zx}\)). The printed \(C_H\) is the
**engineering Voigt** \(6\times 6\) (order \(11,22,33,23,13,12\),
\(\gamma=2\varepsilon\)). A homogeneous isotropic material therefore reports
\(C_{44}=\mu\), not \(2\mu\). The driver also prints \(S=C_H^{-1}\),
compliance Poisson ratios, SPD / \(\lambda_{\min}\), cubic spreads, and
periodic percolation. `--dump-dir` writes gathered `h.bin` / `h.xdmf`.

`HOMOGENIZATION_CHECKSUM` is \(\lVert C_H\rVert_F\); the smoke test greps it.

## Inverse driver (Allen–Cahn)

```bash
mpirun -n 1 ./apps/inverse_homogenization/openpfc_inverse_homogenize \
  --target=isotropic --E-target=0.9 --nu-target=0.25 \
  --volume=0.5 --nx=16 --ny=16 --nz=16 --steps=10 --init=noise
```

`--target` is `isotropic`, `auxetic` (negative Poisson via `--nu-target`),
`orthotropic` (`--C11 --C22 --C12 --C66`), or `file` (`--C-target-file`
with a 6×6 Voigt text matrix). `--init` includes `spinodal`, `yang-a3`,
and `--load-bin` for a Fortran `h` brick. The loop is Takezawa-style
Allen–Cahn, not Cahn–Hilliard and not MMA. `INVERSE_CHECKSUM` is the last
\(J\). Optional `--csv=PATH` writes the per-step history.

Final reports (OpenPFC #28 / research #476) print the full \(6\times 6\)
\(C_H\) of the **initial**, **final physical**, and **\(h>0.5\)
thresholded** fields, plus compliance Poisson ratios
\(\nu_{xy}=-S_{12}/S_{11}\) (and cyclic), SPD / \(\lambda_{\min}\),
cubic spreads, elasticity residuals, and periodic percolation on any
rank count. `--dump-dir` writes gathered Fortran `h_init.bin`,
`h_final.bin`, `h_thresh.bin` and optional `h_%04d.bin` snapshots.
The shortcut \(C_{12}/(C_{11}+C_{12})\) is still printed as
`nu_shortcut`; qualification uses the compliance tensor.

### Stage 5 campaigns (LUMI-C `standard`)

Not paper claims. Grey linear interpolation **cannot** produce \(\nu<0\).

| Job | stepper | isotropic rel-F | volume | notes |
|-----|---------|-----------------|--------|--------|
| 21949415 | raw gradient, `dt=0.08` | **0.050** | 0.31 | first step collapsed volume \(0.55\to0.25\) |
| 21949811 | RMS-normalised, `dt=0.03` | 0.129 | 0.40 | step 1 is \(0.55\to0.52\); volume still bleeds |
| 21949859 | + volume projection | 0.375 | **0.50** | volume held; design stayed fully grey (Voigt-like) |
| 21950094 | + SIMP \(p=3\) | 0.209 | **0.50** | still fully grey after 40 steps; \(C_{12}\) still \(>0\) |
| 21954427 | elastic-only RMS + SIMP \(1\to 3\), \(\lambda_r\) \(0.05\to 0.4\) | 0.261 | **0.50** | **grey \(0.999\to 0.674\)** — first real binarization |
| 21954505 | same continuation, 80 steps; 2-D auxetic \(E_{\mathrm{void}}=0.08\) | 0.261 | **0.50** | auxetic grey \(1\to 0.79\), \(C_{12}=+0.25\) still not negative; orthotropic grey \(0.88\) |
| 21954959 / 21955207 | 64² auxetic, two seeds; continue +150 steps | — | **0.50** | grey 0.63 / 0.58; morphologies **uncorrelated** (\(r=0.19\)). SIMP \(C_{12}\) 0.065→0.026; **physical** \(C_{12}=0.22\). Thresholded \(h>0.5\): \(\nu_{\mathrm{bin}}=0.032\) (seed continue), still \(>0\). |
| 21956076 | **rotating-square seed** (hinged, half=0.200) | — | 0.64 | **\(\nu=-0.123\)**, \(C_{12}<0\)**. Inverse 80 steps still auxetic (\(\nu_{\mathrm{bin}}=-0.080\)). Re-entrant honeycomb on this grid is not auxetic. |
| 21958468 | Stage 6 CH process family vs rotating square | — | 0.35–0.65 | **No CH \((c_0,\kappa,a_y)\) is auxetic.** Best spinodal \(\nu_{\mathrm{bin}}=+0.23\); rotating square \(\nu=-0.123\). Process restriction loses the auxetic quadrant. |
| 21959215 | Stage 7 manufacturability | — | — | Rotating square: 1 solid component, percolates \(x,y\), opening loss \(r=1\) is 0. Disconnected squares (`half=0.16`): 4 islands, no solid percolation. Spinodal: 7+7 components, grey 0.83. |
| 21964843 | Stage 8 $64^3$ 3-D inverse (CPU) | — | 0.5 | 1 rank: 12 steps, 6 solves/step, 7.7–4.7 s/step, 224 MB. 8 ranks: 1.05–0.62 s/step. |
| 21967096 | $128^3$ spinodal $\mathbf{C}_H$, 8 CPU ranks | — | 0.5 | MPI-destructor fix: **finished**. 237 MB/rank, $\nu=0.30$. |
| 21967095 | HIP Green on 1 GCD | — | — | Homogeneous $C_{11}=1.346153846$ (CPU match). Rotating-square $32^2$: **$\nu=-0.075$**, $C_{12}<0$. $64^3$ homogeneous wall **7.59 s**. Inverse 2 steps converged. |
| 22151582 / **22151622** | #31 3-D rotating-cube **forward** oracle (not inverse) | — | 0.51 | Eight cubes, Attard–Grima RRU idea. Job 22151582 (`half=0.210`) $\nu=+0.155$ at $32^3$, SPD, perc $xyz$. One hinge correction 22151622 (`half=0.200`, $\angle=0.45$): $32^3$ **$\nu=+0.159$**, $64^3$ **$\nu=+0.158$**, both SPD, 1 solid component, perc $xyz$. **Not auxetic.** Stop; no Allen–Cahn. |
| **22156718** | #33 extruded rotating-square **consistency** oracle | — | 0.64 | Same seed as 21956076 (`half=0.200`, $\angle=0.45$, $E_{\mathrm{void}}=0.02$). $64\times64\times1$: $\nu_{\mathrm{shortcut}}=-0.123$, $\nu_{xy}=-0.139$. $64\times64\times4$ extrusion: **identical** $C_H$, $\nu_{xy}=-0.139$, perc $xyz$, SPD. Homogeneous $\nu=0.3$. **3-D path preserves in-plane auxeticity.** PR #32 is a geometry failure, not a forward-model failure. |
| **22157558** | #36 3-D re-entrant honeycomb **forward** oracle | — | 0.29 | Yang/Evans TRH, $t=0.06$, inset=0.14, all four 0.5-squares. $32^3$ $\nu_{xy}=+0.213$, $64^3$ $\nu_{xy}=+0.219$, $\nu_{zx}\approx+0.31$, SPD, perc $xyz$, 1 component. **Not auxetic.** One representation correction used. Not inverse. |
| **22159921** | #40 threshold archived 2-D inverse (jobs 22112442/443) | — | 0.50/0.57 | Step 600 **not rerun**. $256^2$ continuous $\nu_{xy}=-0.140$ (grey 0.32); $h>0.5$ $\nu_{xy}=-0.166$, perc $xy$, SPD. $1024^2$ continuous $\nu_{xy}=-0.128$ (grey 0.45); $h>0.5$ $\nu_{xy}=-0.162$. **Binarization preserves auxeticity.** |
| **22160325** | #40 volume-matched binary of the same bricks | — | **0.50** | $t_v$ from order statistics of $h$ (not from $\nu$). $256^2$ $t_v=0.6978$, $\nu_{xy}=-0.166$; $1024^2$ $t_v=0.6738$, $\nu_{xy}=-0.171$. Both SPD, 1 solid component, perc $xy$. **Volume-preserving binary stays auxetic.** |
| **22160647** | #43 Yang A3 density-qualified forward | — | 0.26/0.25 | Published box $2(H-L\cos\theta)$, square $t$, two $z$-stories. $64\times64\times121$ $\nu_{zx}=+0.365$ vf $0.258$; $96\times96\times182$ $\nu_{zx}=+0.365$ vf $0.248$ (Table 1 $0.233$, Wang $0.165$). SPD, perc $xyz$, 1 component. **Not auxetic.** dirty=1 was Slurm-only (blobs match clean commit). |
| **22160921 / 22160972 / 22160975** | #43 $E_{\mathrm{void}}$ ladder, $64\times64\times121$, dirty=0 | — | 0.258 | Frozen $E_v/E_s=0.02,0.002,0.0002$. $\nu_{zx}=+0.365,+0.407,+0.418$ (all SPD, converged). $\nu_{xy}$ becomes negative at lower contrast; **Yang's $\nu_{zx}$ does not.** |
| **22161950** | #9 model-specific 3-D oracle, $E_{\mathrm{void}}=0.002$ | — | 0.248 | $96\times96\times182$ $\nu_{xy}=-0.267$, $\nu_{zx}=+0.398$, SPD, perc $xyz$, dirty=0. Sign-stable vs 22160972 ($64\times64\times121$ $\nu_{xy}=-0.262$). **Not Yang $\nu_{zx}$.** |
| **22162138** | #9 A/B inverse, $64\times64\times121$, $C_{\mathrm{target}}$ from 22160972 | — | 0.258 | Spinodal **and** Yang seed both reach thresholded $\nu_{xy}<0$ (A $-0.383$, B $-0.327$), $J_{\mathrm{tensor}}\to0$. Generic basin access **succeeds** for this attainable tensor. Previous $64^3$ isotropic-auxetic failure was target/protocol specific. |

The double well drives a *uniform* grey field to the wells (Catch2). A
sharp/tanh interface with large \(\lambda_r/\varepsilon\) inverts bands
because the spectral Laplacian Gibbs term dominates \(W'\). Auxetic at
\(E_{\mathrm{void}}=0.02\) lost Eyre–Milton convergence at step 13
(grey already \(0.96\)).

## Tests

`ctest -R homogenization` (HeFFTe builds):

* `apps-common-homogenization` — homogeneous oracle, laminate/Postma,
  cubic symmetry, closed-form homogeneous sensitivity, finite-difference
  gradient check on random \(h\).
* `apps-common-homogenization-mpi` — same binary on two ranks.
* `inverse-homogenization-homogeneous-smoke` — the CLI entry point.

Build through `./scripts/build.sh` only. On LUMI, **configure** may run on
the login node (FetchContent needs the network); **compile, test, debug and
run** go to `standard` (CPU) or `standard-g` (HIP). Never compile on the
login node. CPU suite:

```bash
sbatch apps/inverse_homogenization/slurm/build_and_test_cpu.sbatch
```

## Workflow

```
h(x) → C(x) → six periodic elasticity solves → C_H[h] → J → dJ/dh
     → Allen–Cahn step or CH process map → new h(x)
```

`PeriodicHomogenizer::compute` and `objective_sensitivity` are the first two
arrows after \(C(x)\). `openpfc_inverse_homogenize` runs the rest as
Takezawa-style Allen–Cahn; Stage 6 substitutes a Cahn–Hilliard process
map for the free-topology step. HIP twins
(`openpfc_homogenize_hip`, `openpfc_inverse_homogenize_hip`) use a
device Green operator (FFT + Eyre–Milton) on the same loop.

## Showcase (issue #8)

One LUMI-G node (8 GCDs). Heavy frames go to scratch, never flash.
The production evidence is jobs 22112442 (256²) and 22112443 (1024²).

```bash
./scripts/build.sh --machine=lumi --partition=standard-g --no-test --no-submit \
  --build-dir=/flash/project_462001519/juaho/build/openpfc-lumi-showcase-inv
sbatch apps/inverse_homogenization/slurm/showcase_inverse2d.sbatch
SHOWCASE_MODE=prod sbatch apps/inverse_homogenization/slurm/showcase_inverse2d.sbatch
```

`openpfc_inverse_homogenize_hip` writes `history.csv` and MPI-IO `h`
bricks plus an XDMF sidecar. Size later grids from `HIP_MEM
bytes_per_cell` in the HIP driver log, not from `sacct` MaxRSS.
Do not call a grey morphology auxetic unless `nu_eff` / `nu_xy` is
negative. The default seed is rotating-squares.

### HIP endpoint material records

With `--dump-dir`, the HIP inverse driver also writes
`h_final_material.json` and `h_thresh_material.json` beside the exact accepted
`h_final.bin` and its strict `h>0.5` thresholded field. Each record identifies
the accepted step, termination reason, grid and spacing. `MAX_STEPS` remains
nonconvergence, even when the endpoint elasticity solves succeed.

The records preserve all 36 raw unpenalized stiffness entries. Diagnostics use
an explicitly separate symmetric tensor, the existing `diagnose_stiffness`
implementation, engineering-Voigt order `[xx, yy, zz, yz, xz, xy]`, and
engineering shear strain. Compliance-based `nu_xy` means transverse y response
under uniaxial x stress; all six ordered axis pairs are reported. These replace
the isotropic `C12/(C11+C12)` shortcut for anisotropic material assessment.
The compliance and Poisson fields are null unless all six solves converge,
the tensor is finite, positive definite and invertible, and compliance
normalization is defined. Per-load iterations and residuals remain available
for rejected diagnostics. This flag certifies neither inverse convergence nor
physical validation of a material model.

Both tensors were already computed by the driver; reporting adds no elasticity
solve and changes no iterate, scientific target or convergence tolerance.

### Certified trajectory animation

`render_certified_trajectory.py` consumes the HIP snapshot manifest and
`h_final_material.json`. It requires a converged endpoint with successful
material diagnostics, matching grid/spacing and exact terminal snapshot bytes.
The h=0.5 isosurface uses physical spacing, periodic boundary planes and one
fixed camera; the final frame is labelled with its certified accepted step.
It does not synthesize intermediate designs or affine deformation.

```sh
python3.11 apps/inverse_homogenization/scripts/render_certified_trajectory.py \
  /path/to/fields/run_manifest.json /new/movie-directory
```

Rendering requires NumPy, PyVista/VTK with offscreen OpenGL, and ffmpeg.
`--validate-only` checks input identity without graphics dependencies. The
output `frames.json` records field/frame/movie checksums and renderer versions.
An animation illustrates the admitted trajectory; it is not a convergence proof.

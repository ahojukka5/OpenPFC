<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Magnetic reconnection diagnostics (issue #26)

Research note for the offline topology/reconnection experiment on
the merged #24 2-D incompressible visco-resistive MHD prototype.
The solver is unchanged.

## Decision

**Current sheets but no demonstrated reconnection.**

Independent diagnostics do **not** jointly support a converged
connectivity-changing reconnection event in the documented
\(\nu=\eta=0.005\), \(t\le 2.5\), 256²/512² window.

## Question

At fixed physical parameters and after timestep and spatial
convergence, does the solution exhibit a reproducible magnetic
connectivity change with a rate that topology, electric-field, and
flux diagnostics agree on?

## Model and signs (unchanged from #24)

\[
\mathbf B=(\partial_y a,-\partial_x a),\qquad j=-\nabla^2 a,
\]
\[
E_z=-\partial_t a,\qquad
\partial_t a+\mathbf u\cdot\nabla a=-\eta j.
\]
At a true X-point \(\nabla a=0\), so \(E_z=\eta j\) if \(u=0\) or
\(u\cdot\nabla a=0\) there. That identity is a local Ohm check, not
a reconnection proof.

## Tracker

`scripts/mhd_topology.py` locates \(\nabla a=0\) by spectral
derivatives plus Newton refinement in the cell, classifies from the
Hessian, tracks periodically, and pairs X/O by Morse
eigendirection walks. Hessian conditioning is reported
(`eig_ratio`, `hess_cond`).

Synthetic tests (`scripts/test_mhd_topology.py`):

* stationary \(\sin x\sin y\): 4 X + 2 O\(_\max\) + 2 O\(_\min\) at
  the exact points, sub-cell (not node-stuck);
* rigid translation: 8 tracks, no birth/death, X follows
  \((vt,wt)\);
* OT flux: 8 critical points;
* fold creation/annihilation: count changes, births and deaths;
* decaying eigenmode \(e^{-t}\sin x\sin y\): counts fixed, no
  birth/death (false-reconnection control).

## Baseline \(\nu=\eta=0.005\)

Source of fields: #24 runs
`/scratch/project_462001519/juaho/mhd2d-23/ot{256,512}_nu0005_peak/`,
`--cfl 0.4`, dumps every \(\Delta t\approx 0.039\).

**X-track \(E_z\) vs \(\eta j\)** (longest X track, 256², full
cadence): they agree to a few parts in \(10^3\) while the Hessian
is well-conditioned (\(t\lesssim 1.4\)). At \(t=1.178\),
\(E_z=5.25\times10^{-2}\), \(\eta j=5.26\times10^{-2}\). Cadence
stride 2 (twice the dump interval) does not change this. Stride 4
(\(\Delta t\sim 0.16\)) shifts \(E_z\) by \(\sim 2\)–\(10\%\).
Repeating 256² at `--cfl 0.2` gives the same \(E_z(t)\) to all
printed digits at \(t=1.178\). The Ohm check is not an artefact of
\(\Delta t\) or of the finest finite difference.

**Hessian degeneracy.** From \(t\approx 0.82\), X-points become
degenerate (`hess_cond` \(\sim 10^2\)–\(10^3\)) as the current
sheet thins. \(n_O=4\) is stable; \(n_X\) falls and is replaced by
degenerate nulls. Peak \(|j|\) at \(t=2.238\) sits in this
ill-conditioned regime. An X-point reconnection rate at peak
current is **not** well-posed on these grids.

**Connectivity.** After replacing a noisy Morse-\(a\) fingerprint
with \((n_X,n_O,n_{\mathrm{degen}})\), the force-free control has
constant 4 X + 4 O. OT shows count changes only when X-points
degenerate, not a clean X–O pair creation. Morse pairing jitter is
**not** accepted as connectivity change.

**256 vs 512.** Field-level common-band errors from #24 remain
small. At \(t=2.238\) both resolutions report 2 well-labelled X
and 4 O, with \(a_X\) agreeing to \(\sim 2\times10^{-4}\). That is
count agreement in a degenerate window, not a converged
reconnection rate.

**Sheet.** Operational FWHM of \(|j|\) at peak is \(0.12\)–\(0.14\)
(\(\sim 5\) cells on 256²). Local \(S\sim L_{\mathrm{sheet}}/\eta\)
with \(L_{\mathrm{sheet}}\sim\sqrt{\mathrm{area}_{|j|>j_{\max}/2}}\)
is \(O(10^2)\), not a domain-scale \(S=2\pi/\eta\sim 10^3\). Neither
is in the high-\(S\) plasmoid regime.

## Controls

* Force-free \(a=\sin x\sin y\), \(u=0\): 4 X + 4 O persist;
  \(a_X=0\) so \(E_z=\eta j=0\) at saddles; no false reconnection
  rate.
* Decaying synthetic eigenmode: no births/deaths.
* Hydro \(a=0\): no magnetic nulls to track.

## \(\eta=0.0025\)

Not used for a reconnection claim. #24: 512² `--cfl 0.4` was
timestep-limited; 256² is spatially under-resolved. A reconnection
rate there would be numerically unresolved.

## Literature (applicability)

* Orszag & Tang, JFM 90 (1979): this is their incompressible
  vortex. Applies.
* Sweet/Parker: assumes a quasi-steady, isolated current sheet
  with a well-defined inflow and length. OT is a decaying,
  time-dependent vortex; SP scalings are a comparison, not a
  prediction we have tested.
* Loureiro et al. (2007), Huang & Bhattacharjee (2010): plasmoid
  chains at \(S\gtrsim 10^4\). Our local \(S\) is \(O(10^2)\)–\(10^3\).
  The theories do not apply as onset criteria here.
* García Morillo & Alexakis, JFM 1007 (2025) R3: OT plasmoids can
  be under-resolution artefacts; well-resolved spectral OT showed
  none even at much larger \(S\). Used as a **falsification
  baseline**, not a threshold to chase.

## What would be required to flip the decision

* well-conditioned X-points through the claimed event;
* a connectivity diagnostic that is silent on force-free and
  decaying-eigenmode controls, and that agrees between 256² and
  512²;
* \(E_z\), \(\eta j\), and \(d\Delta a/dt\) agreeing on a
  dt- and N-converged interval.

That package is not available in the present window. The
scientifically honest result is robust current sheets without a
demonstrated, converged reconnection event.

## Commands

```bash
python3 examples/ns2d_vorticity/scripts/test_mhd_topology.py
python3 examples/ns2d_vorticity/scripts/analyze_mhd_reconnection.py \
  --dir /scratch/.../ot256_nu0005_peak --n 256 --eta 0.005 \
  --fine-dir /scratch/.../ot512_nu0005_peak --fine-n 512
python3 examples/ns2d_vorticity/scripts/plot_mhd_topology.py \
  --dir /scratch/.../ot256_nu0005_peak --n 256 --inc 228 \
  --json /scratch/.../ot256_nu0005_peak/reconn_s1.json \
  --out /scratch/.../ot256_nu0005_peak/topology_0228.png
```

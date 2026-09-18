<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Magnetic reconnection diagnostics (issue #26)

Research note for the offline topology/reconnection experiment on
the merged #24 2-D incompressible visco-resistive MHD prototype.
The solver is unchanged.

## Decision

**Current sheets but no demonstrated topology-changing
reconnection.**

The previous “no enclosed O” claim was a simple-graph artefact.
OT islands exist at \(t=0\) with \(\Delta a=\pm 1\). Their faces
persist until Hessian degeneracy; the only 256/512-agreed
\(\Delta a(t)\) is a slow drift at a persistent X, not a new-island
event. Peak \(|j|\) remains outside the well-conditioned window.

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

`scripts/mhd_topology.py` locates \(\nabla a=0\) by bilinear-seeded
Newton, then **evaluates \(a\), \(\nabla a\) and the Hessian at the
converged point from the DFT** (cell-centered phase). Classification
uses that Fourier Hessian (`eig_ratio`, `hess_cond`). \(j=-\nabla^2 a\)
at the point is spectral, not bilinear.

Magnetic connectivity traces the critical level \(a=a_X\) along
\(\mathbf B=(\partial_y a,-\partial_x a)\), not \(\pm\nabla a\).
Four rays come from the Hessian quadratic form
\(\lambda_1\xi_1^2+\lambda_2\xi_2^2=0\) (generally **not** the
eigenvectors). \(\Delta a=a_O-a_X\) is stored only if an
\(a=a_X\) X–X cycle winds around that O.

Morse \(\pm\nabla a\) walks remain a scalar diagnostic. On
\(a=\sin x\sin y\) they run on the diagonals to O-points; magnetic
separatrices are the axes and connect X to X. A test requires both
graphs and fails if they are swapped.

Tracking uses a **physical** gate \(\Delta r \le 2\,\Delta t\)
(VA~\(O(1)\) bound on \([0,2\pi]^2\)), not \(4\Delta x\). The same
translating topology keeps 8 tracks with no birth/death at
\(N=32,64,128\) and \(\Delta t=0.025,0.05,0.10\).

Connectivity is the X–O adjacency graph (which O-track basins each
well-conditioned X connects). \((n_X,n_O,n_{\mathrm{degen}})\) is
only a sentinel.

`halfmax_area_sqrt` is an operational area proxy, **not** a
sheet-axis length. No upstream \(B\) is measured, so no physical
local Lundquist number is reported.

Synthetic tests (`scripts/test_mhd_topology.py`):

* stationary \(\sin x\sin y\): 4 X + 2 O\(_\max\) + 2 O\(_\min\) at
  the exact points; Fourier \(|\nabla a|^2\sim10^{-32}\);
* rigid translation: 8 tracks, no birth/death, X follows
  \((vt,wt)\);
* same motion at three \(N\) and three cadences;
* OT flux: 8 critical points;
* fold creation/annihilation: count changes, births and deaths;
* decaying eigenmode \(e^{-t}\sin x\sin y\): counts fixed, no
  birth/death (false-reconnection control);
* nearest-O trap: Morse walk reaches the far connected O;
* magnetic vs Morse on \(\sin x\sin y\): axes/X–X vs diagonals/X–O,
  at \(N=32,64,128\);
* OT \(t=0\) multigraph: parallel X–X edges and \(\Delta a=\pm 1\).

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

**Hessian degeneracy is Fourier-real and 256/512-agreed.**
Minimum well-conditioned X `hess_cond` at \(t=0.785\) is 10.06063
on both grids. At \(t=0.825\) both report \(n_X=3\),
\(n_{\mathrm{degen}}=1\). By \(t=1.178\) both have \(n_X=0\),
\(n_{\mathrm{degen}}=4\). \(n_O=4\) throughout. Peak \(|j|\) at
\(t=2.238\) is in this degenerate window (\(n_X=2\), cond of the
surviving labelled X \(\approx 12.68\) on both grids; the tracked
long-lived X that still has an \(a_X(t)\) has cond \(\sim 700\)).
This is not a bilinear-interpolation artefact.

**Magnetic connectivity is a multigraph.** Distinct \(a=a_X\)
branches between the same X pair are stored as parallel edges.
Two-edge faces are the OT islands. A simple neighbour graph
missed them and falsely reported no enclosed O.

Analytic OT at \(t=0\): 4 X (\(a=\pm 0.5\)) and 4 O
(\(a=\pm 1.5\)); parallel X–X edges; each \(\mathrm{O}_{\max}\)
(resp. \(\mathrm{O}_{\min}\)) is enclosed by the \(+0.5\)
(resp. \(-0.5\)) network with \(\Delta a=\pm 1\). Tests at
\(N=64,128\) require this.

**Time series \(\nu=\eta=0.005\).** Until \(t\approx 0.82\), both
256² and 512² keep 4 X, 8 enclosures and 12 extra parallel
edges. \(\Delta a\) for the tracked \(a=-0.5\) island goes
\(-1.000\to -0.979\), identical on both grids. That is a slow
resistive drift of island flux at a *persistent* X, not a new
X–O pair or a peak-current burst. After \(t=0.82\) X-points
degenerate; \(\Delta a\) at \(t=2.24\) is not a trusted rate.
Force-free: 0 graph changes. No \(\eta\) lowering.

**256 vs 512 Ohm check.** \(E_z(t)\) and \(\eta j_X(t)\) on the
longest X track agree between resolutions to the printed digits.

**Sheet.** FWHM of \(|j|\) at peak remains \(0.12\)–\(0.14\).
`halfmax_area_sqrt` is not used as a Lundquist length.

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

A topology-changing event in the peak-current window is not
available. Persistent-X island flux now has a valid \(\Delta a\),
but it is a slow drift, not a reconnection burst, and it ends when
the X-points become degenerate.

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

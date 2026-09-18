<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Resistive reconnection scaling (issue #38)

Research note for the follow-on to closed #26 / merged #27. The
solver is unchanged. Topology and flux-budget machinery is reused,
not duplicated. Local sheet definitions were frozen before any
\(\eta\) comparison.

## Question

For the persistent-X-point reconnection established in #26, how do
the normalized reconnection rate and local current-sheet geometry
scale with resistivity at \(P_m=1\), and is the converged behaviour
compatible with Sweet–Parker scaling?

Do **not** assume a power law. A negative or unresolved result is
valid.

## Predeclared outcomes

1. Sweet–Parker-like: \(R\sim S_{\mathrm{local}}^{-1/2}\).
2. Systematically different scaling.
3. No defensible single scaling law over the accessible \(\eta\)
   range.
4. Lower-\(\eta\) regime unresolved.

This pass does **not** claim an exponent. Only \(\eta=0.01\) and
\(\eta=0.005\) are compared, and local Sweet–Parker geometry is not
yet a qualified observable on the Orszag–Tang X-point in
\(t\le 0.80\).

## Frozen local-sheet definitions

Around the tracked persistent reconnection X-point
(`scripts/mhd_sheet_geometry.py`):

* \(\hat n\): eigenvector of the most negative Hessian of \(|j|\)
  at the X-point (Fourier). Fallback: largest-eigenvalue direction
  of the \(\nabla|j|\) structure tensor in a disk of radius
  \(R_{\mathrm{ST}}=0.5\).
* \(\hat t = (-\hat n_y,\hat n_x)\).

**Thickness** \(\delta\): sub-grid FWHM of \(|j|\) along \(\hat n\)
through the X-point. Physical length, not a cell count.

**Length** \(L\): extent along \(\hat t\) where
\(|j|\ge \alpha |j|_X\) with \(\alpha=0.5\). This is not
\(\sqrt{\text{half-max area}}\).

**Upstream field** \(B_{\mathrm{up}}\): reconnecting component
\(\mathbf B\cdot\hat t\) sampled at \(n=\pm\kappa\delta\) with
\(\kappa=2\). Mean of the two absolute values. The sampling
distance is not retuned per \(\eta\).

Density is 1, so \(V_A=B_{\mathrm{up}}\).

\[
S_{\mathrm{local}} = L V_A/\eta,\qquad
R = |E_{z,X}|/(B_{\mathrm{up}} V_A) = |E_{z,X}|/B_{\mathrm{up}}^2.
\]

Search windows are frozen: \(s_n\in[-\pi/2,\pi/2]\),
\(s_t\in[-\pi,\pi]\). If the half-max set reaches a window
endpoint, \(\delta\) or \(L\) is **capped** and is not a physical
sheet width. `sheet_ok` requires uncapped \(\delta\), uncapped
\(L\), and \(B_{\mathrm{up}}>10^{-3}\).

Capped frames must not enter a scaling point.

## Synthetic qualification

`scripts/test_mhd_sheet_geometry.py` (Gaussian ridge and Harris
sheet, \(N=64,128,256\)):

* rotated sheets recover \(\hat n\) (catches tangent/normal swap);
* FWHM \(\delta\) and length \(L\) match the known Gaussian widths
  and stay distinct;
* grid-aligned Harris \(B_{\mathrm{up}}\approx B_0\) at
  \(\pm 2\delta\);
* both sides of a rotated Harris are sampled;
* a sheet wider than the search window is flagged capped, not
  treated as a physical FWHM.

The geometry code is qualified on synthetics. That is necessary
and not sufficient for Orszag–Tang.

## Proposed scalar reconnection statistic (frozen here)

Track the same #26 symmetry-related persistent island / X family.

* Primary un-normalized rate: \(E_{z,X}(t)\) and island flux
  \(F=a_O-a_X\), already N-converged in #26.
* Primary *normalized* \(R\): time average of \(R(t)\) on
  \(t\in[0.10,0.70]\) **restricted to `sheet_ok` frames**, and only
  if that interval is common (every compared run is
  well-conditioned and `sheet_ok` throughout).
* Always show the full \(R(t)\) with saturation flags.
* Peak \(R\) is secondary and is not used to pick \(\eta\) points.

**Finding:** a common thin-sheet interval cannot be defined in the
accepted \(t\le 0.80\) window. Most of \([0.10,0.70]\) is FWHM-
capped at both \(\eta\). The few `sheet_ok` frames sit at
\(t\gtrsim 0.59\), where the X Hessian condition number is already
rising toward the #26 degeneracy near \(t\approx 0.82\). Those
frames are not a well-conditioned common interval. Do not extract
a single \(R(S_{\mathrm{local}})\) from them.

Event-based fallback, if a later campaign needs one number per
\(\eta\): match the same tracked X/O pair on a predeclared
`sheet_ok` run of at least three consecutive dumps, after 256² vs
512² geometry agreement, and still show full \(R(t)\). That rule
is recorded before \(\eta=0.0025\). It is **not** applied in this
pass.

## Re-analysis of \(\eta=\nu=0.005\) (existing #26 dumps)

No new 256²/512² runs. Cadence \(\Delta t_{\mathrm{dump}}=\pi/80\),
\(t\le 0.80\).

Flux / Ohm (necessary, not sufficient for #38):

|  N  | rel \(\eta(j)\) budget | mean \(E_{z,X}\) | \(F(t_{\mathrm{last}})\) |
| --: | ---------------------: | ---------------: | -----------------------: |
| 256 |                5.65e-4 |          0.01169 |                 -0.97862 |
| 512 |                5.65e-4 |          0.01169 |                 -0.97862 |

Ohm at the X-point remains \(E_{z,X}\approx\eta j_X\) on the
tracked pair. 256² vs 512² flux diagnostics agree.

Local geometry in \([0.10,0.70]\):

|  N  | capped / 15 | `sheet_ok` | mean \(\delta\) (all) | mean \(\delta\) (`ok`) |
| --: | ----------: | ---------: | --------------------: | ---------------------: |
| 256 |          12 |          3 |                  2.83 |                   1.85 |
| 512 |          12 |          3 |                  2.79 |                   1.86 |

FWHM is at the search-window cap \(\delta=\pi\) for 13 of 21 dumps
(through \(t\approx 0.47\)). The current around the X-point is
**not a thin Sweet–Parker sheet** during most of the accepted
reconnection window.

Late thinning (not a scaling interval):

| \(t\)  | \(\delta\) 256 | \(\delta\) 512 | \(L\) 256 | \(L\) 512 | `sheet_ok` |
| -----: | -------------: | -------------: | --------: | --------: | ---------: |
| 0.589  |          2.038 |          1.966 |     2.218 |     1.985 | yes / yes  |
| 0.628  |          1.877 |          1.822 |     2.194 |     2.227 | yes / yes  |
| 0.668  |          1.641 |          1.802 |     2.112 |     1.556 | yes / yes  |
| 0.707  |          1.275 |          1.771 |     2.037 |     6.283 | yes / L-cap |
| 0.746  |          1.501 |          1.117 |     0.807 |     2.054 | yes / yes  |
| 0.785  |          0.242 |          0.242 |     1.902 |     1.829 | yes / yes  |

\(\delta\) at \(t=0.785\) agrees at 256² and 512², but that dump
is at Hessian cond \(\approx 18\), next to the #26 stop. \(L\) and
\(B_{\mathrm{up}}\) do **not** agree across \(N\) on the late
`sheet_ok` frames. Normalized \(R\) on those three interior
`sheet_ok` dumps is \(0.12\pm 0.05\) (256) vs \(0.23\pm 0.22\)
(512). Local SP geometry is **not** N-converged in this window.

Naive \(R\) averaged over all of \([0.10,0.70]\) (including capped
frames) is \(0.071\) vs \(3.09\) and is discarded.

## New physical point \(\eta=\nu=0.01\)

Same Orszag–Tang initial condition, same frozen geometry constants,
same island family. New runs: 128² CFL 0.4; 256² CFL 0.4 and CFL
0.2. Timestep refinement at 256² leaves flux and geometry traces
indistinguishable at dump cadence.

|  N  | CFL | rel \(\eta(j)\) | mean \(E_{z,X}\) | \(F(t_{\mathrm{last}})\) |
| --: | --: | --------------: | ---------------: | -----------------------: |
| 128 | 0.4 |         4.57e-4 |          0.02236 |                 -0.95823 |
| 256 | 0.4 |         4.55e-4 |          0.02236 |                 -0.95823 |
| 256 | 0.2 |         4.55e-4 |          0.02236 |                 -0.95823 |

Mean \(E_{z,X}\) is about twice the \(\eta=0.005\) value
(\(0.02236/0.01169\approx 1.91\)), consistent with
\(E_{z,X}\approx\eta j_X\) at similar \(j_X\). Fractional X-point
contribution to \(dF/dt\) remains \(\approx 0.40\).

Local geometry is again FWHM-capped for most of \([0.10,0.70]\)
(10–11 of 15 dumps). `sheet_ok` counts: 2 (128²) and 4 (256²).
No common thin-sheet interval.

## Two physical points, no exponent

Un-normalized, N-converged rate diagnostics exist at two
resistivities:

| \(\eta=\nu\) | mean \(E_{z,X}\) on \(t\le 0.80\) | \(\Delta F\) |
| -----------: | --------------------------------: | -----------: |
|        0.010 |                           0.02236 |       0.0418 |
|        0.005 |                           0.01169 |       0.0214 |

That is **not** a Sweet–Parker test. \(S_{\mathrm{local}}\) and
normalized \(R\) are not admitted while \(\delta\) is search-window
capped and \(L\), \(B_{\mathrm{up}}\) are not N-converged.

Working classification against the predeclared outcomes: **(3)** no
defensible single local-sheet scaling law on the accessible
\(\eta\) range in \(t\le 0.80\), with **(4)** still open if a later
well-conditioned thin sheet appears outside this window. Outcome
**(1)** is not supported by this pass.

## Gate: \(\eta=0.0025\) at 512²/1024²

**Not justified for local Sweet–Parker geometry.**

The geometry diagnostic works on analytic sheets. On the #26
Orszag–Tang X-point it reports the absence of a thin sheet in the
accepted reconnection window, and the quantities that would enter
\(R(S_{\mathrm{local}})\) do not converge between 256² and 512².
A 1024² run at \(\eta=0.0025\) cannot repair that: it would spend
the expensive ladder on a diagnostic that is already saturated at
higher \(\eta\).

Un-normalized \(E_{z,X}\) and \(F\) *are* N-converged at both
\(\eta\) already in hand. They do not by themselves authorize the
1024² gate.

Do not hunt plasmoids. Do not lower \(\eta\) below 0.0025. Do not
extend \(t\) past 0.80 in this campaign (Hessian degeneracy).

## Commands

```bash
python3 examples/ns2d_vorticity/scripts/test_mhd_sheet_geometry.py
python3 examples/ns2d_vorticity/scripts/sheet_scaling.py \
  --dir /scratch/.../ot256_nu0005_peak --n 256 --eta 0.005 --t-max 0.80
python3 examples/ns2d_vorticity/scripts/sheet_scaling.py \
  --dir /scratch/.../ot512_nu0005_peak --n 512 --eta 0.005 --t-max 0.80
python3 examples/ns2d_vorticity/scripts/sheet_scaling.py \
  --dir /scratch/.../ot128_nu001_cfl04 --n 128 --eta 0.01 --t-max 0.80
python3 examples/ns2d_vorticity/scripts/sheet_scaling.py \
  --dir /scratch/.../ot256_nu001_cfl04 --n 256 --eta 0.01 --t-max 0.80
```

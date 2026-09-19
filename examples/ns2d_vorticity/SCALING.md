<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Resistive reconnection scaling (issue #38)

Research note for the follow-on to closed #26 / merged #27. The
solver is unchanged. Topology and flux-budget machinery is reused,
not duplicated. Local sheet constants were frozen before any
\(\eta\) comparison and were **not** retuned after the connected-
component correction.

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
\(\eta=0.005\) are compared.

## Frozen local-sheet definitions

Around the tracked persistent reconnection X-point
(`scripts/mhd_sheet_geometry.py`). Constants
\(\alpha=0.5\), \(\kappa=2\), search windows, and
\(t\in[0.10,0.70]\) are unchanged.

Orientation uses the **smooth signed ridge**
\(q=\mathrm{sign}(j_X)\,j\), never Fourier derivatives of \(|j|\).
\(\hat n\) is the most negative Hessian eigenvector of \(q\) at the
X-point. Fallback: largest-eigenvalue direction of the \(\nabla q\)
structure tensor in a disk of radius \(R_{\mathrm{ST}}=0.5\). If
\(|j_X|<10^{-12}\) or both constructions are degenerate, fail
closed. \(\hat t=(-\hat n_y,\hat n_x)\).

**Thickness** \(\delta\): sub-grid width of the **connected**
\(|j|\ge\tfrac12|j_X|\) component containing \(s=0\) along
\(\hat n\). Remote superlevel lobes are ignored. Not a cell count.

**Length** \(L\): connected \(|j|\ge\alpha|j_X|\) component
containing \(s=0\) along \(\hat t\). Not
\(\sqrt{\text{half-max area}}\).

**Upstream field** \(B_{\mathrm{up}}\): reconnecting component
\(\mathbf B\cdot\hat t\) at \(n=\pm\kappa\delta\), with
\(\mathbf B=(a_y,-a_x)\) from Fourier derivatives of \(a\) (not
bilinear interpolation). Mean of the two absolute values. Sampling
distance is not retuned per \(\eta\).

Density is 1, so \(V_A=B_{\mathrm{up}}\).

\[
S_{\mathrm{local}} = L V_A/\eta,\qquad
R = |E_{z,X}|/(B_{\mathrm{up}} V_A) = |E_{z,X}|/B_{\mathrm{up}}^2.
\]

Search windows: \(s_n\in[-\pi/2,\pi/2]\), \(s_t\in[-\pi,\pi]\).
`capped` is true only if **that connected component** reaches a
window endpoint. `sheet_ok` requires uncapped \(\delta\), uncapped
\(L\), and \(B_{\mathrm{up}}>10^{-3}\).

## Synthetic qualification

`scripts/test_mhd_sheet_geometry.py` (\(N=64,128,256\)):

* rotated Gaussian ridges recover \(\hat n\), \(\delta\), and \(L\);
* a separated remote ridge above the same threshold is ignored for
  both \(\delta\) and \(L\); a remote lobe at a window endpoint does
  **not** mark the local sheet capped;
* \(j\to -j\) leaves \(\hat n\), \(\hat t\) (up to eigenvector
  sign), \(\delta\), and \(L\) unchanged on the Harris orientation
  path;
* grid-aligned Harris \(B_{\mathrm{up}}\approx B_0\) at
  \(\pm 2\delta\);
* a sheet wider than the search window is flagged capped.

## Preregistered windows

Track the same #26 island / X family. Full \(R(t)\) is always shown.
Peak \(R\) is secondary.

**Stage 1 (failed, kept as history).**
\(t\in[0.10,0.70]\). Not a common `sheet_ok` interval: the first
dumps are genuinely \(L\)-capped at \(2\pi\). This window is **not**
moved and is **not** the scaling statistic.

**Stage 2 (frozen before \(\eta=0.0025\)).**

\[
T_{\mathrm{SCALE,LO}}=0.314,\qquad T_{\mathrm{SCALE,HI}}=0.70.
\]

Reasons, recorded before any low-\(\eta\) experiment:

* at both \(\eta=0.01\) and \(\eta=0.005\) the same persistent-X
  family becomes finite-length `sheet_ok` at the first common dump
  \(t=8\pi/80\approx 0.314\);
* geometry is spatially and timestep converged through \(t=0.70\);
* the cut at \(0.70\) stays off the later Hessian-degeneracy region
  (cond \(\approx 13\) already at \(t=0.707\)).

Do **not** move these endpoints after seeing \(\eta=0.0025\).

Primary normalized \(R\): time average of
\(R=|E_{z,X}|/B_{\mathrm{up}}^2\) on \([0.314,0.70]\), admitted only
if every dump in that window is `sheet_ok` and well-conditioned.

## Corrected re-analysis of \(\eta=\nu=0.005\)

Existing #26 dumps only. Cadence \(\Delta t_{\mathrm{dump}}=\pi/80\),
\(t\le 0.80\).

Flux / Ohm is unchanged and N-converged:

|  N  | rel \(\eta(j)\) budget | mean \(E_{z,X}\) | \(F(t_{\mathrm{last}})\) |
| --: | ---------------------: | ---------------: | -----------------------: |
| 256 |                5.65e-4 |          0.01169 |                 -0.97862 |
| 512 |                5.65e-4 |          0.01169 |                 -0.97862 |

Local geometry, after the connected-component / signed-\(q\) /
spectral-\(B_{\mathrm{up}}\) correction:

| \(t\)  | \(\delta\) | \(L\) | \(B_{\mathrm{up}}\) | \(R\) | `sheet_ok` |
| -----: | ---------: | ----: | ------------------: | ----: | ---------: |
| 0.000  |      0.723 | 6.283 |               0.248 | 0.081 | L-cap      |
| 0.118  |      0.705 | 6.283 |               0.246 | 0.089 | L-cap      |
| 0.275  |      0.626 | 6.283 |               0.219 | 0.149 | L-cap      |
| 0.314  |      0.598 | 6.076 |               0.209 | 0.180 | yes        |
| 0.393  |      0.534 | 5.193 |               0.192 | 0.261 | yes        |
| 0.471  |      0.466 | 2.781 |               0.184 | 0.351 | yes        |
| 0.550  |      0.400 | 2.379 |               0.190 | 0.412 | yes        |
| 0.628  |      0.340 | 2.229 |               0.207 | 0.431 | yes        |
| 0.707  |      0.287 | 2.163 |               0.230 | 0.433 | yes        |
| 0.785  |      0.241 | 2.140 |               0.250 | 0.426 | yes        |

256² and 512² agree to printed precision on every listed geometry
quantity (relative difference \(\lesssim 10^{-5}\) on
\(\delta,L,B_{\mathrm{up}},R\)). Timestep is the #26 accepted CFL.

\(\delta\) is **never** search-window capped. Early \(L=2\pi\) is a
genuine tangent-window cap: the connected \(|j|\ge\tfrac12|j_X|\)
component along \(\hat t\) fills \(s_t\in[-\pi,\pi]\). That is a
broad current structure, not a finite-length Sweet–Parker sheet.

From the first `sheet_ok` dump \(t=0.314\) through \(t=0.785\),
\(\delta\) thins from 0.60 to 0.24, \(L/\delta\sim 4\)–\(10\), and
the same dumps are `sheet_ok` at 256² and 512².

The frozen interval \([0.10,0.70]\) has 5 L-capped dumps and 10
`sheet_ok` dumps. `common_thin_sheet_interval` is **false**. The
primary normalized \(R\) is therefore **not** admitted as a single
scaling point.

## New physical point \(\eta=\nu=0.01\)

Same IC, same frozen constants, same island family. 128² CFL 0.4;
256² CFL 0.4 and 0.2.

|  N  | CFL | rel \(\eta(j)\) | mean \(E_{z,X}\) | \(F(t_{\mathrm{last}})\) |
| --: | --: | --------------: | ---------------: | -----------------------: |
| 128 | 0.4 |         4.57e-4 |          0.02236 |                 -0.95823 |
| 256 | 0.4 |         4.55e-4 |          0.02236 |                 -0.95823 |
| 256 | 0.2 |         4.55e-4 |          0.02236 |                 -0.95823 |

Local geometry is again L-capped for \(t\le 0.275\) and `sheet_ok`
from \(t=0.314\). 128², 256², and CFL 0.2 agree to printed
precision. Representative `sheet_ok` dumps (256², CFL 0.4):

| \(t\)  | \(\delta\) | \(L\) | \(B_{\mathrm{up}}\) | \(R\) | \(S_{\mathrm{local}}\) |
| -----: | ---------: | ----: | ------------------: | ----: | ---------------------: |
| 0.314  |      0.598 | 6.098 |               0.209 | 0.355 |                    128 |
| 0.471  |      0.469 | 2.825 |               0.176 | 0.748 |                     50 |
| 0.628  |      0.347 | 2.248 |               0.180 | 1.081 |                     40 |
| 0.785  |      0.253 | 2.167 |               0.189 | 1.360 |                     41 |

The same \(t=0.314\) onset of `sheet_ok` occurs at both
resistivities.

## Two-point baseline on the frozen stage-2 window

All numbers below are time averages on exactly
\([0.314,0.70]\) (10 dumps, \(t=0.3142\) to \(0.6676\)). Every dump
is `sheet_ok`. No power-law exponent is fitted from two points.

\(\eta=\nu=0.005\) (256² vs 512², #26 dumps):

| quantity | mean | std | min | max | 256 vs 512 rel |
| --- | ---: | ---: | ---: | ---: | ---: |
| \(E_{z,X}\) | 0.01327 | 0.00409 | 0.00786 | 0.02062 | \(9\times10^{-9}\) |
| \(\eta j_X\) | 0.01324 | 0.00408 | 0.00784 | 0.02059 | \(1\times10^{-8}\) |
| \(dF/dt\) | 0.02871 | 0.00419 | 0.02311 | 0.03619 | \(6\times10^{-8}\) |
| \(\delta\) | 0.4517 | 0.0924 | 0.3122 | 0.5976 | \(6\times10^{-8}\) |
| \(L\) | 3.490 | 1.473 | 2.189 | 6.076 | \(2\times10^{-7}\) |
| \(L/\delta\) | 7.458 | 1.696 | 5.831 | 10.17 | \(2\times10^{-7}\) |
| \(B_{\mathrm{up}}\) | 0.1971 | 0.0109 | 0.1844 | 0.2185 | \(1\times10^{-6}\) |
| \(S_{\mathrm{local}}\) | 137.9 | 60.6 | 90.4 | 254.2 | \(1\times10^{-6}\) |
| \(R\) | 0.3403 | 0.0893 | 0.1797 | 0.4321 | \(2\times10^{-6}\) |

\(\eta=\nu=0.01\) (128² CFL 0.4 vs 256² CFL 0.4; 256² CFL 0.2
matches 256² CFL 0.4 to \(\sim10^{-6}\)):

| quantity | mean | std | min | max | 128 vs 256 rel |
| --- | ---: | ---: | ---: | ---: | ---: |
| \(E_{z,X}\) | 0.02546 | 0.00740 | 0.01549 | 0.03853 | \(3\times10^{-7}\) |
| \(\eta j_X\) | 0.02542 | 0.00740 | 0.01545 | 0.03849 | \(3\times10^{-7}\) |
| \(dF/dt\) | 0.05614 | 0.00756 | 0.04585 | 0.06939 | \(7\times10^{-7}\) |
| \(\delta\) | 0.4554 | 0.0898 | 0.3206 | 0.5977 | \(1\times10^{-6}\) |
| \(L\) | 3.549 | 1.482 | 2.208 | 6.098 | \(3\times10^{-6}\) |
| \(L/\delta\) | 7.516 | 1.726 | 5.842 | 10.20 | \(4\times10^{-6}\) |
| \(B_{\mathrm{up}}\) | 0.1840 | 0.0111 | 0.1735 | 0.2090 | \(1\times10^{-5}\) |
| \(S_{\mathrm{local}}\) | 66.75 | 32.17 | 40.41 | 127.5 | \(2\times10^{-5}\) |
| \(R\) | 0.7736 | 0.2633 | 0.3545 | 1.138 | \(3\times10^{-5}\) |

\(L\) and \(B_{\mathrm{up}}\) are **not** constant in time, and they
are not the same at the two resistivities. \(\delta\) means are
almost equal (0.452 vs 0.455); \(L\) is slightly larger at
\(\eta=0.01\). A later three-point slope in \(R(S_{\mathrm{local}})\)
must be read against this evolving OT geometry, not as a test of a
steady Sweet–Parker sheet with fixed \(L\) and \(B_{\mathrm{up}}\).

The earlier “no thin sheet / FWHM at \(\pi\)” reading was a
**measurement artefact** and is withdrawn.

## Gate: \(\eta=0.0025\)

The stage-2 window is now frozen. Two admitted \(\eta\) points exist.
\(\eta=\nu=0.0025\) is the next physical point.

Admission plan (do not move \(T_{\mathrm{SCALE}}\)):

1. Exploratory 512² only, `--cfl 0.2` (the #24 stable smaller-CFL
   regime; do not repeat the timestep-limited `--cfl 0.4` crash as
   a scaling point). Same OT initial condition and X/O family.
2. 512² is **not** admitted alone. Require: the same persistent X/O
   through \([0.314,0.70]\); well-conditioned X and O;
   \(E_{z,X}\approx\eta j_X\); flux budget closed; `sheet_ok`
   throughout the frozen window; \(\delta\) not grid-limited;
   \(L\), \(B_{\mathrm{up}}\), \(R\) not obviously unresolved.
3. Launch 1024² only if 512² defines a spatial-convergence question.
   Then require convergence of \(E_{z,X}\), \(F\), \(\delta\), \(L\),
   \(B_{\mathrm{up}}\), \(S_{\mathrm{local}}\), and \(R\). Flux-budget
   agreement alone is not enough.
4. Fit \(\log R\) vs \(\log S_{\mathrm{local}}\) only after three
   admitted \(\eta\) values. Also test \(\delta/L\) vs \(S\) and
   whether \(L\) and \(B_{\mathrm{up}}\) change with \(\eta\).

Do not hunt plasmoids. Do not lower \(\eta\) below 0.0025. Do not
extend \(t\) past 0.80.

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

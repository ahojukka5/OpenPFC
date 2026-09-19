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

## Proposed scalar reconnection statistic (frozen)

Track the same #26 island / X family.

* Primary un-normalized rate: \(E_{z,X}(t)\) and \(F=a_O-a_X\).
* Primary *normalized* \(R\): time average of \(R(t)\) on
  \(t\in[0.10,0.70]\) restricted to `sheet_ok` frames, **and only
  if that interval is common** (every compared run is
  well-conditioned and `sheet_ok` throughout).
* Always show the full \(R(t)\) with saturation flags.
* Peak \(R\) is secondary.

The window \([0.10,0.70]\) is **not** retuned after the connected-
component correction.

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

## Two \(\eta\) values, no exponent

Un-normalized N-converged rates:

| \(\eta=\nu\) | mean \(E_{z,X}\) on \(t\le 0.80\) | \(\Delta F\) |
| -----------: | --------------------------------: | -----------: |
|        0.010 |                           0.02236 |       0.0418 |
|        0.005 |                           0.01169 |       0.0214 |

Local geometry is now a qualified, N-converged observable after
\(t=0.314\). The frozen \([0.10,0.70]\) window is still not a
common `sheet_ok` interval, so no single \(R(S_{\mathrm{local}})\)
is extracted. Full \(R(t)\) is the comparison. No power law is
fitted.

The earlier “no thin sheet / FWHM at \(\pi\)” reading was a
**measurement artefact** (union of disconnected superlevel lobes,
and Fourier derivatives of \(|j|\)). It is withdrawn.

Working classification: **(3)** no defensible single scaling law
from the frozen common-interval statistic, with **(4)** still open
at lower \(\eta\). Outcome **(1)** is not supported. This is not
the same as “no local sheet exists”.

## Gate: \(\eta=0.0025\) at 512²/1024²

**Not launched in this pass.**

The geometry diagnostic is now qualified on Orszag–Tang: \(\delta\),
\(L\), \(B_{\mathrm{up}}\), and \(R\) agree between 256² and 512²
and between CFL 0.4 and 0.2. A local connected current sheet is
present and thinning for \(t\ge 0.314\).

That does **not** by itself authorize the expensive ladder:

* the frozen \([0.10,0.70]\) interval is not fully `sheet_ok`;
* a later common dump set must be **predeclared** before looking
  at \(\eta=0.0025\), not chosen after seeing two \(\eta\) values;
* two resistivities are not a scaling law;
* the X Hessian condition number is already 3 at \(t=0.314\) and
  18 at \(t=0.785\), next to the #26 stop.

A candidate common window for a later pass, recorded here so it is
not invented after a low-\(\eta\) run: every dump with
\(0.314\le t\le 0.80\) is `sheet_ok` at both \(\eta=0.01\) and
\(\eta=0.005\), 128²–512², and both CFLs. That window is **not**
used as the primary statistic in this PR.

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

<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Magnetic-island coalescence (issue #113)

Stage-0 qualification of a doubly-periodic coalescence control for
the accepted Orszag–Tang persistent-X result. Parent research work
order: [ahojukka5/research#543](https://github.com/ahojukka5/research/issues/543).

This note freezes the benchmark, the perturbation, and — after the
\(\eta=0.01\) run — the cross-\(\eta\) event statistic. It does
**not** run \(\eta=0.005\), fit a scaling exponent, add noise, hunt
plasmoids, or add Hall/kinetic physics.

The solver is unchanged. Topology, flux-budget, and local-sheet
definitions are the accepted #26/#38 machinery.

## Literature-matched field

Ng & Ragunathan (arXiv:1106.0521) use, on the unit square,

\[
A(x,y,0)=\bar A\sin(2\pi x)\sin(2\pi y),\qquad \bar A=0.4.
\]

On OpenPFC's \([0,2\pi]^2\) domain the coordinate-equivalent flux is

\[
a(x,y,0)=0.4\sin x\sin y.
\]

Then \(j=-\nabla^2 a=2a\), so \(\mathbf B\cdot\nabla j=0\). With
\(u=0\) the magnetic field is an exact static state of the current
vorticity formulation; it decays only as the \(k^2=2\) resistive
eigenmode \(a(t)=a(0)e^{-2\eta t}\).

Critical points of \(a\):

| kind | locations | \(a\) |
|------|-----------|------:|
| X | \((0,0)\), \((\pi,0)\), \((0,\pi)\), \((\pi,\pi)\) | \(0\) |
| \(\mathrm{O}_{\max}\) | \((\pi/2,\pi/2)\), \((3\pi/2,3\pi/2)\) | \(+0.4\) |
| \(\mathrm{O}_{\min}\) | \((\pi/2,3\pi/2)\), \((3\pi/2,\pi/2)\) | \(-0.4\) |

Like-signed islands sit on the diagonals. Magnetic separatrices of
the \(a=0\) network are the coordinate axes, not the Morse
diagonals (already tested on \(a=\sin x\sin y\)).

Tracked **primary family**: the X nearest \((\pi,\pi)\) and the
enclosed \(\mathrm{O}_{\max}\) nearest \((\pi/2,\pi/2)\). At \(t=0\),
\(F=a_O-a_X=0.4\). The negative pair that coalesces toward
\((0,0)\) is the symmetry image and is not a second experiment.

## Frozen perturbation

Ng & Ragunathan state that a small flow triggers the coalescence
instability but do not give its form in the short paper. Random
noise is forbidden in the accepted baseline.

The smallest deterministic, divergence-free, low-wavenumber
streamfunction that displaces like-signed islands toward one
another is the \(k=1\) diagonal strain

\[
\phi=\varepsilon(\cos x-\cos y),\qquad
\varepsilon=0.01\ \text{(frozen)}.
\]

Then

\[
\mathbf u=(\varepsilon\sin y,\ \varepsilon\sin x),\qquad
\omega=\varepsilon(\cos x-\cos y).
\]

Direction check, \(\varepsilon>0\):

- \(\mathrm{O}_{\max}(\pi/2,\pi/2)\): \(\mathbf u=\varepsilon(1,1)\)
  toward \((\pi,\pi)\).
- \(\mathrm{O}_{\max}(3\pi/2,3\pi/2)\): \(\mathbf u=\varepsilon(-1,-1)\)
  toward \((\pi,\pi)\).
- \(\mathrm{O}_{\min}(\pi/2,3\pi/2)\): \(\mathbf u=\varepsilon(-1,1)\)
  toward \((0,0)\).
- \(\mathrm{O}_{\min}(3\pi/2,\pi/2)\): \(\mathbf u=\varepsilon(1,-1)\)
  toward \((0,0)\).
- All four X-points are stagnation points.

This is not \(\phi\propto a\) (which stagnates at the O-points) and
not a single Fourier mode (which only translates the lattice). The
shape and amplitude are frozen during Stage 0 and are **not**
retuned with \(\eta\). Constants live in
`include/ns2d/mhd_cases.hpp` as `coalescence_abar` and
`coalescence_eps`.

Stage-0 parameters: \(\nu=\eta=0.01\), \(P_m=1\).

## Diagnostics (unchanged definitions)

Reuse `scripts/mhd_topology.py`, `island_flux_budget.py`,
`mhd_sheet_geometry.py`, and `sheet_scaling.py`:

- Fourier sub-grid X/O tracker;
- magnetic separatrix / enclosure;
- \(F=a_O-a_X\);
- \(E_{z,X}=-\mathrm{d}a_X/\mathrm{d}t\);
- \(\eta j_X\) Ohm check;
- X/O flux budget;
- signed-current-ridge sheet frame;
- connected-component \(\delta\) and \(L\);
- spectral \(B_{\mathrm{up}}\);
- \(S_{\mathrm{local}}=L B_{\mathrm{up}}/\eta\);
- \(R=|E_{z,X}|/B_{\mathrm{up}}^2\);
- Hessian-conditioning and Elsasser-CFL fail-closed gates.

Do not use \(\max|j|\) as a reconnection proxy. Pass
`--family coalescence` so the tracker seeds \((\pi,\pi)\) rather
than the OT \(a\approx-0.5\) island.

## Cross-\(\eta\) event statistic (frozen from Stage 0 only)

Do **not** copy the OT window \([0.314,0.70]\).

Define island-flux progress from the tracked family

\[
p(t)=\mathrm{sign}\frac{F(0)-F(t)}{|F(0)|},
\]

with the sign taken from the transfer direction at the first
`sheet_ok` dump (decrease of \(F\) is the expected coalescence
direction for the positive island).

The comparison interval is the **first crossing** of a frozen
progress bracket

\[
p\in[\alpha_{\mathrm{lo}},\alpha_{\mathrm{hi}}]
\]

that lies after a finite current sheet exists (`sheet_ok`) and
before the first rebound. Rebound is a consecutive sign reversal of
\(\mathrm{d}F/\mathrm{d}t\) against the transfer direction
(`REBOUND_STREAK=2` in `scripts/coalescence_event.py`).

From the \(\eta=0.01\) series only:

- first `sheet_ok` at \(t=9.032\), \(p\approx 0.189\);
- no rebound through \(p\approx 0.92\) (`dF/dt` stays negative);
- Hessian \(\mathrm{cond}(X)\) stays below 10 and \(B_{\mathrm{up}}\)
  stays finite for \(p\le 0.60\); both degrade for \(p\gtrsim 0.70\).

The frozen bracket is therefore

\[
\alpha_{\mathrm{lo}}=0.30,\qquad \alpha_{\mathrm{hi}}=0.60.
\]

It starts after a finite sheet exists, covers a substantial first
transfer of the initial island flux, and stops before late merger
(ill-conditioned X, collapsing \(B_{\mathrm{up}}\), \(R\) blow-up).
It is **not** an Orszag–Tang clock window. Constants live in
`scripts/coalescence_event.py` as `PROGRESS_LO` / `PROGRESS_HI`.
Do not retune them after seeing \(\eta=0.005\).

On that frozen event interval later runs report:

| quantity | definition |
|----------|------------|
| mean \(R\) | time mean of \(\|E_{z,X}\|/B_{\mathrm{up}}^2\) |
| mean \(S_{\mathrm{local}}\) | time mean of \(L B_{\mathrm{up}}/\eta\) |
| mean \(\delta/L\) | time mean of connected-component aspect |
| mean \(j_X\) | time mean of spectral \(j\) at the tracked X |
| mean \(L\) | time mean of connected-component length |
| mean \(B_{\mathrm{up}}\) | time mean of spectral upstream field |
| flux pile-up | \(B_{\mathrm{up}}^{\max}/B_{\mathrm{up}}(t_{\mathrm{first}})\) on the event window |
| sloshing sentinel | first rebound time, or none |

Peak reconnection rate is secondary only. A two-point log-log slope
is out of scope for this issue's Stage 0.

### Frozen numbers (Stage 0)

| symbol | value | status |
|--------|------:|--------|
| \(\varepsilon\) | \(0.01\) | frozen before any run |
| \(\bar A\) | \(0.4\) | literature match |
| \(\alpha_{\mathrm{lo}}\) | \(0.30\) | frozen from \(\eta=0.01\) |
| \(\alpha_{\mathrm{hi}}\) | \(0.60\) | frozen from \(\eta=0.01\) |
| event clock window | n/a | not used as the comparator |

## Qualification ladder

Start at \(128^2\) and \(256^2\). Timestep refinement at \(256^2\)
(`--cfl 0.4` vs `0.2`). Increase \(N\) only if a concrete
convergence failure requires it.

Demonstrate:

1. analytic \(t=0\) topology;
2. intended islands approach and coalesce;
3. one unambiguous primary X/O family;
4. X/O topology survives refinement;
5. \(E_{z,X}\approx\eta j_X\);
6. island-flux budget closes;
7. \(\delta\), \(L\), and \(B_{\mathrm{up}}\) finite and
   `sheet_ok`;
8. the same event spatially and timestep converged;
9. representative \(j+a\) contours + X/O frames preserved.

## Commands

CPU HeFFTe / LUMI-C. Large I/O under scratch.

```bash
# 128^2 scout, eta=nu=0.01, nominal CFL 0.4
mhd2d --case island_coalescence --N 128 --cfl 0.4 \
  --nu 0.01 --eta 0.01 --steps 2038 --diag 20 --dump 20 \
  --outdir /scratch/project_462001519/juaho/mhd2d-113/ic128_nu001_cfl04

# 256^2, CFL 0.4
mhd2d --case island_coalescence --N 256 --cfl 0.4 \
  --nu 0.01 --eta 0.01 --steps 4074 --diag 40 --dump 40 \
  --outdir /scratch/project_462001519/juaho/mhd2d-113/ic256_nu001_cfl04

# 256^2 timestep refinement, CFL 0.2
mhd2d --case island_coalescence --N 256 --cfl 0.2 \
  --nu 0.01 --eta 0.01 --steps 8148 --diag 80 --dump 80 \
  --outdir /scratch/project_462001519/juaho/mhd2d-113/ic256_nu001_cfl02
```

`--steps` above reaches \(t\approx 40\) at each nominal CFL. Shorten
only after the scout shows that the first coalescence and any first
rebound both sit well below that time.

```bash
python3 examples/ns2d_vorticity/scripts/island_flux_budget.py \
  --dir .../ic256_nu001_cfl04 --n 256 --eta 0.01 --t-max 40 \
  --family coalescence --json-out .../flux.json

python3 examples/ns2d_vorticity/scripts/sheet_scaling.py \
  --dir .../ic256_nu001_cfl04 --n 256 --eta 0.01 --t-max 40 \
  --family coalescence --json-out .../sheet.json

python3 examples/ns2d_vorticity/scripts/coalescence_event.py \
  --json .../sheet.json --json-out .../event.json

python3 examples/ns2d_vorticity/scripts/plot_mhd_topology.py \
  --dir .../ic256_nu001_cfl04 --n 256 --inc INC \
  --out .../frame.png
```

Slurm: `slurm/coalescence_stage0.sbatch` and
`slurm/coalescence_analyze.sbatch`.

## Stage-0 evidence

Parent SHA at run time: `e98d8a87`. Jobs used the dirty
`feat/113-island-coalescence` worktree that this PR records.
Binary:
`/flash/project_462001519/juaho/build/openpfc-113-cpu/examples/ns2d_vorticity/mhd2d`.
Raw dumps:
`/scratch/project_462001519/juaho/mhd2d-113/`.
CPU HeFFTe 2.3.0, LUMI-C `debug`, 8 OpenMP threads.

| job | run | \(N\) | CFL | steps | dt |
|----:|-----|------:|----:|------:|---:|
| 22188502 | `ic128_nu001_cfl04` | 128 | 0.4 | 2038 | 0.019635 |
| 22188521 | `ic256_nu001_cfl04` | 256 | 0.4 | 4074 | 0.009817 |
| 22188522 | `ic256_nu001_cfl02` | 256 | 0.2 | 8148 | 0.004909 |
| 22188569 | analyze 256 CFL 0.4 | | | | |
| 22188570 | analyze 256 CFL 0.2 | | | | |

128² topology/sheet analysis ran on login with cray-python 3.11
after job 22188502; 256² analysis is jobs 22188569/22188570.

CTest on compute job 22188495: 6/6 MHD tests passed (`mhd2d-cpu`,
`mhd2d-cli-verify`, `mhd2d-cli-coalescence`). Topology script with
cray-python 3.11: `ALL TOPOLOGY TESTS PASSED`.

### Ladder

1. **t=0 topology.** Four X, two \(\mathrm{O}_{\max}\), two
   \(\mathrm{O}_{\min}\). Tracked family: X nearest \((\pi,\pi)\),
   enclosed \(\mathrm{O}_{\max}\) nearest \((\pi/2,\pi/2)\),
   \(F(0)=0.40000\). Analytic test in
   `scripts/test_mhd_topology.py`. Frame:
   `figures/coalescence/ic256_t0.png`.
2. **Islands approach and coalesce.** The \(\mathrm{O}_{\max}\)
   walks along the diagonal toward the stationary X at
   \((\pi,\pi)\): at \(t=0\), \((x_O,y_O)\approx(1.57,1.57)\); at
   first `sheet_ok`, \(\approx(1.66,1.66)\); by \(t\approx 40\),
   \(\approx(2.67,2.67)\). \(F: 0.400\to 0.031\) (\(p\approx 0.92\)).
3. **One primary family.** The tracker holds the \((\pi,\pi)\) X
   and its enclosed \(\mathrm{O}_{\max}\) through the run.
   Hessian \(\mathrm{cond}(X)\) at \(t=0\) is 1.
4. **Topology survives refinement.** At matched times, \(F(t)\)
   and the event means agree among 128² CFL 0.4, 256² CFL 0.4, and
   256² CFL 0.2 to relative \(\sim 10^{-11}\). The X coordinate
   differs by half a cell (stationary saddle at \(\pi\)); that
   does not move the scalars. No 512² run is required.
5. **Ohm.** On 256² CFL 0.4,
   \(\mathrm{rms}(E_{z,X}-\eta j_X)=1.86\times 10^{-6}\).
6. **Flux budget.** Identity
   \(\mathrm{rms}(\mathrm{d}F/\mathrm{d}t-(E_{z,X}-E_{z,O}))
   =1.8\times 10^{-17}\). Resistive residual
   \(\mathrm{rms}(\mathrm{d}F/\mathrm{d}t-\eta(j_X-j_O))
   =2.0\times 10^{-6}\).
7. **Sheet.** First `sheet_ok` at \(t=9.032\), \(p=0.189\),
   \(\delta\approx 0.45\), \(L\approx 1.00\), \(B_{\mathrm{up}}\approx 0.23\).
   The frozen event window is common `sheet_ok` (32/32 dumps).
8. **Same event, spatially and timestep converged.** Frozen-window
   means on 256² CFL 0.4 (128² and CFL 0.2 identical at printed
   precision):

   | quantity | value |
   |----------|------:|
   | mean \(R\) | \(0.1404\) |
   | mean \(S_{\mathrm{local}}\) | \(17.15\) |
   | mean \(\delta/L\) | \(0.4385\) |
   | mean \(j_X\) | \(-0.4062\) |
   | mean \(L\) | \(0.9773\) |
   | mean \(B_{\mathrm{up}}\) | \(0.1767\) |
   | flux pile-up ratio | \(1.00\) (false; \(B_{\mathrm{up}}\) falls) |
   | sloshing sentinel | none |

   Relative 128² vs 256² and CFL 0.4 vs 0.2 differences are
   \(\lesssim 2\times 10^{-11}\). Global KE at matched times is
   likewise roundoff-close. Elsasser-sum CFL stayed \(\le 0.17\).
9. **Frames.** `figures/coalescence/`: \(t=0\), first `sheet_ok`,
   mid-event \(p\approx 0.40\), and the \(F(t)\) / O-path series.
   Late \(p\gtrsim 0.70\) frames are not comparison evidence
   (collapsing \(B_{\mathrm{up}}\)).

\(\max|j|\) decays from 0.80 because that peak is O-point current,
not a reconnection proxy.

Do not proceed to \(\eta=0.005\) in this PR. Do not fit a scaling
exponent.


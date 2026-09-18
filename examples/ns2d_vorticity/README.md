<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# 2-D vorticity–streamfunction Navier–Stokes (issue #21)

Research prototype, not a shipped `apps/` catalog entry. It lives under
`examples/` so the sixteen-application catalog stays closed. Shared
spectral pieces are `include/ns2d/spectral.hpp`. Navier–Stokes is
`vorticity_stream.hpp` (#21/#22). Two-dimensional incompressible
visco-resistive MHD is `mhd.hpp` (#23). This is **not** Strauss reduced
MHD and it is **not** Cahn–Hilliard–Navier–Stokes.

## Equations

Periodic 2-D incompressible Navier–Stokes in vorticity–streamfunction
form:

\[
\partial_t\omega+\mathbf u\cdot\nabla\omega=\nu\nabla^2\omega,
\qquad
\nabla^2\psi=-\omega,
\qquad
\mathbf u=(\partial_y\psi,-\partial_x\psi).
\]

OpenPFC's FFT stack is three-dimensional; the grid is an
\(N\times N\times 1\) slab and \(k_z=0\) for every mode.

Taylor–Green uses \([0,2\pi]^2\). The Minion–Brown / Bell double shear
uses the **unit square**. Do not put \(\rho=30\) on \([0,2\pi]^2\): that
makes the layer \(2\pi\) times thinner than the benchmark. The same
physical layer on \([0,2\pi]^2\) would need \(\rho'=\rho/(2\pi)\) and
\(\nu'=2\pi\nu\). This code does not do that transformation.

### Zero mode

\(\hat\psi(\mathbf 0)=0\). Mean vorticity is conserved: \(L(0)=0\) and
\(\hat N(\mathbf 0)=0\). The code does not project
\(\hat\omega(\mathbf 0)\) to zero.

### Discretization

- FFT Poisson: \(\hat\psi_{\mathbf k}=\hat\omega_{\mathbf k}/|\mathbf k|^2\)
  for \(\mathbf k\neq 0\).
- Spectral derivatives use \(i k\) with the Nyquist component zeroed
  (`k_component_odd`).
- **2/3 dealiasing of the state and of \(\hat N\).** The initial
  condition and every IFRK4 stage are projected onto the retained band
  *before* real-space products. Masking only \(\hat N\) after the product
  is not enough if the state still holds modes above the cutoff.
- **Integrating-factor RK4** on \(L=-\nu|\mathbf k|^2\). The viscous
  piece is \(e^{L\Delta t}\); the Jacobian is classical RK4 in the IF
  variable. When \(N\equiv 0\) (Taylor–Green) the step is exact. When
  \(L\equiv 0\) it is RK4, whose stability region includes the imaginary
  axis up to \(|z|\approx 2\sqrt{2}\). ETDRK2/Heun does not
  (\(|1+i\omega+(i\omega)^2/2|>1\) for every \(\omega\neq 0\)).

## Diagnostics

2-D means over the \(N^2\) cells (\(N_z=1\)):

| Name | Normalization |
|------|----------------|
| `ke` | \(\frac12\langle u^2+v^2\rangle\) |
| `enstrophy` | \(\frac12\langle\omega^2\rangle\) |
| `max_abs_omega` | \(\max\|\omega\|\) |
| `div_linf` / `div_l2` | \(L^\infty\) and RMS of spectral \(\partial_x u+\partial_y v\) |
| `cfl` | \(\Delta t\,\max(\|u\|,\|v\|)/\min(\Delta x,\Delta y)\) |
| `mean_omega` | \(\langle\omega\rangle\) (conserved) |
| `wall_step_s` | wall time of the last timed step |

Shear resolution (unit square, thickness \(1/\rho\)):

\[
\text{cells/thickness}=N/\rho,\qquad
\text{2/3-effective}=(2N/3)/\rho.
\]

For \(\rho=30\): \(N=128\) has 4.27 cells (2.84 effective); \(N=256\)
has 8.53 (5.69); \(N=512\) has 17.07 (11.38).

## HIP / GPU

Not implemented. `GPUSpectralStack` / `SpectralETDOps` expose pointwise
\(N(\psi)\) plus a diagonal \(L(k)\), not Poisson plus four spectral
derivatives.

## Commands

```bash
./scripts/build.sh --machine=lumi --cpu --no-submit \
  --build-dir=/flash/project_462001519/juaho/build/openpfc-ns2d-21-cpu
ctest --test-dir <build> -R ns2d-vorticity --output-on-failure
```

Taylor–Green (correctness, not a movie):

```bash
./examples/ns2d_vorticity/ns2d_vorticity --verify --N 32 --steps 20 \
  --dt 0.05 --nu 0.1 --outdir results/ns2d/tg
```

Minion–Brown unit-square shear, CFL \(0.4\) so
\(\Delta t=0.4/N\), \(\nu=10^{-4}\), \(\rho=30\), \(\varepsilon=0.05\),
to the standard \(t=1.2\):

```bash
# N=256: 8.5 cells/thickness; animation-ready VTK
./examples/ns2d_vorticity/ns2d_vorticity --case shear --N 256 \
  --steps 768 --cfl 0.4 --nu 1e-4 --rho 30 --eps 0.05 --dump 64 \
  --outdir /scratch/project_462001519/juaho/ns2d-21-ifrk4/shear256
```

On LUMI write large series under scratch, not flash. ParaView loads
`omega_*.vti`. Rank 0 writes `diagnostics.csv` and `run.json`.

The driver aborts on non-finite diagnostics or CFL \(> 2\).

## Evidence (corrected integrator / scaling / dealias)

CPU HeFFTe 2.3.0 FFTW/Milan, 1 rank. Catch2: 10 cases, 341 assertions.
CLI `--verify` PASS. Source: this branch after the PR #22 review.

### Taylor–Green \(N=32\), \(\nu=0.1\), \(\Delta t=0.05\), \(t=1\)

- \(L^\infty(\omega)=2.76\times10^{-15}\)
- KE / enstrophy match \(e^{-4\nu t}/4\) and \(e^{-4\nu t}/2\) to
  \(\sim10^{-15}\)
- `div_linf` \(\sim10^{-15}\)

### Nonlinear tests (Taylor–Green cannot satisfy these)

- Two-mode IFRK4: error ratio \(>8\) when \(\Delta t\) is halved
  (first-order ETD1-Euler would be \(\approx 2\)).
- Inviscid two-mode, 80 steps, \(\Delta t=0.02\): finite, KE relative
  drift \(<10^{-3}\), mean \(\omega\) conserved.

### Aliasing

On \(N=32\), \(\sin(12x)\sin(13x)\) aliases into the retained \(k=1\)
cosine. Projecting each factor onto the 2/3 band before the product
removes that contamination.

### Minion–Brown shear, \(\rho=30\), \(\nu=10^{-4}\), CFL \(0.4\)

All three resolutions finished \(t=1.2\) finite. Mean \(\omega\) is
constant to printer precision. Spectral divergence stays \(\le10^{-12}\).

| \(N\) | cells / \(1/\rho\) | KE(\(t=1.2\)) | \(Z(t=1.2)\) | \(\max\|\omega\|\) | \(\max\|u\|\) peak |
|------:|-------------------:|--------------:|-------------:|-------------------:|-------------------:|
| 128 | 4.27 (2.84 eff.) | 0.425162 | 32.1945 | 28.297 | 1.438 at \(t=0.8\) |
| 256 | 8.53 (5.69) | 0.425162 | 32.1923 | 28.206 | 1.439 at \(t=0.8\) |
| 512 | 17.07 (11.38) | 0.425162 | 32.1923 | 28.206 | 1.439 at \(t=0.8\) |

256 and 512 agree to five digits on KE, enstrophy and
\(\max\|\omega\|\) at \(t=1.2\). \(\max\|u\|\) grows \(1.00\to 1.44\)
by \(t=0.8\) (KH roll-up) then sits near \(1.28\)–\(1.33\) through
\(t=1.6\) on \(N=256\) with no blow-up.

Animation-ready series: `shear256_t16` (17 VTK frames, \(t=0\)–\(1.6\))
under `/scratch/project_462001519/juaho/ns2d-21-ifrk4/`. Frames were
not rendered in this session; the claim of roll-up is from the
converged diagnostics, not from a screenshot.

### Preserved ETD1 failure (do not treat as the #21 gate)

The first PR #22 runs used ETD1 (forward Euler on the Jacobian),
\(\rho=30\) on \([0,2\pi]^2\) (layer \(2\pi\) times too thin), and
post-product-only 2/3 masking. Those blew up: \(N=128\), \(\nu=10^{-4}\)
NaN by \(t=4\); \(N=256\), \(\nu=5\times10^{-4}\) NaN by \(t=4.5\) at
CFL \(0.15\). Raw files remain at
`/scratch/project_462001519/juaho/ns2d-21/`. That failure was a
numerical-method artefact, not evidence that vorticity–streamfunction
NS is a poor fit.

## Decision gate (issue #21)

After the three review blockers:

The corrected CPU solver is a **credible 2-D incompressible core**:
Taylor–Green is spectrally exact, IFRK4 has nonlinear order well above
one, 2/3 projection passes an aliasing test, and the standard
Minion–Brown shear is grid-converged at 256²/512² through roll-up
without blow-up.

Issue #23 (below) is the MHD coupling experiment. CHNS is still a
separate later choice.

Still unverified for NS: HIP; multi-rank; ParaView inspection of VTK;
vortex *merger* as a distinct later-time event beyond the \(t=0.8\)
roll-up / \(t=1.6\) persistence already integrated.

## 2-D incompressible visco-resistive MHD (issue #23)

Not Strauss (1976) reduced MHD (no guide field, no parallel dynamics).
Literature for this experiment: Orszag & Tang, JFM 90 (1979);
Pouquet, JFM 88 (1978).

### Model and signs

\[
\mathbf u=(\partial_y\phi,-\partial_x\phi),\quad
\omega=-\nabla^2\phi,\quad
\mathbf B=(\partial_y a,-\partial_x a),\quad
j=-\nabla^2 a,
\]

\[
\partial_t\omega+\mathbf u\cdot\nabla\omega
=\mathbf B\cdot\nabla j+\nu\nabla^2\omega,
\qquad
\partial_t a+\mathbf u\cdot\nabla a=\eta\nabla^2 a.
\]

The Lorentz term is **\(+B\cdot\nabla j\)**. A test that flips the sign
fails on an Alfvénic two-mode state (\(N_\omega\) jumps from
\(<10^{-10}\) to \(>1\)).

Shared with NS: `SpectralPlane` Poisson, odd-\(k\) derivatives, state
2/3 projection, IFRK4. Prognostic fields are \(\omega\) (stack field)
and \(a\) (second inbox `Field`). Density and \(\mu_0\) are 1.

### Diagnostics (2-D means)

| Name | Normalization |
|------|----------------|
| `ke` | \(\frac12\langle\|u\|^2\rangle\) |
| `me` | \(\frac12\langle\|B\|^2\rangle\) |
| `energy` | `ke+me` |
| `cross_helicity` | \(\langle u\cdot B\rangle\) |
| `a2` | \(\frac12\langle(a-\langle a\rangle)^2\rangle\), with \(\hat a(0)=0\) |
| `enstrophy` | \(\frac12\langle\omega^2\rangle\) so \(\langle\omega^2\rangle=2Z\) |
| `mean_sq_j` | \(\langle j^2\rangle\) |
| `dissipation` \(D\) | \(\nu\langle\omega^2\rangle+\eta\langle j^2\rangle\) |
| `budget_residual` | \((E_n-E_{n-1})/\Delta t_{\mathrm{diag}}+\frac12(D_n+D_{n-1})\) |
| `cfl_nominal` | \(\Delta t/\Delta x\) (what `--cfl` sets) |
| `cfl_ub` | \(\Delta t\,\max_i(|u_i|,|B_i|)/\Delta x\) (deprecated) |
| `cfl_elsasser` | \(\Delta t\,\max_i|z^\pm_i|/\Delta x\), \(z^\pm=u\pm B\) |

`--cfl` remains a **nominal fixed-\(\Delta t\) selector**:
\(\Delta t=\mathrm{CFL}_{\mathrm{nom}}\Delta x\) (characteristic speed 1).
The measured MHD CFL is the Elsasser quantity `cfl_elsasser`. The
driver aborts if that measured value exceeds 2. There is no adaptive
timestepping.

\(A_2\) is gauge-safe: the DC mode of \(a\) is zeroed after every
stage, and the reported invariant uses the variance of \(a\).

### Commands

```bash
# force-free magnetic eigenmode (CLI verify)
./examples/ns2d_vorticity/mhd2d --verify --N 16 --steps 8 \
  --dt 0.05 --nu 0.1 --eta 0.1

# incompressible Orszag–Tang, Pm=1, peak-current window at nu=eta=0.005
./examples/ns2d_vorticity/mhd2d --case orszag_tang --N 256 \
  --steps 255 --cfl 0.4 --nu 0.005 --eta 0.005 --diag 4 --dump 4 \
  --outdir /scratch/project_462001519/juaho/mhd2d-23/ot256_nu0005_peak

./examples/ns2d_vorticity/mhd2d --case orszag_tang --N 512 \
  --steps 509 --cfl 0.4 --nu 0.005 --eta 0.005 --diag 8 --dump 8 \
  --outdir /scratch/project_462001519/juaho/mhd2d-23/ot512_nu0005_peak

# matched hydro control (same u, a=0, same nu)
./examples/ns2d_vorticity/mhd2d --case hydro_control --N 256 \
  --steps 255 --cfl 0.4 --nu 0.005 --eta 0.005 --diag 4 \
  --outdir /scratch/project_462001519/juaho/mhd2d-23/hydro256_nu0005_peak
```

`--diag` writes CSV/stdout without requiring a VTK dump. Field dumps
are Fortran-order `double` bricks (`a_%04d.bin`, `j_%04d.bin`,
`omega_%04d.bin`) plus VTK. Overlay \(j\) with \(a\) contours; do
**not** call that reconnection.

Common-band comparison and rendering (cray-python 3.11 + numpy;
matplotlib for figures):

```bash
python3 examples/ns2d_vorticity/scripts/compare_mhd_fields.py \
  --coarse-dir /scratch/project_462001519/juaho/mhd2d-23/ot256_nu0005_peak \
  --fine-dir   /scratch/project_462001519/juaho/mhd2d-23/ot512_nu0005_peak \
  --coarse-n 256 --fine-n 512 --coarse-inc 228 --fine-inc 456

python3 examples/ns2d_vorticity/scripts/render_mhd_frames.py \
  --dir /scratch/project_462001519/juaho/mhd2d-23/ot256_nu0005_peak \
  --n 256 --inc 228 \
  --out /scratch/project_462001519/juaho/mhd2d-23/ot256_nu0005_peak/frame_0228.png

python3 examples/ns2d_vorticity/scripts/plot_mhd_diagnostics.py \
  --mhd /scratch/project_462001519/juaho/mhd2d-23/ot256_nu0005_peak/diagnostics.csv \
  --hydro /scratch/project_462001519/juaho/mhd2d-23/hydro256_nu0005_peak/diagnostics.csv \
  --out /scratch/project_462001519/juaho/mhd2d-23/ot256_nu0005_peak/mhd_vs_hydro.png
```

### Verification (Catch2 `mhd2d-cpu`, 12 cases)

- \(a=0\) MHD matches NS on two-mode IC (\(L^\infty<10^{-11}\)).
- Force-free \(a=\sin x\sin y\), \(u=0\): \(a(t)=a(0)e^{-2\eta t}\)
  to \(10^{-11}\); velocity stays 0.
- Alfvénic \(u=B\): \(N_\omega\sim 0\) with Lorentz \(+\); \(N_\omega>1\)
  if the sign is flipped.
- Elsasser: OT at \(t=0\) has \(\max_i|z^\pm_i|=2\), so
  `cfl_elsasser = 2 cfl_nominal`. Aligned \(u=B=(1,0)\) gives
  \(\max|z^+|=2\).
- Gauge: \(a+\mathrm{const}\) is projected to \(\langle a\rangle=0\)
  and \(A_2=0.3125\) for the OT flux.
- Ideal OT, \(N=32\), \(T=0.2\), \(\nu=\eta=0\). Relative drift of
  \(E\), \(H_c\) and \(A_2\) (high-precision CSV):

  | \(\Delta t\) | \(\delta E/E\) | \(\delta H_c/H_c\) | \(\delta A_2/A_2\) |
  |-------------:|---------------:|-------------------:|-------------------:|
  | 0.02 | \(1.18\times10^{-9}\) | \(2.44\times10^{-9}\) | \(5.60\times10^{-10}\) |
  | 0.01 | \(3.54\times10^{-11}\) | \(8.19\times10^{-11}\) | \(2.25\times10^{-11}\) |
  | 0.005 | \(1.00\times10^{-12}\) | \(2.91\times10^{-12}\) | \(1.02\times10^{-12}\) |
  | 0.0025 | \(2.15\times10^{-14}\) | \(1.13\times10^{-13}\) | \(4.90\times10^{-14}\) |

  Halving \(\Delta t\) from 0.02 to 0.01 reduces the three drifts by
  \(\approx 25\)–\(33\). That is clearly higher than first order
  (ratio 2). The window is too close to roundoff on the last halving
  to claim a formal RK4/fourth-order rate.
- Trapezoidal budget residual drops as the diagnostic interval is
  coarsened in reverse: every-step residual \(<5\times10^{-3}\) and
  smaller than the 4-step and 8-step residuals.
- `restrict_hat_by_k` recovers a shared trigonometric polynomial
  from \(N=32\) onto \(N=16\).
- \(\nabla\cdot u\) and \(\nabla\cdot B\) at roundoff.

### Orszag–Tang ladder (incompressible, \([0,2\pi]^2\))

\(\phi=\cos x+\cos y\), \(a=\frac12\cos 2x+\cos y\).
CPU HeFFTe 2.3.0 FFTW/Milan, 1 rank. Raw:
`/scratch/project_462001519/juaho/mhd2d-23/`.
Nominal `--cfl 0.4` so \(\Delta t=0.4\cdot 2\pi/N\). Measured Elsasser
CFL starts at 0.8.

**\(P_m=1\), \(\nu=\eta=0.02\), \(t=2\):** 128² and 256² agree to six
digits on \(E\), KE, ME, \(\max\|j\|\). Finite, `div` \(\sim10^{-14}\).

| \(N\) | \(E(t=2)\) | KE | ME | \(\max\|j\|\) | \(\max\|\omega\|\) |
|------:|----------:|---:|---:|-------------:|------------------:|
| 128 | 0.760592 | 0.219881 | 0.540711 | 15.752 | 4.322 |
| 256 | 0.760592 | 0.219881 | 0.540711 | 15.752 | 4.322 |

**Matched hydro control.** The OT velocity has vanishing Jacobian, so
hydro is an exact decaying eigenmode. At \(\nu=0.005\), \(N=256\),
\(t=2.50\):

| | KE | ME | \(E\) | \(\max\|\omega\|\) | \(\max\|j\|\) |
|---|---:|---:|---:|---:|---:|
| hydro | 0.4876 | 0 | 0.4876 | 1.975 | 0 |
| MHD | 0.2299 | 0.6219 | 0.8518 | 13.58 | 31.62 |

Hydro KE decays \(0.50\to 0.488\). MHD transfers kinetic to magnetic
energy and forms current sheets. Figure:
`ot256_nu0005_peak/mhd_vs_hydro.png`.

**\(\nu=\eta=0.005\), peak current from the time series.** \(t=1.5\)
is still on the rise. On 256² with `--diag 4` the global
\(\max\|j\|\) peaks at \(t=2.238\) (\(j=38.237\)), with an earlier
shoulder at \(t=1.924\) (\(j=35.751\)). 512² agrees in those
scalars to five digits. Measured Elsasser CFL at the peak is 1.10
(nominal 0.4). Extending 256² toward \(t=3\) **fails closed** at
step 312 (\(t=3.063\)): `cfl_elsasser` jumps \(1.28\to 8.71\) after
\(\max\|j\|\) explodes. The defensible window is \(t\le 2.5\).

**Field-level 256² vs 512²** (restrict 512 r2c hats onto the 256
integer-\(k\) lattice, then relative \(L^2\)):

| \(t\) | \(\|a\|_{2,\mathrm{rel}}\) | \(\|j\|_{2,\mathrm{rel}}\) | \(\|\omega\|_{2,\mathrm{rel}}\) | ME-spectrum | \(j\)-spectrum |
|------:|---------------------------:|---------------------------:|-------------------------------:|------------:|---------------:|
| 1.492 | \(4.4\times10^{-8}\) | \(9.6\times10^{-5}\) | \(5.2\times10^{-5}\) | \(2.3\times10^{-9}\) | \(6.4\times10^{-8}\) |
| 1.924 | \(5.6\times10^{-7}\) | \(1.0\times10^{-3}\) | \(4.9\times10^{-4}\) | \(4.4\times10^{-9}\) | \(1.3\times10^{-6}\) |
| 2.238 | \(5.1\times10^{-7}\) | \(8.7\times10^{-4}\) | \(4.3\times10^{-4}\) | \(6.5\times10^{-9}\) | \(9.9\times10^{-7}\) |

At the \(\max\|j\|\) time the operational sheet thickness from
periodic FWHM of \(|j|\) is 5 cells on 256² (\(\approx 0.12\)).
256² and 512² frames at \(t=2.24\) are visually the same:
current sheets, deformed flux contours, no claim of reconnection
or plasmoids. Rendered:
`ot256_nu0005_peak/frame_0228.png`,
`ot512_nu0005_peak/frame_0456.png`.

**Exploratory \(\nu=\eta=0.0025\).** 256² survives to \(t=2.50\)
with peak \(\max\|j\|=54.97\) at \(t=1.964\). 512² agrees in \(E\)
but not in the sheet: at \(t=1.885\), common-band
\(\|j\|_{2,\mathrm{rel}}=1.4\times10^{-2}\),
\(\max\|j\|\) 54.63 vs 55.6. 512² then **blows up** at
\(t=2.474\) (`cfl_elsasser` \(1.70\to 54\)). 256² is
under-resolved; 512² is not a stable late-time solution at this
dissipation. 1024² was **not** run. Do not go to still lower
\(\eta\) in this PR.

### Decision gate (issue #23)

OpenPFC now has a verified and spatially converged 2-D
incompressible MHD research prototype whose magnetic coupling
generates robust current-sheet dynamics beyond the matched
hydrodynamic control.

That statement is limited to \(P_m=1\), \(\nu=\eta=0.005\),
\(t\le 2.5\), 256²/512² common-band fields. It does **not**
cover \(\nu=\eta=0.0025\), \(t>2.5\), reconnection, or plasmoids.

A later issue may add quantitative magnetic reconnection /
topology diagnostics. This PR does not implement that project
and does not start Strauss reduced MHD.

HIP is still blocked by the same pointwise `SpectralETDOps` pipeline.

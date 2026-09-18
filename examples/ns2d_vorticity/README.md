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
| `a2` | \(\frac12\langle a^2\rangle\) |
| `enstrophy` | \(\frac12\langle\omega^2\rangle\) |
| `mean_sq_j` | \(\langle j^2\rangle\) |
| `dissipation` | \(\nu\langle\omega^2\rangle+\eta\langle j^2\rangle\) |
| `budget_residual` | \(\Delta E/\Delta t_{\mathrm{dump}}+\mathrm{dissipation}\) |
| `cfl` | \(\Delta t\,\max(\|u\|,\|B\|)/\Delta x\) |

### Commands

```bash
# force-free magnetic eigenmode (CLI verify)
./examples/ns2d_vorticity/mhd2d --verify --N 16 --steps 8 \
  --dt 0.05 --nu 0.1 --eta 0.1

# incompressible Orszag–Tang, Pm=1, enough dissipation for 128=256
./examples/ns2d_vorticity/mhd2d --case orszag_tang --N 256 \
  --steps 204 --cfl 0.4 --nu 0.02 --eta 0.02 --dump 17 \
  --outdir /scratch/project_462001519/juaho/mhd2d-23/ot256

# matched hydro control (same u, a=0)
./examples/ns2d_vorticity/mhd2d --case hydro_control --N 256 \
  --steps 204 --cfl 0.4 --nu 0.02 --eta 0.02 --dump 17 \
  --outdir /scratch/project_462001519/juaho/mhd2d-23/hydro256
```

VTK: `omega_*.vti`, `a_*.vti`, `j_*.vti`. Overlay \(j\) with \(a\)
contours in ParaView; do **not** call that reconnection.

### Verification (Catch2 `mhd2d-cpu`, 7 cases)

- \(a=0\) MHD matches NS on two-mode IC (\(L^\infty<10^{-11}\)).
- Force-free \(a=\sin x\sin y\), \(u=0\): \(a(t)=a(0)e^{-2\eta t}\)
  to \(10^{-11}\); velocity stays 0.
- Alfvénic \(u=B\): \(N_\omega\sim 0\) with Lorentz \(+\); \(N_\omega>1\)
  if the sign is flipped.
- Ideal OT, 40 steps, \(\nu=\eta=0\): relative drift of \(E\) and \(A_2\)
  \(<2\times10^{-3}\).
- Dissipative budget residual \(<5\times10^{-3}\) when sampled every
  step.
- \(\nabla\cdot u\) and \(\nabla\cdot B\) at roundoff.

### Orszag–Tang ladder (incompressible, \([0,2\pi]^2\))

\(\phi=\cos x+\cos y\), \(a=\frac12\cos 2x+\cos y\).
CPU HeFFTe 2.3.0, 1 rank. Raw:
`/scratch/project_462001519/juaho/mhd2d-23/`.

**\(P_m=1\), \(\nu=\eta=0.02\), CFL 0.4, \(t=2\):** 128² and 256²
agree to six digits on \(E\), KE, ME, \(\max\|j\|\). Finite,
`div` \(\sim10^{-14}\).

| \(N\) | \(E(t=2)\) | KE | ME | \(\max\|j\|\) | \(\max\|\omega\|\) |
|------:|----------:|---:|---:|-------------:|------------------:|
| 128 | 0.760592 | 0.219881 | 0.540711 | 15.752 | 4.322 |
| 256 | 0.760592 | 0.219881 | 0.540711 | 15.752 | 4.322 |

KE falls \(0.50\to 0.22\); ME rises \(0.50\to 0.54\). \(\max\|j\|\)
grows \(3\to 15.8\) (current-sheet formation). No reconnection
diagnostic is implemented.

**Matched hydro control** (256², same \(u\), \(a=0\)): the OT
velocity has vanishing Jacobian, so hydro is an exact decaying
eigenmode (KE \(0.50\to 0.46\), \(\max\|\omega\|\) \(2\to 1.92\)).
Magnetic coupling is the entire nonlinear dynamics.

**Thinner sheets \(\nu=\eta=0.005\), \(t=1.5\):**

| \(N\) | \(E\) | \(\max\|j\|\) | \(\max\|\omega\|\) |
|------:|------:|-------------:|------------------:|
| 128 | 0.94775 | 25.73 | 7.70 |
| 256 | 0.94700 | 26.90 | 8.00 |
| 512 | 0.94700 | 26.90 | 7.99 |

256 and 512 agree; 128 under-resolves peak current by \(\sim 4\%\).
Budget residuals after the dump-interval fix are \(O(10^{-3})\).

Animation-ready: `ot256` and `ot256_nu0005` (`omega`, `a`, `j` VTK).
Frames were not rendered here.

### Decision gate (issue #23)

- Magnetic dynamics are **robust and grid-converged** at \(P_m=1\) for
  \(\nu=\eta=0.02\) already at 128², and for \(\nu=\eta=0.005\) at
  256²/512².
- MHD **does** add scientifically distinct behaviour: the hydro control
  is a linear eigenmode; MHD transfers kinetic to magnetic energy and
  forms current sheets.
- Current-sheet formation is converged. That **justifies a later
  reconnection / topology issue** with a quantitative X-point or flux
  diagnostic. This PR does **not** claim reconnection and does **not**
  start Strauss RMHD.
- Keep this as an **MHD verification / teaching prototype** until that
  reconnection issue is posed. Do not pivot to CHNS on this evidence:
  the magnetic coupling is the interesting part.

HIP is still blocked by the same pointwise `SpectralETDOps` pipeline.

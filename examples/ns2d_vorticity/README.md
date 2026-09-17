<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# 2-D vorticity–streamfunction Navier–Stokes (issue #21)

Research prototype, not a shipped `apps/` catalog entry. It lives under
`examples/` so the sixteen-application catalog stays closed. The
incompressible-flow core is `include/ns2d/vorticity_stream.hpp`. This
directory does **not** implement reduced MHD or Cahn–Hilliard–Navier–Stokes.

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

That is enough to **keep the prototype and allow a later coupling
issue** (reduced MHD or CHNS — choose in a new issue, do not start
either here). It is no longer accurate to park the work solely because
ETD1 blew up on a mis-scaled layer.

Still unverified: HIP; multi-rank; visual inspection of the VTK
series in ParaView; vortex *merger* as a distinct later-time event
beyond the \(t=0.8\) roll-up / \(t=1.6\) persistence already
integrated.

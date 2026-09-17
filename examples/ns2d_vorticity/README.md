<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# 2-D vorticity–streamfunction Navier–Stokes (issue #21)

Research prototype, not a shipped `apps/` catalog entry. It lives under
`examples/` so the sixteen-application catalog stays closed. The
incompressible-flow core is `include/ns2d/vorticity_stream.hpp`; a later
reduced-MHD or Cahn–Hilliard–Navier–Stokes issue could call it. This
directory does **not** implement either of those models.

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

The grid is an \(N\times N\times 1\) slab on \([0,2\pi]^2\). OpenPFC's
FFT stack is three-dimensional; \(k_z=0\) for every mode.

### Zero mode

The Poisson problem determines \(\psi\) only up to a constant. The
solver sets \(\hat\psi(\mathbf 0)=0\) (zero-mean streamfunction). Mean
vorticity \(\hat\omega(\mathbf 0)\) is conserved circulation: \(L(0)=0\)
and the Jacobian of a periodic incompressible flow has zero spatial
mean, which is enforced by writing \(\hat N(\mathbf 0)=0\) after the
dealiased product. The code does **not** project \(\hat\omega(\mathbf 0)\)
to zero.

### Discretization

- FFT Poisson: \(\hat\psi_{\mathbf k}=\hat\omega_{\mathbf k}/|\mathbf k|^2\)
  for \(\mathbf k\neq 0\).
- Spectral derivatives use \(i k\) with the Nyquist component zeroed
  (`k_component_odd`), matching `SpectralGradient`.
- Pseudospectral Jacobian \(N=-u\cdot\nabla\omega\), then Orszag 2/3
  dealiasing of \(\hat N\) via `fill_two_thirds_mask`. The mask is the
  existing cubic-safe helper; for a quadratic term it is conservative.
- Time integrator: ETD1 on \(L=-\nu|\mathbf k|^2\) with the dealiased
  Jacobian as the explicit term (`fill_spectral_exp_coeffs` /
  `apply_etd1_update`).

Taylor–Green is a single Fourier mode whose Jacobian vanishes, so ETD1
is exact in time for that linear decay. Spatial exactness is therefore
**not** a mesh-refinement study. Timestep order is measured on the
two-mode nonlinear initial condition.

## Diagnostics

Every quantity below is a **2-D mean** over the \(N^2\) cells
(\(N_z=1\)), not a 3-D volume integral.

| Name | Normalization |
|------|----------------|
| `ke` | \(\frac12\langle u^2+v^2\rangle\) |
| `enstrophy` | \(\frac12\langle\omega^2\rangle\) |
| `max_abs_omega` | \(\max\|\omega\|\) |
| `div_linf` / `div_l2` | \(L^\infty\) and RMS of spectral \(\partial_x u+\partial_y v\) |
| `cfl` | \(\Delta t\,\max(\|u\|,\|v\|)/\min(\Delta x,\Delta y)\) |
| `mean_omega` | \(\langle\omega\rangle\) (conserved) |
| `wall_step_s` | wall time of the last timed step |

## HIP / GPU

Not implemented. `GPUSpectralStack` and `SpectralETDOps` expose a
pointwise \(N(\psi)\) plus a diagonal \(L(k)\). The Navier–Stokes
Jacobian needs a Poisson solve and four spectral derivatives, which
that pipeline does not compose. Building a second device FFT/Poisson
stack for this prototype would duplicate runtime infrastructure.
A later issue can add device kernels **on top of** `IDeviceFFT` once
the CPU decision gate is passed.

## Commands

Build with the repository script (CPU, HeFFTe FFTW):

```bash
./scripts/build.sh --machine=lumi --cpu --no-submit \
  --build-dir=/flash/project_462001519/juaho/build/openpfc-ns2d-21-cpu
```

Tests (Catch2 mathematics + CLI entry point):

```bash
ctest --test-dir <build> -R ns2d-vorticity --output-on-failure
```

Taylor–Green verification (quantitative; this is the correctness gate):

```bash
srun -n 1 ./examples/ns2d_vorticity --verify --N 32 --steps 20 \
  --dt 0.05 --nu 0.1 --outdir results/ns2d/tg
```

Double shear / Kelvin–Helmholtz showcase (visual only; not a pass):

```bash
srun -n 1 ./examples/ns2d_vorticity --case shear --N 256 --steps 2000 \
  --dt 0.002 --nu 1e-4 --rho 30 --eps 0.05 --dump 20 \
  --outdir results/ns2d/shear
```

On LUMI write large series under
`/scratch/project_462001519/juaho/`, not flash. ParaView loads
`omega_*.vti`. Rank 0 also writes `diagnostics.csv` and `run.json`.

A cheap shear smoke (roll-up is weaker than 256²):

```bash
srun -n 1 ./examples/ns2d_vorticity --case shear --N 128 --steps 400 \
  --dt 0.004 --nu 5e-4 --dump 20 --outdir results/ns2d/shear128
```

The driver aborts if a diagnostic becomes non-finite or if CFL exceeds 2.
That is a fail-closed stop, not a successful roll-up.

## Evidence recorded for issue #21

CPU HeFFTe 2.3.0 (FFTW/Milan), login-node 1 rank, source this branch.
Catch2: 8/8 cases, 336 assertions. CLI `--verify` PASS.

Taylor–Green \(N=32\), \(\nu=0.1\), \(\Delta t=0.05\), \(t=1\):

- \(L^\infty(\omega)=1.93\times10^{-15}\)
- `ke` and enstrophy match \(e^{-4\nu t}/4\) and \(e^{-4\nu t}/2\) to
  \(\sim10^{-15}\)
- `div_linf` \(\sim10^{-15}\), mean \(\omega\) \(\sim10^{-18}\)

Two-mode nonlinear IC: mean \(\omega\) conserved to \(10^{-12}\);
ETD1 observed order on \(\Delta t\)-refinement is first-order
(ratio \(\approx 2\) when \(\Delta t\) is halved).

Double shear (Bell–Mei, \(\rho=30\), \(\varepsilon=0.05\)):

| Run | Result |
|-----|--------|
| \(N=128\), \(\nu=10^{-4}\), \(\Delta t=0.005\) | KH onset \(t\sim 1\)–\(3\) (`max_speed` \(1.06\to 1.97\)), then enstrophy explosion at \(t=3.5\) (`max\|\omega\|\) \(31\to 455\), CFL \(0.20\to 1.01\)), NaN by \(t=4\). |
| \(N=256\), \(\nu=5\times10^{-4}\), \(\Delta t=0.002\) | Finite through \(t=4\): `max_speed` \(1.00\to 1.82\), `max\|\omega\|` decays \(30\to 14.6\) then jumps to \(22.6\) at \(t=4\) (roll-up onset). Diverges to NaN by \(t=4.5\) with CFL still \(0.15\). |

The \(t=4\) 256² VTK series is the animation-ready onset dump. It is
**not** a completed vortex-merger showcase. The blow-up at CFL \(\ll 1\)
is a resolution / ETD1-Jacobian limitation, not a Poisson bug (TG is
spectrally exact).

## Decision gate (issue #21)

**Keep only as a verification / teaching prototype** until a
CFL-controlled, grid-converged shear run produces a finite interacting
vortex field. Do **not** start reduced MHD or CHNS on this evidence:
the incompressible core is correct on Taylor–Green, but the visual
transport experiment is not yet robust.

Do not treat a visually attractive VTK series as verification.

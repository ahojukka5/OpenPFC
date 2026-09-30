<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Periodic incompressible flow

Maintained application for the periodic Fourier velocity. It is built
when HeFFTe is enabled. It is not a copy of
`examples/ns2d_vorticity/`. That directory remains the 2-D
vorticity–streamfunction prototype and the owner of its spatial,
temporal, and closed-box checks. This binary calls the installed Leray
projector, the 2/3-dealiased rotational term, and integrating-factor
RK4.

## Equation

The cube is \([0,2\pi]^3\). One MPI rank advances

\[
\partial_t\hat{\mathbf u}=P(\mathbf u\times\boldsymbol\omega)
-\nu|k|^2\hat{\mathbf u}.
\]

\(P\) is the Fourier Leray projector. The 2/3 mask stays on for this
quadratic term. There is no body force. Viscosity is the diagonal
symbol \(-\nu|k_{\mathrm{odd}}|^2\), so an unforced mean velocity is
invariant. CPU only.

## Taylor–Green

The only shipped case is

\[
u=\sin x\cos y\cos z,\qquad
v=-\cos x\sin y\cos z,\qquad
w=0.
\]

Kinematic viscosity defaults to \(0.05\). The binary does not assign a
Reynolds number. The refinement tables for this field live in
[`examples/taylor_green3d/README.md`](../../examples/taylor_green3d/README.md).
At this viscosity the \(N=256\) series is the spatial reference, and
the timestep pairs that clear \(10^{-11}\) sit near order 4. This
application does not rerun that ladder.

## Command

```bash
./apps/incompressible_flow/incompressible_flow \
  --case taylor-green --n 32 --nu 0.05 --dt 0.01 --time 0.1 \
  --outdir results/taylor-green
```

`--time` must be an integer number of steps. \(N\) is even and at
least 8. Speed CFL is checked on the initial and final samples. Above
2 the run stops with `status=cfl`. A non-finite field leaves
`status=nonfinite`. The box length is \(2\pi\) and is not a flag.
`diagnostics.csv` records those two samples.

## Limits

The run is periodic, three-dimensional, unforced, and one rank. It
does not add walls, a channel, a pipe, a second viscosity, or a
forcing spectrum. A 2-D double shear is not reimplemented here.

<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# When OpenPFC is (and is not) the right tool

OpenPFC is a C++ / MPI framework for high-performance PDEs on structured
grids. Spectral (HeFFTe) and finite-difference discretizations are both in
scope. Phase-field crystal models are the origin of the project and one
important use, not the boundary of what the framework is for. This page is
an honest fit guide, not a comparison against every other code.

Who owns a concept, as opposed to whether the code fits a problem, is the
[semantic ownership](concepts/architecture.md#semantic-ownership) table.

## Good fit

- You want distributed structured grids and MPI, as a library or through a
  configuration-driven application. The roster is
  [`apps/README.md`](../apps/README.md).
- You are building a spectral model (linear pieces in wave-number space,
  nonlinear pieces in real space) or a finite-difference model on the same
  decomposition and halo infrastructure
  ([`halo_exchange.md`](concepts/halo_exchange.md)). PFC applications such
  as tungsten are one spectral family, not the only one.
- You can invest in a consistent MPI build, and in HeFFTe when you take the
  spectral path, including optional CUDA or HIP
  ([`INSTALL.md`](../INSTALL.md),
  [`troubleshooting.md`](troubleshooting.md)).

## Limits that are still true

Spectral discretizations fit many periodic and FFT-friendly problems. They
are a poor fit when you need local stencil control, some boundary
treatments, or mixed operators that fight a global FFT.

Finite-difference building blocks are already in the tree. `pfc::field::fd`
is declared in
[`finite_difference.hpp`](../include/openpfc/kernel/field/finite_difference.hpp).
Applications such as heat, acoustic waves, and Allen–Cahn use that path. A single
user-facing switch that selects spectral versus finite-difference for the
same operator in every application is still in progress. Shipped spectral
applications, including tungsten, stay FFT-centric until that wiring
exists. The design note is
[ADR 0002](adr/0002-gradient-operators-fd-vs-spectral.md).

OpenPFC is not an unstructured-mesh, finite-element, or adaptive-mesh
code. It is not a turn-key GUI, and it does not offer a production-support
contract. Plan engineering effort for your site.

Supported builds require MPI. `-DOpenPFC_ENABLE_MPI=OFF` is rejected.
HeFFTe is the default spectral stack. `-DOpenPFC_ENABLE_HEFFTE=OFF` is a
finite-difference / kernel-only build: spectral applications and examples
are skipped. See [`build_options.md`](reference/build_options.md).

## When another tool might be simpler

| Situation | Consider |
|-----------|----------|
| Small 1D/2D toy FFTs, and you do not want MPI | A NumPy, SciPy, or FFTW tutorial is faster to learn. There is no supported build without MPI. |
| You specifically want the 2-D vorticity–streamfunction prototype | [`examples/ns2d_vorticity`](../examples/ns2d_vorticity/README.md) (issue #21) is a bounded experiment, not a shipped application. Periodic 3-D incompressible Navier–Stokes is [`apps/incompressible_flow`](../apps/incompressible_flow/README.md). |
| Unstructured meshes, finite elements, or AMR as the discretization | A code built for that mesh. OpenPFC stays on structured grids. |
| You must match another community's formulation and ecosystem exactly | That community's code. PFC is one OpenPFC use; what the tungsten app solves is [`tungsten_quicklook.md`](science/tungsten_quicklook.md). |
| Guaranteed production support or a turn-key GUI | OpenPFC is a research-oriented C++ framework. |

## See also

- [`spectral_stack.md`](concepts/spectral_stack.md) — default spectral
  data flow
- [`numerics_limits.md`](science/numerics_limits.md) — stability and
  discretization caveats
- [`architecture.md`](concepts/architecture.md) — layers and semantic
  ownership

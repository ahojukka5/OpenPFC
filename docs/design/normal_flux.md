<!-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Node-centred normal-flux contract

The CPU point kernel uses the compact reflected diagonal-norm SBP family.
H D2(B)=-M(B)+E^T N B S. Natural-flux SAT cancels the derivative trace,
leaving H L(u)=-M(B)u-E^T J. Positive J is outward. Normal derivative q
means J=-D q, with q already differentiated along the outward normal.
The left and right q therefore have opposite signs for a linear field.

Global axes require at least 17 nodes so boundary closures do not overlap.
There is no minimum local extent: an accessor must supply every requested
solution/material node, using prepared halos if distributed. Only global
endpoint owners apply flux. Internal partition edges are not physical faces.
Tensor-product inventory uses the product of axis H weights; face area uses
all transverse weights. Periodic axes have uniform weights and wrapped
interior stencils. A condition on a periodic face is invalid.

Prepare state and diffusivity before evaluation through the existing stage
preparation service. Bind each prescribed time function to the actual
StageContext.time, not the beginning of the step. The kernel performs no
communication. FluxLedger is rank-local and transactional: begin an attempt,
accumulate actual stage quadrature, accept only the integrator's accepted
attempt, or reject. Callers explicitly reduce accepted totals. Negative RK
weights are allowed; nonfinite data and negative area are not.

## Qualification frozen before production evaluation

- Roundoff inventory residual <=1e-11 times the sum of absolute weighted RHS
  and imposed flux, with unit floor; same relative bound for symmetry.
- Zero-flux energy nonpositive within 1e-11 scaled roundoff.
- Linear/curved polynomial manufactured residual <=1e-8 at 33 nodes.
- Manufactured solution convergence: 32/64/128 intervals, final time .02,
  RK4 dt<=h^2/20; finest two global H-norm rates >=3.5; half-step error
  below 1% of spatial error. Constant and variable D and zero-flux decay.
- Constant/time-dependent flux inventory and accepted/rejected ledger tests.
- Real communication on 1/2/4 ranks, including a one-node local partition,
  must agree with the serial operator to 1e-11 scaled roundoff.
- Mixed periodic/physical axes, opposite normals, invalid faces, nonfinite
  state, zero-D/nonzero physical flux and invalid ledger lifecycle reject.
- Installed out-of-tree consumption is required before admission.

Passing the serial kernel does not close the distributed integration gate.
GPU support and nonlinear state-dependent reaction callbacks are deferred.

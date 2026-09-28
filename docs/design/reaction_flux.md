<!-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Local outer-face reaction contract

This CPU adapter evaluates a local law once per physical face and integration
stage. Inputs are immutable state, other local fields, position, Cartesian
outward unit normal, actual stage time, and application parameters. The
application supplies a state-admissibility predicate. A local implicit solve
must report success explicitly. Exceptions, invalid input or nonfinite rates
invalidate the active flux trial; it cannot be accepted afterward.

The returned constitutive FaceCondition contains that evaluated rate. Use it
both in DiffusionAxis and in FluxLedger.stage with the identical transverse
norm and stage quadrature. Only the physical edge owner evaluates it. Internal
partition faces are not reaction surfaces. The callback performs no collective
and must not mutate accepted state. Stage preparation remains the caller's
existing field/halo preparation. No implicit global integrator is introduced.

## Qualification fixed before evaluation

- Constant callback equals prescribed-flux RHS to roundoff (1e-11 scaled).
- Reversible exchange J=k(u-u_eq), on both faces, relaxes to u_eq. Compare
  the cosine Robin eigenmode with analytic exponential decay; H-norm error
  <=1e-4 and final distance from equilibrium <=1e-4.
- State/field/position/normal/time dependence is checked against a separately
  evaluated nonlinear expression; inventory equals minus face flux to 1e-11
  scaled roundoff. Linear-in-time stage quadrature integrates exactly.
- Failed admissibility, NaN, failed local solve and callback exception poison
  the trial. A rejected trial changes neither accepted integral nor state.
- Repeat the distributed physical ownership test on 1/2/4 ranks, including
  one-node local extent. Compare with serial and the integrated face oracle.
- Compile and execute a copied consumer against an installed package.

This qualifies a neutral callback, not any material reaction or validation.
Embedded interfaces, chemistry, GPU callbacks and mechanics remain separate.

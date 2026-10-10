<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Central 3D derivatives

`openpfc/kernel/field/central_derivatives.hpp` provides
`pfc::field::fd::evaluate<Order>(sample, hx, hy, hz)` for orders 2, 4 and 6.
It returns `Derivatives3D<T>` with `x,y,z,xx,yy,zz,xy,xz,yz`. The spacing type
sets the floating result type. The caller provides a const callable
`sample(dx,dy,dz)` returning the scalar at relative integer offsets from the
point of evaluation. This callable may read host or resident device storage;
its realization must be callable in the execution context that evaluates it.
The evaluator allocates and synchronizes nothing.

```cpp
struct Sample {
  const double* origin;
  int sy, sz;
  OPENPFC_HD double operator()(int dx, int dy, int dz) const {
    return origin[dx + dy*sy + dz*sz];
  }
};
auto derivatives = pfc::field::fd::evaluate<6>(sample, hx, hy, hz);
```

Here `origin` must point to an evaluation cell in storage with valid initialized
stencil neighbours. Positive finite spacings, finite samples and valid loads
are preconditions; they are not dynamically checked. Signed index arithmetic
must fit the accessor's index type. The evaluator uses existing central
first/second coefficient tables. Pure second derivatives use neighbour-minus-
centre differences. Mixed derivatives apply the tensor product of the two
first-derivative stencils, with normalization `D1²*h_axis*h_other`.
For a smooth function and compatible sampling the interior truncation error
is order `h^Order`, including all mixed Hessian entries. Unequal positive
spacings are supported.

The greatest offset in an individual axis is `Order/2`: 1, 2 or 3 cells.
Pure derivatives read axis neighbours; mixed derivatives additionally read
pairs of offsets in two axes. These need edge-filled ghost data for a padded
3D array. There is no automatic halo exchange or boundary-condition choice.
The helper does not enable mixed entries in the separate model-bound
`FDGradient<G>` evaluator, whose existing compatibility contract is unchanged.

Boundary accuracy depends on the extension and grid-coordinate convention.
For cell centres `x_i=(i+1/2)h` with a Neumann face at x=0, even reflection is
`u[-1]=u[0]`, `u[-2]=u[1]`, `u[-3]=u[2]`. The first derivative at the first
cell generally is not zero; the derivative at the boundary face is zero.
For a node at the boundary, `x_i=ih`, even reflection is `u[-i]=u[i]` and the
centred first derivative at node zero vanishes. These conventions differ.
Repeating the closest boundary cell for every out-of-range offset (clamping)
is another extension and does not imply fourth/sixth-order Neumann accuracy.
A reflected extension must also have the required smoothness; satisfying only
a zero first normal derivative does not guarantee all higher compatibility
conditions. Periodic samples or fully populated ghosts are other valid policies.

The caller also owns support-growth margins when using these stencils in a
transactional multi-field stepper: wider stencils can create positive support
farther from the previous support. A previous one-cell contact margin does
not automatically certify a three-cell stencil. No model, timestep stability,
physical anisotropy, ownership propagation or distributed implementation is
provided by this derivative helper.

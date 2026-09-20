<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Certified 128-grid trajectory

The movie contains all 70 manifest-indexed accepted snapshots, ending at state
1371, whose original convergence certificate completed the full verification
hold. Its final h field is byte-identical to h_final and has SHA256
213dfde9192eccddfb456cb507acef0325a79113ef4c06880ff1f63e72bdd5f5.
The decoded final MP4 frame was visually checked against the labelled terminal
PNG. Video compression is lossy; field admission uses original Float64 bytes.

[Animation](certified-trajectory.mp4) and [final frame](final.png) show only
the periodic h=0.5 isosurface, in physical cell64x64x121. No smoothing,
time-interpolated field or affine deformation is applied. The fixed camera,
field/PNG/movie checksums and accepted steps are recorded in `frames.json`.
The final image is held for two seconds. This is a trajectory visualization,
not a new convergence or material validation result.

`provenance.json` pins the renderer and the complete field-regeneration archive
from issue98. Decompress that archive's exact initial64 seed, recreate the128
interpolant, and replay its pinned producer protocol to regenerate omitted
fields. The archived render command then runs this renderer on the manifest;
only paths should change. `raw.json.gz` preserves the command, environment,
source, scheduler identity and render log. Python3.11, PyVista0.46.4,
VTK9.5.2 and ffmpeg were used; exact package versions are in the raw bundle.

Graphics-context job22182930 failed because MI250X is a compute chip. The
unchanged renderer succeeded under Mesa software rendering in job22182959
(accountproject_462001245,44seconds):

```sh
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
export MESA_LOADER_DRIVER_OVERRIDE=swrast
export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
python3.11 render_certified_trajectory.py \
  /fields/n0fixed128_manifest.json /new/render --ffmpeg /path/to/ffmpeg
```

The failed attempt is retained; no material state or threshold was changed to
make rendering succeed. The visualization environment is separate from the
solver and is not needed to replay or audit its numerical certificate.

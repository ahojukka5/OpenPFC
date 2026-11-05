<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Installed backend acceptance

The independent consumer in `tests/packaging/backend_acceptance` checks an
installed SDK on CPU, CUDA and HIP, each with 1, 2 and 4 MPI ranks. It contains
no solidification model or scientific acceptance claim. Use the compiler and
MPI recorded in the selected SDK manifest; all APIs and targets are installed.

```sh
cmake -S tests/packaging/backend_acceptance -B /path/to/consumer-build \
  -DCMAKE_PREFIX_PATH=/path/to/sdk -Dbackend=cpu
cmake --build /path/to/consumer-build --parallel 8
python3 -m pytest tests/packaging/backend_acceptance/test_matrix.py
python3 tests/packaging/backend_acceptance/run_matrix.py \
  --cpu /path/to/consumer-build/backend_acceptance \
  --output /path/to/results/cpu-acceptance
```

Build separate directories with `-Dbackend=cuda` or `hip`, using the matching
device compiler and architecture, then pass `--cuda` and `--hip` binaries.
The SDK must enable the requested device backend, device FFT and device-buffer
MPI; CPU requires distributed HeFFTe. Missing capabilities stop configuration
with `NOT_QUALIFIED`. Device cases never substitute host fields or host FFTs.

Use allocated resources and production partitions for matrix qualification.
The default launcher is `mpiexec -n`; for example,
`--launcher 'srun --mpi=pmi2' --count-flag=-n` selects a site launcher.
GPU ranks must see assigned devices through scheduler visibility variables.
A separate preflight verifies distinct device UUIDs per node and exchanges
actual device buffers over MPI. Unavailable preflight capabilities leave the
row unqualified. Missing binaries leave that backend's rows unqualified.

The runner retains launcher logs and `matrix.json`. Exit 0 requires all nine
rows; 77 denotes a partial matrix, and 1 a failed available case. Machine-specific
fragments can be reviewed together; a partial fragment is never an overall pass.
The existing script suite invokes the runner policy tests. The
`BackendAcceptance_InstalledCPU` CTest installs and relocates the built SDK,
builds the independent consumer and requires all three CPU rows to pass,
while retaining six unqualified GPU rows. It skips qualification for dirty or
unidentified SDK sources or an unavailable MPI launcher. No CI workflow is changed.
Every invocation keeps its own receipts. Timeouts retain partial launcher output
and terminate the invocation's process group. Direct and merged matrices reject
mixed compiled cases, and each backend must use one provider build across ranks.

## Fixed case and criteria

Two fields on a 12 × 10 × 8 grid with spacing (0.7, 0.9, 1.1) obey
`du/dt = lap(u) + 0.3(v-u)` and `dv/dt = 2 lap(v) + 0.3(u-v)`.
The x walls use cell-face even reflection and zero face current; y/z are
periodic. Both initial means are one. Mode amplitudes 0.2 and -0.1 multiply
`cos(pi(i+1/2)/12) sin(2pi*j/10) cos(2pi*k/8)`.

OpenPFC owns decomposition, padded fields, full halo transport, stage
preparation, reflection, face operators, vector algebra, reductions, field
bundles and FFTs. The consumer supplies the constitutive law and coordinates
eight Euler candidates with dt = 0.001. An independent 2 × 2 amplitude
recurrence uses the exact discrete Laplacian eigenvalue. It checks:

- Complete fields after every step and initial/final ghost faces, edges and
  corners: maximum absolute error below 2e-12.
- Closed boundary mass ledger and quadratic energy bounded by its initial value: absolute
  tolerance 3e-11. Criteria are identical across backends.
- A NaN proposed on one rank: collective rejection leaves accepted fields,
  prepared ghosts and increment exactly unchanged.
- Actual accepted fields written on one rank, reloaded on two/four ranks and
  evolved eight more steps; method, time, increment and full fields agree.
- Both evolved field patterns round-trip through the selected backend's FFT
  for every reduced axis: absolute error below 2e-12.

A zero exit alone cannot qualify a row. Receipts must identify every rank,
the requested compiled backend, distinct per-node GPU UUIDs, a clean SDK
revision, manifest SHA-256 and compiled consumer source-content SHA-256.

Per-rank wall time and allocated bytes of state/RHS/candidate/coefficient
fields are recorded. These are an allocation inventory, not peak memory or
a performance claim. Verification/checkpoint paths inspect host snapshots;
device RHS/vector algebra execute on the device, and device observables use
compact owned-cell reductions. Preserve logs with their source identities.

Combine machine-specific fragments with the same compiled case:

```sh
python3 tests/packaging/backend_acceptance/run_matrix.py \
  --merge /path/to/cpu/matrix.json /path/to/cuda/matrix.json /path/to/hip/matrix.json \
  --output /path/to/results/combined
```

The merger revalidates all passing receipts, requires the same compiled case
across backends and the same provider build within each backend's 1/2/4 rows,
and preserves supplied failures. Missing rows remain unqualified. It records
provider revisions per backend explicitly: a report qualifies those exact
builds and does not certify an unrecorded SDK or assume identical provider
revisions across platforms.

The SDK compatibility API certifies host/MPI evidence and intentionally reports
GPU runtime compatibility as unverified. Configuration keeps its required host
check, checks selected backend/FFT/device-MPI capabilities and exact recorded
architectures, and records the GPU API's unverified report. A GPU row can pass
only after the built device executable passes its assigned-device MPI preflight
and every runtime check. Compile-only success never qualifies a GPU row.

On Cray, the HIP driver cannot inherit MPI from the host compiler wrapper.
The consumer requests explicit FindMPI wrapper metadata for HIP; normal
MPI compile/link and SDK identity checks still apply. Use a fresh backend
build directory when changing compiler/MPI selections.

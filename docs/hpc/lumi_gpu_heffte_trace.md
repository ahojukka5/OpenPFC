<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# LUMI-G heFFTe phase tracing (diagnostic)

Diagnostic only. These timings do **not** replace admitted tungsten_hip
wall times. The production `heffte-rocm/2.4.1` module is built with
`Heffte_ENABLE_TRACING` undefined; `init_tracing` is a no-op there.

## Why a separate HeFFTe install

Stock heFFTe 2.4.1 tracing (`-DHeffte_ENABLE_TRACING=ON`) logs
`fft-1d` / `packing` / `all2all` / `unpacking` with `MPI_Wtime`.
`rocfft_execute` is asynchronous. Without a device wait, `fft-1d`
records enqueue time and the following packing `synchronize_device()`
absorbs the local transform. That would falsely support a
transpose-dominated explanation.

The diagnostic install therefore:

1. enables `Heffte_ENABLE_TRACING`;
2. applies `cmake/heffte-2.4.1-gpu-trace-sync.patch` so `add_trace`
   calls `hipDeviceSynchronize` at both ends of each interval.

It does **not** apply the production `p2p-plined-packall` patch; the
admitted LUMI-G module was built from vanilla 2.4.1 plus GPU-aware MPI.

Install prefix: `~/opt/heffte/2.4.1-rocm-trace`. Module:
`heffte-rocm-trace/2.4.1`. The `heffte-rocm` default module is not
updated.

## Build

From a GPU partition (not a login node):

```bash
sbatch docs/lumi_slurm/heffte_rocm_trace.sbatch
```

Then compile OpenPFC against that module in a **separate** build tree
(GPU node, not login):

```bash
sbatch docs/lumi_slurm/openpfc_heffte_trace_build.sbatch
```

or equivalently:

```bash
HEFFTE_MODULE=heffte-rocm-trace/2.4.1 \
BUILD_DIR=/flash/project_462001519/juaho/build/openpfc-lumi-rocm-heffte-trace \
./scripts/build.sh --machine=lumi --with-rocm --no-test --no-submit
```

The second form must not be run on a login node.

## Cases

`submit_tungsten_hip_trace.sh` submits the six 4-node / 32-GCD layouts
that discriminate local-FFT vs transpose after the matched-work slab
falsification:

| name | grid | process grid |
|------|------|----------------|
| `thip-tr-1200c` | 1200³ | 1×8×4 (admitted non-slab) |
| `thip-tr-1200x1152` | 1200×1200×1152 | 1×1×32 |
| `thip-tr-1200x1216` | 1200×1200×1216 | 1×1×32 |
| `thip-tr-1200x1280` | 1200×1200×1280 | 1×1×32 |
| `thip-tr-1152c` | 1152³ | 1×1×32 |
| `thip-tr-1216c` | 1216³ | 1×1×32 |

```bash
export TUNGSTEN_HIP_BIN=/flash/project_462001519/juaho/build/openpfc-lumi-rocm-heffte-trace/apps/tungsten/tungsten_hip
./docs/lumi_slurm/submit_tungsten_hip_trace.sh
python3 apps/tungsten/scripts/parse_heffte_trace.py \
  /scratch/project_462001519/juaho/openpfc-heffte-trace/runs
```

Set `OPENPFC_HEFFTE_TRACE` is handled by the sbatch (per-run
`heffte_trace_<rank>.log`). The extra device synchronizes change host
overlap; they do not change the numerical algorithm or process grid.

## What the 2× `waitany` count is

Whole-log `n_waitany` is 20 tungsten steps × rounds × 31 peers.
`n_steps_kept` (19) is only the timing-profile warmup skip. Do not
divide raw `waitany` by 19.

For `1×1×32` z-slab real inboxes with r2c along x, OpenPFC places the
complex outbox on **y-slabs** (`1×32×1`) when `Ny % 32 == 0`. HeFFTe
then finishes the z-FFT already in the outbox layout and skips the
pencils-back-to-z-slab reshape. `1152` and `1216` take that path
(4 all-peer rounds per step). `1200 % 32 != 0`, so the outbox stays a
z-slab and that extra transpose pair remains (8 rounds). The 2× count
is extra distributed reshape stages, not a logging artifact.

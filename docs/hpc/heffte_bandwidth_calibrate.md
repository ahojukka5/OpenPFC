<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# HeFFTe-trace MPI bandwidth vs OSU (issue #119)

Diagnostic LUMI-G calibration: is the Heat3D spectral extra wall at
8--32 nodes MPI-limited at the Slingshot injection cap, or is pack/sync
hiding a slower collective?

This page is the **recipe**. It does not replace the admitted production
Heat3D weak curve or the extra-wall envelope in the research report.
Do not mix those tables with this campaign.

Issue: [#119](https://github.com/ahojukka5/OpenPFC/issues/119). Research
[#589](https://github.com/ahojukka5/research/issues/589).

## Question

Measure MPI-only payload rate
\(B_{\mathrm{mpi}} =\) `bytes_per_timestep` / `T_mpi` from HeFFTe-trace
`all2all` / `all2allv` / `waitany`, and compare it to GPU-buffer OSU
`alltoall` / `alltoallv` at the same rank map and `bytes_per_peer`,
against the documented 12.5 GB/s unidirectional GCD NIC share.

Stop if 8--32 node HeFFTe-trace MPI rate is within a factor of about 2
of OSU at the same size. If it is far below OSU, that is packing/sync,
not the NIC. Either way do not restart 512-node Heat3D tournaments.

## Frozen factors

- Constant local inbox \(768^3\) / GCD, grow only \(N_z\).
- Diagnostic `heffte-rocm-trace/2.4.1` (device-sync tracing, no
  production `p2p-plined-packall` patch). Walls are **not** production
  numbers. Label `p2p_plined` rows as diagnostic.
- Heat3D 20 steps, warmup 1, I/O off, GPU-aware MPI, slabs.
- One protocol per `srun`. At 32 nodes run only `alltoall`.
- OSU 7.5 `--enable-rocm`, `-d rocm`, message size = `bytes_per_peer`.
- Account `project_462001519`. Partition `dev-g` (MaxNodes=32). Refuse
  `project_462001245`.

Bytes (comm-plan replica, `n_mpi=1`, 2 FFTs/step):

| nodes | ranks | `bytes_per_timestep` | `bytes_per_peer` | Heat3D | OSU |
|------:|------:|---------------------:|-----------------:|--------|-----|
| 1 | 8 | 7266631680 | 454164480 | `p2p_plined`, `alltoall` | `alltoall` |
| 2 | 16 | 7266631680 | 227082240 | `p2p_plined`, `alltoall` | `alltoall` |
| 8 | 64 | 7266631680 | 56770560 | `alltoall`, `alltoallv` | both |
| 32 | 256 | 7266631680 | 14192640 | `alltoall` | `alltoall` |

`B_osu = bytes_per_peer * (P-1) / latency` (excludes the local copy).

## Account and scratch

```text
/scratch/project_462001519/juaho/openpfc-scaling/heffte-bandwidth-calibrate/
```

Diagnostic Heat3D tree (not the tungsten 1200-axis trace install):

```text
/flash/project_462001519/juaho/build/openpfc-lumi-rocm-heffte-trace-heat3d/
```

OSU prefix:

```text
/flash/project_462001519/juaho/opt/osu-micro-benchmarks/7.5-rocm/
```

## Submit

From a LUMI login node. Builds go to `dev-g`; do not compile on login.

```bash
./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh check
./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh build
./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh osu-build
```

After both builds finish:

```bash
HEAT3D_DEPENDENCY=<build-job> OSU_DEPENDENCY=<osu-build-job> \
  ./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh 768
./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh collect
```

If the binaries already exist, omit the dependency variables.
Optional: `NODES=1,2`, `DRY_RUN=1`, `PARTITION=standard-g`.

`OPENPFC_HEFFTE_TRACE` is set by the sbatch. HeFFTe writes
`heffte_trace_<rank>.log`. Collect rebuilds CSV from run directories.
Do not commit raw traces.

## What this does not change

- Production `heffte-rocm` module.
- Production `p2p_plined` default.
- The tungsten 1200-axis trace install.
- 128+ node Heat3D jobs.

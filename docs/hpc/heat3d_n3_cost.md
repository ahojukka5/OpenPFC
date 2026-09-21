<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Heat3D 1-node GPU cost matrix (research #592)

Frozen LUMI-G equal-grid timings at \(N=512\) and \(N=1024\) for the
`periodic-operator-choice` \(N^3\) referee gate. Scientific owner:
[research#592](https://github.com/ahojukka5/research/issues/592). This
page is the executable recipe (OpenPFC #124).

Do **not** splice these numbers onto the historical `b91d2575` GPU table.
The fresh \(N=1024\) cells are the same-binary anchor for \(N=512\).

## Frozen protocol

| Knob | Value |
|------|--------|
| Partition | `standard-g` |
| Account | `project_462001519` |
| Nodes / GCDs | 1 / 8, one MPI rank per GCD |
| MPI | GPU-aware (`MPICH_GPU_SUPPORT_ENABLED=1`) |
| Spectral layout | production `use_pencils=false` slabs |
| Precision / I/O | double / off |
| `dt` | 0.01 |
| Steps | 30 CLI, `HEAT3D_WARMUP=5`, 25 accepted |
| Statistic | median barriered `wall_step` |
| Repeats | 3 independent 1-node allocations |
| Binary | one production HIP Release tree |

Cells: spectral, FD-2, FD-8, FD-12 at both sizes (24 runs). Not in the
matrix: \(N=768/1280/1536\), FD-4/6, CPU, multi-node, HeFFTe protocol
A/B.

Scratch:
`/scratch/project_462001519/juaho/openpfc-scaling/heat3d-n3-cost-592/`.
Build tree:
`/flash/project_462001519/juaho/build/openpfc-lumi-rocm-n3-cost-592`.

## Submit (LUMI login)

```bash
./docs/lumi_slurm/submit_heat3d_n3_cost.sh check
./docs/lumi_slurm/submit_heat3d_n3_cost.sh build
# after the HIP job finishes:
HEAT3D_SPECTRAL_HIP_BIN=/flash/project_462001519/juaho/build/openpfc-lumi-rocm-n3-cost-592/apps/heat3d/heat3d_spectral_hip \
HEAT3D_HIP_BIN=/flash/project_462001519/juaho/build/openpfc-lumi-rocm-n3-cost-592/apps/heat3d/heat3d_fd_hip \
  ./docs/lumi_slurm/submit_heat3d_n3_cost.sh submit
./docs/lumi_slurm/submit_heat3d_n3_cost.sh collect
```

`collect` writes `docs/report/data/heat3d_n3_cost_repeats.csv` and
`heat3d_n3_cost_summary.csv`. Ranking recompute belongs in research #592.

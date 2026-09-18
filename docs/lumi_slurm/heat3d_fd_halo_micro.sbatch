#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# Issue #25 halo-only control: examples/23_halo_microtiming --hip
# Required: HEAT3D_HALO_BIN, HEAT3D_NX, HEAT3D_NY, HEAT3D_NZ,
#           OPENPFC_FD_PROC_GRID

set -euo pipefail

: "${HEAT3D_HALO_BIN:?set HEAT3D_HALO_BIN to 23_halo_microtiming}"
: "${HEAT3D_NX:?}"
: "${HEAT3D_NY:?}"
: "${HEAT3D_NZ:?}"
: "${OPENPFC_FD_PROC_GRID:?set OPENPFC_FD_PROC_GRID=gx,gy,gz}"

NX="${HEAT3D_NX}"
NY="${HEAT3D_NY}"
NZ="${HEAT3D_NZ}"
ITERS="${HEAT3D_HALO_ITERS:-100}"
WARMUP="${HEAT3D_WARMUP:-5}"
JOBROOT="${HEAT3D_WEAK_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling/heat3d-fd-weak}"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES \
  OPENPFC_HIP_FORCE_PACKED_HALO OPENPFC_CUDA_FORCE_PACKED_HALO \
  OPENPFC_ASSUME_GPU_AWARE_MPI || true

NTASKS="${SLURM_NTASKS:?SLURM_NTASKS is not set}"
RUNDIR="${JOBROOT}/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${RUNDIR}"
cd "${RUNDIR}"
cp -f "${BASH_SOURCE[0]}" "${RUNDIR}/heat3d_fd_halo_micro.sbatch"

PER_NODE="${SLURM_NTASKS_PER_NODE:-${NTASKS}}"
SRUN_BIND=()
CPU_BIND="none"
if (( PER_NODE == 8 )) && [[ "${SLURM_JOB_PARTITION}" == "standard-g" ]]; then
  CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"
  SRUN_BIND=(--cpu-bind="${CPU_BIND}")
elif (( PER_NODE > 8 )); then
  echo "at most 8 ranks per node (one per GCD); ntasks-per-node=${PER_NODE}" >&2
  exit 1
fi

BIN_SHA=""
if command -v sha256sum >/dev/null 2>&1 && [[ -f "${HEAT3D_HALO_BIN}" ]]; then
  BIN_SHA="$(sha256sum "${HEAT3D_HALO_BIN}" | awk '{print $1}')"
fi
REV="${OPENPFC_REVISION:-}"
DIRTY="${OPENPFC_DIRTY:-}"
SRC="${OPENPFC_SRC:-}"

{
  echo "=== 23_halo_microtiming weak control (issue #25) ==="
  echo "job=${SLURM_JOB_ID}"
  echo "name=${SLURM_JOB_NAME}"
  echo "nodes=${SLURM_NNODES}"
  echo "ntasks=${NTASKS}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "bin=${HEAT3D_HALO_BIN}"
  echo "bin_sha256=${BIN_SHA}"
  echo "Nx=${NX}"
  echo "Ny=${NY}"
  echo "Nz=${NZ}"
  echo "iters=${ITERS}"
  echo "warmup=${WARMUP}"
  echo "OPENPFC_FD_PROC_GRID=${OPENPFC_FD_PROC_GRID}"
  echo "cpu-bind=${CPU_BIND}"
  echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
  echo "revision=${REV}"
  echo "dirty=${DIRTY}"
  echo "src=${SRC}"
  echo "cwd=${RUNDIR}"
  module list
} | tee "${RUNDIR}/run_meta.txt"

srun "${SRUN_BIND[@]}" \
  "${HEAT3D_HALO_BIN}" --hip \
  --nx "${NX}" --ny "${NY}" --nz "${NZ}" \
  --halo 1 --iters "${ITERS}" --warmup "${WARMUP}" \
  --output "${RUNDIR}/timing_profile.json" \
  2>&1 | tee "${RUNDIR}/run.log"

echo "=== done ==="

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
#SBATCH --cpus-per-task=7
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=h3-n3-r
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# OpenPFC #124: one independent repeat of the frozen
# 8-cell Heat3D GPU cost matrix on one LUMI-G node.

set -euo pipefail

: "${HEAT3D_SPECTRAL_HIP_BIN:?set HEAT3D_SPECTRAL_HIP_BIN}"
: "${HEAT3D_HIP_BIN:?set HEAT3D_HIP_BIN}"
REPEAT="${HEAT3D_REPEAT:?set HEAT3D_REPEAT to 1, 2, or 3}"
ROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling/heat3d-n3-cost-592}"
STEPS="${HEAT3D_STEPS:-30}"
WARMUP="${HEAT3D_WARMUP:-5}"
DT="${HEAT3D_DT:-0.01}"
ACCOUNT="${SLURM_JOB_ACCOUNT:-project_462001519}"

if [[ "${ACCOUNT}" != "project_462001519" ]]; then
  echo "refusing account ${ACCOUNT}" >&2
  exit 2
fi
if [[ "${REPEAT}" != "1" && "${REPEAT}" != "2" && "${REPEAT}" != "3" ]]; then
  echo "HEAT3D_REPEAT must be 1, 2, or 3" >&2
  exit 2
fi

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm/2.4.1 2>/dev/null || module load heffte-rocm || true

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES \
  HEAT3D_USE_PENCILS OPENPFC_FFT_PROC_GRID OPENPFC_FFT_NODE_GRID \
  OPENPFC_FFT_SLAB_AXIS OPENPFC_ASSUME_GPU_AWARE_MPI || true

CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"
SPEC_SHA="$(sha256sum "${HEAT3D_SPECTRAL_HIP_BIN}" | awk '{print $1}')"
FD_SHA="$(sha256sum "${HEAT3D_HIP_BIN}" | awk '{print $1}')"
HOST="$(hostname)"
REVISION="${OPENPFC_REVISION:-unknown}"
DIRTY="${OPENPFC_DIRTY:-unknown}"

mkdir -p "${ROOT}/runs"
{
  echo "=== heat3d n3 cost matrix repeat ${REPEAT} (OpenPFC #124) ==="
  echo "job=${SLURM_JOB_ID}"
  echo "account=${ACCOUNT}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "nodes=${SLURM_NNODES}"
  echo "ntasks=${SLURM_NTASKS}"
  echo "host=${HOST}"
  echo "revision=${REVISION}"
  echo "dirty=${DIRTY}"
  echo "spectral_bin=${HEAT3D_SPECTRAL_HIP_BIN}"
  echo "fd_bin=${HEAT3D_HIP_BIN}"
  echo "spectral_sha256=${SPEC_SHA}"
  echo "fd_sha256=${FD_SHA}"
  echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
  echo "HEFFTE_MODULE=${HEFFTE_MODULE:-heffte-rocm}"
  module list 2>&1 | sed 's/^/module: /' || true
} | tee "${ROOT}/repeat-${REPEAT}-${SLURM_JOB_ID}.meta"

export HEAT3D_WARMUP="${WARMUP}"

run_cell() {
  local n="$1" method="$2" order="$3"
  local tag
  if [[ "${method}" == "spectral" ]]; then
    tag="n${n}_spectral_r${REPEAT}"
  else
    tag="n${n}_fd${order}_r${REPEAT}"
  fi
  local rundir="${ROOT}/runs/${tag}_${SLURM_JOB_ID}"
  mkdir -p "${rundir}"
  local profile="${rundir}/timing_profile.json"
  local log="${rundir}/cell.log"
  export HEAT3D_PROFILE_JSON="${profile}"
  local bin_sha="${SPEC_SHA}"
  if [[ "${method}" != "spectral" ]]; then
    bin_sha="${FD_SHA}"
  fi
  {
    echo "=== cell ${tag} ==="
    echo "job=${SLURM_JOB_ID}"
    echo "tag=${tag}"
    echo "repeat=${REPEAT}"
    echo "N=${n}"
    echo "method=${method}"
    echo "fd_order=${order}"
    echo "account=${ACCOUNT}"
    echo "partition=${SLURM_JOB_PARTITION}"
    echo "nodes=${SLURM_NNODES}"
    echo "ntasks=${SLURM_NTASKS}"
    echo "host=${HOST}"
    echo "revision=${REVISION}"
    echo "dirty=${DIRTY}"
    echo "steps=${STEPS}"
    echo "warmup=${WARMUP}"
    echo "dt=${DT}"
    echo "bin_sha256=${bin_sha}"
    echo "gpu_aware=1"
    if [[ "${method}" == "spectral" ]]; then
      echo "use_pencils=0"
      echo "bin=${HEAT3D_SPECTRAL_HIP_BIN}"
    else
      echo "bin=${HEAT3D_HIP_BIN}"
    fi
    echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
  } | tee "${rundir}/run_meta.txt"

  if [[ "${method}" == "spectral" ]]; then
    srun --ntasks=8 --gpus-per-node=8 --cpus-per-task=7 \
      --cpu-bind="${CPU_BIND}" \
      "${HEAT3D_SPECTRAL_HIP_BIN}" "${n}" "${STEPS}" "${DT}" \
      | tee "${log}"
  else
    srun --ntasks=8 --gpus-per-node=8 --cpus-per-task=7 \
      --cpu-bind="${CPU_BIND}" \
      "${HEAT3D_HIP_BIN}" "${n}" "${STEPS}" "${DT}" "${order}" \
      | tee "${log}"
  fi
}

run_cell 512 spectral 0
run_cell 512 fd 2
run_cell 512 fd 8
run_cell 512 fd 12
run_cell 1024 spectral 0
run_cell 1024 fd 2
run_cell 1024 fd 8
run_cell 1024 fd 12

echo "=== repeat ${REPEAT} done ==="

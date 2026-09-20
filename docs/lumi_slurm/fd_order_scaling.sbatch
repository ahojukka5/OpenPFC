#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001245
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001245/juaho/logs/%x-%j.out
#
# Issue #108: FD-order × halo-overlap weak scaling on LUMI-G.
# Submit with docs/lumi_slurm/submit_fd_order_scaling.sh.
#
# Fail-closed: refuse any billed account other than project_462001245.
# Clean runs drop HEAT3D_DIAG_TIMING / HEAT3D_HALO_OVERLAP so production
# two-stream overlap remains. Diagnostic jobs set HEAT3D_KEEP_OVERRIDES=1
# and HEAT3D_DIAG_TIMING=1 only.

set -euo pipefail

ALLOWED_ACCOUNTS="project_462001245"

: "${HEAT3D_HIP_BIN:?set HEAT3D_HIP_BIN to heat3d_fd_hip}"
: "${HEAT3D_NX:?}"
: "${HEAT3D_NY:?}"
: "${HEAT3D_NZ:?}"
: "${OPENPFC_FD_PROC_GRID:?set OPENPFC_FD_PROC_GRID=gx,gy,gz}"

if [[ ! -x "${HEAT3D_HIP_BIN}" ]]; then
  echo "HEAT3D_HIP_BIN is not executable: ${HEAT3D_HIP_BIN}" >&2
  exit 1
fi

ACCOUNT_NOW="${SLURM_JOB_ACCOUNT:-}"
account_ok=0
for a in ${ALLOWED_ACCOUNTS}; do
  if [[ "${ACCOUNT_NOW}" == "${a}" ]]; then
    account_ok=1
    break
  fi
done
if [[ "${account_ok}" -ne 1 ]]; then
  echo "refusing #108 job billed to ${ACCOUNT_NOW:-missing} (allowed: ${ALLOWED_ACCOUNTS})" >&2
  exit 2
fi

if [[ "${HEAT3D_KEEP_OVERRIDES:-0}" != "1" ]]; then
  unset HEAT3D_HALO_OVERLAP HEAT3D_DIAG_TIMING || true
fi

NX="${HEAT3D_NX}"
NY="${HEAT3D_NY}"
NZ="${HEAT3D_NZ}"
STEPS="${HEAT3D_STEPS:-105}"
DT="${HEAT3D_DT:-0.01}"
FD_ORDER="${HEAT3D_FD_ORDER:-2}"
MODE="${OPENPFC_FD_MODE:-clean}"
REPEAT="${OPENPFC_REPEAT:-1}"
JOBROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001245/juaho/openpfc-scaling/fd-order-108}"

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
cp -f "${BASH_SOURCE[0]}" "${RUNDIR}/fd_order_scaling.sbatch"

export HEAT3D_PROFILE_JSON="${RUNDIR}/timing_profile.json"
export HEAT3D_WARMUP="${HEAT3D_WARMUP:-5}"
export HEAT3D_REQUIRE_INTERIOR="${HEAT3D_REQUIRE_INTERIOR:-256x256x256}"

PER_NODE="${SLURM_NTASKS_PER_NODE:-${NTASKS}}"
SRUN_BIND=()
CPU_BIND="none"
if (( PER_NODE == 8 )) && [[ "${SLURM_JOB_PARTITION}" == "standard-g" ]]; then
  CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"
  SRUN_BIND=(--cpu-bind="${CPU_BIND}")
elif (( PER_NODE > 8 )); then
  echo "at most 8 ranks per GCD; ntasks-per-node=${PER_NODE}" >&2
  exit 1
fi

BIN_SHA=""
if command -v sha256sum >/dev/null 2>&1; then
  BIN_SHA="$(sha256sum "${HEAT3D_HIP_BIN}" | awk '{print $1}')"
fi
REV="${OPENPFC_REVISION:-}"
DIRTY="${OPENPFC_DIRTY:-}"
SRC="${OPENPFC_SRC:-}"
if [[ -z "${REV}" ]] && command -v git >/dev/null 2>&1 && [[ -n "${SRC}" ]]; then
  if git -C "${SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    REV="$(git -C "${SRC}" rev-parse HEAD 2>/dev/null || true)"
    DIRTY="$(git -C "${SRC}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  fi
fi

halo_width=$((FD_ORDER / 2))
halo_face_bytes=$((6 * 256 * 256 * halo_width * 8))

{
  echo "=== fd-order scaling (issue #108) ==="
  echo "job=${SLURM_JOB_ID}"
  echo "name=${SLURM_JOB_NAME}"
  echo "account=${ACCOUNT_NOW:-unset}"
  echo "allowed_accounts=${ALLOWED_ACCOUNTS}"
  echo "nodes=${SLURM_NNODES}"
  echo "ntasks=${NTASKS}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "bin=${HEAT3D_HIP_BIN}"
  echo "bin_sha256=${BIN_SHA}"
  echo "Nx=${NX}"
  echo "Ny=${NY}"
  echo "Nz=${NZ}"
  echo "N=${NX}x${NY}x${NZ}"
  echo "steps=${STEPS}"
  echo "dt=${DT}"
  echo "fd_order=${FD_ORDER}"
  echo "halo_width=${halo_width}"
  echo "halo_face_bytes=${halo_face_bytes}"
  echo "mode=${MODE}"
  echo "repeat=${REPEAT}"
  echo "warmup=${HEAT3D_WARMUP}"
  echo "require_interior=${HEAT3D_REQUIRE_INTERIOR}"
  echo "OPENPFC_FD_PROC_GRID=${OPENPFC_FD_PROC_GRID}"
  echo "HEAT3D_DIAG_TIMING=${HEAT3D_DIAG_TIMING:-unset}"
  echo "HEAT3D_HALO_OVERLAP=${HEAT3D_HALO_OVERLAP:-unset}"
  echo "HEAT3D_KEEP_OVERRIDES=${HEAT3D_KEEP_OVERRIDES:-unset}"
  echo "cpu-bind=${CPU_BIND}"
  echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
  echo "revision=${REV}"
  echo "dirty=${DIRTY}"
  echo "src=${SRC}"
  echo "cwd=${RUNDIR}"
  module list
} | tee "${RUNDIR}/run_meta.txt"

set +e
# sbatch --export=NONE is inherited by srun unless overridden. Job
# 22186610 then lost LD_LIBRARY_PATH (libfabric.so.1) and died 127,
# the same class as HeFFTe tournament job 22175215.
srun --export=ALL "${SRUN_BIND[@]}" \
  "${HEAT3D_HIP_BIN}" "${NX}" "${NY}" "${NZ}" "${STEPS}" "${DT}" "${FD_ORDER}" \
  2>&1 | tee "${RUNDIR}/run.log"
SRUN_RC="${PIPESTATUS[0]}"
set -e

if [[ -f fd_placement.txt ]]; then
  cp -f fd_placement.txt "${RUNDIR}/fd_placement.txt"
fi

ADMIT="ok"
REASON="none"
if (( SRUN_RC != 0 )); then
  ADMIT="reject"
  REASON="srun_exit_${SRUN_RC}"
elif ! grep -q 'HEAT3D_FD_DECOMP' "${RUNDIR}/run.log"; then
  ADMIT="reject"
  REASON="missing_decomp_banner"
elif grep -qiE 'HEAT3D_HIP_CHECKSUM .*nan|HEAT3D_HIP_CHECKSUM .*inf' "${RUNDIR}/run.log"; then
  ADMIT="reject"
  REASON="checksum_nonfinite"
fi

{
  echo "admit=${ADMIT}"
  echo "reason=${REASON}"
  echo "srun_rc=${SRUN_RC}"
} | tee "${RUNDIR}/admit.txt"

echo "=== fd-order scaling done rc=${SRUN_RC} admit=${ADMIT} ==="
if [[ "${ADMIT}" != "ok" ]]; then
  exit 1
fi
exit 0

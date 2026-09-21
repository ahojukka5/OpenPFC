#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# Issue #119: one Heat3D-trace protocol or one GPU OSU collective per job.
# Diagnostic walls. Do not mix with production Heat3D tables.
# Submit with docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh.
#
# Required: RUN_KIND=heat3d|osu
# heat3d: HEAT3D_SPECTRAL_HIP_BIN, HEAT3D_NX/NY/NZ, HEAT3D_RESHAPE_ALG
# osu:    OSU_BIN, OSU_MSG_SIZE, OSU_KIND
#
# Refuse if billed to project_462001245.

set -euo pipefail

ALLOWED_ACCOUNTS="project_462001519"
ALLOWED_ALGS="p2p_plined p2p alltoallv alltoall"
ACCOUNT_NOW="${SLURM_JOB_ACCOUNT:-}"
account_ok=0
for a in ${ALLOWED_ACCOUNTS}; do
  if [[ "${ACCOUNT_NOW}" == "${a}" ]]; then
    account_ok=1
    break
  fi
done
if [[ "${account_ok}" -ne 1 ]]; then
  echo "refusing bandwidth job billed to ${ACCOUNT_NOW:-missing} (allowed: ${ALLOWED_ACCOUNTS})" >&2
  exit 2
fi

KIND="${RUN_KIND:-heat3d}"
JOBROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling/heffte-bandwidth-calibrate}"
NTASKS="${SLURM_NTASKS:?SLURM_NTASKS is not set}"
RUNDIR="${JOBROOT}/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${RUNDIR}"
cd "${RUNDIR}"
cp -f "${BASH_SOURCE[0]}" "${RUNDIR}/heffte_bandwidth_calibrate.sbatch"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
if [[ "${KIND}" == "heat3d" ]]; then
  module load heffte-rocm-trace/2.4.1
fi

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true

PER_NODE="${SLURM_NTASKS_PER_NODE:-${NTASKS}}"
SRUN_BIND=()
CPU_BIND="none"
# Exclusive standard-g nodes own the CCD CPU masks. dest-g / small-g
# 8-GPU jobs can get a thin CPU set; srun then rejects the mask.
if (( PER_NODE == 8 )) && [[ "${SLURM_JOB_PARTITION}" == "standard-g" ]]; then
  CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"
  SRUN_BIND=(--cpu-bind="${CPU_BIND}")
elif (( PER_NODE > 8 )); then
  echo "at most 8 ranks per node (one per GCD); ntasks-per-node=${PER_NODE}" >&2
  exit 1
fi

BIN_SHA=""
REV="${OPENPFC_REVISION:-}"
DIRTY="${OPENPFC_DIRTY:-}"
SRC="${OPENPFC_SRC:-}"

banner_field() {
  local log="$1"
  local key="$2"
  awk -v k="${key}" '
    $1 == "HEAT3D_SPECTRAL_HIP" {
      for (i = 2; i <= NF; i++) {
        if ($i ~ ("^" k "=")) {
          split($i, a, "=")
          print a[2]
          exit
        }
      }
    }
  ' "${log}"
}

if [[ "${KIND}" == "heat3d" ]]; then
  : "${HEAT3D_SPECTRAL_HIP_BIN:?set HEAT3D_SPECTRAL_HIP_BIN}"
  : "${HEAT3D_NX:?}"
  : "${HEAT3D_NY:?}"
  : "${HEAT3D_NZ:?}"
  : "${HEAT3D_RESHAPE_ALG:?}"
  ALG="${HEAT3D_RESHAPE_ALG}"
  alg_ok=0
  for x in ${ALLOWED_ALGS}; do
    if [[ "${x}" == "${ALG}" ]]; then
      alg_ok=1
      break
    fi
  done
  if [[ "${alg_ok}" -ne 1 ]]; then
    echo "unknown reshape algorithm: ${ALG}" >&2
    exit 2
  fi
  if [[ ! -x "${HEAT3D_SPECTRAL_HIP_BIN}" ]]; then
    echo "HEAT3D_SPECTRAL_HIP_BIN is not executable: ${HEAT3D_SPECTRAL_HIP_BIN}" >&2
    exit 1
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    BIN_SHA="$(sha256sum "${HEAT3D_SPECTRAL_HIP_BIN}" | awk '{print $1}')"
  fi
  unset HEAT3D_USE_PENCILS HEAT3D_USE_REORDER HEAT3D_GPU_AWARE \
    HEAT3D_DIAG_TIMING HEAT3D_KEEP_OVERRIDES \
    OPENPFC_FFT_PROC_GRID OPENPFC_FFT_NODE_GRID OPENPFC_FFT_SLAB_AXIS \
    OPENPFC_FFT_COMPLEX_PROC_GRID OPENPFC_FFT_COMPLEX_OUTBOX || true
  STEPS="${HEAT3D_STEPS:-20}"
  DT="${HEAT3D_DT:-0.01}"
  WARMUP="${HEAT3D_WARMUP:-1}"
  export HEAT3D_RESHAPE_ALG="${ALG}"
  export HEAT3D_PROFILE_JSON="${RUNDIR}/timing_profile.json"
  export HEAT3D_WARMUP="${WARMUP}"
  export OPENPFC_HEFFTE_TRACE="${RUNDIR}/heffte_trace"
  {
    echo "=== heffte bandwidth calibrate (issue #119) ==="
    echo "kind=heat3d_trace"
    echo "job=${SLURM_JOB_ID}"
    echo "name=${SLURM_JOB_NAME}"
    echo "account=${ACCOUNT_NOW}"
    echo "nodes=${SLURM_NNODES}"
    echo "ntasks=${NTASKS}"
    echo "partition=${SLURM_JOB_PARTITION}"
    echo "bin=${HEAT3D_SPECTRAL_HIP_BIN}"
    echo "bin_sha256=${BIN_SHA}"
    echo "Nx=${HEAT3D_NX}"
    echo "Ny=${HEAT3D_NY}"
    echo "Nz=${HEAT3D_NZ}"
    echo "steps=${STEPS}"
    echo "dt=${DT}"
    echo "warmup=${WARMUP}"
    echo "reshape=${ALG}"
    echo "cpu-bind=${CPU_BIND}"
    echo "OPENPFC_HEFFTE_TRACE=${OPENPFC_HEFFTE_TRACE}"
    echo "HEFFTE_ROOT=${HEFFTE_ROOT:-unset}"
    echo "revision=${REV}"
    echo "dirty=${DIRTY}"
    echo "src=${SRC}"
    echo "cwd=${RUNDIR}"
    module list
  } | tee "${RUNDIR}/run_meta.txt"

  set +e
  srun --export=ALL "${SRUN_BIND[@]}" \
    "${HEAT3D_SPECTRAL_HIP_BIN}" \
    "${HEAT3D_NX}" "${HEAT3D_NY}" "${HEAT3D_NZ}" "${STEPS}" "${DT}" \
    2>&1 | tee "${RUNDIR}/run.log"
  SRUN_RC="${PIPESTATUS[0]}"
  set -e

  BANNER_ALG="$(banner_field "${RUNDIR}/run.log" reshape)"
  BANNER_PENCILS="$(banner_field "${RUNDIR}/run.log" use_pencils)"
  BANNER_AWARE="$(banner_field "${RUNDIR}/run.log" gpu_aware)"
  NTRACE="$(find "${RUNDIR}" -maxdepth 1 -name 'heffte_trace_*.log' | wc -l | tr -d ' ')"
  ADMIT="ok"
  REASON="none"
  if (( SRUN_RC != 0 )); then
    ADMIT="reject"
    REASON="srun_exit_${SRUN_RC}"
  elif [[ -z "${BANNER_ALG}" ]]; then
    ADMIT="reject"
    REASON="missing_reshape_banner"
  elif [[ "${BANNER_ALG}" != "${ALG}" ]]; then
    ADMIT="reject"
    REASON="banner_reshape_${BANNER_ALG}_expected_${ALG}"
  elif [[ "${BANNER_PENCILS}" != "0" ]]; then
    ADMIT="reject"
    REASON="banner_use_pencils_${BANNER_PENCILS}"
  elif [[ "${BANNER_AWARE}" != "1" ]]; then
    ADMIT="reject"
    REASON="banner_gpu_aware_${BANNER_AWARE}"
  elif (( NTRACE < 1 )); then
    ADMIT="reject"
    REASON="missing_trace"
  fi
  {
    echo "admit=${ADMIT}"
    echo "reason=${REASON}"
    echo "srun_rc=${SRUN_RC}"
    echo "banner_reshape=${BANNER_ALG:-missing}"
    echo "banner_use_pencils=${BANNER_PENCILS:-missing}"
    echo "banner_gpu_aware=${BANNER_AWARE:-missing}"
    echo "expected_reshape=${ALG}"
    echo "n_trace_logs=${NTRACE}"
  } | tee "${RUNDIR}/admit.txt"
  echo "=== heat3d trace done admit=${ADMIT} ==="
  if [[ "${ADMIT}" != "ok" ]]; then
    exit 1
  fi
  exit 0
fi

if [[ "${KIND}" != "osu" ]]; then
  echo "RUN_KIND must be heat3d or osu (got ${KIND})" >&2
  exit 2
fi

: "${OSU_BIN:?set OSU_BIN}"
: "${OSU_MSG_SIZE:?}"
OSU_KIND="${OSU_KIND:-alltoall}"
if [[ ! -x "${OSU_BIN}" ]]; then
  echo "OSU_BIN is not executable: ${OSU_BIN}" >&2
  exit 1
fi
if command -v sha256sum >/dev/null 2>&1; then
  BIN_SHA="$(sha256sum "${OSU_BIN}" | awk '{print $1}')"
fi
ITERS="${OSU_ITERS:-20}"
WARM="${OSU_WARMUP:-5}"
{
  echo "=== heffte bandwidth calibrate (issue #119) ==="
  echo "kind=osu"
  echo "job=${SLURM_JOB_ID}"
  echo "name=${SLURM_JOB_NAME}"
  echo "account=${ACCOUNT_NOW}"
  echo "nodes=${SLURM_NNODES}"
  echo "ntasks=${NTASKS}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "bin=${OSU_BIN}"
  echo "bin_sha256=${BIN_SHA}"
  echo "osu_kind=${OSU_KIND}"
  echo "reshape=${OSU_KIND}"
  echo "osu_msg_size=${OSU_MSG_SIZE}"
  echo "cpu-bind=${CPU_BIND}"
  echo "revision=${REV}"
  echo "dirty=${DIRTY}"
  echo "src=${SRC}"
  echo "cwd=${RUNDIR}"
  module list
} | tee "${RUNDIR}/run_meta.txt"

set +e
srun --export=ALL "${SRUN_BIND[@]}" \
  "${OSU_BIN}" -d rocm -x "${WARM}" -i "${ITERS}" \
  -m "${OSU_MSG_SIZE}:${OSU_MSG_SIZE}" \
  2>&1 | tee "${RUNDIR}/osu.out"
SRUN_RC="${PIPESTATUS[0]}"
set -e
if (( SRUN_RC != 0 )); then
  echo "admit=reject" | tee "${RUNDIR}/admit.txt"
  echo "reason=srun_exit_${SRUN_RC}" | tee -a "${RUNDIR}/admit.txt"
  exit 1
fi
echo "admit=ok" | tee "${RUNDIR}/admit.txt"
echo "reason=none" | tee -a "${RUNDIR}/admit.txt"
echo "=== osu done ==="

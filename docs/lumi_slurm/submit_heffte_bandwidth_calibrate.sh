#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Issue #119: submit HeFFTe-trace vs GPU OSU alltoall on LUMI-G.
# Bill project_462001519. Refuse project_462001245.
#
# Usage (from a LUMI login node):
#   ./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh check
#   ./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh build
#   ./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh osu-build
#   HEAT3D_SPECTRAL_HIP_BIN=... OSU_ALLTOALL_BIN=... \
#     ./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh 768
#   ./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh pencils
#   ./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh collect
#
# Optional: ACCOUNT, PARTITION, NODES, DRY_RUN=1, OPENPFC_SCALING_ROOT,
#           HEAT3D_DEPENDENCY, OSU_DEPENDENCY (Slurm job ids).
# GPU-aware MPI is always on. pencils = issue #121 32-node A/B.

set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/heffte_bandwidth_calibrate.sbatch"
BUILD_SBATCH="${SCRIPT_DIR}/openpfc_heffte_trace_heat3d_build.sbatch"
OSU_BUILD_SBATCH="${SCRIPT_DIR}/osu_rocm_build.sbatch"
PY="${SCRIPT_DIR}/../../apps/heat3d/scripts/heffte_bandwidth_calibrate.py"

ALLOWED_ACCOUNTS="project_462001519"
ACCOUNT="${ACCOUNT:-project_462001519}"
PARTITION="${PARTITION:-standard-g}"
CAMPAIGN_ROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling/heffte-bandwidth-calibrate}"
LOGDIR="${OPENPFC_LOG_DIR:-/scratch/project_462001519/juaho/logs}"
DRY_RUN="${DRY_RUN:-0}"
NODES_FILTER="${NODES:-}"
DEFAULT_H3D_BIN="/flash/project_462001519/juaho/build/openpfc-lumi-rocm-heffte-trace-heat3d/apps/heat3d/heat3d_spectral_hip"
DEFAULT_OSU_PREFIX="/flash/project_462001519/juaho/opt/osu-micro-benchmarks/7.5-rocm"
DEFAULT_OSU_COLL="${DEFAULT_OSU_PREFIX}/libexec/osu-micro-benchmarks/mpi/collective"
# Login shells leak the production HIP tree (jobs 22205703/704 wrote no
# traces). Keep only a binary whose path names the diagnostic install.
if [[ -n "${HEAT3D_SPECTRAL_HIP_BIN:-}" &&
      "${HEAT3D_SPECTRAL_HIP_BIN}" != *heffte-trace* ]]; then
  echo "ignoring leaked HEAT3D_SPECTRAL_HIP_BIN=${HEAT3D_SPECTRAL_HIP_BIN}" >&2
  unset HEAT3D_SPECTRAL_HIP_BIN
fi
HEAT3D_SPECTRAL_HIP_BIN="${HEAT3D_SPECTRAL_HIP_BIN:-${DEFAULT_H3D_BIN}}"
OSU_ALLTOALL_BIN="${OSU_ALLTOALL_BIN:-${DEFAULT_OSU_COLL}/osu_alltoall}"
OSU_ALLTOALLV_BIN="${OSU_ALLTOALLV_BIN:-${DEFAULT_OSU_COLL}/osu_alltoallv}"

account_allowed() {
  local a="$1" x
  for x in ${ALLOWED_ACCOUNTS}; do
    if [[ "${x}" == "${a}" ]]; then
      return 0
    fi
  done
  return 1
}
if ! account_allowed "${ACCOUNT}"; then
  echo "refusing ACCOUNT=${ACCOUNT}; allowed: ${ALLOWED_ACCOUNTS}" >&2
  exit 2
fi

case "${MODE}" in
  check|build|osu-build|768|pencils|collect) ;;
  *)
    echo "usage: $0 check|build|osu-build|768|pencils|collect" >&2
    exit 1
    ;;
esac

export OPENPFC_SCALING_ROOT="${CAMPAIGN_ROOT}"

if [[ "${MODE}" == "check" ]]; then
  python3 "${PY}" --check
  exit 0
fi
if [[ "${MODE}" == "collect" ]]; then
  python3 "${PY}" --harvest "${CAMPAIGN_ROOT}" \
    --out "${CAMPAIGN_ROOT}/results"
  exit 0
fi

SRC="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${OPENPFC_SRC:-}" ]]; then
  leaked="$(cd "${OPENPFC_SRC}" 2>/dev/null && pwd || true)"
  if [[ -n "${leaked}" && "${leaked}" != "${SRC}" ]]; then
    echo "ignoring leaked OPENPFC_SRC=${OPENPFC_SRC}; using ${SRC}" >&2
  fi
fi
export OPENPFC_SRC="${SRC}"
if command -v git >/dev/null 2>&1 &&
   git -C "${SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  export OPENPFC_REVISION="$(git -C "${SRC}" rev-parse HEAD)"
  export OPENPFC_DIRTY="$(git -C "${SRC}" status --porcelain | wc -l | tr -d ' ')"
fi
export OPENPFC_REVISION="${OPENPFC_REVISION:-unknown}"
export OPENPFC_DIRTY="${OPENPFC_DIRTY:-unknown}"

# Login shells may export SBATCH_ACCOUNT=project_462001245; drop it so
# the explicit --account=project_462001519 cannot be overridden.
unset HEAT3D_USE_PENCILS HEAT3D_USE_REORDER HEAT3D_GPU_AWARE \
  HEAT3D_DIAG_TIMING HEAT3D_KEEP_OVERRIDES HEAT3D_PROTOCOLS \
  OPENPFC_FFT_PROC_GRID OPENPFC_FFT_NODE_GRID OPENPFC_FFT_SLAB_AXIS \
  OPENPFC_FFT_COMPLEX_PROC_GRID OPENPFC_FFT_COMPLEX_OUTBOX \
  SBATCH_ACCOUNT SLURM_ACCOUNT SBATCH_PARTITION SLURM_PARTITION || true

if [[ "${MODE}" == "build" ]]; then
  echo "submit OpenPFC heffte-rocm-trace Heat3D build account=${ACCOUNT}"
  extra=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --job-name="openpfc-heffte-trace-h3d"
    --output="${LOGDIR}/%x-%j.out"
    --error="${LOGDIR}/%x-%j.err"
    --export="NONE,OPENPFC_SRC=${SRC},BUILD_DIR=/flash/project_462001519/juaho/build/openpfc-lumi-rocm-heffte-trace-heat3d"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo sbatch "${extra[@]}" "${BUILD_SBATCH}"
    exit 0
  fi
  mkdir -p "${LOGDIR}"
  sbatch "${extra[@]}" "${BUILD_SBATCH}"
  exit 0
fi

if [[ "${MODE}" == "osu-build" ]]; then
  echo "submit OSU ROCm build account=${ACCOUNT}"
  extra=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --job-name="osu-rocm-build"
    --output="${LOGDIR}/%x-%j.out"
    --error="${LOGDIR}/%x-%j.err"
    --export="NONE"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo sbatch "${extra[@]}" "${OSU_BUILD_SBATCH}"
    exit 0
  fi
  mkdir -p "${LOGDIR}"
  sbatch "${extra[@]}" "${OSU_BUILD_SBATCH}"
  exit 0
fi

want_nodes() {
  local n="$1"
  if [[ -z "${NODES_FILTER}" ]]; then
    return 0
  fi
  local tok
  IFS=',' read -ra toks <<< "${NODES_FILTER}"
  for tok in "${toks[@]}"; do
    if [[ "${tok}" == "${n}" ]]; then
      return 0
    fi
  done
  return 1
}

walltime_for_nodes() {
  local nodes="$1"
  if (( nodes <= 2 )); then
    echo "00:20:00"
  elif (( nodes <= 8 )); then
    echo "00:30:00"
  else
    echo "00:45:00"
  fi
}

bytes_peer() {
  case "$1" in
    1) echo 454164480 ;;
    2) echo 227082240 ;;
    8) echo 56770560 ;;
    32) echo 14192640 ;;
    *) echo 0 ;;
  esac
}

submit_heat3d() {
  local nodes="$1"
  local proto="$2"
  local nx="$3"
  local ny="$4"
  local nz="$5"
  local pencils="${6:-0}"
  local issue="${7:-119}"
  local ntasks=$((nodes * 8))
  local job_name="h3dbw-768-${nodes}n-${proto}"
  if [[ "${pencils}" == "1" ]]; then
    job_name="${job_name}-pencils"
  fi
  local time_lim
  time_lim="$(walltime_for_nodes "${nodes}")"
  local extra=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --nodes="${nodes}"
    --ntasks="${ntasks}"
    --ntasks-per-node=8
    --gpus-per-node=8
    --mem=0
    --time="${time_lim}"
    --job-name="${job_name}"
    --output="${LOGDIR}/%x-%j.out"
    --export="NONE,RUN_KIND=heat3d,HEAT3D_SPECTRAL_HIP_BIN=${HEAT3D_SPECTRAL_HIP_BIN},HEAT3D_NX=${nx},HEAT3D_NY=${ny},HEAT3D_NZ=${nz},HEAT3D_RESHAPE_ALG=${proto},HEAT3D_USE_PENCILS=${pencils},HEAT3D_GPU_AWARE=1,HEAT3D_STEPS=20,HEAT3D_WARMUP=1,HEAT3D_DT=0.01,OPENPFC_ISSUE=${issue},OPENPFC_SCALING_ROOT=${CAMPAIGN_ROOT},OPENPFC_REVISION=${OPENPFC_REVISION},OPENPFC_DIRTY=${OPENPFC_DIRTY},OPENPFC_SRC=${OPENPFC_SRC}"
  )
  if [[ -n "${HEAT3D_DEPENDENCY:-}" ]]; then
    extra+=(--dependency="afterok:${HEAT3D_DEPENDENCY}")
  fi
  echo "submit heat3d nodes=${nodes} ranks=${ntasks} ${nx}x${ny}x${nz} reshape=${proto} pencils=${pencils} gpu_aware=1 account=${ACCOUNT} partition=${PARTITION}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo sbatch "${extra[@]}" "${SBATCH}"
    return 0
  fi
  mkdir -p "${LOGDIR}" "${CAMPAIGN_ROOT}/runs" "${CAMPAIGN_ROOT}/results"
  sbatch "${extra[@]}" "${SBATCH}"
}

submit_osu() {
  local nodes="$1"
  local kind="$2"
  local ntasks=$((nodes * 8))
  local job_name="osubw-768-${nodes}n-${kind}"
  local time_lim
  time_lim="$(walltime_for_nodes "${nodes}")"
  local bin="${OSU_ALLTOALL_BIN}"
  if [[ "${kind}" == "alltoallv" ]]; then
    bin="${OSU_ALLTOALLV_BIN}"
  fi
  local msg
  msg="$(bytes_peer "${nodes}")"
  local extra=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --nodes="${nodes}"
    --ntasks="${ntasks}"
    --ntasks-per-node=8
    --gpus-per-node=8
    --mem=0
    --time="${time_lim}"
    --job-name="${job_name}"
    --output="${LOGDIR}/%x-%j.out"
    --export="NONE,RUN_KIND=osu,OSU_BIN=${bin},OSU_KIND=${kind},OSU_MSG_SIZE=${msg},OPENPFC_SCALING_ROOT=${CAMPAIGN_ROOT},OPENPFC_REVISION=${OPENPFC_REVISION},OPENPFC_DIRTY=${OPENPFC_DIRTY},OPENPFC_SRC=${OPENPFC_SRC}"
  )
  if [[ -n "${OSU_DEPENDENCY:-}" ]]; then
    extra+=(--dependency="afterok:${OSU_DEPENDENCY}")
  fi
  echo "submit osu nodes=${nodes} ranks=${ntasks} kind=${kind} msg=${msg} account=${ACCOUNT} partition=${PARTITION}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo sbatch "${extra[@]}" "${SBATCH}"
    return 0
  fi
  mkdir -p "${LOGDIR}" "${CAMPAIGN_ROOT}/runs" "${CAMPAIGN_ROOT}/results"
  sbatch "${extra[@]}" "${SBATCH}"
}

need_osu=1
if [[ "${MODE}" == "pencils" ]]; then
  need_osu=0
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -x "${HEAT3D_SPECTRAL_HIP_BIN}" && -z "${HEAT3D_DEPENDENCY:-}" ]]; then
    echo "HEAT3D_SPECTRAL_HIP_BIN is not executable: ${HEAT3D_SPECTRAL_HIP_BIN}" >&2
    echo "submit build first, or set HEAT3D_DEPENDENCY" >&2
    exit 1
  fi
  if [[ "${need_osu}" == "1" && ! -x "${OSU_ALLTOALL_BIN}" && -z "${OSU_DEPENDENCY:-}" ]]; then
    echo "OSU_ALLTOALL_BIN is not executable: ${OSU_ALLTOALL_BIN}" >&2
    echo "submit osu-build first, or set OSU_DEPENDENCY" >&2
    exit 1
  fi
fi

if [[ "${MODE}" == "pencils" ]]; then
  echo "Issue #121 32-node pencil A/B. account=${ACCOUNT} rev=${OPENPFC_REVISION} dirty=${OPENPFC_DIRTY}"
  echo "root=${CAMPAIGN_ROOT}"
  echo "gpu_aware=1 HEAT3D_USE_PENCILS=1 vs admitted slab 22207838"
  submit_heat3d 32 alltoall 768 768 196608 1 121
  echo "Logs: ${LOGDIR}/"
  echo "Runs: ${CAMPAIGN_ROOT}/runs/"
  exit 0
fi

echo "Issue #119 bandwidth 768. account=${ACCOUNT} rev=${OPENPFC_REVISION} dirty=${OPENPFC_DIRTY}"
echo "root=${CAMPAIGN_ROOT}"

python3 "${PY}" --ladder | while read -r nodes nx ny nz ranks heat3d osu bytes_peer; do
  [[ "${nodes}" == "nodes" ]] && continue
  want_nodes "${nodes}" || continue
  for proto in ${heat3d//,/ }; do
    submit_heat3d "${nodes}" "${proto}" "${nx}" "${ny}" "${nz}"
  done
  for kind in ${osu//,/ }; do
    submit_osu "${nodes}" "${kind}"
  done
done

echo "Logs: ${LOGDIR}/"
echo "Runs: ${CAMPAIGN_ROOT}/runs/"

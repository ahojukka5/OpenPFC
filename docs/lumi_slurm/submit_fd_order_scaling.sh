#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Issue #108: FD-order × large-scale halo-overlap scaling on LUMI-G.
# Do not submit against project_462001519. Do not mix with #106.
#
# Usage:
#   HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip \
#     ./submit_fd_order_scaling.sh check
#   HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip \
#     ./submit_fd_order_scaling.sh clean
#   HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip \
#     ./submit_fd_order_scaling.sh diag
#   ./submit_fd_order_scaling.sh collect
#   ./submit_fd_order_scaling.sh analyze
#   ./submit_fd_order_scaling.sh harvest
#   ./submit_fd_order_scaling.sh geometry
#
# Optional: ACCOUNT, PARTITION, NODES, ORDERS, REPEATS, DRY_RUN=1,
#           OPENPFC_SCALING_ROOT

set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/fd_order_scaling.sbatch"
PY="${SCRIPT_DIR}/../../apps/heat3d/scripts/fd_order_scaling.py"

ALLOWED_ACCOUNTS="project_462001245"
ACCOUNT="${ACCOUNT:-project_462001245}"
PARTITION="${PARTITION:-standard-g}"
CAMPAIGN_ROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001245/juaho/openpfc-scaling/fd-order-108}"
LOGDIR="${OPENPFC_LOG_DIR:-/scratch/project_462001245/juaho/logs}"
DRY_RUN="${DRY_RUN:-0}"
REPEATS="${REPEATS:-}"
NODES_FILTER="${NODES:-}"
ORDERS_FILTER="${ORDERS:-}"

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
  check|clean|diag|collect|analyze|harvest|geometry) ;;
  *)
    echo "usage: $0 check|clean|diag|collect|analyze|harvest|geometry" >&2
    exit 1
    ;;
esac

export OPENPFC_SCALING_ROOT="${CAMPAIGN_ROOT}"

if [[ "${MODE}" == "check" ]]; then
  python3 "${PY}" --check
  exit 0
fi
if [[ "${MODE}" == "collect" ]]; then
  python3 "${PY}" --collect "${CAMPAIGN_ROOT}" \
    --out "${CAMPAIGN_ROOT}/results/runs.csv"
  exit 0
fi
if [[ "${MODE}" == "analyze" ]]; then
  python3 "${PY}" --analyze "${CAMPAIGN_ROOT}/results/runs.csv" \
    --out "${CAMPAIGN_ROOT}/results"
  exit 0
fi
if [[ "${MODE}" == "harvest" ]]; then
  python3 "${PY}" --harvest "${CAMPAIGN_ROOT}" \
    --out "${CAMPAIGN_ROOT}/results"
  exit 0
fi
if [[ "${MODE}" == "geometry" ]]; then
  python3 "${PY}" --geometry \
    --out "${SCRIPT_DIR}/../hpc/fd_order_geometry.csv"
  exit 0
fi

: "${HEAT3D_HIP_BIN:?set HEAT3D_HIP_BIN to heat3d_fd_hip}"
if [[ ! -x "${HEAT3D_HIP_BIN}" ]]; then
  echo "HEAT3D_HIP_BIN is not executable: ${HEAT3D_HIP_BIN}" >&2
  exit 1
fi

SRC="${OPENPFC_SRC:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export OPENPFC_SRC="${SRC}"
if command -v git >/dev/null 2>&1 &&
   git -C "${SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  export OPENPFC_REVISION="$(git -C "${SRC}" rev-parse HEAD)"
  export OPENPFC_DIRTY="$(git -C "${SRC}" status --porcelain | wc -l | tr -d ' ')"
fi
if [[ -z "${OPENPFC_REVISION:-}" ]]; then
  echo "OPENPFC_REVISION is empty; set OPENPFC_SRC" >&2
  exit 2
fi
export OPENPFC_DIRTY="${OPENPFC_DIRTY:-unknown}"
export HEAT3D_STEPS="${HEAT3D_STEPS:-105}"
export HEAT3D_WARMUP="${HEAT3D_WARMUP:-5}"
export HEAT3D_DT="${HEAT3D_DT:-0.01}"
export HEAT3D_REQUIRE_INTERIOR="${HEAT3D_REQUIRE_INTERIOR:-256x256x256}"
export HEAT3D_HIP_BIN

unset OPENPFC_HIP_FORCE_PACKED_HALO OPENPFC_CUDA_FORCE_PACKED_HALO \
  OPENPFC_ASSUME_GPU_AWARE_MPI OPENPFC_FFT_PROC_GRID OPENPFC_FFT_NODE_GRID \
  OPENPFC_FFT_SLAB_AXIS HEAT3D_HALO_OVERLAP \
  SBATCH_ACCOUNT SLURM_ACCOUNT SBATCH_PARTITION SLURM_PARTITION || true

if [[ "${MODE}" == "diag" ]]; then
  export HEAT3D_DIAG_TIMING=1
  export HEAT3D_KEEP_OVERRIDES=1
  PREFIX="h3d108d"
else
  unset HEAT3D_DIAG_TIMING HEAT3D_KEEP_OVERRIDES || true
  PREFIX="h3d108c"
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

want_order() {
  local o="$1"
  if [[ -z "${ORDERS_FILTER}" ]]; then
    return 0
  fi
  local tok
  IFS=',' read -ra toks <<< "${ORDERS_FILTER}"
  for tok in "${toks[@]}"; do
    if [[ "${tok}" == "${o}" ]]; then
      return 0
    fi
  done
  return 1
}

repeats_for() {
  local nodes="$1"
  if [[ -n "${REPEATS}" ]]; then
    echo "${REPEATS}"
    return
  fi
  if (( nodes == 128 || nodes == 512 || nodes == 1024 )); then
    echo 3
  else
    echo 1
  fi
}

walltime_for_nodes() {
  local nodes="$1"
  if (( nodes <= 32 )); then
    echo "00:30:00"
  elif (( nodes <= 128 )); then
    echo "01:00:00"
  else
    echo "01:30:00"
  fi
}

submit_one() {
  local order="$1"
  local nodes="$2"
  local nx="$3"
  local ny="$4"
  local nz="$5"
  local grid="$6"
  local repeat="$7"
  local ntasks=$((nodes * 8))
  local job_name="${PREFIX}-fd${order}-${nodes}n-r${repeat}"
  local time_lim
  time_lim="$(walltime_for_nodes "${nodes}")"
  local keep="${HEAT3D_KEEP_OVERRIDES:-0}"
  local diag="${HEAT3D_DIAG_TIMING:-}"
  local export_list="NONE"
  export_list+=",HEAT3D_HIP_BIN=${HEAT3D_HIP_BIN}"
  export_list+=",HEAT3D_STEPS=${HEAT3D_STEPS}"
  export_list+=",HEAT3D_WARMUP=${HEAT3D_WARMUP}"
  export_list+=",HEAT3D_DT=${HEAT3D_DT}"
  export_list+=",HEAT3D_FD_ORDER=${order}"
  export_list+=",HEAT3D_REQUIRE_INTERIOR=${HEAT3D_REQUIRE_INTERIOR}"
  export_list+=",HEAT3D_NX=${nx}"
  export_list+=",HEAT3D_NY=${ny}"
  export_list+=",HEAT3D_NZ=${nz}"
  export_list+=",OPENPFC_FD_PROC_GRID=${grid}"
  export_list+=",OPENPFC_SCALING_ROOT=${CAMPAIGN_ROOT}"
  export_list+=",OPENPFC_REVISION=${OPENPFC_REVISION}"
  export_list+=",OPENPFC_DIRTY=${OPENPFC_DIRTY}"
  export_list+=",OPENPFC_SRC=${OPENPFC_SRC}"
  export_list+=",OPENPFC_REPEAT=${repeat}"
  export_list+=",OPENPFC_FD_MODE=${MODE}"
  if [[ "${keep}" == "1" ]]; then
    export_list+=",HEAT3D_KEEP_OVERRIDES=1"
  fi
  if [[ -n "${diag}" ]]; then
    export_list+=",HEAT3D_DIAG_TIMING=${diag}"
  fi
  echo "submit mode=${MODE} fd_order=${order} nodes=${nodes} ${nx}x${ny}x${nz} grid=${grid} repeat=${repeat} account=${ACCOUNT}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo sbatch --account="${ACCOUNT}" --partition="${PARTITION}" \
      --nodes="${nodes}" --ntasks="${ntasks}" --ntasks-per-node=8 \
      --gpus-per-node=8 --mem=0 --time="${time_lim}" \
      --job-name="${job_name}" --output="${LOGDIR}/%x-%j.out" \
      --export="${export_list}" "${SBATCH}"
    return 0
  fi
  mkdir -p "${LOGDIR}" "${CAMPAIGN_ROOT}/runs" "${CAMPAIGN_ROOT}/results"
  sbatch \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes="${nodes}" \
    --ntasks="${ntasks}" \
    --ntasks-per-node=8 \
    --gpus-per-node=8 \
    --mem=0 \
    --time="${time_lim}" \
    --job-name="${job_name}" \
    --output="${LOGDIR}/%x-%j.out" \
    --export="${export_list}" \
    "${SBATCH}"
}

echo "Issue #108 FD-order ${MODE}. account=${ACCOUNT} rev=${OPENPFC_REVISION} dirty=${OPENPFC_DIRTY}"
echo "root=${CAMPAIGN_ROOT}"

python3 "${PY}" --ladder | while read -r nodes nx ny nz grid ranks; do
  [[ "${nodes}" == "nodes" ]] && continue
  want_nodes "${nodes}" || continue
  if [[ "${MODE}" == "diag" ]]; then
    case "${nodes}" in
      8|128|1024) ;;
      *) continue ;;
    esac
  fi
  for order in 2 4 8 12 20; do
    want_order "${order}" || continue
    nrep=1
    if [[ "${MODE}" == "clean" ]]; then
      nrep="$(repeats_for "${nodes}")"
    fi
    r=1
    while (( r <= nrep )); do
      submit_one "${order}" "${nodes}" "${nx}" "${ny}" "${nz}" "${grid}" "${r}"
      r=$((r + 1))
    done
  done
done

echo "Logs: ${LOGDIR}/"
echo "Runs: ${CAMPAIGN_ROOT}/runs/"

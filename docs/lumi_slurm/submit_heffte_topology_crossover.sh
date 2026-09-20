#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Issue #106: HeFFTe protocol crossovers and LUMI-G topology regimes.
# Do not submit against project_462001519. Do not mix with the #61 tree.
#
# Usage (from a LUMI login node):
#   HEAT3D_SPECTRAL_HIP_BIN=/path/to/heat3d_spectral_hip \
#     ./submit_heffte_topology_crossover.sh check
#   HEAT3D_SPECTRAL_HIP_BIN=/path/to/heat3d_spectral_hip \
#     ./submit_heffte_topology_crossover.sh wave1
#   ./submit_heffte_topology_crossover.sh collect
#   ./submit_heffte_topology_crossover.sh analyze
#
# Optional: ACCOUNT, PARTITION, NODES (comma list), REPEATS, DRY_RUN=1,
#           OPENPFC_SCALING_ROOT, HEAT3D_STEPS, HEAT3D_WARMUP, HEAT3D_DT

set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/heffte_topology_crossover.sbatch"
PY="${SCRIPT_DIR}/../../apps/heat3d/scripts/heffte_topology_crossover.py"

ALLOWED_ACCOUNTS="project_462001245"
ACCOUNT="${ACCOUNT:-project_462001245}"
PARTITION="${PARTITION:-standard-g}"
CAMPAIGN_ROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001245/juaho/openpfc-scaling/heffte-topology-106}"
LOGDIR="${OPENPFC_LOG_DIR:-/scratch/project_462001245/juaho/logs}"
DRY_RUN="${DRY_RUN:-0}"
REPEATS="${REPEATS:-3}"
NODES_FILTER="${NODES:-}"
FAMILY="${HEAT3D_FAMILY:-768}"

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
  check|wave1|collect|analyze) ;;
  *)
    echo "usage: $0 check|wave1|collect|analyze" >&2
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

: "${HEAT3D_SPECTRAL_HIP_BIN:?set HEAT3D_SPECTRAL_HIP_BIN to heat3d_spectral_hip}"
if [[ ! -x "${HEAT3D_SPECTRAL_HIP_BIN}" ]]; then
  echo "HEAT3D_SPECTRAL_HIP_BIN is not executable: ${HEAT3D_SPECTRAL_HIP_BIN}" >&2
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
export HEAT3D_STEPS="${HEAT3D_STEPS:-20}"
export HEAT3D_WARMUP="${HEAT3D_WARMUP:-1}"
export HEAT3D_DT="${HEAT3D_DT:-0.01}"
export HEAT3D_SPECTRAL_HIP_BIN

unset HEAT3D_USE_PENCILS HEAT3D_RESHAPE_ALG HEAT3D_USE_REORDER \
  HEAT3D_GPU_AWARE HEAT3D_DIAG_TIMING HEAT3D_KEEP_OVERRIDES \
  OPENPFC_FFT_PROC_GRID OPENPFC_FFT_NODE_GRID OPENPFC_FFT_SLAB_AXIS \
  OPENPFC_FFT_COMPLEX_PROC_GRID OPENPFC_FFT_COMPLEX_OUTBOX \
  SBATCH_ACCOUNT SLURM_ACCOUNT SBATCH_PARTITION SLURM_PARTITION || true

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
  if (( nodes <= 136 )); then
    echo "02:00:00"
  elif (( nodes <= 264 )); then
    echo "03:00:00"
  else
    echo "04:00:00"
  fi
}

submit_one() {
  local family="$1"
  local nodes="$2"
  local nx="$3"
  local ny="$4"
  local nz="$5"
  local repeat="$6"
  local seed="$7"
  local protocols="$8"
  local ntasks=$((nodes * 8))
  local job_name="h3d106-${family}-${nodes}n-r${repeat}"
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
    --export="NONE,HEAT3D_SPECTRAL_HIP_BIN=${HEAT3D_SPECTRAL_HIP_BIN},HEAT3D_STEPS=${HEAT3D_STEPS},HEAT3D_WARMUP=${HEAT3D_WARMUP},HEAT3D_DT=${HEAT3D_DT},HEAT3D_PROTOCOLS=${protocols},HEAT3D_PROTOCOL_SEED=${seed},OPENPFC_SCALING_ROOT=${CAMPAIGN_ROOT},OPENPFC_REVISION=${OPENPFC_REVISION},OPENPFC_DIRTY=${OPENPFC_DIRTY},OPENPFC_SRC=${OPENPFC_SRC},OPENPFC_FAMILY=${family},OPENPFC_REPEAT=${repeat},HEAT3D_NX=${nx},HEAT3D_NY=${ny},HEAT3D_NZ=${nz}"
  )
  echo "submit family=${family} nodes=${nodes} ranks=${ntasks} ${nx}x${ny}x${nz} repeat=${repeat} seed=${seed} order=${protocols} account=${ACCOUNT}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo sbatch "${extra[@]}" "${SBATCH}"
    return 0
  fi
  mkdir -p "${LOGDIR}" "${CAMPAIGN_ROOT}/runs" "${CAMPAIGN_ROOT}/results"
  sbatch "${extra[@]}" "${SBATCH}"
}

echo "Issue #106 topology crossover ${MODE}. account=${ACCOUNT} rev=${OPENPFC_REVISION} dirty=${OPENPFC_DIRTY}"
echo "root=${CAMPAIGN_ROOT}"

python3 "${PY}" --wave1 | while read -r nodes nx ny nz ranks; do
  [[ "${nodes}" == "nodes" ]] && continue
  want_nodes "${nodes}" || continue
  r=1
  while (( r <= REPEATS )); do
    order_line="$(python3 "${PY}" --order --family "${FAMILY}" --nodes "${nodes}" --repeat "${r}")"
    seed="${order_line#seed=}"
    seed="${seed%% *}"
    protocols="${order_line#*protocols=}"
    submit_one "${FAMILY}" "${nodes}" "${nx}" "${ny}" "${nz}" "${r}" "${seed}" "${protocols}"
    r=$((r + 1))
  done
done

echo "Logs: ${LOGDIR}/"
echo "Runs: ${CAMPAIGN_ROOT}/runs/"

#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Issue #61: submit the HeFFTe reshape-algorithm tournament on LUMI-G.
# Do not submit against project_462001519. Do not inherit leaked HEAT3D_*
# or OPENPFC_FFT_* variables (job 22166458).
#
# Usage (from a LUMI login node):
#   HEAT3D_SPECTRAL_HIP_BIN=/path/to/heat3d_spectral_hip \
#     ./submit_heffte_protocol_tournament.sh check
#   HEAT3D_SPECTRAL_HIP_BIN=/path/to/heat3d_spectral_hip \
#     ./submit_heffte_protocol_tournament.sh 768
#   HEAT3D_SPECTRAL_HIP_BIN=/path/to/heat3d_spectral_hip \
#     ./submit_heffte_protocol_tournament.sh 512
#   HEAT3D_SPECTRAL_HIP_BIN=/path/to/heat3d_spectral_hip \
#     ./submit_heffte_protocol_tournament.sh 1gcd
#   ./submit_heffte_protocol_tournament.sh collect
#   ./submit_heffte_protocol_tournament.sh analyze
#
# Optional: ACCOUNT, PARTITION, NODES (comma list), REPEATS, DRY_RUN=1,
#           OPENPFC_SCALING_ROOT, HEAT3D_STEPS, HEAT3D_WARMUP, HEAT3D_DT,
#           HEAT3D_PROTOCOLS

set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/heffte_protocol_tournament.sbatch"
PY="${SCRIPT_DIR}/../../apps/heat3d/scripts/heffte_protocol_tournament.py"

# The campaign allocation is fixed; fair-share pressure does not authorize
# substituting another account.
ALLOWED_ACCOUNTS="project_462001245"
ACCOUNT="${ACCOUNT:-project_462001245}"
PARTITION="${PARTITION:-standard-g}"
CAMPAIGN_ROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001245/juaho/openpfc-scaling/heffte-protocol-tournament}"
LOGDIR="${OPENPFC_LOG_DIR:-/scratch/project_462001245/juaho/logs}"
DRY_RUN="${DRY_RUN:-0}"
REPEATS="${REPEATS:-}"
NODES_FILTER="${NODES:-}"

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
  check|768|512|1gcd|collect|analyze) ;;
  *)
    echo "usage: $0 check|768|512|1gcd|collect|analyze" >&2
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
# Colon-separated: sbatch --export parses commas as variable separators
# (job 22175215 ran only p2p_plined).
export HEAT3D_PROTOCOLS="${HEAT3D_PROTOCOLS:-p2p_plined:p2p:alltoallv:alltoall}"
export HEAT3D_SPECTRAL_HIP_BIN

# Fail closed: never export leaked A/B knobs into the job.
# Login shells may export SBATCH_ACCOUNT=project_462001519; drop it so
# the explicit --account=project_462001245 cannot be overridden.
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
  if (( nodes <= 8 )); then
    echo "01:00:00"
  elif (( nodes <= 32 )); then
    echo "01:30:00"
  elif (( nodes <= 128 )); then
    echo "02:00:00"
  elif (( nodes <= 256 )); then
    echo "03:00:00"
  else
    echo "04:00:00"
  fi
}

repeats_for_nodes() {
  local nodes="$1"
  if [[ -n "${REPEATS}" ]]; then
    echo "${REPEATS}"
    return
  fi
  if (( nodes >= 32 )); then
    echo 2
  else
    echo 1
  fi
}

EXPORT_LIST="NONE"
EXPORT_LIST+=",HEAT3D_SPECTRAL_HIP_BIN=${HEAT3D_SPECTRAL_HIP_BIN}"
EXPORT_LIST+=",HEAT3D_STEPS=${HEAT3D_STEPS}"
EXPORT_LIST+=",HEAT3D_WARMUP=${HEAT3D_WARMUP}"
EXPORT_LIST+=",HEAT3D_DT=${HEAT3D_DT}"
EXPORT_LIST+=",HEAT3D_PROTOCOLS=${HEAT3D_PROTOCOLS}"
EXPORT_LIST+=",OPENPFC_SCALING_ROOT=${CAMPAIGN_ROOT}"
EXPORT_LIST+=",OPENPFC_REVISION=${OPENPFC_REVISION}"
EXPORT_LIST+=",OPENPFC_DIRTY=${OPENPFC_DIRTY}"
EXPORT_LIST+=",OPENPFC_SRC=${OPENPFC_SRC}"

submit_one() {
  local family="$1"
  local nodes="$2"
  local per_node="$3"
  local nx="$4"
  local ny="$5"
  local nz="$6"
  local repeat="$7"
  local ntasks=$((nodes * per_node))
  local gpus="${per_node}"
  local job_name="h3dpt-${family}-${nodes}n-r${repeat}"
  if (( per_node == 1 && nodes == 1 )); then
    job_name="h3dpt-${family}-1gcd-r${repeat}"
  fi
  local time_lim
  time_lim="$(walltime_for_nodes "${nodes}")"
  local mem_opt=(--mem=0)
  local part="${PARTITION}"
  if (( per_node < 8 )); then
    part="${PARTITION_SMALL:-small-g}"
    mem_opt=(--mem="$((per_node * 64))G")
    time_lim="00:30:00"
  fi
  local extra=(
    --account="${ACCOUNT}"
    --partition="${part}"
    --nodes="${nodes}"
    --ntasks="${ntasks}"
    --ntasks-per-node="${per_node}"
    --gpus-per-node="${gpus}"
    "${mem_opt[@]}"
    --time="${time_lim}"
    --job-name="${job_name}"
    --output="${LOGDIR}/%x-%j.out"
    --export="${EXPORT_LIST},HEAT3D_NX=${nx},HEAT3D_NY=${ny},HEAT3D_NZ=${nz},OPENPFC_FAMILY=${family}"
  )
  echo "submit family=${family} nodes=${nodes} ranks=${ntasks} ${nx}x${ny}x${nz} repeat=${repeat} account=${ACCOUNT} partition=${part}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo sbatch "${extra[@]}" "${SBATCH}"
    return 0
  fi
  mkdir -p "${LOGDIR}" "${CAMPAIGN_ROOT}/runs" "${CAMPAIGN_ROOT}/results"
  sbatch "${extra[@]}" "${SBATCH}"
}

echo "Issue #61 tournament ${MODE}. account=${ACCOUNT} rev=${OPENPFC_REVISION} dirty=${OPENPFC_DIRTY}"
echo "root=${CAMPAIGN_ROOT}"

if [[ "${MODE}" == "1gcd" ]]; then
  submit_one 768 1 1 768 768 768 1
  submit_one 512 1 1 512 512 512 1
  exit 0
fi

FAMILY="${MODE}"
LOCAL="${FAMILY}"
python3 "${PY}" --ladder "${FAMILY}" | while read -r nodes nx ny nz ranks; do
  [[ "${nodes}" == "nodes" ]] && continue
  want_nodes "${nodes}" || continue
  nrep="$(repeats_for_nodes "${nodes}")"
  r=1
  while (( r <= nrep )); do
    submit_one "${FAMILY}" "${nodes}" 8 "${nx}" "${ny}" "${nz}" "${r}"
    r=$((r + 1))
  done
done

echo "Logs: ${LOGDIR}/"
echo "Runs: ${CAMPAIGN_ROOT}/runs/"

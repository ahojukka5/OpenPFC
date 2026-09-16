#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Issue #13: tungsten_hip weak-scaling ladder on LUMI-G.
# Submit from a LUMI login node. Do not run the jobs on the login node.
#
# Usage:
#   TUNGSTEN_HIP_BIN=/path/to/tungsten_hip ./submit_tungsten_hip_flagship.sh check
#   TUNGSTEN_HIP_BIN=/path/to/tungsten_hip ./submit_tungsten_hip_flagship.sh pilot
#   TUNGSTEN_HIP_BIN=/path/to/tungsten_hip ./submit_tungsten_hip_flagship.sh weak
#   TUNGSTEN_HIP_BIN=/path/to/tungsten_hip ./submit_tungsten_hip_flagship.sh strong
#   FLAGSHIP_ALLOW_60=1 TUNGSTEN_HIP_BIN=... ./submit_tungsten_hip_flagship.sh max
#
# Frozen protocol: tungsten_hip_scaling.toml (I/O off, dt=1).
# Multi-node uses OPENPFC_FFT_NODE_GRID=1 (1x8xnnodes). The 60-node
# point is gated until the 32-node ladder is healthy.

set -euo pipefail

: "${TUNGSTEN_HIP_BIN:?set TUNGSTEN_HIP_BIN to the 0.2 tungsten_hip binary}"
if [[ ! -x "${TUNGSTEN_HIP_BIN}" ]]; then
  echo "TUNGSTEN_HIP_BIN is not executable: ${TUNGSTEN_HIP_BIN}" >&2
  exit 1
fi

MODE="${1:-}"
if [[ "${MODE}" != "check" && "${MODE}" != "pilot" && "${MODE}" != "weak" &&
      "${MODE}" != "strong" && "${MODE}" != "max" ]]; then
  echo "usage: $0 check|pilot|weak|strong|max" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/tungsten_hip_scaling.sbatch"
TEMPLATE="${SCRIPT_DIR}/tungsten_hip_scaling.toml"
PARTITION="${PARTITION:-standard-g}"
ACCOUNT="${ACCOUNT:-project_462001519}"
STEPS="${TUNGSTEN_STEPS:-20}"
export TUNGSTEN_HIP_BIN
export TUNGSTEN_STEPS="${STEPS}"
export TUNGSTEN_SCALING_TEMPLATE="${TEMPLATE}"
export OPENPFC_SCALING_ROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling}"
export OPENPFC_FFT_NODE_GRID="${OPENPFC_FFT_NODE_GRID:-1}"

# nodes  gcds  N     time
# 1      8     768   01:00:00
# 2      16    960
# 4      32    1200
# 8      64    1536
# 16     128   1920
# 32     256   2400
# 60     480   3000  (gated)

submit_one() {
  local nodes="$1"
  local n="$2"
  local time_lim="$3"
  local job_name="$4"
  local per_node=8
  local ntasks=$((nodes * per_node))
  export TUNGSTEN_LX="${n}"
  sbatch \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes="${nodes}" \
    --ntasks="${ntasks}" \
    --ntasks-per-node="${per_node}" \
    --gpus-per-node="${per_node}" \
    --time="${time_lim}" \
    --job-name="${job_name}" \
    --export=ALL \
    "${SBATCH}"
}

case "${MODE}" in
  check)
    python3 "${SCRIPT_DIR}/../../apps/tungsten/scripts/flagship_ladder.py" --check
    ;;
  pilot)
    echo "1-node 768^3 pilot (8 GCDs, I/O off, ${STEPS} steps)"
    submit_one 1 768 "01:00:00" "thip-flag-1n-768"
    ;;
  weak)
    echo "Weak ladder 1/2/4/8/16/32 nodes (not 60). NODE_GRID=${OPENPFC_FFT_NODE_GRID}"
    submit_one 1 768 "01:00:00" "thip-flag-1n-768"
    submit_one 2 960 "01:00:00" "thip-flag-2n-960"
    submit_one 4 1200 "01:00:00" "thip-flag-4n-1200"
    submit_one 8 1536 "02:00:00" "thip-flag-8n-1536"
    submit_one 16 1920 "02:00:00" "thip-flag-16n-1920"
    submit_one 32 2400 "02:00:00" "thip-flag-32n-2400"
    ;;
  strong)
    LX="${TUNGSTEN_LX:-768}"
    echo "Strong-scaling control ${LX}^3 at 1/2/4 nodes (8 GCD/node)"
    submit_one 1 "${LX}" "01:00:00" "thip-flag-strong-1n-lx${LX}"
    submit_one 2 "${LX}" "01:00:00" "thip-flag-strong-2n-lx${LX}"
    submit_one 4 "${LX}" "01:00:00" "thip-flag-strong-4n-lx${LX}"
    ;;
  max)
    if [[ "${FLAGSHIP_ALLOW_60:-0}" != "1" ]]; then
      echo "60-node / 3000^3 is gated. Set FLAGSHIP_ALLOW_60=1 after the" >&2
      echo "32-node ladder is healthy and memory/decomposition are checked." >&2
      exit 2
    fi
    echo "60-node 3000^3 (480 GCD). NODE_GRID=${OPENPFC_FFT_NODE_GRID}"
    submit_one 60 3000 "04:00:00" "thip-flag-60n-3000"
    ;;
esac

echo "Logs: /scratch/project_462001519/juaho/logs/"
echo "Runs: ${OPENPFC_SCALING_ROOT}/runs/"
echo "Summarise into docs/report/data/tungsten_lumi_g_flagship.csv"
echo "See docs/hpc/lumi_gpu_flagship.md"

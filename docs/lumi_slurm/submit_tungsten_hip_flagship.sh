#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Issue #13: tungsten_hip weak-scaling ladder on LUMI-G.
# Submit from a LUMI login node. Do not run the jobs on the login node.
#
# Usage:
#   ./submit_tungsten_hip_flagship.sh check
#   TUNGSTEN_HIP_BIN=/path/to/tungsten_hip ./submit_tungsten_hip_flagship.sh pilot
#   TUNGSTEN_HIP_BIN=/path/to/tungsten_hip ./submit_tungsten_hip_flagship.sh weak
#   TUNGSTEN_HIP_BIN=/path/to/tungsten_hip ./submit_tungsten_hip_flagship.sh strong
#   TUNGSTEN_HIP_BIN=/path/to/tungsten_hip ./submit_tungsten_hip_flagship.sh control
#   FLAGSHIP_ALLOW_60=1 TUNGSTEN_HIP_BIN=... ./submit_tungsten_hip_flagship.sh max
#
# Frozen protocol: tungsten_hip_scaling.toml (I/O off, dt=1).
# Multi-node weak/max uses OPENPFC_FFT_NODE_GRID=1 (1x8xnnodes). The
# 60-node point is gated until the 32-node ladder is healthy.
# `control` is the issue #13 matched-decomposition / process-grid set.

set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "check" && "${MODE}" != "pilot" && "${MODE}" != "weak" &&
      "${MODE}" != "strong" && "${MODE}" != "max" && "${MODE}" != "collect" &&
      "${MODE}" != "control" ]]; then
  echo "usage: $0 check|pilot|weak|strong|max|control|collect" >&2
  exit 1
fi

if [[ "${MODE}" != "check" && "${MODE}" != "collect" ]]; then
  : "${TUNGSTEN_HIP_BIN:?set TUNGSTEN_HIP_BIN to the 0.2 tungsten_hip binary}"
  if [[ ! -x "${TUNGSTEN_HIP_BIN}" ]]; then
    echo "TUNGSTEN_HIP_BIN is not executable: ${TUNGSTEN_HIP_BIN}" >&2
    exit 1
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/tungsten_hip_scaling.sbatch"
TEMPLATE="${SCRIPT_DIR}/tungsten_hip_scaling.toml"
PARTITION="${PARTITION:-standard-g}"
ACCOUNT="${ACCOUNT:-project_462001519}"
STEPS="${TUNGSTEN_STEPS:-20}"
if [[ -n "${TUNGSTEN_HIP_BIN:-}" ]]; then
  export TUNGSTEN_HIP_BIN
fi
export TUNGSTEN_STEPS="${STEPS}"
export TUNGSTEN_SCALING_TEMPLATE="${TEMPLATE}"
export OPENPFC_SCALING_ROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling}"

# Capture git on the login node. Compute nodes often have no `git` after
# `module purge`, so revision/dirty must travel in the job environment.
# Git worktrees have `.git` as a file, not a directory.
SRC="${OPENPFC_SRC:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export OPENPFC_SRC="${SRC}"
if command -v git >/dev/null 2>&1 &&
   git -C "${SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  export OPENPFC_REVISION="$(git -C "${SRC}" rev-parse HEAD)"
  export OPENPFC_DIRTY="$(git -C "${SRC}" status --porcelain | wc -l | tr -d ' ')"
fi
if [[ "${MODE}" != "check" && "${MODE}" != "collect" ]]; then
  if [[ -z "${OPENPFC_REVISION:-}" ]]; then
    echo "OPENPFC_REVISION is empty; refusing to submit without provenance" >&2
    echo "Set OPENPFC_SRC to a git checkout (worktrees are OK) or export" >&2
    echo "OPENPFC_REVISION / OPENPFC_DIRTY from the login node." >&2
    exit 2
  fi
  export OPENPFC_DIRTY="${OPENPFC_DIRTY:-unknown}"
fi

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
    --export=ALL,OPENPFC_REVISION,OPENPFC_DIRTY,OPENPFC_SRC \
    "${SBATCH}"
}

case "${MODE}" in
  check)
    python3 "${SCRIPT_DIR}/../../apps/tungsten/scripts/flagship_ladder.py" --check
    python3 "${SCRIPT_DIR}/../../apps/tungsten/scripts/flagship_ladder.py" \
      --collect-self-test
    ;;
  collect)
    python3 "${SCRIPT_DIR}/../../apps/tungsten/scripts/flagship_ladder.py" \
      --collect "${OPENPFC_SCALING_ROOT}" \
      --out "${SCRIPT_DIR}/../../docs/report/data/tungsten_lumi_g_flagship.csv"
    ;;
  pilot)
    export OPENPFC_FFT_NODE_GRID="${OPENPFC_FFT_NODE_GRID:-1}"
    echo "1-node 768^3 pilot (8 GCDs, I/O off, ${STEPS} steps) rev=${OPENPFC_REVISION:-empty}"
    submit_one 1 768 "01:00:00" "thip-flag-1n-768"
    ;;
  weak)
    export OPENPFC_FFT_NODE_GRID="${OPENPFC_FFT_NODE_GRID:-1}"
    echo "Weak ladder 1/2/4/8/16/32 nodes (not 60). NODE_GRID=${OPENPFC_FFT_NODE_GRID} rev=${OPENPFC_REVISION:-empty}"
    submit_one 1 768 "01:00:00" "thip-flag-1n-768"
    submit_one 2 960 "01:00:00" "thip-flag-2n-960"
    submit_one 4 1200 "01:00:00" "thip-flag-4n-1200"
    submit_one 8 1536 "02:00:00" "thip-flag-8n-1536"
    submit_one 16 1920 "02:00:00" "thip-flag-16n-1920"
    submit_one 32 2400 "02:00:00" "thip-flag-32n-2400"
    ;;
  control)
    echo "Issue #13 matched-decomposition / process-grid controls. rev=${OPENPFC_REVISION:-empty}"
    echo "1-node 768^3: min-surface, explicit 1x8x1, 1x8x1+pencils"
    unset OPENPFC_FFT_PROC_GRID TUNGSTEN_USE_PENCILS || true
    export OPENPFC_FFT_NODE_GRID=1
    submit_one 1 768 "01:00:00" "thip-c1-ms"
    unset OPENPFC_FFT_NODE_GRID || true
    export OPENPFC_FFT_PROC_GRID=1,8,1
    submit_one 1 768 "01:00:00" "thip-c1-181"
    export TUNGSTEN_USE_PENCILS=1
    submit_one 1 768 "01:00:00" "thip-c1-181p"
    unset OPENPFC_FFT_PROC_GRID TUNGSTEN_USE_PENCILS || true
    echo "2-node 960^3: 1x8x2 pencils, 1x1x16 slab, 2x2x4, 1x4x4"
    export OPENPFC_FFT_NODE_GRID=1
    submit_one 2 960 "01:00:00" "thip-c2-182"
    unset OPENPFC_FFT_NODE_GRID || true
    submit_one 2 960 "01:00:00" "thip-c2-slab"
    export OPENPFC_FFT_PROC_GRID=2,2,4
    submit_one 2 960 "01:00:00" "thip-c2-224"
    export OPENPFC_FFT_PROC_GRID=1,4,4
    submit_one 2 960 "01:00:00" "thip-c2-144"
    unset OPENPFC_FFT_PROC_GRID || true
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
    export OPENPFC_FFT_NODE_GRID="${OPENPFC_FFT_NODE_GRID:-1}"
    echo "60-node 3000^3 (480 GCD). NODE_GRID=${OPENPFC_FFT_NODE_GRID}"
    submit_one 60 3000 "04:00:00" "thip-flag-60n-3000"
    ;;
esac

echo "Logs: /scratch/project_462001519/juaho/logs/"
echo "Runs: ${OPENPFC_SCALING_ROOT}/runs/"
echo "Collect CSV: $0 collect"
echo "See docs/hpc/lumi_gpu_flagship.md"

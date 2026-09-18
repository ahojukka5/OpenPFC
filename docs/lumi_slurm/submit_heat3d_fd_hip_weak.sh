#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Issue #25: heat3d_fd_hip weak scaling at constant 256^3 interior / GCD.
# Submit from a LUMI login node. Do not run the jobs on the login node.
#
# Usage:
#   HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip ./submit_heat3d_fd_hip_weak.sh check
#   HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip ./submit_heat3d_fd_hip_weak.sh clean
#   HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip ./submit_heat3d_fd_hip_weak.sh diag
#   ./submit_heat3d_fd_hip_weak.sh collect
#
# `clean` is the admitted FD-2 series (no HEAT3D_DIAG_TIMING).
# `diag` repeats the same grids with attribution timers.

set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "check" && "${MODE}" != "clean" && "${MODE}" != "diag" &&
      "${MODE}" != "collect" ]]; then
  echo "usage: $0 check|clean|diag|collect" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/heat3d_fd_hip_weak.sbatch"
ACCOUNT="${ACCOUNT:-project_462001519}"
PARTITION="${PARTITION:-standard-g}"
# Do not inherit a leftover OPENPFC_SCALING_ROOT from another campaign.
export OPENPFC_SCALING_ROOT="${HEAT3D_WEAK_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling/heat3d-fd-weak}"
export HEAT3D_STEPS="${HEAT3D_STEPS:-105}"
export HEAT3D_WARMUP="${HEAT3D_WARMUP:-5}"
export HEAT3D_DT="${HEAT3D_DT:-0.01}"
export HEAT3D_FD_ORDER="${HEAT3D_FD_ORDER:-2}"
export HEAT3D_REQUIRE_INTERIOR="${HEAT3D_REQUIRE_INTERIOR:-256x256x256}"

SRC="${OPENPFC_SRC:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export OPENPFC_SRC="${SRC}"
if command -v git >/dev/null 2>&1 &&
   git -C "${SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  export OPENPFC_REVISION="$(git -C "${SRC}" rev-parse HEAD)"
  export OPENPFC_DIRTY="$(git -C "${SRC}" status --porcelain | wc -l | tr -d ' ')"
fi

if [[ "${MODE}" == "check" ]]; then
  python3 "${SCRIPT_DIR}/../../apps/heat3d/scripts/fd_weak_ladder.py" --check
  exit 0
fi
if [[ "${MODE}" == "collect" ]]; then
  python3 "${SCRIPT_DIR}/../../apps/heat3d/scripts/fd_weak_ladder.py" \
    --collect "${OPENPFC_SCALING_ROOT}" \
    --out "${SCRIPT_DIR}/../../docs/report/data/heat3d_fd_lumi_g_weak.csv"
  exit 0
fi

: "${HEAT3D_HIP_BIN:?set HEAT3D_HIP_BIN to heat3d_fd_hip}"
if [[ ! -x "${HEAT3D_HIP_BIN}" ]]; then
  echo "HEAT3D_HIP_BIN is not executable: ${HEAT3D_HIP_BIN}" >&2
  exit 1
fi
if [[ -z "${OPENPFC_REVISION:-}" ]]; then
  echo "OPENPFC_REVISION is empty; set OPENPFC_SRC" >&2
  exit 2
fi
export OPENPFC_DIRTY="${OPENPFC_DIRTY:-unknown}"
export HEAT3D_HIP_BIN
# Do not inherit packed-halo / FFT campaign leftovers via --export=ALL.
unset OPENPFC_HIP_FORCE_PACKED_HALO OPENPFC_CUDA_FORCE_PACKED_HALO \
  OPENPFC_ASSUME_GPU_AWARE_MPI OPENPFC_FFT_PROC_GRID OPENPFC_FFT_NODE_GRID \
  OPENPFC_FFT_SLAB_AXIS || true

if [[ "${MODE}" == "diag" ]]; then
  export HEAT3D_DIAG_TIMING=1
  PREFIX="h3dfd-d"
else
  unset HEAT3D_DIAG_TIMING || true
  PREFIX="h3dfd-w"
fi

submit_one() {
  local nodes="$1"
  local nx="$2"
  local ny="$3"
  local nz="$4"
  local grid="$5"
  local job_name="$6"
  local ntasks=$((nodes * 8))
  export HEAT3D_NX="${nx}"
  export HEAT3D_NY="${ny}"
  export HEAT3D_NZ="${nz}"
  export OPENPFC_FD_PROC_GRID="${grid}"
  sbatch \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes="${nodes}" \
    --ntasks="${ntasks}" \
    --ntasks-per-node=8 \
    --gpus-per-node=8 \
    --time=01:00:00 \
    --job-name="${job_name}" \
    --export=ALL,OPENPFC_REVISION,OPENPFC_DIRTY,OPENPFC_SRC,HEAT3D_HIP_BIN,HEAT3D_NX,HEAT3D_NY,HEAT3D_NZ,OPENPFC_FD_PROC_GRID,HEAT3D_STEPS,HEAT3D_WARMUP,HEAT3D_DT,HEAT3D_FD_ORDER,HEAT3D_REQUIRE_INTERIOR,OPENPFC_SCALING_ROOT \
    "${SBATCH}"
}

echo "Issue #25 FD-2 weak ${MODE}. rev=${OPENPFC_REVISION} dirty=${OPENPFC_DIRTY}"
submit_one 1 512 512 512 "2,2,2" "${PREFIX}1n-512"
submit_one 2 512 512 1024 "2,2,4" "${PREFIX}2n-512x1024"
submit_one 4 512 1024 1024 "2,4,4" "${PREFIX}4n-512x1024x1024"
submit_one 8 1024 1024 1024 "4,4,4" "${PREFIX}8n-1024"
submit_one 16 1024 1024 2048 "4,4,8" "${PREFIX}16n-1024x2048"

echo "Logs: /scratch/project_462001519/juaho/logs/"
echo "Runs: ${OPENPFC_SCALING_ROOT}/runs/"

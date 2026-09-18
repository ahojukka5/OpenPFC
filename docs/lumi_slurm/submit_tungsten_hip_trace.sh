#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Minimal 4-node / 32-GCD heFFTe phase-attribution set for issue #13.
# Reproduces the admitted layouts from research PR #487; does not extend
# the node ladder. Wall times from this binary are diagnostic, not admitted
# production numbers.
#
# Usage:
#   TUNGSTEN_HIP_BIN=/path/to/tungsten_hip ./submit_tungsten_hip_trace.sh
#   TUNGSTEN_HIP_BIN=... ./submit_tungsten_hip_trace.sh thip-tr-1152c thip-tr-1216c

set -euo pipefail

: "${TUNGSTEN_HIP_BIN:?set TUNGSTEN_HIP_BIN to the diagnostic tracing tungsten_hip}"
if [[ ! -x "${TUNGSTEN_HIP_BIN}" ]]; then
  echo "TUNGSTEN_HIP_BIN is not executable: ${TUNGSTEN_HIP_BIN}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/tungsten_hip_trace.sbatch"
TEMPLATE="${SCRIPT_DIR}/tungsten_hip_scaling.toml"
PARTITION="${PARTITION:-standard-g}"
ACCOUNT="${ACCOUNT:-project_462001519}"
STEPS="${TUNGSTEN_STEPS:-20}"
# Ignore a stale OPENPFC_SRC from another worktree in the login environment.
SRC="$(cd "${SCRIPT_DIR}/../.." && pwd)"
unset OPENPFC_REVISION OPENPFC_DIRTY

export TUNGSTEN_HIP_BIN
export TUNGSTEN_STEPS="${STEPS}"
export TUNGSTEN_SCALING_TEMPLATE="${TEMPLATE}"
export OPENPFC_SCALING_ROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001519/juaho/openpfc-heffte-trace}"
export OPENPFC_SRC="${SRC}"

if command -v git >/dev/null 2>&1 &&
   git -C "${SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  export OPENPFC_REVISION="$(git -C "${SRC}" rev-parse HEAD)"
  export OPENPFC_DIRTY="$(git -C "${SRC}" status --porcelain | wc -l | tr -d ' ')"
fi
if [[ -z "${OPENPFC_REVISION:-}" ]]; then
  echo "OPENPFC_REVISION is empty; refusing to submit without provenance" >&2
  exit 2
fi
if [[ "${OPENPFC_DIRTY}" != "0" ]]; then
  echo "refusing dirty tree (dirty=${OPENPFC_DIRTY}); commit before submitting" >&2
  exit 2
fi

submit_one() {
  local name="$1"
  local lx="$2"
  local ly="$3"
  local lz="$4"
  local grid="$5"
  local cgrid="${6:-}"
  export TUNGSTEN_LX="${lx}"
  export TUNGSTEN_LY="${ly}"
  export TUNGSTEN_LZ="${lz}"
  export OPENPFC_FFT_PROC_GRID="${grid}"
  unset OPENPFC_FFT_NODE_GRID TUNGSTEN_USE_PENCILS \
        OPENPFC_FFT_COMPLEX_OUTBOX || true
  if [[ -n "${cgrid}" ]]; then
    export OPENPFC_FFT_COMPLEX_PROC_GRID="${cgrid}"
  else
    unset OPENPFC_FFT_COMPLEX_PROC_GRID || true
  fi
  sbatch \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes=4 \
    --ntasks=32 \
    --ntasks-per-node=8 \
    --gpus-per-node=8 \
    --time=01:00:00 \
    --job-name="${name}" \
    --export=ALL,OPENPFC_REVISION,OPENPFC_DIRTY,OPENPFC_SRC,OPENPFC_FFT_PROC_GRID,OPENPFC_FFT_COMPLEX_PROC_GRID,TUNGSTEN_LX,TUNGSTEN_LY,TUNGSTEN_LZ \
    "${SBATCH}"
}

CASES=(
  "thip-tr-1200c:1200:1200:1200:1,8,4"
  "thip-tr-1200x1152:1200:1200:1152:1,1,32"
  "thip-tr-1200x1216:1200:1200:1216:1,1,32"
  "thip-tr-1200x1280:1200:1200:1280:1,1,32"
  "thip-tr-1152c:1152:1152:1152:1,1,32"
  "thip-tr-1216c:1216:1216:1216:1,1,32"
  "thip-tr-1200x1152-c2161:1200:1200:1152:1,1,32:2,16,1"
  "thip-tr-1200x1152-c481:1200:1200:1152:1,1,32:4,8,1"
)

echo "heFFTe phase-trace set (4 nodes / 32 GCDs) rev=${OPENPFC_REVISION} dirty=${OPENPFC_DIRTY}"
want=("$@")
for spec in "${CASES[@]}"; do
  IFS=':' read -r name lx ly lz grid cgrid <<< "${spec}"
  if (( ${#want[@]} > 0 )); then
    keep=0
    for w in "${want[@]}"; do
      if [[ "${w}" == "${name}" ]]; then
        keep=1
        break
      fi
    done
    if (( keep == 0 )); then
      continue
    fi
  elif [[ -n "${cgrid}" ]]; then
    # Complex-outbox experiments are opt-in; do not resubmit with the
    # original six-case set.
    continue
  fi
  submit_one "${name}" "${lx}" "${ly}" "${lz}" "${grid}" "${cgrid}"
done

echo "Logs: /scratch/project_462001519/juaho/logs/"
echo "Runs: ${OPENPFC_SCALING_ROOT}/runs/"
echo "Collect: python3 apps/tungsten/scripts/parse_heffte_trace.py ${OPENPFC_SCALING_ROOT}/runs"

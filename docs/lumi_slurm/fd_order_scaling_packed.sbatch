#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001245
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001245/juaho/logs/%x-%j.out
#
# Issue #108: sequential cheap-rung matrix on one allocation.
# FD wall/step is milliseconds, so 15 (order × repeat) cells fit in
# one 1-node or 8-node job and stay under AssocMaxSubmitJobLimit.
# Submit with docs/lumi_slurm/submit_fd_order_scaling.sh.

set -euo pipefail

ALLOWED_ACCOUNTS="project_462001245"

: "${HEAT3D_HIP_BIN:?set HEAT3D_HIP_BIN to heat3d_fd_hip}"
: "${HEAT3D_NX:?}"
: "${HEAT3D_NY:?}"
: "${HEAT3D_NZ:?}"
: "${OPENPFC_FD_PROC_GRID:?set OPENPFC_FD_PROC_GRID=gx x gy x gz}"

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
STEPS="${HEAT3D_STEPS:-5005}"
DT="${HEAT3D_DT:-0.01}"
MODE="${OPENPFC_FD_MODE:-clean}"
JOBROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001245/juaho/openpfc-scaling/fd-order-108}"
ORDERS="${HEAT3D_ORDERS:-2:4:8:12:20}"
ORDERS="${ORDERS//,/:}"
NREP="${OPENPFC_REPEATS:-3}"
SKIP="${HEAT3D_PACK_SKIP:-}"
SKIP="${SKIP//,/:}"
GAP="${HEAT3D_CELL_GAP_S:-15}"
PREFIX="${OPENPFC_JOB_PREFIX:-h3d108c}"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES \
  OPENPFC_HIP_FORCE_PACKED_HALO OPENPFC_CUDA_FORCE_PACKED_HALO \
  OPENPFC_ASSUME_GPU_AWARE_MPI || true

NTASKS="${SLURM_NTASKS:?SLURM_NTASKS is not set}"
export HEAT3D_WARMUP="${HEAT3D_WARMUP:-50}"
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

want_skip() {
  local key="$1"
  local tok
  IFS=':' read -ra toks <<< "${SKIP}"
  for tok in "${toks[@]}"; do
    if [[ "${tok}" == "${key}" ]]; then
      return 0
    fi
  done
  return 1
}

run_cell() {
  local order="$1"
  local repeat="$2"
  local name="${PREFIX}-fd${order}-${SLURM_NNODES}n-r${repeat}"
  local rundir="${JOBROOT}/runs/${name}_${SLURM_JOB_ID}"
  mkdir -p "${rundir}"
  cd "${rundir}"
  cp -f "${BASH_SOURCE[0]}" "${rundir}/fd_order_scaling.sbatch"
  export HEAT3D_PROFILE_JSON="${rundir}/timing_profile.json"
  local halo_width=$((order / 2))
  local halo_face_bytes=$((6 * 256 * 256 * halo_width * 8))
  {
    echo "=== fd-order scaling (issue #108) ==="
    echo "job=${SLURM_JOB_ID}"
    echo "name=${name}"
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
    echo "fd_order=${order}"
    echo "halo_width=${halo_width}"
    echo "halo_face_bytes=${halo_face_bytes}"
    echo "mode=${MODE}"
    echo "repeat=${repeat}"
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
    echo "cwd=${rundir}"
    echo "packed=1"
    module list
  } | tee "${rundir}/run_meta.txt"

  set +e
  srun --export=ALL "${SRUN_BIND[@]}" \
    "${HEAT3D_HIP_BIN}" "${NX}" "${NY}" "${NZ}" "${STEPS}" "${DT}" "${order}" \
    2>&1 | tee "${rundir}/run.log"
  local srun_rc="${PIPESTATUS[0]}"
  set -e

  local admit="ok"
  local reason="none"
  if (( srun_rc != 0 )); then
    admit="reject"
    reason="srun_exit_${srun_rc}"
  elif ! grep -q 'HEAT3D_FD_DECOMP' "${rundir}/run.log"; then
    admit="reject"
    reason="missing_decomp_banner"
  elif grep -qiE 'HEAT3D_HIP_CHECKSUM .*nan|HEAT3D_HIP_CHECKSUM .*inf' "${rundir}/run.log"; then
    admit="reject"
    reason="checksum_nonfinite"
  fi

  if [[ -f fd_placement.txt ]]; then
    dest="${rundir}/fd_placement.txt"
    if [[ ! fd_placement.txt -ef "${dest}" ]]; then
      cp -f fd_placement.txt "${dest}"
    fi
  fi

  {
    echo "admit=${admit}"
    echo "reason=${reason}"
    echo "srun_rc=${srun_rc}"
  } | tee "${rundir}/admit.txt"
  echo "=== packed cell ${name} rc=${srun_rc} admit=${admit} ==="
  if (( GAP > 0 )); then
    echo "=== drain HIP ${GAP}s after ${name} ==="
    sleep "${GAP}"
  fi
  if [[ "${admit}" != "ok" ]]; then
    return 1
  fi
  return 0
}

echo "=== fd-order packed scaling (issue #108) job=${SLURM_JOB_ID} nodes=${SLURM_NNODES} ==="
failed=0
IFS=':' read -ra order_list <<< "${ORDERS}"
for order in "${order_list[@]}"; do
  r=1
  while (( r <= NREP )); do
    key="${order}r${r}"
    if want_skip "${key}"; then
      echo "skip packed cell fd${order}-${SLURM_NNODES}n-r${r}"
      r=$((r + 1))
      continue
    fi
    if ! run_cell "${order}" "${r}"; then
      failed=$((failed + 1))
    fi
    r=$((r + 1))
  done
done

echo "=== packed scaling done failed_cells=${failed} ==="
if (( failed > 0 )); then
  exit 1
fi
exit 0

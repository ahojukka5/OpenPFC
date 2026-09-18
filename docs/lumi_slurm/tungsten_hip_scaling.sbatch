#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# tungsten_hip on LUMI-G: one MPI rank per GCD. Size one GCD, strong-scale
# 1/2/4/8 GCDs on one node, then 16/32 GCDs (2/4 nodes). I/O off. Submit with
# docs/lumi_slurm/submit_tungsten_hip_scaling.sh so ntasks and gpus-per-node
# match. Do not use the 0.1.4 / project_462001245 paths in tungsten_gpu.sbatch.
#
# Required in the job environment:
#   TUNGSTEN_HIP_BIN   absolute path to a 0.2 tungsten_hip
# Optional:
#   TUNGSTEN_LX        grid points in x (default 512)
#   TUNGSTEN_LY        grid points in y (default TUNGSTEN_LX)
#   TUNGSTEN_LZ        grid points in z (default TUNGSTEN_LX)
#   TUNGSTEN_STEPS     accepted steps; t1 = steps with dt=1 (default 20)
#   OPENPFC_SCALING_ROOT  scratch run parent
#     (default /scratch/project_462001519/juaho/openpfc-scaling)

set -euo pipefail

: "${TUNGSTEN_HIP_BIN:?set TUNGSTEN_HIP_BIN to the 0.2 tungsten_hip binary}"

LX="${TUNGSTEN_LX:-512}"
LY="${TUNGSTEN_LY:-${LX}}"
LZ="${TUNGSTEN_LZ:-${LX}}"
STEPS="${TUNGSTEN_STEPS:-20}"
JOBROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling}"

# Slurm runs a spool copy of this script; do not resolve the TOML relative to
# BASH_SOURCE. The submit helper exports TUNGSTEN_SCALING_TEMPLATE.
TEMPLATE="${TUNGSTEN_SCALING_TEMPLATE:-}"
if [[ -z "${TEMPLATE}" || ! -f "${TEMPLATE}" ]]; then
  echo "set TUNGSTEN_SCALING_TEMPLATE to docs/lumi_slurm/tungsten_hip_scaling.toml" >&2
  exit 1
fi

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
# Leave every GCD visible: bind_local_device() selects local_rank % n_devices
# so GPU-aware MPI can use intra-node HIP IPC. Do not mask with
# ROCR_VISIBLE_DEVICES. MPICH_OFI_NIC_POLICY=GPU made 16-GCD 768³ slower
# than the Cray default BLOCK (373 ms vs 145 ms); override in the job
# environment to experiment.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true

NTASKS="${SLURM_NTASKS:?SLURM_NTASKS is not set}"
RUNDIR="${JOBROOT}/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${RUNDIR}"

sed -e "s/__LX__/${LX}/g" -e "s/__LY__/${LY}/g" -e "s/__LZ__/${LZ}/g" \
    -e "s/__T1__/${STEPS}/g" "${TEMPLATE}" \
  > "${RUNDIR}/input.toml"
if [[ "${TUNGSTEN_USE_PENCILS:-}" == "1" ]]; then
  sed -i 's/^use_pencils = false/use_pencils = true/' "${RUNDIR}/input.toml"
fi
cp -f "${BASH_SOURCE[0]}" "${RUNDIR}/tungsten_hip_scaling.sbatch"
cd "${RUNDIR}"
USE_PENCILS="$(awk '/^use_pencils/{print $3; exit}' "${RUNDIR}/input.toml")"

# Full-node 8-GCD map from LUMI-G docs, applied per node. Partial GCD
# allocations (dev-g / small-g with 1–4 GPUs) do not own those CPUs;
# map_cpu:49 then fails. Do not wrap with ROCR_VISIBLE_DEVICES: the binary
# calls bind_local_device() so every GCD stays visible for GPU-aware IPC.
PER_NODE="${SLURM_NTASKS_PER_NODE:-${NTASKS}}"
SRUN_BIND=()
CPU_BIND="none"
if (( PER_NODE == 8 )); then
  # 7 cores per GCD (LUMI CCD mask), not a single map_cpu thread.
  CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"
  SRUN_BIND=(--cpu-bind="${CPU_BIND}")
elif (( PER_NODE > 8 )); then
  echo "at most 8 ranks per node (one per GCD); ntasks-per-node=${PER_NODE}" >&2
  exit 1
fi

SRC="${OPENPFC_SRC:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"

# Resolve the layout OpenPFC will actually use (brick_split.hpp). Do not
# assume 8 ranks are min-surface if OPENPFC_FFT_PROC_GRID is set.
PROC_GRID=""
LOCAL_BRICK=""
forced="${OPENPFC_FFT_PROC_GRID:-}"
if [[ -n "${forced}" ]]; then
  forced="${forced//x/,}"
  forced="${forced//X/,}"
  IFS=',' read -r gx gy gz <<< "${forced}"
  if [[ -n "${gx}" && -n "${gy}" && -n "${gz}" ]] &&
     (( gx * gy * gz == NTASKS && LX % gx == 0 && LY % gy == 0 && LZ % gz == 0 )); then
    PROC_GRID="${gx}x${gy}x${gz}"
    LOCAL_BRICK="$((LX / gx))x$((LY / gy))x$((LZ / gz))"
  fi
fi
if [[ -z "${PROC_GRID}" ]]; then
  if (( NTASKS >= 9 )) && [[ "${OPENPFC_FFT_NODE_GRID:-}" == "1" ]] &&
     (( LY % 8 == 0 && LZ % SLURM_NNODES == 0 )); then
    PROC_GRID="1x8x${SLURM_NNODES}"
    LOCAL_BRICK="${LX}x$((LY / 8))x$((LZ / SLURM_NNODES))"
  elif (( NTASKS >= 9 && LZ % NTASKS == 0 )); then
    PROC_GRID="1x1x${NTASKS}"
    LOCAL_BRICK="${LX}x${LY}x$((LZ / NTASKS))"
  elif (( NTASKS >= 9 && LY % 8 == 0 && LZ % SLURM_NNODES == 0 )); then
    PROC_GRID="1x8x${SLURM_NNODES}"
    LOCAL_BRICK="${LX}x$((LY / 8))x$((LZ / SLURM_NNODES))"
  elif (( NTASKS == 8 && LX % 2 == 0 && LY % 2 == 0 && LZ % 2 == 0 )); then
    # Cubes: min_surface_proc_grid is 2x2x2 (pinned in test_brick_split).
    PROC_GRID="min-surface(2x2x2)"
    LOCAL_BRICK="$((LX / 2))x$((LY / 2))x$((LZ / 2))"
  else
    PROC_GRID="min-surface"
    LOCAL_BRICK="unknown"
  fi
fi

BIN_SHA=""
if command -v sha256sum >/dev/null 2>&1 && [[ -f "${TUNGSTEN_HIP_BIN}" ]]; then
  BIN_SHA="$(sha256sum "${TUNGSTEN_HIP_BIN}" | awk '{print $1}')"
fi
# Revision must be exported at submit time. Compute nodes often have no
# `git` after `module purge`, and git worktrees have `.git` as a file.
REV="${OPENPFC_REVISION:-}"
DIRTY="${OPENPFC_DIRTY:-}"
if [[ -z "${REV}" ]] && command -v git >/dev/null 2>&1; then
  if git -C "${SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    REV="$(git -C "${SRC}" rev-parse HEAD 2>/dev/null || true)"
    DIRTY="$(git -C "${SRC}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  fi
fi

{
  echo "=== tungsten_hip scaling ==="
  echo "job=${SLURM_JOB_ID}"
  echo "name=${SLURM_JOB_NAME}"
  echo "nodes=${SLURM_NNODES}"
  echo "ntasks=${NTASKS}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "bin=${TUNGSTEN_HIP_BIN}"
  echo "bin_sha256=${BIN_SHA}"
  echo "Lx=${LX}"
  echo "Ly=${LY}"
  echo "Lz=${LZ}"
  echo "N=${LX}x${LY}x${LZ}"
  echo "steps=${STEPS}"
  echo "cwd=${RUNDIR}"
  echo "input=${RUNDIR}/input.toml"
  echo "cpu-bind=${CPU_BIND}"
  echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
  echo "MPICH_OFI_NIC_POLICY=${MPICH_OFI_NIC_POLICY:-unset}"
  echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
  echo "OPENPFC_FFT_SLAB_AXIS=${OPENPFC_FFT_SLAB_AXIS:-unset}"
  echo "OPENPFC_FFT_NODE_GRID=${OPENPFC_FFT_NODE_GRID:-unset}"
  echo "OPENPFC_FFT_PROC_GRID=${OPENPFC_FFT_PROC_GRID:-unset}"
  echo "TUNGSTEN_USE_PENCILS=${TUNGSTEN_USE_PENCILS:-unset}"
  echo "use_pencils=${USE_PENCILS:-unset}"
  echo "proc_grid=${PROC_GRID}"
  echo "local_brick=${LOCAL_BRICK}"
  echo "revision=${REV}"
  echo "dirty=${DIRTY}"
  echo "src=${SRC}"
  module list
} | tee "${RUNDIR}/run_meta.txt"

srun "${SRUN_BIND[@]}" \
  "${TUNGSTEN_HIP_BIN}" "${RUNDIR}/input.toml"

echo "=== done ==="

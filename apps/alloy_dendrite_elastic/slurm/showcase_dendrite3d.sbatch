#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00
#SBATCH --job-name=dend3d-512
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Issue #12: 3-D isothermal dendrite for ParaView (.bin + .xdmf).
# Spherical seed (nz=nx). Elasticity off. HBM is not the limit
# (64^3 used_max ~0.24 GiB/GCD); unique-frame dumps GPU-faulted under
# GPU-aware MPI subarrays, so this path forces packed-halo MPI.
#
#   SHOWCASE_MODE=pilot   64^3 dump-every-1 (dense-dump smoke)
#   SHOWCASE_MODE=prod    512^3, 16 nodes, dump every 40 steps (~400 frames)
# Scratch only.

set -euo pipefail

SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-dend3d-512}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/openpfc-showcase/dendrite3d}"
MODE="${SHOWCASE_MODE:-pilot}"

# Bake revision on the login-exported env. Compute nodes often have no git
# after module purge, and worktrees have .git as a file.
REV="${OPENPFC_REVISION:-}"
DIRTY="${OPENPFC_DIRTY:-}"
if [[ -z "${REV}" ]] && command -v git >/dev/null 2>&1; then
  if git -C "${SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    REV="$(git -C "${SRC}" rev-parse HEAD 2>/dev/null || true)"
    DIRTY="$(git -C "${SRC}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  fi
fi

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export MPICH_GPU_SUPPORT_ENABLED=1
export OPENPFC_HIP_FORCE_PACKED_HALO="${OPENPFC_HIP_FORCE_PACKED_HALO:-1}"
export OPENPFC_FFT_NODE_GRID="${OPENPFC_FFT_NODE_GRID:-1}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true
export CC=cc CXX=CC

mkdir -p "${OUT}/fields" "${OUT}/frames"
{
  echo "job ${SLURM_JOB_ID} partition ${SLURM_JOB_PARTITION} host $(hostname)"
  echo "date $(date -Iseconds)"
  echo "src ${SRC}"
  echo "revision ${REV}"
  echo "dirty ${DIRTY}"
  echo "mode ${MODE}"
  echo "OPENPFC_HIP_FORCE_PACKED_HALO=${OPENPFC_HIP_FORCE_PACKED_HALO}"
  echo "3-D spherical seed; isothermal; elasticity off for the memory path"
  echo "anisotropy: fourfold cubic eps4=0.05; <100> along the brick axes"
  module list
} | tee "${OUT}/provenance.txt"

if [[ ! -f "${BUILD}/build.ninja" && ! -f "${BUILD}/Makefile" ]]; then
  echo "No HIP build tree at ${BUILD}. Configure on the login node with"
  echo "  ./scripts/build.sh --machine=lumi --partition=standard-g --no-test --no-submit --build-dir=${BUILD}"
  exit 2
fi
cmake --build "${BUILD}" -j 8 --target alloy_dendrite_hip_growth
BIN="${BUILD}/apps/alloy_dendrite_elastic/alloy_dendrite_hip_growth"
test -x "${BIN}"

CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"

if [[ "${MODE}" == "prod" ]]; then
  NX="${SHOWCASE_NX:-512}"
  STEPS="${SHOWCASE_STEPS:-16000}"
  EVERY="${SHOWCASE_SAMPLE:-40}"
  DUMP="${SHOWCASE_DUMP_EVERY:-1}"
else
  NX="${SHOWCASE_NX:-64}"
  STEPS="${SHOWCASE_STEPS:-80}"
  EVERY="${SHOWCASE_SAMPLE:-1}"
  DUMP="${SHOWCASE_DUMP_EVERY:-1}"
fi
DX="${SHOWCASE_DX:-0.8}"
NY="${SHOWCASE_NY:-${NX}}"
NZ="${SHOWCASE_NZ:-${NX}}"

CMD=(
  "${BIN}" --nx="${NX}" --ny="${NY}" --nz="${NZ}" --dx="${DX}"
  --steps="${STEPS}" --sample-every="${EVERY}"
  --Dl=2 --k=0.15 --eps4=0.05 --evolve-theta=0 --seed-radius=8
  --elastic=0
  --csv="${OUT}/growth.csv" --run-id=dendrite3d
  --fields-dir="${OUT}/fields" --fields-every="${DUMP}"
)
printf '%q ' "${CMD[@]}" | tee -a "${OUT}/provenance.txt"
echo | tee -a "${OUT}/provenance.txt"

NTASKS="${SLURM_NTASKS:-8}"
srun --ntasks="${NTASKS}" --gpus-per-node=8 --cpu-bind="${CPU_BIND}" "${CMD[@]}" \
  | tee "${OUT}/run.log"

echo "HBM is HIP_MEM in ${OUT}/run.log (hipMemGetInfo). sacct MaxRSS is host RSS." \
  | tee -a "${OUT}/provenance.txt"

XDMF="${SRC}/scripts/xdmfgen.py"
if [[ -f "${XDMF}" && -f "${OUT}/fields/dendrite3d_manifest.json" ]]; then
  python3 "${XDMF}" --manifest "${OUT}/fields/dendrite3d_manifest.json" || true
fi

echo "=== done ${OUT} ==="
echo "ParaView: File → Open ${OUT}/fields/dendrite3d.xdmf"
echo "Frames: pvbatch ${OUT}/paraview_render.py"

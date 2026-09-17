#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --job-name=dend3d-show
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Issue #12: genuine 3-D isothermal dendrite on one LUMI-G node.
# Spherical seed (nz=nx). No elasticity on this path: measure bytes/cell
# before coupling Green at this scale. 2048^3 is a later node-count target.
#
#   SHOWCASE_MODE=pilot   64^3 (default)
#   SHOWCASE_MODE=prod    candidate 256^3 until the pilot reports HBM/wall

set -euo pipefail

SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-showcase-dend}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/openpfc-showcase/dendrite3d_${SLURM_JOB_ID}}"
MODE="${SHOWCASE_MODE:-pilot}"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export MPICH_GPU_SUPPORT_ENABLED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true
export CC=cc CXX=CC

mkdir -p "${OUT}/fields"
{
  echo "job ${SLURM_JOB_ID} partition ${SLURM_JOB_PARTITION} host $(hostname)"
  echo "date $(date -Iseconds)"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
  echo "mode ${MODE}"
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
  NX="${SHOWCASE_NX:-256}"
  STEPS="${SHOWCASE_STEPS:-2000}"
  EVERY="${SHOWCASE_SAMPLE:-50}"
  DUMP="${SHOWCASE_DUMP_EVERY:-2}"
  if [[ -z "${SHOWCASE_NX:-}" ]]; then
    echo "SHOWCASE_NX unset: candidate ${NX}^3; re-choose from pilot HBM/wall" | tee -a "${OUT}/provenance.txt"
  fi
else
  NX="${SHOWCASE_NX:-64}"
  STEPS="${SHOWCASE_STEPS:-400}"
  EVERY="${SHOWCASE_SAMPLE:-20}"
  DUMP="${SHOWCASE_DUMP_EVERY:-1}"
fi
DX="${SHOWCASE_DX:-0.8}"

CMD=(
  "${BIN}" --nx="${NX}" --ny="${NX}" --nz="${NX}" --dx="${DX}"
  --steps="${STEPS}" --sample-every="${EVERY}"
  --Dl=2 --k=0.15 --eps4=0.05 --evolve-theta=0 --seed-radius=8
  --elastic=0
  --csv="${OUT}/growth.csv" --run-id=dendrite3d
  --fields-dir="${OUT}/fields" --fields-every="${DUMP}"
)
printf '%q ' "${CMD[@]}" | tee -a "${OUT}/provenance.txt"
echo | tee -a "${OUT}/provenance.txt"

srun --ntasks=8 --gpus-per-node=8 --cpu-bind="${CPU_BIND}" "${CMD[@]}" \
  | tee "${OUT}/run.log"

echo "HBM is HIP_MEM in ${OUT}/run.log (hipMemGetInfo). sacct MaxRSS is host RSS." \
  | tee -a "${OUT}/provenance.txt"

echo "=== done ${OUT} ==="
echo "Render: python3 ${SRC}/apps/alloy_dendrite_elastic/scripts/render_dendrite3d_showcase.py --run ${OUT}"

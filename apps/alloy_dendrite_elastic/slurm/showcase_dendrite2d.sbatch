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
#SBATCH --time=08:00:00
#SBATCH --job-name=dend2d-show
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Issue #11: isothermal 2-D dendrite on the device Green path.
# Both arms keep Al-4.5wt%Cu material and the elastic solve; only
# lambda_el is toggled (0 vs calibrated lambda=D_l/a2).
# SHOWCASE_MODE=pilot (default) or prod. Prod NX is a candidate until the
# 8-GCD pilot reports wall time and HBM.
# Film: >=600 diagnostic snapshots (10 s at 60 fps). Same dt/material;
# sample-every is lowered so dumps are unique frames. Scratch only.

set -euo pipefail

SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-showcase-dend}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/openpfc-showcase/dendrite2d_${SLURM_JOB_ID}}"
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

mkdir -p "${OUT}"
{
  echo "job ${SLURM_JOB_ID} partition ${SLURM_JOB_PARTITION} host $(hostname)"
  echo "date $(date -Iseconds)"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
  echo "mode ${MODE}"
  echo "device Green path: alloy_dendrite_hip_growth (no host elasticity)"
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
  NX="${SHOWCASE_NX:-2048}"
  STEPS="${SHOWCASE_STEPS:-8000}"
  EVERY="${SHOWCASE_SAMPLE:-13}"
  if [[ -z "${SHOWCASE_NX:-}" ]]; then
    echo "SHOWCASE_NX unset: using candidate ${NX}; re-choose from pilot HBM/wall" | tee -a "${OUT}/provenance.txt"
  fi
else
  NX="${SHOWCASE_NX:-256}"
  STEPS="${SHOWCASE_STEPS:-800}"
  EVERY="${SHOWCASE_SAMPLE:-1}"
fi
DX="${SHOWCASE_DX:-0.8}"
SHEAR="${SHOWCASE_LIQUID_SHEAR:-0.05}"

run_one() {
  local id="$1"; shift
  mkdir -p "${OUT}/fields_${id}"
  srun --ntasks=8 --gpus-per-node=8 --cpu-bind="${CPU_BIND}" \
    "${BIN}" --nx="${NX}" --ny="${NX}" --nz=1 --dx="${DX}" \
    --steps="${STEPS}" --sample-every="${EVERY}" \
    --Dl=2 --k=0.15 --eps4=0.05 --evolve-theta=0 \
    --elastic=1 --eps-T=0 --n-el-substep=1 \
    --liquid-shear="${SHEAR}" \
    --csv="${OUT}/${id}.csv" --run-id="${id}" \
    --fields-dir="${OUT}/fields_${id}" --fields-every=1 \
    "$@" | tee "${OUT}/${id}.log"
}

# Same material/Green arm; feedback off vs calibrated lambda_el=lambda.
run_one off --lambda-el=0
run_one on

echo "HBM is HIP_MEM in ${OUT}/*.log (hipMemGetInfo). sacct MaxRSS is host RSS." \
  | tee -a "${OUT}/provenance.txt"

echo "=== done ${OUT} ==="
echo "Render: python3 ${SRC}/apps/alloy_dendrite_elastic/scripts/render_dendrite_showcase.py --run ${OUT}"

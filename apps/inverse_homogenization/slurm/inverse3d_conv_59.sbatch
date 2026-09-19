#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001245
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --job-name=inv3d-conv-59
#SBATCH --output=/scratch/project_462001245/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001245/juaho/logs/%x-%j.err
#
# OpenPFC #59/#63: converge the matched 3-D inverse instead of stopping at N.
# CONV_MODE=qual|64|128  (default 64)
#
# If the environment has SBATCH_ACCOUNT=project_462001519, submit with
#   sbatch --account=project_462001245 ...
# so the env does not override this file.
#
# SRC_DIR must be the checkout with the #63 CSV/SIMP fixes. Override if
# this default worktree is not that branch.
#
#   CONV_MODE=qual sbatch ...   32x32x61, short continuation, machinery check
#   CONV_MODE=64   sbatch ...   64x64x121, same target as job 22162138
#   CONV_MODE=128  sbatch ...   128x128x242

set -euo pipefail

SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/wt/openpfc-conv2}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-inv-conv63}"
OUT="${OUT_DIR:-/scratch/project_462001245/juaho/inverse-homogenization/conv_${SLURM_JOB_ID}}"
MODE="${CONV_MODE:-64}"
CT="${SRC}/apps/inverse_homogenization/evidence/2026-09-19-yang-a3-oracle-ev002.txt"
BIN="${BUILD}/apps/inverse_homogenization/openpfc_inverse_homogenize_hip"

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

mkdir -p "${OUT}/fields" /scratch/project_462001245/juaho/logs
{
  echo "job ${SLURM_JOB_ID} partition ${SLURM_JOB_PARTITION} host $(hostname)"
  echo "date $(date -Iseconds)"
  echo "account ${SLURM_JOB_ACCOUNT}"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
  echo "mode ${MODE}"
} | tee "${OUT}/provenance.txt"

if [[ ! -f "${BUILD}/CMakeCache.txt" ]]; then
  cmake -S "${SRC}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DHeffte_DIR=/users/juaho/opt/heffte/2.4.1-rocm/lib64/cmake/Heffte \
    -DOpenPFC_ENABLE_HIP=ON \
    -DOpenPFC_ENABLE_HIP_SPECTRAL=ON \
    -DOpenPFC_BUILD_TESTS=ON \
    -DOpenPFC_BUILD_APPS=ON \
    -DOpenPFC_BUILD_EXAMPLES=OFF
fi
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" \
  --target openpfc_inverse_homogenize_hip test_inverse_convergence

srun --ntasks=1 --gpus-per-node=1 \
  "${BUILD}/apps/inverse_homogenization/test_inverse_convergence"

COMMON=(
  --E-solid=1.0 --nu-solid=0.3 --E-void=0.002 --nu-void=0.3
  --target=file --C-target-file="${CT}"
  --volume=0.2575865186 --init-volume=0.2575865186
  --normalize=1 --dt=0.04 --max-delta=0.04 --project-volume=1
  --lambda-reg=0.05 --lambda-reg-end=0.2 --simp=1 --simp-end=2
  --n-el-iter=400
  --init=spinodal --seed=1
)

case "${MODE}" in
  qual)
    GRID=(--nx=32 --ny=32 --nz=61 --dx=1)
    STOP=(--continuation-steps=40 --max-steps=200 --conv-window=10
          --verify-convergence-steps=20 --dump-every=5)
    ;;
  64)
    GRID=(--nx=64 --ny=64 --nz=121 --dx=1)
    STOP=(--continuation-steps=300 --max-steps=5000 --conv-window=20
          --verify-convergence-steps=100 --dump-every=10)
    ;;
  128)
    GRID=(--nx=128 --ny=128 --nz=242 --dx=1)
    STOP=(--continuation-steps=300 --max-steps=5000 --conv-window=20
          --verify-convergence-steps=100 --dump-every=20)
    ;;
  *)
    echo "unknown CONV_MODE=${MODE}" >&2
    exit 2
    ;;
esac

echo "=== HIP inverse ${MODE} ==="
srun --ntasks=8 --gpus-per-node=8 "${BIN}" \
  "${COMMON[@]}" "${GRID[@]}" "${STOP[@]}" \
  --run-id="inv3d_${MODE}" \
  --csv="${OUT}/history.csv" \
  --dump-dir="${OUT}/fields" \
  | tee "${OUT}/run.log"

echo "done job ${SLURM_JOB_ID}" | tee -a "${OUT}/provenance.txt"

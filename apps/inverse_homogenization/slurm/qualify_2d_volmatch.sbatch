#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --job-name=q2d-volmatch
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Volume-matched binary of archived step-600 fields. t_v from order statistics
# of h so vf=0.5; chosen without looking at Poisson.

set -euo pipefail
SRC="${SRC_DIR:-/flash/project_462001519/juaho/wt/openpfc-476}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-inv-476}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/q2dvol_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/inverse_homogenization/openpfc_homogenize"
A256="/scratch/project_462001519/juaho/openpfc-showcase/inverse2d_22112442/fields/inverse2d_h_0600.bin"
A1024="/scratch/project_462001519/juaho/openpfc-showcase/inverse2d_22112443/fields/inverse2d_h_0600.bin"
# Histogram/order-statistic thresholds (computed before this job; not from nu).
TV256=0.6978081159092979
TV1024=0.6738155870227969

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

mkdir -p "${OUT}/n256_vol" "${OUT}/n1024_vol"
{
  echo "job ${SLURM_JOB_ID} host $(hostname)"
  echo "date $(date -Iseconds)"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
  echo "TV256=${TV256} TV1024=${TV1024} target_vf=0.5"
} | tee "${OUT}/provenance.txt"

cmake --build "${BUILD}" -j "${SLURM_NTASKS}" --target openpfc_homogenize
test -x "${BIN}"

run_one() {
  local nx="$1" bin="$2" dest="$3" tv="$4"
  echo "=== ${dest} tv=${tv} ==="
  srun --ntasks=8 --cpus-per-task=8 "${BIN}" \
    --nx="${nx}" --ny="${nx}" --nz=1 --dx=1 \
    --E-solid=1.0 --nu-solid=0.3 --E-void=0.02 --nu-void=0.3 \
    --n-el-iter=200 \
    --load-bin="${bin}" --binarize="${tv}" \
    --dump-dir="${dest}" \
    | tee "${dest}/run.log"
}

run_one 256 "${A256}" "${OUT}/n256_vol" "${TV256}"
run_one 1024 "${A1024}" "${OUT}/n1024_vol" "${TV1024}"

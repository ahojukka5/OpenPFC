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
#SBATCH --job-name=yang-a3-contrast
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Predeclared ersatz-stiffness ladder (frozen before Poisson):
#   E_void / E_solid = 0.02, 0.002, 0.0002
# Geometry, grid 64x64x121, n_el_iter=80 identical to job 22160647.

set -euo pipefail
SRC="${SRC_DIR:-/flash/project_462001519/juaho/wt/openpfc-yang}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-yang}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/yang43_contrast_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/inverse_homogenization/openpfc_homogenize"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

mkdir -p "${OUT}"
{
  echo "job ${SLURM_JOB_ID} host $(hostname)"
  echo "date $(date -Iseconds)"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
  echo "grid 64x64x121"
  echo "E_void_over_E_solid 0.02 0.002 0.0002"
} | tee "${OUT}/provenance.txt"

cmake --build "${BUILD}" -j "${SLURM_NTASKS}" --target openpfc_homogenize
test -x "${BIN}"

run_one() {
  local ev="$1"
  local dest="${OUT}/ev${ev}"
  mkdir -p "${dest}"
  echo "=== yang-a3 64x64x121 E_void=${ev} ==="
  srun --ntasks=8 --cpus-per-task=8 "${BIN}" \
    --shape=yang-a3 --nx=64 --ny=64 --nz=121 --dx=1 \
    --E-solid=1.0 --nu-solid=0.3 --E-void="${ev}" --nu-void=0.3 \
    --n-el-iter=80 \
    --dump-dir="${dest}" \
    | tee "${dest}/run.log"
}

run_one 0.02
run_one 0.002
run_one 0.0002

echo "=== summary ==="
for d in ev0.02 ev0.002 ev0.0002; do
  echo "----- ${d} -----"
  grep -E 'shape |nu_|spd |volume_fraction|min_eig|el_iters|el_residual|converged' \
    "${OUT}/${d}/run.log" || true
done
echo YANG_CONTRAST_OK

#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --job-name=q2d-thresh
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Qualify archived step-600 2-D inverse fields. Not an optimization rerun.

set -euo pipefail
SRC="${SRC_DIR:-/flash/project_462001519/juaho/wt/openpfc-476}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-inv-476}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/q2d_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/inverse_homogenization/openpfc_homogenize"
A256="/scratch/project_462001519/juaho/openpfc-showcase/inverse2d_22112442/fields/inverse2d_h_0600.bin"
A1024="/scratch/project_462001519/juaho/openpfc-showcase/inverse2d_22112443/fields/inverse2d_h_0600.bin"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

mkdir -p "${OUT}/n256_cont" "${OUT}/n256_bin" "${OUT}/n1024_cont" "${OUT}/n1024_bin"
{
  echo "job ${SLURM_JOB_ID} host $(hostname)"
  echo "date $(date -Iseconds)"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
  echo "fields ${A256} ${A1024}"
  echo "E_solid=1 nu_solid=0.3 E_void=0.02 nu_void=0.3 n_el_iter=200"
} | tee "${OUT}/provenance.txt"

cmake --build "${BUILD}" -j "${SLURM_NTASKS}" --target openpfc_homogenize
test -x "${BIN}"

run_one() {
  local nx="$1" bin="$2" dest="$3" extra="$4"
  echo "=== ${dest} ==="
  srun --ntasks=8 --cpus-per-task=8 "${BIN}" \
    --nx="${nx}" --ny="${nx}" --nz=1 --dx=1 \
    --E-solid=1.0 --nu-solid=0.3 --E-void=0.02 --nu-void=0.3 \
    --n-el-iter=200 \
    --load-bin="${bin}" ${extra} \
    --dump-dir="${dest}" \
    | tee "${dest}/run.log"
}

run_one 256 "${A256}" "${OUT}/n256_cont" ""
run_one 256 "${A256}" "${OUT}/n256_bin" "--binarize=0.5"
run_one 1024 "${A1024}" "${OUT}/n1024_cont" ""
run_one 1024 "${A1024}" "${OUT}/n1024_bin" "--binarize=0.5"

echo "=== summary ==="
for d in n256_cont n256_bin n1024_cont n1024_bin; do
  echo "----- ${d} -----"
  grep -E 'shape |nu_|spd |perc_|volume|min_eig|solid_components|grey|opening' \
    "${OUT}/${d}/run.log" || true
done

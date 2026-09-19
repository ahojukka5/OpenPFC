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
#SBATCH --time=01:00:00
#SBATCH --job-name=yang-a3-43
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# OpenPFC #43: Yang 2015 A3 3-D re-entrant honeycomb.
# Resolutions frozen before Poisson from Lz/Lx = 2(H-L cos θ)/(2 L sin θ)
# ≈ 1.896: 64x64x121 and 96x96x182. Do not run until the density gate
# in Catch2 [yang] passes.

set -euo pipefail
SRC="${SRC_DIR:-/flash/project_462001519/juaho/wt/openpfc-yang}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-yang}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/yang43_${SLURM_JOB_ID}}"
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

mkdir -p "${OUT}/n64" "${OUT}/n96"
{
  echo "job ${SLURM_JOB_ID} host $(hostname)"
  echo "date $(date -Iseconds)"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
  echo "Yang A3 H=7.74 L=3.78 t=0.8 theta=45deg"
  echo "grids 64x64x121 and 96x96x182"
  echo "E_solid=1 nu=0.3 E_void=0.02"
} | tee "${OUT}/provenance.txt"

cmake --build "${BUILD}" -j "${SLURM_NTASKS}" --target openpfc_homogenize
test -x "${BIN}"

run_one() {
  local nx="$1" ny="$2" nz="$3" dest="$4"
  echo "=== yang-a3 ${nx}x${ny}x${nz} ==="
  srun --ntasks=8 --cpus-per-task=8 "${BIN}" \
    --shape=yang-a3 --nx="${nx}" --ny="${ny}" --nz="${nz}" --dx=1 \
    --E-solid=1.0 --nu-solid=0.3 --E-void=0.02 --nu-void=0.3 \
    --dump-dir="${dest}" \
    | tee "${dest}/run.log"
}

run_one 64 64 121 "${OUT}/n64"
run_one 96 96 182 "${OUT}/n96"

echo "=== summary ==="
for d in n64 n96; do
  echo "----- ${d} -----"
  grep -E 'shape |nu_|spd |perc_|volume|min_eig|solid_components|H ' \
    "${OUT}/${d}/run.log" || true
done

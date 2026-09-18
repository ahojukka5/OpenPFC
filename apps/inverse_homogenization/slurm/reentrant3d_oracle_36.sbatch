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
#SBATCH --time=00:30:00
#SBATCH --job-name=reent3d-36
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# OpenPFC #36: Yang/Evans 3-D re-entrant honeycomb forward oracle.
# Not inverse design. Resolutions 32^3 and 64^3 predeclared.

set -euo pipefail
SRC="${SRC_DIR:-/flash/project_462001519/juaho/wt/openpfc-476}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-inv-476}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/oracle36_${SLURM_JOB_ID}}"
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

mkdir -p "${OUT}/n32" "${OUT}/n64"
{
  echo "job ${SLURM_JOB_ID} host $(hostname) ranks ${SLURM_NTASKS}"
  echo "date $(date -Iseconds)"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
  echo "bin ${BIN}"
  echo "shape reentrant-3d thickness=0.06 inset=0.14 z=0.10,0.30,0.70,0.90"
  echo "E_solid=1 nu_solid=0.3 E_void=0.02 nu_void=0.3"
  echo "citation Yang et al. IJSS 69-70 (2015) 475-490; Evans et al. 1994"
  module list
} | tee "${OUT}/provenance.txt"

if [[ ! -f "${BUILD}/CMakeCache.txt" ]]; then
  cmake -S "${SRC}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DHeffte_DIR=/users/juaho/opt/heffte/2.4.1-rocm/lib64/cmake/Heffte \
    -DOpenPFC_ENABLE_HIP=OFF \
    -DOpenPFC_BUILD_TESTS=ON \
    -DOpenPFC_BUILD_APPS=ON \
    -DOpenPFC_BUILD_EXAMPLES=OFF
fi
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" \
  --target openpfc_homogenize test_inverse_homogenization
test -x "${BIN}"

echo "=== Catch2 geometry ==="
srun --ntasks=1 --cpus-per-task=8 \
  "${BUILD}/apps/inverse_homogenization/test_inverse_homogenization" \
  "[reentrant3d]"

run_one() {
  local n="$1"
  local dest="$2"
  echo "=== reentrant-3d ${n}^3 ==="
  srun --ntasks=8 --cpus-per-task=8 "${BIN}" \
    --shape=reentrant-3d --nx="${n}" --ny="${n}" --nz="${n}" --dx=1 \
    --E-solid=1.0 --nu-solid=0.3 --E-void=0.02 --nu-void=0.3 \
    --thickness=0.06 --inset=0.14 \
    --dump-dir="${dest}" \
    | tee "${dest}/run.log"
}

run_one 32 "${OUT}/n32"
run_one 64 "${OUT}/n64"

echo "=== done ${OUT} ==="
for d in n32 n64; do
  echo "----- ${d} -----"
  grep -E 'shape |nu_|spd |perc_|volume|min_eig|solid_components|converged' \
    "${OUT}/${d}/run.log" || true
done

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
#SBATCH --job-name=fwd-oracle-31
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# OpenPFC #31: forward-homogenize a prescribed 3-D rotating-cube cell
# at 32^3 and 64^3. Not an inverse-design run.

set -euo pipefail
SRC="${SRC_DIR:-/flash/project_462001519/juaho/wt/openpfc-476}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-inv-476}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/oracle31_${SLURM_JOB_ID}}"
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
  echo "shape rotating-cubes half=0.200 angle=0.45"
  echo "E_solid=1 nu_solid=0.3 E_void=0.05 nu_void=0.3"
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
  --target openpfc_homogenize test_inverse_homogenization test_homogenization
test -x "${BIN}"

echo "=== Catch2 rotating-cube geometry ==="
srun --ntasks=1 --cpus-per-task=8 \
  "${BUILD}/apps/inverse_homogenization/test_inverse_homogenization" \
  "[inverse][auxetic][geometry]"

run_one() {
  local n="$1"
  local dest="$2"
  echo "=== rotating-cubes ${n}^3 ==="
  srun --ntasks=8 --cpus-per-task=8 "${BIN}" \
    --shape=rotating-cubes --nx="${n}" --ny="${n}" --nz="${n}" --dx=1 \
    --E-solid=1.0 --nu-solid=0.3 --E-void=0.05 --nu-void=0.3 \
    --half=0.200 --angle=0.45 \
    --dump-dir="${dest}" \
    | tee "${dest}/run.log"
}

run_one 32 "${OUT}/n32"
run_one 64 "${OUT}/n64"

echo "=== done ${OUT} ==="
echo "----- 32^3 -----"
grep -E 'nu_|spd |perc_|volume|min_eig|solid_components|converged' "${OUT}/n32/run.log" || true
echo "----- 64^3 -----"
grep -E 'nu_|spd |perc_|volume|min_eig|solid_components|converged' "${OUT}/n64/run.log" || true

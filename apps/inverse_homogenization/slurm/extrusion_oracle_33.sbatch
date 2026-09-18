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
#SBATCH --job-name=ext-oracle-33
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# OpenPFC #33: 2-D rotating-square seed vs its z-extrusion through the
# 3-D homogenizer. Not inverse design and not a 3-D metamaterial.

set -euo pipefail
SRC="${SRC_DIR:-/flash/project_462001519/juaho/wt/openpfc-476}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-inv-476}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/oracle33_${SLURM_JOB_ID}}"
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

mkdir -p "${OUT}/n32_nz1" "${OUT}/n64_nz1" "${OUT}/n32_nz4" "${OUT}/n64_nz4" \
         "${OUT}/homogeneous"
{
  echo "job ${SLURM_JOB_ID} host $(hostname) ranks ${SLURM_NTASKS}"
  echo "date $(date -Iseconds)"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
  echo "bin ${BIN}"
  echo "shape rotating-squares half=0.200 angle=0.45"
  echo "E_solid=1 nu_solid=0.3 E_void=0.02 nu_void=0.3"
  echo "2-D reference job 21956076 nu_eff=-0.123 grid 64x64x1"
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

echo "=== Catch2 geometry ==="
srun --ntasks=1 --cpus-per-task=8 \
  "${BUILD}/apps/inverse_homogenization/test_inverse_homogenization" \
  "[inverse][auxetic][geometry]"

echo "=== homogeneous isotropic sanity 16^3 ==="
srun --ntasks=1 --cpus-per-task=8 "${BIN}" \
  --shape=homogeneous --nx=16 --ny=16 --nz=16 --volume=1 \
  --E-solid=1.0 --nu-solid=0.3 --E-void=1.0 --nu-void=0.3 \
  --dump-dir="${OUT}/homogeneous" \
  | tee "${OUT}/homogeneous/run.log"

run_sq() {
  local nx="$1" ny="$2" nz="$3" dest="$4"
  echo "=== rotating-squares ${nx}x${ny}x${nz} ==="
  srun --ntasks=8 --cpus-per-task=8 "${BIN}" \
    --shape=rotating-squares --nx="${nx}" --ny="${ny}" --nz="${nz}" --dx=1 \
    --E-solid=1.0 --nu-solid=0.3 --E-void=0.02 --nu-void=0.3 \
    --half=0.200 --angle=0.45 \
    --dump-dir="${dest}" \
    | tee "${dest}/run.log"
}

run_sq 32 32 1 "${OUT}/n32_nz1"
run_sq 64 64 1 "${OUT}/n64_nz1"
run_sq 32 32 4 "${OUT}/n32_nz4"
run_sq 64 64 4 "${OUT}/n64_nz4"

echo "=== done ${OUT} ==="
for d in n32_nz1 n64_nz1 n32_nz4 n64_nz4 homogeneous; do
  echo "----- ${d} -----"
  grep -E 'shape |nu_|spd |perc_|volume|min_eig|solid_components|converged' \
    "${OUT}/${d}/run.log" || true
done

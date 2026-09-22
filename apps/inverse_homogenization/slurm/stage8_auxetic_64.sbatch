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
#SBATCH --time=02:00:00
#SBATCH --job-name=inv-aux-64
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Frozen first experiment,  / OpenPFC #28.
# Do not retune after seeing the result.

set -euo pipefail
SRC="${SRC_DIR:-/flash/project_462001519/juaho/wt/openpfc-476}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-inv-476}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/auxetic64_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/inverse_homogenization/openpfc_inverse_homogenize"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

mkdir -p "${OUT}/fields"
{
  echo "job ${SLURM_JOB_ID} host $(hostname) ranks ${SLURM_NTASKS}"
  echo "date $(date -Iseconds)"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
  echo "bin ${BIN}"
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
  --target openpfc_inverse_homogenize test_homogenization
test -x "${BIN}"
srun --ntasks=1 --cpus-per-task=8 \
  "${BUILD}/apps/common/test_homogenization" "[homogenization][algebra]"

srun --ntasks=8 --cpus-per-task=8 "${BIN}" \
  --nx=64 --ny=64 --nz=64 --dx=1 \
  --E-solid=1.0 --nu-solid=0.3 --E-void=0.05 --nu-void=0.3 \
  --target=auxetic --E-target=0.4 --nu-target=-0.2 \
  --volume=0.5 --init=spinodal --seed=1 \
  --ch-kappa=1 --ch-dt=0.2 --ch-steps=80 --init-amp=0.12 \
  --normalize=1 --dt=0.04 --max-delta=0.04 --project-volume=1 \
  --lambda-reg=0.05 --lambda-reg-end=0.2 --simp=1 --simp-end=2 \
  --steps=300 \
  --csv="${OUT}/history.csv" \
  --dump-dir="${OUT}/fields" --dump-every=10 \
  | tee "${OUT}/run.log"

echo "=== done ${OUT} ==="
tail -40 "${OUT}/run.log"

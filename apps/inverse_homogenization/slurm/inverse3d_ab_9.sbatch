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
#SBATCH --time=04:00:00
#SBATCH --job-name=inv3d-ab-9
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# OpenPFC #9: matched A/B inverse. Protocol frozen in
# evidence/2026-09-19-inverse3d-ab-protocol.json. Do not retune.

set -euo pipefail
SRC="${SRC_DIR:-/flash/project_462001519/juaho/wt/openpfc-9}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-inv-9}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/inv3d_ab_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/inverse_homogenization/openpfc_inverse_homogenize"
CT="${SRC}/apps/inverse_homogenization/evidence/2026-09-19-yang-a3-oracle-ev002.txt"

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
  echo "protocol evidence/2026-09-19-inverse3d-ab-protocol.json"
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
  --target openpfc_inverse_homogenize test_inverse_homogenization

COMMON=(
  --nx=64 --ny=64 --nz=121 --dx=1
  --E-solid=1.0 --nu-solid=0.3 --E-void=0.002 --nu-void=0.3
  --target=file --C-target-file="${CT}"
  --volume=0.2575865186 --init-volume=0.2575865186
  --normalize=1 --dt=0.04 --max-delta=0.04 --project-volume=1
  --lambda-reg=0.05 --lambda-reg-end=0.2 --simp=1 --simp-end=2
  --n-el-iter=400 --steps=300
)

echo "=== Catch2 target-io / spinodal ==="
srun --ntasks=1 --cpus-per-task=8 \
  "${BUILD}/apps/inverse_homogenization/test_inverse_homogenization" \
  "[target-io],[spinodal]" || true

run_arm() {
  local name="$1"
  shift
  mkdir -p "${OUT}/${name}/fields"
  echo "=== arm ${name} ==="
  srun --ntasks=8 --cpus-per-task=8 "${BIN}" \
    "${COMMON[@]}" "$@" \
    --csv="${OUT}/${name}/history.csv" \
    --dump-dir="${OUT}/${name}/fields" --dump-every=10 \
    | tee "${OUT}/${name}/run.log"
}

run_arm A --init=spinodal --seed=1 --ch-steps=80 --ch-kappa=1 --ch-dt=0.2 --init-amp=0.12
run_arm B --init=yang-a3

echo INVERSE3D_AB_DONE

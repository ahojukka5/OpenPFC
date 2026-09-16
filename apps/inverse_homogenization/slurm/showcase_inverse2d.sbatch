#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --job-name=invhom-show
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Issue #8: one LUMI-G node (8 MI250X GCDs) inverse-homogenization showcase.
# Heavy frames go to scratch, never flash. Configure the HIP tree on the
# login node first; this job compiles the HIP inverse binary and runs.
#
#   SHOWCASE_MODE=pilot   256^2, few steps (default)
#   SHOWCASE_MODE=prod    candidate 1024^2 until the 8-GCD pilot reports
#                         wall time and HBM; override with SHOWCASE_NX
#
# Required: HIP build tree already configured (FetchContent needs the network).

set -euo pipefail

SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-showcase-inv}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/openpfc-showcase/inverse2d_${SLURM_JOB_ID}}"
MODE="${SHOWCASE_MODE:-pilot}"

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

mkdir -p "${OUT}/fields"
{
  echo "job ${SLURM_JOB_ID} partition ${SLURM_JOB_PARTITION} host $(hostname)"
  echo "date $(date -Iseconds)"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
  echo "mode ${MODE}"
  echo "ntasks ${SLURM_NTASKS} gpus ${SLURM_GPUS_ON_NODE:-}"
  module list
} | tee "${OUT}/provenance.txt"

if [[ ! -f "${BUILD}/build.ninja" && ! -f "${BUILD}/Makefile" ]]; then
  echo "No HIP build tree at ${BUILD}. Configure on the login node:"
  echo "  ./scripts/build.sh --machine=lumi --partition=standard-g --no-test --no-submit --build-dir=${BUILD}"
  exit 2
fi
cmake --build "${BUILD}" -j 8 --target openpfc_inverse_homogenize_hip
BIN="${BUILD}/apps/inverse_homogenization/openpfc_inverse_homogenize_hip"
test -x "${BIN}"

CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"

if [[ "${MODE}" == "prod" ]]; then
  NX="${SHOWCASE_NX:-1024}"
  STEPS="${SHOWCASE_STEPS:-80}"
  EVERY="${SHOWCASE_DUMP_EVERY:-2}"
  if [[ -z "${SHOWCASE_NX:-}" ]]; then
    echo "SHOWCASE_NX unset: candidate ${NX}^2; re-choose from pilot HBM/wall" | tee -a "${OUT}/provenance.txt"
  fi
else
  NX="${SHOWCASE_NX:-256}"
  STEPS="${SHOWCASE_STEPS:-20}"
  EVERY="${SHOWCASE_DUMP_EVERY:-1}"
fi

CMD=(
  "${BIN}"
  --nx="${NX}" --ny="${NX}" --nz=1
  --target=auxetic --nu-target=0.4 --E-target=0.4
  --E-solid=1 --E-void=0.02 --nu-solid=0.3 --nu-void=0.3
  --init=rotating-squares --init-half=0.200 --init-angle=0.45
  --volume=0.5 --project-volume=1 --lambda-volume=1
  --lambda-reg=0.05 --dt=0.05 --steps="${STEPS}"
  --run-id=inverse2d --csv="${OUT}/history.csv"
  --dump-dir="${OUT}/fields" --dump-every="${EVERY}"
)
printf '%q ' "${CMD[@]}" | tee -a "${OUT}/provenance.txt"
echo | tee -a "${OUT}/provenance.txt"

srun --ntasks=8 --gpus-per-node=8 --cpu-bind="${CPU_BIND}" "${CMD[@]}" \
  | tee "${OUT}/run.log"

echo "=== done ${OUT} ==="
echo "Render with:"
echo "  python3 ${SRC}/apps/inverse_homogenization/scripts/render_inverse_showcase.py --run ${OUT}"

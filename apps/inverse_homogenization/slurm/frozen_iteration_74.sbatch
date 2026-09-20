#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#SBATCH --account=project_462001245
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=8
#SBATCH --time=02:30:00
#SBATCH --job-name=inverse-frozen74
#SBATCH --output=/scratch/project_462001245/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001245/juaho/logs/%x-%j.err

# Build/test first through scripts/build.sh. Submit this run-only recipe with
# --dependency=afterok:<build-job> --account=project_462001245 and explicit
# SOURCE_DIR, BINARY, BINARY_SOURCE_SHA, INITIAL_FIELD environment variables.
set -euo pipefail
test "${SLURM_JOB_ACCOUNT}" = project_462001245
: "${SOURCE_DIR:?}" "${BINARY:?}" "${BINARY_SOURCE_SHA:?}" "${INITIAL_FIELD:?}"
module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
module use "${HOME}/privatemodules"
module load heffte-rocm
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_EXPORT_ENV=ALL
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
unset OPENPFC_HEFFTE_RESHAPE_ALGORITHM OPENPFC_HEFFTE_USE_PENCILS
ROOT="/scratch/project_462001245/juaho/inverse-homogenization/frozen74_${SLURM_JOB_ID}"
mkdir "${ROOT}"
cp "${BINARY}" "${ROOT}/inverse_hip"
cp "${SOURCE_DIR}/apps/inverse_homogenization/scripts/qualify_frozen_iteration.py" "${ROOT}/driver.py"
cp "${INITIAL_FIELD}" "${ROOT}/initial_h190.bin"
cp "${SOURCE_DIR}/apps/inverse_homogenization/evidence/2026-09-19-yang-a3-oracle-ev002.txt" "${ROOT}/target.txt"
git -C "${SOURCE_DIR}" rev-parse HEAD > "${ROOT}/driver_revision.txt"
git -C "${SOURCE_DIR}" status --porcelain > "${ROOT}/driver_dirty.txt"
module list 2> "${ROOT}/modules.txt"
scontrol show job "${SLURM_JOB_ID}" > "${ROOT}/slurm.txt"
python3 "${ROOT}/driver.py" --binary "${ROOT}/inverse_hip" \
  --source-sha "${BINARY_SOURCE_SHA}" --target "${ROOT}/target.txt" \
  --initial-field "${ROOT}/initial_h190.bin" --output "${ROOT}/results" \
  --phase "${PHASE:-all}"

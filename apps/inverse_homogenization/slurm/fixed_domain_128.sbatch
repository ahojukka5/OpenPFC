#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#SBATCH --account=project_462001245
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=inverse128-fixed-domain
#SBATCH --output=/scratch/project_462001245/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001245/juaho/logs/%x-%j.err
# Run-only fixed-domain control. Build/test the executable via scripts/build.sh.
set -euo pipefail
test "${SLURM_JOB_ACCOUNT}" = project_462001245
: "${SOURCE_DIR:?}" "${BINARY:?}" "${BINARY_SOURCE_SHA:?}" "${INITIAL64:?}"
module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
module use "${HOME}/privatemodules"
module load heffte-rocm
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPICH_GPU_SUPPORT_ENABLED=1 SLURM_EXPORT_ENV=ALL OMP_NUM_THREADS=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
OUT="/scratch/project_462001245/juaho/inverse-homogenization/fixed128_${SLURM_JOB_ID}"
mkdir "${OUT}"
cp "${BINARY}" "${OUT}/inverse_hip"
cp "${INITIAL64}" "${OUT}/initial64.bin"
cp "${SOURCE_DIR}/apps/inverse_homogenization/evidence/2026-09-19-yang-a3-oracle-ev002.txt" "${OUT}/target.txt"
cp "${SOURCE_DIR}/apps/inverse_homogenization/evidence/2026-09-20-fixed-domain-protocol.json" "${OUT}/protocol.json"
cp "${SOURCE_DIR}/apps/inverse_homogenization/scripts/refine_periodic_initial.py" "${OUT}/refine.py"
printf '%s\n' "${BINARY_SOURCE_SHA}" > "${OUT}/binary-source.txt"
git -C "${SOURCE_DIR}" rev-parse HEAD > "${OUT}/driver-source.txt"
git -C "${SOURCE_DIR}" status --porcelain > "${OUT}/driver-dirty.txt"
python3 "${OUT}/refine.py" "${OUT}/initial64.bin" "${OUT}/initial128.bin" \
 --shape 64 64 121 --dx=1
sha256sum "${OUT}/inverse_hip" "${OUT}/target.txt" "${OUT}/initial64.bin" \
 "${OUT}/initial128.bin" "${OUT}/refine.py" > "${OUT}/checksums.txt"
scontrol show job "${SLURM_JOB_ID}" > "${OUT}/slurm.txt"
module list 2> "${OUT}/modules.txt"
ARGS=(--nx=128 --ny=128 --nz=242 --dx=0.5
 --E-solid=1 --nu-solid=0.3 --E-void=0.002 --nu-void=0.3
 --target=file --C-target-file="${OUT}/target.txt"
 --volume=0.2575865186 --init-volume=0.2575865186 --lambda-volume=1 --epsilon=2
 --normalize=1 --dt=0.04 --max-delta=0.04 --project-volume=1
 --lambda-reg=0.05 --lambda-reg-end=0.2 --simp=1 --simp-end=2
 --n-el-iter=400 --init=uniform --load-bin="${OUT}/initial128.bin"
 --continuation-steps=300 --max-steps=5000 --conv-window=20
 --verify-convergence-steps=100 --tol-design=0.0001
 --tol-objective=0.000001 --tol-tensor=0.0001 --dump-every=20
 --run-id=fixed128 --csv="${OUT}/history.csv" --dump-dir="${OUT}/fields")
printf '%q ' "${OUT}/inverse_hip" "${ARGS[@]}" > "${OUT}/command.sh"
srun --export=ALL --ntasks=16 --gpus-per-node=8 "${OUT}/inverse_hip" "${ARGS[@]}" \
 > "${OUT}/run.log" 2> "${OUT}/run.err"

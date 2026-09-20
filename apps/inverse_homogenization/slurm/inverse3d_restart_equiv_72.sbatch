#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#SBATCH --account=project_462001245
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --job-name=inverse-restart-72
#SBATCH --output=/scratch/project_462001245/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001245/juaho/logs/%x-%j.err
# Build first through scripts/build.sh. This batch only runs experiments.
# Required: INVERSE_SRC, INVERSE_BINARY, INVERSE_BINARY_SOURCE.
# Optional: INVERSE_CPU_BINARY (run the same controls on two CPU MPI ranks).
set -euo pipefail
test "${SLURM_JOB_ACCOUNT}" = project_462001245
: "${INVERSE_SRC:?Set the clean source checkout}"
: "${INVERSE_BINARY:?Build the HIP executable through scripts/build.sh first}"
: "${INVERSE_BINARY_SOURCE:?Record the immutable binary build source}"
OUT="${INVERSE_OUT:-/scratch/project_462001245/juaho/inverse-homogenization/restart72_${SLURM_JOB_ID}}"
mkdir "$OUT"
module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
module use "${HOME}/privatemodules"
module load heffte-rocm
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPICH_GPU_SUPPORT_ENABLED=1 SLURM_EXPORT_ENV=ALL OMP_NUM_THREADS=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true
cp "$INVERSE_BINARY" "$OUT/inverse_hip"
cp "$INVERSE_SRC/apps/inverse_homogenization/scripts/replay_checkpoint_state.py" "$OUT/driver.py"
cp "$INVERSE_SRC/apps/inverse_homogenization/evidence/2026-09-19-yang-a3-oracle-ev002.txt" "$OUT/target.txt"
cp "$0" "$OUT/command.sbatch"
module -t list > "$OUT/modules.txt" 2>&1
scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm.txt"
{
  printf 'binary_source=%s\n' "$INVERSE_BINARY_SOURCE"
  git -C "$INVERSE_SRC" rev-parse HEAD
  git -C "$INVERSE_SRC" status --porcelain
} > "$OUT/provenance.txt"
sha256sum "$OUT/inverse_hip" "$OUT/driver.py" "$OUT/target.txt" > "$OUT/checksums.txt"
python3.11 "$OUT/driver.py" --binary "$OUT/inverse_hip" --target "$OUT/target.txt" \
  --output "$OUT/hip" --ranks 8 > "$OUT/hip.log" 2>&1
if [[ -n "${INVERSE_CPU_BINARY:-}" ]]; then
  cp "$INVERSE_CPU_BINARY" "$OUT/inverse_cpu"
  sha256sum "$OUT/inverse_cpu" >> "$OUT/checksums.txt"
  python3.11 "$OUT/driver.py" --cpu --binary "$OUT/inverse_cpu" --target "$OUT/target.txt" \
    --output "$OUT/cpu" --ranks 2 > "$OUT/cpu.log" 2>&1
fi

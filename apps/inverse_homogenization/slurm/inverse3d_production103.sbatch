#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#SBATCH --account=project_462001245
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --job-name=inverse-production103
#SBATCH --output=/scratch/project_462001245/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001245/juaho/logs/%x-%j.err
# Fresh start only. Preserve this allocation and inspect CURRENT before resuming.
set -euo pipefail
test "${SLURM_JOB_ACCOUNT}" = project_462001245
: "${CAMPAIGN_SOURCE:?Set the committed runner checkout}"
: "${GRID:?128 or 256}"
case "$GRID" in
128) NZ=242; DX=.5; RANKS=16; NODES=2 ;;
256) NZ=484; DX=.25; RANKS=128; NODES=16 ;;
*) exit 2 ;;
esac
test "$SLURM_NTASKS" = "$RANKS"
test "$SLURM_JOB_NUM_NODES" = "$NODES"
module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
module use "${HOME}/privatemodules"
module load heffte-rocm
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPICH_GPU_SUPPORT_ENABLED=1 SLURM_EXPORT_ENV=ALL OMP_NUM_THREADS=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true
OUT=/scratch/project_462001245/juaho/inverse-homogenization/production103_${GRID}_${SLURM_JOB_ID}
mkdir "$OUT"
cp /flash/project_462001519/juaho/build/openpfc-qualified-checkpoint/apps/inverse_homogenization/openpfc_inverse_homogenize_hip "$OUT/inverse_hip"
cp /scratch/project_462001245/juaho/inverse-homogenization/n0fixed128_22180901/initial.bin "$OUT/initial128.bin"
cp "$CAMPAIGN_SOURCE/apps/inverse_homogenization/evidence/2026-09-19-yang-a3-oracle-ev002.txt" "$OUT/target.txt"
cp "$CAMPAIGN_SOURCE/apps/inverse_homogenization/evidence/production103/protocol.md" "$OUT/protocol.md"
cp "$0" "$OUT/command.sbatch"
git -C "$CAMPAIGN_SOURCE" show a342b407005c778703704ee0b39fcd1faa3e9877:apps/inverse_homogenization/scripts/refine_periodic_initial.py > "$OUT/refine_periodic_initial.py"
module -t list > "$OUT/modules.txt" 2>&1
scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm.txt"
{
 printf 'binary_source=43d0d4224340532594f2ae6f618a7e661cbd0ff9\n'
 git -C "$CAMPAIGN_SOURCE" rev-parse HEAD
 git -C "$CAMPAIGN_SOURCE" status --porcelain
} > "$OUT/provenance.txt"
cd "$OUT"
sha256sum -c <<'HASHES'
590a350fae0cc8d27511b415aeb1337bea72724fe0c8d8c814127b90ea674fa6  inverse_hip
ecf733fbbbe8e0f831bc3daa6352090c6d1775eab82d58f4b6f28d150ff8f65f  initial128.bin
81cec52af7d770af2244e7b6c00a7d3e96f9c6b3df921ba3badfbf3e4f75ebc8  target.txt
HASHES
if test "$GRID" = 256; then
 python3.11 refine_periodic_initial.py initial128.bin initial.bin --shape 128 128 242 --dx .5
else
 cp initial128.bin initial.bin
fi
sha256sum inverse_hip initial128.bin initial.bin target.txt command.sbatch protocol.md refine_periodic_initial.py > checksums.txt
ARGS=(--nx="$GRID" --ny="$GRID" --nz="$NZ" --dx="$DX"
 --E-solid=1 --nu-solid=.3 --E-void=.002 --nu-void=.3
 --target=file --C-target-file="$OUT/target.txt"
 --volume=.2575865186 --init-volume=.2575865186 --lambda-volume=1
 --normalize=0 --dt=.01 --max-delta=.04 --project-volume=1
 --lambda-reg=.05 --lambda-reg-end=.2 --simp=1 --simp-end=2
 --epsilon=2 --n-el-iter=400 --init=uniform --load-bin="$OUT/initial.bin"
 --continuation-steps=1197 --max-steps=20000 --conv-window=80 --verify-convergence-steps=400
 --tol-design=.000025 --tol-objective=.00000025 --tol-tensor=.000025
 --dump-every=80 --run-id="production${GRID}" --csv="$OUT/history-01.csv"
 --dump-dir="$OUT/fields" --checkpoint-dir="$OUT/checkpoint" --stop-after=1500)
printf '%q ' "$OUT/inverse_hip" "${ARGS[@]}" > command.sh
srun --export=ALL --ntasks="$RANKS" --gpus-per-node=8 "$OUT/inverse_hip" "${ARGS[@]}" > run-01.log 2> run-01.err

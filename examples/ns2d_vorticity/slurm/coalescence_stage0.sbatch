#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Stage-0 island coalescence (#113), eta=nu=0.01 only.
# CPU tessellation on LUMI-C. Do not submit this to standard-g.
#
#SBATCH --account=project_462001519
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=mhd-ic-113
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail

RUN="${1:?usage: sbatch coalescence_stage0.sbatch RUN}"
# RUN is ic128_cfl04 | ic256_cfl04 | ic256_cfl02

ROOT=/pfs/lustref1/flash/project_462001519/juaho/dev/worktrees/openpfc/113
BUILD=/flash/project_462001519/juaho/build/openpfc-113-cpu
OUTROOT=/scratch/project_462001519/juaho/mhd2d-113
MHD2D="${BUILD}/examples/ns2d_vorticity/mhd2d"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8

mkdir -p "${OUTROOT}" /scratch/project_462001519/juaho/logs

case "${RUN}" in
  ic128_cfl04)
    OUT="${OUTROOT}/ic128_nu001_cfl04"
    N=128; CFL=0.4; STEPS=2038; EVERY=20
    ;;
  ic256_cfl04)
    OUT="${OUTROOT}/ic256_nu001_cfl04"
    N=256; CFL=0.4; STEPS=4074; EVERY=40
    ;;
  ic256_cfl02)
    OUT="${OUTROOT}/ic256_nu001_cfl02"
    N=256; CFL=0.2; STEPS=8148; EVERY=80
    ;;
  *)
    echo "unknown RUN=${RUN}" >&2
    exit 1
    ;;
esac

echo "job=${SLURM_JOB_ID:-none} run=${RUN} N=${N} cfl=${CFL} bin=${MHD2D}"
srun --cpu-bind=cores "${MHD2D}" \
  --case island_coalescence --N "${N}" --cfl "${CFL}" \
  --nu 0.01 --eta 0.01 --steps "${STEPS}" \
  --diag "${EVERY}" --dump "${EVERY}" \
  --outdir "${OUT}"
echo "done ${OUT}"

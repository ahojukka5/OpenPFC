#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Stage-2 island coalescence (#113): the third resistivity.
# eta = nu = 0.0025 at fixed Pm = 1. Two resolutions; no eta is admitted
# from a single unresolved grid. Geometry, perturbation and the frozen
# bracket are NOT retuned for this eta. CPU LUMI-C; never standard-g.
#
#SBATCH --account=project_462001519
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mhd-ic-113-s2
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail

RUN="${1:?usage: sbatch coalescence_stage1.sbatch RUN}"
# RUN is ic512_e00025_cfl04 | ic512_e00025_cfl02 | ic1024_e00025_cfl04

BUILD=/flash/project_462001519/juaho/build/openpfc-113-cpu
OUTROOT=/scratch/project_462001519/juaho/mhd2d-113
MHD2D="${BUILD}/examples/ns2d_vorticity/mhd2d"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8

mkdir -p "${OUTROOT}" /scratch/project_462001519/juaho/logs

# dt is set by the CFL condition and the grid, not by eta, so the step
# counts reproduce the Stage-0 end time t ~ 40 at each (N, CFL).
case "${RUN}" in
  ic512_e00025_cfl04)
    OUT="${OUTROOT}/ic512_e00025_cfl04"
    N=512; CFL=0.4; STEPS=8148; EVERY=80
    ;;
  ic1024_e00025_cfl04)
    OUT="${OUTROOT}/ic1024_e00025_cfl04"
    N=1024; CFL=0.4; STEPS=16297; EVERY=160
    ;;
  ic512_e00025_cfl02)
    OUT="${OUTROOT}/ic512_e00025_cfl02"
    N=512; CFL=0.2; STEPS=16297; EVERY=160
    ;;
  *)
    echo "unknown RUN=${RUN}" >&2
    exit 1
    ;;
esac

echo "job=${SLURM_JOB_ID:-none} run=${RUN} N=${N} cfl=${CFL} eta=0.0025 bin=${MHD2D}"
srun --cpu-bind=cores "${MHD2D}" \
  --case island_coalescence --N "${N}" --cfl "${CFL}" \
  --nu 0.0025 --eta 0.0025 --steps "${STEPS}" \
  --diag "${EVERY}" --dump "${EVERY}" \
  --outdir "${OUT}"
echo "done ${OUT}"

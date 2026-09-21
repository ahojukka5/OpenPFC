#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Stage-2 analysis (#113), eta = nu = 0.0025. Runs each grid twice: at the
# default X/O classification floor and at --degen-ratio 0.01, exactly as
# Stage 1 did. The frozen progress bracket is unchanged.
#
#SBATCH --account=project_462001519
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mhd-ic-an-s2
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail
RUN="${1:?usage: sbatch coalescence_analyze_stage2.sbatch RUN}"
ROOT=/pfs/lustref1/flash/project_462001519/juaho/dev/worktrees/openpfc/coalescence-stage1
SCR="${ROOT}/examples/ns2d_vorticity/scripts"
OUTROOT=/scratch/project_462001519/juaho/mhd2d-113
ETA=0.0025

module purge
module load LUMI/25.09 partition/C cray-python
export PYTHONPATH=
export OMP_NUM_THREADS=8

case "${RUN}" in
  ic512_e00025_cfl04)  DIR="${OUTROOT}/ic512_e00025_cfl04";  N=512  ;;
  ic1024_e00025_cfl04) DIR="${OUTROOT}/ic1024_e00025_cfl04"; N=1024 ;;
  *) echo "unknown RUN=${RUN}" >&2; exit 1 ;;
esac

echo "job=${SLURM_JOB_ID:-none} run=${RUN} N=${N} eta=${ETA}"

echo "### default classification floor"
python3 "${SCR}/island_flux_budget.py" --dir "${DIR}" --n "${N}" --eta "${ETA}" \
  --t-max 40 --family coalescence --json-out "${DIR}/flux_default.json"
python3 "${SCR}/sheet_scaling.py" --dir "${DIR}" --n "${N}" --eta "${ETA}" \
  --t-max 40 --family coalescence --t-lo 0 --t-hi 40 \
  --json-out "${DIR}/sheet_default.json"

echo "### degen-ratio 0.01 (Stage-1 sensitivity arm)"
python3 "${SCR}/island_flux_budget.py" --dir "${DIR}" --n "${N}" --eta "${ETA}" \
  --t-max 40 --family coalescence --degen-ratio 0.01 \
  --json-out "${DIR}/flux_degen001.json"
python3 "${SCR}/sheet_scaling.py" --dir "${DIR}" --n "${N}" --eta "${ETA}" \
  --t-max 40 --family coalescence --t-lo 0 --t-hi 40 --degen-ratio 0.01 \
  --json-out "${DIR}/sheet_degen001.json"
echo "done ${DIR}"

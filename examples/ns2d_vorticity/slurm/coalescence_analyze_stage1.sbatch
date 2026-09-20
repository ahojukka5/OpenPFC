#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Topology / flux / sheet / event analysis for Stage-1 coalescence (#113),
# eta = nu = 0.005. Definitions and the frozen progress bracket are the
# Stage-0 ones, unchanged. CPU only; requires cray-python 3.11 + numpy.
#
#SBATCH --account=project_462001519
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=mhd-ic-an-s1
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail

RUN="${1:?usage: sbatch coalescence_analyze_stage1.sbatch RUN}"
ROOT=/pfs/lustref1/flash/project_462001519/juaho/dev/worktrees/openpfc/coalescence-stage1
OUTROOT=/scratch/project_462001519/juaho/mhd2d-113
SCR="${ROOT}/examples/ns2d_vorticity/scripts"
ETA=0.005

module purge
module load LUMI/25.09 partition/C cray-python
export PYTHONPATH=
export OMP_NUM_THREADS=8
PY="${PYTHON:-python3}"

case "${RUN}" in
  ic256_e0005_cfl04) DIR="${OUTROOT}/ic256_e0005_cfl04"; N=256 ;;
  ic256_e0005_cfl02) DIR="${OUTROOT}/ic256_e0005_cfl02"; N=256 ;;
  ic512_e0005_cfl04) DIR="${OUTROOT}/ic512_e0005_cfl04"; N=512 ;;
  *) echo "unknown RUN=${RUN}" >&2; exit 1 ;;
esac

echo "job=${SLURM_JOB_ID:-none} run=${RUN} N=${N} eta=${ETA} dir=${DIR}"
"${PY}" "${SCR}/island_flux_budget.py" \
  --dir "${DIR}" --n "${N}" --eta "${ETA}" --t-max 40 \
  --family coalescence --json-out "${DIR}/flux.json"
"${PY}" "${SCR}/sheet_scaling.py" \
  --dir "${DIR}" --n "${N}" --eta "${ETA}" --t-max 40 \
  --family coalescence --t-lo 0 --t-hi 40 \
  --json-out "${DIR}/sheet.json"
"${PY}" "${SCR}/coalescence_event.py" \
  --json "${DIR}/sheet.json" --json-out "${DIR}/event.json"
echo "done ${DIR}"

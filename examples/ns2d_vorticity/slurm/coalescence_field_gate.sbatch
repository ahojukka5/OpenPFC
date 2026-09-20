#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Field-level common-band spatial gate for coalescence Stage 0 (#113).
#SBATCH --account=project_462001519
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --job-name=mhd-ic-field
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail

ROOT=/pfs/lustref1/flash/project_462001519/juaho/dev/worktrees/openpfc/113
OUT=/scratch/project_462001519/juaho/mhd2d-113
SCR="${ROOT}/examples/ns2d_vorticity/scripts"

module purge
module load LUMI/25.09 partition/C cray-python
export PYTHONPATH=
export OMP_NUM_THREADS=8
PY="${PYTHON:-python3}"

C="${OUT}/ic128_nu001_cfl04"
F="${OUT}/ic256_nu001_cfl04"

# Exact matched physical times because dt_128 = 2*dt_256.
# 460*(0.4*2pi/128) = 920*(0.4*2pi/256) = 9.032078879...
"${PY}" "${SCR}/compare_mhd_fields.py" \
  --coarse-dir "${C}" --coarse-n 128 --coarse-inc 460 \
  --fine-dir "${F}" --fine-n 256 --fine-inc 920 \
  --json-out "${OUT}/commonband_t09032.json"

# Developed first-transfer state, still well before the late-X/Bup degradation.
# t = 15.707963268...
"${PY}" "${SCR}/compare_mhd_fields.py" \
  --coarse-dir "${C}" --coarse-n 128 --coarse-inc 800 \
  --fine-dir "${F}" --fine-n 256 --fine-inc 1600 \
  --json-out "${OUT}/commonband_t15708.json"

"${PY}" "${SCR}/coalescence_field_gate.py" \
  "${OUT}/commonband_t09032.json" "${OUT}/commonband_t15708.json"

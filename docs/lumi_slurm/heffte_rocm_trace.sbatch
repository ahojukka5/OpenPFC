#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --job-name=heffte-trace-build
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Compile the diagnostic HeFFTe tracing install on a GPU node.
# Does not replace the production heffte-rocm module.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
SRC="${OPENPFC_SRC:-/pfs/lustref1/flash/project_462001519/juaho/dev/worktrees/openpfc/openpfc-heffte-trace}"
cd "${SRC}"
chmod +x scripts/build_heffte_rocm_trace.sh
exec scripts/build_heffte_rocm_trace.sh

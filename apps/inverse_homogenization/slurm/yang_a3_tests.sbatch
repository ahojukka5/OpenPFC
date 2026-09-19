#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --job-name=yang-a3-tests
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Compile and run Yang A3 Catch2 geometry/density tests. No Poisson oracle.

set -euo pipefail
SRC="${SRC_DIR:-/flash/project_462001519/juaho/wt/openpfc-yang}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-yang}"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

cmake --build "${BUILD}" -j "${SLURM_NTASKS}" \
  --target test_inverse_homogenization openpfc_homogenize

echo "=== Catch2 [yang] ==="
srun --ntasks=1 --cpus-per-task=8 \
  "${BUILD}/apps/inverse_homogenization/test_inverse_homogenization" \
  "[yang]" -s
echo "=== Catch2 [reentrant3d] ==="
srun --ntasks=1 --cpus-per-task=8 \
  "${BUILD}/apps/inverse_homogenization/test_inverse_homogenization" \
  "[reentrant3d]"
echo YANG_TESTS_OK

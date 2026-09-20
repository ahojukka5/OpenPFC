#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Compile and run MHD tests for #113 on LUMI-C. Configure first on a
# login node (FetchContent). Do not use GPU partitions.
#
#SBATCH --account=project_462001519
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:30:00
#SBATCH --job-name=mhd-113-build
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail

ROOT=/pfs/lustref1/flash/project_462001519/juaho/dev/worktrees/openpfc/113
BUILD=/flash/project_462001519/juaho/build/openpfc-113-cpu

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw cray-python lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CRAYPE_LINK_TYPE=dynamic
export CC=cc
export CXX=CC

cmake --build "${BUILD}" --parallel 8 --target mhd2d test_mhd2d ns2d_vorticity
ctest --test-dir "${BUILD}" -R 'mhd' --output-on-failure --parallel 8

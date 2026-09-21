#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=02:00:00
#SBATCH --job-name=openpfc-heffte-trace-h3d
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Issue #119: compile heat3d_spectral_hip against heffte-rocm-trace.
# Separate tree from production HIP and from the tungsten 1200-axis
# trace install. Skip the full test suite; this binary is diagnostic.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
SRC="${OPENPFC_SRC:-/pfs/lustref1/flash/project_462001519/juaho/dev/worktrees/openpfc/heffte-bw}"
BUILD_DIR="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-rocm-heffte-trace-heat3d}"
cd "${SRC}"
export HEFFTE_MODULE=heffte-rocm-trace/2.4.1
export BUILD_DIR
exec "${SRC}/scripts/build.sh" \
  --machine=lumi \
  --with-rocm \
  --no-test \
  --no-mpi-tests \
  --no-submit \
  --build-dir="${BUILD_DIR}" \
  --jobs=16

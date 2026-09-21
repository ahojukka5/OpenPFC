#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --job-name=osu-rocm-build
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Issue #119: GPU-buffer OSU 7.5 (osu_alltoall / osu_alltoallv).
# Compile on a GPU node. Does not replace any production binary.

set -euo pipefail

SRC="${OSU_SRC:-/flash/project_462001519/juaho/src/osu-micro-benchmarks}"
PREFIX="${OSU_PREFIX:-/flash/project_462001519/juaho/opt/osu-micro-benchmarks/7.5-rocm}"
BUILD="${OSU_BUILD:-/flash/project_462001519/juaho/build/osu-micro-benchmarks-7.5-rocm}"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export MPICH_GPU_SUPPORT_ENABLED=1
export CC=cc
export CXX=CC

ROCM_ROOT="${EBROOTROCM:-${ROCM_PATH:-}}"
if [[ -z "${ROCM_ROOT}" || ! -d "${ROCM_ROOT}" ]]; then
  for cand in /opt/rocm /opt/rocm-6.2.2 /opt/rocm-6.0.3; do
    if [[ -d "${cand}" ]]; then
      ROCM_ROOT="${cand}"
      break
    fi
  done
fi
if [[ -z "${ROCM_ROOT}" || ! -d "${ROCM_ROOT}" ]]; then
  echo "ROCm prefix not found (EBROOTROCM/ROCM_PATH unset)" >&2
  exit 2
fi

if [[ ! -x "${SRC}/configure" ]]; then
  echo "OSU sources missing: ${SRC}" >&2
  exit 2
fi

mkdir -p "${BUILD}" "${PREFIX}"
cd "${BUILD}"
"${SRC}/configure" \
  --prefix="${PREFIX}" \
  --enable-rocm \
  --with-rocm="${ROCM_ROOT}"
make -j16
make install

ls -l "${PREFIX}/libexec/osu-micro-benchmarks/mpi/collective/osu_alltoall" \
  "${PREFIX}/libexec/osu-micro-benchmarks/mpi/collective/osu_alltoallv"
echo "osu prefix=${PREFIX} rocm=${ROCM_ROOT}"

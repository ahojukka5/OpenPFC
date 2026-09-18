#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --job-name=openpfc-outbox-tests
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Compile the diagnostic tree and run the complex-outbox planner tests.
# Not a production HIP qualification.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
SRC="${OPENPFC_SRC:-/pfs/lustref1/flash/project_462001519/juaho/dev/worktrees/openpfc/openpfc-heffte-trace}"
BUILD_DIR=/flash/project_462001519/juaho/build/openpfc-lumi-rocm-heffte-trace
cd "${SRC}"
export HEFFTE_MODULE=heffte-rocm-trace/2.4.1
export BUILD_DIR
"${SRC}/scripts/build.sh" \
  --machine=lumi \
  --with-rocm \
  --no-test \
  --no-mpi-tests \
  --no-submit \
  --cmake-arg=-DOpenPFC_BUILD_TESTS=ON \
  --build-dir="${BUILD_DIR}" \
  --jobs=16

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm-trace/2.4.1
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

TEST_BIN="${BUILD_DIR}/tests/openpfc-tests"
if [[ ! -x "${TEST_BIN}" ]]; then
  TEST_BIN="${BUILD_DIR}/openpfc-tests"
fi
echo "Running ${TEST_BIN} [complex_outbox]"
srun --ntasks=1 --gpus-per-node=1 "${TEST_BIN}" "[complex_outbox]~[MPI]"
echo "Running 4-rank MPI roundtrip"
srun --ntasks=4 --gpus-per-node=4 "${TEST_BIN}" "[complex_outbox][MPI]"
echo "complex-outbox tests passed"

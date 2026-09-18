#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Diagnostic HeFFTe ROCm install for OpenPFC issue #13 / research #486.
# Does NOT replace ~/opt/heffte/2.4.1-rocm or the heffte-rocm module.
#
# Differences from scripts/build_heffte_rocm.sh:
#   - Heffte_ENABLE_TRACING=ON
#   - cmake/heffte-2.4.1-gpu-trace-sync.patch so MPI_Wtime intervals wait for
#     rocFFT kernels (stock tracing records enqueue time only)
#   - install prefix .../2.4.1-rocm-trace
#   - module heffte-rocm-trace/2.4.1
#
# Submit this from a GPU partition. Do not compile HeFFTe on a login node.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HEFFTE_VERSION="${HEFFTE_VERSION:-2.4.1}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ROCM_ARCH="${ROCM_ARCH:-gfx90a}"
JOBS="${JOBS:-16}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/opt/heffte/${HEFFTE_VERSION}-rocm-trace}"
MODULE_FILE="${MODULE_FILE:-$HOME/privatemodules/heffte-rocm-trace/${HEFFTE_VERSION}.lua}"
WORK_ROOT="${WORK_ROOT:-/flash/project_462001519/${USER}/heffte-build-trace}"
LUMI_STACK="${LUMI_STACK:-LUMI/25.09}"
MPI_INC="${MPI_INC:-/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include}"
GPU_AWARE_MPI="${GPU_AWARE_MPI:-ON}"
PATCH="${PATCH:-${REPO_ROOT}/cmake/heffte-2.4.1-gpu-trace-sync.patch}"

die() { echo "ERROR: $*" >&2; exit 1; }

INSTALL_PREFIX="$(readlink -f "${INSTALL_PREFIX}" 2>/dev/null || echo "${INSTALL_PREFIX}")"
SRC_DIR="${WORK_ROOT}/src/heffte-${HEFFTE_VERSION}"
BUILD_DIR="${WORK_ROOT}/build/heffte-${HEFFTE_VERSION}-rocm-trace"
ARCHIVE="${WORK_ROOT}/src/v${HEFFTE_VERSION}.tar.gz"
VANILLA="${WORK_ROOT}/src/heffte-${HEFFTE_VERSION}-vanilla"

echo "HeFFTe ROCm TRACE build (diagnostic; does not replace heffte-rocm)"
echo "  version:       ${HEFFTE_VERSION}"
echo "  install:       ${INSTALL_PREFIX}"
echo "  modulefile:    ${MODULE_FILE}"
echo "  patch:         ${PATCH}"
echo "  GPU-aware MPI: ${GPU_AWARE_MPI}"

if ! command -v module >/dev/null 2>&1; then
  for init_file in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
    if [[ -f "${init_file}" ]]; then
      # shellcheck source=/dev/null
      source "${init_file}"
      break
    fi
  done
fi
command -v module >/dev/null 2>&1 || die "Lmod 'module' command not found"

module --force purge
module load "${LUMI_STACK}" partition/G cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
command -v hipcc >/dev/null 2>&1 || die "hipcc not found after loading ROCm"
[[ -f "${MPI_INC}/mpi.h" ]] || die "Cray MPICH mpi.h not found at ${MPI_INC}"
export CMAKE_PREFIX_PATH="${EBROOTROCM:-}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

mkdir -p "${WORK_ROOT}/src" "${WORK_ROOT}/build"
if [[ ! -d "${VANILLA}" ]]; then
  if [[ -d /flash/project_462001519/${USER}/heffte-build/src/heffte-${HEFFTE_VERSION} ]]; then
    echo "Copying existing HeFFTe ${HEFFTE_VERSION} sources"
    cp -a "/flash/project_462001519/${USER}/heffte-build/src/heffte-${HEFFTE_VERSION}" "${VANILLA}"
  else
    if [[ ! -f "${ARCHIVE}" ]]; then
      url="https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${HEFFTE_VERSION}.tar.gz"
      curl -fsSL -o "${ARCHIVE}" "${url}"
    fi
    tar xf "${ARCHIVE}" -C "${WORK_ROOT}/src"
    mv "${WORK_ROOT}/src/heffte-${HEFFTE_VERSION}" "${VANILLA}"
  fi
fi

rm -rf "${SRC_DIR}"
cp -a "${VANILLA}" "${SRC_DIR}"
[[ -f "${PATCH}" ]] || die "missing patch ${PATCH}"
patch -p1 -d "${SRC_DIR}" < "${PATCH}"
grep -q 'sync_device' "${SRC_DIR}/include/heffte_trace.h" ||
  die "gpu-trace-sync patch did not apply"

rm -rf "${BUILD_DIR}"
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DHeffte_ENABLE_FFTW=ON \
  -DHeffte_ENABLE_ROCM=ON \
  -DHeffte_ENABLE_CUDA=OFF \
  -DHeffte_ENABLE_TRACING=ON \
  -DHeffte_ENABLE_GPU_AWARE_MPI="${GPU_AWARE_MPI}" \
  -DCMAKE_HIP_ARCHITECTURES="${ROCM_ARCH}" \
  -DCMAKE_HIP_FLAGS="-I${MPI_INC}"

cmake --build "${BUILD_DIR}" -j"${JOBS}"
cmake --install "${BUILD_DIR}"

HEFFTE_CMAKE_DIR=""
for cand in "${INSTALL_PREFIX}/lib64/cmake/Heffte" "${INSTALL_PREFIX}/lib/cmake/Heffte"; do
  if [[ -f "${cand}/HeffteConfig.cmake" ]]; then
    HEFFTE_CMAKE_DIR="${cand}"
    break
  fi
done
[[ -n "${HEFFTE_CMAKE_DIR}" ]] || die "HeffteConfig.cmake not found under ${INSTALL_PREFIX}"
grep -q 'Heffte_ENABLE_TRACING' "${INSTALL_PREFIX}/include/heffte_config.h"
if grep -q '/\* #undef Heffte_ENABLE_TRACING' "${INSTALL_PREFIX}/include/heffte_config.h"; then
  die "installed heffte_config.h still has tracing undefined"
fi
grep -q 'sync_device' "${INSTALL_PREFIX}/include/heffte_trace.h" ||
  die "installed heffte_trace.h is missing the GPU-sync patch"

mkdir -p "$(dirname "${MODULE_FILE}")"
cat > "${MODULE_FILE}" <<LUA
-- Name: heffte-rocm-trace
-- Version: ${HEFFTE_VERSION}
-- Diagnostic tracing build. Does not replace heffte-rocm.

help([[Diagnostic HeFFTe ${HEFFTE_VERSION} ROCm build with Heffte_ENABLE_TRACING
and GPU-synchronous add_trace. Do not use for admitted production timings.]])

local module_base = "${INSTALL_PREFIX}"

prepend_path("CMAKE_PREFIX_PATH", module_base)
prepend_path("PATH", pathJoin(module_base, "bin"))
prepend_path("LD_LIBRARY_PATH", pathJoin(module_base, "lib64"))
prepend_path("LD_LIBRARY_PATH", pathJoin(module_base, "lib"))
prepend_path("CPATH", pathJoin(module_base, "include"))
prepend_path("INCLUDE", pathJoin(module_base, "include"))
prepend_path("PKG_CONFIG_PATH", pathJoin(module_base, "lib64", "pkgconfig"))
prepend_path("PKG_CONFIG_PATH", pathJoin(module_base, "lib", "pkgconfig"))
setenv("HEFFTE_DIR", pathJoin(module_base, "lib64", "cmake", "Heffte"))
setenv("HEFFTE_ROOT", module_base)
setenv("OPENPFC_HEFFTE_TRACING", "1")

whatis("Name: heffte-rocm-trace")
whatis("Version: ${HEFFTE_VERSION}")
whatis("Description: Diagnostic HeFFTe ${HEFFTE_VERSION} ROCm tracing build")
LUA

{
  echo "heffte_trace_build=1"
  echo "heffte_version=${HEFFTE_VERSION}"
  echo "heffte_enable_tracing=ON"
  echo "heffte_gpu_trace_sync_patch=cmake/heffte-2.4.1-gpu-trace-sync.patch"
  echo "install_prefix=${INSTALL_PREFIX}"
  echo "modulefile=${MODULE_FILE}"
  echo "gpu_aware_mpi=${GPU_AWARE_MPI}"
  echo "rocm_arch=${ROCM_ARCH}"
  echo "build_type=${BUILD_TYPE}"
  echo "does_not_replace=heffte-rocm/2.4.1"
} | tee "${INSTALL_PREFIX}/TRACE_BUILD.txt"

echo
echo "HeFFTe ROCm TRACE build: PASS"
echo "  Install prefix: ${INSTALL_PREFIX}"
echo "  Modulefile:     ${MODULE_FILE}"
echo "Load with: module use \$HOME/privatemodules && module load heffte-rocm-trace"

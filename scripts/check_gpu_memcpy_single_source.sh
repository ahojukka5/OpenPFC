#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# M3 single-source GPU runtime: native cudaMemcpy / hipMemcpy in include/ and
# src/ must live under runtime/gpu/. Vendor trees are thin includes or
# re-exports (plus FFT until M5) and must not grow their own memcpy calls.
#
# Matches the DoD grep:
#   grep -rn "hipMemcpy\|cudaMemcpy" include/ src/ | grep -v runtime/gpu
#
# Ripgrep is used when that binary actually runs. A missing rg, a broken
# symlink, or an rg that exits with an error falls through to grep. If neither
# can scan, the script exits non-zero. It does not report a clean tree.
# runtime/gpu/ stays the only allowed path.
#
# Usage:
#   check_gpu_memcpy_single_source.sh             # run the check
#   check_gpu_memcpy_single_source.sh --self-test # verify the checker

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# True only when rg is executable and answers --version. command -v can
# miss a broken symlink; -x rejects one that it still prints.
rg_ready() {
  local bin
  bin="$(command -v rg 2>/dev/null || true)"
  [[ -n "${bin}" && -x "${bin}" ]] || return 1
  "${bin}" --version >/dev/null 2>&1
}

# Print path:line:text for an ERE. Return 0 after a real scan, 2 when
# neither ripgrep nor grep could scan.
search_ere() {
  local ere="$1"
  shift
  if rg_ready; then
    local out status=0
    out="$(rg -n -H --no-heading -e "${ere}" "$@" 2>/dev/null)" || status=$?
    if [[ "${status}" -le 1 ]]; then
      [[ -n "${out}" ]] && printf '%s\n' "${out}"
      return 0
    fi
  fi
  if ! printf 'x\n' | grep -E -I -e 'x' >/dev/null 2>&1; then
    return 2
  fi
  local path file hits row
  for path in "$@"; do
    if [[ -f "${path}" ]]; then
      hits="$(grep -n -E -I -e "${ere}" -- "${path}" 2>/dev/null || true)"
      if [[ -n "${hits}" ]]; then
        while IFS= read -r row; do
          printf '%s:%s\n' "${path}" "${row}"
        done <<<"${hits}"
      fi
      continue
    fi
    [[ -d "${path}" ]] || continue
    while IFS= read -r -d '' file; do
      hits="$(grep -n -E -I -e "${ere}" -- "${file}" 2>/dev/null || true)"
      [[ -z "${hits}" ]] && continue
      while IFS= read -r row; do
        printf '%s:%s\n' "${file}" "${row}"
      done <<<"${hits}"
    done < <(find "${path}" -type f -print0)
  done
  return 0
}

# Store search_ere output in the named variable. Exit the process if the
# scan did not run: an empty result must not look like a clean tree.
capture_hits() {
  local __name="$1"
  shift
  local __out __status=0
  __out="$("$@")" || __status=$?
  if [[ "${__status}" -ne 0 ]]; then
    echo "ERROR: ${0##*/}: ripgrep is missing or failed, and grep -E -I could not scan. Install ripgrep, or use a grep that supports -E and -I." >&2
    exit 1
  fi
  printf -v "${__name}" '%s' "${__out}"
}

run_checks() {
  local base="$1"
  local hits="" filtered="" line
  capture_hits hits search_ere 'hipMemcpy|cudaMemcpy' \
    "${base}/include" "${base}/src"
  if [[ -n "${hits}" ]]; then
    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      # Allow include/.../runtime/gpu/ and src/.../runtime/gpu/ only.
      [[ "${line}" == *"/runtime/gpu/"* ]] && continue
      filtered+="${line}"$'\n'
    done <<<"${hits}"
  fi
  if [[ -n "${filtered}" ]]; then
    echo "ERROR: cudaMemcpy/hipMemcpy outside runtime/gpu/ (include/ and src/):"
    printf '%s' "${filtered}"
    return 1
  fi
  return 0
}

self_test() {
  local tmp
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064 # path is fixed when the trap is set
  trap "rm -rf $(printf '%q' "${tmp}")" EXIT
  mkdir -p "${tmp}/include/openpfc/runtime/gpu" \
           "${tmp}/include/openpfc/kernel/data" \
           "${tmp}/src/openpfc/runtime/gpu" \
           "${tmp}/src/openpfc/runtime/cuda"

  cat >"${tmp}/include/openpfc/runtime/gpu/ok.hpp" <<'EOF'
#pragma once
inline void copy(void *d, const void *s, std::size_t n) {
  cudaMemcpy(d, s, n, cudaMemcpyHostToDevice);
}
EOF
  cat >"${tmp}/src/openpfc/runtime/gpu/ok.inc" <<'EOF'
hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
EOF
  if ! run_checks "${tmp}" >/dev/null; then
    echo "SELF-TEST FAILED: allowed runtime/gpu memcpy was flagged."; return 1
  fi

  cat >"${tmp}/include/openpfc/kernel/data/bad.hpp" <<'EOF'
#pragma once
inline void leak() { cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyDefault); }
EOF
  if run_checks "${tmp}" >/dev/null; then
    echo "SELF-TEST FAILED: kernel cudaMemcpy was NOT detected."; return 1
  fi
  rm -f "${tmp}/include/openpfc/kernel/data/bad.hpp"

  cat >"${tmp}/src/openpfc/runtime/cuda/bad.cpp" <<'EOF'
#include <cuda_runtime.h>
void leak() { cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyDefault); }
EOF
  if run_checks "${tmp}" >/dev/null; then
    echo "SELF-TEST FAILED: vendor-tree cudaMemcpy was NOT detected."; return 1
  fi
  rm -f "${tmp}/src/openpfc/runtime/cuda/bad.cpp"

  echo "SELF-TEST OK: runtime/gpu memcpy allowed; leaks outside it detected."
  return 0
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit $?
fi

if run_checks "${ROOT}"; then
  echo "OK: cudaMemcpy/hipMemcpy in include/ and src/ stay under runtime/gpu/."
else
  exit 1
fi

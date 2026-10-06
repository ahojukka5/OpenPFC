#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Architecture layering enforcement (audit §11 / M0). Layers, top to bottom:
#
#   frontend  ->  runtime  ->  kernel
#
# A lower layer must never #include a higher one. This script enforces all three
# downward-only rules:
#   1. kernel   must not #include openpfc/frontend/...
#   2. kernel   must not #include openpfc/runtime/...   (kernel defines tags;
#      runtime injects specializations, so the dependency is runtime -> kernel)
#   3. runtime  must not #include openpfc/frontend/...
#
# Only real preprocessor directives count: the pattern anchors to the start of a
# line (optional leading whitespace), so guidance strings such as
#   static_assert(false, "... requires #include <openpfc/runtime/...>")
# are NOT flagged.
#
# Ripgrep is used when that binary actually runs. A missing rg, a broken
# symlink, or an rg that exits with an error falls through to grep. If neither
# can scan, the script exits non-zero. It does not report a clean tree.
#
# Usage:
#   check_kernel_no_frontend_includes.sh             # run the checks
#   check_kernel_no_frontend_includes.sh --self-test # verify the checker detects
#                                                     # a deliberate violation
# See docs/concepts/architecture.md (Include audit) and docs/adr/.

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

# Print path:line:text for an anchored ERE. Return 0 after a real scan,
# 2 when neither ripgrep nor grep could scan.
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
  local inc_kernel="${base}/include/openpfc/kernel"
  local src_kernel="${base}/src/openpfc/kernel"
  local inc_runtime="${base}/include/openpfc/runtime"
  local src_runtime="${base}/src/openpfc/runtime"
  local failed=0
  local m

  capture_hits m search_ere \
    '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]openpfc/frontend/' \
    "${inc_kernel}" "${src_kernel}"
  if [[ -n "${m}" ]]; then
    echo "ERROR: kernel must not #include openpfc/frontend headers:"; echo "${m}"; failed=1
  fi

  capture_hits m search_ere \
    '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]openpfc/runtime/' \
    "${inc_kernel}" "${src_kernel}"
  if [[ -n "${m}" ]]; then
    echo "ERROR: kernel must not #include openpfc/runtime headers (dependency is runtime -> kernel):"; echo "${m}"; failed=1
  fi

  capture_hits m search_ere \
    '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]openpfc/frontend/' \
    "${inc_runtime}" "${src_runtime}"
  if [[ -n "${m}" ]]; then
    echo "ERROR: runtime must not #include openpfc/frontend headers:"; echo "${m}"; failed=1
  fi

  return "${failed}"
}

self_test() {
  local tmp
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064 # path is fixed when the trap is set
  trap "rm -rf $(printf '%q' "${tmp}")" EXIT
  mkdir -p "${tmp}/include/openpfc/kernel/data" \
           "${tmp}/include/openpfc/runtime/cuda" \
           "${tmp}/src/openpfc/kernel" \
           "${tmp}/src/openpfc/runtime"

  # A clean kernel header (including a guidance STRING mentioning runtime) must pass.
  cat >"${tmp}/include/openpfc/kernel/data/clean.hpp" <<'EOF'
#pragma once
#include <vector>
// static_assert(false, "CudaTag requires #include <openpfc/runtime/cuda/x.hpp>");
EOF
  if ! run_checks "${tmp}" >/dev/null; then
    echo "SELF-TEST FAILED: clean tree (with guidance string) was flagged."; return 1
  fi

  # Each planted violation is removed before the next case. Leaving one in
  # place would make a later miss look like a detection.
  cat >"${tmp}/include/openpfc/kernel/data/bad_runtime.hpp" <<'EOF'
#pragma once
#include <openpfc/runtime/cuda/x.hpp>
EOF
  if run_checks "${tmp}" >/dev/null; then
    echo "SELF-TEST FAILED: a kernel->runtime violation was NOT detected."; return 1
  fi
  rm -f "${tmp}/include/openpfc/kernel/data/bad_runtime.hpp"

  # A real kernel-header frontend include must be detected on its own.
  cat >"${tmp}/include/openpfc/kernel/data/bad.hpp" <<'EOF'
#pragma once
#include <openpfc/frontend/ui/app.hpp>
EOF
  if run_checks "${tmp}" >/dev/null; then
    echo "SELF-TEST FAILED: a kernel->frontend violation was NOT detected."; return 1
  fi
  rm -f "${tmp}/include/openpfc/kernel/data/bad.hpp"

  # The same frontend include under src/openpfc/kernel must be detected too.
  # Header-only scanning would miss it.
  cat >"${tmp}/src/openpfc/kernel/bad.cpp" <<'EOF'
#include <openpfc/frontend/ui/app.hpp>
EOF
  if run_checks "${tmp}" >/dev/null; then
    echo "SELF-TEST FAILED: a kernel source ->frontend violation was NOT detected."; return 1
  fi
  rm -f "${tmp}/src/openpfc/kernel/bad.cpp"

  # runtime -> frontend is a separate rule from the kernel rules above.
  cat >"${tmp}/include/openpfc/runtime/bad_frontend.hpp" <<'EOF'
#pragma once
#include <openpfc/frontend/ui/app.hpp>
EOF
  if run_checks "${tmp}" >/dev/null; then
    echo "SELF-TEST FAILED: a runtime->frontend violation was NOT detected."; return 1
  fi
  rm -f "${tmp}/include/openpfc/runtime/bad_frontend.hpp"

  echo "SELF-TEST OK: guidance string passes; kernel->runtime header, kernel->frontend header, kernel source, and runtime->frontend includes are rejected."
  return 0
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit $?
fi

if run_checks "${ROOT}"; then
  echo "OK: layering respected (kernel !-> frontend/runtime; runtime !-> frontend)."
else
  exit 1
fi

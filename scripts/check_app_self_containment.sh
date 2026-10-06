#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Application boundary. Each apps/<name> may include its own headers and the
# installed OpenPFC API. It must not include another application, the removed
# apps/common tree, or <openpfc_apps/...>. Public OpenPFC headers and sources
# must not include apps/.
#
# Ripgrep is used when that binary actually runs. A missing rg, a broken
# symlink, or an rg that exits with an error falls through to grep. If neither
# can scan, the script exits non-zero. It does not report a clean tree.
#
# Usage:
#   check_app_self_containment.sh
#   check_app_self_containment.sh --self-test

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

# A comma-separated extension list restricts the fallback the way rg -g
# '*.{hpp,cpp,...}' does. An empty list scans every file rg would open.
file_selected() {
  local file="$1" exts="$2" exclude="$3"
  local base="${file##*/}" ext
  if [[ -n "${exclude}" && "${base}" == "${exclude}" ]]; then
    return 1
  fi
  [[ -z "${exts}" ]] && return 0
  ext="${base##*.}"
  [[ "${base}" == "${ext}" ]] && return 1
  [[ ",${exts}," == *",${ext},"* ]]
}

# Print path:line:text for an ERE. exts and exclude match the rg globs used
# by the app boundary (empty means no such filter). Return 0 after a real
# scan, 2 when neither ripgrep nor grep could scan.
search_ere() {
  local ere="$1" exts="$2" exclude="$3"
  shift 3
  if rg_ready; then
    local -a args=(-n -H --no-heading -e "${ere}")
    [[ -n "${exts}" ]] && args+=(-g "*.{${exts}}")
    [[ -n "${exclude}" ]] && args+=(-g "!**/${exclude}")
    local out status=0
    out="$(rg "${args[@]}" "$@" 2>/dev/null)" || status=$?
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
      file_selected "${path}" "${exts}" "${exclude}" || continue
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
      file_selected "${file}" "${exts}" "${exclude}" || continue
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
  local failed=0
  local apps="${base}/apps"
  local app name other hits

  if [[ -d "${apps}/common" ]]; then
    echo "ERROR: apps/common must not exist"
    failed=1
  fi

  if [[ -d "${apps}" ]]; then
    for app in "${apps}"/*; do
      [[ -d "${app}" ]] || continue
      name="$(basename "${app}")"
      capture_hits hits search_ere \
        '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]openpfc_apps/' \
        '' 'README.md' "${app}"
      if [[ -n "${hits}" ]]; then
        echo "ERROR: ${name} includes <openpfc_apps/...>:"; echo "${hits}"; failed=1
      fi
      capture_hits hits search_ere \
        '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"][^>"]*apps/' \
        'hpp,cpp,hip,cu,h' '' "${app}"
      if [[ -n "${hits}" ]]; then
        echo "ERROR: ${name} includes an apps/ path:"; echo "${hits}"; failed=1
      fi
      if [[ -f "${app}/CMakeLists.txt" ]]; then
        capture_hits hits search_ere \
          'openpfc_apps_common|apps/common' \
          '' '' "${app}/CMakeLists.txt"
        if [[ -n "${hits}" ]]; then
          echo "ERROR: ${name} CMake still depends on apps/common:"; echo "${hits}"; failed=1
        fi
      fi
      for other in "${apps}"/*; do
        [[ -d "${other}" ]] || continue
        [[ "${other}" == "${app}" ]] && continue
        local oname
        oname="$(basename "${other}")"
        capture_hits hits search_ere \
          "^[[:space:]]*#[[:space:]]*include.*[^A-Za-z0-9_]${oname}/" \
          'hpp,cpp,hip,cu,h' '' "${app}"
        if [[ -n "${hits}" ]]; then
          echo "ERROR: ${name} includes ${oname} headers:"; echo "${hits}"; failed=1
        fi
      done
    done
  fi

  if [[ -d "${base}/include/openpfc" || -d "${base}/src" ]]; then
    local -a roots=()
    [[ -d "${base}/include/openpfc" ]] && roots+=("${base}/include/openpfc")
    [[ -d "${base}/src" ]] && roots+=("${base}/src")
    capture_hits hits search_ere \
      '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"][^>"]*apps/' \
      'hpp,cpp,hip,cu,h' '' "${roots[@]}"
    if [[ -n "${hits}" ]]; then
      echo "ERROR: public OpenPFC includes apps/:"; echo "${hits}"; failed=1
    fi
  fi

  return "${failed}"
}

self_test() {
  local tmp
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064 # path is fixed when the trap is set
  trap "rm -rf $(printf '%q' "${tmp}")" EXIT
  mkdir -p "${tmp}/apps/alpha" "${tmp}/apps/beta" \
           "${tmp}/include/openpfc/kernel" "${tmp}/src"

  cat >"${tmp}/apps/alpha/own.hpp" <<'EOF'
#include <alpha/local.hpp>
#include <openpfc/kernel/data/domain.hpp>
EOF
  cat >"${tmp}/apps/alpha/CMakeLists.txt" <<'EOF'
target_link_libraries(alpha PRIVATE OpenPFC)
EOF
  cat >"${tmp}/include/openpfc/kernel/clean.hpp" <<'EOF'
#include <openpfc/kernel/data/domain.hpp>
EOF
  if ! run_checks "${tmp}"; then
    echo "self-test: clean tree was rejected"
    return 1
  fi

  echo '#include <openpfc_apps/cli.hpp>' >"${tmp}/apps/alpha/bad.hpp"
  if run_checks "${tmp}"; then
    echo "self-test: openpfc_apps include was accepted"
    return 1
  fi
  rm -f "${tmp}/apps/alpha/bad.hpp"

  echo '#include <beta/secret.hpp>' >"${tmp}/apps/alpha/cross.hpp"
  if run_checks "${tmp}"; then
    echo "self-test: cross-app include was accepted"
    return 1
  fi
  rm -f "${tmp}/apps/alpha/cross.hpp"

  echo '#include <apps/shared/secret.hpp>' >"${tmp}/apps/alpha/viapath.hpp"
  if run_checks "${tmp}"; then
    echo "self-test: app include of an apps/ path was accepted"
    return 1
  fi
  rm -f "${tmp}/apps/alpha/viapath.hpp"

  mkdir -p "${tmp}/apps/common"
  if run_checks "${tmp}"; then
    echo "self-test: apps/common directory was accepted"
    return 1
  fi
  rm -rf "${tmp}/apps/common"

  printf 'target_link_libraries(alpha PRIVATE OpenPFC)\nadd_subdirectory(apps/common)\n' \
    >"${tmp}/apps/alpha/CMakeLists.txt"
  if run_checks "${tmp}"; then
    echo "self-test: apps/common CMake reference was accepted"
    return 1
  fi

  echo 'target_link_libraries(alpha PRIVATE openpfc_apps_common)' \
    >"${tmp}/apps/alpha/CMakeLists.txt"
  if run_checks "${tmp}"; then
    echo "self-test: openpfc_apps_common link was accepted"
    return 1
  fi
  printf 'target_link_libraries(alpha PRIVATE OpenPFC)\n' \
    >"${tmp}/apps/alpha/CMakeLists.txt"

  echo '#include <apps/beta/secret.hpp>' >"${tmp}/include/openpfc/kernel/leak.hpp"
  if run_checks "${tmp}"; then
    echo "self-test: public include of apps/ was accepted"
    return 1
  fi
  rm -f "${tmp}/include/openpfc/kernel/leak.hpp"

  echo "check_app_self_containment: self-test passed"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
else
  run_checks "${ROOT}"
fi

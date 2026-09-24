#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Application boundary. Each apps/<name> may include its own headers and the
# installed OpenPFC API. It must not include another application, the removed
# apps/common tree, or <openpfc_apps/...>. Public OpenPFC headers and sources
# must not include apps/.
#
# Usage:
#   check_app_self_containment.sh
#   check_app_self_containment.sh --self-test

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v rg >/dev/null 2>&1; then
  echo "check_app_self_containment: ripgrep (rg) not found; install ripgrep or skip in minimal environments."
  exit 0
fi

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
      hits="$(rg -n --no-heading -g '!**/README.md' \
        -e '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]openpfc_apps/' \
        "${app}" 2>/dev/null || true)"
      if [[ -n "${hits}" ]]; then
        echo "ERROR: ${name} includes <openpfc_apps/...>:"; echo "${hits}"; failed=1
      fi
      hits="$(rg -n --no-heading -g '*.{hpp,cpp,hip,cu,h}' \
        -e '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"][^>"]*apps/' \
        "${app}" 2>/dev/null || true)"
      if [[ -n "${hits}" ]]; then
        echo "ERROR: ${name} includes an apps/ path:"; echo "${hits}"; failed=1
      fi
      if [[ -f "${app}/CMakeLists.txt" ]]; then
        hits="$(rg -n --no-heading -e 'openpfc_apps_common' -e 'apps/common' \
          "${app}/CMakeLists.txt" 2>/dev/null || true)"
        if [[ -n "${hits}" ]]; then
          echo "ERROR: ${name} CMake still depends on apps/common:"; echo "${hits}"; failed=1
        fi
      fi
      for other in "${apps}"/*; do
        [[ -d "${other}" ]] || continue
        [[ "${other}" == "${app}" ]] && continue
        local oname
        oname="$(basename "${other}")"
        hits="$(rg -n --no-heading -g '*.{hpp,cpp,hip,cu,h}' \
          -e "^[[:space:]]*#[[:space:]]*include.*[^A-Za-z0-9_]${oname}/" \
          "${app}" 2>/dev/null || true)"
        if [[ -n "${hits}" ]]; then
          echo "ERROR: ${name} includes ${oname} headers:"; echo "${hits}"; failed=1
        fi
      done
    done
  fi

  if [[ -d "${base}/include/openpfc" || -d "${base}/src" ]]; then
    hits="$(rg -n --no-heading -g '*.{hpp,cpp,hip,cu,h}' \
      -e '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"][^>"]*apps/' \
      "${base}/include/openpfc" "${base}/src" 2>/dev/null || true)"
    if [[ -n "${hits}" ]]; then
      echo "ERROR: public OpenPFC includes apps/:"; echo "${hits}"; failed=1
    fi
  fi

  return "${failed}"
}

self_test() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp}"' RETURN
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

  echo 'target_link_libraries(alpha PRIVATE openpfc_apps_common)' \
    >"${tmp}/apps/alpha/CMakeLists.txt"
  if run_checks "${tmp}"; then
    echo "self-test: openpfc_apps_common link was accepted"
    return 1
  fi

  echo '#include <apps/beta/secret.hpp>' >"${tmp}/include/openpfc/kernel/leak.hpp"
  printf 'target_link_libraries(alpha PRIVATE OpenPFC)\n' \
    >"${tmp}/apps/alpha/CMakeLists.txt"
  if run_checks "${tmp}"; then
    echo "self-test: public include of apps/ was accepted"
    return 1
  fi

  echo "check_app_self_containment: self-test passed"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
else
  run_checks "${ROOT}"
fi

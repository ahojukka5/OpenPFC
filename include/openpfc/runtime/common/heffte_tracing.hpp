// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file heffte_tracing.hpp
 * @brief Optional HeFFTe event tracing around MPI_Init / MPI_Finalize.
 *
 * @details
 * When `OPENPFC_HEFFTE_TRACE` is set to a filename stem, call
 * `heffte::init_tracing` after `MPI_Init` and `heffte::finalize_tracing`
 * before `MPI_Finalize`. This is a no-op unless HeFFTe was compiled with
 * `Heffte_ENABLE_TRACING=ON`. The production `heffte-rocm` module leaves
 * tracing compiled out; use the diagnostic `heffte-rocm-trace` install.
 */

#include <cstdlib>
#include <iostream>

#ifdef OpenPFC_ENABLE_HEFFTE
#include <heffte.h>
#endif

namespace pfc::runtime {

inline void maybe_init_heffte_tracing(int rank) {
#ifdef OpenPFC_ENABLE_HEFFTE
  const char *root = std::getenv("OPENPFC_HEFFTE_TRACE");
  if (root == nullptr || root[0] == '\0') {
    return;
  }
#ifndef Heffte_ENABLE_TRACING
  if (rank == 0) {
    std::cerr << "OPENPFC_HEFFTE_TRACE is set but Heffte_ENABLE_TRACING is "
                 "compiled out; phase logs will be empty. Load heffte-rocm-trace "
                 "and rebuild.\n";
  }
  return;
#endif
  heffte::init_tracing(root);
  if (rank == 0) {
    std::cout << "heFFTe tracing enabled, root=" << root << "\n";
  }
#else
  (void)rank;
#endif
}

inline void maybe_finalize_heffte_tracing() {
#ifdef OpenPFC_ENABLE_HEFFTE
  const char *root = std::getenv("OPENPFC_HEFFTE_TRACE");
  if (root == nullptr || root[0] == '\0') {
    return;
  }
  heffte::finalize_tracing();
#endif
}

} // namespace pfc::runtime

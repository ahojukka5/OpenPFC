// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file incompressible_device_kernels.cu
 * @brief CUDA translation unit for the periodic rotational Navier–Stokes step.
 */

#if !defined(OpenPFC_ENABLE_CUDA)
#error "incompressible_device_kernels.cu requires OpenPFC_ENABLE_CUDA"
#endif

#include <openpfc/kernel/field/incompressible_device_impl.hpp>

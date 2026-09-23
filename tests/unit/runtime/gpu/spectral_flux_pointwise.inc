// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/simulation/spectral_flux.hpp>
#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>

#include "spectral_flux_mobilities.hpp"

OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(pfc::sim::MobilityGradPointwise<FluxScale>)
OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(
    pfc::sim::MobilityGradPointwise<pfc::sim::AsStored>)

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <openpfc/kernel/data/host_device.hpp>

/// Constant factor for the spectral-flux host/device parity test.
struct FluxScale {
  double value{1.0};
  [[nodiscard]] OPENPFC_HD double operator()(double) const { return value; }
};

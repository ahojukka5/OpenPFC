// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cmath>
#include <openpfc/kernel/data/host_device.hpp>
#include <tuple>
namespace coupled_rk {
struct UGrads {
  double value{}, x{}, xx{}, yy{}, zz{}, xy{}, yz{};
};
struct VGrads {
  double value{}, xz{};
};
struct Local {
  UGrads u;
  VGrads v;
};
struct Increments {
  double du{}, dv{};
  auto as_tuple() { return std::tie(du, dv); }
  auto as_tuple() const { return std::tie(du, dv); }
};
struct Model {
  double a{.3}, b{.04}, c{-.2}, d{2. / 15.};
  double first{}, nonlinear{}, time_source{};
  bool manufactured_uniform{false};
  OPENPFC_HD Increments rhs(double t, const Local &g) const noexcept {
    Increments q{g.v.value + a * g.u.xy + first * g.u.x +
                     nonlinear * g.u.value * g.u.value + time_source * t,
                 b * (g.u.xx + g.u.yy + g.u.zz) + c * g.v.xz + d * g.u.yz -
                     nonlinear * g.u.value * g.v.value + time_source * std::sin(t)};
    if (manufactured_uniform) {
      // Exact spatially uniform solution (exp(t),exp(-t)); the nonlinear
      // terms stay active. These imposed sources depend only on stage time.
      q.du +=
          std::exp(t) - std::exp(-t) - nonlinear * std::exp(2 * t) - time_source * t;
      q.dv += -std::exp(-t) + nonlinear - time_source * std::sin(t);
    }
    return q;
  }
};
} // namespace coupled_rk

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace pfc::field::fd {
namespace detail {
struct BoundaryCoefficient {
  int row, column, material;
  double value;
};
// Unit-spacing M(B) coefficients, not D2. One-based source indices converted
// to zero-based. Right closure reflects both solution and material indices.
// Sources: Mattsson2012.jl at 1cd1bdae0b12626085acebbc2b50bc2d80b6c6b1;
// Uppsala sbplib d2_variable_4.m at e64d8c6a740b6d977a2ebcf70f10693c1121a464
// agrees on this left closure. The latter uses a distinct right closure.
inline constexpr std::array<BoundaryCoefficient, 132> boundary_coefficients{{
    {0, 0, 0, 12.0 / 17.0},
    {0, 1, 0, -59.0 / 68.0},
    {0, 2, 0, 2.0 / 17.0},
    {0, 3, 0, 3.0 / 68.0},
    {1, 0, 0, -59.0 / 68.0},
    {1, 1, 0, 3481.0 / 3264.0},
    {1, 2, 0, -59.0 / 408.0},
    {1, 3, 0, -59.0 / 1088.0},
    {2, 0, 0, 2.0 / 17.0},
    {2, 1, 0, -59.0 / 408.0},
    {2, 2, 0, 1.0 / 51.0},
    {2, 3, 0, 1.0 / 136.0},
    {3, 0, 0, 3.0 / 68.0},
    {3, 1, 0, -59.0 / 1088.0},
    {3, 2, 0, 1.0 / 136.0},
    {3, 3, 0, 3.0 / 1088.0},
    {0, 0, 1, 59.0 / 192.0},
    {0, 2, 1, -59.0 / 192.0},
    {2, 0, 1, -59.0 / 192.0},
    {2, 2, 1, 59.0 / 192.0},
    {0, 0, 2, 27010400129.0 / 345067064608.0},
    {0, 1, 2, -6025413881.0 / 21126554976.0},
    {0, 2, 2, 2083938599.0 / 8024815456.0},
    {0, 3, 2, -1244724001.0 / 21126554976.0},
    {0, 4, 2, 49579087.0 / 10149031312.0},
    {0, 5, 2, 1.0 / 784.0},
    {1, 0, 2, -6025413881.0 / 21126554976.0},
    {1, 1, 2, 9258282831623875.0 / 7669235228057664.0},
    {1, 2, 2, -29294615794607.0 / 29725717938208.0},
    {1, 3, 2, 260297319232891.0 / 2556411742685888.0},
    {1, 4, 2, -1328188692663.0 / 37594290333616.0},
    {1, 5, 2, -8673.0 / 2904112.0},
    {2, 0, 2, 2083938599.0 / 8024815456.0},
    {2, 1, 2, -29294615794607.0 / 29725717938208.0},
    {2, 2, 2, 378288882302546512209.0 / 270764341349677687456.0},
    {2, 3, 2, -4836340090442187227.0 / 5525802884687299744.0},
    {2, 4, 2, 1613976761032884305.0 / 7963657098519931984.0},
    {2, 5, 2, 33235054191.0 / 26452850508784.0},
    {3, 0, 2, -1244724001.0 / 21126554976.0},
    {3, 1, 2, 260297319232891.0 / 2556411742685888.0},
    {3, 2, 2, -4836340090442187227.0 / 5525802884687299744.0},
    {3, 3, 2, 507284006600757858213.0 / 475219048083107777984.0},
    {3, 4, 2, -4959271814984644613.0 / 20965546238960637264.0},
    {3, 5, 2, 752806667.0 / 539854092016.0},
    {4, 0, 2, 49579087.0 / 10149031312.0},
    {4, 1, 2, -1328188692663.0 / 37594290333616.0},
    {4, 2, 2, 1613976761032884305.0 / 7963657098519931984.0},
    {4, 3, 2, -4959271814984644613.0 / 20965546238960637264.0},
    {4, 4, 2, 8386761355510099813.0 / 128413970713633903242.0},
    {4, 5, 2, -13091810925.0 / 13226425254392.0},
    {5, 0, 2, 1.0 / 784.0},
    {5, 1, 2, -8673.0 / 2904112.0},
    {5, 2, 2, 33235054191.0 / 26452850508784.0},
    {5, 3, 2, 752806667.0 / 539854092016.0},
    {5, 4, 2, -13091810925.0 / 13226425254392.0},
    {5, 5, 2, 660204843.0 / 13226425254392.0},
    {0, 0, 3, 69462376031.0 / 2070402387648.0},
    {0, 1, 3, -537416663.0 / 7042184992.0},
    {0, 2, 3, 213318005.0 / 16049630912.0},
    {0, 3, 3, 752806667.0 / 21126554976.0},
    {0, 4, 3, -49579087.0 / 10149031312.0},
    {0, 5, 3, -1.0 / 784.0},
    {1, 0, 3, -537416663.0 / 7042184992.0},
    {1, 1, 3, 236024329996203.0 / 1278205871342944.0},
    {1, 2, 3, -2944673881023.0 / 29725717938208.0},
    {1, 3, 3, -60834186813841.0 / 1278205871342944.0},
    {1, 4, 3, 1328188692663.0 / 37594290333616.0},
    {1, 5, 3, 8673.0 / 2904112.0},
    {2, 0, 3, 213318005.0 / 16049630912.0},
    {2, 1, 3, -2944673881023.0 / 29725717938208.0},
    {2, 2, 3, 13777050223300597.0 / 26218083221499456.0},
    {2, 3, 3, -17220493277981.0 / 89177153814624.0},
    {2, 4, 3, -10532412077335.0 / 42840005263888.0},
    {2, 5, 3, -960119.0 / 1280713392.0},
    {3, 0, 3, 752806667.0 / 21126554976.0},
    {3, 1, 3, -60834186813841.0 / 1278205871342944.0},
    {3, 2, 3, -17220493277981.0 / 89177153814624.0},
    {3, 3, 3, 1950062198436997.0 / 3834617614028832.0},
    {3, 4, 3, -15998714909649.0 / 37594290333616.0},
    {3, 5, 3, 1063649.0 / 8712336.0},
    {4, 0, 3, -49579087.0 / 10149031312.0},
    {4, 1, 3, 1328188692663.0 / 37594290333616.0},
    {4, 2, 3, -10532412077335.0 / 42840005263888.0},
    {4, 3, 3, -15998714909649.0 / 37594290333616.0},
    {4, 4, 3, 2224717261773437.0 / 2763180339520776.0},
    {4, 5, 3, -35039615.0 / 213452232.0},
    {5, 0, 3, -1.0 / 784.0},
    {5, 1, 3, 8673.0 / 2904112.0},
    {5, 2, 3, -960119.0 / 1280713392.0},
    {5, 3, 3, 1063649.0 / 8712336.0},
    {5, 4, 3, -35039615.0 / 213452232.0},
    {5, 5, 3, 3290636.0 / 80044587.0},
    {2, 2, 4, 564461.0 / 13384296.0},
    {2, 3, 4, -125059.0 / 743572.0},
    {2, 4, 4, 564461.0 / 4461432.0},
    {2, 5, 4, -3391.0 / 6692148.0},
    {3, 2, 4, -125059.0 / 743572.0},
    {3, 3, 4, 1869103.0 / 2230716.0},
    {3, 4, 4, -375177.0 / 743572.0},
    {3, 5, 4, -368395.0 / 2230716.0},
    {4, 2, 4, 564461.0 / 4461432.0},
    {4, 3, 4, -375177.0 / 743572.0},
    {4, 4, 4, 280535.0 / 371786.0},
    {4, 5, 4, -1118749.0 / 2230716.0},
    {4, 6, 4, 1.0 / 8.0},
    {5, 2, 4, -3391.0 / 6692148.0},
    {5, 3, 4, -368395.0 / 2230716.0},
    {5, 4, 4, -1118749.0 / 2230716.0},
    {5, 5, 4, 5580181.0 / 6692148.0},
    {5, 6, 4, -1.0 / 6.0},
    {3, 3, 5, 1.0 / 24.0},
    {3, 4, 5, -1.0 / 6.0},
    {3, 5, 5, 1.0 / 8.0},
    {4, 3, 5, -1.0 / 6.0},
    {4, 4, 5, 5.0 / 6.0},
    {4, 5, 5, -1.0 / 2.0},
    {4, 6, 5, -1.0 / 6.0},
    {5, 3, 5, 1.0 / 8.0},
    {5, 4, 5, -1.0 / 2.0},
    {5, 5, 5, 3.0 / 4.0},
    {5, 6, 5, -1.0 / 2.0},
    {5, 7, 5, 1.0 / 8.0},
    {4, 4, 6, 1.0 / 24.0},
    {4, 5, 6, -1.0 / 6.0},
    {4, 6, 6, 1.0 / 8.0},
    {5, 4, 6, -1.0 / 6.0},
    {5, 5, 6, 5.0 / 6.0},
    {5, 6, 6, -1.0 / 2.0},
    {5, 7, 6, -1.0 / 6.0},
    {5, 5, 7, 1.0 / 24.0},
    {5, 6, 7, -1.0 / 6.0},
    {5, 7, 7, 1.0 / 8.0},
}};
} // namespace detail

/// Prescribed outward derivative and physical outward flux are distinct data.
enum class BoundaryQuantity { NormalDerivative, ConstitutiveFlux };
struct FaceCondition {
  BoundaryQuantity quantity = BoundaryQuantity::ConstitutiveFlux;
  std::function<double(double)> value = [](double) { return 0.0; };
  double flux(double time, double diffusivity) const {
    if (!std::isfinite(time) || !std::isfinite(diffusivity) || diffusivity < 0)
      throw std::invalid_argument("invalid face time or diffusivity");
    const double q = value(time);
    if (!std::isfinite(q)) throw std::domain_error("nonfinite boundary value");
    if (quantity == BoundaryQuantity::ConstitutiveFlux && diffusivity == 0 && q != 0)
      throw std::domain_error("nonzero constitutive flux at zero diffusivity");
    const double j =
        quantity == BoundaryQuantity::NormalDerivative ? -diffusivity * q : q;
    if (!std::isfinite(j)) throw std::domain_error("nonfinite constitutive flux");
    return j;
  }
};

/// One tensor-product axis. Readers use global axis indices on an already
/// prepared field/halo. No communication or hidden reduction occurs here.
/// Global N>=17 prevents boundary-closure overlap; local extent can be one.
class DiffusionAxis {
  int count_;
  double spacing_;
  bool periodic_;
  int wrapped(int i) const { return (i % count_ + count_) % count_; }

public:
  DiffusionAxis(int count, double spacing, bool periodic = false)
      : count_(count), spacing_(spacing), periodic_(periodic) {
    if (count < 17 || !std::isfinite(spacing) || spacing <= 0)
      throw std::invalid_argument("diffusion axis needs N>=17 and positive spacing");
  }
  int size() const { return count_; }
  double weight(int i) const {
    if (i < 0 || i >= count_) throw std::out_of_range("axis index");
    constexpr std::array<double, 4> w{17. / 48, 59. / 48, 43. / 48, 49. / 48};
    int d = std::min(i, count_ - 1 - i);
    return spacing_ * (!periodic_ && d < 4 ? w[d] : 1.);
  }
  /// Visit M(B) tensor entries for row i: (solution node, material node, value).
  template <class Visit> void coefficients(int i, Visit visit) const {
    (void)weight(i);
    const bool right = !periodic_ && i >= count_ - 6;
    if (!periodic_ && (i < 6 || right)) {
      const int row = right ? count_ - 1 - i : i;
      for (const auto &c : detail::boundary_coefficients)
        if (c.row == row)
          visit(right ? count_ - 1 - c.column : c.column,
                right ? count_ - 1 - c.material : c.material, c.value);
      return;
    }
    auto add = [&](int j, int k, double v) {
      visit(wrapped(i + j), wrapped(i + k), v);
    };
    add(-2, -2, 1. / 8);
    add(-2, -1, -1. / 6);
    add(-2, 0, 1. / 8);
    add(-1, -2, -1. / 6);
    add(-1, -1, -1. / 2);
    add(-1, 0, -1. / 2);
    add(-1, 1, -1. / 6);
    add(0, -2, 1. / 24);
    add(0, -1, 5. / 6);
    add(0, 0, 3. / 4);
    add(0, 1, 5. / 6);
    add(0, 2, 1. / 24);
    add(1, -1, -1. / 6);
    add(1, 0, -1. / 2);
    add(1, 1, -1. / 2);
    add(1, 2, -1. / 6);
    add(2, 0, 1. / 8);
    add(2, 1, -1. / 6);
    add(2, 2, 1. / 8);
  }
  /// Return diffusion RHS. Only global endpoint owners supply face data.
  /// Positive outward J removes inventory. Tangential H weights form area.
  template <class State, class Material>
  double apply(int i, State state, Material material, double time,
               const FaceCondition *left = nullptr,
               const FaceCondition *right = nullptr) const {
    if (periodic_ && (left || right))
      throw std::invalid_argument("BC on periodic face");
    double sum = 0;
    coefficients(i, [&](int j, int k, double c) {
      const double u = state(j), d = material(k);
      if (!std::isfinite(u) || !std::isfinite(d) || d < 0)
        throw std::domain_error("invalid diffusion state or coefficient");
      sum += c * d * u;
    });
    double result = -sum / (spacing_ * weight(i));
    if (!periodic_ && i == 0 && left)
      result -= left->flux(time, material(0)) / weight(i);
    if (!periodic_ && i == count_ - 1 && right)
      result -= right->flux(time, material(i)) / weight(i);
    if (!std::isfinite(result))
      throw std::domain_error("nonfinite diffusion result");
    return result;
  }
};

/// Transactional integral of outward flux. Stage weight includes dt and RK
/// weight; area is the same tangential norm used in the inventory. Caller
/// reduces rank-local accepted totals explicitly. Rejected trials add zero.
class FluxLedger {
  double accepted_ = 0, trial_ = 0;
  bool active_ = false, failed_ = false;

public:
  void begin() {
    if (active_) throw std::logic_error("flux trial already active");
    active_ = true;
    failed_ = false;
    trial_ = 0;
  }
  bool active() const { return active_; }
  void invalidate() {
    if (!active_) throw std::logic_error("no flux trial");
    failed_ = true;
  }
  void stage(double flux, double area, double time_weight) {
    if (!active_) throw std::logic_error("no flux trial");
    const double next = trial_ + flux * area * time_weight;
    if (!std::isfinite(flux) || !std::isfinite(area) || area < 0 ||
        !std::isfinite(time_weight) || !std::isfinite(next)) {
      failed_ = true;
      throw std::domain_error("invalid flux quadrature");
    }
    trial_ = next;
  }
  void accept() {
    if (!active_) throw std::logic_error("no flux trial");
    if (failed_) throw std::logic_error("failed flux trial must be rejected");
    const double next = accepted_ + trial_;
    if (!std::isfinite(next)) throw std::overflow_error("flux integral overflow");
    accepted_ = next;
    active_ = false;
    trial_ = 0;
  }
  void reject() {
    if (!active_) throw std::logic_error("no flux trial");
    active_ = false;
    trial_ = 0;
  }
  double accepted() const { return accepted_; }
};
} // namespace pfc::field::fd

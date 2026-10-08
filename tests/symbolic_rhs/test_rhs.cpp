// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "generated_rhs.hpp"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <openpfc/kernel/simulation/for_each_interior.hpp>
#include <tuple>

namespace {
struct U {
  double xx{}, yy{}, zz{}, xy{}, yz{};
};
struct V {
  double value{}, xz{};
};
struct Local {
  U u;
  V v;
};
struct Parameters {
  double a{}, b{}, c{}, d{};
};
struct Increments {
  double du{}, dv{};
  auto as_tuple() const { return std::tie(du, dv); }
};
struct Model {
  Parameters p{.3, .04, -.2, 2. / 15.};
  std::array<double, 3> position{2, 3, 5};
  Increments rhs(double t, const Local &g) const {
    return coupled::generated::rhs<Increments>(t, position, g, p, .125);
  }
};
struct Eval {
  int prepared = 0;
  void prepare() { ++prepared; }
  int imin() const { return 0; }
  int imax() const { return 8; }
  int jmin() const { return 0; }
  int jmax() const { return 1; }
  int kmin() const { return 0; }
  int kmax() const { return 1; }
  std::size_t idx(int i, int, int) const { return i; }
  Local operator()(int i, int, int) const {
    return {{double(i), 2, 3, 4, 5}, {6, 7}};
  }
};
} // namespace
int main(int argc, char **) {
  if (argc > 1) {
    double t, rate;
    std::array<double, 3> x{};
    Local g{};
    Parameters p{};
    std::cout << std::setprecision(17);
    while (std::cin >> t >> x[0] >> x[1] >> x[2] >> rate >> p.a >> p.b >> p.c >>
           p.d >> g.v.value >> g.u.xx >> g.u.yy >> g.u.zz >> g.u.xy >> g.u.yz >>
           g.v.xz) {
      const auto inc = coupled::generated::rhs<Increments>(t, x, g, p, rate);
      std::cout << inc.du << ' ' << inc.dv << '\n';
    }
    return std::cin.eof() ? 0 : 2;
  }
  Eval eval;
  Model model;
  const auto bare = coupled::generated::rhs<Increments>(2., model.position,
                                                        eval(0, 0, 0), model.p);
  if (std::abs(bare.du - 7.2) > 1e-13 ||
      std::abs(bare.dv - (.04 * 5 - .2 * 7 + 2. / 15. * 5)) > 1e-13)
    return 1;
  std::array<double, 10> du{}, dv{};
  du.fill(-999);
  dv.fill(-999);
  pfc::sim::for_each_interior(model, eval, std::make_tuple(du.data(), dv.data()),
                              2.);
  if (eval.prepared != 1 || du[8] != -999 || dv[9] != -999) return 1;
  for (int i = 0; i < 8; ++i) {
    const double a = 6 + .3 * 4 + .125 * 2 * 10;
    const double b = .04 * (i + 2 + 3) - .2 * 7 + (2. / 15.) * 5 + .125 * 2 * 10;
    if (std::abs(du[i] - a) > 1e-13 || std::abs(dv[i] - b) > 1e-13) return 1;
  }
  std::cout << "Generated RHS canonical tuple scatter: 8 cells passed\n";
}

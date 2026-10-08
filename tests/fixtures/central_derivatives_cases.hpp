// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <array>
#include <cmath>
#include <openpfc/kernel/field/central_derivatives.hpp>
namespace derivative_cases {
using Result = pfc::field::fd::Derivatives3D<double>;
inline std::array<double, 9> entries(Result a) {
  return {a.x, a.y, a.z, a.xx, a.yy, a.zz, a.xy, a.xz, a.yz};
}
struct Polynomial {
  double x = .7, y = -.4, z = .9, hx = .125, hy = .25, hz = .5;
  OPENPFC_HD double operator()(int a, int b, int c) const {
    double X = x + a * hx, Y = y + b * hy, Z = z + c * hz;
    return 2 * X * X + 3 * Y * Y + 4 * Z * Z + 5 * X * Y + 6 * X * Z + 7 * Y * Z +
           X - 2 * Y + 3 * Z + 9;
  }
  Result exact() const {
    return {4 * x + 5 * y + 6 * z + 1,
            6 * y + 5 * x + 7 * z - 2,
            8 * z + 6 * x + 7 * y + 3,
            4,
            6,
            8,
            5,
            6,
            7};
  }
};
struct Periodic {
  double x = .37, y = .61, z = .93, h = .2;
  OPENPFC_HD double operator()(int a, int b, int c) const {
    double X = x + a * h, Y = y + b * h, Z = z + c * h;
    return sin(X) + cos(Y) + sin(Z) + sin(X + Y + Z);
  }
  Result exact() const {
    double s = sin(x + y + z), c = cos(x + y + z);
    return {cos(x) + c,  -sin(y) + c, cos(z) + c, -sin(x) - s, -cos(y) - s,
            -sin(z) - s, -s,          -s,         -s};
  }
};
// Cell centres x=(i+.5)h on [0,pi]. Even reflection at both boundary faces.
struct Reflected {
  int n = 16, i = 0;
  double y = .4, z = .8;
  OPENPFC_HD double operator()(int a, int b, int c) const {
    int j = (i + a) % (2 * n);
    if (j < 0) j += 2 * n;
    if (j >= n) j = 2 * n - 1 - j;
    const double h = 3.14159265358979323846 / n;
    return cos((j + .5) * h) * cos(y + b * h) * cos(z + c * h);
  }
  Result exact() const {
    const double x = (i + .5) * 3.14159265358979323846 / n;
    double u = cos(x) * cos(y) * cos(z);
    return {-sin(x) * cos(y) * cos(z),
            -cos(x) * sin(y) * cos(z),
            -cos(x) * cos(y) * sin(z),
            -u,
            -u,
            -u,
            sin(x) * sin(y) * cos(z),
            sin(x) * cos(y) * sin(z),
            cos(x) * sin(y) * sin(z)};
  }
};
} // namespace derivative_cases

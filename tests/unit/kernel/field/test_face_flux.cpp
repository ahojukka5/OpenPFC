// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Neutral checks for `pfc::field::fd::divergence_separated`: face averages,
 * a constant-coefficient stencil, discrete conservation, a split box, and
 * a rank that owns one cell on an axis the global grid still uses.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/field/face_flux.hpp>

using pfc::field::fd::average_face;
using pfc::field::fd::divergence_separated;
using pfc::field::fd::FaceAverage;

namespace {

std::array<bool, 3> axes_of_whole_grid(int nx, int ny, int nz) {
  return {nx > 1, ny > 1, nz > 1};
}

std::size_t idx(int ix, int iy, int iz, int nx, int ny) {
  return static_cast<std::size_t>(ix) +
         static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
         static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx) *
             static_cast<std::size_t>(ny);
}

struct Halos {
  std::array<std::vector<double>, 6> storage{};
  std::array<const double *, 6> ptrs{};

  void wrap(const std::vector<double> &core, int nx, int ny, int nz) {
    storage[0].assign(static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz),
                      0.0);
    storage[1] = storage[0];
    storage[2].assign(static_cast<std::size_t>(nx) * static_cast<std::size_t>(nz),
                      0.0);
    storage[3] = storage[2];
    storage[4].assign(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny),
                      0.0);
    storage[5] = storage[4];
    for (int iz = 0; iz < nz; ++iz) {
      for (int iy = 0; iy < ny; ++iy) {
        const std::size_t face =
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny) +
            static_cast<std::size_t>(iy);
        storage[0][face] = core[idx(0, iy, iz, nx, ny)];
        storage[1][face] = core[idx(nx - 1, iy, iz, nx, ny)];
      }
    }
    for (int iz = 0; iz < nz; ++iz) {
      for (int ix = 0; ix < nx; ++ix) {
        const std::size_t face =
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);
        storage[2][face] = core[idx(ix, 0, iz, nx, ny)];
        storage[3][face] = core[idx(ix, ny - 1, iz, nx, ny)];
      }
    }
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const std::size_t face =
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);
        storage[4][face] = core[idx(ix, iy, 0, nx, ny)];
        storage[5][face] = core[idx(ix, iy, nz - 1, nx, ny)];
      }
    }
    for (int i = 0; i < 6; ++i)
      ptrs[static_cast<std::size_t>(i)] =
          storage[static_cast<std::size_t>(i)].data();
  }
};

double sum_of(const std::vector<double> &v) {
  double s = 0.0;
  for (double x : v) s += x;
  return s;
}

} // namespace

TEST_CASE("arithmetic and harmonic face averages", "[face_flux]") {
  REQUIRE(average_face(FaceAverage::Arithmetic, 1.0, 3.0) == 2.0);
  REQUIRE(average_face(FaceAverage::Arithmetic, 0.0, 4.0) == 2.0);
  REQUIRE(average_face(FaceAverage::Harmonic, 2.0, 6.0) == 3.0);
  REQUIRE(average_face(FaceAverage::Harmonic, 0.0, 4.0) == 0.0);
  REQUIRE(average_face(FaceAverage::Harmonic, 5.0, 0.0) == 0.0);
  REQUIRE(average_face(FaceAverage::Harmonic, 0.0, 0.0) == 0.0);
  REQUIRE(average_face(FaceAverage::Harmonic, -1.0, 1.0) == 0.0);
}

TEST_CASE("constant coefficient matches the three-point second difference",
          "[face_flux]") {
  constexpr int nx = 16;
  constexpr int ny = 8;
  constexpr int nz = 1;
  constexpr double dx = 0.5;
  constexpr double dy = 0.25;
  constexpr double a = 2.5;
  const double lx = static_cast<double>(nx) * dx;
  const double k = 2.0 * pfc::pi / lx;

  std::vector<double> coeff(static_cast<std::size_t>(nx * ny), a);
  std::vector<double> potential(coeff.size());
  std::vector<double> rhs(coeff.size());
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const double x = (static_cast<double>(ix) + 0.5) * dx;
      potential[idx(ix, iy, 0, nx, ny)] = std::cos(k * x);
    }
  }
  Halos ch, ph;
  ch.wrap(coeff, nx, ny, nz);
  ph.wrap(potential, nx, ny, nz);
  divergence_separated(coeff, potential, ch.ptrs, ph.ptrs, rhs, nx, ny, nz,
                       axes_of_whole_grid(nx, ny, nz), dx, dy, 1.0,
                       FaceAverage::Arithmetic);

  const double stencil = a * (2.0 * std::cos(k * dx) - 2.0) / (dx * dx);
  double err = 0.0;
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const double x = (static_cast<double>(ix) + 0.5) * dx;
      const double expected = stencil * std::cos(k * x);
      err = std::max(err, std::abs(rhs[idx(ix, iy, 0, nx, ny)] - expected));
    }
  }
  REQUIRE(err < 1.0e-12);
  REQUIRE(std::abs(sum_of(rhs)) < 1.0e-10);
}

TEST_CASE("variable coefficient matches a scalar face loop", "[face_flux]") {
  constexpr int nx = 6;
  constexpr int ny = 5;
  constexpr int nz = 1;
  constexpr double dx = 0.3;
  constexpr double dy = 0.4;
  std::vector<double> coeff(static_cast<std::size_t>(nx * ny));
  std::vector<double> potential(coeff.size());
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const auto c = idx(ix, iy, 0, nx, ny);
      coeff[c] = 0.3 + 0.2 * static_cast<double>((ix + 2 * iy) % 5);
      potential[c] = std::sin(0.5 * ix) - 0.25 * iy;
    }
  }
  Halos ch, ph;
  ch.wrap(coeff, nx, ny, nz);
  ph.wrap(potential, nx, ny, nz);
  std::vector<double> rhs(coeff.size());
  divergence_separated(coeff, potential, ch.ptrs, ph.ptrs, rhs, nx, ny, nz,
                       axes_of_whole_grid(nx, ny, nz), dx, dy, 1.0,
                       FaceAverage::Harmonic);

  auto at = [&](const std::vector<double> &core, const Halos &halo, int ix, int iy,
                int axis, int side) {
    if (axis == 0) {
      if (side > 0)
        return (ix + 1 < nx) ? core[idx(ix + 1, iy, 0, nx, ny)]
                             : halo.storage[0][static_cast<std::size_t>(iy)];
      return (ix - 1 >= 0) ? core[idx(ix - 1, iy, 0, nx, ny)]
                           : halo.storage[1][static_cast<std::size_t>(iy)];
    }
    if (side > 0)
      return (iy + 1 < ny) ? core[idx(ix, iy + 1, 0, nx, ny)]
                           : halo.storage[2][static_cast<std::size_t>(ix)];
    return (iy - 1 >= 0) ? core[idx(ix, iy - 1, 0, nx, ny)]
                         : halo.storage[3][static_cast<std::size_t>(ix)];
  };
  double err = 0.0;
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const double ac = coeff[idx(ix, iy, 0, nx, ny)];
      const double pc = potential[idx(ix, iy, 0, nx, ny)];
      const double fxp = pfc::field::fd::face_flux(
          average_face(FaceAverage::Harmonic, ac, at(coeff, ch, ix, iy, 0, +1)),
          at(potential, ph, ix, iy, 0, +1), pc, dx);
      const double fxm = pfc::field::fd::face_flux(
          average_face(FaceAverage::Harmonic, at(coeff, ch, ix, iy, 0, -1), ac), pc,
          at(potential, ph, ix, iy, 0, -1), dx);
      const double fyp = pfc::field::fd::face_flux(
          average_face(FaceAverage::Harmonic, ac, at(coeff, ch, ix, iy, 1, +1)),
          at(potential, ph, ix, iy, 1, +1), pc, dy);
      const double fym = pfc::field::fd::face_flux(
          average_face(FaceAverage::Harmonic, at(coeff, ch, ix, iy, 1, -1), ac), pc,
          at(potential, ph, ix, iy, 1, -1), dy);
      const double expect = (fxp - fxm) / dx + (fyp - fym) / dy;
      err = std::max(err, std::abs(rhs[idx(ix, iy, 0, nx, ny)] - expect));
    }
  }
  REQUIRE(err < 1.0e-12);
}

TEST_CASE("periodic divergence conserves a variable coefficient", "[face_flux]") {
  constexpr int nx = 7;
  constexpr int ny = 5;
  constexpr int nz = 4;
  std::vector<double> coeff(static_cast<std::size_t>(nx * ny * nz));
  std::vector<double> potential(coeff.size());
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const auto c = idx(ix, iy, iz, nx, ny);
        coeff[c] = 0.2 + 0.05 * static_cast<double>((3 * ix + 5 * iy + 7 * iz) % 9);
        potential[c] = std::sin(0.3 * ix) + 0.1 * iy - 0.2 * iz;
      }
    }
  }
  Halos ch, ph;
  ch.wrap(coeff, nx, ny, nz);
  ph.wrap(potential, nx, ny, nz);
  for (const auto kind : {FaceAverage::Arithmetic, FaceAverage::Harmonic}) {
    std::vector<double> rhs(coeff.size());
    divergence_separated(coeff, potential, ch.ptrs, ph.ptrs, rhs, nx, ny, nz,
                         axes_of_whole_grid(nx, ny, nz), 0.4, 0.5, 0.8, kind);
    REQUIRE(std::abs(sum_of(rhs)) < 1.0e-10);
  }
}

TEST_CASE("a split periodic box matches the whole-grid divergence", "[face_flux]") {
  constexpr int nx = 8;
  constexpr int ny = 4;
  constexpr int nz = 1;
  constexpr int left_n = 4;
  std::vector<double> coeff(static_cast<std::size_t>(nx * ny));
  std::vector<double> potential(coeff.size());
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const auto c = idx(ix, iy, 0, nx, ny);
      coeff[c] = 1.0 + 0.1 * ix + 0.01 * iy;
      potential[c] = std::cos(0.7 * ix) * std::sin(0.4 * iy);
    }
  }
  Halos ch, ph;
  ch.wrap(coeff, nx, ny, nz);
  ph.wrap(potential, nx, ny, nz);

  auto piece = [&](int x0, int nloc) {
    std::vector<double> c(static_cast<std::size_t>(nloc * ny));
    std::vector<double> p(c.size());
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nloc; ++ix) {
        c[idx(ix, iy, 0, nloc, ny)] = coeff[idx(x0 + ix, iy, 0, nx, ny)];
        p[idx(ix, iy, 0, nloc, ny)] = potential[idx(x0 + ix, iy, 0, nx, ny)];
      }
    }
    Halos hc, hp;
    hc.storage[0].resize(static_cast<std::size_t>(ny));
    hc.storage[1].resize(static_cast<std::size_t>(ny));
    hc.storage[2].resize(static_cast<std::size_t>(nloc));
    hc.storage[3].resize(static_cast<std::size_t>(nloc));
    hc.storage[4].resize(static_cast<std::size_t>(nloc * ny));
    hc.storage[5] = hc.storage[4];
    hp.storage = hc.storage;
    for (int iy = 0; iy < ny; ++iy) {
      const int xp = (x0 + nloc) % nx;
      const int xm = (x0 + nx - 1) % nx;
      hc.storage[0][static_cast<std::size_t>(iy)] = coeff[idx(xp, iy, 0, nx, ny)];
      hc.storage[1][static_cast<std::size_t>(iy)] = coeff[idx(xm, iy, 0, nx, ny)];
      hp.storage[0][static_cast<std::size_t>(iy)] =
          potential[idx(xp, iy, 0, nx, ny)];
      hp.storage[1][static_cast<std::size_t>(iy)] =
          potential[idx(xm, iy, 0, nx, ny)];
    }
    for (int ix = 0; ix < nloc; ++ix) {
      hc.storage[2][static_cast<std::size_t>(ix)] = c[idx(ix, 0, 0, nloc, ny)];
      hc.storage[3][static_cast<std::size_t>(ix)] = c[idx(ix, ny - 1, 0, nloc, ny)];
      hp.storage[2][static_cast<std::size_t>(ix)] = p[idx(ix, 0, 0, nloc, ny)];
      hp.storage[3][static_cast<std::size_t>(ix)] = p[idx(ix, ny - 1, 0, nloc, ny)];
    }
    for (int i = 0; i < 6; ++i) {
      hc.ptrs[static_cast<std::size_t>(i)] =
          hc.storage[static_cast<std::size_t>(i)].data();
      hp.ptrs[static_cast<std::size_t>(i)] =
          hp.storage[static_cast<std::size_t>(i)].data();
    }
    std::vector<double> rhs(c.size());
    divergence_separated(c, p, hc.ptrs, hp.ptrs, rhs, nloc, ny, nz,
                         axes_of_whole_grid(nx, ny, nz), 1.0, 1.0, 1.0,
                         FaceAverage::Harmonic);
    return rhs;
  };

  // Harmonic on the pieces must match harmonic on the whole grid.
  std::vector<double> full_h(coeff.size());
  divergence_separated(coeff, potential, ch.ptrs, ph.ptrs, full_h, nx, ny, nz,
                       axes_of_whole_grid(nx, ny, nz), 1.0, 1.0, 1.0,
                       FaceAverage::Harmonic);
  const auto left = piece(0, left_n);
  const auto right = piece(left_n, nx - left_n);
  double err = 0.0;
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < left_n; ++ix) {
      err = std::max(err, std::abs(left[idx(ix, iy, 0, left_n, ny)] -
                                   full_h[idx(ix, iy, 0, nx, ny)]));
    }
    for (int ix = 0; ix < nx - left_n; ++ix) {
      err = std::max(err, std::abs(right[idx(ix, iy, 0, nx - left_n, ny)] -
                                   full_h[idx(left_n + ix, iy, 0, nx, ny)]));
    }
  }
  REQUIRE(err < 1.0e-12);
  REQUIRE(std::abs(sum_of(full_h)) < 1.0e-10);
}

TEST_CASE("a locally owned extent of 1 still fluxes an active global axis",
          "[face_flux]") {
  constexpr int nx = 4;
  constexpr int ny = 3;
  constexpr int nz = 1;
  const std::array<bool, 3> active{true, true, false};
  std::vector<double> coeff(static_cast<std::size_t>(nx * ny));
  std::vector<double> potential(coeff.size());
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const auto c = idx(ix, iy, 0, nx, ny);
      coeff[c] = 0.4 + 0.3 * ix;
      potential[c] = std::sin(0.8 * ix) + 0.15 * iy;
    }
  }
  Halos ch, ph;
  ch.wrap(coeff, nx, ny, nz);
  ph.wrap(potential, nx, ny, nz);
  std::vector<double> full(coeff.size());
  divergence_separated(coeff, potential, ch.ptrs, ph.ptrs, full, nx, ny, nz, active,
                       0.5, 0.5, 1.0, FaceAverage::Arithmetic);

  double err = 0.0;
  double split_sum = 0.0;
  for (int x0 = 0; x0 < nx; ++x0) {
    std::vector<double> c(static_cast<std::size_t>(ny));
    std::vector<double> p(c.size());
    for (int iy = 0; iy < ny; ++iy) {
      c[static_cast<std::size_t>(iy)] = coeff[idx(x0, iy, 0, nx, ny)];
      p[static_cast<std::size_t>(iy)] = potential[idx(x0, iy, 0, nx, ny)];
    }
    Halos hc, hp;
    hc.storage[0].resize(static_cast<std::size_t>(ny));
    hc.storage[1].resize(static_cast<std::size_t>(ny));
    hc.storage[2].resize(1);
    hc.storage[3].resize(1);
    hc.storage[4].resize(static_cast<std::size_t>(ny));
    hc.storage[5] = hc.storage[4];
    hp.storage = hc.storage;
    const int xp = (x0 + 1) % nx;
    const int xm = (x0 + nx - 1) % nx;
    for (int iy = 0; iy < ny; ++iy) {
      hc.storage[0][static_cast<std::size_t>(iy)] = coeff[idx(xp, iy, 0, nx, ny)];
      hc.storage[1][static_cast<std::size_t>(iy)] = coeff[idx(xm, iy, 0, nx, ny)];
      hp.storage[0][static_cast<std::size_t>(iy)] =
          potential[idx(xp, iy, 0, nx, ny)];
      hp.storage[1][static_cast<std::size_t>(iy)] =
          potential[idx(xm, iy, 0, nx, ny)];
    }
    hc.storage[2][0] = c[0];
    hc.storage[3][0] = c[static_cast<std::size_t>(ny - 1)];
    hp.storage[2][0] = p[0];
    hp.storage[3][0] = p[static_cast<std::size_t>(ny - 1)];
    for (int i = 0; i < 6; ++i) {
      hc.ptrs[static_cast<std::size_t>(i)] =
          hc.storage[static_cast<std::size_t>(i)].data();
      hp.ptrs[static_cast<std::size_t>(i)] =
          hp.storage[static_cast<std::size_t>(i)].data();
    }
    std::vector<double> rhs(c.size());
    divergence_separated(c, p, hc.ptrs, hp.ptrs, rhs, 1, ny, nz, active, 0.5, 0.5,
                         1.0, FaceAverage::Arithmetic);
    for (int iy = 0; iy < ny; ++iy) {
      err = std::max(err, std::abs(rhs[static_cast<std::size_t>(iy)] -
                                   full[idx(x0, iy, 0, nx, ny)]));
      split_sum += rhs[static_cast<std::size_t>(iy)];
    }
  }
  REQUIRE(err < 1.0e-12);
  REQUIRE(std::abs(split_sum) < 1.0e-10);
  REQUIRE(std::abs(split_sum - sum_of(full)) < 1.0e-12);
  double peak = 0.0;
  for (double v : full) peak = std::max(peak, std::abs(v));
  REQUIRE(peak > 1.0e-6);
}

TEST_CASE("divergence_separated span extents", "[face_flux]") {
  std::array<const double *, 6> absent{};
  const std::array<bool, 3> idle{false, false, false};
  // A non-positive extent returns before any span is read.
  divergence_separated({}, {}, absent, absent, {}, 0, 2, 2, idle, 1.0, 1.0, 1.0,
                       FaceAverage::Arithmetic);

  const std::vector<double> coeff{1.0};
  const std::vector<double> potential{4.0};
  std::vector<double> rhs{9.0};
  divergence_separated(coeff, potential, absent, absent, rhs, 1, 1, 1, idle, 1.0,
                       1.0, 1.0, FaceAverage::Arithmetic);
  REQUIRE(rhs[0] == 0.0);

  const std::vector<double> too_long{1.0, 2.0};
  REQUIRE_THROWS_AS(divergence_separated(too_long, potential, absent, absent, rhs, 1,
                                         1, 1, idle, 1.0, 1.0, 1.0,
                                         FaceAverage::Arithmetic),
                    std::invalid_argument);
}

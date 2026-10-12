// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

// A coupled two-field test problem on FieldVectors (#471), shared by the host
// and device tests. Periodic n x n grid, h = 1/n:
//
//   (A u)_0 = c0 u0 - lap u0               + g u1 (+ u0^3)
//   (A u)_1 = c1 u1 - div(kappa grad u1)   + g u0 (+ u1^3)
//
// The stencil runs on host views (with_host_read/with_host_view), so on a
// device memory space the test exercises the device vector operations, halo
// exchange and multigrid while the action itself stays simple.

#include <cmath>
#include <cstddef>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/solvers/field_multigrid.hpp>
#include <openpfc/solvers/field_operations.hpp>
#include <openpfc/solvers/krylov.hpp>

namespace field_ops_test {

constexpr double kPi = 3.14159265358979323846;
constexpr double c0 = 1.0, c1 = 2.0, g = 0.3;

inline double kappa(double x, double y) {
  return 1.0 + 0.5 * std::sin(2 * kPi * x) * std::sin(2 * kPi * y);
}
inline double rhs(int comp, int i, int j, int n) {
  const double x = (i + 0.5) / n, y = (j + 0.5) / n;
  return comp == 0 ? std::cos(2 * kPi * x) + 0.5 * std::sin(4 * kPi * y)
                   : std::sin(2 * kPi * (x + y)) - 0.25;
}

/// Global flat reference: index (comp * n + j) * n + i, periodic, nonlinear if
/// cubic.
inline void flat_apply(int n, bool cubic, const std::vector<double> &u,
                       std::vector<double> &out,
                       const std::vector<double> *lin = nullptr) {
  const double ih2 = double(n) * n;
  auto at = [&](int c, int i, int j) {
    return u[(std::size_t(c) * n + ((j + n) % n)) * n + ((i + n) % n)];
  };
  out.assign(u.size(), 0.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n, y = (j + 0.5) / n, h = 1.0 / n;
      const double u0 = at(0, i, j), u1 = at(1, i, j);
      const double lap0 = (at(0, i + 1, j) + at(0, i - 1, j) + at(0, i, j + 1) +
                           at(0, i, j - 1) - 4 * u0) *
                          ih2;
      const double ke = kappa(x + h / 2, y), kw = kappa(x - h / 2, y),
                   kn = kappa(x, y + h / 2), ks = kappa(x, y - h / 2);
      const double div1 =
          (ke * (at(1, i + 1, j) - u1) - kw * (u1 - at(1, i - 1, j)) +
           kn * (at(1, i, j + 1) - u1) - ks * (u1 - at(1, i, j - 1))) *
          ih2;
      double n0 = 0, n1 = 0;
      if (cubic) {
        n0 = u0 * u0 * u0;
        n1 = u1 * u1 * u1;
      }
      if (lin) {
        const double l0 = (*lin)[std::size_t(j) * n + i];
        const double l1 = (*lin)[(std::size_t(n) + j) * n + i];
        n0 = 3 * l0 * l0 * u0;
        n1 = 3 * l1 * l1 * u1;
      }
      out[std::size_t(j) * n + i] = c0 * u0 - lap0 + g * u1 + n0;
      out[(std::size_t(n) + j) * n + i] = c1 * u1 - div1 + g * u0 + n1;
    }
}

template <class Space> struct Problem {
  using Ops = pfc::solvers::FieldOperations<Space>;
  using Vector = typename Ops::Vector;
  using FieldT = typename Ops::FieldT;
  using MG = pfc::solvers::FieldMultigrid<Space>;

  int n, rank = 0, nproc = 1;
  pfc::decomposition::Decomposition decomp;
  pfc::Box3i box;
  Ops ops;
  pfc::solvers::FieldHalo<Space> halo;
  MG mg0, mg1;
  Vector lin; ///< linearisation point of the cubic term (Newton), zero otherwise
  bool cubic = false;

  static pfc::decomposition::Decomposition make_decomp(int n) {
    int p = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    const double h = 1.0 / n;
    auto d = pfc::domain::with_spacing({n, n, 1}, {h, h, 1.0}, {true, true, true});
    return pfc::decomposition::create(d, p);
  }
  static int my_rank() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
  }
  static pfc::comm::HaloExchangeOptions faces(int base) {
    pfc::comm::HaloExchangeOptions o;
    o.directions = pfc::halo::presets::Axes2D();
    o.exchange_base = base;
    return o;
  }

  explicit Problem(int n_)
      : n(n_), rank(my_rank()), decomp(make_decomp(n_)),
        box(pfc::decomposition::local_box(decomp, rank)),
        ops(pfc::decomposition::domain(decomp), box, 2, 1, MPI_COMM_WORLD),
        halo(ops, decomp, rank, MPI_COMM_WORLD, faces(500)),
        mg0(decomp, rank, MPI_COMM_WORLD, controls(7000)),
        mg1(decomp, rank, MPI_COMM_WORLD, controls(17000)),
        lin(ops.make(ops.size())) {
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    update_preconditioner();
  }
  static typename MG::Controls controls(int base) {
    typename MG::Controls c;
    c.tag_base = base;
    return c;
  }

  /// Loop over owned cells with global indices.
  template <class Fn> void owned(const FieldT &f, Fn &&fn) const {
    const auto s = f.local_size();
    for (int j = 0; j < s[1]; ++j)
      for (int i = 0; i < s[0]; ++i) fn(i, j, box.low[0] + i, box.low[1] + j);
  }

  /// Multigrid coefficients: c (+ 3 lin^2 when linearised), D = 1 and kappa.
  void update_preconditioner() {
    const double h = 1.0 / n;
    std::vector<double> l0, l1;
    read(lin, 0, l0);
    read(lin, 1, l1);
    for (int comp = 0; comp < 2; ++comp) {
      MG &mg = comp ? mg1 : mg0;
      const auto &l = comp ? l1 : l0;
      auto set = [&](FieldT &f, auto &&value) {
        f.with_host_view([&](double *p, std::size_t) {
          owned(f, [&](int i, int j, int gi, int gj) {
            p[f.idx(i, j, 0)] = value(i, j, gi, gj);
          });
        });
      };
      const auto s = box.size;
      set(mg.c(), [&](int i, int j, int, int) {
        const double v = l[std::size_t(j) * s[0] + i];
        return (comp ? c1 : c0) + 3 * v * v;
      });
      set(mg.dx(), [&](int, int, int gi, int gj) {
        return comp ? kappa((gi + 1.0) * h, (gj + 0.5) * h) : 1.0;
      });
      set(mg.dy(), [&](int, int, int gi, int gj) {
        return comp ? kappa((gi + 0.5) * h, (gj + 1.0) * h) : 1.0;
      });
      set(mg.dz(), [](int, int, int, int) { return 0.0; });
      set(mg.mask(), [](int, int, int, int) { return 1.0; });
      mg.update();
    }
  }

  /// Owned values of component `comp` (row-major, x fastest).
  void read(const Vector &v, int comp, std::vector<double> &out) const {
    const auto &f = v[comp];
    out.assign(std::size_t(box.size[0]) * box.size[1], 0.0);
    const_cast<FieldT &>(f).with_host_read([&](const double *p, std::size_t) {
      owned(f, [&](int i, int j, int, int) {
        out[std::size_t(j) * box.size[0] + i] = p[f.idx(i, j, 0)];
      });
    });
  }
  void write(Vector &v, int comp, const std::vector<double> &in) const {
    auto &f = v[comp];
    f.with_host_view([&](double *p, std::size_t) {
      owned(f, [&](int i, int j, int, int) {
        p[f.idx(i, j, 0)] = in[std::size_t(j) * box.size[0] + i];
      });
    });
    f.sync_to_device();
  }

  Vector right_hand_side() {
    Vector b = ops.make(ops.size());
    for (int comp = 0; comp < 2; ++comp) {
      std::vector<double> v(std::size_t(box.size[0]) * box.size[1]);
      owned(b[comp], [&](int i, int j, int gi, int gj) {
        v[std::size_t(j) * box.size[0] + i] = rhs(comp, gi, gj, n);
      });
      write(b, comp, v);
    }
    return b;
  }

  /// y = A x (cubic: the nonlinear operator; jacobian: linearised at `lin`).
  void apply(const Vector &x, Vector &y, bool nonlinear, bool jacobian) {
    const Vector &u = halo(x);
    const double ih2 = double(n) * n, h = 1.0 / n;
    std::vector<double> l0, l1;
    if (jacobian) {
      read(lin, 0, l0);
      read(lin, 1, l1);
    }
    const FieldT &a = u[0], &b = u[1];
    std::vector<double> o0(std::size_t(box.size[0]) * box.size[1]), o1(o0.size());
    const_cast<FieldT &>(a).with_host_read([&](const double *p0, std::size_t) {
      const_cast<FieldT &>(b).with_host_read([&](const double *p1, std::size_t) {
        owned(a, [&](int i, int j, int gi, int gj) {
          const double x0 = (gi + 0.5) * h, y0 = (gj + 0.5) * h;
          auto A = [&](int di, int dj) { return p0[a.idx(i + di, j + dj, 0)]; };
          auto B = [&](int di, int dj) { return p1[b.idx(i + di, j + dj, 0)]; };
          const double u0 = A(0, 0), u1 = B(0, 0);
          const double lap0 =
              (A(1, 0) + A(-1, 0) + A(0, 1) + A(0, -1) - 4 * u0) * ih2;
          const double ke = kappa(x0 + h / 2, y0), kw = kappa(x0 - h / 2, y0),
                       kn = kappa(x0, y0 + h / 2), ks = kappa(x0, y0 - h / 2);
          const double div1 = (ke * (B(1, 0) - u1) - kw * (u1 - B(-1, 0)) +
                               kn * (B(0, 1) - u1) - ks * (u1 - B(0, -1))) *
                              ih2;
          const std::size_t q = std::size_t(j) * box.size[0] + i;
          double n0 = 0, n1 = 0;
          if (nonlinear) {
            n0 = u0 * u0 * u0;
            n1 = u1 * u1 * u1;
          }
          if (jacobian) {
            n0 = 3 * l0[q] * l0[q] * u0;
            n1 = 3 * l1[q] * l1[q] * u1;
          }
          o0[q] = c0 * u0 - lap0 + g * u1 + n0;
          o1[q] = c1 * u1 - div1 + g * u0 + n1;
        });
      });
    });
    write(y, 0, o0);
    write(y, 1, o1);
  }

  /// Block-diagonal multigrid z = diag(M0, M1)^{-1} r.
  void precondition(const Vector &r, Vector &z) {
    mg0.apply(r[0], z[0]);
    mg1.apply(r[1], z[1]);
  }

  /// Gather the owned values into the global flat layout (all ranks get all).
  std::vector<double> gather(const Vector &v) const {
    std::vector<double> local(std::size_t(2) * n * n, 0.0), global(local.size());
    for (int comp = 0; comp < 2; ++comp) {
      std::vector<double> o;
      read(v, comp, o);
      owned(v[comp], [&](int i, int j, int gi, int gj) {
        local[(std::size_t(comp) * n + gj) * n + gi] =
            o[std::size_t(j) * box.size[0] + i];
      });
    }
    MPI_Allreduce(local.data(), global.data(), int(local.size()), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    return global;
  }
};

/// Newton model on the cubic problem: R(u) = A_cubic u - b.
template <class Space> struct CubicModel {
  using P = Problem<Space>;
  P &p;
  typename P::Vector b;
  int linearizations = 0;
  explicit CubicModel(P &problem) : p(problem), b(problem.right_hand_side()) {}
  void residual(const typename P::Vector &x, typename P::Vector &r) {
    p.apply(x, r, true, false);
    p.ops.axpy(r, -1.0, b);
  }
  void linearize(const typename P::Vector &x) {
    p.ops.copy(x, p.lin);
    p.update_preconditioner();
    ++linearizations;
  }
  void jacobian(const typename P::Vector &v, typename P::Vector &w) {
    p.apply(v, w, false, true);
  }
  void precondition(const typename P::Vector &r, typename P::Vector &z, int) {
    p.precondition(r, z);
  }
  void begin_trial(const typename P::Vector &) {}
  void accept_trial(const typename P::Vector &) {}
  void reject_trial() {}
};

} // namespace field_ops_test

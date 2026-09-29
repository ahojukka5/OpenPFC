// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tensor.hpp
 * @brief Second-order tensor storage for the finite-strain FFT solver.
 *
 * Components use the reference-configuration convention `c[i][j]`, with `i`
 * the row and `j` the column, matching `F_ij = dx_i / dX_j`.
 */

#pragma once

#include <cmath>
#include <cstddef>

namespace pfc::finite_strain {

/// A 3×3 tensor. Plain storage so a grid of them is an FFT-sized array.
struct Tensor2 {
  double c[3][3]{};

  double &operator()(int i, int j) noexcept { return c[i][j]; }
  double operator()(int i, int j) const noexcept { return c[i][j]; }
};

[[nodiscard]] inline Tensor2 identity2() noexcept {
  Tensor2 out;
  out(0, 0) = 1.0;
  out(1, 1) = 1.0;
  out(2, 2) = 1.0;
  return out;
}

[[nodiscard]] inline Tensor2 transposed(const Tensor2 &a) noexcept {
  Tensor2 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = a(j, i);
    }
  }
  return out;
}

[[nodiscard]] inline double trace(const Tensor2 &a) noexcept {
  return a(0, 0) + a(1, 1) + a(2, 2);
}

[[nodiscard]] inline Tensor2 matmul(const Tensor2 &a, const Tensor2 &b) noexcept {
  Tensor2 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = a(i, 0) * b(0, j) + a(i, 1) * b(1, j) + a(i, 2) * b(2, j);
    }
  }
  return out;
}

[[nodiscard]] inline Tensor2 scaled(const Tensor2 &a, double s) noexcept {
  Tensor2 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = s * a(i, j);
    }
  }
  return out;
}

[[nodiscard]] inline Tensor2 add(const Tensor2 &a, const Tensor2 &b) noexcept {
  Tensor2 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = a(i, j) + b(i, j);
    }
  }
  return out;
}

[[nodiscard]] inline Tensor2 axpy(double alpha, const Tensor2 &x,
                                  const Tensor2 &y) noexcept {
  Tensor2 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = alpha * x(i, j) + y(i, j);
    }
  }
  return out;
}

/// Symmetric part `(A + A^T) / 2`.
[[nodiscard]] inline Tensor2 symmetric_part(const Tensor2 &a) noexcept {
  Tensor2 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = 0.5 * (a(i, j) + a(j, i));
    }
  }
  return out;
}

[[nodiscard]] inline double frobenius_dot(const Tensor2 &a,
                                          const Tensor2 &b) noexcept {
  double sum = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      sum += a(i, j) * b(i, j);
    }
  }
  return sum;
}

[[nodiscard]] inline double frobenius_norm(const Tensor2 &a) noexcept {
  return std::sqrt(frobenius_dot(a, a));
}

} // namespace pfc::finite_strain

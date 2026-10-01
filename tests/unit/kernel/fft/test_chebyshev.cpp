// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/fft/chebyshev.hpp>

using Catch::Matchers::WithinAbs;

namespace {

[[nodiscard]] std::vector<double> mode_values(int degree, int mode) {
  const auto nodes = pfc::fft::chebyshev_lobatto(degree);
  std::vector<double> values(nodes.size());
  for (std::size_t j = 0; j < nodes.size(); ++j) {
    values[j] = std::cos(mode * std::acos(std::clamp(nodes[j], -1.0, 1.0)));
  }
  return values;
}

[[nodiscard]] std::vector<double>
matrix_derivative(const std::vector<double> &values) {
  const int n = static_cast<int>(values.size()) - 1;
  const auto x = pfc::fft::chebyshev_lobatto(n);
  std::vector<double> c(values.size(), 1.0);
  c.front() = 2.0;
  c.back() = 2.0;
  for (int i = 0; i <= n; ++i) {
    if (i % 2 != 0) c[static_cast<std::size_t>(i)] *= -1.0;
  }
  std::vector<std::vector<double>> dense(values.size(),
                                         std::vector<double>(values.size()));
  for (int i = 0; i <= n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j <= n; ++j) {
      if (i == j) continue;
      const double entry = (c[static_cast<std::size_t>(i)] /
                            c[static_cast<std::size_t>(j)]) /
                           (x[static_cast<std::size_t>(i)] -
                            x[static_cast<std::size_t>(j)]);
      dense[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = entry;
      row_sum += entry;
    }
    dense[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = -row_sum;
  }
  std::vector<double> derivative(values.size());
  for (int i = 0; i <= n; ++i) {
    double sum = 0.0;
    for (int j = 0; j <= n; ++j) {
      sum += dense[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
             values[static_cast<std::size_t>(j)];
    }
    derivative[static_cast<std::size_t>(i)] = sum;
  }
  return derivative;
}

template <typename Sample>
[[nodiscard]] std::vector<double> tensor_field(int nx, int degree, double period,
                                               Sample sample) {
  const auto z = pfc::fft::chebyshev_lobatto(degree);
  std::vector<double> values(static_cast<std::size_t>(nx) * z.size());
  for (int iz = 0; iz <= degree; ++iz) {
    for (int ix = 0; ix < nx; ++ix) {
      const double x =
          period * static_cast<double>(ix) / static_cast<double>(nx);
      values[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx) +
             static_cast<std::size_t>(ix)] =
          sample(x, z[static_cast<std::size_t>(iz)]);
    }
  }
  return values;
}

[[nodiscard]] double interval_integral(const std::vector<double> &values) {
  const auto coefficients = pfc::fft::chebyshev_coefficients(values);
  double sum = 0.0;
  for (int k = 0; k < static_cast<int>(coefficients.size()); ++k) {
    if (k % 2 != 0) continue;
    const double mode = static_cast<double>(k);
    const double weight = (k == 0) ? 2.0 : 2.0 / (1.0 - mode * mode);
    sum += coefficients[static_cast<std::size_t>(k)] * weight;
  }
  return sum;
}

[[nodiscard]] double max_abs_diff(const std::vector<double> &left,
                                  const std::vector<double> &right) {
  double peak = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    peak = std::max(peak, std::abs(left[i] - right[i]));
  }
  return peak;
}

} // namespace

TEST_CASE("Chebyshev coefficients reproduce the Lobatto samples",
          "[fft][chebyshev]") {
  const auto nodes = pfc::fft::chebyshev_lobatto(8);
  REQUIRE_THAT(nodes.front(), WithinAbs(1.0, 0.0));
  REQUIRE_THAT(nodes.back(), WithinAbs(-1.0, 1e-15));

  for (int mode = 0; mode <= 8; ++mode) {
    const auto values = mode_values(8, mode);
    const auto coefficients = pfc::fft::chebyshev_coefficients(values);
    for (int k = 0; k <= 8; ++k) {
      const double expect = k == mode ? 1.0 : 0.0;
      REQUIRE_THAT(coefficients[static_cast<std::size_t>(k)],
                   WithinAbs(expect, 1e-12));
    }
    const auto back = pfc::fft::chebyshev_values(coefficients);
    REQUIRE(max_abs_diff(back, values) < 1e-12);
  }
}

TEST_CASE("Chebyshev derivative matches the differentiation matrix",
          "[fft][chebyshev]") {
  const int degree = 12;
  const auto nodes = pfc::fft::chebyshev_lobatto(degree);
  std::vector<double> values(nodes.size());
  for (std::size_t j = 0; j < nodes.size(); ++j) {
    values[j] = std::exp(nodes[j]);
  }
  const auto spectral = pfc::fft::chebyshev_derivative(values);
  const auto matrix = matrix_derivative(values);
  REQUIRE(max_abs_diff(spectral, matrix) < 1e-10);

  const auto linear = mode_values(1, 1);
  const auto slope = pfc::fft::chebyshev_derivative(linear);
  REQUIRE_THAT(slope[0], WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(slope[1], WithinAbs(1.0, 1e-12));

  const auto grid = pfc::fft::chebyshev_lobatto(6);
  const auto quadratic = mode_values(6, 2);
  const auto quad = pfc::fft::chebyshev_derivative(quadratic);
  REQUIRE_THAT(quad.front(), WithinAbs(4.0, 1e-11));
  REQUIRE_THAT(quad.back(), WithinAbs(-4.0, 1e-11));
  for (std::size_t j = 1; j + 1 < quad.size(); ++j) {
    REQUIRE_THAT(quad[j], WithinAbs(4.0 * grid[j], 1e-11));
  }
}

TEST_CASE("Chebyshev derivative of exp converges on the Lobatto grid",
          "[fft][chebyshev]") {
  const auto error_at = [](int degree) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> values(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) values[j] = std::exp(nodes[j]);
    const auto derivative = pfc::fft::chebyshev_derivative(values);
    double peak = 0.0;
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      peak = std::max(peak, std::abs(derivative[j] - values[j]));
    }
    return peak;
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-8);
  REQUIRE(fine < coarse);
}

TEST_CASE("Fourier-Chebyshev Laplacian matches separable polynomials",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto laplace = [&](int nx, int degree, auto sample) {
    const auto values = tensor_field(nx, degree, period, sample);
    return pfc::fft::fourier_chebyshev_laplacian(values, nx, period);
  };

  const auto constant = laplace(8, 4, [](double, double) { return 1.0; });
  REQUIRE(max_abs_diff(constant, std::vector<double>(constant.size(), 0.0)) <
          1e-12);

  const auto along_z = laplace(1, 8, [](double, double z) {
    return 2.0 * z * z - 1.0;
  });
  REQUIRE(max_abs_diff(along_z, std::vector<double>(along_z.size(), 4.0)) <
          1e-10);

  const auto product = tensor_field(16, 8, period, [](double x, double z) {
    return std::cos(2.0 * x) * (2.0 * z * z - 1.0);
  });
  const auto product_lap =
      pfc::fft::fourier_chebyshev_laplacian(product, 16, period);
  std::vector<double> product_expect(product.size());
  const auto z = pfc::fft::chebyshev_lobatto(8);
  for (int iz = 0; iz <= 8; ++iz) {
    const double t2 = 2.0 * z[static_cast<std::size_t>(iz)] *
                          z[static_cast<std::size_t>(iz)] -
                      1.0;
    for (int ix = 0; ix < 16; ++ix) {
      const double x = period * static_cast<double>(ix) / 16.0;
      product_expect[static_cast<std::size_t>(iz) * 16 +
                     static_cast<std::size_t>(ix)] =
          std::cos(2.0 * x) * (4.0 - 4.0 * t2);
    }
  }
  REQUIRE(max_abs_diff(product_lap, product_expect) < 1e-9);

  const auto nyquist = tensor_field(8, 2, period, [](double x, double) {
    return std::cos(4.0 * x);
  });
  const auto nyquist_lap =
      pfc::fft::fourier_chebyshev_laplacian(nyquist, 8, period);
  std::vector<double> nyquist_expect(nyquist.size());
  for (std::size_t i = 0; i < nyquist.size(); ++i) {
    nyquist_expect[i] = -16.0 * nyquist[i];
  }
  REQUIRE(max_abs_diff(nyquist_lap, nyquist_expect) < 1e-9);
}

TEST_CASE("Fourier-Chebyshev Laplacian of a smooth mode converges in z",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto error_at = [&](int degree) {
    const auto values = tensor_field(8, degree, period, [](double x, double z) {
      return std::sin(2.0 * x) * std::exp(z);
    });
    const auto got =
        pfc::fft::fourier_chebyshev_laplacian(values, 8, period);
    const auto z = pfc::fft::chebyshev_lobatto(degree);
    double peak = 0.0;
    for (int iz = 0; iz <= degree; ++iz) {
      for (int ix = 0; ix < 8; ++ix) {
        const double x = period * static_cast<double>(ix) / 8.0;
        const double expect = -3.0 * std::sin(2.0 * x) *
                              std::exp(z[static_cast<std::size_t>(iz)]);
        const auto index = static_cast<std::size_t>(iz) * 8 +
                           static_cast<std::size_t>(ix);
        peak = std::max(peak, std::abs(got[index] - expect));
      }
    }
    return peak;
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-8);
  REQUIRE(fine < coarse);
}

TEST_CASE("Fourier-Chebyshev Laplacian rejects a bad grid",
          "[fft][chebyshev]") {
  const std::vector<double> values{1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_laplacian({}, 4, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_laplacian(values, 0, 1.0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_laplacian(values, 3, 1.0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_laplacian(values, 4, 0.0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_laplacian(values, 4, -1.0),
      std::invalid_argument);
}

TEST_CASE("Chebyshev Dirichlet Poisson recovers polynomial solutions",
          "[fft][chebyshev]") {
  const auto error_of = [](int degree, auto force, auto exact, double at_plus,
                           double at_minus) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = force(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got =
        pfc::fft::chebyshev_dirichlet_poisson(forcing, at_plus, at_minus);
    REQUIRE_THAT(got.front(), WithinAbs(at_plus, 1e-12));
    REQUIRE_THAT(got.back(), WithinAbs(at_minus, 1e-12));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              8, [](double) { return -2.0; },
              [](double x) { return 1.0 - x * x; }, 0.0, 0.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return 0.0; }, [](double x) { return x; }, 1.0,
              -1.0) < 1e-12);
  REQUIRE(error_of(
              10, [](double x) { return -6.0 * x; },
              [](double x) { return x - x * x * x; }, 0.0, 0.0) < 1e-11);

  const auto nodes = pfc::fft::chebyshev_lobatto(12);
  const std::vector<double> forcing(nodes.size(), -2.0);
  const auto solution =
      pfc::fft::chebyshev_dirichlet_poisson(forcing, 0.0, 0.0);
  const auto second =
      pfc::fft::chebyshev_derivative(pfc::fft::chebyshev_derivative(solution));
  REQUIRE(max_abs_diff(second, forcing) < 1e-9);
}

TEST_CASE("Chebyshev Dirichlet Poisson converges for a smooth forcing",
          "[fft][chebyshev]") {
  const auto error_at = [](int degree) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = std::exp(nodes[j]);
      truth[j] = std::exp(nodes[j]);
    }
    const auto got = pfc::fft::chebyshev_dirichlet_poisson(
        forcing, std::exp(1.0), std::exp(-1.0));
    return max_abs_diff(got, truth);
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-10);
  REQUIRE(fine < coarse);
}

TEST_CASE("Chebyshev Dirichlet Poisson rejects a one-point grid",
          "[fft][chebyshev]") {
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_dirichlet_poisson({}, 0.0, 0.0),
                    std::invalid_argument);
  const std::vector<double> one{0.0};
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_dirichlet_poisson(one, 0.0, 1.0),
                    std::invalid_argument);
}

TEST_CASE("Chebyshev Neumann Poisson recovers mean-zero polynomials",
          "[fft][chebyshev]") {
  const auto error_of = [](int degree, auto force, auto exact, double at_plus,
                           double at_minus) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = force(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got = pfc::fft::chebyshev_neumann_poisson(forcing, at_plus, at_minus);
    const auto slope = pfc::fft::chebyshev_derivative(got);
    REQUIRE_THAT(slope.front(), WithinAbs(at_plus, 1e-9));
    REQUIRE_THAT(slope.back(), WithinAbs(at_minus, 1e-9));
    REQUIRE_THAT(interval_integral(got), WithinAbs(0.0, 1e-10));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              8, [](double) { return -2.0; },
              [](double x) { return 1.0 / 3.0 - x * x; }, -2.0, 2.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return 0.0; }, [](double x) { return x; }, 1.0, 1.0) <
          1e-12);
  REQUIRE(error_of(
              10, [](double x) { return -6.0 * x; },
              [](double x) { return x - x * x * x; }, -2.0, -2.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return -2.0; },
              [](double x) { return 1.0 / 3.0 - x * x + x; }, -1.0, 3.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return 0.0; }, [](double) { return 0.0; }, 0.0, 0.0) <
          1e-12);

  const auto nodes = pfc::fft::chebyshev_lobatto(12);
  const std::vector<double> forcing(nodes.size(), -2.0);
  const auto solution = pfc::fft::chebyshev_neumann_poisson(forcing, -2.0, 2.0);
  const auto second =
      pfc::fft::chebyshev_derivative(pfc::fft::chebyshev_derivative(solution));
  REQUIRE(max_abs_diff(second, forcing) < 1e-9);
}

TEST_CASE("Chebyshev Neumann Poisson rejects incompatible data",
          "[fft][chebyshev]") {
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_poisson({}, 0.0, 0.0),
                    std::invalid_argument);
  const std::vector<double> one{0.0};
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_poisson(one, 0.0, 0.0),
                    std::invalid_argument);

  const auto nodes = pfc::fft::chebyshev_lobatto(8);
  const std::vector<double> constant(nodes.size(), -2.0);
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_poisson(constant, 0.0, 0.0),
                    std::invalid_argument);
  const std::vector<double> zero(nodes.size(), 0.0);
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_poisson(zero, 1.0, -1.0),
                    std::invalid_argument);
}

TEST_CASE("Chebyshev Dirichlet Helmholtz recovers polynomial solutions",
          "[fft][chebyshev]") {
  const auto error_of = [](int degree, double lambda, auto force, auto exact,
                           double at_plus, double at_minus) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = force(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got =
        pfc::fft::chebyshev_dirichlet_helmholtz(forcing, lambda, at_plus, at_minus);
    REQUIRE_THAT(got.front(), WithinAbs(at_plus, 1e-12));
    REQUIRE_THAT(got.back(), WithinAbs(at_minus, 1e-12));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              8, 0.0, [](double) { return -2.0; },
              [](double x) { return 1.0 - x * x; }, 0.0, 0.0) < 1e-11);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return x * x - 3.0; },
              [](double x) { return 1.0 - x * x; }, 0.0, 0.0) < 1e-11);
  REQUIRE(error_of(
              8, -1.0, [](double x) { return x; }, [](double x) { return x; }, 1.0,
              -1.0) < 1e-12);

  const auto nodes = pfc::fft::chebyshev_lobatto(12);
  std::vector<double> forcing(nodes.size());
  std::vector<double> field(nodes.size());
  for (std::size_t j = 0; j < nodes.size(); ++j) {
    const double x = nodes[j];
    forcing[j] = x * x - 3.0;
    field[j] = 1.0 - x * x;
  }
  const auto solution =
      pfc::fft::chebyshev_dirichlet_helmholtz(forcing, 1.0, 0.0, 0.0);
  const auto second =
      pfc::fft::chebyshev_derivative(pfc::fft::chebyshev_derivative(solution));
  std::vector<double> residual(solution.size());
  for (std::size_t j = 0; j < solution.size(); ++j) {
    residual[j] = second[j] - solution[j];
  }
  REQUIRE(max_abs_diff(residual, forcing) < 1e-9);
  REQUIRE(max_abs_diff(solution, field) < 1e-11);
}

TEST_CASE("Chebyshev Dirichlet Helmholtz converges for a smooth solution",
          "[fft][chebyshev]") {
  const auto error_at = [](int degree) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    const std::vector<double> forcing(nodes.size(), 0.0);
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      truth[j] = std::exp(nodes[j]);
    }
    const auto got = pfc::fft::chebyshev_dirichlet_helmholtz(
        forcing, 1.0, std::exp(1.0), std::exp(-1.0));
    return max_abs_diff(got, truth);
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-10);
  REQUIRE(fine < coarse);
}

TEST_CASE("Chebyshev Dirichlet Helmholtz rejects a one-point grid",
          "[fft][chebyshev]") {
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_dirichlet_helmholtz({}, 1.0, 0.0, 0.0),
                    std::invalid_argument);
  const std::vector<double> one{0.0};
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_dirichlet_helmholtz(one, 1.0, 0.0, 1.0),
                    std::invalid_argument);
}

TEST_CASE("Chebyshev Neumann Helmholtz recovers polynomial solutions",
          "[fft][chebyshev]") {
  const auto error_of = [](int degree, double lambda, auto force, auto exact,
                           double at_plus, double at_minus) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = force(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got =
        pfc::fft::chebyshev_neumann_helmholtz(forcing, lambda, at_plus, at_minus);
    const auto slope = pfc::fft::chebyshev_derivative(got);
    REQUIRE_THAT(slope.front(), WithinAbs(at_plus, 1e-9));
    REQUIRE_THAT(slope.back(), WithinAbs(at_minus, 1e-9));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              8, 0.0, [](double) { return -2.0; },
              [](double x) { return 1.0 / 3.0 - x * x; }, -2.0, 2.0) < 1e-11);
  const auto gauged = pfc::fft::chebyshev_lobatto(8);
  std::vector<double> gauged_force(gauged.size(), -2.0);
  const auto gauged_solution =
      pfc::fft::chebyshev_neumann_helmholtz(gauged_force, 0.0, -2.0, 2.0);
  REQUIRE_THAT(interval_integral(gauged_solution), WithinAbs(0.0, 1e-10));

  REQUIRE(error_of(
              8, 1.0, [](double x) { return x * x - 7.0 / 3.0; },
              [](double x) { return 1.0 / 3.0 - x * x; }, -2.0, 2.0) < 1e-11);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return -x; }, [](double x) { return x; }, 1.0,
              1.0) < 1e-12);

  const auto nodes = pfc::fft::chebyshev_lobatto(12);
  std::vector<double> forcing(nodes.size());
  for (std::size_t j = 0; j < nodes.size(); ++j) {
    forcing[j] = nodes[j] * nodes[j] - 7.0 / 3.0;
  }
  const auto solution =
      pfc::fft::chebyshev_neumann_helmholtz(forcing, 1.0, -2.0, 2.0);
  const auto second =
      pfc::fft::chebyshev_derivative(pfc::fft::chebyshev_derivative(solution));
  std::vector<double> residual(solution.size());
  for (std::size_t j = 0; j < solution.size(); ++j) {
    residual[j] = second[j] - solution[j];
  }
  REQUIRE(max_abs_diff(residual, forcing) < 1e-9);
}

TEST_CASE("Chebyshev Neumann Helmholtz converges for a smooth solution",
          "[fft][chebyshev]") {
  const auto error_at = [](int degree) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    const std::vector<double> forcing(nodes.size(), 0.0);
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      truth[j] = std::exp(nodes[j]);
    }
    const auto got = pfc::fft::chebyshev_neumann_helmholtz(
        forcing, 1.0, std::exp(1.0), std::exp(-1.0));
    return max_abs_diff(got, truth);
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-10);
  REQUIRE(fine < coarse);
}

TEST_CASE("Chebyshev Neumann Helmholtz rejects a one-point grid",
          "[fft][chebyshev]") {
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_helmholtz({}, 1.0, 0.0, 0.0),
                    std::invalid_argument);
  const std::vector<double> one{0.0};
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_helmholtz(one, 1.0, 1.0, 1.0),
                    std::invalid_argument);
}

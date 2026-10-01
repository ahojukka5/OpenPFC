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

TEST_CASE("Fourier-Chebyshev Dirichlet Poisson recovers separable solutions",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto error_of = [&](int nx, int degree, auto exact, auto force) {
    const auto forcing = tensor_field(nx, degree, period, force);
    const auto truth = tensor_field(nx, degree, period, exact);
    std::vector<double> at_plus(static_cast<std::size_t>(nx));
    std::vector<double> at_minus(static_cast<std::size_t>(nx));
    for (int ix = 0; ix < nx; ++ix) {
      const double x = period * static_cast<double>(ix) / static_cast<double>(nx);
      at_plus[static_cast<std::size_t>(ix)] = exact(x, 1.0);
      at_minus[static_cast<std::size_t>(ix)] = exact(x, -1.0);
    }
    const auto got = pfc::fft::fourier_chebyshev_dirichlet_poisson(
        forcing, nx, period, at_plus, at_minus);
    for (int ix = 0; ix < nx; ++ix) {
      const auto plus = static_cast<std::size_t>(ix);
      const auto minus =
          static_cast<std::size_t>(degree) * static_cast<std::size_t>(nx) + plus;
      REQUIRE_THAT(got[plus], WithinAbs(at_plus[plus], 1e-10));
      REQUIRE_THAT(got[minus], WithinAbs(at_minus[plus], 1e-10));
    }
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              1, 8, [](double, double z) { return 1.0 - z * z; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(error_of(
              16, 8,
              [](double x, double z) { return std::cos(2.0 * x) * (1.0 - z * z); },
              [](double x, double z) {
                return std::cos(2.0 * x) * (4.0 * z * z - 6.0);
              }) < 1e-10);
  REQUIRE(error_of(
              16, 8, [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -4.0 * z * std::cos(2.0 * x); }) <
          1e-10);
  REQUIRE(error_of(
              8, 8, [](double, double z) { return 1.0 - z * z + 0.25 * z; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(error_of(
              8, 8,
              [](double x, double z) { return std::cos(4.0 * x) * (1.0 - z * z); },
              [](double x, double z) {
                return std::cos(4.0 * x) * (16.0 * z * z - 18.0);
              }) < 1e-10);

  const auto forcing = tensor_field(16, 8, period, [](double x, double z) {
    return std::cos(2.0 * x) * (4.0 * z * z - 6.0);
  });
  const std::vector<double> zeros(16, 0.0);
  const auto got = pfc::fft::fourier_chebyshev_dirichlet_poisson(forcing, 16, period,
                                                                 zeros, zeros);
  const auto residual = pfc::fft::fourier_chebyshev_laplacian(got, 16, period);
  REQUIRE(max_abs_diff(residual, forcing) < 1e-8);
}

TEST_CASE("Fourier-Chebyshev Dirichlet Poisson rejects a bad grid",
          "[fft][chebyshev]") {
  const std::vector<double> forcing{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> trace{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_trace{0.0};
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_dirichlet_poisson({}, 4, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_dirichlet_poisson(forcing, 0, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_dirichlet_poisson(forcing, 3, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_dirichlet_poisson(forcing, 4, 0.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_dirichlet_poisson(
                        forcing, 4, 1.0, short_trace, trace),
                    std::invalid_argument);
  const std::vector<double> one_line{1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_dirichlet_poisson(one_line, 4, 1.0, trace, trace),
      std::invalid_argument);
}

TEST_CASE("Fourier-Chebyshev Neumann Poisson recovers separable solutions",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto error_of = [&](int nx, int degree, auto exact, auto force, auto slope) {
    const auto forcing = tensor_field(nx, degree, period, force);
    const auto truth = tensor_field(nx, degree, period, exact);
    std::vector<double> slope_plus(static_cast<std::size_t>(nx));
    std::vector<double> slope_minus(static_cast<std::size_t>(nx));
    for (int ix = 0; ix < nx; ++ix) {
      const double x = period * static_cast<double>(ix) / static_cast<double>(nx);
      slope_plus[static_cast<std::size_t>(ix)] = slope(x, 1.0);
      slope_minus[static_cast<std::size_t>(ix)] = slope(x, -1.0);
    }
    const auto got = pfc::fft::fourier_chebyshev_neumann_poisson(
        forcing, nx, period, slope_plus, slope_minus);
    double mean_integral = 0.0;
    for (int ix = 0; ix < nx; ++ix) {
      std::vector<double> column(static_cast<std::size_t>(degree) + 1);
      for (int iz = 0; iz <= degree; ++iz) {
        column[static_cast<std::size_t>(iz)] =
            got[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx) +
                static_cast<std::size_t>(ix)];
      }
      const auto derivative = pfc::fft::chebyshev_derivative(column);
      REQUIRE_THAT(derivative.front(),
                   WithinAbs(slope_plus[static_cast<std::size_t>(ix)], 1e-9));
      REQUIRE_THAT(derivative.back(),
                   WithinAbs(slope_minus[static_cast<std::size_t>(ix)], 1e-9));
      mean_integral += interval_integral(column);
    }
    REQUIRE_THAT(mean_integral / static_cast<double>(nx), WithinAbs(0.0, 1e-10));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              1, 8, [](double, double z) { return 1.0 / 3.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double z) { return -2.0 * z; }) < 1e-11);
  REQUIRE(
      error_of(
          16, 8,
          [](double x, double z) { return std::cos(2.0 * x) * (1.0 / 3.0 - z * z); },
          [](double x, double z) {
            return std::cos(2.0 * x) * (4.0 * z * z - 10.0 / 3.0);
          },
          [](double x, double z) { return std::cos(2.0 * x) * (-2.0 * z); }) <
      1e-10);
  REQUIRE(error_of(
              16, 8, [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -4.0 * z * std::cos(2.0 * x); },
              [](double x, double) { return std::cos(2.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              8, 8, [](double, double z) { return 1.0 / 3.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double z) { return -2.0 * z; }) < 1e-11);
  REQUIRE(
      error_of(
          8, 8,
          [](double x, double z) { return std::cos(4.0 * x) * (1.0 / 3.0 - z * z); },
          [](double x, double z) {
            return std::cos(4.0 * x) * (16.0 * z * z - 22.0 / 3.0);
          },
          [](double x, double z) { return std::cos(4.0 * x) * (-2.0 * z); }) <
      1e-10);
  REQUIRE(error_of(
              16, 8,
              [](double x, double z) {
                return 1.0 / 3.0 - z * z + z * std::cos(2.0 * x);
              },
              [](double x, double z) { return -2.0 - 4.0 * z * std::cos(2.0 * x); },
              [](double x, double z) { return -2.0 * z + std::cos(2.0 * x); }) <
          1e-10);

  const auto forcing = tensor_field(16, 8, period, [](double x, double z) {
    return std::cos(2.0 * x) * (4.0 * z * z - 10.0 / 3.0);
  });
  std::vector<double> slope_plus(16);
  std::vector<double> slope_minus(16);
  for (int ix = 0; ix < 16; ++ix) {
    const double x = period * static_cast<double>(ix) / 16.0;
    slope_plus[static_cast<std::size_t>(ix)] = -2.0 * std::cos(2.0 * x);
    slope_minus[static_cast<std::size_t>(ix)] = 2.0 * std::cos(2.0 * x);
  }
  const auto got = pfc::fft::fourier_chebyshev_neumann_poisson(
      forcing, 16, period, slope_plus, slope_minus);
  const auto residual = pfc::fft::fourier_chebyshev_laplacian(got, 16, period);
  REQUIRE(max_abs_diff(residual, forcing) < 1e-8);
}

TEST_CASE("Fourier-Chebyshev Neumann Poisson rejects a bad condition",
          "[fft][chebyshev]") {
  const std::vector<double> forcing{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> trace{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_trace{0.0};
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson({}, 4, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 0, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 3, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 4, 0.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 4, 1.0,
                                                                short_trace, trace),
                    std::invalid_argument);
  const std::vector<double> one_line{1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson(one_line, 4, 1.0, trace, trace),
      std::invalid_argument);

  const double period = 2.0 * std::acos(-1.0);
  const auto field = tensor_field(8, 8, period, [](double, double) { return -2.0; });
  const std::vector<double> flat(8, 0.0);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson(field, 8, period, flat, flat),
      std::invalid_argument);
}

TEST_CASE("Fourier-Chebyshev Robin Poisson recovers separable solutions",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto error_of = [&](int nx, int degree, double value_plus, double slope_plus,
                            double value_minus, double slope_minus, auto exact,
                            auto force, auto data) {
    const auto forcing = tensor_field(nx, degree, period, force);
    const auto truth = tensor_field(nx, degree, period, exact);
    std::vector<double> data_plus(static_cast<std::size_t>(nx));
    std::vector<double> data_minus(static_cast<std::size_t>(nx));
    for (int ix = 0; ix < nx; ++ix) {
      const double x = period * static_cast<double>(ix) / static_cast<double>(nx);
      data_plus[static_cast<std::size_t>(ix)] = data(x, 1.0);
      data_minus[static_cast<std::size_t>(ix)] = data(x, -1.0);
    }
    const auto got = pfc::fft::fourier_chebyshev_robin_poisson(
        forcing, nx, period, value_plus, slope_plus, data_plus, value_minus,
        slope_minus, data_minus);
    double mean_integral = 0.0;
    for (int ix = 0; ix < nx; ++ix) {
      std::vector<double> column(static_cast<std::size_t>(degree) + 1);
      for (int iz = 0; iz <= degree; ++iz) {
        column[static_cast<std::size_t>(iz)] =
            got[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx) +
                static_cast<std::size_t>(ix)];
      }
      const auto derivative = pfc::fft::chebyshev_derivative(column);
      REQUIRE_THAT(value_plus * column.front() + slope_plus * derivative.front(),
                   WithinAbs(data_plus[static_cast<std::size_t>(ix)], 1e-9));
      REQUIRE_THAT(value_minus * column.back() + slope_minus * derivative.back(),
                   WithinAbs(data_minus[static_cast<std::size_t>(ix)], 1e-9));
      mean_integral += interval_integral(column);
    }
    if (value_plus == 0.0 && value_minus == 0.0) {
      REQUIRE_THAT(mean_integral / static_cast<double>(nx), WithinAbs(0.0, 1e-10));
    }
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              1, 8, 1.0, 1.0, 1.0, -1.0,
              [](double, double z) { return 1.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(error_of(
              1, 8, 1.0, 1.0, 1.0, -1.0,
              [](double, double z) { return 1.0 - z * z + 0.5 * z; },
              [](double, double) { return -2.0; },
              [](double, double z) { return z > 0.0 ? -1.0 : -3.0; }) < 1e-11);
  REQUIRE(error_of(
              8, 8, 1.0, 1.0, 1.0, -1.0,
              [](double, double z) { return 1.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(
      error_of(
          16, 8, 1.0, 1.0, 1.0, -1.0,
          [](double x, double z) { return std::cos(2.0 * x) * (1.0 - z * z); },
          [](double x, double z) { return std::cos(2.0 * x) * (4.0 * z * z - 6.0); },
          [](double x, double) { return -2.0 * std::cos(2.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 1.0, 1.0, 1.0, -1.0,
              [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -4.0 * z * std::cos(2.0 * x); },
              [](double x, double z) {
                return (z > 0.0 ? 2.0 : -2.0) * std::cos(2.0 * x);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 1.0, 1.0, 1.0, -1.0,
              [](double x, double z) { return std::cos(4.0 * x) * (1.0 - z * z); },
              [](double x, double z) {
                return std::cos(4.0 * x) * (16.0 * z * z - 18.0);
              },
              [](double x, double) { return -2.0 * std::cos(4.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 1.0, 1.0, 1.0, -1.0,
              [](double x, double z) { return 1.0 - z * z + z * std::cos(2.0 * x); },
              [](double x, double z) { return -2.0 - 4.0 * z * std::cos(2.0 * x); },
              [](double x, double z) {
                return -2.0 + (z > 0.0 ? 2.0 : -2.0) * std::cos(2.0 * x);
              }) < 1e-10);
  REQUIRE(
      error_of(
          16, 8, 1.0, 0.0, 1.0, 0.0,
          [](double x, double z) { return std::cos(2.0 * x) * (1.0 - z * z); },
          [](double x, double z) { return std::cos(2.0 * x) * (4.0 * z * z - 6.0); },
          [](double, double) { return 0.0; }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 2.0, 0.0, 2.0, 0.0,
              [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -4.0 * z * std::cos(2.0 * x); },
              [](double x, double z) { return 2.0 * z * std::cos(2.0 * x); }) <
          1e-10);
  REQUIRE(error_of(
              8, 8, 0.0, 1.0, 0.0, 1.0,
              [](double, double z) { return 1.0 / 3.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double z) { return -2.0 * z; }) < 1e-11);
  REQUIRE(
      error_of(
          16, 8, 0.0, 1.0, 0.0, 1.0,
          [](double x, double z) { return std::cos(2.0 * x) * (1.0 / 3.0 - z * z); },
          [](double x, double z) {
            return std::cos(2.0 * x) * (4.0 * z * z - 10.0 / 3.0);
          },
          [](double x, double z) { return -2.0 * z * std::cos(2.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 1.0, 0.0, 0.0, 1.0,
              [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -4.0 * z * std::cos(2.0 * x); },
              [](double x, double) { return std::cos(2.0 * x); }) < 1e-10);

  const auto forcing = tensor_field(16, 8, period, [](double x, double z) {
    return std::cos(2.0 * x) * (4.0 * z * z - 6.0);
  });
  std::vector<double> data_plus(16);
  std::vector<double> data_minus(16);
  for (int ix = 0; ix < 16; ++ix) {
    const double x = period * static_cast<double>(ix) / 16.0;
    data_plus[static_cast<std::size_t>(ix)] = -2.0 * std::cos(2.0 * x);
    data_minus[static_cast<std::size_t>(ix)] = -2.0 * std::cos(2.0 * x);
  }
  const auto got = pfc::fft::fourier_chebyshev_robin_poisson(
      forcing, 16, period, 1.0, 1.0, data_plus, 1.0, -1.0, data_minus);
  const auto residual = pfc::fft::fourier_chebyshev_laplacian(got, 16, period);
  REQUIRE(max_abs_diff(residual, forcing) < 1e-8);
}

TEST_CASE("Fourier-Chebyshev Robin Poisson rejects a bad condition",
          "[fft][chebyshev]") {
  const std::vector<double> forcing{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> trace{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_trace{0.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        {}, 4, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 3, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 4, 0.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 4, 1.0, 1.0, 1.0, short_trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  const std::vector<double> one_line{1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        one_line, 4, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);

  const double period = 2.0 * std::acos(-1.0);
  const auto field = tensor_field(8, 8, period, [](double, double) { return -2.0; });
  const std::vector<double> flat(8, 0.0);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        field, 8, period, 0.0, 0.0, flat, 1.0, 0.0, flat),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        field, 8, period, 1.0, 0.0, flat, 0.5, 1.0, flat),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        field, 8, period, 0.0, 1.0, flat, 0.0, 1.0, flat),
                    std::invalid_argument);
}

TEST_CASE("Fourier-Chebyshev Robin Helmholtz recovers separable solutions",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto error_of = [&](int nx, int degree, double lambda, double value_plus,
                            double slope_plus, double value_minus,
                            double slope_minus, auto exact, auto force, auto data) {
    const auto forcing = tensor_field(nx, degree, period, force);
    const auto truth = tensor_field(nx, degree, period, exact);
    std::vector<double> data_plus(static_cast<std::size_t>(nx));
    std::vector<double> data_minus(static_cast<std::size_t>(nx));
    for (int ix = 0; ix < nx; ++ix) {
      const double x = period * static_cast<double>(ix) / static_cast<double>(nx);
      data_plus[static_cast<std::size_t>(ix)] = data(x, 1.0);
      data_minus[static_cast<std::size_t>(ix)] = data(x, -1.0);
    }
    const auto got = pfc::fft::fourier_chebyshev_robin_helmholtz(
        forcing, nx, period, lambda, value_plus, slope_plus, data_plus, value_minus,
        slope_minus, data_minus);
    double mean_integral = 0.0;
    for (int ix = 0; ix < nx; ++ix) {
      std::vector<double> column(static_cast<std::size_t>(degree) + 1);
      for (int iz = 0; iz <= degree; ++iz) {
        column[static_cast<std::size_t>(iz)] =
            got[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx) +
                static_cast<std::size_t>(ix)];
      }
      const auto derivative = pfc::fft::chebyshev_derivative(column);
      REQUIRE_THAT(value_plus * column.front() + slope_plus * derivative.front(),
                   WithinAbs(data_plus[static_cast<std::size_t>(ix)], 1e-9));
      REQUIRE_THAT(value_minus * column.back() + slope_minus * derivative.back(),
                   WithinAbs(data_minus[static_cast<std::size_t>(ix)], 1e-9));
      mean_integral += interval_integral(column);
    }
    if (lambda == 0.0 && value_plus == 0.0 && value_minus == 0.0) {
      REQUIRE_THAT(mean_integral / static_cast<double>(nx), WithinAbs(0.0, 1e-10));
    }
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              1, 8, 1.0, 1.0, 1.0, 1.0, -1.0,
              [](double, double z) { return 1.0 - z * z; },
              [](double, double z) { return z * z - 3.0; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(error_of(
              8, 8, 1.0, 1.0, 1.0, 1.0, -1.0,
              [](double, double z) { return 1.0 - z * z; },
              [](double, double z) { return z * z - 3.0; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(
      error_of(
          16, 8, 1.0, 1.0, 1.0, 1.0, -1.0,
          [](double x, double z) { return std::cos(2.0 * x) * (1.0 - z * z); },
          [](double x, double z) { return std::cos(2.0 * x) * (5.0 * z * z - 7.0); },
          [](double x, double) { return -2.0 * std::cos(2.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 1.0, 1.0, 1.0, 1.0, -1.0,
              [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -5.0 * z * std::cos(2.0 * x); },
              [](double x, double z) {
                return (z > 0.0 ? 2.0 : -2.0) * std::cos(2.0 * x);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 1.0, 1.0, 1.0, 1.0, -1.0,
              [](double x, double z) { return std::cos(4.0 * x) * (1.0 - z * z); },
              [](double x, double z) {
                return std::cos(4.0 * x) * (17.0 * z * z - 19.0);
              },
              [](double x, double) { return -2.0 * std::cos(4.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 1.0, 1.0, 1.0, 1.0, -1.0,
              [](double x, double z) { return 1.0 - z * z + z * std::cos(2.0 * x); },
              [](double x, double z) {
                return z * z - 3.0 - 5.0 * z * std::cos(2.0 * x);
              },
              [](double x, double z) {
                return -2.0 + (z > 0.0 ? 2.0 : -2.0) * std::cos(2.0 * x);
              }) < 1e-10);
  REQUIRE(
      error_of(
          16, 8, 1.0, 1.0, 0.0, 1.0, 0.0,
          [](double x, double z) { return std::cos(2.0 * x) * (1.0 - z * z); },
          [](double x, double z) { return std::cos(2.0 * x) * (5.0 * z * z - 7.0); },
          [](double, double) { return 0.0; }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 1.0, 0.0, 1.0, 0.0, 1.0,
              [](double, double z) { return 1.0 / 3.0 - z * z; },
              [](double, double z) { return z * z - 7.0 / 3.0; },
              [](double, double z) { return -2.0 * z; }) < 1e-11);
  REQUIRE(error_of(
              16, 8, 1.0, 1.0, 0.0, 0.0, 1.0,
              [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -5.0 * z * std::cos(2.0 * x); },
              [](double x, double) { return std::cos(2.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 0.0, 1.0, 1.0, 1.0, -1.0,
              [](double, double z) { return 1.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(error_of(
              8, 8, 0.0, 0.0, 1.0, 0.0, 1.0,
              [](double, double z) { return 1.0 / 3.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double z) { return -2.0 * z; }) < 1e-11);

  const auto forcing = tensor_field(16, 8, period, [](double x, double z) {
    return std::cos(2.0 * x) * (5.0 * z * z - 7.0);
  });
  std::vector<double> data_plus(16);
  std::vector<double> data_minus(16);
  for (int ix = 0; ix < 16; ++ix) {
    const double x = period * static_cast<double>(ix) / 16.0;
    data_plus[static_cast<std::size_t>(ix)] = -2.0 * std::cos(2.0 * x);
    data_minus[static_cast<std::size_t>(ix)] = -2.0 * std::cos(2.0 * x);
  }
  const auto got = pfc::fft::fourier_chebyshev_robin_helmholtz(
      forcing, 16, period, 1.0, 1.0, 1.0, data_plus, 1.0, -1.0, data_minus);
  auto residual = pfc::fft::fourier_chebyshev_laplacian(got, 16, period);
  for (std::size_t i = 0; i < residual.size(); ++i) {
    residual[i] -= got[i];
  }
  REQUIRE(max_abs_diff(residual, forcing) < 1e-8);
}

TEST_CASE("Fourier-Chebyshev Robin Helmholtz rejects a bad condition",
          "[fft][chebyshev]") {
  const std::vector<double> forcing{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> trace{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_trace{0.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        {}, 4, 1.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        forcing, 0, 1.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        forcing, 3, 1.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        forcing, 4, 0.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(forcing, 4, 1.0, 1.0, 1.0, 1.0,
                                                  short_trace, 1.0, -1.0, trace),
      std::invalid_argument);
  const std::vector<double> one_line{1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        one_line, 4, 1.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);

  const double period = 2.0 * std::acos(-1.0);
  const auto field = tensor_field(8, 8, period, [](double, double) { return -2.0; });
  const std::vector<double> flat(8, 0.0);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        field, 8, period, 1.0, 0.0, 0.0, flat, 1.0, 0.0, flat),
                    std::invalid_argument);
  const std::vector<double> pair(8, 0.0);
  const std::vector<double> pair_trace(4, 0.0);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(pair, 4, period, 1.0, 1.0, 0.0,
                                                  pair_trace, 0.5, 1.0, pair_trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        field, 8, period, 0.0, 0.0, 1.0, flat, 0.0, 1.0, flat),
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

TEST_CASE("Chebyshev Robin Poisson recovers polynomial solutions",
          "[fft][chebyshev]") {
  const auto error_of = [](int degree, auto force, auto exact, double value_plus,
                           double slope_plus, double data_plus, double value_minus,
                           double slope_minus, double data_minus) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = force(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got =
        pfc::fft::chebyshev_robin_poisson(forcing, value_plus, slope_plus, data_plus,
                                          value_minus, slope_minus, data_minus);
    const auto slope = pfc::fft::chebyshev_derivative(got);
    REQUIRE_THAT(value_plus * got.front() + slope_plus * slope.front(),
                 WithinAbs(data_plus, 1e-9));
    REQUIRE_THAT(value_minus * got.back() + slope_minus * slope.back(),
                 WithinAbs(data_minus, 1e-9));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              8, [](double) { return -2.0; }, [](double x) { return 1.0 - x * x; },
              1.0, 0.0, 0.0, 1.0, 0.0, 0.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return 0.0; }, [](double x) { return x; }, 2.0, 0.0,
              2.0, 2.0, 0.0, -2.0) < 1e-12);
  REQUIRE(error_of(
              8, [](double) { return -2.0; },
              [](double x) { return 1.0 / 3.0 - x * x; }, 0.0, 1.0, -2.0, 0.0, 1.0,
              2.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return 0.0; }, [](double x) { return x; }, 1.0, 0.0,
              1.0, 0.0, 1.0, 1.0) < 1e-12);
  REQUIRE(error_of(
              8, [](double) { return -2.0; }, [](double x) { return 1.0 - x * x; },
              1.0, 1.0, -2.0, 1.0, 1.0, 2.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return -2.0; },
              [](double x) { return 1.0 - x * x + 0.5 * x; }, 1.0, 1.0, -1.0, 1.0,
              1.0, 2.0) < 1e-11);

  const auto nodes = pfc::fft::chebyshev_lobatto(8);
  const std::vector<double> forcing(nodes.size(), -2.0);
  const auto gauged =
      pfc::fft::chebyshev_robin_poisson(forcing, 0.0, 1.0, -2.0, 0.0, 1.0, 2.0);
  REQUIRE_THAT(interval_integral(gauged), WithinAbs(0.0, 1e-10));
  const auto second =
      pfc::fft::chebyshev_derivative(pfc::fft::chebyshev_derivative(gauged));
  REQUIRE(max_abs_diff(second, forcing) < 1e-9);
}

TEST_CASE("Chebyshev Robin Poisson rejects an unusable condition",
          "[fft][chebyshev]") {
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_poisson({}, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      std::invalid_argument);
  const std::vector<double> one{0.0};
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_poisson(one, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      std::invalid_argument);

  const auto nodes = pfc::fft::chebyshev_lobatto(8);
  const std::vector<double> forcing(nodes.size(), -2.0);
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_poisson(forcing, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_poisson(forcing, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_poisson(forcing, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0),
      std::invalid_argument);
}

TEST_CASE("Chebyshev Robin Helmholtz recovers polynomial solutions",
          "[fft][chebyshev]") {
  const auto error_of = [](int degree, double lambda, auto force, auto exact,
                           double value_plus, double slope_plus, double data_plus,
                           double value_minus, double slope_minus,
                           double data_minus) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = force(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got = pfc::fft::chebyshev_robin_helmholtz(
        forcing, lambda, value_plus, slope_plus, data_plus, value_minus, slope_minus,
        data_minus);
    const auto slope = pfc::fft::chebyshev_derivative(got);
    REQUIRE_THAT(value_plus * got.front() + slope_plus * slope.front(),
                 WithinAbs(data_plus, 1e-9));
    REQUIRE_THAT(value_minus * got.back() + slope_minus * slope.back(),
                 WithinAbs(data_minus, 1e-9));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              8, 0.0, [](double) { return -2.0; },
              [](double x) { return 1.0 - x * x; }, 1.0, 1.0, -2.0, 1.0, 1.0,
              2.0) < 1e-11);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return x * x - 3.0; },
              [](double x) { return 1.0 - x * x; }, 1.0, 1.0, -2.0, 1.0, -1.0,
              -2.0) < 1e-11);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return x * x - 0.5 * x - 3.0; },
              [](double x) { return 1.0 - x * x + 0.5 * x; }, 1.0, 1.0, -1.0, 1.0,
              -1.0, -3.0) < 1e-11);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return -x; }, [](double x) { return x; }, 1.0,
              0.0, 1.0, 0.0, 1.0, 1.0) < 1e-12);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return -x; }, [](double x) { return x; }, 2.0,
              0.0, 2.0, 2.0, 0.0, -2.0) < 1e-12);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return x * x - 7.0 / 3.0; },
              [](double x) { return 1.0 / 3.0 - x * x; }, 0.0, 1.0, -2.0, 0.0, 1.0,
              2.0) < 1e-11);

  const auto nodes = pfc::fft::chebyshev_lobatto(8);
  const std::vector<double> forcing(nodes.size(), -2.0);
  const auto gauged = pfc::fft::chebyshev_robin_helmholtz(forcing, 0.0, 0.0, 1.0,
                                                          -2.0, 0.0, 1.0, 2.0);
  REQUIRE_THAT(interval_integral(gauged), WithinAbs(0.0, 1e-10));

  const auto residual_nodes = pfc::fft::chebyshev_lobatto(12);
  std::vector<double> residual_forcing(residual_nodes.size());
  for (std::size_t j = 0; j < residual_nodes.size(); ++j) {
    residual_forcing[j] = residual_nodes[j] * residual_nodes[j] - 3.0;
  }
  const auto solution = pfc::fft::chebyshev_robin_helmholtz(
      residual_forcing, 1.0, 1.0, 1.0, -2.0, 1.0, -1.0, -2.0);
  const auto second =
      pfc::fft::chebyshev_derivative(pfc::fft::chebyshev_derivative(solution));
  std::vector<double> residual(solution.size());
  for (std::size_t j = 0; j < solution.size(); ++j) {
    residual[j] = second[j] - solution[j];
  }
  REQUIRE(max_abs_diff(residual, residual_forcing) < 1e-9);
}

TEST_CASE("Chebyshev Robin Helmholtz converges for a smooth solution",
          "[fft][chebyshev]") {
  const auto error_at = [](int degree) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    const std::vector<double> forcing(nodes.size(), 0.0);
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      truth[j] = std::exp(nodes[j]);
    }
    const auto got = pfc::fft::chebyshev_robin_helmholtz(
        forcing, 1.0, 1.0, 1.0, 2.0 * std::exp(1.0), 1.0, -1.0, 0.0);
    return max_abs_diff(got, truth);
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-10);
  REQUIRE(fine < coarse);
}

TEST_CASE("Chebyshev Robin Helmholtz rejects an unusable condition",
          "[fft][chebyshev]") {
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_helmholtz({}, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      std::invalid_argument);
  const std::vector<double> one{0.0};
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_helmholtz(one, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      std::invalid_argument);

  const auto nodes = pfc::fft::chebyshev_lobatto(8);
  const std::vector<double> forcing(nodes.size(), -2.0);
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_robin_helmholtz(forcing, 1.0, 0.0, 0.0, 0.0,
                                                        1.0, 0.0, 0.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_robin_helmholtz(forcing, 0.0, 0.0, 1.0, 0.0,
                                                        0.0, 1.0, 0.0),
                    std::invalid_argument);
  const std::vector<double> pair{0.0, 0.0};
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_helmholtz(pair, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0),
      std::invalid_argument);
}

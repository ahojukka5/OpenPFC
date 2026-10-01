// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file chebyshev.cpp
 * @brief Even-extension FFT for a Chebyshev–Lobatto series.
 *
 * A pure mode T_k, 0 < k < n, has forward bins U_k = U_{2n-k} = n.
 * The endpoint bins use weight 2, so U_0 = 2n a_0 and U_n = 2n a_n.
 * The interior derivative is the theta derivative divided by -sin(theta).
 * Endpoints use the same weighted sum as Trefethen's chebfft.
 * Dirichlet Poisson integrates the coefficients twice. The two
 * constants are fixed by the endpoint values. Neumann Poisson uses
 * the same integrals, rejects a slope jump the forcing cannot
 * support, and fixes the constant so the integral is zero.
 * Dirichlet Helmholtz uses a tau recurrence for u'' - lambda u = f.
 * The two highest coefficients match the endpoint values. A zero
 * lambda reuses the Poisson integral. Neumann Helmholtz matches both
 * endpoint slopes. A zero lambda reuses the Neumann Poisson solve.
 * Robin Poisson mixes the value and the slope at each end. Pure
 * value data and pure slope data reuse those solves. Robin
 * Helmholtz uses that mix for u'' - lambda u = f. A zero lambda
 * reuses the Robin Poisson integral. A Fourier × Chebyshev
 * Dirichlet Poisson transforms the periodic direction and solves
 * u'' - k^2 u = f on each mode. Neumann slopes use the same
 * reduction. A zero wavenumber keeps the integral gauge. Robin
 * data uses that reduction with weights fixed along x. Helmholtz
 * adds λ to k². A zero λ reuses the Robin Poisson solve.
 * A second periodic direction adds k_y² on the same rank.
 */

#include <openpfc/kernel/fft/chebyshev.hpp>

#include <cmath>
#include <complex>
#include <fftw3.h>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pfc::fft {
namespace {

struct Plan {
  int length{0};
  fftw_complex *in{nullptr};
  fftw_complex *out{nullptr};
  fftw_plan forward{};
  fftw_plan backward{};

  explicit Plan(int n) : length(n) {
    if (n < 2) {
      throw std::invalid_argument("chebyshev: FFT length must be at least 2");
    }
    const auto bytes = sizeof(fftw_complex) * static_cast<std::size_t>(n);
    in = static_cast<fftw_complex *>(fftw_malloc(bytes));
    out = static_cast<fftw_complex *>(fftw_malloc(bytes));
    if (in == nullptr || out == nullptr) {
      fftw_free(in);
      fftw_free(out);
      throw std::bad_alloc();
    }
    forward = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    backward = fftw_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (forward == nullptr || backward == nullptr) {
      fftw_destroy_plan(forward);
      fftw_destroy_plan(backward);
      fftw_free(in);
      fftw_free(out);
      throw std::runtime_error("chebyshev: FFTW plan failed");
    }
  }

  Plan(const Plan &) = delete;
  Plan &operator=(const Plan &) = delete;

  ~Plan() {
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(in);
    fftw_free(out);
  }

  void load(const std::vector<std::complex<double>> &samples) {
    for (int i = 0; i < length; ++i) {
      in[i][0] = samples[static_cast<std::size_t>(i)].real();
      in[i][1] = samples[static_cast<std::size_t>(i)].imag();
    }
  }

  [[nodiscard]] std::vector<std::complex<double>> read() const {
    std::vector<std::complex<double>> samples(static_cast<std::size_t>(length));
    for (int i = 0; i < length; ++i) {
      samples[static_cast<std::size_t>(i)] = {out[i][0], out[i][1]};
    }
    return samples;
  }
};

[[nodiscard]] int degree_of(std::span<const double> samples, const char *what) {
  if (samples.empty()) {
    throw std::invalid_argument(what);
  }
  if (samples.size() > static_cast<std::size_t>(std::vector<double>{}.max_size() / 2)) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  return static_cast<int>(samples.size()) - 1;
}

[[nodiscard]] std::vector<std::complex<double>>
even_extension(std::span<const double> values) {
  const int n = degree_of(values, "chebyshev: values are empty");
  const int m = 2 * n;
  std::vector<std::complex<double>> extended(static_cast<std::size_t>(m));
  extended[0] = values[0];
  for (int j = 1; j < n; ++j) {
    extended[static_cast<std::size_t>(j)] = values[static_cast<std::size_t>(j)];
    extended[static_cast<std::size_t>(m - j)] = values[static_cast<std::size_t>(j)];
  }
  extended[static_cast<std::size_t>(n)] = values[static_cast<std::size_t>(n)];
  return extended;
}

[[nodiscard]] std::vector<std::complex<double>>
forward_bins(std::span<const double> values) {
  auto extended = even_extension(values);
  Plan plan(static_cast<int>(extended.size()));
  plan.load(extended);
  fftw_execute(plan.forward);
  return plan.read();
}

[[nodiscard]] double endpoint_weight(int k, int n) {
  return (k == 0 || k == n) ? 2.0 : 1.0;
}

[[nodiscard]] std::size_t mixed_index(int iz, int ix, int nx) {
  return static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx) +
         static_cast<std::size_t>(ix);
}

[[nodiscard]] std::size_t tensor_index(int iz, int iy, int ix, int ny, int nx) {
  return (static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny) +
          static_cast<std::size_t>(iy)) *
             static_cast<std::size_t>(nx) +
         static_cast<std::size_t>(ix);
}

[[nodiscard]] double periodic_wavenumber(int mode, int count, double period) {
  int alias = mode;
  if (mode > count / 2) alias = mode - count;
  const double pi = std::acos(-1.0);
  return (2.0 * pi * static_cast<double>(alias)) / period;
}

[[nodiscard]] std::vector<double>
second_derivative(std::span<const double> values) {
  return chebyshev_derivative(chebyshev_derivative(values));
}

/// Indefinite integral with the constant of integration set to zero.
/// a_k = (c_{k-1} b_{k-1} - b_{k+1}) / (2k), c_0 = 2.
[[nodiscard]] std::vector<double>
integrate_coefficients(std::span<const double> derivative) {
  const int n = static_cast<int>(derivative.size()) - 1;
  std::vector<double> integral(static_cast<std::size_t>(n + 2), 0.0);
  for (int k = 1; k <= n + 1; ++k) {
    const double weight = (k == 1) ? 2.0 : 1.0;
    const double previous = derivative[static_cast<std::size_t>(k - 1)];
    const double next =
        (k + 1 <= n) ? derivative[static_cast<std::size_t>(k + 1)] : 0.0;
    integral[static_cast<std::size_t>(k)] =
        (weight * previous - next) / (2.0 * static_cast<double>(k));
  }
  return integral;
}

[[nodiscard]] double clenshaw(std::span<const double> coefficients, double x) {
  double ahead = 0.0;
  double current = 0.0;
  for (int k = static_cast<int>(coefficients.size()) - 1; k >= 1; --k) {
    const double updated = 2.0 * x * current - ahead +
                           coefficients[static_cast<std::size_t>(k)];
    ahead = current;
    current = updated;
  }
  const double constant = coefficients.empty() ? 0.0 : coefficients.front();
  return constant + x * current - ahead;
}

/// Coefficient of T_k in u'', for u = sum a_j T_j.
/// Only modes j >= k + 2 of the same parity contribute.
[[nodiscard]] double
second_derivative_coefficient(std::span<const double> coefficients, int mode) {
  const double weight = (mode == 0) ? 2.0 : 1.0;
  double sum = 0.0;
  const double wave = static_cast<double>(mode);
  for (int j = mode + 2; j < static_cast<int>(coefficients.size()); j += 2) {
    const double higher = static_cast<double>(j);
    sum += higher * (higher * higher - wave * wave) *
           coefficients[static_cast<std::size_t>(j)];
  }
  return sum / weight;
}

/// Coefficients of a polynomial solution with the two highest modes set.
[[nodiscard]] std::vector<double> helmholtz_series(std::span<const double> forcing,
                                                   double lambda, double highest,
                                                   double next) {
  const int degree = static_cast<int>(forcing.size()) - 1;
  std::vector<double> coefficients(forcing.size(), 0.0);
  coefficients[static_cast<std::size_t>(degree)] = highest;
  if (degree >= 1) {
    coefficients[static_cast<std::size_t>(degree - 1)] = next;
  }
  for (int mode = degree - 2; mode >= 0; --mode) {
    const double second = second_derivative_coefficient(coefficients, mode);
    coefficients[static_cast<std::size_t>(mode)] =
        (second - forcing[static_cast<std::size_t>(mode)]) / lambda;
  }
  return coefficients;
}

struct EndpointSum {
  double at_plus{0.0};
  double at_minus{0.0};
};

[[nodiscard]] EndpointSum endpoint_sum(std::span<const double> coefficients) {
  EndpointSum sum;
  for (int mode = 0; mode < static_cast<int>(coefficients.size()); ++mode) {
    const double term = coefficients[static_cast<std::size_t>(mode)];
    sum.at_plus += term;
    sum.at_minus += (mode % 2 == 0) ? term : -term;
  }
  return sum;
}

/// T_k'(1) = k^2 and T_k'(-1) = (-1)^{k-1} k^2.
[[nodiscard]] EndpointSum slope_sum(std::span<const double> coefficients) {
  EndpointSum sum;
  for (int mode = 1; mode < static_cast<int>(coefficients.size()); ++mode) {
    const double weight = static_cast<double>(mode) * static_cast<double>(mode) *
                          coefficients[static_cast<std::size_t>(mode)];
    sum.at_plus += weight;
    sum.at_minus += (mode % 2 == 1) ? weight : -weight;
  }
  return sum;
}

enum class HelmholtzMatch { values, slopes };

[[nodiscard]] EndpointSum matched_ends(std::span<const double> coefficients,
                                       HelmholtzMatch match) {
  if (match == HelmholtzMatch::values) return endpoint_sum(coefficients);
  return slope_sum(coefficients);
}

[[nodiscard]] bool coefficients_finite(std::span<const double> coefficients) {
  for (double value : coefficients) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

[[nodiscard]] double column_scale(double at_plus, double at_minus) {
  double scale = 1.0;
  const double plus = std::abs(at_plus);
  const double minus = std::abs(at_minus);
  if (plus > scale) scale = plus;
  if (minus > scale) scale = minus;
  return scale;
}

[[nodiscard]] std::vector<double>
evaluate_series(std::span<const double> coefficients, int degree) {
  const auto nodes = chebyshev_lobatto(degree);
  std::vector<double> values(nodes.size());
  for (std::size_t j = 0; j < nodes.size(); ++j) {
    values[j] = clenshaw(coefficients, nodes[j]);
  }
  return values;
}

[[nodiscard]] std::vector<double>
solve_matched_helmholtz(std::span<const double> forcing, double lambda,
                        double at_plus, double at_minus, HelmholtzMatch match,
                        const char *singular) {
  const int degree = static_cast<int>(forcing.size()) - 1;
  const auto coefficients = chebyshev_coefficients(forcing);
  const auto particular = helmholtz_series(coefficients, lambda, 0.0, 0.0);
  const std::vector<double> zero(coefficients.size(), 0.0);
  const auto along_highest = helmholtz_series(zero, lambda, 1.0, 0.0);
  const auto along_next = helmholtz_series(zero, lambda, 0.0, 1.0);
  if (!coefficients_finite(particular) || !coefficients_finite(along_highest) ||
      !coefficients_finite(along_next)) {
    throw std::invalid_argument("chebyshev: Helmholtz degree is too large");
  }

  const auto particular_ends = matched_ends(particular, match);
  const auto highest_ends = matched_ends(along_highest, match);
  const auto next_ends = matched_ends(along_next, match);
  const double highest_scale =
      column_scale(highest_ends.at_plus, highest_ends.at_minus);
  const double next_scale = column_scale(next_ends.at_plus, next_ends.at_minus);
  const double highest_plus = highest_ends.at_plus / highest_scale;
  const double highest_minus = highest_ends.at_minus / highest_scale;
  const double next_plus = next_ends.at_plus / next_scale;
  const double next_minus = next_ends.at_minus / next_scale;
  const double determinant = highest_plus * next_minus - next_plus * highest_minus;
  if (!(std::abs(determinant) > 1e-8)) {
    throw std::invalid_argument(singular);
  }
  const double rhs_plus = at_plus - particular_ends.at_plus;
  const double rhs_minus = at_minus - particular_ends.at_minus;
  const double highest =
      (rhs_plus * next_minus - next_plus * rhs_minus) / determinant / highest_scale;
  const double next = (highest_plus * rhs_minus - highest_minus * rhs_plus) /
                      determinant / next_scale;

  std::vector<double> solution(coefficients.size());
  for (std::size_t mode = 0; mode < solution.size(); ++mode) {
    solution[mode] =
        particular[mode] + highest * along_highest[mode] + next * along_next[mode];
  }
  if (!coefficients_finite(solution)) {
    throw std::invalid_argument("chebyshev: Helmholtz degree is too large");
  }
  return evaluate_series(solution, degree);
}

struct RobinWeights {
  double value_plus{0.0};
  double slope_plus{0.0};
  double value_minus{0.0};
  double slope_minus{0.0};
};

[[nodiscard]] EndpointSum robin_boundary(std::span<const double> coefficients,
                                         RobinWeights weights) {
  const auto values = endpoint_sum(coefficients);
  const auto slopes = slope_sum(coefficients);
  return EndpointSum{
      weights.value_plus * values.at_plus + weights.slope_plus * slopes.at_plus,
      weights.value_minus * values.at_minus + weights.slope_minus * slopes.at_minus};
}

[[nodiscard]] std::vector<double>
solve_robin_helmholtz(std::span<const double> forcing, double lambda,
                      RobinWeights weights, double data_plus, double data_minus) {
  const int degree = static_cast<int>(forcing.size()) - 1;
  const auto coefficients = chebyshev_coefficients(forcing);
  const auto particular = helmholtz_series(coefficients, lambda, 0.0, 0.0);
  const std::vector<double> zero(coefficients.size(), 0.0);
  const auto along_highest = helmholtz_series(zero, lambda, 1.0, 0.0);
  const auto along_next = helmholtz_series(zero, lambda, 0.0, 1.0);
  if (!coefficients_finite(particular) || !coefficients_finite(along_highest) ||
      !coefficients_finite(along_next)) {
    throw std::invalid_argument("chebyshev: Helmholtz degree is too large");
  }

  const auto particular_ends = robin_boundary(particular, weights);
  const auto highest_ends = robin_boundary(along_highest, weights);
  const auto next_ends = robin_boundary(along_next, weights);
  const double highest_scale =
      column_scale(highest_ends.at_plus, highest_ends.at_minus);
  const double next_scale = column_scale(next_ends.at_plus, next_ends.at_minus);
  const double highest_plus = highest_ends.at_plus / highest_scale;
  const double highest_minus = highest_ends.at_minus / highest_scale;
  const double next_plus = next_ends.at_plus / next_scale;
  const double next_minus = next_ends.at_minus / next_scale;
  const double determinant = highest_plus * next_minus - next_plus * highest_minus;
  if (!(std::abs(determinant) > 1e-8)) {
    throw std::invalid_argument("chebyshev: Helmholtz Robin problem is singular");
  }
  const double rhs_plus = data_plus - particular_ends.at_plus;
  const double rhs_minus = data_minus - particular_ends.at_minus;
  const double highest =
      (rhs_plus * next_minus - next_plus * rhs_minus) / determinant / highest_scale;
  const double next = (highest_plus * rhs_minus - highest_minus * rhs_plus) /
                      determinant / next_scale;

  std::vector<double> solution(coefficients.size());
  for (std::size_t mode = 0; mode < solution.size(); ++mode) {
    solution[mode] =
        particular[mode] + highest * along_highest[mode] + next * along_next[mode];
  }
  if (!coefficients_finite(solution)) {
    throw std::invalid_argument("chebyshev: Helmholtz degree is too large");
  }
  return evaluate_series(solution, degree);
}

} // namespace

std::vector<double> chebyshev_lobatto(int degree) {
  if (degree < 0) {
    throw std::invalid_argument("chebyshev: degree must be non-negative");
  }
  std::vector<double> nodes(static_cast<std::size_t>(degree) + 1);
  if (degree == 0) {
    nodes[0] = 1.0;
    return nodes;
  }
  const double step = std::acos(-1.0) / static_cast<double>(degree);
  for (int j = 0; j <= degree; ++j) {
    nodes[static_cast<std::size_t>(j)] = std::cos(step * static_cast<double>(j));
  }
  return nodes;
}

std::vector<double> chebyshev_coefficients(std::span<const double> values) {
  const int n = degree_of(values, "chebyshev: values are empty");
  std::vector<double> coefficients(values.size());
  if (n == 0) {
    coefficients[0] = values[0];
    return coefficients;
  }
  const auto bins = forward_bins(values);
  for (int k = 0; k <= n; ++k) {
    coefficients[static_cast<std::size_t>(k)] =
        bins[static_cast<std::size_t>(k)].real() /
        (static_cast<double>(n) * endpoint_weight(k, n));
  }
  return coefficients;
}

std::vector<double> chebyshev_values(std::span<const double> coefficients) {
  const int n = degree_of(coefficients, "chebyshev: coefficients are empty");
  std::vector<double> values(coefficients.size());
  if (n == 0) {
    values[0] = coefficients[0];
    return values;
  }
  const int m = 2 * n;
  std::vector<std::complex<double>> bins(static_cast<std::size_t>(m));
  const double scale = static_cast<double>(n);
  for (int k = 0; k <= n; ++k) {
    bins[static_cast<std::size_t>(k)] =
        scale * endpoint_weight(k, n) * coefficients[static_cast<std::size_t>(k)];
  }
  for (int k = 1; k < n; ++k) {
    bins[static_cast<std::size_t>(m - k)] = bins[static_cast<std::size_t>(k)];
  }
  Plan plan(m);
  plan.load(bins);
  fftw_execute(plan.backward);
  const auto samples = plan.read();
  for (int j = 0; j <= n; ++j) {
    values[static_cast<std::size_t>(j)] =
        samples[static_cast<std::size_t>(j)].real() / static_cast<double>(m);
  }
  return values;
}

std::vector<double> chebyshev_derivative(std::span<const double> values) {
  const int n = degree_of(values, "chebyshev: values are empty");
  std::vector<double> derivative(values.size(), 0.0);
  if (n == 0) return derivative;

  const auto bins = forward_bins(values);
  const int m = 2 * n;
  std::vector<std::complex<double>> spectrum(static_cast<std::size_t>(m));
  for (int k = 0; k < m; ++k) {
    int wave = 0;
    if (k < n) wave = k;
    else if (k > n) wave = k - m;
    const auto bin = bins[static_cast<std::size_t>(k)];
    spectrum[static_cast<std::size_t>(k)] = std::complex<double>(0.0, 1.0) *
                                            static_cast<double>(wave) * bin;
  }
  Plan plan(m);
  plan.load(spectrum);
  fftw_execute(plan.backward);
  const auto theta = plan.read();
  const auto nodes = chebyshev_lobatto(n);
  for (int j = 1; j < n; ++j) {
    const double x = nodes[static_cast<std::size_t>(j)];
    const double sine = std::sqrt(1.0 - x * x);
    derivative[static_cast<std::size_t>(j)] =
        -theta[static_cast<std::size_t>(j)].real() /
        (static_cast<double>(m) * sine);
  }

  double left = 0.0;
  double right = 0.0;
  for (int k = 0; k < n; ++k) {
    const double term = static_cast<double>(k) * static_cast<double>(k) *
                        bins[static_cast<std::size_t>(k)].real();
    const double sign = (k % 2 == 0) ? -1.0 : 1.0;
    left += term;
    right += sign * term;
  }
  const double nyquist = bins[static_cast<std::size_t>(n)].real();
  const double inv = 1.0 / static_cast<double>(n);
  const double end_sign = (n % 2 == 0) ? -1.0 : 1.0;
  derivative[0] = left * inv + 0.5 * static_cast<double>(n) * nyquist;
  derivative[static_cast<std::size_t>(n)] =
      right * inv + 0.5 * end_sign * static_cast<double>(n) * nyquist;
  return derivative;
}

std::vector<double>
fourier_chebyshev_laplacian(std::span<const double> values, int nx,
                            double period) {
  if (nx < 1) {
    throw std::invalid_argument(
        "chebyshev: periodic count must be positive");
  }
  if (!(period > 0.0)) {
    throw std::invalid_argument("chebyshev: period must be positive");
  }
  if (values.empty() ||
      values.size() % static_cast<std::size_t>(nx) != 0) {
    throw std::invalid_argument(
        "chebyshev: values do not match the periodic count");
  }
  const auto lines = values.size() / static_cast<std::size_t>(nx);
  if (lines > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  const int nline = static_cast<int>(lines);
  std::vector<std::complex<double>> modes(values.size());

  if (nx == 1) {
    for (int iz = 0; iz < nline; ++iz) {
      modes[static_cast<std::size_t>(iz)] = values[static_cast<std::size_t>(iz)];
    }
  } else {
    Plan plan(nx);
    std::vector<std::complex<double>> line(static_cast<std::size_t>(nx));
    for (int iz = 0; iz < nline; ++iz) {
      for (int ix = 0; ix < nx; ++ix) {
        line[static_cast<std::size_t>(ix)] = values[mixed_index(iz, ix, nx)];
      }
      plan.load(line);
      fftw_execute(plan.forward);
      const auto bins = plan.read();
      for (int mode = 0; mode < nx; ++mode) {
        modes[mixed_index(iz, mode, nx)] = bins[static_cast<std::size_t>(mode)];
      }
    }
  }

  for (int mode = 0; mode < nx; ++mode) {
    const double wavenumber = periodic_wavenumber(mode, nx, period);
    const double square = wavenumber * wavenumber;
    std::vector<double> real(static_cast<std::size_t>(nline));
    std::vector<double> imag(static_cast<std::size_t>(nline));
    for (int iz = 0; iz < nline; ++iz) {
      const auto coefficient = modes[mixed_index(iz, mode, nx)];
      real[static_cast<std::size_t>(iz)] = coefficient.real();
      imag[static_cast<std::size_t>(iz)] = coefficient.imag();
    }
    const auto dreal = second_derivative(real);
    const auto dimag = second_derivative(imag);
    for (int iz = 0; iz < nline; ++iz) {
      const auto coefficient = modes[mixed_index(iz, mode, nx)];
      modes[mixed_index(iz, mode, nx)] =
          std::complex<double>(dreal[static_cast<std::size_t>(iz)],
                               dimag[static_cast<std::size_t>(iz)]) -
          square * coefficient;
    }
  }

  std::vector<double> laplacian(values.size());
  if (nx == 1) {
    for (int iz = 0; iz < nline; ++iz) {
      laplacian[static_cast<std::size_t>(iz)] =
          modes[static_cast<std::size_t>(iz)].real();
    }
    return laplacian;
  }

  Plan plan(nx);
  std::vector<std::complex<double>> line(static_cast<std::size_t>(nx));
  const double scale = static_cast<double>(nx);
  for (int iz = 0; iz < nline; ++iz) {
    for (int mode = 0; mode < nx; ++mode) {
      line[static_cast<std::size_t>(mode)] = modes[mixed_index(iz, mode, nx)];
    }
    plan.load(line);
    fftw_execute(plan.backward);
    const auto samples = plan.read();
    for (int ix = 0; ix < nx; ++ix) {
      laplacian[mixed_index(iz, ix, nx)] =
          samples[static_cast<std::size_t>(ix)].real() / scale;
    }
  }
  return laplacian;
}

std::vector<double> fourier_chebyshev_laplacian(std::span<const double> values,
                                                int nx, int ny, double period_x,
                                                double period_y) {
  if (nx < 1 || ny < 1) {
    throw std::invalid_argument("chebyshev: periodic count must be positive");
  }
  if (!(period_x > 0.0) || !(period_y > 0.0)) {
    throw std::invalid_argument("chebyshev: period must be positive");
  }
  const auto plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  if (values.empty() || values.size() % plane != 0) {
    throw std::invalid_argument("chebyshev: values do not match the periodic count");
  }
  const auto lines = values.size() / plane;
  if (lines > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  if (ny == 1) {
    return fourier_chebyshev_laplacian(values, nx, period_x);
  }
  if (nx == 1) {
    return fourier_chebyshev_laplacian(values, ny, period_y);
  }

  const int nline = static_cast<int>(lines);
  std::vector<std::complex<double>> modes(values.size());
  Plan plan_x(nx);
  std::vector<std::complex<double>> line_x(static_cast<std::size_t>(nx));
  for (int iz = 0; iz < nline; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        line_x[static_cast<std::size_t>(ix)] =
            values[tensor_index(iz, iy, ix, ny, nx)];
      }
      plan_x.load(line_x);
      fftw_execute(plan_x.forward);
      const auto bins = plan_x.read();
      for (int mode = 0; mode < nx; ++mode) {
        modes[tensor_index(iz, iy, mode, ny, nx)] =
            bins[static_cast<std::size_t>(mode)];
      }
    }
  }

  Plan plan_y(ny);
  std::vector<std::complex<double>> line_y(static_cast<std::size_t>(ny));
  for (int iz = 0; iz < nline; ++iz) {
    for (int mx = 0; mx < nx; ++mx) {
      for (int iy = 0; iy < ny; ++iy) {
        line_y[static_cast<std::size_t>(iy)] =
            modes[tensor_index(iz, iy, mx, ny, nx)];
      }
      plan_y.load(line_y);
      fftw_execute(plan_y.forward);
      const auto bins = plan_y.read();
      for (int my = 0; my < ny; ++my) {
        modes[tensor_index(iz, my, mx, ny, nx)] = bins[static_cast<std::size_t>(my)];
      }
    }
  }

  for (int mx = 0; mx < nx; ++mx) {
    const double kx = periodic_wavenumber(mx, nx, period_x);
    for (int my = 0; my < ny; ++my) {
      const double ky = periodic_wavenumber(my, ny, period_y);
      const double square = kx * kx + ky * ky;
      std::vector<double> real(static_cast<std::size_t>(nline));
      std::vector<double> imag(static_cast<std::size_t>(nline));
      for (int iz = 0; iz < nline; ++iz) {
        const auto coefficient = modes[tensor_index(iz, my, mx, ny, nx)];
        real[static_cast<std::size_t>(iz)] = coefficient.real();
        imag[static_cast<std::size_t>(iz)] = coefficient.imag();
      }
      const auto dreal = second_derivative(real);
      const auto dimag = second_derivative(imag);
      for (int iz = 0; iz < nline; ++iz) {
        const auto coefficient = modes[tensor_index(iz, my, mx, ny, nx)];
        modes[tensor_index(iz, my, mx, ny, nx)] =
            std::complex<double>(dreal[static_cast<std::size_t>(iz)],
                                 dimag[static_cast<std::size_t>(iz)]) -
            square * coefficient;
      }
    }
  }

  for (int iz = 0; iz < nline; ++iz) {
    for (int mx = 0; mx < nx; ++mx) {
      for (int my = 0; my < ny; ++my) {
        line_y[static_cast<std::size_t>(my)] =
            modes[tensor_index(iz, my, mx, ny, nx)];
      }
      plan_y.load(line_y);
      fftw_execute(plan_y.backward);
      const auto samples = plan_y.read();
      for (int iy = 0; iy < ny; ++iy) {
        modes[tensor_index(iz, iy, mx, ny, nx)] =
            samples[static_cast<std::size_t>(iy)];
      }
    }
  }

  std::vector<double> laplacian(values.size());
  const double scale = static_cast<double>(nx) * static_cast<double>(ny);
  for (int iz = 0; iz < nline; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int mx = 0; mx < nx; ++mx) {
        line_x[static_cast<std::size_t>(mx)] =
            modes[tensor_index(iz, iy, mx, ny, nx)];
      }
      plan_x.load(line_x);
      fftw_execute(plan_x.backward);
      const auto samples = plan_x.read();
      for (int ix = 0; ix < nx; ++ix) {
        laplacian[tensor_index(iz, iy, ix, ny, nx)] =
            samples[static_cast<std::size_t>(ix)].real() / scale;
      }
    }
  }
  return laplacian;
}

std::vector<double> fourier_chebyshev_dirichlet_poisson(
    std::span<const double> forcing, int nx, double period,
    std::span<const double> value_at_plus, std::span<const double> value_at_minus) {
  if (nx < 1) {
    throw std::invalid_argument("chebyshev: periodic count must be positive");
  }
  if (!(period > 0.0)) {
    throw std::invalid_argument("chebyshev: period must be positive");
  }
  if (forcing.empty() || forcing.size() % static_cast<std::size_t>(nx) != 0) {
    throw std::invalid_argument(
        "chebyshev: forcing does not match the periodic count");
  }
  if (value_at_plus.size() != static_cast<std::size_t>(nx) ||
      value_at_minus.size() != static_cast<std::size_t>(nx)) {
    throw std::invalid_argument(
        "chebyshev: boundary trace does not match the periodic count");
  }
  const auto lines = forcing.size() / static_cast<std::size_t>(nx);
  if (lines > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  const int nline = static_cast<int>(lines);
  if (nline < 2) {
    throw std::invalid_argument("chebyshev: Dirichlet Poisson needs both endpoints");
  }

  std::vector<std::complex<double>> modes(forcing.size());
  std::vector<std::complex<double>> plus_hat(static_cast<std::size_t>(nx));
  std::vector<std::complex<double>> minus_hat(static_cast<std::size_t>(nx));
  if (nx == 1) {
    for (int iz = 0; iz < nline; ++iz) {
      modes[static_cast<std::size_t>(iz)] = forcing[static_cast<std::size_t>(iz)];
    }
    plus_hat[0] = value_at_plus[0];
    minus_hat[0] = value_at_minus[0];
  } else {
    Plan plan(nx);
    std::vector<std::complex<double>> line(static_cast<std::size_t>(nx));
    for (int iz = 0; iz < nline; ++iz) {
      for (int ix = 0; ix < nx; ++ix) {
        line[static_cast<std::size_t>(ix)] = forcing[mixed_index(iz, ix, nx)];
      }
      plan.load(line);
      fftw_execute(plan.forward);
      const auto bins = plan.read();
      for (int mode = 0; mode < nx; ++mode) {
        modes[mixed_index(iz, mode, nx)] = bins[static_cast<std::size_t>(mode)];
      }
    }
    for (int ix = 0; ix < nx; ++ix) {
      line[static_cast<std::size_t>(ix)] =
          value_at_plus[static_cast<std::size_t>(ix)];
    }
    plan.load(line);
    fftw_execute(plan.forward);
    plus_hat = plan.read();
    for (int ix = 0; ix < nx; ++ix) {
      line[static_cast<std::size_t>(ix)] =
          value_at_minus[static_cast<std::size_t>(ix)];
    }
    plan.load(line);
    fftw_execute(plan.forward);
    minus_hat = plan.read();
  }

  for (int mode = 0; mode < nx; ++mode) {
    const double wavenumber = periodic_wavenumber(mode, nx, period);
    const double lambda = wavenumber * wavenumber;
    std::vector<double> real(static_cast<std::size_t>(nline));
    std::vector<double> imag(static_cast<std::size_t>(nline));
    for (int iz = 0; iz < nline; ++iz) {
      const auto coefficient = modes[mixed_index(iz, mode, nx)];
      real[static_cast<std::size_t>(iz)] = coefficient.real();
      imag[static_cast<std::size_t>(iz)] = coefficient.imag();
    }
    const auto real_solution = chebyshev_dirichlet_helmholtz(
        real, lambda, plus_hat[static_cast<std::size_t>(mode)].real(),
        minus_hat[static_cast<std::size_t>(mode)].real());
    const auto imag_solution = chebyshev_dirichlet_helmholtz(
        imag, lambda, plus_hat[static_cast<std::size_t>(mode)].imag(),
        minus_hat[static_cast<std::size_t>(mode)].imag());
    for (int iz = 0; iz < nline; ++iz) {
      modes[mixed_index(iz, mode, nx)] =
          std::complex<double>(real_solution[static_cast<std::size_t>(iz)],
                               imag_solution[static_cast<std::size_t>(iz)]);
    }
  }

  std::vector<double> solution(forcing.size());
  if (nx == 1) {
    for (int iz = 0; iz < nline; ++iz) {
      solution[static_cast<std::size_t>(iz)] =
          modes[static_cast<std::size_t>(iz)].real();
    }
    return solution;
  }

  Plan plan(nx);
  std::vector<std::complex<double>> line(static_cast<std::size_t>(nx));
  const double scale = static_cast<double>(nx);
  for (int iz = 0; iz < nline; ++iz) {
    for (int mode = 0; mode < nx; ++mode) {
      line[static_cast<std::size_t>(mode)] = modes[mixed_index(iz, mode, nx)];
    }
    plan.load(line);
    fftw_execute(plan.backward);
    const auto samples = plan.read();
    for (int ix = 0; ix < nx; ++ix) {
      solution[mixed_index(iz, ix, nx)] =
          samples[static_cast<std::size_t>(ix)].real() / scale;
    }
  }
  return solution;
}

std::vector<double> fourier_chebyshev_neumann_poisson(
    std::span<const double> forcing, int nx, double period,
    std::span<const double> slope_at_plus, std::span<const double> slope_at_minus) {
  if (nx < 1) {
    throw std::invalid_argument("chebyshev: periodic count must be positive");
  }
  if (!(period > 0.0)) {
    throw std::invalid_argument("chebyshev: period must be positive");
  }
  if (forcing.empty() || forcing.size() % static_cast<std::size_t>(nx) != 0) {
    throw std::invalid_argument(
        "chebyshev: forcing does not match the periodic count");
  }
  if (slope_at_plus.size() != static_cast<std::size_t>(nx) ||
      slope_at_minus.size() != static_cast<std::size_t>(nx)) {
    throw std::invalid_argument(
        "chebyshev: boundary trace does not match the periodic count");
  }
  const auto lines = forcing.size() / static_cast<std::size_t>(nx);
  if (lines > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  const int nline = static_cast<int>(lines);
  if (nline < 2) {
    throw std::invalid_argument("chebyshev: Neumann Poisson needs both endpoints");
  }

  std::vector<std::complex<double>> modes(forcing.size());
  std::vector<std::complex<double>> plus_hat(static_cast<std::size_t>(nx));
  std::vector<std::complex<double>> minus_hat(static_cast<std::size_t>(nx));
  if (nx == 1) {
    for (int iz = 0; iz < nline; ++iz) {
      modes[static_cast<std::size_t>(iz)] = forcing[static_cast<std::size_t>(iz)];
    }
    plus_hat[0] = slope_at_plus[0];
    minus_hat[0] = slope_at_minus[0];
  } else {
    Plan plan(nx);
    std::vector<std::complex<double>> line(static_cast<std::size_t>(nx));
    for (int iz = 0; iz < nline; ++iz) {
      for (int ix = 0; ix < nx; ++ix) {
        line[static_cast<std::size_t>(ix)] = forcing[mixed_index(iz, ix, nx)];
      }
      plan.load(line);
      fftw_execute(plan.forward);
      const auto bins = plan.read();
      for (int mode = 0; mode < nx; ++mode) {
        modes[mixed_index(iz, mode, nx)] = bins[static_cast<std::size_t>(mode)];
      }
    }
    for (int ix = 0; ix < nx; ++ix) {
      line[static_cast<std::size_t>(ix)] =
          slope_at_plus[static_cast<std::size_t>(ix)];
    }
    plan.load(line);
    fftw_execute(plan.forward);
    plus_hat = plan.read();
    for (int ix = 0; ix < nx; ++ix) {
      line[static_cast<std::size_t>(ix)] =
          slope_at_minus[static_cast<std::size_t>(ix)];
    }
    plan.load(line);
    fftw_execute(plan.forward);
    minus_hat = plan.read();
  }

  for (int mode = 0; mode < nx; ++mode) {
    const double wavenumber = periodic_wavenumber(mode, nx, period);
    const double lambda = wavenumber * wavenumber;
    std::vector<double> real(static_cast<std::size_t>(nline));
    std::vector<double> imag(static_cast<std::size_t>(nline));
    for (int iz = 0; iz < nline; ++iz) {
      const auto coefficient = modes[mixed_index(iz, mode, nx)];
      real[static_cast<std::size_t>(iz)] = coefficient.real();
      imag[static_cast<std::size_t>(iz)] = coefficient.imag();
    }
    const auto real_solution = chebyshev_neumann_helmholtz(
        real, lambda, plus_hat[static_cast<std::size_t>(mode)].real(),
        minus_hat[static_cast<std::size_t>(mode)].real());
    const auto imag_solution = chebyshev_neumann_helmholtz(
        imag, lambda, plus_hat[static_cast<std::size_t>(mode)].imag(),
        minus_hat[static_cast<std::size_t>(mode)].imag());
    for (int iz = 0; iz < nline; ++iz) {
      modes[mixed_index(iz, mode, nx)] =
          std::complex<double>(real_solution[static_cast<std::size_t>(iz)],
                               imag_solution[static_cast<std::size_t>(iz)]);
    }
  }

  std::vector<double> solution(forcing.size());
  if (nx == 1) {
    for (int iz = 0; iz < nline; ++iz) {
      solution[static_cast<std::size_t>(iz)] =
          modes[static_cast<std::size_t>(iz)].real();
    }
    return solution;
  }

  Plan plan(nx);
  std::vector<std::complex<double>> line(static_cast<std::size_t>(nx));
  const double scale = static_cast<double>(nx);
  for (int iz = 0; iz < nline; ++iz) {
    for (int mode = 0; mode < nx; ++mode) {
      line[static_cast<std::size_t>(mode)] = modes[mixed_index(iz, mode, nx)];
    }
    plan.load(line);
    fftw_execute(plan.backward);
    const auto samples = plan.read();
    for (int ix = 0; ix < nx; ++ix) {
      solution[mixed_index(iz, ix, nx)] =
          samples[static_cast<std::size_t>(ix)].real() / scale;
    }
  }
  return solution;
}

std::vector<double> fourier_chebyshev_robin_poisson(
    std::span<const double> forcing, int nx, double period, double value_weight_plus,
    double slope_weight_plus, std::span<const double> data_at_plus,
    double value_weight_minus, double slope_weight_minus,
    std::span<const double> data_at_minus) {
  if (nx < 1) {
    throw std::invalid_argument("chebyshev: periodic count must be positive");
  }
  if (!(period > 0.0)) {
    throw std::invalid_argument("chebyshev: period must be positive");
  }
  if (forcing.empty() || forcing.size() % static_cast<std::size_t>(nx) != 0) {
    throw std::invalid_argument(
        "chebyshev: forcing does not match the periodic count");
  }
  if (data_at_plus.size() != static_cast<std::size_t>(nx) ||
      data_at_minus.size() != static_cast<std::size_t>(nx)) {
    throw std::invalid_argument(
        "chebyshev: boundary trace does not match the periodic count");
  }
  const auto lines = forcing.size() / static_cast<std::size_t>(nx);
  if (lines > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  const int nline = static_cast<int>(lines);
  if (nline < 2) {
    throw std::invalid_argument("chebyshev: Robin Poisson needs both endpoints");
  }

  std::vector<std::complex<double>> modes(forcing.size());
  std::vector<std::complex<double>> plus_hat(static_cast<std::size_t>(nx));
  std::vector<std::complex<double>> minus_hat(static_cast<std::size_t>(nx));
  if (nx == 1) {
    for (int iz = 0; iz < nline; ++iz) {
      modes[static_cast<std::size_t>(iz)] = forcing[static_cast<std::size_t>(iz)];
    }
    plus_hat[0] = data_at_plus[0];
    minus_hat[0] = data_at_minus[0];
  } else {
    Plan plan(nx);
    std::vector<std::complex<double>> line(static_cast<std::size_t>(nx));
    for (int iz = 0; iz < nline; ++iz) {
      for (int ix = 0; ix < nx; ++ix) {
        line[static_cast<std::size_t>(ix)] = forcing[mixed_index(iz, ix, nx)];
      }
      plan.load(line);
      fftw_execute(plan.forward);
      const auto bins = plan.read();
      for (int mode = 0; mode < nx; ++mode) {
        modes[mixed_index(iz, mode, nx)] = bins[static_cast<std::size_t>(mode)];
      }
    }
    for (int ix = 0; ix < nx; ++ix) {
      line[static_cast<std::size_t>(ix)] =
          data_at_plus[static_cast<std::size_t>(ix)];
    }
    plan.load(line);
    fftw_execute(plan.forward);
    plus_hat = plan.read();
    for (int ix = 0; ix < nx; ++ix) {
      line[static_cast<std::size_t>(ix)] =
          data_at_minus[static_cast<std::size_t>(ix)];
    }
    plan.load(line);
    fftw_execute(plan.forward);
    minus_hat = plan.read();
  }

  for (int mode = 0; mode < nx; ++mode) {
    const double wavenumber = periodic_wavenumber(mode, nx, period);
    const double lambda = wavenumber * wavenumber;
    std::vector<double> real(static_cast<std::size_t>(nline));
    std::vector<double> imag(static_cast<std::size_t>(nline));
    for (int iz = 0; iz < nline; ++iz) {
      const auto coefficient = modes[mixed_index(iz, mode, nx)];
      real[static_cast<std::size_t>(iz)] = coefficient.real();
      imag[static_cast<std::size_t>(iz)] = coefficient.imag();
    }
    const auto real_solution = chebyshev_robin_helmholtz(
        real, lambda, value_weight_plus, slope_weight_plus,
        plus_hat[static_cast<std::size_t>(mode)].real(), value_weight_minus,
        slope_weight_minus, minus_hat[static_cast<std::size_t>(mode)].real());
    const auto imag_solution = chebyshev_robin_helmholtz(
        imag, lambda, value_weight_plus, slope_weight_plus,
        plus_hat[static_cast<std::size_t>(mode)].imag(), value_weight_minus,
        slope_weight_minus, minus_hat[static_cast<std::size_t>(mode)].imag());
    for (int iz = 0; iz < nline; ++iz) {
      modes[mixed_index(iz, mode, nx)] =
          std::complex<double>(real_solution[static_cast<std::size_t>(iz)],
                               imag_solution[static_cast<std::size_t>(iz)]);
    }
  }

  std::vector<double> solution(forcing.size());
  if (nx == 1) {
    for (int iz = 0; iz < nline; ++iz) {
      solution[static_cast<std::size_t>(iz)] =
          modes[static_cast<std::size_t>(iz)].real();
    }
    return solution;
  }

  Plan plan(nx);
  std::vector<std::complex<double>> line(static_cast<std::size_t>(nx));
  const double scale = static_cast<double>(nx);
  for (int iz = 0; iz < nline; ++iz) {
    for (int mode = 0; mode < nx; ++mode) {
      line[static_cast<std::size_t>(mode)] = modes[mixed_index(iz, mode, nx)];
    }
    plan.load(line);
    fftw_execute(plan.backward);
    const auto samples = plan.read();
    for (int ix = 0; ix < nx; ++ix) {
      solution[mixed_index(iz, ix, nx)] =
          samples[static_cast<std::size_t>(ix)].real() / scale;
    }
  }
  return solution;
}

std::vector<double> fourier_chebyshev_robin_helmholtz(
    std::span<const double> forcing, int nx, double period, double lambda,
    double value_weight_plus, double slope_weight_plus,
    std::span<const double> data_at_plus, double value_weight_minus,
    double slope_weight_minus, std::span<const double> data_at_minus) {
  if (lambda == 0.0) {
    return fourier_chebyshev_robin_poisson(
        forcing, nx, period, value_weight_plus, slope_weight_plus, data_at_plus,
        value_weight_minus, slope_weight_minus, data_at_minus);
  }
  if (nx < 1) {
    throw std::invalid_argument("chebyshev: periodic count must be positive");
  }
  if (!(period > 0.0)) {
    throw std::invalid_argument("chebyshev: period must be positive");
  }
  if (forcing.empty() || forcing.size() % static_cast<std::size_t>(nx) != 0) {
    throw std::invalid_argument(
        "chebyshev: forcing does not match the periodic count");
  }
  if (data_at_plus.size() != static_cast<std::size_t>(nx) ||
      data_at_minus.size() != static_cast<std::size_t>(nx)) {
    throw std::invalid_argument(
        "chebyshev: boundary trace does not match the periodic count");
  }
  const auto lines = forcing.size() / static_cast<std::size_t>(nx);
  if (lines > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  const int nline = static_cast<int>(lines);
  if (nline < 2) {
    throw std::invalid_argument("chebyshev: Robin Helmholtz needs both endpoints");
  }

  std::vector<std::complex<double>> modes(forcing.size());
  std::vector<std::complex<double>> plus_hat(static_cast<std::size_t>(nx));
  std::vector<std::complex<double>> minus_hat(static_cast<std::size_t>(nx));
  if (nx == 1) {
    for (int iz = 0; iz < nline; ++iz) {
      modes[static_cast<std::size_t>(iz)] = forcing[static_cast<std::size_t>(iz)];
    }
    plus_hat[0] = data_at_plus[0];
    minus_hat[0] = data_at_minus[0];
  } else {
    Plan plan(nx);
    std::vector<std::complex<double>> line(static_cast<std::size_t>(nx));
    for (int iz = 0; iz < nline; ++iz) {
      for (int ix = 0; ix < nx; ++ix) {
        line[static_cast<std::size_t>(ix)] = forcing[mixed_index(iz, ix, nx)];
      }
      plan.load(line);
      fftw_execute(plan.forward);
      const auto bins = plan.read();
      for (int mode = 0; mode < nx; ++mode) {
        modes[mixed_index(iz, mode, nx)] = bins[static_cast<std::size_t>(mode)];
      }
    }
    for (int ix = 0; ix < nx; ++ix) {
      line[static_cast<std::size_t>(ix)] =
          data_at_plus[static_cast<std::size_t>(ix)];
    }
    plan.load(line);
    fftw_execute(plan.forward);
    plus_hat = plan.read();
    for (int ix = 0; ix < nx; ++ix) {
      line[static_cast<std::size_t>(ix)] =
          data_at_minus[static_cast<std::size_t>(ix)];
    }
    plan.load(line);
    fftw_execute(plan.forward);
    minus_hat = plan.read();
  }

  for (int mode = 0; mode < nx; ++mode) {
    const double wavenumber = periodic_wavenumber(mode, nx, period);
    const double modal = wavenumber * wavenumber + lambda;
    std::vector<double> real(static_cast<std::size_t>(nline));
    std::vector<double> imag(static_cast<std::size_t>(nline));
    for (int iz = 0; iz < nline; ++iz) {
      const auto coefficient = modes[mixed_index(iz, mode, nx)];
      real[static_cast<std::size_t>(iz)] = coefficient.real();
      imag[static_cast<std::size_t>(iz)] = coefficient.imag();
    }
    const auto real_solution = chebyshev_robin_helmholtz(
        real, modal, value_weight_plus, slope_weight_plus,
        plus_hat[static_cast<std::size_t>(mode)].real(), value_weight_minus,
        slope_weight_minus, minus_hat[static_cast<std::size_t>(mode)].real());
    const auto imag_solution = chebyshev_robin_helmholtz(
        imag, modal, value_weight_plus, slope_weight_plus,
        plus_hat[static_cast<std::size_t>(mode)].imag(), value_weight_minus,
        slope_weight_minus, minus_hat[static_cast<std::size_t>(mode)].imag());
    for (int iz = 0; iz < nline; ++iz) {
      modes[mixed_index(iz, mode, nx)] =
          std::complex<double>(real_solution[static_cast<std::size_t>(iz)],
                               imag_solution[static_cast<std::size_t>(iz)]);
    }
  }

  std::vector<double> solution(forcing.size());
  if (nx == 1) {
    for (int iz = 0; iz < nline; ++iz) {
      solution[static_cast<std::size_t>(iz)] =
          modes[static_cast<std::size_t>(iz)].real();
    }
    return solution;
  }

  Plan plan(nx);
  std::vector<std::complex<double>> line(static_cast<std::size_t>(nx));
  const double scale = static_cast<double>(nx);
  for (int iz = 0; iz < nline; ++iz) {
    for (int mode = 0; mode < nx; ++mode) {
      line[static_cast<std::size_t>(mode)] = modes[mixed_index(iz, mode, nx)];
    }
    plan.load(line);
    fftw_execute(plan.backward);
    const auto samples = plan.read();
    for (int ix = 0; ix < nx; ++ix) {
      solution[mixed_index(iz, ix, nx)] =
          samples[static_cast<std::size_t>(ix)].real() / scale;
    }
  }
  return solution;
}

std::vector<double>
chebyshev_dirichlet_poisson(std::span<const double> forcing,
                            double value_at_plus, double value_at_minus) {
  if (forcing.size() < 2) {
    throw std::invalid_argument(
        "chebyshev: Dirichlet Poisson needs both endpoints");
  }
  if (forcing.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  const int degree = static_cast<int>(forcing.size()) - 1;
  const auto slope = integrate_coefficients(chebyshev_coefficients(forcing));
  auto solution = integrate_coefficients(slope);

  double at_plus = 0.0;
  double at_minus = 0.0;
  for (int k = 0; k < static_cast<int>(solution.size()); ++k) {
    const double term = solution[static_cast<std::size_t>(k)];
    at_plus += term;
    at_minus += (k % 2 == 0) ? term : -term;
  }
  solution.front() +=
      0.5 * ((value_at_plus + value_at_minus) - (at_plus + at_minus));
  solution[1] +=
      0.5 * ((value_at_plus - value_at_minus) - (at_plus - at_minus));

  const auto nodes = chebyshev_lobatto(degree);
  std::vector<double> values(nodes.size());
  for (std::size_t j = 0; j < nodes.size(); ++j) {
    values[j] = clenshaw(solution, nodes[j]);
  }
  return values;
}

std::vector<double> chebyshev_neumann_poisson(std::span<const double> forcing,
                                              double slope_at_plus,
                                              double slope_at_minus) {
  if (forcing.size() < 2) {
    throw std::invalid_argument("chebyshev: Neumann Poisson needs both endpoints");
  }
  if (forcing.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  const int degree = static_cast<int>(forcing.size()) - 1;
  auto slope = integrate_coefficients(chebyshev_coefficients(forcing));

  double series_plus = 0.0;
  double series_minus = 0.0;
  for (int k = 0; k < static_cast<int>(slope.size()); ++k) {
    const double term = slope[static_cast<std::size_t>(k)];
    series_plus += term;
    series_minus += (k % 2 == 0) ? term : -term;
  }
  const double jump = slope_at_plus - slope_at_minus;
  const double series_jump = series_plus - series_minus;
  const double scale = 1.0 + std::abs(slope_at_plus) + std::abs(slope_at_minus) +
                       std::abs(series_plus) + std::abs(series_minus);
  if (std::abs(jump - series_jump) > 1e-8 * scale) {
    throw std::invalid_argument(
        "chebyshev: Neumann slopes are incompatible with the forcing");
  }
  slope.front() +=
      0.5 * ((slope_at_plus + slope_at_minus) - (series_plus + series_minus));
  auto solution = integrate_coefficients(slope);

  double even_tail = 0.0;
  for (int k = 2; k < static_cast<int>(solution.size()); k += 2) {
    const double mode = static_cast<double>(k);
    even_tail += solution[static_cast<std::size_t>(k)] / (1.0 - mode * mode);
  }
  solution.front() = -even_tail;

  const auto nodes = chebyshev_lobatto(degree);
  std::vector<double> values(nodes.size());
  for (std::size_t j = 0; j < nodes.size(); ++j) {
    values[j] = clenshaw(solution, nodes[j]);
  }
  return values;
}

std::vector<double> chebyshev_dirichlet_helmholtz(std::span<const double> forcing,
                                                  double lambda,
                                                  double value_at_plus,
                                                  double value_at_minus) {
  if (forcing.size() < 2) {
    throw std::invalid_argument(
        "chebyshev: Helmholtz Dirichlet needs both endpoints");
  }
  if (forcing.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  if (lambda == 0.0) {
    return chebyshev_dirichlet_poisson(forcing, value_at_plus, value_at_minus);
  }
  return solve_matched_helmholtz(
      forcing, lambda, value_at_plus, value_at_minus, HelmholtzMatch::values,
      "chebyshev: Helmholtz Dirichlet problem is singular");
}

std::vector<double> chebyshev_neumann_helmholtz(std::span<const double> forcing,
                                                double lambda, double slope_at_plus,
                                                double slope_at_minus) {
  if (forcing.size() < 2) {
    throw std::invalid_argument("chebyshev: Helmholtz Neumann needs both endpoints");
  }
  if (forcing.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  if (lambda == 0.0) {
    return chebyshev_neumann_poisson(forcing, slope_at_plus, slope_at_minus);
  }
  return solve_matched_helmholtz(forcing, lambda, slope_at_plus, slope_at_minus,
                                 HelmholtzMatch::slopes,
                                 "chebyshev: Helmholtz Neumann problem is singular");
}

std::vector<double>
chebyshev_robin_poisson(std::span<const double> forcing, double value_weight_plus,
                        double slope_weight_plus, double data_plus,
                        double value_weight_minus, double slope_weight_minus,
                        double data_minus) {
  if (forcing.size() < 2) {
    throw std::invalid_argument("chebyshev: Robin Poisson needs both endpoints");
  }
  if (forcing.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  const bool plus_empty = value_weight_plus == 0.0 && slope_weight_plus == 0.0;
  const bool minus_empty = value_weight_minus == 0.0 && slope_weight_minus == 0.0;
  if (plus_empty || minus_empty) {
    throw std::invalid_argument("chebyshev: Robin condition needs a weight");
  }
  if (slope_weight_plus == 0.0 && slope_weight_minus == 0.0) {
    return chebyshev_dirichlet_poisson(forcing, data_plus / value_weight_plus,
                                       data_minus / value_weight_minus);
  }
  if (value_weight_plus == 0.0 && value_weight_minus == 0.0) {
    return chebyshev_neumann_poisson(forcing, data_plus / slope_weight_plus,
                                     data_minus / slope_weight_minus);
  }

  const int degree = static_cast<int>(forcing.size()) - 1;
  const auto slope = integrate_coefficients(chebyshev_coefficients(forcing));
  auto solution = integrate_coefficients(slope);
  const auto values = endpoint_sum(solution);
  const auto slopes = endpoint_sum(slope);
  const double plus_constant = value_weight_plus;
  const double plus_linear = value_weight_plus + slope_weight_plus;
  const double minus_constant = value_weight_minus;
  const double minus_linear = -value_weight_minus + slope_weight_minus;
  const double rhs_plus = data_plus - value_weight_plus * values.at_plus -
                          slope_weight_plus * slopes.at_plus;
  const double rhs_minus = data_minus - value_weight_minus * values.at_minus -
                           slope_weight_minus * slopes.at_minus;
  const double constant_scale = column_scale(plus_constant, minus_constant);
  const double linear_scale = column_scale(plus_linear, minus_linear);
  const double plus_constant_scaled = plus_constant / constant_scale;
  const double minus_constant_scaled = minus_constant / constant_scale;
  const double plus_linear_scaled = plus_linear / linear_scale;
  const double minus_linear_scaled = minus_linear / linear_scale;
  const double determinant = plus_constant_scaled * minus_linear_scaled -
                             minus_constant_scaled * plus_linear_scaled;
  if (!(std::abs(determinant) > 1e-8)) {
    throw std::invalid_argument("chebyshev: Robin Poisson problem is singular");
  }
  const double constant =
      (rhs_plus * minus_linear_scaled - plus_linear_scaled * rhs_minus) /
      determinant / constant_scale;
  const double linear =
      (plus_constant_scaled * rhs_minus - minus_constant_scaled * rhs_plus) /
      determinant / linear_scale;
  solution.front() += constant;
  solution[1] += linear;
  return evaluate_series(solution, degree);
}

std::vector<double>
chebyshev_robin_helmholtz(std::span<const double> forcing, double lambda,
                          double value_weight_plus, double slope_weight_plus,
                          double data_plus, double value_weight_minus,
                          double slope_weight_minus, double data_minus) {
  if (forcing.size() < 2) {
    throw std::invalid_argument("chebyshev: Robin Helmholtz needs both endpoints");
  }
  if (forcing.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("chebyshev: grid is too large");
  }
  const bool plus_empty = value_weight_plus == 0.0 && slope_weight_plus == 0.0;
  const bool minus_empty = value_weight_minus == 0.0 && slope_weight_minus == 0.0;
  if (plus_empty || minus_empty) {
    throw std::invalid_argument(
        "chebyshev: Robin Helmholtz condition needs a weight");
  }
  if (lambda == 0.0) {
    return chebyshev_robin_poisson(forcing, value_weight_plus, slope_weight_plus,
                                   data_plus, value_weight_minus, slope_weight_minus,
                                   data_minus);
  }
  if (slope_weight_plus == 0.0 && slope_weight_minus == 0.0) {
    return chebyshev_dirichlet_helmholtz(forcing, lambda,
                                         data_plus / value_weight_plus,
                                         data_minus / value_weight_minus);
  }
  if (value_weight_plus == 0.0 && value_weight_minus == 0.0) {
    return chebyshev_neumann_helmholtz(forcing, lambda,
                                       data_plus / slope_weight_plus,
                                       data_minus / slope_weight_minus);
  }
  return solve_robin_helmholtz(forcing, lambda,
                               RobinWeights{value_weight_plus, slope_weight_plus,
                                            value_weight_minus, slope_weight_minus},
                               data_plus, data_minus);
}

} // namespace pfc::fft

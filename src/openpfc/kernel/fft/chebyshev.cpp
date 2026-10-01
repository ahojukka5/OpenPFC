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
 */

#include <openpfc/kernel/fft/chebyshev.hpp>

#include <cmath>
#include <complex>
#include <fftw3.h>
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

} // namespace pfc::fft

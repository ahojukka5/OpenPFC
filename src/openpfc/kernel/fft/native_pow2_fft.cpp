// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/fft/native_pow2_fft.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__)
#include <immintrin.h>
#define OPENPFC_NATIVE_POW2_AVX2 1
#else
#define OPENPFC_NATIVE_POW2_AVX2 0
#endif

namespace pfc::fft {
namespace {

[[nodiscard]] bool pow2(int n) noexcept {
  const auto value = static_cast<unsigned>(n);
  return n > 0 && (value & (value - 1U)) == 0;
}

[[nodiscard]] std::size_t checked_count(const std::array<int, 3> &n) {
  std::size_t count = 1;
  for (int v : n) {
    if (!pow2(v))
      throw std::invalid_argument(
          "NativePow2FFT: each grid dimension must be a power of two; "
          "choose HeFFTe for other grid sizes");
    // Two reduced half-spectrum and two packed-x SoA arrays are allocated.
    const auto max_count =
        std::numeric_limits<std::size_t>::max() / (2 * sizeof(double));
    if (count > max_count / static_cast<std::size_t>(v))
      throw std::invalid_argument(
          "NativePow2FFT: volume exceeds addressable memory");
    count *= static_cast<std::size_t>(v);
  }
  return count;
}

void check_buffer(std::size_t actual, std::size_t expected, const char *context) {
  if (actual != expected) {
    std::string message = "NativePow2FFT: ";
    message += context;
    message += " size ";
    message += std::to_string(actual);
    message += ", expected ";
    message += std::to_string(expected);
    throw std::invalid_argument(message);
  }
}

struct Axis {
  std::size_t length;
  std::size_t stride;
  std::vector<std::pair<std::size_t, std::size_t>> bit_swaps;
  std::vector<double> wr, wi;

  Axis(int n, std::size_t step)
      : length(static_cast<std::size_t>(n)), stride(step), wr(length / 2),
        wi(length / 2) {
    std::size_t width = 0;
    for (std::size_t l = length; l > 1; l >>= 1) ++width;
    for (std::size_t i = 0; i < length; ++i) {
      std::size_t j = 0, value = i;
      for (std::size_t bit = 0; bit < width; ++bit) {
        j = (j << 1) | (value & 1);
        value >>= 1;
      }
      if (i < j) bit_swaps.emplace_back(i, j);
    }
    for (std::size_t k = 0; k < length / 2; ++k) {
      const double angle = -2.0 * std::numbers::pi * double(k) / double(length);
      wr[k] = std::cos(angle);
      wi[k] = std::sin(angle);
    }
  }

  [[nodiscard]] std::size_t allocated_bytes() const noexcept {
    return bit_swaps.capacity() * sizeof(bit_swaps.front()) +
           (wr.capacity() + wi.capacity()) * sizeof(double);
  }
};

inline void butterfly_scalar(double *real, double *imag, std::size_t a,
                             std::size_t b, std::size_t lanes, double wr,
                             double wi) noexcept {
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    const double ur = real[a + lane], ui = imag[a + lane];
    const double vr = real[b + lane], vi = imag[b + lane];
    const double tr = vr * wr - vi * wi;
    const double ti = vr * wi + vi * wr;
    real[a + lane] = ur + tr;
    imag[a + lane] = ui + ti;
    real[b + lane] = ur - tr;
    imag[b + lane] = ui - ti;
  }
}

#if OPENPFC_NATIVE_POW2_AVX2
// Explicit SIMD across four independent contiguous x-lanes, not four
// dependent butterflies in one transform. This function is never called
// unless both AVX2 and FMA were detected on the current CPU.
// clang-format off
// Compiler-specific target attribute: keep this declaration together.
[[gnu::target("avx2,fma")]] void butterfly_avx2(
    double *real, double *imag, std::size_t a, std::size_t b,
    std::size_t lanes, double wr, double wi) noexcept {
  const __m256d wre = _mm256_set1_pd(wr);
  const __m256d wim = _mm256_set1_pd(wi);
  std::size_t i = 0;
  for (; i + 4 <= lanes; i += 4) {
    const __m256d ur = _mm256_loadu_pd(real + a + i);
    const __m256d ui = _mm256_loadu_pd(imag + a + i);
    const __m256d vr = _mm256_loadu_pd(real + b + i);
    const __m256d vi = _mm256_loadu_pd(imag + b + i);
    const __m256d tr = _mm256_fmsub_pd(vr, wre, _mm256_mul_pd(vi, wim));
    const __m256d ti = _mm256_fmadd_pd(vr, wim, _mm256_mul_pd(vi, wre));
    _mm256_storeu_pd(real + a + i, _mm256_add_pd(ur, tr));
    _mm256_storeu_pd(imag + a + i, _mm256_add_pd(ui, ti));
    _mm256_storeu_pd(real + b + i, _mm256_sub_pd(ur, tr));
    _mm256_storeu_pd(imag + b + i, _mm256_sub_pd(ui, ti));
  }
  if (i < lanes) butterfly_scalar(real, imag, a + i, b + i, lanes - i, wr, wi);
}
// clang-format on

[[nodiscard]] bool cpu_has_avx2_fma() noexcept {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#else
[[nodiscard]] bool cpu_has_avx2_fma() noexcept { return false; }
#endif

#if OPENPFC_NATIVE_POW2_AVX2
// Four independent x-line transforms per 256-bit vector. The packed
// even/odd samples use [group][frequency][four_rows] SoA order, so every
// AVX2 load/store is contiguous. This method changes *layout* only for
// the plan-owned temporary arrays; caller-visible output is unchanged.
//
// clang-format off
[[gnu::target("avx2,fma")]] void transform_x_batch4(
    std::vector<double> &re, std::vector<double> &im,
    const Axis &axis, std::size_t groups, bool inverse) {
  const std::size_t n = axis.length;
  for (std::size_t group = 0; group < groups; ++group) {
    const std::size_t base = group * n * 4;
    for (const auto &[i, j] : axis.bit_swaps) {
      const auto a = base + i * 4;
      const auto b = base + j * 4;
      const __m256d ar = _mm256_loadu_pd(re.data() + a);
      const __m256d ai = _mm256_loadu_pd(im.data() + a);
      const __m256d br = _mm256_loadu_pd(re.data() + b);
      const __m256d bi = _mm256_loadu_pd(im.data() + b);
      _mm256_storeu_pd(re.data() + a, br);
      _mm256_storeu_pd(im.data() + a, bi);
      _mm256_storeu_pd(re.data() + b, ar);
      _mm256_storeu_pd(im.data() + b, ai);
    }
    for (std::size_t width = 2; width <= n; width *= 2) {
      const std::size_t half = width / 2;
      const std::size_t step = n / width;
      for (std::size_t start = 0; start < n; start += width) {
        for (std::size_t k = 0; k < half; ++k) {
          const auto a = base + (start + k) * 4;
          const auto b = a + half * 4;
          const double sign = inverse ? -1.0 : 1.0;
          const __m256d wr = _mm256_set1_pd(axis.wr[k * step]);
          const __m256d wi = _mm256_set1_pd(sign * axis.wi[k * step]);
          const __m256d ur = _mm256_loadu_pd(re.data() + a);
          const __m256d ui = _mm256_loadu_pd(im.data() + a);
          const __m256d vr = _mm256_loadu_pd(re.data() + b);
          const __m256d vi = _mm256_loadu_pd(im.data() + b);
          const __m256d tr = _mm256_fmsub_pd(vr, wr, _mm256_mul_pd(vi, wi));
          const __m256d ti = _mm256_fmadd_pd(vr, wi, _mm256_mul_pd(vi, wr));
          _mm256_storeu_pd(re.data() + a, _mm256_add_pd(ur, tr));
          _mm256_storeu_pd(im.data() + a, _mm256_add_pd(ui, ti));
          _mm256_storeu_pd(re.data() + b, _mm256_sub_pd(ur, tr));
          _mm256_storeu_pd(im.data() + b, _mm256_sub_pd(ui, ti));
        }
      }
    }
  }
}
// clang-format on
#endif

void transform_axis(std::vector<double> &re, std::vector<double> &im,
                    const Axis &axis, bool inverse, bool avx2) {
  const std::size_t n = axis.length, stride = axis.stride;
  if (n == 1) return;
  const std::size_t group = n * stride;
  for (std::size_t outer = 0; outer < re.size(); outer += group) {
    for (const auto &[i, j] : axis.bit_swaps) {
      const std::size_t a = outer + i * stride;
      const std::size_t b = outer + j * stride;
      for (std::size_t lane = 0; lane < stride; ++lane) {
        std::swap(re[a + lane], re[b + lane]);
        std::swap(im[a + lane], im[b + lane]);
      }
    }
    for (std::size_t width = 2; width <= n; width *= 2) {
      const std::size_t half = width / 2, step = n / width;
      for (std::size_t start = 0; start < n; start += width) {
        for (std::size_t k = 0; k < half; ++k) {
          const std::size_t a = outer + (start + k) * stride;
          const std::size_t b = a + half * stride;
          const double wr = axis.wr[k * step];
          const double wi = inverse ? -axis.wi[k * step] : axis.wi[k * step];
#if OPENPFC_NATIVE_POW2_AVX2
          if (avx2 && stride >= 4)
            butterfly_avx2(re.data(), im.data(), a, b, stride, wr, wi);
          else
#endif
            butterfly_scalar(re.data(), im.data(), a, b, stride, wr, wi);
        }
      }
    }
  }
}

} // namespace

struct NativePow2FFT::Impl {
  const std::array<int, 3> n;
  const std::size_t nr, nc, nx, nxh, packed_x, rows;
  const bool avx2;
  const bool batch_x4;
  // Re/imaginary arrays are indexed as [z][y][kx], kx=0..nx/2.
  // Packed arrays store two real x samples per complex number.
  std::vector<double> re, im, packed_re, packed_im, x_wr, x_wi;
  std::array<Axis, 3> axes;
  double time = 0.0;

  // clang-format off
  // Deliberately one initializer per semantic field; no temporary allocations.
  explicit Impl(const std::array<int, 3> &size)
      : n(size), nr(checked_count(size)),
        nc((static_cast<std::size_t>(n[0]) / 2 + 1) *
           static_cast<std::size_t>(n[1]) * static_cast<std::size_t>(n[2])),
        nx(static_cast<std::size_t>(n[0])), nxh(nx / 2 + 1),
        packed_x(nx == 1 ? 0 : nx / 2), rows(nr / nx),
        avx2(cpu_has_avx2_fma()),
#if defined(OPENPFC_NATIVE_POW2_DISABLE_X_BATCH)
        batch_x4(false),
#else
        batch_x4(avx2 && packed_x >= 2 && rows >= 4),
#endif
        re(nc), im(nc),
        packed_re(rows * packed_x), packed_im(rows * packed_x),
        x_wr(packed_x + (nx == 1 ? 0 : 1)),
        x_wi(packed_x + (nx == 1 ? 0 : 1)),
        axes{Axis(nx == 1 ? 1 : n[0] / 2, 1), Axis(n[1], nxh),
             Axis(n[2], nxh * static_cast<std::size_t>(n[1]))} {
    // Frequencies for reconstructing an Nx real transform from Nx/2
    // complex coefficients. These are constant over plan execution.
    for (std::size_t k = 0; k <= packed_x && nx > 1; ++k) {
      const double phase = static_cast<double>(k) / static_cast<double>(nx);
      const double angle = -2.0 * std::numbers::pi * phase;
      x_wr[k] = std::cos(angle);
      x_wi[k] = std::sin(angle);
    }
  }
  // clang-format on

  // Row-major when scalar; AoSoA [group][x_frequency][lane] for
  // four simultaneously transformed rows on AVX2. The number of
  // x-lines (ny*nz) is a power of two, so the four-lane case has
  // no partial group or masked SIMD operation.
  [[nodiscard]] std::size_t packed_base(std::size_t row) const noexcept {
    if (batch_x4) return (row / 4) * packed_x * 4 + (row % 4);
    return row * packed_x;
  }

  void transform_packed_x(bool inverse) {
#if OPENPFC_NATIVE_POW2_AVX2
    if (batch_x4) {
      transform_x_batch4(packed_re, packed_im, axes[0], rows / 4, inverse);
      return;
    }
#endif
    transform_axis(packed_re, packed_im, axes[0], inverse, false);
  }

  // Pack even/odd real samples into one half-length complex transform.
  // No full Nx-complex spectrum is ever allocated or transformed.
  void real_x_forward(const RealVector &input) {
    if (nx == 1) {
      for (std::size_t i = 0; i < nr; ++i) {
        re[i] = input[i];
        im[i] = 0.0;
      }
      return;
    }
    for (std::size_t row = 0; row < rows; ++row) {
      const auto real_base = row * nx;
      const auto complex_base = packed_base(row);
      const auto stride = batch_x4 ? 4U : 1U;
      for (std::size_t j = 0; j < packed_x; ++j) {
        packed_re[complex_base + j * stride] = input[real_base + 2 * j];
        packed_im[complex_base + j * stride] = input[real_base + 2 * j + 1];
      }
    }
    transform_packed_x(false);
    for (std::size_t row = 0; row < rows; ++row) {
      const auto base = packed_base(row);
      const auto stride = batch_x4 ? 4U : 1U;
      const auto spectral_base = row * nxh;
      for (std::size_t k = 0; k <= packed_x; ++k) {
        const std::size_t left = k == packed_x ? 0 : k;
        const std::size_t right = k == 0 ? 0 : packed_x - k;
        const double ar = packed_re[base + left * stride];
        const double ai = packed_im[base + left * stride];
        const double br = packed_re[base + right * stride];
        const double bi = -packed_im[base + right * stride];
        const double dr = ar - br, di = ai - bi;
        const double tr = x_wr[k] * dr - x_wi[k] * di;
        const double ti = x_wi[k] * dr + x_wr[k] * di;
        re[spectral_base + k] = 0.5 * (ar + br + ti);
        im[spectral_base + k] = 0.5 * (ai + bi - tr);
      }
    }
  }

  // Recover the packed even/odd complex sequence from each half-spectrum
  // AFTER the inverse y/z FFTs. At fixed spatial y/z, the x spectrum
  // once again obeys one-dimensional Hermitian symmetry.
  void real_x_backward(RealVector &output) {
    const double scale = 1.0 / static_cast<double>(nr);
    if (nx == 1) {
      for (std::size_t i = 0; i < nr; ++i) output[i] = re[i] * scale;
      return;
    }
    for (std::size_t row = 0; row < rows; ++row) {
      const auto base = packed_base(row);
      const auto stride = batch_x4 ? 4U : 1U;
      const auto spectral_base = row * nxh;
      for (std::size_t k = 0; k < packed_x; ++k) {
        const std::size_t partner = packed_x - k;
        const double ar = re[spectral_base + k];
        const double ai = im[spectral_base + k];
        const double br = re[spectral_base + partner];
        const double bi = -im[spectral_base + partner];
        const double dr = ar - br, di = ai - bi;
        // Multiply (F[k]-conj(F[m-k])) by conj(W_N^k);
        // then multiply by i. The factor two compensates the
        // half-size inverse FFT's unnormalised convention.
        const double tr = x_wr[k] * dr + x_wi[k] * di;
        const double ti = -x_wi[k] * dr + x_wr[k] * di;
        packed_re[base + k * stride] = ar + br - ti;
        packed_im[base + k * stride] = ai + bi + tr;
      }
    }
    transform_packed_x(true);
    for (std::size_t row = 0; row < rows; ++row) {
      const auto real_base = row * nx;
      const auto base = packed_base(row);
      const auto stride = batch_x4 ? 4U : 1U;
      for (std::size_t j = 0; j < packed_x; ++j) {
        output[real_base + 2 * j] = packed_re[base + j * stride] * scale;
        const double odd = packed_im[base + j * stride] * scale;
        output[real_base + 2 * j + 1] = odd;
      }
    }
  }

  // Cross-axis SIMD still handles only Nx/2+1 complex frequencies.
  // The x-axis uses four adjacent independent real rows per AVX2 group,
  // with an unmodified scalar fallback for short lines and older CPUs.
  void forward_yz() {
    transform_axis(re, im, axes[1], false, avx2);
    transform_axis(re, im, axes[2], false, avx2);
  }
  void backward_zy() {
    transform_axis(re, im, axes[2], true, avx2);
    transform_axis(re, im, axes[1], true, avx2);
  }

  [[nodiscard]] std::size_t allocated_bytes() const noexcept {
    std::size_t capacity = re.capacity() + im.capacity();
    capacity += packed_re.capacity() + packed_im.capacity();
    capacity += x_wr.capacity() + x_wi.capacity();
    std::size_t bytes = capacity * sizeof(double);
    for (const auto &axis : axes) bytes += axis.allocated_bytes();
    return bytes;
  }
};

NativePow2FFT::NativePow2FFT(const std::array<int, 3> &size)
    : m_impl(std::make_unique<Impl>(size)) {}
NativePow2FFT::~NativePow2FFT() = default;

void NativePow2FFT::forward(const RealVector &in, ComplexVector &out) {
  check_buffer(in.size(), m_impl->nr, "forward real input");
  check_buffer(out.size(), m_impl->nc, "forward complex output");
  const auto begin = std::chrono::steady_clock::now();
  m_impl->real_x_forward(in);
  m_impl->forward_yz();
  for (std::size_t i = 0; i < m_impl->nc; ++i)
    out[i] = {m_impl->re[i], m_impl->im[i]};
  m_impl->time +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
}

void NativePow2FFT::backward(const ComplexVector &in, RealVector &out) {
  check_buffer(in.size(), m_impl->nc, "backward complex input");
  check_buffer(out.size(), m_impl->nr, "backward real output");
  const auto begin = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < m_impl->nc; ++i) {
    m_impl->re[i] = in[i].real();
    m_impl->im[i] = in[i].imag();
  }
  m_impl->backward_zy();
  m_impl->real_x_backward(out);
  m_impl->time +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
}

void NativePow2FFT::reset_fft_time() { m_impl->time = 0.0; }
double NativePow2FFT::get_fft_time() const { return m_impl->time; }
std::size_t NativePow2FFT::size_inbox() const { return m_impl->nr; }
std::size_t NativePow2FFT::size_outbox() const { return m_impl->nc; }
std::size_t NativePow2FFT::size_workspace() const {
  return (m_impl->allocated_bytes() + sizeof(std::complex<double>) - 1) /
         sizeof(std::complex<double>);
}
std::size_t NativePow2FFT::get_allocated_memory_bytes() const {
  return m_impl->allocated_bytes();
}
Box3i NativePow2FFT::get_inbox_bounds() const {
  const auto &n = m_impl->n;
  return Box3i::from_bounds({0, 0, 0}, {n[0] - 1, n[1] - 1, n[2] - 1});
}
Box3i NativePow2FFT::get_outbox_bounds() const {
  const auto &n = m_impl->n;
  return Box3i::from_bounds({0, 0, 0}, {n[0] / 2, n[1] - 1, n[2] - 1});
}

} // namespace pfc::fft

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
    // Two full-volume double-precision SoA arrays are allocated.
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

[[nodiscard]] bool cpu_has_avx2_fma() noexcept {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#else
[[nodiscard]] bool cpu_has_avx2_fma() noexcept { return false; }
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
  const std::size_t nr, nc;
  const bool avx2;
  std::vector<double> re, im;
  std::array<Axis, 3> axes;
  double time = 0.0;

  explicit Impl(const std::array<int, 3> &size)
      : n(size), nr(checked_count(size)),
        nc((static_cast<std::size_t>(n[0]) / 2 + 1) *
           static_cast<std::size_t>(n[1]) * static_cast<std::size_t>(n[2])),
        avx2(cpu_has_avx2_fma()), re(nr), im(nr),
        axes{Axis(n[0], 1), Axis(n[1], static_cast<std::size_t>(n[0])),
             Axis(n[2], static_cast<std::size_t>(n[0]) * n[1])} {}

  [[nodiscard]] std::size_t index(std::size_t x, std::size_t y,
                                  std::size_t z) const noexcept {
    return (z * static_cast<std::size_t>(n[1]) + y) * n[0] + x;
  }

  [[nodiscard]] std::size_t half_index(std::size_t x, std::size_t y,
                                       std::size_t z) const noexcept {
    return (z * static_cast<std::size_t>(n[1]) + y) *
               (static_cast<std::size_t>(n[0]) / 2 + 1) +
           x;
  }

  void fft(bool inverse) {
    if (inverse) {
      transform_axis(re, im, axes[2], true, avx2);
      transform_axis(re, im, axes[1], true, avx2);
      transform_axis(re, im, axes[0], true, avx2);
    } else {
      transform_axis(re, im, axes[0], false, avx2);
      transform_axis(re, im, axes[1], false, avx2);
      transform_axis(re, im, axes[2], false, avx2);
    }
  }

  [[nodiscard]] std::size_t allocated_bytes() const noexcept {
    std::size_t bytes = (re.capacity() + im.capacity()) * sizeof(double);
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
  for (std::size_t i = 0; i < m_impl->nr; ++i) {
    m_impl->re[i] = in[i];
    m_impl->im[i] = 0.0;
  }
  m_impl->fft(false);
  const std::size_t nxh = static_cast<std::size_t>(m_impl->n[0]) / 2 + 1;
  for (int z = 0; z < m_impl->n[2]; ++z) {
    for (int y = 0; y < m_impl->n[1]; ++y) {
      for (std::size_t x = 0; x < nxh; ++x) {
        const auto src = m_impl->index(x, y, z);
        out[m_impl->half_index(x, y, z)] = {m_impl->re[src], m_impl->im[src]};
      }
    }
  }
  m_impl->time +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
}

void NativePow2FFT::backward(const ComplexVector &in, RealVector &out) {
  check_buffer(in.size(), m_impl->nc, "backward complex input");
  check_buffer(out.size(), m_impl->nr, "backward real output");
  const auto begin = std::chrono::steady_clock::now();
  const std::size_t nx = static_cast<std::size_t>(m_impl->n[0]);
  const std::size_t ny = static_cast<std::size_t>(m_impl->n[1]);
  const std::size_t nz = static_cast<std::size_t>(m_impl->n[2]);
  const std::size_t nxh = nx / 2 + 1;
  for (std::size_t z = 0; z < nz; ++z) {
    for (std::size_t y = 0; y < ny; ++y) {
      for (std::size_t x = 0; x < nxh; ++x) {
        const auto value = in[m_impl->half_index(x, y, z)];
        const auto dst = m_impl->index(x, y, z);
        m_impl->re[dst] = value.real();
        m_impl->im[dst] = value.imag();
      }
      for (std::size_t x = nxh; x < nx; ++x) {
        // Hermitian partner must reflect ALL Fourier axes, not only x.
        const auto value =
            in[m_impl->half_index(nx - x, (ny - y) % ny, (nz - z) % nz)];
        const auto dst = m_impl->index(x, y, z);
        m_impl->re[dst] = value.real();
        m_impl->im[dst] = -value.imag();
      }
    }
  }
  m_impl->fft(true);
  const double scale = 1.0 / static_cast<double>(m_impl->nr);
  for (std::size_t i = 0; i < m_impl->nr; ++i) out[i] = m_impl->re[i] * scale;
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

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/fft/native_pow2_fft.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#if defined(OpenPFC_ENABLE_HEFFTE)
#include <mpi.h>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/local_fft.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#endif

namespace {

[[nodiscard]] std::vector<double> make_signal(std::size_t length) {
  std::vector<double> result(length);
  for (std::size_t i = 0; i < length; ++i)
    result[i] = std::sin(0.237 * static_cast<double>(i)) +
                0.18 * std::cos(0.31 * static_cast<double>(i));
  return result;
}

[[nodiscard]] std::complex<double> direct_dft(const std::vector<double> &data,
                                              const std::array<int, 3> &n, int kx,
                                              int ky, int kz) {
  std::complex<double> result{0.0, 0.0};
  for (int z = 0; z < n[2]; ++z)
    for (int y = 0; y < n[1]; ++y)
      for (int x = 0; x < n[0]; ++x) {
        const double phase_x = double(kx * x) / n[0];
        const double phase_y = double(ky * y) / n[1];
        const double phase_z = double(kz * z) / n[2];
        const double phase_sum = phase_x + phase_y + phase_z;
        const double phase = -2.0 * std::numbers::pi * phase_sum;
        result += data[(z * n[1] + y) * n[0] + x] * std::polar(1.0, phase);
      }
  return result;
}

} // namespace

TEST_CASE("NativePow2FFT returns the independent real 3D Fourier spectrum",
          "[native_pow2]") {
  // Covers singleton axes and x-fastest half-complex layout.
  for (const std::array<int, 3> dims :
       {std::array<int, 3>{1, 1, 1}, {8, 1, 1}, {4, 2, 4}, {8, 4, 2}}) {
    INFO("dims=" << dims[0] << "x" << dims[1] << "x" << dims[2]);
    pfc::fft::NativePow2FFT fft(dims);
    auto input = make_signal(fft.size_inbox());
    const auto original = input;
    pfc::fft::ComplexVector frequency(fft.size_outbox());
    fft.forward(input, frequency);
    REQUIRE(input == original);
    REQUIRE(fft.get_inbox_bounds().size == dims);
    REQUIRE(fft.get_outbox_bounds().size[0] == dims[0] / 2 + 1);
    for (int z = 0; z < dims[2]; ++z)
      for (int y = 0; y < dims[1]; ++y)
        for (int x = 0; x <= dims[0] / 2; ++x) {
          const auto expected = direct_dft(input, dims, x, y, z);
          const auto packed_index = (z * dims[1] + y) * (dims[0] / 2 + 1) + x;
          const auto actual = frequency[packed_index];
          CHECK(std::abs(actual - expected) < 1e-11 * (1.0 + std::abs(expected)));
        }
    const auto saved_spectrum = frequency;
    pfc::fft::RealVector restored(fft.size_inbox());
    fft.backward(frequency, restored);
    REQUIRE(frequency == saved_spectrum);
    for (std::size_t i = 0; i < input.size(); ++i)
      CHECK(std::abs(restored[i] - input[i]) < 1e-11);
    REQUIRE(fft.get_fft_time() >= 0.0);
    fft.reset_fft_time();
    CHECK(fft.get_fft_time() == 0.0);
  }
}

TEST_CASE("NativePow2FFT validates its deliberately restricted domain",
          "[native_pow2]") {
  auto reject = [](std::array<int, 3> dims) {
    [[maybe_unused]] pfc::fft::NativePow2FFT candidate(dims);
  };
  REQUIRE_THROWS_AS(reject({0, 4, 4}), std::invalid_argument);
  REQUIRE_THROWS_AS(reject({-2, 8, 8}), std::invalid_argument);
  REQUIRE_THROWS_AS(reject({6, 8, 8}), std::invalid_argument);
  REQUIRE_THROWS_AS(reject({8, 3, 8}), std::invalid_argument);
  REQUIRE_THROWS_AS(reject({8, 4, 7}), std::invalid_argument);
  pfc::fft::NativePow2FFT fft({8, 4, 2});
  pfc::fft::RealVector real(fft.size_inbox());
  pfc::fft::RealVector wrong_real(fft.size_inbox() - 1);
  pfc::fft::ComplexVector frequency(fft.size_outbox());
  pfc::fft::ComplexVector wrong_frequency(fft.size_outbox() + 1);
  REQUIRE_THROWS_AS(fft.forward(wrong_real, frequency), std::invalid_argument);
  REQUIRE_THROWS_AS(fft.forward(real, wrong_frequency), std::invalid_argument);
  REQUIRE_THROWS_AS(fft.backward(wrong_frequency, real), std::invalid_argument);
  REQUIRE_THROWS_AS(fft.backward(frequency, wrong_real), std::invalid_argument);
  CHECK(fft.size_workspace() > 0);
  CHECK(fft.get_allocated_memory_bytes() >= 2 * real.size() * sizeof(double));
}

#if defined(OpenPFC_ENABLE_HEFFTE)
TEST_CASE("NativePow2FFT matches LocalFFT and HeFFTe-compatible layout",
          "[native_pow2]") {
  for (const std::array<int, 3> dims :
       {std::array<int, 3>{8, 8, 8}, {32, 8, 4}, {2, 16, 8}}) {
    pfc::fft::NativePow2FFT native(dims);
    pfc::fft::LocalFFT local(dims);
    REQUIRE(native.get_inbox_bounds() == local.get_inbox_bounds());
    REQUIRE(native.get_outbox_bounds() == local.get_outbox_bounds());
    const auto input = make_signal(native.size_inbox());
    pfc::fft::ComplexVector native_fft(native.size_outbox());
    pfc::fft::ComplexVector fftw_fft(local.size_outbox());
    native.forward(input, native_fft);
    local.forward(input, fftw_fft);
    for (std::size_t i = 0; i < native_fft.size(); ++i) {
      const auto scale = 1.0 + std::abs(fftw_fft[i]);
      CHECK(std::abs(native_fft[i] - fftw_fft[i]) < 1e-11 * scale);
    }
    pfc::fft::RealVector restored(input.size());
    native.backward(native_fft, restored);
    for (std::size_t i = 0; i < input.size(); ++i)
      CHECK(std::abs(restored[i] - input[i]) < 1e-11);
  }
}

TEST_CASE("NativePow2FFT stack requires one rank and a power-of-two grid",
          "[native_pow2][MPI]") {
  using Stack = pfc::sim::stacks::SpectralCPUStack;
  const auto selected = pfc::fft::Implementation::NativePow2;
  auto domain = pfc::domain::create({8, 8, 8});
  Stack stack(domain, 0, 1, MPI_COMM_SELF, selected);
  CHECK(stack.fft_choice() == selected);
  CHECK(dynamic_cast<pfc::fft::NativePow2FFT *>(&stack.fft()) != nullptr);
  REQUIRE_THROWS_AS((Stack(domain, 0, 2, MPI_COMM_SELF, selected)),
                    std::invalid_argument);
  REQUIRE_THROWS_AS((Stack(domain, 1, 1, MPI_COMM_SELF, selected)),
                    std::invalid_argument);
  REQUIRE_THROWS_AS((Stack(domain, 0, 1, MPI_COMM_NULL, selected)),
                    std::invalid_argument);
  auto unsupported = pfc::domain::create({8, 6, 8});
  REQUIRE_THROWS_AS((Stack(unsupported, 0, 1, MPI_COMM_SELF, selected)),
                    std::invalid_argument);
  // Do not silently change backend when native preconditions are violated.
  int world_size = 0, world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_size > 1)
    REQUIRE_THROWS_AS(
        (Stack(domain, world_rank, world_size, MPI_COMM_WORLD, selected)),
        std::invalid_argument);
}
#endif

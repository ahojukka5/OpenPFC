// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <openpfc/kernel/fft/fft_interface.hpp>

namespace pfc::fft {

/**
 * @brief Explicit one-rank, CPU, power-of-two real-to-complex 3D FFT.
 *
 * All dimensions (x, y, z) must be positive powers of two (1 is allowed).
 * Input is real with x fastest; complex output reduces x to nx/2+1.
 * Forward is unscaled, backward divides by nx*ny*nz. Both inputs are
 * preserved, all transform scratch is owned and reused, and no FFTW/HeFFTe
 * planning or communication occurs.
 *
 * This implementation is intentionally not a general FFT library: for
 * non-power-of-two grids, distributed decompositions, and GPU execution,
 * select HeFFTe instead. The spectral stack enforces one MPI rank for
 * this backend. A plan is not reentrant; use one instance per concurrent
 * execution. On supported x86 GCC/Clang builds, AVX2/FMA is selected at
 * runtime for batches of independent y/z butterflies; otherwise a scalar
 * implementation is used.
 */
class NativePow2FFT final : public IHostFFT {
public:
  explicit NativePow2FFT(const std::array<int, 3> &size);
  ~NativePow2FFT() override;
  NativePow2FFT(const NativePow2FFT &) = delete;
  NativePow2FFT &operator=(const NativePow2FFT &) = delete;
  NativePow2FFT(NativePow2FFT &&) = delete;
  NativePow2FFT &operator=(NativePow2FFT &&) = delete;

  void forward(const RealVector &in, ComplexVector &out) override;
  void backward(const ComplexVector &in, RealVector &out) override;
  void reset_fft_time() override;
  [[nodiscard]] double get_fft_time() const override;
  [[nodiscard]] std::size_t size_inbox() const override;
  [[nodiscard]] std::size_t size_outbox() const override;
  /// Owned scratch expressed in complex-sized slots.
  [[nodiscard]] std::size_t size_workspace() const override;
  /// Owned vector element storage, including twiddle/swap tables.
  [[nodiscard]] std::size_t get_allocated_memory_bytes() const override;
  [[nodiscard]] Box3i get_inbox_bounds() const override;
  [[nodiscard]] Box3i get_outbox_bounds() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace pfc::fft

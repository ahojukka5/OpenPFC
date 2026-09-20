// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_heat_propagator_hip.hpp
 * @brief HIP implicit-Euler heat propagator (2 device FFTs/step).
 *
 * @details Device twin of `SpectralHeatPropagator`. The multiplier table is
 *          filled on the host by `fill_implicit_euler_symbol` and uploaded
 *          once. Each `step` is a device forward FFT, a complex×real multiply,
 *          and a device inverse FFT — the same 2-FFT implicit Euler as the
 *          CPU spectral Heat3D driver, not the 4-FFT point-wise path.
 */

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)

#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <heat3d/spectral_heat_propagator.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/elementwise_ops_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

namespace heat3d {

/// Device times in milliseconds for one `step`, after the device is idle.
struct SpectralHipPhaseTimes {
  float forward_ms{};
  float multiply_ms{};
  float backward_ms{};
};

class SpectralHeatPropagatorHIP {
public:
  using FFT = pfc::fft::IDeviceFFT<pfc::HIPSpace>;
  using RealField = pfc::data::Field<double, pfc::HIPSpace>;

  SpectralHeatPropagatorHIP(const SpectralHeatPropagatorHIP &) = delete;
  SpectralHeatPropagatorHIP &operator=(const SpectralHeatPropagatorHIP &) = delete;
  SpectralHeatPropagatorHIP(SpectralHeatPropagatorHIP &&) = delete;
  SpectralHeatPropagatorHIP &operator=(SpectralHeatPropagatorHIP &&) = delete;

  /**
   * @param fft Device FFT plan to reuse (borrowed; must outlive the propagator).
   * @param u   Field whose global grid + spacing define the symbol table.
   * @param D   Diffusion coefficient.
   * @param dt  Time-step size.
   */
  SpectralHeatPropagatorHIP(FFT &fft, const RealField &u, double D, double dt)
      : m_fft(fft), m_psi_F(fft.size_outbox()), m_opL(fft.size_outbox()) {
    std::vector<double> host_opL(fft.size_outbox());
    fill_implicit_euler_symbol(host_opL, fft.get_outbox_bounds(), u.global_size(),
                               u.spacing(), D, dt);
    m_opL.copy_from_host(host_opL);
    auto ev = [](hipEvent_t *e, const char *what) {
      const hipError_t err = hipEventCreate(e);
      if (err != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
      }
    };
    ev(&m_ev[0], "hipEventCreate begin");
    ev(&m_ev[1], "hipEventCreate after forward");
    ev(&m_ev[2], "hipEventCreate after multiply");
    ev(&m_ev[3], "hipEventCreate after backward");
  }

  ~SpectralHeatPropagatorHIP() {
    for (hipEvent_t e : m_ev) {
      if (e != nullptr) {
        static_cast<void>(hipEventDestroy(e));
      }
    }
  }

  /** Advance `u` by one implicit-Euler step (1 fwd FFT + 1 inv FFT). */
  void step(RealField &u) {
    u.sync_to_device();
    auto rec = [](hipEvent_t e, const char *what) {
      const hipError_t err = hipEventRecord(e);
      if (err != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
      }
    };
    rec(m_ev[0], "hipEventRecord begin");
    m_fft.forward(u.buffer(), m_psi_F);
    rec(m_ev[1], "hipEventRecord after forward");
    pfc::multiply_complex_real_hip_impl(m_psi_F.data(), m_opL.data(), m_psi_F.data(),
                                        m_psi_F.size());
    rec(m_ev[2], "hipEventRecord after multiply");
    m_fft.backward(m_psi_F, u.buffer());
    rec(m_ev[3], "hipEventRecord after backward");
    u.note_device_write();
  }

  /**
   * @brief Forward / multiply / backward ms for the last `step`.
   *
   * Call after the device has caught up (`hipDeviceSynchronize`).
   */
  [[nodiscard]] SpectralHipPhaseTimes phase_times_ms() const {
    SpectralHipPhaseTimes t{};
    auto span = [](hipEvent_t a, hipEvent_t b, float *ms, const char *what) {
      const hipError_t err = hipEventElapsedTime(ms, a, b);
      if (err != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
      }
    };
    span(m_ev[0], m_ev[1], &t.forward_ms, "hipEventElapsedTime forward");
    span(m_ev[1], m_ev[2], &t.multiply_ms, "hipEventElapsedTime multiply");
    span(m_ev[2], m_ev[3], &t.backward_ms, "hipEventElapsedTime backward");
    return t;
  }

private:
  FFT &m_fft;
  FFT::ComplexBuffer m_psi_F;
  FFT::RealBuffer m_opL;
  hipEvent_t m_ev[4]{};
};

} // namespace heat3d

#endif // OpenPFC_ENABLE_HIP_SPECTRAL

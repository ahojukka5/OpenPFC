// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file vorticity_stream.hpp
 * @brief Periodic 2-D vorticity–streamfunction Navier–Stokes on the CPU
 *        spectral stack (OpenPFC issue #21).
 *
 * @details
 * Prognostic equation
 * \f[
 *   \partial_t\omega+\mathbf u\cdot\nabla\omega=\nu\nabla^2\omega,
 * \f]
 * with the Poisson streamfunction
 * \f[
 *   \nabla^2\psi=-\omega,\qquad
 *   \mathbf u=(\partial_y\psi,-\partial_x\psi).
 * \f]
 * In Fourier space, \f$\hat\psi_{\mathbf k}=\hat\omega_{\mathbf k}/|\mathbf k|^2\f$
 * for \f$\mathbf k\neq 0\f$.
 *
 * **Zero mode.** The Poisson problem determines \f$\psi\f$ only up to a
 * constant. This solver sets \f$\hat\psi(\mathbf 0)=0\f$ (zero-mean
 * streamfunction). Mean vorticity \f$\hat\omega(\mathbf 0)\f$ is a conserved
 * circulation: the viscous symbol \f$L(0)=0\f$ and the Jacobian of a
 * periodic incompressible flow has zero mean, which is enforced by writing
 * \f$\hat N(\mathbf 0)=0\f$ after the dealiased product. The code does
 * **not** project \f$\hat\omega(\mathbf 0)\f$ to zero.
 *
 * **Anti-aliasing.** The Jacobian is quadratic. Orszag's 2/3 rule
 * (`pfc::fft::kspace::fill_two_thirds_mask`) is applied to \f$\hat N\f$.
 * That mask is the existing cubic-safe helper; for a quadratic term it is
 * conservative (3/2 padding would keep more modes). No second dealias
 * abstraction is introduced.
 *
 * **Time integrator.** ETD1 on the viscous symbol \f$L=-\nu|\mathbf k|^2\f$
 * with the dealiased Jacobian as the explicit term:
 * \f$\hat\omega^{n+1}=e^{L\Delta t}\hat\omega^n+\varphi_1(L\Delta t)\hat N^n\f$.
 * For Taylor–Green the Jacobian vanishes, so ETD1 is exact in time for the
 * linear decay. Timestep refinement therefore uses a two-mode nonlinear IC.
 *
 * **HIP.** `GPUSpectralStack` / `SpectralETDOps` implement pointwise
 * \f$N(\psi)\f$ plus a diagonal \f$L(k)\f$. The NS Jacobian needs a Poisson
 * solve and four spectral derivatives, which that pipeline does not expose.
 * This prototype stays on `SpectralCPUStack`. See the example README.
 *
 * This header is the incompressible-flow core a later reduced-MHD or
 * Cahn–Hilliard–Navier–Stokes experiment could call. It is not a generic
 * framework.
 */

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/integrator/etd1_apply.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

namespace ns2d {

struct Params {
  double nu{0.1};
  double dt{0.01};
};

struct Diagnostics {
  double time{0.0};
  /// Mean kinetic-energy density: \f$\frac12\langle u^2+v^2\rangle\f$.
  double ke{0.0};
  /// Mean enstrophy density: \f$\frac12\langle\omega^2\rangle\f$.
  double enstrophy{0.0};
  double max_abs_omega{0.0};
  double max_speed{0.0};
  /// \f$L^\infty\f$ of \f$\partial_x u+\partial_y v\f$ (spectral).
  double div_linf{0.0};
  /// RMS of the same divergence field.
  double div_l2{0.0};
  /// \f$\Delta t\,\max(|u|,|v|)/\min(\Delta x,\Delta y)\f$.
  double cfl{0.0};
  /// Conserved mean vorticity (circulation / area).
  double mean_omega{0.0};
};

/**
 * @brief CPU vorticity–streamfunction stepper bound to a `SpectralCPUStack`.
 *
 * Prognostic real field is `stack.u()` (vorticity). The stack must outlive
 * this object. Extra FFT work arrays are owned here.
 */
class VorticityStreamCPU {
public:
  using Complex = std::complex<double>;

  VorticityStreamCPU(pfc::sim::stacks::SpectralCPUStack &stack, Params params)
      : m_stack(&stack), m_fft(&stack.fft()), m_params(params),
        m_inbox(stack.fft().get_inbox_bounds()),
        m_outbox(stack.fft().get_outbox_bounds()),
        m_gsize(stack.u().global_size()), m_spacing(stack.u().spacing()) {
    if (params.nu < 0.0 || params.dt <= 0.0) {
      throw std::invalid_argument("ns2d: nu >= 0 and dt > 0 are required");
    }
    if (m_gsize[2] != 1) {
      throw std::invalid_argument(
          "ns2d: expected a one-cell z slab (Nx x Ny x 1)");
    }
    const std::size_t in_n = m_fft->size_inbox();
    const std::size_t out_n = m_fft->size_outbox();
    m_u.assign(in_n, 0.0);
    m_v.assign(in_n, 0.0);
    m_wx.assign(in_n, 0.0);
    m_wy.assign(in_n, 0.0);
    m_n.assign(in_n, 0.0);
    m_div.assign(in_n, 0.0);
    m_omega_hat.assign(out_n, Complex{});
    m_psi_hat.assign(out_n, Complex{});
    m_n_hat.assign(out_n, Complex{});
    m_tmp_hat.assign(out_n, Complex{});
    m_L.assign(out_n, 0.0);
    m_exp_Ldt.assign(out_n, 0.0);
    m_phi1.assign(out_n, 0.0);
    m_mask.assign(out_n, 0.0);
    m_kx_odd.assign(out_n, 0.0);
    m_ky_odd.assign(out_n, 0.0);
    m_inv_k2.assign(out_n, 0.0);

    pfc::fft::kspace::fill_two_thirds_mask(m_outbox, m_gsize, m_spacing,
                                           m_mask.data(), m_mask.size());
    pfc::fft::kspace::for_each_kpoint(
        m_outbox, m_gsize, m_spacing,
        [&](std::size_t idx, double kx, double ky, double /*kz*/, int i, int j,
            int k) {
          const double kx_odd =
              pfc::fft::kspace::is_nyquist_index(i, m_gsize[0]) ? 0.0 : kx;
          const double ky_odd =
              pfc::fft::kspace::is_nyquist_index(j, m_gsize[1]) ? 0.0 : ky;
          m_kx_odd[idx] = kx_odd;
          m_ky_odd[idx] = ky_odd;
          const double k2 = kx * kx + ky * ky; // kz = 0 on the 2-D slab
          m_L[idx] = -m_params.nu * k2;
          const bool dc = (i == 0 && j == 0 && k == 0);
          m_inv_k2[idx] = (dc || k2 == 0.0) ? 0.0 : 1.0 / k2;
          if (dc) m_has_dc = true;
        });
    pfc::integrator::fill_spectral_exp_coeffs(std::span<const double>(m_L),
                                              m_params.dt,
                                              std::span<double>(m_exp_Ldt),
                                              std::span<double>(m_phi1));
  }

  [[nodiscard]] Params params() const noexcept { return m_params; }
  [[nodiscard]] double time() const noexcept { return m_time; }
  [[nodiscard]] pfc::data::Field<double> &omega() noexcept { return m_stack->u(); }
  [[nodiscard]] const pfc::data::Field<double> &omega() const noexcept {
    return m_stack->u();
  }
  [[nodiscard]] const std::vector<double> &u() const noexcept { return m_u; }
  [[nodiscard]] const std::vector<double> &v() const noexcept { return m_v; }
  [[nodiscard]] const std::vector<double> &psi_inv_k2() const noexcept {
    return m_inv_k2;
  }
  [[nodiscard]] const std::vector<double> &dealias_mask() const noexcept {
    return m_mask;
  }
  [[nodiscard]] bool rank_owns_zero_mode() const noexcept { return m_has_dc; }

  template <class Fn> void initialize_omega(Fn &&fn) {
    m_stack->u().apply(std::forward<Fn>(fn));
    m_time = 0.0;
    recover_velocity_from_omega();
  }

  /// One ETD1 step. Updates vorticity, velocity, and \f$t\f$.
  void step() {
    auto &w = m_stack->u();
    m_fft->forward(w.vec(), m_omega_hat);
    poisson_psi();
    invert_ik(m_psi_hat, m_ky_odd, +1.0, m_u); // u =  ∂y ψ
    invert_ik(m_psi_hat, m_kx_odd, -1.0, m_v); // v = -∂x ψ
    invert_ik(m_omega_hat, m_kx_odd, +1.0, m_wx);
    invert_ik(m_omega_hat, m_ky_odd, +1.0, m_wy);
    const std::size_t n = m_n.size();
    for (std::size_t i = 0; i < n; ++i) {
      m_n[i] = -(m_u[i] * m_wx[i] + m_v[i] * m_wy[i]);
    }
    m_fft->forward(m_n, m_n_hat);
    for (std::size_t i = 0; i < m_n_hat.size(); ++i) {
      m_n_hat[i] *= m_mask[i];
    }
    zero_dc(m_n_hat);
    pfc::integrator::apply_etd1_update(std::span<const double>(m_exp_Ldt),
                                       std::span<const double>(m_phi1),
                                       std::span<const Complex>(m_omega_hat),
                                       std::span<const Complex>(m_n_hat),
                                       std::span<Complex>(m_tmp_hat));
    m_omega_hat.swap(m_tmp_hat);
    m_fft->backward(m_omega_hat, w.vec());
    m_time += m_params.dt;
  }

  /// Rebuild \f$(u,v)\f$ from the current vorticity (no time advance).
  void recover_velocity_from_omega() {
    m_fft->forward(m_stack->u().vec(), m_omega_hat);
    poisson_psi();
    invert_ik(m_psi_hat, m_ky_odd, +1.0, m_u);
    invert_ik(m_psi_hat, m_kx_odd, -1.0, m_v);
  }

  [[nodiscard]] Diagnostics diagnostics(MPI_Comm comm) {
    recover_velocity_from_omega();
    spectral_divergence();

    double ke_sum = 0.0;
    double ens_sum = 0.0;
    double w_sum = 0.0;
    double max_w = 0.0;
    double max_speed = 0.0;
    double div_linf = 0.0;
    double div_sq = 0.0;
    const auto &wfield = m_stack->u();
    wfield.for_each_owned([&](int i, int j, int k) {
      const std::size_t c = real_idx(i, j, k);
      const double w = wfield(i, j, k);
      const double uu = m_u[c];
      const double vv = m_v[c];
      ke_sum += 0.5 * (uu * uu + vv * vv);
      ens_sum += 0.5 * w * w;
      w_sum += w;
      max_w = std::max(max_w, std::abs(w));
      max_speed = std::max(max_speed, std::max(std::abs(uu), std::abs(vv)));
      const double d = std::abs(m_div[c]);
      div_linf = std::max(div_linf, d);
      div_sq += m_div[c] * m_div[c];
    });

    double g_ke = 0.0, g_ens = 0.0, g_w = 0.0, g_divsq = 0.0;
    double g_maxw = 0.0, g_maxspd = 0.0, g_divinf = 0.0;
    MPI_Allreduce(&ke_sum, &g_ke, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&ens_sum, &g_ens, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&w_sum, &g_w, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&div_sq, &g_divsq, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&max_w, &g_maxw, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&max_speed, &g_maxspd, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&div_linf, &g_divinf, 1, MPI_DOUBLE, MPI_MAX, comm);

    const double ncells = static_cast<double>(m_gsize[0]) *
                          static_cast<double>(m_gsize[1]) *
                          static_cast<double>(m_gsize[2]);
    const double dx = std::min(m_spacing[0], m_spacing[1]);
    Diagnostics d;
    d.time = m_time;
    d.ke = g_ke / ncells;
    d.enstrophy = g_ens / ncells;
    d.max_abs_omega = g_maxw;
    d.max_speed = g_maxspd;
    d.div_linf = g_divinf;
    d.div_l2 = std::sqrt(g_divsq / ncells);
    d.cfl = m_params.dt * g_maxspd / dx;
    d.mean_omega = g_w / ncells;
    return d;
  }

  [[nodiscard]] double linf_omega_error(MPI_Comm comm,
                                        const auto &exact) const {
    double local = 0.0;
    m_stack->u().for_each_owned([&](double x, double y, double /*z*/, double w) {
      local = std::max(local, std::abs(w - exact(x, y)));
    });
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global;
  }

private:
  [[nodiscard]] std::size_t real_idx(int i, int j, int k) const noexcept {
    const auto nx = static_cast<std::size_t>(m_inbox.size[0]);
    const auto ny = static_cast<std::size_t>(m_inbox.size[1]);
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * nx +
           static_cast<std::size_t>(k) * nx * ny;
  }

  void poisson_psi() {
    for (std::size_t i = 0; i < m_psi_hat.size(); ++i) {
      m_psi_hat[i] = m_omega_hat[i] * m_inv_k2[i];
    }
  }

  void invert_ik(const std::vector<Complex> &in, const std::vector<double> &kodd,
                 double sign, std::vector<double> &out) {
    for (std::size_t i = 0; i < in.size(); ++i) {
      m_tmp_hat[i] = in[i] * Complex(0.0, sign * kodd[i]);
    }
    m_fft->backward(m_tmp_hat, out);
  }

  void zero_dc(std::vector<Complex> &hat) {
    if (!m_has_dc) return;
    pfc::fft::kspace::for_each_kpoint(
        m_outbox, m_gsize, m_spacing,
        [&](std::size_t idx, double, double, double, int i, int j, int k) {
          if (i == 0 && j == 0 && k == 0) hat[idx] = Complex{0.0, 0.0};
        });
  }

  void spectral_divergence() {
    m_fft->forward(m_u, m_tmp_hat);
    for (std::size_t i = 0; i < m_tmp_hat.size(); ++i) {
      m_n_hat[i] = m_tmp_hat[i] * Complex(0.0, m_kx_odd[i]);
    }
    m_fft->forward(m_v, m_tmp_hat);
    for (std::size_t i = 0; i < m_tmp_hat.size(); ++i) {
      m_n_hat[i] += m_tmp_hat[i] * Complex(0.0, m_ky_odd[i]);
    }
    m_fft->backward(m_n_hat, m_div);
  }

  pfc::sim::stacks::SpectralCPUStack *m_stack{nullptr};
  pfc::fft::IHostFFT *m_fft{nullptr};
  Params m_params{};
  double m_time{0.0};
  pfc::fft::Box3i m_inbox{};
  pfc::fft::Box3i m_outbox{};
  pfc::Int3 m_gsize{};
  pfc::Real3 m_spacing{};
  bool m_has_dc{false};

  std::vector<double> m_u, m_v, m_wx, m_wy, m_n, m_div;
  std::vector<Complex> m_omega_hat, m_psi_hat, m_n_hat, m_tmp_hat;
  std::vector<double> m_L, m_exp_Ldt, m_phi1, m_mask, m_kx_odd, m_ky_odd,
      m_inv_k2;
};

[[nodiscard]] inline pfc::Domain make_periodic_square(int n) {
  if (n < 4) {
    throw std::invalid_argument("ns2d: N >= 4 is required");
  }
  const double dx = 2.0 * pfc::pi / static_cast<double>(n);
  return pfc::domain::create(pfc::GridSize({n, n, 1}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({dx, dx, 2.0 * pfc::pi}));
}

} // namespace ns2d

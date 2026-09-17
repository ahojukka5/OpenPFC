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
 * with \f$\nabla^2\psi=-\omega\f$ and
 * \f$\mathbf u=(\partial_y\psi,-\partial_x\psi)\f$.
 *
 * **Zero mode.** \f$\hat\psi(\mathbf 0)=0\f$. Mean vorticity is conserved:
 * \f$L(0)=0\f$ and \f$\hat N(\mathbf 0)=0\f$. The code does not project
 * \f$\hat\omega(\mathbf 0)\f$ to zero.
 *
 * **Anti-aliasing.** Orszag 2/3 is applied to the *state* as well as to
 * \f$\hat N\f$. The initial condition and every IFRK4 stage are projected
 * onto the retained band *before* real-space products. Masking only
 * \f$\hat N\f$ after the product is not enough if the state still holds
 * modes above the cutoff: those modes alias into retained wavenumbers
 * before the mask. 3/2 padding is not used.
 *
 * **Time integrator.** Integrating-factor RK4 on
 * \f$L=-\nu|\mathbf k|^2\f$ with the dealiased Jacobian as \f$N\f$.
 * The linear viscous piece is advanced by \f$e^{L\Delta t}\f$; the
 * nonlinear piece is classical RK4 in the integrating-factor variable.
 * When \f$N\equiv 0\f$ (Taylor–Green) the step is exact. When
 * \f$L\equiv 0\f$ it reduces to RK4, whose stability region includes
 * the imaginary axis up to \f$|z|\approx 2\sqrt{2}\f$. ETDRK2/Heun does
 * not: \f$|1+i\omega+(\mathrm{i}\omega)^2/2|>1\f$ for every
 * \f$\omega\neq 0\f$. Coefficients are only \f$e^{L\Delta t}\f$ and
 * \f$e^{L\Delta t/2}\f$ — no extra generic \f$\varphi_k\f$ framework.
 *
 * **HIP.** `GPUSpectralStack` / `SpectralETDOps` do not compose Poisson
 * plus four spectral derivatives. CPU only; see the example README.
 */

#include <algorithm>
#include <array>
#include <cmath>
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
  double div_linf{0.0};
  double div_l2{0.0};
  /// \f$\Delta t\,\max(|u|,|v|)/\min(\Delta x,\Delta y)\f$.
  double cfl{0.0};
  double mean_omega{0.0};
};

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
    m_tmp_hat.assign(out_n, Complex{});
    m_stage_hat.assign(out_n, Complex{});
    m_n1_hat.assign(out_n, Complex{});
    m_n2_hat.assign(out_n, Complex{});
    m_n3_hat.assign(out_n, Complex{});
    m_n4_hat.assign(out_n, Complex{});
    m_L.assign(out_n, 0.0);
    m_exp_Ldt.assign(out_n, 0.0);
    m_exp_half.assign(out_n, 0.0);
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
          const double k2 = kx * kx + ky * ky;
          m_L[idx] = -m_params.nu * k2;
          m_exp_Ldt[idx] = std::exp(m_L[idx] * m_params.dt);
          m_exp_half[idx] = std::exp(m_L[idx] * 0.5 * m_params.dt);
          const bool dc = (i == 0 && j == 0 && k == 0);
          m_inv_k2[idx] = (dc || k2 == 0.0) ? 0.0 : 1.0 / k2;
          if (dc) m_has_dc = true;
        });
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
  [[nodiscard]] const std::vector<Complex> &last_n_hat() const noexcept {
    return m_n1_hat;
  }
  [[nodiscard]] bool rank_owns_zero_mode() const noexcept { return m_has_dc; }

  void project_hat(std::vector<Complex> &hat) const {
    for (std::size_t i = 0; i < hat.size(); ++i) hat[i] *= m_mask[i];
  }

  template <class Fn> void initialize_omega(Fn &&fn) {
    m_stack->u().apply(std::forward<Fn>(fn));
    m_time = 0.0;
    project_current_state();
    recover_velocity_from_omega();
  }

  /// FFT, 2/3-project, inverse FFT. Public for aliasing tests.
  void project_current_state() {
    auto &w = m_stack->u();
    m_fft->forward(w.vec(), m_omega_hat);
    project_hat(m_omega_hat);
    m_fft->backward(m_omega_hat, w.vec());
  }

  /**
   * @brief Jacobian \f$\hat N\f$ of the current real vorticity.
   *
   * If @p project_inputs is true, the spectrum is 2/3-masked before
   * products (the production path). If false, high-k content is left in
   * so a test can show aliasing into retained modes.
   */
  void jacobian_from_current(bool project_inputs) {
    m_fft->forward(m_stack->u().vec(), m_omega_hat);
    if (project_inputs) project_hat(m_omega_hat);
    nonlinear_from_hat(m_omega_hat, m_n1_hat);
  }

  /// Magnitude of local outbox mode (ix,iy,0), or -1 if this rank does not own it.
  [[nodiscard]] double mode_abs(const std::vector<Complex> &hat, int ix,
                                int iy) const {
    double local = -1.0;
    pfc::fft::kspace::for_each_kpoint(
        m_outbox, m_gsize, m_spacing,
        [&](std::size_t idx, double, double, double, int i, int j, int k) {
          if (i == ix && j == iy && k == 0) local = std::abs(hat[idx]);
        });
    return local;
  }

  void step() {
    auto &w = m_stack->u();
    const double dt = m_params.dt;
    m_fft->forward(w.vec(), m_omega_hat);
    project_hat(m_omega_hat);

    nonlinear_from_hat(m_omega_hat, m_n1_hat);

    for (std::size_t i = 0; i < m_omega_hat.size(); ++i) {
      m_stage_hat[i] =
          m_exp_half[i] * (m_omega_hat[i] + (0.5 * dt) * m_n1_hat[i]);
    }
    project_hat(m_stage_hat);
    nonlinear_from_hat(m_stage_hat, m_n2_hat);

    for (std::size_t i = 0; i < m_omega_hat.size(); ++i) {
      m_stage_hat[i] =
          m_exp_half[i] * m_omega_hat[i] + (0.5 * dt) * m_n2_hat[i];
    }
    project_hat(m_stage_hat);
    nonlinear_from_hat(m_stage_hat, m_n3_hat);

    for (std::size_t i = 0; i < m_omega_hat.size(); ++i) {
      m_stage_hat[i] =
          m_exp_Ldt[i] * m_omega_hat[i] + dt * m_exp_half[i] * m_n3_hat[i];
    }
    project_hat(m_stage_hat);
    nonlinear_from_hat(m_stage_hat, m_n4_hat);

    const double dt6 = dt / 6.0;
    for (std::size_t i = 0; i < m_omega_hat.size(); ++i) {
      m_omega_hat[i] =
          m_exp_Ldt[i] * m_omega_hat[i] +
          dt6 * (m_exp_Ldt[i] * m_n1_hat[i] +
                 2.0 * m_exp_half[i] * (m_n2_hat[i] + m_n3_hat[i]) +
                 m_n4_hat[i]);
    }
    project_hat(m_omega_hat);
    m_fft->backward(m_omega_hat, w.vec());
    m_time += dt;
  }

  void recover_velocity_from_omega() {
    m_fft->forward(m_stack->u().vec(), m_omega_hat);
    project_hat(m_omega_hat);
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

  void nonlinear_from_hat(const std::vector<Complex> &omega_hat,
                          std::vector<Complex> &n_hat) {
    for (std::size_t i = 0; i < omega_hat.size(); ++i) {
      m_psi_hat[i] = omega_hat[i] * m_inv_k2[i];
    }
    invert_ik(m_psi_hat, m_ky_odd, +1.0, m_u);
    invert_ik(m_psi_hat, m_kx_odd, -1.0, m_v);
    invert_ik(omega_hat, m_kx_odd, +1.0, m_wx);
    invert_ik(omega_hat, m_ky_odd, +1.0, m_wy);
    for (std::size_t i = 0; i < m_n.size(); ++i) {
      m_n[i] = -(m_u[i] * m_wx[i] + m_v[i] * m_wy[i]);
    }
    m_fft->forward(m_n, n_hat);
    project_hat(n_hat);
    zero_dc(n_hat);
  }

  void spectral_divergence() {
    m_fft->forward(m_u, m_tmp_hat);
    for (std::size_t i = 0; i < m_tmp_hat.size(); ++i) {
      m_n1_hat[i] = m_tmp_hat[i] * Complex(0.0, m_kx_odd[i]);
    }
    m_fft->forward(m_v, m_tmp_hat);
    for (std::size_t i = 0; i < m_tmp_hat.size(); ++i) {
      m_n1_hat[i] += m_tmp_hat[i] * Complex(0.0, m_ky_odd[i]);
    }
    m_fft->backward(m_n1_hat, m_div);
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
  std::vector<Complex> m_omega_hat, m_psi_hat, m_tmp_hat, m_stage_hat;
  std::vector<Complex> m_n1_hat, m_n2_hat, m_n3_hat, m_n4_hat;
  std::vector<double> m_L, m_exp_Ldt, m_exp_half, m_mask, m_kx_odd, m_ky_odd,
      m_inv_k2;
};

[[nodiscard]] inline pfc::Domain make_slab(int n, double length) {
  if (n < 4) {
    throw std::invalid_argument("ns2d: N >= 4 is required");
  }
  if (length <= 0.0) {
    throw std::invalid_argument("ns2d: box length must be positive");
  }
  const double dx = length / static_cast<double>(n);
  return pfc::domain::create(pfc::GridSize({n, n, 1}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({dx, dx, length}));
}

[[nodiscard]] inline pfc::Domain make_twopi_slab(int n) {
  return make_slab(n, 2.0 * pfc::pi);
}

[[nodiscard]] inline pfc::Domain make_unit_slab(int n) { return make_slab(n, 1.0); }

/// Back-compat name used by Taylor–Green tests.
[[nodiscard]] inline pfc::Domain make_periodic_square(int n) {
  return make_twopi_slab(n);
}

} // namespace ns2d

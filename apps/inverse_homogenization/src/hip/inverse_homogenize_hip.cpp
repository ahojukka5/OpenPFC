// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "openpfc_inverse_homogenize_hip requires HIP spectral (rocFFT HeFFTe)"
#endif

/**
 * HIP inverse homogenization: device Green/FFT elasticity, host
 * Allen–Cahn on the downloaded sensitivity. Same loop as the CPU driver,
 * including rotating-square auxetic seeds and MPI-IO design snapshots.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <inverse_homogenization/auxetic_geometry.hpp>
#include <inverse_homogenization/field_output.hpp>
#include <inverse_homogenization/inverse_checkpoint.hpp>
#include <inverse_homogenization/inverse_convergence.hpp>
#include <inverse_homogenization/manufacturability.hpp>
#include <inverse_homogenization/material_report.hpp>
#include <inverse_homogenization/phase_field_inverse.hpp>
#include <inverse_homogenization/simp_penalty.hpp>
#include <inverse_homogenization/spinodal_generator.hpp>
#include <inverse_homogenization/target_io.hpp>
#include <inverse_homogenization/yang_reentrant.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <openpfc_apps/homogenization_hip.hpp>

namespace {

using RealField = pfc::data::Field<double>;
using ComplexField = pfc::data::Field<std::complex<double>>;
using FFT = pfc::fft::IDeviceFFT<pfc::HIPSpace>;

void report_hbm(int rank, int nproc, MPI_Comm comm, long long n_global,
                std::size_t known_owned_bytes) {
  std::size_t free_b = 0, total_b = 0;
  const hipError_t err = hipMemGetInfo(&free_b, &total_b);
  if (err != hipSuccess) {
    if (rank == 0)
      std::cerr << "hipMemGetInfo failed: " << hipGetErrorString(err) << '\n';
    return;
  }
  const unsigned long long used = static_cast<unsigned long long>(total_b - free_b);
  const unsigned long long free_ull = static_cast<unsigned long long>(free_b);
  const unsigned long long total_ull = static_cast<unsigned long long>(total_b);
  unsigned long long used_sum = 0, used_max = 0, free_min = 0, total_gcd = 0;
  MPI_Reduce(&used, &used_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&used, &used_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, comm);
  MPI_Reduce(&free_ull, &free_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, comm);
  MPI_Reduce(&total_ull, &total_gcd, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, comm);
  if (rank == 0) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    std::cout << std::setprecision(6) << "HIP_MEM ranks=" << nproc
              << " used_sum_gib=" << used_sum / gib
              << " used_max_gib=" << used_max / gib
              << " free_min_gib=" << free_min / gib
              << " total_gcd_gib=" << total_gcd / gib
              << " bytes_per_cell=" << used_sum / static_cast<double>(n_global)
              << " known_owned_h_bytes=" << known_owned_bytes << '\n'
              << std::flush;
  }
}

struct Config {
  int nx{16}, ny{16}, nz{1};
  double dx{1.0};
  double E_solid{1.0}, nu_solid{0.3};
  double E_void{0.02}, nu_void{0.3};
  std::string target{"auxetic"};
  double E_target{0.4}, nu_target{0.4};
  double volume{0.5};
  double lambda_volume{1.0};
  double lambda_reg{0.05};
  double epsilon{2.0};
  double dt{0.1};
  int steps{5000};
  int continuation_steps{300};
  int conv_window{20};
  int verify_steps{100};
  double tol_design{1e-4};
  double tol_objective{1e-6};
  double tol_tensor{1e-4};
  std::string init{"rotating-squares"};
  double init_volume{0.55};
  double init_half{0.200};
  double init_angle{0.45};
  unsigned seed{1};
  std::string csv{};
  std::string run_id{"inverse2d"};
  int normalize{1};
  double max_delta{0.05};
  int project_volume{1};
  double simp{1.0};
  double simp_end{-1.0};
  double lambda_reg_end{-1.0};
  double w12{1.0};
  double init_amp{0.12};
  int n_el_iter{200};
  int ch_steps{80};
  double ch_kappa{1.0};
  double ch_dt{0.2};
  std::string load_bin{};
  std::string C_target_file{};
  std::string checkpoint_dir{};
  std::string restart_dir{};
  int stop_after{0};
  pfc::apps::inverse::FieldOutputConfig fields{};
};

void usage(std::ostream &os, const char *exe) {
  os << "Usage: " << exe << " [--key=value]...\n"
     << "  HIP inverse homogenization (device Green, host Allen-Cahn).\n"
     << "  --nx --ny --nz --dx --target isotropic|auxetic|file\n"
     << "  --C-target-file=PATH 6x6 Voigt text\n"
     << "  --init rotating-squares|noise|uniform|spinodal|yang-a3\n"
     << "  --load-bin=PATH Fortran float64 brick --csv --dump-dir\n"
     << "  --checkpoint-dir --restart  continue the same frozen problem\n"
     << "  --stop-after=N              checkpoint running state and exit\n"
     << "  --continuation-steps --max-steps|--steps --conv-window\n"
     << "  --verify-convergence-steps --tol-design --tol-objective --tol-tensor\n";
}

bool parse_double(std::string_view v, double &out) {
  try {
    std::size_t n = 0;
    out = std::stod(std::string(v), &n);
    return n == v.size();
  } catch (...) {
    return false;
  }
}
bool parse_int(std::string_view v, int &out) {
  try {
    std::size_t n = 0;
    out = std::stoi(std::string(v), &n);
    return n == v.size();
  } catch (...) {
    return false;
  }
}

bool parse_args(int argc, char **argv, Config &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view tok(argv[i]);
    if (tok == "--help" || tok == "-h") return false;
    const auto eq = tok.find('=');
    if (!tok.starts_with("--") || eq == std::string_view::npos) return false;
    const auto key = tok.substr(2, eq - 2);
    const auto val = tok.substr(eq + 1);
    bool ok = true;
    if (key == "nx")
      ok = parse_int(val, cfg.nx) && cfg.nx > 0;
    else if (key == "ny")
      ok = parse_int(val, cfg.ny) && cfg.ny > 0;
    else if (key == "nz")
      ok = parse_int(val, cfg.nz) && cfg.nz > 0;
    else if (key == "dx")
      ok = parse_double(val, cfg.dx) && cfg.dx > 0.0;
    else if (key == "E-solid")
      ok = parse_double(val, cfg.E_solid);
    else if (key == "nu-solid")
      ok = parse_double(val, cfg.nu_solid);
    else if (key == "E-void")
      ok = parse_double(val, cfg.E_void);
    else if (key == "nu-void")
      ok = parse_double(val, cfg.nu_void);
    else if (key == "target")
      cfg.target = std::string(val);
    else if (key == "E-target")
      ok = parse_double(val, cfg.E_target);
    else if (key == "nu-target")
      ok = parse_double(val, cfg.nu_target);
    else if (key == "volume")
      ok = parse_double(val, cfg.volume);
    else if (key == "lambda-volume")
      ok = parse_double(val, cfg.lambda_volume);
    else if (key == "lambda-reg")
      ok = parse_double(val, cfg.lambda_reg);
    else if (key == "epsilon")
      ok = parse_double(val, cfg.epsilon) && cfg.epsilon > 0.0;
    else if (key == "dt")
      ok = parse_double(val, cfg.dt) && cfg.dt > 0.0;
    else if (key == "steps" || key == "max-steps")
      ok = parse_int(val, cfg.steps) && cfg.steps >= 0;
    else if (key == "continuation-steps")
      ok = parse_int(val, cfg.continuation_steps) && cfg.continuation_steps >= 0;
    else if (key == "conv-window")
      ok = parse_int(val, cfg.conv_window) && cfg.conv_window > 0;
    else if (key == "verify-convergence-steps")
      ok = parse_int(val, cfg.verify_steps) && cfg.verify_steps >= 0;
    else if (key == "tol-design")
      ok = parse_double(val, cfg.tol_design) && cfg.tol_design > 0.0;
    else if (key == "tol-objective")
      ok = parse_double(val, cfg.tol_objective) && cfg.tol_objective > 0.0;
    else if (key == "tol-tensor")
      ok = parse_double(val, cfg.tol_tensor) && cfg.tol_tensor > 0.0;
    else if (key == "init")
      cfg.init = std::string(val);
    else if (key == "init-volume")
      ok = parse_double(val, cfg.init_volume);
    else if (key == "init-half")
      ok = parse_double(val, cfg.init_half) && cfg.init_half > 0.0;
    else if (key == "init-angle")
      ok = parse_double(val, cfg.init_angle);
    else if (key == "seed") {
      int s = 1;
      ok = parse_int(val, s);
      cfg.seed = static_cast<unsigned>(s);
    } else if (key == "csv")
      cfg.csv = std::string(val);
    else if (key == "run-id")
      cfg.run_id = std::string(val);
    else if (key == "normalize")
      ok = parse_int(val, cfg.normalize);
    else if (key == "max-delta")
      ok = parse_double(val, cfg.max_delta) && cfg.max_delta >= 0.0;
    else if (key == "project-volume")
      ok = parse_int(val, cfg.project_volume);
    else if (key == "simp")
      ok = parse_double(val, cfg.simp) && cfg.simp >= 1.0;
    else if (key == "simp-end")
      ok = parse_double(val, cfg.simp_end) && cfg.simp_end >= 1.0;
    else if (key == "lambda-reg-end")
      ok = parse_double(val, cfg.lambda_reg_end) && cfg.lambda_reg_end >= 0.0;
    else if (key == "W-12")
      ok = parse_double(val, cfg.w12) && cfg.w12 >= 0.0;
    else if (key == "init-amp")
      ok = parse_double(val, cfg.init_amp) && cfg.init_amp >= 0.0;
    else if (key == "n-el-iter")
      ok = parse_int(val, cfg.n_el_iter) && cfg.n_el_iter > 0;
    else if (key == "ch-steps")
      ok = parse_int(val, cfg.ch_steps) && cfg.ch_steps >= 0;
    else if (key == "ch-kappa")
      ok = parse_double(val, cfg.ch_kappa) && cfg.ch_kappa > 0.0;
    else if (key == "ch-dt")
      ok = parse_double(val, cfg.ch_dt) && cfg.ch_dt > 0.0;
    else if (key == "load-bin")
      cfg.load_bin = std::string(val);
    else if (key == "C-target-file")
      cfg.C_target_file = std::string(val);
    else if (key == "dump-dir")
      cfg.fields.dir = std::string(val);
    else if (key == "dump-every")
      ok = parse_int(val, cfg.fields.every) && cfg.fields.every > 0;
    else if (key == "checkpoint-dir")
      cfg.checkpoint_dir = std::string(val);
    else if (key == "restart")
      cfg.restart_dir = std::string(val);
    else if (key == "stop-after")
      ok = parse_int(val, cfg.stop_after) && cfg.stop_after >= 0;
    else
      return false;
    if (!ok) return false;
  }
  if (cfg.target != "isotropic" && cfg.target != "auxetic" && cfg.target != "file")
    return false;
  if (cfg.target == "file" && cfg.C_target_file.empty()) return false;
  if (cfg.init != "uniform" && cfg.init != "noise" &&
      cfg.init != "rotating-squares" && cfg.init != "spinodal" &&
      cfg.init != "yang-a3")
    return false;
  return true;
}

pfc::apps::Voigt6 make_target(const Config &cfg) {
  if (cfg.target == "file") {
    pfc::apps::Voigt6 C;
    if (!pfc::apps::inverse::load_voigt6_file(cfg.C_target_file, C))
      throw std::runtime_error("C-target-file unreadable");
    return C;
  }
  const double nu =
      (cfg.target == "auxetic") ? -std::abs(cfg.nu_target) : cfg.nu_target;
  return pfc::apps::voigt_from_stiffness(
      pfc::solvers::Stiffness::isotropic(cfg.E_target, nu));
}

void spectral_laplacian_hip(const pfc::Domain &domain, FFT &fft, const RealField &h,
                            ComplexField &hat, RealField &lap,
                            FFT::RealBuffer &d_real, FFT::ComplexBuffer &d_hat) {
  d_real.copy_from_host(h.data(), h.size());
  fft.forward(d_real, d_hat);
  d_hat.copy_to_host(hat.data(), hat.size());
  pfc::fft::kspace::for_each_kpoint(
      fft.get_outbox_bounds(), domain,
      [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
        hat.data()[idx] *= pfc::fft::kspace::k_laplacian_value(kx, ky, kz);
      });
  hat.note_host_write();
  d_hat.copy_from_host(hat.data(), hat.size());
  fft.backward(d_hat, d_real);
  d_real.copy_to_host(lap.data(), lap.size());
  lap.note_host_write();
}

double mean_value(const RealField &h, MPI_Comm comm, double n_global) {
  double local = 0.0;
  const double *p = h.data();
  for (std::size_t i = 0; i < h.size(); ++i) local += p[i];
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global / n_global;
}

void project_mean(RealField &h, double target, MPI_Comm comm, double n_global) {
  for (int it = 0; it < 6; ++it) {
    const double vf = mean_value(h, comm, n_global);
    const double shift = target - vf;
    if (std::abs(shift) < 1.0e-12) break;
    double *p = h.data();
    for (std::size_t i = 0; i < h.size(); ++i)
      p[i] = std::min(1.0, std::max(0.0, p[i] + shift));
    h.note_host_write();
  }
}

pfc::apps::inverse::InverseStepReport
ac_step(pfc::apps::PeriodicHomogenizerHIP &hom, const pfc::Domain &domain, FFT &fft,
        RealField &h, const pfc::apps::inverse::InverseSpec &spec, ComplexField &hat,
        RealField &lap, RealField &dJdh, RealField &g, RealField &penalized,
        FFT::RealBuffer &d_real, FFT::ComplexBuffer &d_hat, MPI_Comm comm) {
  pfc::apps::inverse::InverseStepReport out;
  const std::size_t n_local = h.size();
  const auto gs = h.global_size();
  const double n_global = static_cast<double>(gs[0]) * static_cast<double>(gs[1]) *
                          static_cast<double>(gs[2]);
  const double vf0 = mean_value(h, comm, n_global);
  const double dv = vf0 - spec.volume_target;
  out.volume_accepted = vf0;
  out.J_volume = spec.lambda_volume * dv * dv;

  const RealField *h_el = &h;
  if (spec.simp_p != 1.0) {
    double *pp = penalized.data();
    const double *hd = h.data();
    for (std::size_t i = 0; i < n_local; ++i)
      pp[i] = pfc::apps::inverse::simp_density(hd[i], spec.simp_p);
    penalized.note_host_write();
    h_el = &penalized;
  }
  const auto r = hom.compute(*h_el);
  out.elasticity_converged = r.all_converged();
  out.J_tensor = pfc::apps::tensor_mismatch(r.stiffness, spec.C_target, spec.W);
  out.C = r.stiffness.symmetrized();
  out.C11 = out.C(0, 0);
  out.C12 = out.C(0, 1);
  out.C_fro = out.C.frobenius_norm();
  hom.objective_sensitivity(*h_el, spec.C_target, spec.W, dJdh);
  if (spec.simp_p != 1.0) {
    double *dj = dJdh.data();
    const double *hd = h.data();
    for (std::size_t i = 0; i < n_local; ++i)
      dj[i] *= pfc::apps::inverse::simp_chain(hd[i], spec.simp_p);
    dJdh.note_host_write();
  }
  if (spec.lambda_reg != 0.0)
    spectral_laplacian_hip(domain, fft, h, hat, lap, d_real, d_hat);
  else
    std::fill(lap.vec().begin(), lap.vec().end(), 0.0);

  double local_reg = 0.0, local_el2 = 0.0, local_grey = 0.0, local_hLap = 0.0;
  const double *hp = h.data();
  const double *djel = dJdh.data();
  const double *lp = lap.data();
  double *gp = g.data();
  const double inv_eps = 1.0 / spec.epsilon;
  for (std::size_t i = 0; i < n_local; ++i) {
    const double well = pfc::apps::inverse::double_well(hp[i]);
    local_reg += 0.5 * spec.epsilon * (-hp[i] * lp[i]) + inv_eps * well;
    local_hLap += -hp[i] * lp[i];
    if (hp[i] > 0.1 && hp[i] < 0.9) local_grey += 1.0;
    const double g_el = n_global * djel[i];
    gp[i] = g_el;
    local_el2 += g_el * g_el;
  }
  double glo[4] = {0, 0, 0, 0};
  const double loc[4] = {local_reg, local_el2, local_grey, local_hLap};
  MPI_Allreduce(loc, glo, 4, MPI_DOUBLE, MPI_SUM, comm);
  out.J_reg = spec.lambda_reg * (glo[0] / n_global);
  out.J = out.J_tensor + out.J_volume + out.J_reg;
  const double el_rms = std::sqrt(glo[1] / n_global);
  out.grey_fraction = glo[2] / n_global;
  out.perimeter = std::sqrt(std::max(0.0, glo[3] / n_global));
  const double el_scale =
      (spec.normalize_grad && el_rms > 1.0e-30) ? (1.0 / el_rms) : 1.0;
  double local_g2 = 0.0;
  for (std::size_t i = 0; i < n_local; ++i) {
    const double g_el = el_scale * gp[i];
    const double g_vol = spec.lambda_volume * 2.0 * dv;
    const double g_reg =
        spec.lambda_reg * (-spec.epsilon * lp[i] +
                           inv_eps * pfc::apps::inverse::double_well_prime(hp[i]));
    gp[i] = g_el + g_vol + g_reg;
    local_g2 += gp[i] * gp[i];
  }
  double glo_g2 = 0.0;
  MPI_Allreduce(&local_g2, &glo_g2, 1, MPI_DOUBLE, MPI_SUM, comm);
  out.grad_rms = std::sqrt(glo_g2 / n_global);
  const double scale = spec.dt * spec.mobility;
  const double cap = spec.max_abs_delta;
  double local_dh2 = 0.0;
  for (std::size_t i = 0; i < n_local; ++i) {
    double dh = -scale * gp[i];
    if (cap > 0.0) dh = std::min(cap, std::max(-cap, dh));
    local_dh2 += dh * dh;
    h.data()[i] = std::min(1.0, std::max(0.0, hp[i] + dh));
  }
  h.note_host_write();
  double glo_dh2 = 0.0;
  MPI_Allreduce(&local_dh2, &glo_dh2, 1, MPI_DOUBLE, MPI_SUM, comm);
  out.step_rms = std::sqrt(glo_dh2 / n_global);
  if (spec.project_volume) project_mean(h, spec.volume_target, comm, n_global);
  out.volume_fraction = mean_value(h, comm, n_global);
  return out;
}

int run(int argc, char **argv, int rank, int nproc) {
  Config cfg;
  if (!parse_args(argc, argv, cfg)) {
    if (rank == 0)
      usage(std::cerr, argc >= 1 ? argv[0] : "openpfc_inverse_homogenize_hip");
    return 2;
  }
  if (cfg.checkpoint_dir.empty() && !cfg.restart_dir.empty())
    cfg.checkpoint_dir = cfg.restart_dir;
  if (cfg.checkpoint_dir.empty() && !cfg.fields.dir.empty())
    cfg.checkpoint_dir = cfg.fields.dir + "/checkpoint";
  int rc = 0;
  const pfc::Domain domain = pfc::domain::create(
      pfc::GridSize({cfg.nx, cfg.ny, cfg.nz}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({cfg.dx, cfg.dx, cfg.dx}));
  pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace> stack(domain, rank, nproc,
                                                          MPI_COMM_WORLD);
  auto &fft = stack.fft();
  auto h = pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds());
  const auto n = h.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        double hv = cfg.init_volume;
        if (cfg.init == "noise") {
          const auto gidx = h.global(i, j, k);
          const double twopi = 2.0 * 3.141592653589793;
          const double s = static_cast<double>(cfg.seed);
          hv += 0.25 * std::sin(twopi * (gidx[0] + s) / cfg.nx) *
                std::sin(twopi * (gidx[1] + 2.0 * s) / cfg.ny);
        }
        h(i, j, k) = std::min(1.0, std::max(0.0, hv));
      }
  if (cfg.init == "rotating-squares") {
    pfc::apps::inverse::fill_rotating_squares(h, cfg.nx, cfg.ny, cfg.init_half,
                                              cfg.init_angle);
  } else if (cfg.init == "spinodal") {
    pfc::apps::inverse::SpinodalSpec ch;
    ch.c0 = cfg.init_volume;
    ch.noise = cfg.init_amp;
    ch.seed = cfg.seed;
    pfc::apps::inverse::seed_spinodal_noise(h, cfg.nx, cfg.ny, cfg.nz, ch);
  } else if (cfg.init == "yang-a3") {
    pfc::apps::inverse::fill_yang_a3(h, cfg.nx, cfg.ny, cfg.nz);
  }
  if (!cfg.load_bin.empty()) {
    if (!pfc::apps::inverse::load_fortran_bin(cfg.load_bin, cfg.nx, cfg.ny, cfg.nz,
                                              h)) {
      if (rank == 0)
        std::cerr << "load-bin: unreadable or wrong size " << cfg.load_bin << '\n';
      return 2;
    }
  }
  h.note_host_write();

  pfc::solvers::MicroelasticityParams p;
  p.stiffness_at_one = pfc::solvers::Stiffness::isotropic(cfg.E_solid, cfg.nu_solid);
  p.stiffness_at_zero = pfc::solvers::Stiffness::isotropic(cfg.E_void, cfg.nu_void);
  p.relative_tolerance = 1.0e-8;
  p.max_iterations = cfg.n_el_iter;
  p.warm_start = false;
  p.comm = MPI_COMM_WORLD;

  pfc::apps::inverse::InverseSpec spec;
  try {
    spec.C_target = make_target(cfg);
  } catch (const std::exception &e) {
    if (rank == 0) std::cerr << e.what() << '\n';
    return 2;
  }
  spec.volume_target = cfg.volume;
  spec.lambda_volume = cfg.lambda_volume;
  spec.lambda_reg = cfg.lambda_reg;
  spec.epsilon = cfg.epsilon;
  spec.dt = cfg.dt;
  spec.normalize_grad = cfg.normalize != 0;
  spec.max_abs_delta = cfg.max_delta;
  spec.project_volume = cfg.project_volume != 0;
  spec.simp_p = cfg.simp;
  if (cfg.w12 != 1.0 || cfg.target == "auxetic") {
    const double w = (cfg.target == "auxetic" && cfg.w12 == 1.0) ? 4.0 : cfg.w12;
    spec.W(0, 1) = spec.W(1, 0) = w;
    spec.W(0, 2) = spec.W(2, 0) = w;
    spec.W(1, 2) = spec.W(2, 1) = w;
  }

  pfc::apps::PeriodicHomogenizerHIP hom(domain, fft, p);
  ComplexField hat(domain, fft.get_outbox_bounds(), 0);
  auto lap = pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds());
  auto dJdh = pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds());
  auto g = pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds());
  auto penalized =
      pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds());
  auto h_prev = pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds());
  FFT::RealBuffer d_real(fft.size_inbox());
  FFT::ComplexBuffer d_hat(fft.size_outbox());

  if (rank == 0 && !cfg.fields.dir.empty())
    std::filesystem::create_directories(cfg.fields.dir);
  if (rank == 0 && !cfg.checkpoint_dir.empty())
    std::filesystem::create_directories(cfg.checkpoint_dir);
  MPI_Barrier(MPI_COMM_WORLD);

  const auto box = fft.get_inbox_bounds();
  pfc::apps::inverse::FieldSnapshotWriter snap(
      cfg.fields, cfg.run_id, {cfg.nx, cfg.ny, cfg.nz},
      {box.high[0] - box.low[0] + 1, box.high[1] - box.low[1] + 1,
       box.high[2] - box.low[2] + 1},
      {box.low[0], box.low[1], box.low[2]}, cfg.dx, rank, MPI_COMM_WORLD);

  {
    const long long n_global = static_cast<long long>(cfg.nx) * cfg.ny * cfg.nz;
    const std::size_t n_owned =
        static_cast<std::size_t>(box.high[0] - box.low[0] + 1) *
        static_cast<std::size_t>(box.high[1] - box.low[1] + 1) *
        static_cast<std::size_t>(box.high[2] - box.low[2] + 1);
    report_hbm(rank, nproc, MPI_COMM_WORLD, n_global, n_owned * sizeof(double));
  }

  std::ofstream csv;
  if (rank == 0) {
    std::cout << "backend hip ranks " << nproc << " grid " << cfg.nx << 'x' << cfg.ny
              << 'x' << cfg.nz << " init " << cfg.init << " target " << cfg.target
              << " max_steps " << cfg.steps << " continuation_steps "
              << cfg.continuation_steps << '\n';
    std::cout << "step J J_tensor volume grey C11 C12 nu_eff design_rms dJ_rel "
                 "dC_rel morph_frac step_rms grad_rms simp lambda_reg frozen "
                 "window candidate verify ms conv\n";
    if (!cfg.csv.empty()) {
      const bool resume =
          !cfg.restart_dir.empty() && std::filesystem::exists(cfg.csv);
      csv.open(cfg.csv, resume ? std::ios::app : std::ios::out);
      if (!resume) csv << pfc::apps::inverse::kInverseCsvHeader << '\n';
    }
  }

  auto nu_of = [](double c11, double c12) {
    const double den = c11 + c12;
    return (std::abs(den) > 1.0e-30) ? c12 / den : 0.0;
  };

  int n_snap = 0;
  int last_dumped = -1;
  auto dump = [&](int step, bool force) {
    if (!snap.active()) return;
    if (!force && !snap.due(std::max(0, step))) return;
    if (force && last_dumped == step) return;
    snap.note_step(step);
    snap.write("h", n_snap, h);
    ++n_snap;
    last_dumped = step;
  };

  const double simp0 = cfg.simp;
  const double simp1 = (cfg.simp_end > 0.0) ? cfg.simp_end : cfg.simp;
  const double lreg0 = cfg.lambda_reg;
  const double lreg1 =
      (cfg.lambda_reg_end >= 0.0) ? cfg.lambda_reg_end : cfg.lambda_reg;

  pfc::apps::inverse::ConvergenceTracker tracker;
  tracker.cfg.continuation_steps = cfg.continuation_steps;
  tracker.cfg.max_steps = cfg.steps;
  tracker.cfg.conv_window = cfg.conv_window;
  tracker.cfg.verify_steps = cfg.verify_steps;
  tracker.cfg.tol_design = cfg.tol_design;
  tracker.cfg.tol_objective = cfg.tol_objective;
  tracker.cfg.tol_tensor = cfg.tol_tensor;

  for (std::size_t i = 0; i < h.size(); ++i) h_prev.data()[i] = h.data()[i];
  h_prev.note_host_write();
  pfc::apps::Voigt6 C_prev{};
  double J_prev = 0.0;
  bool have_prev = false;
  int start_s = 0;

  pfc::apps::inverse::FieldOutputConfig ckpt_fields;
  ckpt_fields.dir = cfg.checkpoint_dir;
  ckpt_fields.every = 1;
  pfc::apps::inverse::FieldSnapshotWriter ckpt_snap(
      ckpt_fields, "ckpt", {cfg.nx, cfg.ny, cfg.nz},
      {box.high[0] - box.low[0] + 1, box.high[1] - box.low[1] + 1,
       box.high[2] - box.low[2] + 1},
      {box.low[0], box.low[1], box.low[2]}, cfg.dx, rank, MPI_COMM_WORLD);

  auto write_ckpt = [&](int next_step,
                        pfc::apps::inverse::TerminationReason why =
                            pfc::apps::inverse::TerminationReason::Running) {
    if (cfg.checkpoint_dir.empty()) return true;
    pfc::apps::inverse::InverseCheckpoint ck;
    pfc::apps::inverse::capture_problem(ck, cfg, spec.C_target, spec.W);
    ck.nx = cfg.nx;
    ck.ny = cfg.ny;
    ck.nz = cfg.nz;
    pfc::apps::inverse::capture_tracker(ck, tracker, next_step);
    ck.have_prev = have_prev ? 1 : 0;
    ck.n_snap = n_snap;
    ck.last_dumped = last_dumped;
    ck.termination = static_cast<int>(why);
    ck.J_prev = J_prev;
    pfc::apps::inverse::store_voigt6(ck.C_prev, C_prev);
    const auto root = std::filesystem::path(cfg.checkpoint_dir);
    const auto gen = pfc::apps::inverse::checkpoint_generation_name(next_step);
    int ready = 1;
    if (rank == 0)
      ready = pfc::apps::inverse::prepare_checkpoint_staging(root) ? 1 : 0;
    MPI_Bcast(&ready, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ready) {
      if (rank == 0) std::cerr << "checkpoint: failed to prepare staging\n";
      return false;
    }
    const auto staging =
        pfc::apps::inverse::checkpoint_staging_dir(root).string();
    ckpt_snap.set_directory(staging);
    ckpt_snap.write_named("h.bin", h);
    ckpt_snap.write_named("h_prev.bin", h_prev);
    if (rank == 0) {
      ready = pfc::apps::inverse::checkpoint_field_sizes_match(staging, ck);
      ready = ready &&
              pfc::apps::inverse::write_checkpoint_file(staging + "/state.txt", ck);
      ready = ready && pfc::apps::inverse::write_dump_steps(
                           staging + "/dump_steps.txt", snap.steps());
      ready = ready && pfc::apps::inverse::publish_checkpoint_generation(root, gen);
      if (!ready) std::cerr << "checkpoint: failed to publish " << gen << '\n';
    }
    MPI_Bcast(&ready, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ready != 0;
  };

  if (!cfg.restart_dir.empty()) {
    pfc::apps::inverse::InverseCheckpoint ck;
    std::string bundle;
    std::vector<int> dump_steps;
    int ok = 1;
    if (rank == 0) {
      bundle =
          pfc::apps::inverse::resolve_checkpoint_bundle(cfg.restart_dir).string();
      if (bundle.empty()) {
        ok = 0;
      } else {
        std::ifstream in(bundle + "/state.txt");
        if (!in || !pfc::apps::inverse::read_checkpoint_text(in, ck) ||
            !pfc::apps::inverse::checkpoint_matches_problem(
                ck, cfg, spec.C_target, spec.W) ||
            !pfc::apps::inverse::checkpoint_is_restartable(ck, cfg.steps))
          ok = 0;
        else if (ck.last_dumped > ck.next_step ||
                 !pfc::apps::inverse::read_dump_steps(bundle + "/dump_steps.txt",
                                                      dump_steps, ck.n_snap,
                                                      ck.last_dumped))
          ok = 0;
      }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int n_bundle = static_cast<int>(bundle.size());
    MPI_Bcast(&n_bundle, 1, MPI_INT, 0, MPI_COMM_WORLD);
    bundle.resize(static_cast<std::size_t>(n_bundle));
    if (n_bundle > 0)
      MPI_Bcast(bundle.data(), n_bundle, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ck, static_cast<int>(sizeof(ck)), MPI_BYTE, 0, MPI_COMM_WORLD);
    int n_dump = static_cast<int>(dump_steps.size());
    MPI_Bcast(&n_dump, 1, MPI_INT, 0, MPI_COMM_WORLD);
    dump_steps.resize(static_cast<std::size_t>(std::max(0, n_dump)));
    if (n_dump > 0)
      MPI_Bcast(dump_steps.data(), n_dump, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
      if (rank == 0)
        std::cerr << "restart: unreadable or mismatched " << cfg.restart_dir << '\n';
      return 2;
    }
    snap.restore_steps(std::move(dump_steps));
    if (!pfc::apps::inverse::load_fortran_bin(bundle + "/h.bin", cfg.nx, cfg.ny,
                                              cfg.nz, h) ||
        !pfc::apps::inverse::load_fortran_bin(bundle + "/h_prev.bin", cfg.nx,
                                              cfg.ny, cfg.nz, h_prev)) {
      if (rank == 0) std::cerr << "restart: missing h.bin / h_prev.bin\n";
      return 2;
    }
    h.note_host_write();
    h_prev.note_host_write();
    pfc::apps::inverse::apply_tracker(ck, tracker);
    tracker.cfg.max_steps = cfg.steps;
    start_s = ck.next_step;
    have_prev = ck.have_prev != 0;
    n_snap = ck.n_snap;
    last_dumped = ck.last_dumped;
    J_prev = ck.J_prev;
    pfc::apps::inverse::fill_voigt6(C_prev, ck.C_prev);
    if (rank == 0)
      std::cout << "restart next_step " << start_s << " quiet "
                << tracker.quiet_count << " candidate "
                << (tracker.candidate ? 1 : 0) << '\n';
  }
  if (cfg.restart_dir.empty()) dump(0, false);

  pfc::apps::inverse::InverseStepReport last{};
  auto reason = pfc::apps::inverse::TerminationReason::Running;
  int n_done = start_s;
  const auto gs = h.global_size();
  const double n_global = static_cast<double>(gs[0]) * gs[1] * gs[2];
  for (int s = start_s; s < cfg.steps; ++s) {
    const double t =
        pfc::apps::inverse::continuation_fraction(s, cfg.continuation_steps);
    spec.simp_p = simp0 + t * (simp1 - simp0);
    spec.lambda_reg = lreg0 + t * (lreg1 - lreg0);
    // Accepted-state pair: (h_s, J(h_s), C_H(h_s)) vs previous accepted
    // state. Measure ||h_s-h_{s-1}|| before the update that produces h_{s+1}.
    double local_dh2 = 0.0, local_morph = 0.0;
    for (std::size_t i = 0; i < h.size(); ++i) {
      const double d = h.data()[i] - h_prev.data()[i];
      local_dh2 += d * d;
      if ((h.data()[i] > 0.5) != (h_prev.data()[i] > 0.5)) local_morph += 1.0;
    }
    double glo_dh2 = 0.0, glo_morph = 0.0;
    MPI_Allreduce(&local_dh2, &glo_dh2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_morph, &glo_morph, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double design_rms = std::sqrt(glo_dh2 / n_global);
    const double morph_frac = glo_morph / n_global;
    for (std::size_t i = 0; i < h.size(); ++i) h_prev.data()[i] = h.data()[i];
    h_prev.note_host_write();
    const auto t0 = std::chrono::steady_clock::now();
    last = ac_step(hom, domain, fft, h, spec, hat, lap, dJdh, g, penalized, d_real,
                   d_hat, MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double dC2 = 0.0;
    if (have_prev) {
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
          const double d = last.C(i, j) - C_prev(i, j);
          dC2 += d * d;
        }
    }
    auto metrics = pfc::apps::inverse::make_metrics(
        design_rms, last.J, have_prev ? J_prev : last.J, std::sqrt(dC2),
        have_prev ? C_prev.frobenius_norm() : last.C_fro, morph_frac, tracker.cfg);
    if (!have_prev) metrics.quiet = false;
    reason = tracker.after_step(s, last.elasticity_converged, metrics);
    const double nu = nu_of(last.C11, last.C12);
    const int frozen =
        pfc::apps::inverse::params_frozen(s, cfg.continuation_steps) ? 1 : 0;
    if (rank == 0) {
      std::cout << std::setprecision(8) << s << ' ' << last.J << ' ' << last.J_tensor
                << ' ' << last.volume_accepted << ' ' << last.grey_fraction << ' '
                << last.C11 << ' ' << last.C12 << ' ' << nu << ' ' << design_rms
                << ' ' << metrics.dJ_rel << ' ' << metrics.dC_rel << ' '
                << morph_frac << ' ' << last.step_rms << ' ' << last.grad_rms << ' '
                << spec.simp_p << ' ' << spec.lambda_reg << ' ' << frozen << ' '
                << tracker.quiet_count << ' ' << (tracker.candidate ? 1 : 0) << ' '
                << tracker.verify_left << ' ' << std::setprecision(3) << ms << ' '
                << (last.elasticity_converged ? "yes" : "no") << ' '
                << pfc::apps::inverse::termination_name(reason) << '\n';
      if (csv.is_open()) {
        pfc::apps::inverse::InverseCsvRow row;
        row.step = s;
        row.J = last.J;
        row.J_tensor = last.J_tensor;
        row.J_volume = last.J_volume;
        row.J_reg = last.J_reg;
        row.volume = last.volume_accepted;
        row.grey = last.grey_fraction;
        row.C11 = last.C11;
        row.C12 = last.C12;
        row.nu_eff = nu;
        row.C_fro = last.C_fro;
        row.design_rms = design_rms;
        row.dJ_rel = metrics.dJ_rel;
        row.dC_rel = metrics.dC_rel;
        row.morph_frac = morph_frac;
        row.step_rms = last.step_rms;
        row.grad_rms = last.grad_rms;
        row.simp_p = spec.simp_p;
        row.lambda_reg = spec.lambda_reg;
        row.frozen = frozen;
        row.conv_window = tracker.quiet_count;
        row.candidate = tracker.candidate ? 1 : 0;
        row.verified = tracker.verified ? 1 : 0;
        row.ms = ms;
        row.elasticity = last.elasticity_converged ? 1 : 0;
        row.termination = pfc::apps::inverse::termination_name(reason);
        pfc::apps::inverse::write_inverse_csv_row(csv, row);
        csv.flush();
      }
    }
    n_done = s + 1;
    C_prev = last.C;
    J_prev = last.J;
    have_prev = true;
    if (reason != pfc::apps::inverse::TerminationReason::Running) {
      pfc::apps::inverse::copy_design_buffer(h_prev.data(), h.data(), h.size());
      h.note_host_write();
      if (!write_ckpt(n_done, reason)) {
        rc = 2;
        break;
      }
      if (reason == pfc::apps::inverse::TerminationReason::ElasticityFailure) rc = 1;
      break;
    }
    dump(s + 1, false);
    if (!write_ckpt(n_done)) {
      rc = 2;
      break;
    }
    if (cfg.stop_after > 0 && n_done >= cfg.stop_after) break;
  }
  if (rc == 2) return rc;
  if (cfg.stop_after > 0 &&
      reason == pfc::apps::inverse::TerminationReason::Running) {
    if (rank == 0) std::cout << "STOP_AFTER steps_done " << n_done << '\n';
    return rc;
  }
  const int certified_step = std::max(0, n_done - 1);
  dump(certified_step, true);

  const auto final = hom.compute(h);
  if (rank == 0) {
    const auto &C = final.stiffness;
    const double nu = nu_of(C(0, 0), C(0, 1));
    const double Jt = pfc::apps::tensor_mismatch(C, spec.C_target, spec.W);
    std::cout << std::setprecision(16) << "FINAL_RECOMPUTE J_tensor " << Jt
              << " C11 " << C(0, 0) << " C12 " << C(0, 1) << " nu_eff " << nu
              << " C_fro " << C.symmetrized().frobenius_norm() << " elasticity "
              << (final.all_converged() ? 1 : 0) << '\n';
    if (csv.is_open()) {
      csv << "# CERTIFIED_STEP " << certified_step << " termination "
          << pfc::apps::inverse::termination_name(reason) << '\n';
      csv << "# FINAL_RECOMPUTE unpenalized C_H of certified accepted h; "
             "not an iterate J_tensor="
          << Jt << " C11=" << C(0, 0) << " C12=" << C(0, 1) << " nu_eff=" << nu
          << " C_fro=" << C.symmetrized().frobenius_norm()
          << " elasticity=" << (final.all_converged() ? 1 : 0) << '\n';
      csv.flush();
    }
  }
  auto hbin = h;
  for (std::size_t i = 0; i < h.size(); ++i)
    hbin.data()[i] = (h.data()[i] > 0.5) ? 1.0 : 0.0;
  hbin.note_host_write();
  const auto bin = hom.compute(hbin);
  snap.write_named("h_final.bin", h);
  snap.write_named("h_thresh.bin", hbin);
  snap.write_xdmf_brick("h_final.xdmf", "h_final.bin", "h");
  snap.write_xdmf_brick("h_thresh.xdmf", "h_thresh.bin", "h");
  snap.write_manifest({"h"});
  if (rank == 0) {
    if (!cfg.fields.dir.empty()) {
      const auto write_report = [&](const char *name, const char *field,
                                    const pfc::apps::HomogenizationResult &result) {
        const auto path = std::filesystem::path(cfg.fields.dir) / name;
        std::ofstream stream(path);
        stream << pfc::apps::inverse::material_report(field, certified_step, reason,
                                                      {cfg.nx, cfg.ny, cfg.nz},
                                                      cfg.dx, result, spec.C_target)
                      .dump(2)
               << '\n';
        stream.close();
        if (!stream)
          throw std::runtime_error("cannot write material report: " + path.string());
      };
      write_report("h_final_material.json", "h_final.bin", final);
      write_report("h_thresh_material.json", "h_thresh.bin", bin);
    }
    const auto &C = final.stiffness;
    const double nu = nu_of(C(0, 0), C(0, 1));
    const auto &Cb = bin.stiffness;
    const double nub = nu_of(Cb(0, 0), Cb(0, 1));
    std::cout << std::setprecision(16) << "INVERSE_CHECKSUM " << last.J << '\n';
    std::cout << std::setprecision(8) << "C11 " << C(0, 0) << " C12 " << C(0, 1)
              << " nu_eff " << nu << " grey " << last.grey_fraction << '\n';
    std::cout << "C11_bin " << Cb(0, 0) << " C12_bin " << Cb(0, 1) << " nu_bin "
              << nub << '\n';
    std::cout << "ranks " << nproc << " grid " << cfg.nx << 'x' << cfg.ny << 'x'
              << cfg.nz << " steps_done " << n_done << " certified_step "
              << certified_step << " termination "
              << pfc::apps::inverse::termination_name(reason) << '\n';
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
      if (line.rfind("VmHWM:", 0) == 0 || line.rfind("VmRSS:", 0) == 0)
        std::cout << line << '\n';
    }
  }
  return rc;
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        return run(app_argc, app_argv, rank, nproc);
      });
}

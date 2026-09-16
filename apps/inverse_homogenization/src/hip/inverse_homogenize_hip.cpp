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
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <openpfc_apps/homogenization_hip.hpp>
#include <inverse_homogenization/auxetic_geometry.hpp>
#include <inverse_homogenization/field_output.hpp>
#include <inverse_homogenization/phase_field_inverse.hpp>

namespace {

using RealField = pfc::data::Field<double>;
using ComplexField = pfc::data::Field<std::complex<double>>;
using FFT = pfc::fft::IDeviceFFT<pfc::HIPSpace>;

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
  int steps{40};
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
  double w12{1.0};
  pfc::apps::inverse::FieldOutputConfig fields{};
};

void usage(std::ostream &os, const char *exe) {
  os << "Usage: " << exe << " [--key=value]...\n"
     << "  HIP inverse homogenization (device Green, host Allen-Cahn).\n"
     << "  --nx --ny --nz --dx --target isotropic|auxetic|orthotropic\n"
     << "  --init rotating-squares|noise|uniform --steps --csv --dump-dir\n";
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
    if (key == "nx") ok = parse_int(val, cfg.nx) && cfg.nx > 0;
    else if (key == "ny") ok = parse_int(val, cfg.ny) && cfg.ny > 0;
    else if (key == "nz") ok = parse_int(val, cfg.nz) && cfg.nz > 0;
    else if (key == "dx") ok = parse_double(val, cfg.dx) && cfg.dx > 0.0;
    else if (key == "E-solid") ok = parse_double(val, cfg.E_solid);
    else if (key == "nu-solid") ok = parse_double(val, cfg.nu_solid);
    else if (key == "E-void") ok = parse_double(val, cfg.E_void);
    else if (key == "nu-void") ok = parse_double(val, cfg.nu_void);
    else if (key == "target") cfg.target = std::string(val);
    else if (key == "E-target") ok = parse_double(val, cfg.E_target);
    else if (key == "nu-target") ok = parse_double(val, cfg.nu_target);
    else if (key == "volume") ok = parse_double(val, cfg.volume);
    else if (key == "lambda-volume") ok = parse_double(val, cfg.lambda_volume);
    else if (key == "lambda-reg") ok = parse_double(val, cfg.lambda_reg);
    else if (key == "epsilon") ok = parse_double(val, cfg.epsilon) && cfg.epsilon > 0.0;
    else if (key == "dt") ok = parse_double(val, cfg.dt) && cfg.dt > 0.0;
    else if (key == "steps") ok = parse_int(val, cfg.steps) && cfg.steps >= 0;
    else if (key == "init") cfg.init = std::string(val);
    else if (key == "init-volume") ok = parse_double(val, cfg.init_volume);
    else if (key == "init-half") ok = parse_double(val, cfg.init_half) && cfg.init_half > 0.0;
    else if (key == "init-angle") ok = parse_double(val, cfg.init_angle);
    else if (key == "seed") {
      int s = 1;
      ok = parse_int(val, s);
      cfg.seed = static_cast<unsigned>(s);
    } else if (key == "csv") cfg.csv = std::string(val);
    else if (key == "run-id") cfg.run_id = std::string(val);
    else if (key == "normalize") ok = parse_int(val, cfg.normalize);
    else if (key == "max-delta") ok = parse_double(val, cfg.max_delta) && cfg.max_delta >= 0.0;
    else if (key == "project-volume") ok = parse_int(val, cfg.project_volume);
    else if (key == "simp") ok = parse_double(val, cfg.simp) && cfg.simp >= 1.0;
    else if (key == "W-12") ok = parse_double(val, cfg.w12) && cfg.w12 >= 0.0;
    else if (key == "dump-dir") cfg.fields.dir = std::string(val);
    else if (key == "dump-every") ok = parse_int(val, cfg.fields.every) && cfg.fields.every > 0;
    else return false;
    if (!ok) return false;
  }
  if (cfg.target != "isotropic" && cfg.target != "auxetic") return false;
  if (cfg.init != "uniform" && cfg.init != "noise" &&
      cfg.init != "rotating-squares")
    return false;
  return true;
}

pfc::apps::Voigt6 make_target(const Config &cfg) {
  const double nu = (cfg.target == "auxetic") ? -std::abs(cfg.nu_target)
                                                : cfg.nu_target;
  return pfc::apps::voigt_from_stiffness(
      pfc::apps::Stiffness::isotropic(cfg.E_target, nu));
}

void spectral_laplacian_hip(const pfc::Domain &domain, FFT &fft, const RealField &h,
                             ComplexField &hat, RealField &lap, FFT::RealBuffer &d_real,
                             FFT::ComplexBuffer &d_hat) {
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
        RealField &lap, RealField &dJdh, RealField &g, FFT::RealBuffer &d_real,
        FFT::ComplexBuffer &d_hat, MPI_Comm comm) {
  pfc::apps::inverse::InverseStepReport out;
  const std::size_t n_local = h.size();
  const auto gs = h.global_size();
  const double n_global =
      static_cast<double>(gs[0]) * static_cast<double>(gs[1]) *
      static_cast<double>(gs[2]);
  const double vf0 = mean_value(h, comm, n_global);
  const double dv = vf0 - spec.volume_target;
  out.J_volume = spec.lambda_volume * dv * dv;

  const auto r = hom.compute(h);
  out.elasticity_converged = r.all_converged();
  out.J_tensor = pfc::apps::tensor_mismatch(r.stiffness, spec.C_target, spec.W);
  out.C11 = r.stiffness(0, 0);
  out.C12 = r.stiffness(0, 1);
  hom.objective_sensitivity(h, spec.C_target, spec.W, dJdh);
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
    const double g_reg = spec.lambda_reg * (-spec.epsilon * lp[i] +
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
    if (rank == 0) usage(std::cerr, argc >= 1 ? argv[0] : "openpfc_inverse_homogenize_hip");
    return 2;
  }
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
  }
  h.note_host_write();

  pfc::apps::MicroelasticityParams p;
  p.c_solid = pfc::apps::Stiffness::isotropic(cfg.E_solid, cfg.nu_solid);
  p.c_liquid = pfc::apps::Stiffness::isotropic(cfg.E_void, cfg.nu_void);
  p.tol_el = 1.0e-8;
  p.n_el_iter = 200;
  p.warm_start = false;
  p.comm = MPI_COMM_WORLD;

  pfc::apps::inverse::InverseSpec spec;
  spec.C_target = make_target(cfg);
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
  FFT::RealBuffer d_real(fft.size_inbox());
  FFT::ComplexBuffer d_hat(fft.size_outbox());

  if (rank == 0 && !cfg.fields.dir.empty())
    std::filesystem::create_directories(cfg.fields.dir);
  MPI_Barrier(MPI_COMM_WORLD);

  const auto box = fft.get_inbox_bounds();
  pfc::apps::inverse::FieldSnapshotWriter snap(
      cfg.fields, cfg.run_id, {cfg.nx, cfg.ny, cfg.nz},
      {box.high[0] - box.low[0] + 1, box.high[1] - box.low[1] + 1,
       box.high[2] - box.low[2] + 1},
      {box.low[0], box.low[1], box.low[2]}, cfg.dx, rank, MPI_COMM_WORLD);

  std::ofstream csv;
  if (rank == 0) {
    std::cout << "backend hip ranks " << nproc << " grid " << cfg.nx << 'x'
              << cfg.ny << 'x' << cfg.nz << " init " << cfg.init << " target "
              << cfg.target << " steps " << cfg.steps << '\n';
    std::cout << "step J J_tensor volume grey C11 C12 nu_eff ms conv\n";
    if (!cfg.csv.empty()) {
      csv.open(cfg.csv);
      csv << "step,J,J_tensor,J_volume,J_reg,volume,grey,C11,C12,nu_eff,ms,"
             "converged\n";
    }
  }

  auto nu_of = [](double c11, double c12) {
    const double den = c11 + c12;
    return (std::abs(den) > 1.0e-30) ? c12 / den : 0.0;
  };

  int n_snap = 0;
  auto dump = [&](int step) {
    if (!snap.due(std::max(0, step))) return;
    snap.note_step(step);
    snap.write("h", n_snap, h);
    ++n_snap;
  };
  dump(0);

  pfc::apps::inverse::InverseStepReport last{};
  int n_done = 0;
  for (int s = 0; s < cfg.steps; ++s) {
    const auto t0 = std::chrono::steady_clock::now();
    last = ac_step(hom, domain, fft, h, spec, hat, lap, dJdh, g, d_real, d_hat,
                   MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double nu = nu_of(last.C11, last.C12);
    if (rank == 0) {
      std::cout << std::setprecision(8) << s << ' ' << last.J << ' '
                << last.J_tensor << ' ' << last.volume_fraction << ' '
                << last.grey_fraction << ' ' << last.C11 << ' ' << last.C12 << ' '
                << nu << ' ' << std::setprecision(3) << ms << ' '
                << (last.elasticity_converged ? "yes" : "no") << '\n';
      if (csv.is_open()) {
        csv << s << ',' << last.J << ',' << last.J_tensor << ',' << last.J_volume
            << ',' << last.J_reg << ',' << last.volume_fraction << ','
            << last.grey_fraction << ',' << last.C11 << ',' << last.C12 << ','
            << nu << ',' << ms << ',' << (last.elasticity_converged ? 1 : 0)
            << '\n';
        csv.flush();
      }
    }
    dump(s + 1);
    n_done = s + 1;
    if (!last.elasticity_converged) {
      rc = 1;
      break;
    }
  }

  const auto final = hom.compute(h);
  if (rank == 0 && csv.is_open()) {
    const auto &C = final.stiffness;
    const double nu = nu_of(C(0, 0), C(0, 1));
    const double Jt = pfc::apps::tensor_mismatch(C, spec.C_target, spec.W);
    const double dv = last.volume_fraction - spec.volume_target;
    const double Jv = spec.lambda_volume * dv * dv;
    csv << n_done << ',' << (Jt + Jv + last.J_reg) << ',' << Jt << ',' << Jv
        << ',' << last.J_reg << ',' << last.volume_fraction << ','
        << last.grey_fraction << ',' << C(0, 0) << ',' << C(0, 1) << ',' << nu
        << ',' << 0.0 << ',' << (final.all_converged() ? 1 : 0) << '\n';
    csv.flush();
  }
  auto hbin = h;
  for (std::size_t i = 0; i < h.size(); ++i)
    hbin.data()[i] = (h.data()[i] > 0.5) ? 1.0 : 0.0;
  hbin.note_host_write();
  const auto bin = hom.compute(hbin);
  snap.write_manifest({"h"});
  if (rank == 0) {
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
              << cfg.nz << " steps " << cfg.steps << '\n';
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

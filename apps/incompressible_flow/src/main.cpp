// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file main.cpp
 * @brief Periodic incompressible flow.
 *
 * Taylor–Green and one decaying homogeneous-isotropic realization.
 * The box length is [0, 2π]^3 and is not a flag. Dealiasing stays on.
 * Forcing, walls, and a 2-D streamfunction case are not this executable.
 */

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <openpfc/runtime/common/mpi_main.hpp>

#include <flow/decaying_hit.hpp>
#include <flow/taylor_green.hpp>

namespace {

struct Options {
  std::string which{"taylor-green"};
  std::optional<int> n;
  std::optional<double> nu;
  std::optional<double> dt;
  std::optional<double> time;
  std::optional<std::uint64_t> seed;
  std::string series;
  std::string outdir{"results/incompressible-flow"};
};

void usage(std::ostream &os) {
  os << "Usage: incompressible_flow --case taylor-green|decaying-hit\n"
        "       [--n N] [--nu NU] [--dt DT] [--time T] [--seed SEED]\n"
        "       [--series spatial|temporal] [--outdir DIR]\n"
        "\n"
        "Periodic 3-D Navier-Stokes on [0, 2pi]^3. N is even and at\n"
        "least 8. time is an integer number of steps. The 2/3 mask\n"
        "stays on. Ranks share the HeFFTe pencil. --series runs the\n"
        "locked one-rank decaying-hit ladder and ignores n, nu, dt,\n"
        "time, and seed.\n";
}

std::optional<int> parse_int(std::string_view text) {
  try {
    std::size_t used = 0;
    const int value = std::stoi(std::string(text), &used);
    if (used != text.size()) return std::nullopt;
    return value;
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

std::optional<double> parse_double(std::string_view text) {
  try {
    std::size_t used = 0;
    const double value = std::stod(std::string(text), &used);
    if (used != text.size()) return std::nullopt;
    return value;
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

std::optional<std::uint64_t> parse_u64(std::string_view text) {
  if (text.empty() || text.front() == '-') return std::nullopt;
  try {
    std::size_t used = 0;
    const unsigned long long value = std::stoull(std::string(text), &used, 10);
    if (used != text.size()) return std::nullopt;
    return static_cast<std::uint64_t>(value);
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

bool parse(int argc, char **argv, Options &opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view flag(argv[i]);
    auto need = [&](const char *name) -> std::optional<std::string_view> {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        return std::nullopt;
      }
      return std::string_view(argv[++i]);
    };
    if (flag == "-h" || flag == "--help") {
      usage(std::cout);
      std::exit(EXIT_SUCCESS);
    } else if (flag == "--case") {
      auto value = need("--case");
      if (!value) return false;
      opt.which = std::string(*value);
    } else if (flag == "--n") {
      auto value = need("--n");
      if (!value) return false;
      auto parsed = parse_int(*value);
      if (!parsed) return false;
      opt.n = *parsed;
    } else if (flag == "--nu") {
      auto value = need("--nu");
      if (!value) return false;
      auto parsed = parse_double(*value);
      if (!parsed) return false;
      opt.nu = *parsed;
    } else if (flag == "--dt") {
      auto value = need("--dt");
      if (!value) return false;
      auto parsed = parse_double(*value);
      if (!parsed) return false;
      opt.dt = *parsed;
    } else if (flag == "--time") {
      auto value = need("--time");
      if (!value) return false;
      auto parsed = parse_double(*value);
      if (!parsed) return false;
      opt.time = *parsed;
    } else if (flag == "--seed") {
      auto value = need("--seed");
      if (!value) return false;
      auto parsed = parse_u64(*value);
      if (!parsed) return false;
      opt.seed = *parsed;
    } else if (flag == "--series") {
      auto value = need("--series");
      if (!value) return false;
      opt.series = std::string(*value);
    } else if (flag == "--outdir") {
      auto value = need("--outdir");
      if (!value) return false;
      opt.outdir = std::string(*value);
    } else {
      std::cerr << "unknown argument: " << flag << "\n";
      return false;
    }
  }
  if (opt.which != "taylor-green" && opt.which != "decaying-hit") {
    std::cerr << "unknown case: " << opt.which << "\n";
    return false;
  }
  if (!opt.series.empty() && opt.series != "spatial" && opt.series != "temporal") {
    std::cerr << "unknown series: " << opt.series << "\n";
    return false;
  }
  if (opt.which == "taylor-green" && (opt.seed || !opt.series.empty())) {
    std::cerr << "taylor-green does not take --seed or --series\n";
    return false;
  }
  if (!opt.series.empty() && (opt.n || opt.nu || opt.dt || opt.time || opt.seed)) {
    std::cerr << "the locked series does not take n, nu, dt, time, or seed\n";
    return false;
  }
  return true;
}

void write_taylor_row(std::ostream &os, double time, const flow::Diagnostics &diag,
                      std::string_view status) {
  os << std::scientific << std::setprecision(16) << time << ',' << diag.ke << ','
     << diag.enstrophy << ',' << diag.dissipation << ',' << diag.div_l2 << ','
     << diag.div_linf << ',' << diag.modal_div_max << ',' << diag.w_l2 << ','
     << diag.max_abs_w << ',' << diag.cfl << ',' << status << '\n';
}

int run_taylor(const Options &opt, int rank, int nproc) {
  const int n = opt.n.value_or(32);
  const double nu = opt.nu.value_or(0.05);
  const double dt = opt.dt.value_or(0.01);
  const double time = opt.time.value_or(0.1);
  const long long steps = flow::steps_for(time, dt);
  auto state = flow::make_state(n, nu, dt, rank, nproc);
  flow::initialize_taylor_green(state);
  std::ofstream csv;
  if (rank == 0) {
    std::filesystem::create_directories(opt.outdir);
    csv.open(opt.outdir + "/diagnostics.csv");
    csv << "time,ke,enstrophy,dissipation,div_l2,div_linf,modal_div_max,"
           "w_l2,max_abs_w,cfl,status\n";
  }

  auto report = [&](double sample, const flow::Diagnostics &diag,
                    std::string_view status) {
    if (rank == 0) {
      write_taylor_row(csv, sample, diag, status);
      csv.flush();
    }
    if (rank == 0) {
      std::cout << "incompressible_flow sample t=" << std::scientific
                << std::setprecision(8) << sample << " status=" << status
                << " ke=" << diag.ke << " w_l2=" << diag.w_l2 << std::endl;
    }
  };

  auto initial = flow::diagnose(state);
  std::string status = "ok";
  if (!initial.finite)
    status = "nonfinite";
  else if (initial.cfl > flow::cfl_limit)
    status = "cfl";
  report(0.0, initial, status);
  if (status != "ok") {
    if (rank == 0) std::cout << "incompressible_flow status=" << status << std::endl;
    return EXIT_FAILURE;
  }

  for (long long step = 1; step <= steps; ++step) {
    flow::step(state);
    if (step != steps) continue;
    auto diag = flow::diagnose(state);
    if (!diag.finite)
      status = "nonfinite";
    else if (diag.cfl > flow::cfl_limit)
      status = "cfl";
    report(time, diag, status);
  }
  if (rank == 0) std::cout << "incompressible_flow status=" << status << std::endl;
  return status == "ok" ? EXIT_SUCCESS : EXIT_FAILURE;
}

void write_hit_row(std::ostream &os, double time, const flow::Diagnostics &diag,
                   const flow::Scales &scales, std::string_view status) {
  os << std::scientific << std::setprecision(16) << time << ',' << diag.ke << ','
     << diag.enstrophy << ',' << scales.omega_rms << ',' << diag.dissipation << ','
     << scales.u_rms << ',' << scales.lambda << ',' << scales.re_lambda << ','
     << scales.integral_scale << ',' << scales.eta << ',' << scales.k_max << ','
     << scales.k_max_eta << ',' << diag.div_l2 << ',' << diag.div_linf << ','
     << diag.modal_div_max << ',' << diag.cfl << ',' << scales.outer_ke_fraction
     << ',' << status << '\n';
}

int run_hit(const Options &opt, int rank, int nproc) {
  const int n = opt.n.value_or(32);
  const double nu = opt.nu.value_or(flow::protocol_nu);
  const double dt = opt.dt.value_or(1.0 / 64.0);
  const double time = opt.time.value_or(0.0625);
  const std::uint64_t seed = opt.seed.value_or(flow::protocol_seed);
  const long long steps = flow::steps_for(time, dt);
  auto state = flow::make_state(n, nu, dt, rank, nproc);
  flow::initialize_decaying_hit(state, seed);
  std::ofstream csv;
  std::ofstream spectrum;
  if (rank == 0) {
    std::filesystem::create_directories(opt.outdir);
    csv.open(opt.outdir + "/diagnostics.csv");
    csv << "time,ke,enstrophy,omega_rms,dissipation,u_rms,lambda,re_lambda,"
           "integral_scale,eta,k_max,k_max_eta,div_l2,div_linf,modal_div_max,cfl,"
           "outer_ke_fraction,status\n";
    spectrum.open(opt.outdir + "/spectrum.csv");
    spectrum << "time,shell,shell_ke\n";
  }

  auto report = [&](double sample, const flow::Diagnostics &diag,
                    const flow::Scales &scales, std::string_view status) {
    const auto shells = flow::shell_energies(state);
    if (rank == 0) {
      write_hit_row(csv, sample, diag, scales, status);
      csv.flush();
      spectrum << std::scientific << std::setprecision(16);
      for (const auto &shell : shells) {
        spectrum << sample << ',' << shell.index << ',' << shell.ke << '\n';
      }
      spectrum.flush();
    }
    if (rank == 0) {
      std::cout << "incompressible_flow sample t=" << std::scientific
                << std::setprecision(8) << sample << " status=" << status
                << " ke=" << diag.ke << " re_lambda=" << scales.re_lambda
                << std::endl;
    }
  };

  auto initial = flow::diagnose(state);
  auto initial_scales = flow::measure_scales(state, initial);
  std::string status = "ok";
  if (!initial.finite)
    status = "nonfinite";
  else if (initial.cfl > flow::cfl_limit)
    status = "cfl";
  report(0.0, initial, initial_scales, status);
  if (status != "ok") {
    if (rank == 0) std::cout << "incompressible_flow status=" << status << std::endl;
    return EXIT_FAILURE;
  }

  flow::Diagnostics final_diag = initial;
  flow::Scales final_scales = initial_scales;
  for (long long step = 1; step <= steps; ++step) {
    flow::step(state);
    if (step != steps) continue;
    final_diag = flow::diagnose(state);
    final_scales = flow::measure_scales(state, final_diag);
    if (!final_diag.finite)
      status = "nonfinite";
    else if (final_diag.cfl > flow::cfl_limit)
      status = "cfl";
    report(time, final_diag, final_scales, status);
  }
  const double budget = (final_diag.ke - initial.ke) +
                        0.5 * (initial.dissipation + final_diag.dissipation) * time;
  if (rank == 0) {
    std::ofstream meta(opt.outdir + "/metadata.txt");
    meta << std::setprecision(16);
    meta << "case=decaying-hit\n"
         << "spectrum=yoffe-mccomb-2018-eq10\n"
         << "citation=arXiv:1805.01238\n"
         << "c=" << flow::spectrum_c << "\n"
         << "k0=" << flow::spectrum_k0 << "\n"
         << "seed=" << seed << "\n"
         << "n=" << n << "\n"
         << "nu=" << nu << "\n"
         << "dt=" << std::scientific << dt << "\n"
         << "time=" << time << "\n"
         << "steps=" << steps << "\n"
         << "budget_residual=" << budget << "\n"
         << "assignment=E(|k|)/(4*pi*|k|^2) on 2/3-retained modes\n";
    std::cout << "incompressible_flow status=" << status << std::endl;
  }
  return status == "ok" ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        Options opt;
        if (!parse(app_argc, app_argv, opt)) {
          if (rank == 0) usage(std::cerr);
          return EXIT_FAILURE;
        }
        if (!opt.series.empty() && nproc != 1) {
          if (rank == 0) {
            std::cerr << "incompressible_flow: the locked series is the "
                         "one-rank protocol\n";
          }
          return EXIT_FAILURE;
        }
        try {
          if (!opt.series.empty())
            return flow::run_series(opt.series, opt.outdir, rank);
          if (opt.which == "taylor-green") return run_taylor(opt, rank, nproc);
          return run_hit(opt, rank, nproc);
        } catch (const std::exception &ex) {
          std::cerr << ex.what() << std::endl;
          return EXIT_FAILURE;
        }
      });
}

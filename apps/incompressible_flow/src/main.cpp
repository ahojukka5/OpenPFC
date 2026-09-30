// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file main.cpp
 * @brief Periodic incompressible flow (issue #233).
 *
 * The shipped case is 3-D Taylor–Green on [0, 2π]³. Viscosity, resolution,
 * and the step are arguments. The box length is not. Dealiasing stays on.
 * Forcing, walls, and a 2-D streamfunction case are not this executable.
 */

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

#include <flow/taylor_green.hpp>

namespace {

struct Options {
  std::string which{"taylor-green"};
  int n{32};
  double nu{0.05};
  double dt{0.01};
  double time{0.1};
  std::string outdir{"results/incompressible-flow"};
};

void usage(std::ostream &os) {
  os << "Usage: incompressible_flow --case taylor-green\n"
        "       [--n N] [--nu NU] [--dt DT] [--time T] [--outdir DIR]\n"
        "\n"
        "Periodic 3-D Navier-Stokes on [0, 2pi]^3. The only case is\n"
        "Taylor-Green. N is even and at least 8. time is an integer\n"
        "number of steps. The 2/3 mask stays on. One MPI rank.\n"
        "The 2-D vorticity-streamfunction prototype is examples/ns2d_vorticity.\n";
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
    } else if (flag == "--outdir") {
      auto value = need("--outdir");
      if (!value) return false;
      opt.outdir = std::string(*value);
    } else {
      std::cerr << "unknown argument: " << flag << "\n";
      return false;
    }
  }
  if (opt.which != "taylor-green") {
    std::cerr << "unknown case: " << opt.which << "\n";
    return false;
  }
  return true;
}

void write_row(std::ostream &os, double time, const flow::Diagnostics &diag,
               std::string_view status) {
  os << std::scientific << std::setprecision(16) << time << ',' << diag.ke << ','
     << diag.enstrophy << ',' << diag.dissipation << ',' << diag.div_l2 << ','
     << diag.div_linf << ',' << diag.modal_div_max << ',' << diag.w_l2 << ','
     << diag.max_abs_w << ',' << diag.cfl << ',' << status << '\n';
}

int run(const Options &opt, int rank, int nproc) {
  const long long steps = flow::steps_for(opt.time, opt.dt);
  auto state = flow::make_state(opt.n, opt.nu, opt.dt, rank, nproc);
  flow::initialize_taylor_green(state);
  std::filesystem::create_directories(opt.outdir);
  std::ofstream csv(opt.outdir + "/diagnostics.csv");
  csv << "time,ke,enstrophy,dissipation,div_l2,div_linf,modal_div_max,"
         "w_l2,max_abs_w,cfl,status\n";

  auto report = [&](double time, const flow::Diagnostics &diag,
                    std::string_view status) {
    write_row(csv, time, diag, status);
    csv.flush();
    if (rank == 0) {
      std::cout << "incompressible_flow sample t=" << std::scientific
                << std::setprecision(8) << time << " status=" << status
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
    report(opt.time, diag, status);
  }
  if (rank == 0) std::cout << "incompressible_flow status=" << status << std::endl;
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
        if (nproc != 1) {
          if (rank == 0) {
            std::cerr << "incompressible_flow: one rank is required so every "
                         "Fourier mode is owned locally\n";
          }
          return EXIT_FAILURE;
        }
        try {
          return run(opt, rank, nproc);
        } catch (const std::exception &ex) {
          std::cerr << ex.what() << std::endl;
          return EXIT_FAILURE;
        }
      });
}

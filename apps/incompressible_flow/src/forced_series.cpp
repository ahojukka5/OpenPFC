// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file forced_series.cpp
 * @brief Locked grid and timestep ladder for constant-power HIT.
 *
 * Windows, power, viscosity, and steps live in forced_hit.hpp.
 */

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <flow/forced_hit.hpp>

namespace flow {
namespace {

struct Sample {
  double time{0.0};
  Diagnostics diag{};
  Scales scales{};
  double band_ke{0.0};
  double injection{0.0};
  bool force_applied{false};
  std::string status{"ok"};
};

struct Summary {
  std::string run;
  int n{0};
  double dt{0.0};
  double ke_mean{0.0};
  double ke_stderr{0.0};
  double ke_first{0.0};
  double ke_last{0.0};
  double dissipation_mean{0.0};
  double dissipation_stderr{0.0};
  double injection_mean{0.0};
  double injection_stderr{0.0};
  double re_mean{0.0};
  double re_stderr{0.0};
  double k_max_eta_mean{0.0};
  double budget{0.0};
  double seconds{0.0};
  std::string status{"ok"};
  std::vector<Sample> samples;
};

[[nodiscard]] std::string stop_reason(const Diagnostics &diag) {
  if (!diag.finite) return "nonfinite";
  if (diag.cfl > cfl_limit) return "cfl";
  return "ok";
}

[[nodiscard]] double trapezoid(const std::vector<Sample> &rows, double t0, double t1,
                               double (*value)(const Sample &)) {
  double integral = 0.0;
  const Sample *previous = nullptr;
  for (const auto &row : rows) {
    if (row.time < t0 - 1.0e-12 || row.time > t1 + 1.0e-12) continue;
    if (previous != nullptr) {
      integral +=
          0.5 * (value(*previous) + value(row)) * (row.time - previous->time);
    }
    previous = &row;
  }
  return integral;
}

double sample_ke(const Sample &row) { return row.diag.ke; }
double sample_dissipation(const Sample &row) { return row.diag.dissipation; }
double sample_injection(const Sample &row) { return row.injection; }
double sample_re(const Sample &row) { return row.scales.re_lambda; }
double sample_eta(const Sample &row) { return row.scales.k_max_eta; }

struct Blocks {
  double mean{0.0};
  double stderr{0.0};
  double first{0.0};
  double last{0.0};
};

[[nodiscard]] Blocks block_stats(const std::vector<Sample> &rows,
                                 double (*value)(const Sample &)) {
  const int n_blocks = static_cast<int>(
      std::llround((forced_time - forced_equilibration) / forced_block));
  std::vector<double> means;
  means.reserve(static_cast<std::size_t>(n_blocks));
  for (int b = 0; b < n_blocks; ++b) {
    const double t0 = forced_equilibration + static_cast<double>(b) * forced_block;
    const double t1 = t0 + forced_block;
    const double width = t1 - t0;
    means.push_back(trapezoid(rows, t0, t1, value) / width);
  }
  Blocks out;
  if (means.empty()) return out;
  double sum = 0.0;
  for (double mean : means) sum += mean;
  out.mean = sum / static_cast<double>(means.size());
  out.first = means.front();
  out.last = means.back();
  if (means.size() > 1) {
    double var = 0.0;
    for (double mean : means) {
      const double d = mean - out.mean;
      var += d * d;
    }
    var /= static_cast<double>(means.size() - 1);
    out.stderr = std::sqrt(var / static_cast<double>(means.size()));
  }
  return out;
}

void write_shells(std::ostream &os, const std::string &run, int n, double dt,
                  const Sample &sample, const State &state) {
  os << std::scientific << std::setprecision(16);
  for (const auto &shell : shell_energies(state)) {
    os << run << ',' << n << ',' << dt << ',' << sample.time << ',' << shell.index
       << ',' << shell.ke << '\n';
  }
}

Summary run_one(const std::string &run, int n, double dt, std::ostream &shells,
                int rank) {
  Summary out;
  out.run = run;
  out.n = n;
  out.dt = dt;
  const auto started = std::chrono::steady_clock::now();
  auto stamp = [&]() {
    out.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
  };

  State state = make_state(n, forced_nu, dt, rank, 1);
  initialize_decaying_hit(state, protocol_seed);
  const long long steps = steps_for(forced_time, dt);
  const long long stride = steps_for(forced_sample, dt);

  auto take = [&](double time) -> std::string {
    const auto outbox = state.stack->fft().get_outbox_bounds();
    std::vector<Complex> fu, fv, fw;
    const BandForce force =
        write_band_force(outbox, state.n, state.spacing, state.u, state.v, state.w,
                         fu, fv, fw, forced_power);
    auto diag = diagnose(state);
    auto scales = measure_scales(state, diag);
    Sample sample;
    sample.time = time;
    sample.diag = diag;
    sample.scales = scales;
    sample.band_ke = force.band_ke;
    sample.injection = force.injection;
    sample.force_applied = force.applied;
    sample.status = stop_reason(diag);
    if (rank == 0) {
      std::cout << "incompressible_flow sample series=" << run << " n=" << n
                << " dt=" << std::scientific << std::setprecision(8) << dt
                << " t=" << time << " status=" << sample.status << " ke=" << diag.ke
                << " dissipation=" << diag.dissipation
                << " injection=" << sample.injection
                << " re_lambda=" << scales.re_lambda << std::endl;
    }
    write_shells(shells, run, n, dt, sample, state);
    const std::string status = sample.status;
    out.samples.push_back(std::move(sample));
    return status;
  };

  if (take(0.0) != "ok") {
    out.status = out.samples.back().status;
    stamp();
    return out;
  }
  for (long long taken = 1; taken <= steps; ++taken) {
    step_forced(state, forced_power);
    if (!hats_finite(state)) {
      Sample sample;
      sample.time = static_cast<double>(taken) * dt;
      sample.diag = diagnose(state);
      sample.scales = measure_scales(state, sample.diag);
      sample.status = "nonfinite";
      const double when = sample.time;
      out.samples.push_back(std::move(sample));
      out.status = "nonfinite";
      if (rank == 0) {
        std::cout << "incompressible_flow sample series=" << run << " n=" << n
                  << " dt=" << std::scientific << std::setprecision(8) << dt
                  << " t=" << when << " status=nonfinite" << std::endl;
      }
      stamp();
      return out;
    }
    if (taken % stride != 0) continue;
    if (take(static_cast<double>(taken) * dt) != "ok") {
      out.status = out.samples.back().status;
      stamp();
      return out;
    }
  }
  const auto ke = block_stats(out.samples, sample_ke);
  const auto dissipation = block_stats(out.samples, sample_dissipation);
  const auto injection = block_stats(out.samples, sample_injection);
  const auto re = block_stats(out.samples, sample_re);
  const auto eta = block_stats(out.samples, sample_eta);
  out.ke_mean = ke.mean;
  out.ke_stderr = ke.stderr;
  out.ke_first = ke.first;
  out.ke_last = ke.last;
  out.dissipation_mean = dissipation.mean;
  out.dissipation_stderr = dissipation.stderr;
  out.injection_mean = injection.mean;
  out.injection_stderr = injection.stderr;
  out.re_mean = re.mean;
  out.re_stderr = re.stderr;
  out.k_max_eta_mean = eta.mean;
  const double injected =
      trapezoid(out.samples, forced_equilibration, forced_time, sample_injection);
  const double dissipated =
      trapezoid(out.samples, forced_equilibration, forced_time, sample_dissipation);
  const Sample *start = nullptr;
  const Sample *finish = nullptr;
  for (const auto &sample : out.samples) {
    if (std::abs(sample.time - forced_equilibration) <= 1.0e-10) start = &sample;
    if (std::abs(sample.time - forced_time) <= 1.0e-10) finish = &sample;
  }
  if (start != nullptr && finish != nullptr) {
    out.budget = (finish->diag.ke - start->diag.ke) - (injected - dissipated);
  }
  out.status = "ok";
  stamp();
  return out;
}

void write_samples(const std::filesystem::path &path,
                   const std::vector<Summary> &runs) {
  std::ofstream os(path);
  os << "run,n,dt,time,ke,enstrophy,dissipation,injection,band_ke,u_rms,lambda,"
        "re_lambda,integral_scale,eta,k_max,k_max_eta,div_l2,div_linf,modal_div_max,"
        "cfl,outer_ke_fraction,force_applied,status\n";
  for (const auto &run : runs) {
    for (const auto &sample : run.samples) {
      os << run.run << ',' << run.n << ',' << std::scientific
         << std::setprecision(16) << run.dt << ',' << sample.time << ','
         << sample.diag.ke << ',' << sample.diag.enstrophy << ','
         << sample.diag.dissipation << ',' << sample.injection << ','
         << sample.band_ke << ',' << sample.scales.u_rms << ','
         << sample.scales.lambda << ',' << sample.scales.re_lambda << ','
         << sample.scales.integral_scale << ',' << sample.scales.eta << ','
         << sample.scales.k_max << ',' << sample.scales.k_max_eta << ','
         << sample.diag.div_l2 << ',' << sample.diag.div_linf << ','
         << sample.diag.modal_div_max << ',' << sample.diag.cfl << ','
         << sample.scales.outer_ke_fraction << ',' << (sample.force_applied ? 1 : 0)
         << ',' << sample.status << '\n';
    }
  }
}

void write_summary(const std::filesystem::path &path,
                   const std::vector<Summary> &runs) {
  std::ofstream os(path);
  os << "run,n,dt,ke_mean,ke_stderr,ke_first_block,ke_last_block,dissipation_mean,"
        "dissipation_stderr,injection_mean,injection_stderr,re_lambda_mean,"
        "re_lambda_stderr,k_max_eta_mean,budget_residual,seconds,status\n";
  for (const auto &run : runs) {
    os << run.run << ',' << run.n << ',' << std::scientific << std::setprecision(16)
       << run.dt << ',' << run.ke_mean << ',' << run.ke_stderr << ',' << run.ke_first
       << ',' << run.ke_last << ',' << run.dissipation_mean << ','
       << run.dissipation_stderr << ',' << run.injection_mean << ','
       << run.injection_stderr << ',' << run.re_mean << ',' << run.re_stderr << ','
       << run.k_max_eta_mean << ',' << run.budget << ',' << run.seconds << ','
       << run.status << '\n';
  }
}

void write_metadata(const std::filesystem::path &path) {
  std::ofstream os(path);
  os << std::setprecision(16);
  os << "case=forced-hit\n"
     << "series=stationary\n"
     << "forcing=doering-petrov-2004-eq3\n"
     << "citation=arXiv:physics/0404049\n"
     << "band=|k|=1\n"
     << "power=" << forced_power << "\n"
     << "nu=" << forced_nu << "\n"
     << "seed=" << protocol_seed << "\n"
     << "ic=yoffe-mccomb-2018-eq10\n"
     << "time=" << forced_time << "\n"
     << "sample=" << forced_sample << "\n"
     << "equilibration=" << forced_equilibration << "\n"
     << "block=" << forced_block << "\n"
     << "spatial_n=128,64,32\n"
     << "spatial_dt=" << std::scientific << forced_dt << "\n"
     << "control_dt=" << forced_control_dt << "\n"
     << "coarse_dt=" << forced_coarse_dt << "\n"
     << "band_floor=" << forced_band_floor << "\n";
}

} // namespace

int run_forced_series(std::string_view which, const std::string &outdir, int rank) {
  if (which != "stationary") {
    throw std::invalid_argument(
        "incompressible_flow: forced series must be stationary");
  }
  const double span = forced_time - forced_equilibration;
  if (!(span > 0.0) ||
      std::abs(span / forced_block - std::llround(span / forced_block)) > 1.0e-8) {
    throw std::invalid_argument("incompressible_flow: forced blocks do not tile");
  }
  std::filesystem::create_directories(outdir);
  write_metadata(std::filesystem::path(outdir) / "metadata.txt");
  std::ofstream shells(std::filesystem::path(outdir) / "shells.csv");
  shells << "run,n,dt,time,shell,shell_ke\n";

  std::vector<Summary> runs;
  bool ok = true;
  auto accept = [&](Summary summary) {
    ok = ok && summary.status == "ok" && !summary.samples.empty();
    runs.push_back(std::move(summary));
  };
  for (int n : forced_spatial_n) {
    accept(run_one("spatial", n, forced_dt, shells, rank));
  }
  accept(run_one("control", forced_temporal_n, forced_control_dt, shells, rank));
  accept(run_one("coarse", forced_temporal_n, forced_coarse_dt, shells, rank));

  if (rank == 0) {
    write_samples(std::filesystem::path(outdir) / "samples.csv", runs);
    write_summary(std::filesystem::path(outdir) / "summary.csv", runs);
    std::cout << "incompressible_flow status=" << (ok ? "ok" : "failed")
              << std::endl;
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace flow

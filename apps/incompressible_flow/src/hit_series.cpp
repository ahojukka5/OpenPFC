// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file hit_series.cpp
 * @brief Locked spatial and temporal ladders for decaying HIT.
 *
 * The grids, steps, times, viscosity, seed, and order floor live in
 * decaying_hit.hpp. This file only runs that protocol.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <flow/decaying_hit.hpp>

namespace flow {
namespace {

struct Hats {
  std::array<int, 3> n{};
  std::array<double, 3> spacing{};
  pfc::fft::Box3i outbox{};
  std::vector<Complex> u;
  std::vector<Complex> v;
  std::vector<Complex> w;
};

struct FieldError {
  double velocity_l2{0.0};
  double velocity_l2_rel{0.0};
  double vorticity_l2{0.0};
  double vorticity_l2_rel{0.0};
};

struct Row {
  std::string run;
  int n{0};
  double dt{0.0};
  double time{0.0};
  Diagnostics diag{};
  Scales scales{};
  double budget{0.0};
  std::optional<FieldError> error;
  std::string order_velocity;
  std::string order_vorticity;
  double seconds{0.0};
  std::string status{"ok"};
  Hats hats{};
  bool has_hats{false};
};

constexpr double kSampleTimes[] = {0.0, 0.5, 1.0};

[[nodiscard]] unsigned long long pack_wave(int ki, int kj, int kk) noexcept {
  constexpr unsigned long long mask = (1ull << 21) - 1ull;
  return ((static_cast<unsigned long long>(static_cast<std::uint32_t>(ki)) & mask)
          << 42) |
         ((static_cast<unsigned long long>(static_cast<std::uint32_t>(kj)) & mask)
          << 21) |
         (static_cast<unsigned long long>(static_cast<std::uint32_t>(kk)) & mask);
}

[[nodiscard]] Hats copy_hats(const State &state) {
  Hats hats;
  hats.n = state.n;
  hats.spacing = state.spacing;
  hats.outbox = state.stack->fft().get_outbox_bounds();
  hats.u = state.u;
  hats.v = state.v;
  hats.w = state.w;
  return hats;
}

void inverse_velocity(State &state, const std::vector<Complex> &u,
                      const std::vector<Complex> &v, const std::vector<Complex> &w,
                      std::vector<double> &rx, std::vector<double> &ry,
                      std::vector<double> &rz) {
  auto &fft = state.stack->fft();
  rx.resize(fft.size_inbox());
  ry.resize(fft.size_inbox());
  rz.resize(fft.size_inbox());
  std::vector<Complex> hu = u;
  std::vector<Complex> hv = v;
  std::vector<Complex> hw = w;
  fft.backward(hu, rx);
  fft.backward(hv, ry);
  fft.backward(hw, rz);
}

[[nodiscard]] FieldError field_error(State &coarse, const Hats &fine) {
  const auto coarse_box = coarse.stack->fft().get_outbox_bounds();
  const auto nhat = coarse.u.size();
  const double scale = cell_count(coarse.n) / cell_count(fine.n);

  std::unordered_map<unsigned long long, std::size_t> fmap;
  fmap.reserve(fine.u.size());
  pfc::fft::kspace::for_each_kpoint(
      fine.outbox, fine.n, fine.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        fmap[pack_wave(signed_index(i, fine.n[0]), signed_index(j, fine.n[1]),
                       signed_index(k, fine.n[2]))] = idx;
      });

  std::vector<Complex> ru(nhat), rv(nhat), rw(nhat);
  pfc::fft::kspace::for_each_kpoint(
      coarse_box, coarse.n, coarse.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        const auto it = fmap.find(pack_wave(signed_index(i, coarse.n[0]),
                                            signed_index(j, coarse.n[1]),
                                            signed_index(k, coarse.n[2])));
        if (it == fmap.end()) return;
        ru[idx] = fine.u[it->second] * scale;
        rv[idx] = fine.v[it->second] * scale;
        rw[idx] = fine.w[it->second] * scale;
      });

  std::vector<double> crx, cry, crz, rrx, rry, rrz;
  inverse_velocity(coarse, coarse.u, coarse.v, coarse.w, crx, cry, crz);
  inverse_velocity(coarse, ru, rv, rw, rrx, rry, rrz);

  std::vector<Complex> cox(nhat), coy(nhat), coz(nhat), fox(nhat), foy(nhat),
      foz(nhat);
  pfc::field::curl_hat(coarse_box, coarse.n, coarse.spacing, coarse.u.data(),
                       coarse.v.data(), coarse.w.data(), cox.data(), coy.data(),
                       coz.data(), nhat);
  pfc::field::curl_hat(coarse_box, coarse.n, coarse.spacing, ru.data(), rv.data(),
                       rw.data(), fox.data(), foy.data(), foz.data(), nhat);
  std::vector<double> coxr, coyr, cozr, foxr, foyr, fozr;
  inverse_velocity(coarse, cox, coy, coz, coxr, coyr, cozr);
  inverse_velocity(coarse, fox, foy, foz, foxr, foyr, fozr);

  double ve2 = 0.0;
  double vr2 = 0.0;
  double we2 = 0.0;
  double wr2 = 0.0;
  const auto inbox = coarse.stack->fft().get_inbox_bounds();
  for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
        const auto idx = static_cast<std::size_t>(inbox.to_linear({i, j, k}));
        const double du = crx[idx] - rrx[idx];
        const double dv = cry[idx] - rry[idx];
        const double dw = crz[idx] - rrz[idx];
        ve2 += du * du + dv * dv + dw * dw;
        vr2 += rrx[idx] * rrx[idx] + rry[idx] * rry[idx] + rrz[idx] * rrz[idx];
        const double dox = coxr[idx] - foxr[idx];
        const double doy = coyr[idx] - foyr[idx];
        const double doz = cozr[idx] - fozr[idx];
        we2 += dox * dox + doy * doy + doz * doz;
        wr2 += foxr[idx] * foxr[idx] + foyr[idx] * foyr[idx] + fozr[idx] * fozr[idx];
      }
    }
  }
  const double ncells = cell_count(coarse.n);
  FieldError err;
  err.velocity_l2 = std::sqrt(ve2 / ncells);
  const double vref = std::sqrt(vr2 / ncells);
  err.velocity_l2_rel = (vref > 0.0) ? err.velocity_l2 / vref : 0.0;
  err.vorticity_l2 = std::sqrt(we2 / ncells);
  const double wref = std::sqrt(wr2 / ncells);
  err.vorticity_l2_rel = (wref > 0.0) ? err.vorticity_l2 / wref : 0.0;
  return err;
}

[[nodiscard]] std::string stop_reason(const Diagnostics &diag) {
  if (!diag.finite) return "nonfinite";
  if (diag.cfl > cfl_limit) return "cfl";
  return "ok";
}

void write_shells(std::ostream &os, const std::string &run, int n, double dt,
                  double time, const State &state) {
  os << std::scientific << std::setprecision(16);
  for (const auto &shell : shell_energies(state)) {
    os << run << ',' << n << ',' << dt << ',' << time << ',' << shell.index << ','
       << shell.ke << '\n';
  }
  os.flush();
}

struct Resolution {
  std::vector<Row> rows;
  bool complete{false};
};

Resolution run_resolution(const std::string &run, int n, double dt, bool keep_hats,
                          const std::vector<Row> *reference, std::ostream &shells,
                          int rank) {
  Resolution out;
  const auto started = std::chrono::steady_clock::now();
  auto seconds_now = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
        .count();
  };
  auto stamp = [&]() {
    const double seconds = seconds_now();
    for (auto &row : out.rows) row.seconds = seconds;
  };

  State state = make_state(n, protocol_nu, dt, rank, 1);
  initialize_decaying_hit(state, protocol_seed);
  const long long steps = steps_for(protocol_time, dt);
  std::vector<double> sample_ke;
  std::vector<double> sample_eps;
  std::vector<double> sample_t;

  auto take = [&](double time) {
    auto diag = diagnose(state);
    auto scales = measure_scales(state, diag);
    Row row;
    row.run = run;
    row.n = n;
    row.dt = dt;
    row.time = time;
    row.diag = diag;
    row.scales = scales;
    row.status = stop_reason(diag);
    sample_ke.push_back(diag.ke);
    sample_eps.push_back(diag.dissipation);
    sample_t.push_back(time);
    double integral = 0.0;
    for (std::size_t i = 1; i < sample_t.size(); ++i) {
      integral += 0.5 * (sample_eps[i - 1] + sample_eps[i]) *
                  (sample_t[i] - sample_t[i - 1]);
    }
    row.budget = sample_ke.back() - sample_ke.front() + integral;
    if (reference != nullptr && row.status == "ok") {
      for (const auto &ref : *reference) {
        if (!ref.has_hats || ref.time != time || ref.status != "ok") continue;
        row.error = field_error(state, ref.hats);
        break;
      }
    }
    if (keep_hats && row.status == "ok") {
      row.hats = copy_hats(state);
      row.has_hats = true;
    }
    write_shells(shells, run, n, dt, time, state);
    if (rank == 0) {
      std::cout << "incompressible_flow sample series=" << run << " n=" << n
                << " dt=" << std::scientific << std::setprecision(8) << dt
                << " t=" << time << " status=" << row.status << " ke=" << diag.ke
                << " enstrophy=" << diag.enstrophy
                << " re_lambda=" << scales.re_lambda
                << " k_max_eta=" << scales.k_max_eta << std::endl;
    }
    const std::string status = row.status;
    out.rows.push_back(std::move(row));
    return status;
  };

  if (take(0.0) != "ok") {
    stamp();
    return out;
  }
  for (long long taken = 1; taken <= steps; ++taken) {
    flow::step(state);
    if (!hats_finite(state)) {
      auto diag = diagnose(state);
      Row row;
      row.run = run;
      row.n = n;
      row.dt = dt;
      row.time = static_cast<double>(taken) * dt;
      row.diag = diag;
      row.scales = measure_scales(state, diag);
      row.status = "nonfinite";
      row.budget = diag.ke - sample_ke.front();
      out.rows.push_back(std::move(row));
      if (rank == 0) {
        std::cout << "incompressible_flow sample series=" << run << " n=" << n
                  << " dt=" << std::scientific << std::setprecision(8) << dt
                  << " t=" << static_cast<double>(taken) * dt << " status=nonfinite"
                  << std::endl;
      }
      stamp();
      return out;
    }
    for (double time : kSampleTimes) {
      if (time == 0.0) continue;
      if (taken != std::llround(time / dt)) continue;
      if (take(time) != "ok") {
        stamp();
        return out;
      }
    }
  }
  stamp();
  out.complete = out.rows.size() == 3;
  for (const auto &row : out.rows) {
    if (row.status != "ok") out.complete = false;
  }
  return out;
}

[[nodiscard]] std::string order_text(double coarse, double fine,
                                     bool finer_is_reference) {
  if (finer_is_reference) return {};
  if (!(coarse > protocol_order_floor) || !(fine > protocol_order_floor)) {
    return "roundoff";
  }
  std::ostringstream os;
  os << std::fixed << std::setprecision(8) << std::log2(coarse / fine);
  return os.str();
}

void fill_orders(std::vector<Row> &rows, double reference_dt) {
  for (auto &row : rows) {
    if (row.run != "temporal" || !row.error) continue;
    const double finer = 0.5 * row.dt;
    const bool finer_is_reference = std::abs(finer - reference_dt) <= 1.0e-12;
    const Row *fine_row = nullptr;
    for (const auto &other : rows) {
      if (other.run != "temporal" || other.time != row.time || !other.error)
        continue;
      if (std::abs(other.dt - finer) <= 1.0e-12) fine_row = &other;
    }
    if (fine_row == nullptr) continue;
    row.order_velocity = order_text(
        row.error->velocity_l2, fine_row->error->velocity_l2, finer_is_reference);
    row.order_vorticity = order_text(
        row.error->vorticity_l2, fine_row->error->vorticity_l2, finer_is_reference);
  }
}

void write_opt(std::ostream &os, const std::optional<double> &value) {
  if (!value) return;
  os << std::scientific << std::setprecision(16) << *value;
}

void write_rows(const std::filesystem::path &path, const std::vector<Row> &rows) {
  std::ofstream os(path);
  os << "run,n,dt,time,ke,enstrophy,omega_rms,dissipation,u_rms,lambda,re_lambda,"
        "integral_scale,eta,k_max,k_max_eta,div_l2,div_linf,modal_div_max,cfl,"
        "outer_ke_fraction,modal_ke,budget_residual,velocity_l2,velocity_l2_rel,"
        "vorticity_l2,vorticity_l2_rel,order_velocity,order_vorticity,seconds,"
        "status\n";
  for (const auto &row : rows) {
    os << row.run << ',' << row.n << ',' << std::scientific << std::setprecision(16)
       << row.dt << ',' << row.time << ',' << row.diag.ke << ','
       << row.diag.enstrophy << ',' << row.scales.omega_rms << ','
       << row.diag.dissipation << ',' << row.scales.u_rms << ',' << row.scales.lambda
       << ',' << row.scales.re_lambda << ',' << row.scales.integral_scale << ','
       << row.scales.eta << ',' << row.scales.k_max << ',' << row.scales.k_max_eta
       << ',' << row.diag.div_l2 << ',' << row.diag.div_linf << ','
       << row.diag.modal_div_max << ',' << row.diag.cfl << ','
       << row.scales.outer_ke_fraction << ',' << row.scales.modal_ke << ','
       << row.budget << ',';
    write_opt(os, row.error ? std::optional<double>(row.error->velocity_l2)
                            : std::nullopt);
    os << ',';
    write_opt(os, row.error ? std::optional<double>(row.error->velocity_l2_rel)
                            : std::nullopt);
    os << ',';
    write_opt(os, row.error ? std::optional<double>(row.error->vorticity_l2)
                            : std::nullopt);
    os << ',';
    write_opt(os, row.error ? std::optional<double>(row.error->vorticity_l2_rel)
                            : std::nullopt);
    os << ',' << row.order_velocity << ',' << row.order_vorticity << ','
       << std::scientific << std::setprecision(16) << row.seconds << ','
       << row.status << '\n';
  }
}

void write_metadata(const std::filesystem::path &path, std::string_view which) {
  std::ofstream os(path);
  os << std::setprecision(16);
  os << "case=decaying-hit\n"
     << "series=" << which << "\n"
     << "spectrum=yoffe-mccomb-2018-eq10\n"
     << "citation=arXiv:1805.01238\n"
     << "c=" << spectrum_c << "\n"
     << "k0=" << spectrum_k0 << "\n"
     << "seed=" << protocol_seed << "\n"
     << "nu=" << protocol_nu << "\n"
     << "times=0,0.5,1\n"
     << "order_floor=" << protocol_order_floor << "\n"
     << "cfl_limit=" << cfl_limit << "\n"
     << "assignment=E(|k|)/(4*pi*|k|^2) on 2/3-retained modes\n";
  if (which == "spatial") {
    os << "spatial_n=128,64,32\n"
       << "spatial_dt=" << std::scientific << protocol_dt << "\n"
       << "control_n=64\n"
       << "control_dt=" << protocol_control_dt << "\n";
  } else {
    os << "temporal_n=" << protocol_temporal_n << "\n"
       << "temporal_dt=1/256,1/128,1/64,1/32\n"
       << "reference_dt=" << std::scientific << protocol_temporal_dt[0] << "\n";
  }
}

[[nodiscard]] bool finished(const Resolution &resolution) {
  return resolution.complete;
}

} // namespace

int run_series(std::string_view which, const std::string &outdir, int rank) {
  if (which != "spatial" && which != "temporal") {
    throw std::invalid_argument(
        "incompressible_flow: series must be spatial or temporal");
  }
  std::filesystem::create_directories(outdir);
  write_metadata(std::filesystem::path(outdir) / "metadata.txt", which);
  std::ofstream shells(std::filesystem::path(outdir) / "shells.csv");
  shells << "run,n,dt,time,shell,shell_ke\n";

  std::vector<Row> rows;
  bool ok = true;
  if (which == "spatial") {
    Resolution reference = run_resolution("spatial", protocol_spatial_n[0],
                                          protocol_dt, true, nullptr, shells, rank);
    ok = ok && finished(reference);
    rows.insert(rows.end(), reference.rows.begin(), reference.rows.end());
    const std::vector<Row> *ref_rows =
        reference.complete ? &reference.rows : nullptr;
    Resolution n64;
    for (std::size_t i = 1; i < protocol_spatial_n.size(); ++i) {
      const bool keep = protocol_spatial_n[i] == 64;
      Resolution current = run_resolution("spatial", protocol_spatial_n[i],
                                          protocol_dt, keep, ref_rows, shells, rank);
      ok = ok && finished(current);
      rows.insert(rows.end(), current.rows.begin(), current.rows.end());
      if (keep) n64 = std::move(current);
    }
    Resolution control =
        run_resolution("control", 64, protocol_control_dt, false,
                       n64.complete ? &n64.rows : nullptr, shells, rank);
    ok = ok && finished(control);
    rows.insert(rows.end(), control.rows.begin(), control.rows.end());
  } else {
    Resolution reference =
        run_resolution("temporal", protocol_temporal_n, protocol_temporal_dt[0],
                       true, nullptr, shells, rank);
    ok = ok && finished(reference);
    rows.insert(rows.end(), reference.rows.begin(), reference.rows.end());
    const std::vector<Row> *ref_rows =
        reference.complete ? &reference.rows : nullptr;
    for (std::size_t i = 1; i < protocol_temporal_dt.size(); ++i) {
      Resolution current =
          run_resolution("temporal", protocol_temporal_n, protocol_temporal_dt[i],
                         false, ref_rows, shells, rank);
      ok = ok && finished(current);
      rows.insert(rows.end(), current.rows.begin(), current.rows.end());
    }
    fill_orders(rows, protocol_temporal_dt[0]);
  }

  if (rank == 0) {
    const char *name = (which == "spatial") ? "spatial.csv" : "temporal.csv";
    write_rows(std::filesystem::path(outdir) / name, rows);
    std::cout << "incompressible_flow status=" << (ok ? "ok" : "failed")
              << std::endl;
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace flow

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file channel.hpp
 * @brief Steady laminar balance on one Fourier × Fourier × Chebyshev rank.
 *
 * Plane Couette is u = z. Plane Poiseuille is u = 1 - z² with a
 * constant streamwise body force 2ν. The same force is a constant
 * pressure gradient for that profile. The wall operator removes the
 * pressure gradient so the normal acceleration vanishes at the
 * Chebyshev ends. No-slip is the Dirichlet value of the velocity.
 * The periodic binary stays periodic. One explicit viscous step
 * uses that acceleration. A parallel profile has no convective
 * term, so the step does not evaluate one. Turbulent channel
 * statistics are not this balance.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/fft/chebyshev.hpp>

namespace flow {

enum class LaminarProfile { couette, poiseuille };

struct ChannelBalance {
  double profile_error{0.0};
  double wall_error{0.0};
  double flow_rate{0.0};
  double flow_spread{0.0};
  double shear_plus{0.0};
  double shear_minus{0.0};
  double shear_spread{0.0};
  double momentum{0.0};
  double max_acceleration{0.0};
  double max_normal{0.0};
  double velocity_change{0.0};
  double step_drift{0.0};
};

[[nodiscard]] inline double column_integral(const std::vector<double> &column) {
  const auto coefficients = pfc::fft::chebyshev_coefficients(column);
  double sum = 0.0;
  for (int k = 0; k < static_cast<int>(coefficients.size()); ++k) {
    if (k % 2 != 0) continue;
    const double mode = static_cast<double>(k);
    const double weight = (k == 0) ? 2.0 : 2.0 / (1.0 - mode * mode);
    sum += coefficients[static_cast<std::size_t>(k)] * weight;
  }
  return sum;
}

[[nodiscard]] inline std::size_t channel_index(int iz, int iy, int ix, int ny,
                                               int nx) {
  return (static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny) +
          static_cast<std::size_t>(iy)) *
             static_cast<std::size_t>(nx) +
         static_cast<std::size_t>(ix);
}

/// One explicit step of the linear viscous balance.
/// The tendency is ν times the mixed Laplacian, plus a constant
/// streamwise body force. The impermeable projection removes the
/// pressure gradient. Dirichlet values are then restored at both
/// Chebyshev ends: `wall_plus` and `wall_minus` on the streamwise
/// velocity, and zero on the other two components. A parallel
/// profile has no convective term, so this step does not evaluate
/// one.
inline void viscous_advance(std::vector<double> &velocity_x,
                            std::vector<double> &velocity_y,
                            std::vector<double> &velocity_z, int nx, int ny,
                            double period_x, double period_y, double nu,
                            double force, double step, double wall_plus,
                            double wall_minus) {
  if (!(nu >= 0.0)) {
    throw std::invalid_argument("channel: viscosity must be non-negative");
  }
  if (!(step > 0.0)) {
    throw std::invalid_argument("channel: step must be positive");
  }
  const auto advance_one = [&](const std::vector<double> &velocity, double body) {
    const auto laplacian =
        pfc::fft::fourier_chebyshev_laplacian(velocity, nx, ny, period_x, period_y);
    std::vector<double> tendency(velocity.size(), 0.0);
    for (std::size_t i = 0; i < velocity.size(); ++i) {
      tendency[i] = nu * laplacian[i] + body;
    }
    return tendency;
  };
  const auto tendency_x = advance_one(velocity_x, force);
  const auto tendency_y = advance_one(velocity_y, 0.0);
  const auto tendency_z = advance_one(velocity_z, 0.0);
  const auto acceleration = pfc::fft::impermeable_acceleration(
      tendency_x, tendency_y, tendency_z, nx, ny, period_x, period_y);
  for (std::size_t i = 0; i < velocity_x.size(); ++i) {
    velocity_x[i] += step * acceleration.x[i];
    velocity_y[i] += step * acceleration.y[i];
    velocity_z[i] += step * acceleration.z[i];
  }
  const auto plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  const int nline = static_cast<int>(velocity_x.size() / plane);
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const auto top = channel_index(0, iy, ix, ny, nx);
      const auto bottom = channel_index(nline - 1, iy, ix, ny, nx);
      velocity_x[top] = wall_plus;
      velocity_x[bottom] = wall_minus;
      velocity_y[top] = 0.0;
      velocity_y[bottom] = 0.0;
      velocity_z[top] = 0.0;
      velocity_z[bottom] = 0.0;
    }
  }
}

[[nodiscard]] inline ChannelBalance laminar_balance(LaminarProfile profile, int nx,
                                                    int ny, int degree, double nu) {
  const double period = 2.0 * std::acos(-1.0);
  const auto z = pfc::fft::chebyshev_lobatto(degree);
  const int nline = static_cast<int>(z.size());
  const std::size_t count = static_cast<std::size_t>(nx) *
                            static_cast<std::size_t>(ny) *
                            static_cast<std::size_t>(nline);
  const bool couette = profile == LaminarProfile::couette;
  const double prescribed_plus = couette ? 1.0 : 0.0;
  const double prescribed_minus = couette ? -1.0 : 0.0;
  const double force = couette ? 0.0 : 2.0 * nu;

  std::vector<double> velocity_x(count, 0.0);
  std::vector<double> velocity_y(count, 0.0);
  std::vector<double> velocity_z(count, 0.0);
  for (int iz = 0; iz < nline; ++iz) {
    const double node = z[static_cast<std::size_t>(iz)];
    const double value = couette ? node : 1.0 - node * node;
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        velocity_x[channel_index(iz, iy, ix, ny, nx)] = value;
      }
    }
  }

  const auto laplacian =
      pfc::fft::fourier_chebyshev_laplacian(velocity_x, nx, ny, period, period);
  std::vector<double> tendency_x(count, 0.0);
  std::vector<double> tendency_y(count, 0.0);
  std::vector<double> tendency_z(count, 0.0);
  for (std::size_t i = 0; i < count; ++i) {
    tendency_x[i] = nu * laplacian[i] + force;
  }
  const auto acceleration = pfc::fft::impermeable_acceleration(
      tendency_x, tendency_y, tendency_z, nx, ny, period, period);
  const auto kept = pfc::fft::impermeable_acceleration(
      velocity_x, velocity_y, velocity_z, nx, ny, period, period);

  ChannelBalance report;
  bool first = true;
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      std::vector<double> column(static_cast<std::size_t>(nline));
      for (int iz = 0; iz < nline; ++iz) {
        column[static_cast<std::size_t>(iz)] =
            velocity_x[channel_index(iz, iy, ix, ny, nx)];
      }
      const double rate = column_integral(column);
      const auto slope = pfc::fft::chebyshev_derivative(column);
      const double plus = slope.front();
      const double minus = slope.back();
      if (first) {
        report.flow_rate = rate;
        report.shear_plus = plus;
        report.shear_minus = minus;
        first = false;
      }
      report.flow_spread =
          std::max(report.flow_spread, std::abs(rate - report.flow_rate));
      report.shear_spread =
          std::max(report.shear_spread, std::abs(plus - report.shear_plus));
      report.shear_spread =
          std::max(report.shear_spread, std::abs(minus - report.shear_minus));
    }
  }
  report.momentum = nu * (report.shear_plus - report.shear_minus) + force * 2.0;

  constexpr double step = 0.1;
  auto stepped_x = velocity_x;
  auto stepped_y = velocity_y;
  auto stepped_z = velocity_z;
  viscous_advance(stepped_x, stepped_y, stepped_z, nx, ny, period, period, nu, force,
                  step, prescribed_plus, prescribed_minus);
  for (int iz = 0; iz < nline; ++iz) {
    const double node = z[static_cast<std::size_t>(iz)];
    const double exact = couette ? node : 1.0 - node * node;
    const bool top = iz == 0;
    const bool bottom = iz == nline - 1;
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const auto index = channel_index(iz, iy, ix, ny, nx);
        report.profile_error =
            std::max(report.profile_error, std::abs(velocity_x[index] - exact));
        report.profile_error =
            std::max(report.profile_error, std::abs(velocity_y[index]));
        report.profile_error =
            std::max(report.profile_error, std::abs(velocity_z[index]));
        report.max_acceleration =
            std::max(report.max_acceleration, std::abs(acceleration.x[index]));
        report.max_acceleration =
            std::max(report.max_acceleration, std::abs(acceleration.y[index]));
        report.max_acceleration =
            std::max(report.max_acceleration, std::abs(acceleration.z[index]));
        report.velocity_change = std::max(
            report.velocity_change, std::abs(kept.x[index] - velocity_x[index]));
        report.velocity_change = std::max(
            report.velocity_change, std::abs(kept.y[index] - velocity_y[index]));
        report.velocity_change = std::max(
            report.velocity_change, std::abs(kept.z[index] - velocity_z[index]));
        if (top || bottom) {
          const double prescribed = top ? prescribed_plus : prescribed_minus;
          report.wall_error =
              std::max(report.wall_error, std::abs(velocity_x[index] - prescribed));
          report.wall_error =
              std::max(report.wall_error, std::abs(velocity_y[index]));
          report.wall_error =
              std::max(report.wall_error, std::abs(velocity_z[index]));
          report.max_normal =
              std::max(report.max_normal, std::abs(acceleration.z[index]));
        }
        report.step_drift = std::max(report.step_drift,
                                     std::abs(stepped_x[index] - velocity_x[index]));
        report.step_drift = std::max(report.step_drift,
                                     std::abs(stepped_y[index] - velocity_y[index]));
        report.step_drift = std::max(report.step_drift,
                                     std::abs(stepped_z[index] - velocity_z[index]));
      }
    }
  }
  return report;
}

} // namespace flow

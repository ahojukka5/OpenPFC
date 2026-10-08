// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#define COUPLED_NATIVE_ALLOCATIONS 1
#include "allocation.hpp"
#include "host.hpp"
#include <cstdio>
#include <hip/hip_runtime.h>
#include <iostream>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <openpfc/runtime/hip/databuffer_hip.hpp>
#include <openpfc/runtime/hip/fd_gradient_device.hpp>
#include <openpfc/runtime/hip/for_each_interior_device.hpp>
namespace coupled_rk {
struct Scatter {
  Model model;
  OPENPFC_HD pfc::sim::gpu::DeviceInc2 rhs(double t, const Local &g) const noexcept {
    auto q = model.rhs(t, g);
    return {q.du, q.dv};
  }
};
inline double verify_native(bool fourth, bool uniform = false, double dt = .01,
                            int steps = 1, bool harmonic = false) {
  constexpr int n = 24, hw = 3;
  auto geometry = domain(n);
  auto dec = decomposition(geometry);
  pfc::data::Field<double, pfc::HIPSpace> u(
      geometry, pfc::decomposition::local_box(dec, rank()), hw);
  pfc::data::Field<double, pfc::HIPSpace> v(
      geometry, pfc::decomposition::local_box(dec, rank()), hw);
  pfc::comm::HaloExchangeOptions opts;
  opts.connectivity = pfc::comm::HaloConnectivity::Full;
  pfc::comm::HaloExchange<pfc::HIPSpace, double> halos({&u, &v}, dec, rank(),
                                                       MPI_COMM_WORLD, opts);
  const double h = geometry.spacing[0];
  const auto extent = u.size3();
  const auto local_index = [&](int i, int j, int k) {
    return std::size_t(i) + std::size_t(extent[0]) * (j + extent[1] * k);
  };
  auto gu = pfc::hip::FDGradientDevice<UGrads>(u.data(), extent[0], extent[1],
                                               extent[2], h, h, h, hw, 6);
  auto gv = pfc::hip::FDGradientDevice<VGrads>(v.data(), extent[0], extent[1],
                                               extent[2], h, h, h, hw, 6);
  auto eval = pfc::hip::create_composite_device<Local>(gu, gv);
  using Buffer = pfc::core::DataBuffer<pfc::backend::HIPTag, double>;
  Buffer out_u(u.size()), out_v(v.size());
  Plane host_u(u.size(), -999), host_v(v.size(), -999);
  out_u.copy_from_host(host_u);
  out_v.copy_from_host(host_v);
  const pfc::sim::gpu::DevicePtrPack2 outputs{out_u.data(), out_v.data()};
  Model model;
  model.first = .31;
  model.nonlinear = .17;
  model.time_source = .23;
  if (harmonic) model.first = model.nonlinear = model.time_source = 0;
  model.manufactured_uniform = uniform;
  const double t = uniform ? 0. : .27;
  auto global_initial = initial(n, uniform);
  Oracle first_gold(global_initial, n, t, dt, fourth, model);
  const Oracle *gold = &first_gold;
  int stage = 0;
  double maximum = 0;
  Pair accepted{Plane(std::size_t(extent[0]) * extent[1] * extent[2]),
                Plane(std::size_t(extent[0]) * extent[1] * extent[2])};
  for (int k = 0; k < extent[2]; ++k)
    for (int j = 0; j < extent[1]; ++j)
      for (int i = 0; i < extent[0]; ++i) {
        auto g = u.global(i, j, k);
        for (int f = 0; f < 2; ++f)
          accepted[f][local_index(i, j, k)] =
              global_initial[f][index(g[0], g[1], g[2], n)];
      }
  auto rhs = [&](double time, auto &input, auto &output) {
    allocation_observer::Scope rhs_scope(allocation_observer::Phase::rhs);
    require(stage < gold->stages, "native extra RK stage");
    require(std::abs(time - gold->times[stage]) < 1e-15, "native wrong stage time");
    auto publish = [&](auto &field, const auto &values, int f) {
      allocation_observer::Scope scope(allocation_observer::Phase::upload);
      field.with_host_view([&](double *ptr, std::size_t count) {
        std::fill(ptr, ptr + count, std::numeric_limits<double>::quiet_NaN());
        for (int k = 0; k < extent[2]; ++k)
          for (int j = 0; j < extent[1]; ++j)
            for (int i = 0; i < extent[0]; ++i) {
              auto g = field.global(i, j, k);
              auto at = index(g[0], g[1], g[2], n);
              maximum = std::max(maximum, std::abs(values[local_index(i, j, k)] -
                                                   gold->inputs[stage][f][at]));
              ptr[field.idx(i, j, k)] = values[local_index(i, j, k)];
            }
      });
      field.sync_to_device();
    };
    publish(u, std::get<0>(input), 0);
    publish(v, std::get<1>(input), 1);
    {
      allocation_observer::Scope scope(allocation_observer::Phase::halo);
      halos.exchange();
    } // Native Faces/edges/corners, not a host-filled fixture.
    auto check_halos = [&](auto &field, int f) {
      allocation_observer::Scope scope(allocation_observer::Phase::audit);
      field.with_host_read([&](const double *ptr, std::size_t) {
        for (int k = -hw; k < extent[2] + hw; ++k)
          for (int j = -hw; j < extent[1] + hw; ++j)
            for (int i = -hw; i < extent[0] + hw; ++i) {
              const double got = ptr[field.idx(i, j, k)];
              require(std::isfinite(got), "native stale poisoned Full halo");
              auto g = field.global(i, j, k);
              maximum = std::max(
                  maximum,
                  std::abs(got -
                           gold->inputs[stage][f][index(g[0], g[1], g[2], n)]));
            }
      });
    };
    check_halos(u, 0);
    check_halos(v, 1);
    {
      allocation_observer::Scope scope(allocation_observer::Phase::dispatch);
      pfc::sim::gpu::for_each_interior_device<Scatter, Local, UGrads, VGrads>(
          Scatter{model}, eval.pod(), outputs, time, extent[0], extent[1],
          extent[2]);
    }
    {
      allocation_observer::Scope scope(allocation_observer::Phase::download);
      out_u.copy_to_host(host_u.data(), host_u.size());
      out_v.copy_to_host(host_v.data(), host_v.size());
    }
    for (int k = -hw; k < extent[2] + hw; ++k)
      for (int j = -hw; j < extent[1] + hw; ++j)
        for (int i = -hw; i < extent[0] + hw; ++i) {
          const auto at = u.idx(i, j, k);
          if (i < 0 || j < 0 || k < 0 || i >= extent[0] || j >= extent[1] ||
              k >= extent[2]) {
            require(host_u[at] == -999 && host_v[at] == -999,
                    "native scatter wrote output ghost");
            continue;
          }
          auto g = u.global(i, j, k);
          auto dense = index(g[0], g[1], g[2], n);
          const double a = host_u[at], b = host_v[at];
          require(std::isfinite(a) && std::isfinite(b), "nonfinite native RHS");
          maximum = std::max({maximum, std::abs(a - gold->slopes[stage][0][dense]),
                              std::abs(b - gold->slopes[stage][1][dense])});
          std::get<0>(output)[local_index(i, j, k)] = a;
          std::get<1>(output)[local_index(i, j, k)] = b;
        }
    ++stage;
  };
  pfc::sim::steppers::MultiExplicitRKStepper<decltype(rhs), 2> rk(
      dt, {accepted[0].size(), accepted[1].size()}, tableau(fourth), rhs);
  // Field host mirrors, producer runtime and halo resources warm before the
  // observed production step. Reset inputs; this warm-up is not an accepted step.
  Pair initial_local = accepted;
  rk.step(t, accepted[0], accepted[1]);
  accepted = initial_local;
  stage = 0;
  const auto before = allocation_observer::snapshot();
  allocation_observer::observe(true);
  rk.step(t, accepted[0], accepted[1]);
  allocation_observer::observe(false);
  const auto after = allocation_observer::snapshot();
  allocation_observer::report(before, after);
  require(stage == gold->stages, "native missing RK stage");
  auto compare_accepted = [&] {
    for (int k = 0; k < extent[2]; ++k)
      for (int j = 0; j < extent[1]; ++j)
        for (int i = 0; i < extent[0]; ++i) {
          auto g = u.global(i, j, k);
          auto at = index(g[0], g[1], g[2], n);
          for (int f = 0; f < 2; ++f)
            maximum = std::max(maximum, std::abs(accepted[f][local_index(i, j, k)] -
                                                 gold->result[f][at]));
        }
  };
  compare_accepted();
  require(maximum < 2e-11,
          "observed native RK differs from independent stage/state oracle");
  std::cerr << "observed native numerical error=" << maximum << '\n';
  require(after.phases[std::size_t(allocation_observer::Phase::stepper)] ==
                  before.phases[std::size_t(allocation_observer::Phase::stepper)] &&
              after.phases[std::size_t(allocation_observer::Phase::rhs)] ==
                  before.phases[std::size_t(allocation_observer::Phase::rhs)],
          "C++ allocation in warmed RK orchestration or unclassified RHS");
  Pair global_accepted = first_gold.result;
  for (int step = 1; step < steps; ++step) {
    Oracle expected(global_accepted, n, t + step * dt, dt, fourth, model);
    gold = &expected;
    stage = 0;
    rk.step(t + step * dt, accepted[0], accepted[1]);
    compare_accepted();
    global_accepted = expected.result;
  }
  require(maximum < 2e-11,
          "native RK differs from independent CPU stencil/stage recurrence");
  if (uniform) {
    double temporal = 0;
    for (int f = 0; f < 2; ++f)
      for (double value : accepted[f])
        temporal = std::max(temporal, std::abs(value - std::exp((f == 0 ? 1. : -1.) *
                                                                (t + steps * dt))));
    return temporal;
  }
  std::cout << "successful warmed native stepper/unclassified RHS requests=0; "
               "total observed C++ requests="
            << after.total - before.total << '\n';
  return maximum;
}
} // namespace coupled_rk
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int result = 0;
  try {
    int devices = 0;
    coupled_rk::require(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
                        "actual HIP hardware required");
    coupled_rk::require(hipSetDevice(coupled_rk::rank() % devices) == hipSuccess,
                        "select actual HIP device");
    hipDeviceProp_t prop{};
    coupled_rk::require(
        hipGetDeviceProperties(&prop, coupled_rk::rank() % devices) == hipSuccess,
        "device metadata");
    std::fprintf(
        stderr,
        "native RK rank=%d visible=%d selected=%d device=%s architecture=%s\n",
        coupled_rk::rank(), devices, coupled_rk::rank() % devices, prop.name,
        prop.gcnArchName);
    coupled_rk::require(allocation_observer::verify_provider(),
                        "native allocation provider verification");
    for (bool fourth : {false, true}) {
      std::cout << (fourth ? "RK4" : "RK2")
                << " native independent nonlinear stage/RHS/halo/state error="
                << coupled_rk::verify_native(fourth) << '\n';
      std::cout << (fourth ? "RK4" : "RK2")
                << " native bare harmonic stage/state error="
                << coupled_rk::verify_native(fourth, false, .01, 1, true) << '\n';
      double coarse = coupled_rk::verify_native(fourth, true, .05, 4),
             fine = coupled_rk::verify_native(fourth, true, .025, 8);
      coupled_rk::require(coarse / fine > (fourth ? 12. : 3.5),
                          "native nonlinear-time temporal order");
      coupled_rk::require(fine < (fourth ? 1e-7 : 1e-4),
                          "native nonlinear-time manufactured tolerance");
      std::cout << (fourth ? "RK4" : "RK2")
                << " native nonlinear-time coarse=" << coarse << " fine=" << fine
                << " ratio=" << coarse / fine << '\n';
    }
  } catch (const std::exception &e) {
    allocation_observer::observe(false);
    std::cerr << e.what() << '\n';
    result = 1;
  }
  MPI_Finalize();
  return result;
}

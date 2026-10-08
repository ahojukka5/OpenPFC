// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fixtures/central_derivatives_cases.hpp>
#include <iostream>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#if defined(__HIPCC__) || defined(__HIP__)
using DerivativeBackend = pfc::backend::HIPTag;
#else
using DerivativeBackend = pfc::backend::CUDATag;
#endif
namespace {
using namespace derivative_cases;
template <int Order, class Sample>
__global__ void compute(Sample s, double hx, double hy, double hz, Result *out) {
  if (!threadIdx.x && !blockIdx.x)
    *out = pfc::field::fd::evaluate<Order>(s, hx, hy, hz);
}
template <int Order, class Sample>
void parity(Sample s, double hx, double hy, double hz, double tolerance) {
  pfc::core::DataBuffer<DerivativeBackend, Result> out(1);
  compute<Order><<<1, 1>>>(s, hx, hy, hz, out.data());
  GPU_CHECK(pfc::gpuGetLastError());
  GPU_CHECK(pfc::gpuDeviceSynchronize());
  auto actual = entries(out.to_host()[0]);
  auto host = entries(pfc::field::fd::evaluate<Order>(s, hx, hy, hz)),
       exact = entries(s.exact());
  for (int j = 0; j < 9; ++j) {
    REQUIRE(std::isfinite(actual[j]));
    REQUIRE(actual[j] == Catch::Approx(host[j]).margin(1e-11));
    REQUIRE(actual[j] == Catch::Approx(exact[j]).margin(tolerance));
  }
}
} // namespace
TEST_CASE("Actual device central derivatives match analytic fields and host",
          "[fd][derivatives][device]") {
  int devices = 0;
#if defined(__HIPCC__) || defined(__HIP__)
  REQUIRE(hipGetDeviceCount(&devices) == hipSuccess);
  REQUIRE(devices > 0);
  hipDeviceProp_t properties{};
  REQUIRE(hipGetDeviceProperties(&properties, 0) == hipSuccess);
  std::cout << "Derivative actual device: " << properties.name << " / "
            << properties.gcnArchName << '\n';
#else
  REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess);
  REQUIRE(devices > 0);
#endif
  Polynomial p;
  parity<2>(p, p.hx, p.hy, p.hz, 1e-11);
  parity<4>(p, p.hx, p.hy, p.hz, 1e-11);
  parity<6>(p, p.hx, p.hy, p.hz, 1e-11);
  Periodic t;
  t.h = .08;
  parity<2>(t, t.h, t.h, t.h, .004);
  parity<4>(t, t.h, t.h, t.h, 1e-5);
  parity<6>(t, t.h, t.h, t.h, 1e-7);
  Reflected b{64};
  const double h = 3.14159265358979323846 / b.n;
  parity<2>(b, h, h, h, .001);
  parity<4>(b, h, h, h, 1e-6);
  parity<6>(b, h, h, h, 1e-8);
}

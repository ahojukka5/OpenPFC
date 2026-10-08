// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "generated_rhs.hpp"
#include <algorithm>
#include <cmath>
#include <hip/hip_runtime.h>
#include <iostream>
#include <openpfc/runtime/hip/for_each_interior_device.hpp>
#include <stdexcept>
#include <vector>

namespace {
void check(hipError_t err) {
  if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
}
struct U {
  double xx{}, yy{}, zz{}, xy{}, yz{};
};
struct V {
  double value{}, xz{};
};
struct Local {
  U u;
  V v;
};
struct Parameters {
  double a{}, b{}, c{}, d{};
};
struct Position {
  double p[3]{};
  OPENPFC_HD double operator[](int i) const { return p[i]; }
};
struct Case {
  double t{}, rate{};
  Position x;
  Local g;
  Parameters p;
};
__global__ void evaluate(const Case *cases, pfc::sim::gpu::DevicePtrPack2 outputs,
                         int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto c = cases[i];
    const auto inc = coupled::generated::rhs<pfc::sim::gpu::DeviceInc2>(
        c.t, c.x, c.g, c.p, c.rate);
    pfc::sim::gpu::detail::scatter_device(outputs, inc, i);
  }
}
} // namespace
int main() {
  try {
    int count = 0;
    check(hipGetDeviceCount(&count));
    if (count == 0)
      throw std::runtime_error("native HIP verification requires an actual device");
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, 0));
    constexpr int n = 256;
    std::vector<Case> cases(n);
    for (int i = 0; i < n; ++i) {
      auto &c = cases[i];
      c.t = (i - 128) / 16.;
      c.rate = i < 128 ? 0 : .125;
      c.x = {{(i % 11) / 8., -(i % 7) / 16., (i % 13) / 4.}};
      c.g = {{i / 16., -i / 32., (i % 19) / 8., (i % 23) / 16., -(i % 29) / 32.},
             {(i % 31) / 16., (i % 37) / 32.}};
      c.p = {.3, -.04, .2, -2. / 15.};
    }
    Case *dc = nullptr;
    double *a = nullptr, *b = nullptr;
    check(hipMalloc(&dc, n * sizeof(Case)));
    check(hipMalloc(&a, (n + 2) * sizeof(double)));
    check(hipMalloc(&b, (n + 2) * sizeof(double)));
    std::vector<double> sentinel(n + 2, -999);
    check(hipMemcpy(dc, cases.data(), n * sizeof(Case), hipMemcpyHostToDevice));
    check(hipMemcpy(a, sentinel.data(), sentinel.size() * sizeof(double),
                    hipMemcpyHostToDevice));
    check(hipMemcpy(b, sentinel.data(), sentinel.size() * sizeof(double),
                    hipMemcpyHostToDevice));
    evaluate<<<4, 64>>>(dc, {a, b}, n);
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    std::vector<double> du(n + 2), dv(n + 2);
    check(
        hipMemcpy(du.data(), a, du.size() * sizeof(double), hipMemcpyDeviceToHost));
    check(
        hipMemcpy(dv.data(), b, dv.size() * sizeof(double), hipMemcpyDeviceToHost));
    double worst = 0;
    for (int i = 0; i < n; ++i) {
      const auto &c = cases[i];
      const auto cpu = coupled::generated::rhs<pfc::sim::gpu::DeviceInc2>(
          c.t, c.x, c.g, c.p, c.rate);
      const long double q =
          (long double)c.rate * c.t * ((long double)c.x.p[0] + c.x.p[1] + c.x.p[2]);
      const long double truth0 =
          (long double)c.g.v.value + (long double)c.p.a * c.g.u.xy + q;
      const long double truth1 =
          (long double)c.p.b * ((long double)c.g.u.xx + c.g.u.yy + c.g.u.zz) +
          (long double)c.p.c * c.g.v.xz + (long double)c.p.d * c.g.u.yz + q;
      if (!std::isfinite(du[i]) || !std::isfinite(dv[i]))
        throw std::runtime_error("nonfinite output");
      for (double e :
           {std::abs(du[i] - cpu.v0), std::abs(dv[i] - cpu.v1),
            std::abs(du[i] - (double)truth0), std::abs(dv[i] - (double)truth1)}) {
        worst = std::max(worst, e);
        if (e > 3e-13) throw std::runtime_error("RHS oracle mismatch");
      }
    }
    if (du[n] != -999 || du[n + 1] != -999 || dv[n] != -999 || dv[n + 1] != -999)
      throw std::runtime_error("scatter overwrote output guard");
    check(hipFree(dc));
    check(hipFree(a));
    check(hipFree(b));
    std::cout << prop.name << " / " << prop.gcnArchName << ": " << n
              << " generated CPU/HIP and independent RHS cases, max_abs_error="
              << worst << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

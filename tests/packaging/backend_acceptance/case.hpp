// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#if defined(CHECK_CUDA) || defined(CHECK_HIP)
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <openpfc/runtime/gpu/fft_gpu.hpp>
#include <openpfc/runtime/gpu/observable_reduce_gpu.hpp>
#include <openpfc/runtime/gpu/reflection_ghosts_gpu.hpp>
#endif
#include "identity.hpp"
#include <openpfc/kernel/checkpoint/field_bundle.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/stage_preparation.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/padded_face_flux.hpp>
#include <openpfc/kernel/mpi/collective_check.hpp>
#include <openpfc/solvers/field_operations.hpp>

namespace acceptance {
#if defined(CHECK_CUDA)
using Space = pfc::CUDASpace;
#elif defined(CHECK_HIP)
using Space = pfc::HIPSpace;
#else
using Space = pfc::HostSpace;
#endif
using Field = pfc::data::Field<double, Space>;
using Fields = pfc::solvers::FieldVector<Space>;
using Named = pfc::checkpoint::NamedField<Space>;
using Json = nlohmann::json;
constexpr double pi = 3.14159265358979323846;
constexpr int nx = 12, ny = 10, nz = 8, steps = 8;
constexpr double hx = .7, hy = .9, hz = 1.1, dt = .001, coupling = .3;
constexpr double field_tolerance = 2e-12, ledger_tolerance = 3e-11,
                 fft_tolerance = 2e-12;

inline void require(bool value, const char *message) {
  pfc::mpi::require_all_ranks(value, MPI_COMM_WORLD, message);
}
inline double mode(int i, int j, int k) {
  // Extend the exact cell-centred wall/periodic mode to all halo coordinates.
  return std::cos(pi * (i + .5) / nx) * std::sin(2 * pi * j / ny) *
         std::cos(2 * pi * k / nz);
}
inline std::array<double, 2> amplitudes(int count) {
  double a = .2, b = -.1;
  const double lambda = 2 * (std::cos(pi / nx) - 1) / (hx * hx) +
                        2 * (std::cos(2 * pi / ny) - 1) / (hy * hy) +
                        2 * (std::cos(2 * pi / nz) - 1) / (hz * hz);
  for (int n = 0; n < count; ++n) {
    const double next_a = a + dt * (lambda * a + coupling * (b - a));
    b = b + dt * (2 * lambda * b + coupling * (a - b));
    a = next_a;
  }
  return {a, b};
}
inline auto host_values(const Field &f) {
  std::vector<double> values(f.size());
  f.with_host_read(
      [&](const double *p, std::size_t n) { std::copy_n(p, n, values.data()); });
  return values;
}
inline void seed(Field &f, int component, int count = 0) {
  const auto a = amplitudes(count);
  const auto lo = f.lower_global();
  const auto n = f.local_size();
  f.with_host_view([&](double *p, std::size_t size) {
    std::fill_n(p, size, -99.0);
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i)
          p[f.idx(i, j, k)] =
              1.0 + a[component] * mode(lo[0] + i, lo[1] + j, lo[2] + k);
  });
  f.sync_to_device();
}
#if defined(CHECK_CUDA) || defined(CHECK_HIP)
struct StateReduction {
  const double *v;
  std::size_t sy, sz;
  int halo;
  OPENPFC_HD pfc::sim::gpu::ReductionValues<3> operator()(double u, int i, int j,
                                                          int k) const {
    const double w = v[(i + halo) + (j + halo) * sy + (k + halo) * sz];
    const bool ok = std::isfinite(u) && std::isfinite(w);
    return {{ok ? u + w : 0, ok ? u * u + w * w : 0, ok ? 0.0 : 1.0}};
  }
};

#endif
inline std::array<double, 3> observe(Fields &state) {
#if defined(CHECK_CUDA) || defined(CHECK_HIP)
  using Op = pfc::sim::gpu::ReductionOp;
  const auto &v = state[1];
  const auto r = pfc::sim::gpu::reduce_owned_multi<3>(
      state[0],
      StateReduction{v.data(), std::size_t(v.padded_extent(0)),
                     std::size_t(v.padded_extent(0)) * v.padded_extent(1),
                     v.storage_halo()},
      {{Op::Sum, Op::Sum, Op::Sum}}, MPI_COMM_WORLD);
  return {r[0], r[1], r[2]};
#else
  std::array<double, 3> r{};
  state[0].for_each_owned([&](int i, int j, int k) {
    const double u = state[0](i, j, k), v = state[1](i, j, k);
    if (std::isfinite(u) && std::isfinite(v)) {
      r[0] += u + v;
      r[1] += u * u + v * v;
    } else
      ++r[2];
  });
  pfc::mpi::throw_on_mpi_error(
      MPI_Allreduce(MPI_IN_PLACE, r.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
      "state observables");
  return r;
#endif
}
struct Rhs {
  pfc::field::fd::FaceDivergence<pfc::field::fd::PaddedFaceFlux> du, dv;
  const double *u, *v;
  double *ru, *rv;
  std::ptrdiff_t sy, sz;
  int halo;
  OPENPFC_HD void operator()(int i, int j, int k) const {
    const auto q = (i + halo) + (j + halo) * sy + (k + halo) * sz;
    const double exchange = coupling * (v[q] - u[q]);
    ru[q] = du(i, j, k) + exchange;
    rv[q] = 2 * dv(i, j, k) - exchange;
  }
};
#if defined(CHECK_CUDA) || defined(CHECK_HIP)
__global__ void rhs_kernel(Rhs rhs, int x, int y, int z) {
  const auto q = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q < std::size_t(x) * y * z) rhs(q % x, (q / x) % y, q / (std::size_t(x) * y));
}
#endif
inline void evaluate(Fields &state, Fields &rates, Field &coefficient) {
  const auto n = state[0].local_size(), lo = state[0].lower_global();
  auto walls = pfc::field::fd::model_walls();
  if (lo[0] == 0) walls.xm.policy = pfc::field::fd::WallPolicy::Zero;
  if (lo[0] + n[0] == nx) walls.xp.policy = pfc::field::fd::WallPolicy::Zero;
  const auto uflux = pfc::field::fd::padded_face_flux(coefficient, state[0]);
  const auto vflux = pfc::field::fd::padded_face_flux(coefficient, state[1]);
  Rhs rhs{pfc::field::fd::face_divergence(uflux, n[0], n[1], n[2],
                                          {true, true, true}, hx, hy, hz, walls),
          pfc::field::fd::face_divergence(vflux, n[0], n[1], n[2],
                                          {true, true, true}, hx, hy, hz, walls),
          state[0].data(),
          state[1].data(),
          rates[0].data(),
          rates[1].data(),
          state[0].padded_extent(0),
          std::ptrdiff_t(state[0].padded_extent(0)) * state[0].padded_extent(1),
          1};
#if defined(CHECK_CUDA) || defined(CHECK_HIP)
  const auto count = std::size_t(n[0]) * n[1] * n[2];
  rhs_kernel<<<(count + 255) / 256, 256>>>(rhs, n[0], n[1], n[2]);
  GPU_CHECK(pfc::gpuGetLastError());
  for (int c = 0; c < 2; ++c) rates[c].note_device_write();
#else
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) rhs(i, j, k);
#endif
}
inline double error(const Fields &fields, int count, bool ghosts) {
  const auto a = amplitudes(count);
  double worst = 0;
  for (int c = 0; c < 2; ++c) {
    const auto &f = fields[c];
    const auto n = f.local_size(), lo = f.lower_global();
    const auto data = host_values(f);
    const int h = ghosts ? 1 : 0;
    for (int k = -h; k < n[2] + h; ++k)
      for (int j = -h; j < n[1] + h; ++j)
        for (int i = -h; i < n[0] + h; ++i) {
          const double expected = 1.0 + a[c] * mode(lo[0] + i, lo[1] + j, lo[2] + k);
          const double got = data[f.idx(i, j, k)];
          worst = std::isfinite(got) ? std::max(worst, std::abs(got - expected))
                                     : INFINITY;
        }
  }
  MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return worst;
}
inline void fft_check(const pfc::Domain &domain, int rank, int ranks) {
  const auto decomposition = pfc::decomposition::create(domain, ranks);
  for (int axis = 0; axis < 3; ++axis) {
#if defined(CHECK_CUDA)
    auto fft = pfc::fft::create_cuda(decomposition, rank, MPI_COMM_WORLD, axis);
#elif defined(CHECK_HIP)
    auto fft = pfc::fft::create_hip(decomposition, rank, MPI_COMM_WORLD, axis);
#else
    auto fft = pfc::fft::create(decomposition, rank, MPI_COMM_WORLD, axis);
#endif
    const auto box = fft.get_inbox_bounds();
    std::vector<double> original(fft.size_inbox());
    std::size_t q = 0;
    // Exercise both evolved fields and every spatial direction, independently
    // of the finite-difference partition or its padding.
    for (int component = 0; component < 2; ++component) {
      q = 0;
      const auto a = amplitudes(steps);
      for (int k = box.low[2]; k <= box.high[2]; ++k)
        for (int j = box.low[1]; j <= box.high[1]; ++j)
          for (int i = box.low[0]; i <= box.high[0]; ++i)
            original[q++] = 1.0 + a[component] * mode(i, j, k);
#if defined(CHECK_CUDA) || defined(CHECK_HIP)
      using Tag = pfc::memory_space_to_backend_t<Space>;
      pfc::core::DataBuffer<Tag, double> input(original.size()),
          restored(original.size());
      pfc::core::DataBuffer<Tag, std::complex<double>> spectrum(fft.size_outbox());
      input.copy_from_host(original);
      fft.forward(input, spectrum);
      fft.backward(spectrum, restored);
      const auto result = restored.to_host();
#else
      std::vector<std::complex<double>> spectrum(fft.size_outbox());
      std::vector<double> result(original.size());
      fft.forward(original, spectrum);
      fft.backward(spectrum, result);
#endif
      bool ok = true;
      for (std::size_t i = 0; i < result.size(); ++i)
        ok = ok && std::isfinite(result[i]) &&
             std::abs(result[i] - original[i]) < fft_tolerance;
      require(ok, "FFT round trip differs");
    }
  }
}
inline bool prepare_device(int rank, int ranks, Json &identity) {
#if defined(CHECK_CUDA) || defined(CHECK_HIP)
  MPI_Comm node = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &node);
  int local_rank = 0, local_ranks = 0;
  MPI_Comm_rank(node, &local_rank);
  MPI_Comm_size(node, &local_ranks);
  int count = 0;
  std::array<unsigned char, 16> bytes{};
#if defined(CHECK_CUDA)
  const bool assigned = std::getenv("CUDA_VISIBLE_DEVICES") != nullptr;
  bool ok = assigned && cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
  const bool assigned =
      std::getenv("ROCR_VISIBLE_DEVICES") || std::getenv("HIP_VISIBLE_DEVICES");
  bool ok = assigned && hipGetDeviceCount(&count) == hipSuccess && count > 0;
#endif
  const int device = count == 1 ? 0 : local_rank;
  ok = ok && device < count;
  if (!pfc::mpi::all_ranks_succeeded(ok, MPI_COMM_WORLD)) {
    MPI_Comm_free(&node);
    return false;
  }
#if defined(CHECK_CUDA)
  cudaDeviceProp properties{};
  int driver = 0, runtime = 0;
  ok = cudaSetDevice(device) == cudaSuccess &&
       cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
       cudaDriverGetVersion(&driver) == cudaSuccess &&
       cudaRuntimeGetVersion(&runtime) == cudaSuccess;
  const auto uuid = properties.uuid;
#else
  hipUUID uuid{};
  hipDeviceProp_t properties{};
  int driver = 0, runtime = 0;
  ok = hipSetDevice(device) == hipSuccess &&
       hipDeviceGetUuid(&uuid, device) == hipSuccess &&
       hipGetDeviceProperties(&properties, device) == hipSuccess &&
       hipDriverGetVersion(&driver) == hipSuccess &&
       hipRuntimeGetVersion(&runtime) == hipSuccess;
#endif
  if (!pfc::mpi::all_ranks_succeeded(ok, MPI_COMM_WORLD)) {
    MPI_Comm_free(&node);
    return false;
  }
  std::copy_n(reinterpret_cast<unsigned char *>(uuid.bytes), 16, bytes.data());
  std::vector<unsigned char> ids(16 * local_ranks);
  MPI_Allgather(bytes.data(), 16, MPI_BYTE, ids.data(), 16, MPI_BYTE, node);
  for (int a = 0; a < local_ranks; ++a)
    for (int b = 0; b < a; ++b)
      ok = ok && !std::equal(ids.begin() + 16 * a, ids.begin() + 16 * (a + 1),
                             ids.begin() + 16 * b);
  MPI_Comm_free(&node);
  if (!pfc::mpi::all_ranks_succeeded(ok, MPI_COMM_WORLD)) return false;
  constexpr char hex[] = "0123456789abcdef";
  std::string uuid_text;
  for (auto b : bytes) {
    uuid_text += hex[b / 16];
    uuid_text += hex[b % 16];
  }
  identity["device_uuid"] = uuid_text;
  identity["device_name"] = properties.name;
  identity["driver_version"] = driver;
  identity["runtime_version"] = runtime;
#if defined(CHECK_CUDA)
  identity["device_architecture"] =
      "sm_" + std::to_string(properties.major * 10 + properties.minor);
#else
  identity["device_architecture"] = properties.gcnArchName;
#endif
  identity["visible_device_count"] = count;
  identity["selected_visible_index"] = device;
  // Probe actual device-buffer traffic, including a self-send for one rank.
  // Run this as a separate launcher invocation before the acceptance case.
  using Tag = pfc::memory_space_to_backend_t<Space>;
  pfc::core::DataBuffer<Tag, double> send(1), receive(1);
  send.copy_from_host(std::vector<double>{double(rank) + .125});
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  const int next = (rank + 1) % ranks, previous = (rank + ranks - 1) % ranks;
  const int result =
      MPI_Sendrecv(send.data(), 1, MPI_DOUBLE, next, 41, receive.data(), 1,
                   MPI_DOUBLE, previous, 41, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ok = result == MPI_SUCCESS;
  if (ok) {
    const auto got = receive.to_host();
    ok = got[0] == double(previous) + .125;
  }
  return pfc::mpi::all_ranks_succeeded(ok, MPI_COMM_WORLD);
#else
  (void)rank;
  (void)ranks;
  identity["device_uuid"] = nullptr;
  return true;
#endif
}
inline int run(int argc, char **argv, int rank, int ranks) {
  if (argc != 3)
    throw std::invalid_argument(
        "usage: backend_acceptance run|resume|preflight BUNDLE");
  Json identity = {{"rank", rank},
                   {"backend", compiled_backend},
                   {"sdk_revision", sdk_revision},
                   {"sdk_state", sdk_state},
                   {"sdk_manifest_sha256", sdk_manifest_sha},
                   {"consumer_source_sha256", consumer_sha},
                   {"consumer_compiler", consumer_compiler},
                   {"consumer_host_compiler", consumer_host_compiler}};
  char host[MPI_MAX_PROCESSOR_NAME];
  int host_length = 0;
  MPI_Get_processor_name(host, &host_length);
  identity["host"] = std::string(host, host_length);
  if (!prepare_device(rank, ranks, identity)) {
    if (rank == 0)
      std::cout
          << Json{{"status", "NOT_QUALIFIED"},
                  {"reason",
                   "assigned distinct devices or device-buffer MPI unavailable"}}
                 .dump()
          << '\n';
    return 77;
  }
  std::cout << "IDENTITY " << identity.dump() << std::endl;
  if (std::string(argv[1]) == "preflight") return 0;
  const bool resume = std::string(argv[1]) == "resume";
  require(resume || std::string(argv[1]) == "run", "unknown mode");
  require(std::string(sdk_state) == "clean", "SDK source is not clean");
  const auto begin = std::chrono::steady_clock::now();
  const auto domain =
      pfc::domain::with_spacing({nx, ny, nz}, {hx, hy, hz}, {false, true, true});
  const auto decomposition = pfc::decomposition::create(domain, ranks);
  const auto box = pfc::decomposition::local_box(decomposition, rank);
  pfc::solvers::FieldOperations<Space> algebra(domain, box, 2, 1, MPI_COMM_WORLD);
  auto state = algebra.make(algebra.size()), rates = algebra.make(algebra.size()),
       candidate = algebra.make(algebra.size());
  Field coefficient(domain, box, 1);
  coefficient.with_host_view(
      [](double *p, std::size_t n) { std::fill_n(p, n, 1.0); });
  coefficient.sync_to_device();
  int accepted = 0;
  if (resume) {
    const auto metadata = pfc::checkpoint::load_field_bundle(
        argv[2], std::vector<Named>{{"u", &state[0]}, {"v", &state[1]}},
        MPI_COMM_WORLD);
    require(metadata.method_identity == "acceptance/coupled-euler/v1" &&
                metadata.accepted_increment == steps &&
                metadata.accepted_time == steps * dt,
            "restart metadata differs");
    accepted = steps;
  } else
    for (int c = 0; c < 2; ++c) seed(state[c], c);
  require(error(state, accepted, false) < field_tolerance,
          "initial/restarted fields differ");
  pfc::comm::HaloExchangeOptions options;
  options.connectivity = pfc::comm::HaloConnectivity::Full;
  options.directions = pfc::halo::presets::Full3D();
  options.exchange_base = 300;
  pfc::comm::HaloExchange<Space, double> exchange(
      std::vector<Field *>{&state[0], &state[1]}, decomposition, rank,
      MPI_COMM_WORLD, options);
  pfc::communication::StagePreparationService<double> stage;
  bool sent = false;
  const auto walls = pfc::communication::reflection_walls<double>(
      pfc::communication::ReflectionParity::Even,
      pfc::communication::BoundaryCentering::CellFace);
  const std::vector<std::string_view> names{"u", "v"};
  for (int c = 0; c < 2; ++c) {
    stage.bind(names[c], [&] {
      if (!sent) {
        exchange.exchange();
        sent = true;
      }
    });
    stage.set_reflection(names[c], walls, state[c]);
  }
  auto prepare = [&] {
    sent = false;
    pfc::communication::StagePreparationRequirements req;
    req.needs_halo_exchange = true;
    req.needs_boundary_update = true;
    req.ordering = pfc::communication::BoundaryHaloOrder::HaloThenBoundary;
    stage.prepare(req, names);
  };
  prepare();
  require(error(state, accepted, true) < field_tolerance,
          "initial full halo including corners differs");
  const auto initial = observe(state);
  require(initial[2] == 0, "initial state is not finite");
  for (int step = 0; step < steps; ++step) {
    prepare();
    evaluate(state, rates, coefficient);
    algebra.copy(state, candidate);
    algebra.axpy(candidate, dt, rates);
    const auto values = observe(candidate);
    require(values[2] == 0, "candidate is not finite");
    require(std::abs(values[0] - initial[0]) < ledger_tolerance,
            "closed boundary mass ledger differs");
    require(values[1] <= initial[1] + ledger_tolerance, "coupled energy grew");
    algebra.copy(candidate, state);
    ++accepted;
    require(error(state, accepted, false) < field_tolerance,
            "coupled stage solution differs");
  }
  prepare();
  const double field_error = error(state, accepted, true);
  require(field_error < field_tolerance,
          "final full halo including corners differs");
  // One rank proposes a NaN. The collective finite gate rejects the attempt;
  // accepted fields (including prepared ghosts) and increment remain exact.
  const auto before_u = host_values(state[0]), before_v = host_values(state[1]);
  algebra.copy(state, candidate);
  if (rank == ranks - 1) {
    candidate[1].with_host_view([&](double *p, std::size_t) {
      p[candidate[1].idx(0, 0, 0)] = std::numeric_limits<double>::quiet_NaN();
    });
    candidate[1].sync_to_device();
  }
  const bool finite = observe(candidate)[2] == 0;
  if (pfc::mpi::all_ranks_succeeded(finite, MPI_COMM_WORLD))
    algebra.copy(candidate, state);
  require(!finite && before_u == host_values(state[0]) &&
              before_v == host_values(state[1]),
          "rejected attempt changed accepted state");
  require(accepted == (resume ? 2 * steps : steps),
          "rejected attempt advanced time");
  if (!resume) {
    const auto saved = pfc::checkpoint::publish_field_bundle(
        argv[2], std::vector<Named>{{"u", &state[0]}, {"v", &state[1]}},
        pfc::checkpoint::FieldBundleState{.accepted_time = accepted * dt,
                                          .accepted_increment = accepted,
                                          .method_identity =
                                              "acceptance/coupled-euler/v1"},
        MPI_COMM_WORLD);
    require(saved.ok, "checkpoint publication failed");
  }
  fft_check(domain, rank, ranks);
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
  Json result = {{"status", "PASS"},
                 {"backend", compiled_backend},
                 {"ranks", ranks},
                 {"field_max_error", field_error},
                 {"field_tolerance", field_tolerance},
                 {"ledger_tolerance", ledger_tolerance},
                 {"fft_tolerance", fft_tolerance},
                 {"restart", resume},
                 {"accepted_increment", accepted},
                 {"rank", rank},
                 {"wall_seconds", elapsed},
                 {"resident_state_rhs_candidate_coefficient_bytes",
                  7 * state[0].size() * sizeof(double)}};
  std::cout << "RESULT " << result.dump() << std::endl;
  return 0;
}
inline int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int result = 0;
  try {
    result = run(argc, argv, rank, ranks);
  } catch (const std::exception &e) {
    std::cerr << "FAIL rank " << rank << ": " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    result = 1;
  }
  MPI_Finalize();
  return result;
}
} // namespace acceptance

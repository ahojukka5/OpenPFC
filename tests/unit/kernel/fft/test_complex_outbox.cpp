// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <string>
#include <vector>

#include <mpi.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/complex_outbox.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/fft_layout.hpp>

using namespace pfc;
using pfc::fft::layout::ComplexOutboxCost;
using pfc::fft::layout::complex_proc_grid_for_r2c;
using pfc::fft::layout::complex_proc_grid_is_legal;
using pfc::fft::layout::default_complex_proc_grid_for_r2c;
using pfc::fft::layout::legal_complex_proc_grids;
using pfc::fft::layout::plan_r2c_outbox_cost;
using pfc::fft::layout::r2c_complex_world_box;
using pfc::fft::layout::real_world_box;
using pfc::fft::layout::select_min_distributed_reshape_complex_grid;
using pfc::fft::layout::slab_r2c_plan_options;

namespace {

struct EnvGuard {
  ~EnvGuard() {
    unsetenv("OPENPFC_FFT_COMPLEX_PROC_GRID");
    unsetenv("OPENPFC_FFT_COMPLEX_OUTBOX");
  }
};

ComplexOutboxCost cost_for(const Int3 &real_size, const Int3 &real_grid,
                           const Int3 &complex_grid, int r2c = 0) {
  const auto real_w = real_world_box(real_size);
  const auto cplx_w = r2c_complex_world_box(real_size, r2c);
  return plan_r2c_outbox_cost(real_w, real_grid, cplx_w, complex_grid, r2c,
                              slab_r2c_plan_options());
}

int tungsten_rounds_per_step(const ComplexOutboxCost &c) {
  // Two ETD backward transforms per accepted tungsten step.
  return 2 * c.predicted_all_peer_rounds_per_cycle;
}

} // namespace

TEST_CASE("heFFTe split_world accepts uneven complex y-slabs",
          "[fft][layout][unit][complex_outbox]") {
  const auto world = r2c_complex_world_box({1200, 1200, 1152}, 0);
  REQUIRE(world.size[0] == 601);
  REQUIRE(world.size[1] == 1200);
  REQUIRE(complex_proc_grid_is_legal(world, Int3{1, 32, 1}, 32));
  REQUIRE(complex_proc_grid_is_legal(world, Int3{2, 16, 1}, 32));
  REQUIRE(complex_proc_grid_is_legal(world, Int3{4, 8, 1}, 32));
  REQUIRE(complex_proc_grid_is_legal(world, Int3{1, 16, 2}, 32));
  REQUIRE(complex_proc_grid_is_legal(world, Int3{1, 1, 32}, 32));
  REQUIRE_FALSE(complex_proc_grid_is_legal(world, Int3{1, 31, 1}, 32));
  const auto boxes = heffte::split_world(world, Int3{1, 32, 1});
  REQUIRE(boxes.size() == 32);
  REQUIRE(boxes.front().size[1] == 38);
  REQUIRE(boxes.back().size[1] == 37);
  REQUIRE(boxes.front().size[2] == 1152);
  REQUIRE(boxes.back().size[2] == 1152);
  long long vol = 0;
  for (const auto &b : boxes) {
    vol += b.count();
  }
  REQUIRE(vol == world.count());
}

TEST_CASE("r2c planner reproduces traced 4 vs 8 all-peer rounds",
          "[fft][layout][unit][complex_outbox]") {
  const Int3 z32{1, 1, 32};
  const auto opts = slab_r2c_plan_options();
  {
    const Int3 n{1152, 1152, 1152};
    const auto yslab = cost_for(n, z32, Int3{1, 32, 1});
    REQUIRE(yslab.n_mpi_reshapes == 1);
    REQUIRE(tungsten_rounds_per_step(yslab) == 4);
    REQUIRE(default_complex_proc_grid_for_r2c(
                z32, r2c_complex_world_box(n, 0), 0) == Int3{1, 32, 1});
  }
  {
    const Int3 n{1216, 1216, 1216};
    const auto yslab = cost_for(n, z32, Int3{1, 32, 1});
    REQUIRE(yslab.n_mpi_reshapes == 1);
    REQUIRE(tungsten_rounds_per_step(yslab) == 4);
  }
  {
    const Int3 n{1200, 1200, 1152};
    const auto current = cost_for(n, z32, Int3{1, 1, 32});
    REQUIRE(current.n_mpi_reshapes == 2);
    REQUIRE(tungsten_rounds_per_step(current) == 8);
    REQUIRE(default_complex_proc_grid_for_r2c(
                z32, r2c_complex_world_box(n, 0), 0) == z32);
  }
}

TEST_CASE("1200×1200×1152 has a lower-round complex outbox",
          "[fft][layout][unit][complex_outbox]") {
  const Int3 n{1200, 1200, 1152};
  const Int3 real_g{1, 1, 32};
  const auto real_w = real_world_box(n);
  const auto cplx_w = r2c_complex_world_box(n, 0);
  const auto opts = slab_r2c_plan_options();

  const auto current = cost_for(n, real_g, Int3{1, 1, 32});
  REQUIRE(current.n_mpi_reshapes == 2);

  for (const Int3 g : {Int3{1, 32, 1}, Int3{2, 16, 1}, Int3{4, 8, 1},
                       Int3{8, 4, 1}, Int3{16, 2, 1}, Int3{32, 1, 1}}) {
    const auto c = cost_for(n, real_g, g);
    REQUIRE(c.n_mpi_reshapes == 1);
    REQUIRE(tungsten_rounds_per_step(c) == 4);
  }
  for (const Int3 g : {Int3{1, 16, 2}, Int3{1, 8, 4}, Int3{1, 4, 8},
                       Int3{1, 2, 16}, Int3{2, 8, 2}}) {
    REQUIRE(cost_for(n, real_g, g).n_mpi_reshapes == 2);
  }

  const Int3 selected = select_min_distributed_reshape_complex_grid(
      real_w, real_g, cplx_w, 0, opts);
  const auto best = cost_for(n, real_g, selected);
  REQUIRE(best.n_mpi_reshapes == 1);
  REQUIRE(selected == Int3{2, 16, 1});
  REQUIRE(best.predicted_all_peer_rounds_per_cycle == 2);
  REQUIRE(current.predicted_all_peer_rounds_per_cycle == 4);
}

TEST_CASE("min_reshape keeps even y-slabs when they are already optimal",
          "[fft][layout][unit][complex_outbox]") {
  const auto opts = slab_r2c_plan_options();
  const Int3 z16{1, 1, 16};
  const Int3 z32{1, 1, 32};
  const Int3 z64{1, 1, 64};
  const Int3 z128{1, 1, 128};
  REQUIRE(select_min_distributed_reshape_complex_grid(
              real_world_box({960, 960, 960}), z16,
              r2c_complex_world_box({960, 960, 960}, 0), 0, opts) ==
          Int3{1, 16, 1});
  REQUIRE(select_min_distributed_reshape_complex_grid(
              real_world_box({1152, 1152, 1152}), z32,
              r2c_complex_world_box({1152, 1152, 1152}, 0), 0, opts) ==
          Int3{1, 32, 1});
  REQUIRE(select_min_distributed_reshape_complex_grid(
              real_world_box({1216, 1216, 1216}), z32,
              r2c_complex_world_box({1216, 1216, 1216}, 0), 0, opts) ==
          Int3{1, 32, 1});
  REQUIRE(select_min_distributed_reshape_complex_grid(
              real_world_box({1536, 1536, 1536}), z64,
              r2c_complex_world_box({1536, 1536, 1536}, 0), 0, opts) ==
          Int3{1, 64, 1});
  REQUIRE(select_min_distributed_reshape_complex_grid(
              real_world_box({1920, 1920, 1920}), z128,
              r2c_complex_world_box({1920, 1920, 1920}, 0), 0, opts) ==
          Int3{1, 128, 1});
}

TEST_CASE("already-optimal y-slabs have no lower-round complex outbox",
          "[fft][layout][unit][complex_outbox]") {
  const Int3 n{1152, 1152, 1152};
  const Int3 real_g{1, 1, 32};
  const auto opts = slab_r2c_plan_options();
  const auto yslab = cost_for(n, real_g, Int3{1, 32, 1});
  REQUIRE(yslab.n_mpi_reshapes == 1);
  const Int3 selected = select_min_distributed_reshape_complex_grid(
      real_world_box(n), real_g, r2c_complex_world_box(n, 0), 0, opts);
  REQUIRE(selected == Int3{1, 32, 1});
  REQUIRE(cost_for(n, real_g, selected).n_mpi_reshapes == yslab.n_mpi_reshapes);
}

TEST_CASE("OPENPFC_FFT_COMPLEX_PROC_GRID overrides fail closed",
          "[fft][layout][unit][complex_outbox]") {
  EnvGuard guard;
  REQUIRE(unsetenv("OPENPFC_FFT_COMPLEX_PROC_GRID") == 0);
  REQUIRE(unsetenv("OPENPFC_FFT_COMPLEX_OUTBOX") == 0);
  auto domain = domain::create(Int3{1200, 1200, 1152});
  auto decomp = decomposition::create(domain, Int3{1, 1, 32});

  {
    auto layout = fft::layout::create(decomp, 0);
    const auto &c0 = fft::layout::get_complex_box(layout, 0);
    REQUIRE(c0.size[1] == 1200);
    REQUIRE(c0.size[2] == 36);
  }

  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_PROC_GRID", "2,16,1", 1) == 0);
  {
    auto layout = fft::layout::create(decomp, 0);
    const auto &c0 = fft::layout::get_complex_box(layout, 0);
    REQUIRE(c0.size[0] == 301);
    REQUIRE(c0.size[1] == 75);
    REQUIRE(c0.size[2] == 1152);
    const auto &c31 = fft::layout::get_complex_box(layout, 31);
    REQUIRE(c31.size[0] == 300);
    REQUIRE(c31.size[2] == 1152);
  }

  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_PROC_GRID", "1,32,1", 1) == 0);
  {
    auto layout = fft::layout::create(decomp, 0);
    const auto &c0 = fft::layout::get_complex_box(layout, 0);
    REQUIRE(c0.size[1] == 38);
    REQUIRE(c0.size[2] == 1152);
  }

  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_PROC_GRID", "1,31,1", 1) == 0);
  REQUIRE_THROWS_AS(fft::layout::create(decomp, 0), std::invalid_argument);
  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_PROC_GRID", "nope", 1) == 0);
  REQUIRE_THROWS_AS(fft::layout::create(decomp, 0), std::invalid_argument);
}

TEST_CASE("OPENPFC_FFT_COMPLEX_OUTBOX=min_reshape selects the planner winner",
          "[fft][layout][unit][complex_outbox]") {
  EnvGuard guard;
  REQUIRE(unsetenv("OPENPFC_FFT_COMPLEX_PROC_GRID") == 0);
  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_OUTBOX", "min_reshape", 1) == 0);
  auto domain = domain::create(Int3{1200, 1200, 1152});
  auto decomp = decomposition::create(domain, Int3{1, 1, 32});
  auto layout = fft::layout::create(decomp, 0);
  const auto &c0 = fft::layout::get_complex_box(layout, 0);
  REQUIRE(c0.size[0] == 301);
  REQUIRE(c0.size[1] == 75);
  REQUIRE(c0.size[2] == 1152);

  auto d1152 = domain::create(Int3{1152, 1152, 1152});
  auto de1152 = decomposition::create(d1152, Int3{1, 1, 32});
  auto l1152 = fft::layout::create(de1152, 0);
  const auto &c1152 = fft::layout::get_complex_box(l1152, 0);
  REQUIRE(c1152.size[1] == 36);
  REQUIRE(c1152.size[2] == 1152);

  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_OUTBOX", "not-a-mode", 1) == 0);
  REQUIRE_THROWS_AS(fft::layout::create(decomp, 0), std::invalid_argument);
}

TEST_CASE("uneven y-slab outbox forward/backward roundtrip",
          "[fft][MPI][grid][complex_outbox]") {
  EnvGuard guard;
  int nproc = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (nproc != 4) {
    SKIP("needs 4 ranks");
  }
  REQUIRE(unsetenv("OPENPFC_FFT_COMPLEX_OUTBOX") == 0);
  auto domain = domain::create(Int3{12, 6, 16});
  auto decomp = decomposition::create(domain, Int3{1, 1, 4});
  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_PROC_GRID", "1,4,1", 1) == 0);
  auto layout = fft::layout::create(decomp, 0);
  const auto &c0 = fft::layout::get_complex_box(layout, 0);
  REQUIRE(c0.size[2] == 16);
  REQUIRE(c0.size[1] != 6);

  auto options = heffte::default_options<heffte::backend::fftw>();
  options.use_pencils = false;
  options.use_reorder = true;
  auto fft = fft::create(layout, rank, options, MPI_COMM_WORLD);
  std::vector<double> in(fft.size_inbox());
  for (std::size_t i = 0; i < in.size(); ++i) {
    in[i] = 0.25 * std::sin(0.1 * static_cast<double>(i + 17 * rank)) + 0.5;
  }
  std::vector<std::complex<double>> hat(fft.size_outbox());
  std::vector<double> out(fft.size_inbox(), 0.0);
  fft.forward(in, hat);
  fft.backward(hat, out);
  for (std::size_t i = 0; i < in.size(); ++i) {
    REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs(in[i], 1.0e-10));
  }
}

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <algorithm>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <concepts>
#include <cstdint>
#include <limits>
#include <openpfc/kernel/decomposition/sparse_vector_ops.hpp>
#include <openpfc/runtime/common/selection_geometry.hpp>
#include <vector>
#if defined(OPENPFC_TEST_TYPED_GATHER_HIP) || defined(OPENPFC_TEST_TYPED_GATHER_CUDA)
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/sparse_vector_ops_gpu.hpp>
#endif

namespace {
template <typename Backend, typename T>
void selected_values(const std::vector<T> &values) {
  pfc::core::DataBuffer<Backend, T> source(values.size());
  source.copy_from_host(values);
  const std::vector<std::size_t> indices{3, 1, 3, 0};
  pfc::core::SparseVector<Backend, T> selected(indices);
  pfc::core::gather(selected, source.data(), source.size());
  const auto result = selected.data().to_host();
  const std::vector<T> expected{values[0], values[1], values[3], values[3]};
  REQUIRE(result.size() == expected.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    if constexpr (std::same_as<T, float>) {
      REQUIRE(std::bit_cast<std::uint32_t>(result[i]) ==
              std::bit_cast<std::uint32_t>(expected[i]));
    } else {
      REQUIRE(result[i] == expected[i]);
    }
  }
  const auto untouched = source.to_host();
  for (std::size_t i = 0; i < untouched.size(); ++i) {
    if constexpr (std::same_as<T, float>)
      REQUIRE(std::bit_cast<std::uint32_t>(untouched[i]) ==
              std::bit_cast<std::uint32_t>(values[i]));
    else
      REQUIRE(untouched[i] == values[i]);
  }
  pfc::core::SparseVector<Backend, T> empty(std::vector<std::size_t>{});
  REQUIRE_NOTHROW(pfc::core::gather(empty, static_cast<const T *>(nullptr), 0));
}
const std::vector<std::uint64_t> identities{
    (std::uint64_t{1} << 53) + 1, std::numeric_limits<std::uint64_t>::max(), 0,
    std::numeric_limits<std::uint64_t>::max() - 1};
const std::vector<float> amplitudes{-0.0f,
                                    std::bit_cast<float>(std::uint32_t{0x7fc12345}),
                                    std::numeric_limits<float>::min(), 0.125f};
} // namespace

TEST_CASE("Compact selection geometry avoids signed ceiling overflow",
          "[sparse][typed]") {
  using pfc::core::detail::selection_blocks;
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  REQUIRE(selection_blocks(0, 1024) == 0);
  REQUIRE(selection_blocks(1025, 1024) == 2);
  REQUIRE(selection_blocks(limit, 1) == std::numeric_limits<int>::max());
  REQUIRE(selection_blocks(limit, 512) == 4194304);
  REQUIRE(selection_blocks(limit, 1024) == 2097152);
  REQUIRE_THROWS_AS(selection_blocks(limit + 1, 256), std::overflow_error);
  REQUIRE_THROWS_AS(selection_blocks(1, 0), std::invalid_argument);
  REQUIRE_THROWS_AS(selection_blocks(1, -1), std::invalid_argument);
}

TEST_CASE("CPU compact selection preserves scalar and identity bits",
          "[sparse][typed]") {
  selected_values<pfc::backend::CPUTag>(identities);
  selected_values<pfc::backend::CPUTag>(amplitudes);
}

#if defined(OPENPFC_TEST_TYPED_GATHER_HIP) || defined(OPENPFC_TEST_TYPED_GATHER_CUDA)
#if defined(OPENPFC_TEST_TYPED_GATHER_HIP)
using SelectedBackend = pfc::backend::HIPTag;
#else
using SelectedBackend = pfc::backend::CUDATag;
#endif
TEST_CASE("Device compact selection preserves full-width identities",
          "[sparse][typed][device]") {
  int devices = 0;
#if defined(OPENPFC_TEST_TYPED_GATHER_HIP)
  REQUIRE(hipGetDeviceCount(&devices) == hipSuccess);
#else
  REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess);
#endif
  REQUIRE(devices > 0);
  selected_values<SelectedBackend>(identities);
  selected_values<SelectedBackend>(amplitudes);
}

template <typename T> void rejected_selection() {
  const std::vector<T> values{T{1}, T{2}, T{3}, T{4}};
  pfc::core::DataBuffer<SelectedBackend, T> source(values.size());
  source.copy_from_host(values);
  pfc::core::SparseVector<SelectedBackend, T> selected(
      std::vector<std::size_t>{0, 9}, std::vector<T>{T{7}, T{8}});
  const auto before = selected.data().to_host();
  REQUIRE_THROWS_WITH(pfc::core::gather(selected, source.data(), source.size()),
                      "gather: index out of bounds");
  REQUIRE(selected.data().to_host() == before);
  REQUIRE_THROWS_WITH(
      pfc::core::gather(selected, static_cast<const T *>(nullptr), source.size()),
      "gather: source is null");
  REQUIRE(selected.data().to_host() == before);
  REQUIRE(source.to_host() == values);
}
TEST_CASE("Device compact selection refuses malformed input before publication",
          "[sparse][typed][device]") {
  rejected_selection<float>();
  rejected_selection<std::uint64_t>();
  rejected_selection<double>();
}
#endif

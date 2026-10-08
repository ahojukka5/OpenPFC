// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/grain_tracking.hpp>

namespace pfc::grain::tracking {
#if defined(OPENPFC_GPU_API_HIP)
using Tag = backend::HIPTag;
#else
using Tag = backend::CUDATag;
#endif
namespace {
template <class T> struct Buffer {
  T *data = nullptr;
  explicit Buffer(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::overflow_error("grain allocation byte count overflow");
    if (n) GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&data), n * sizeof(T)));
  }
  ~Buffer() {
    if (data) (void)gpuFree(data);
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
};
std::size_t storage_size(Grid2D grid, Slot slots) {
  const auto n = cell_count(grid);
  if (grid.nx > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      grid.ny > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument(
        "grain tracking axis exceeds signed coordinate limit");
  if (slots == 0 || slots == unassigned)
    throw std::invalid_argument("grain tracking requires a positive slot count");
  if (n > std::numeric_limits<std::size_t>::max() / slots)
    throw std::overflow_error("grain slot storage size overflow");
  if (grid.connectivity != Connectivity::Four &&
      grid.connectivity != Connectivity::Eight)
    throw std::invalid_argument("unknown grain connectivity");
  return n * slots;
}
template <class T> void check_span(std::span<T> data, std::size_t n) {
  if (data.size() != n || (n && !data.data()))
    throw std::invalid_argument("grain device span does not match slot grid");
}
__device__ bool neighbor(Grid2D g, std::size_t cell, int dx, int dy,
                         std::size_t &result) {
  auto x = static_cast<long long>(cell % g.nx) + dx;
  auto y = static_cast<long long>(cell / g.nx) + dy;
  const auto nx = static_cast<long long>(g.nx), ny = static_cast<long long>(g.ny);
  if (g.periodic_x) x = (x % nx + nx) % nx;
  if (g.periodic_y) y = (y % ny + ny) % ny;
  if (x < 0 || y < 0 || x >= nx || y >= ny) return false;
  result = static_cast<std::size_t>(x) + g.nx * static_cast<std::size_t>(y);
  return true;
}
__global__ void initialize(const std::uint8_t *occupancy, const Id *seeds,
                           Id *labels, std::size_t n) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    labels[i] = occupancy[i] ? seeds[i] : background;
}
__global__ void sweep(Grid2D grid, std::size_t n, const std::uint8_t *occupancy,
                      const Id *before, Id *after, unsigned *flags) {
  const auto cells = grid.nx * grid.ny;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    Id chosen = before[i];
    if (occupancy[i] && chosen == background) {
      const auto plane = (i / cells) * cells;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if ((dx == 0 && dy == 0) ||
              (grid.connectivity == Connectivity::Four && dx != 0 && dy != 0))
            continue;
          std::size_t adjacent;
          if (neighbor(grid, i % cells, dx, dy, adjacent)) {
            const Id candidate = before[plane + adjacent];
            if (candidate != background &&
                (chosen == background || candidate < chosen))
              chosen = candidate;
          }
        }
    }
    after[i] = chosen;
    if (chosen != before[i]) atomicOr(flags, 1u);
    if (occupancy[i] && chosen == background) atomicOr(flags, 2u);
  }
}
template <class T> __global__ void zero(T *data, std::size_t n) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    data[i] = 0;
}
__device__ std::size_t find_id(const Id *ids, std::size_t n, Id id) {
  std::size_t first = 0, last = n;
  while (first < last) {
    const auto mid = first + (last - first) / 2;
    if (ids[mid] < id)
      first = mid + 1;
    else
      last = mid;
  }
  return first < n && ids[first] == id ? first : n;
}
__global__ void contacts(Grid2D grid, Slot slots, const Id *labels, const Id *ids,
                         std::size_t grains, unsigned *matrix, unsigned *active,
                         unsigned *error, int radius) {
  const auto cells = grid.nx * grid.ny, n = cells * slots;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const Id id = labels[i];
    if (id == background) continue;
    const auto a = find_id(ids, grains, id);
    if (a == grains) {
      atomicOr(error, 1u);
      continue;
    }
    atomicOr(active + a, 1u);
    for (int dy = -radius; dy <= radius; ++dy)
      for (int dx = -radius; dx <= radius; ++dx) {
        const auto distance = static_cast<std::size_t>(dx < 0 ? -dx : dx) +
                              static_cast<std::size_t>(dy < 0 ? -dy : dy);
        if (grid.connectivity == Connectivity::Four &&
            distance > static_cast<std::size_t>(radius))
          continue;
        std::size_t adjacent;
        if (!neighbor(grid, i % cells, dx, dy, adjacent)) continue;
        for (Slot slot = 0; slot < slots; ++slot) {
          const Id other = labels[static_cast<std::size_t>(slot) * cells + adjacent];
          if (other == background || other == id) continue;
          const auto b = find_id(ids, grains, other);
          if (b == grains) {
            atomicOr(error, 1u);
            continue;
          }
          const auto low = a < b ? a : b, high = a < b ? b : a;
          atomicOr(matrix + low * grains + high, 1u);
        }
      }
  }
}
__global__ void pack(const unsigned *matrix, const Id *ids, std::size_t n,
                     Contact *edges, std::size_t capacity, std::size_t *count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  std::size_t found = 0;
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = a + 1; b < n; ++b)
      if (matrix[a * n + b]) {
        if (found < capacity) {
          edges[found].first = ids[a];
          edges[found].second = ids[b];
        }
        ++found;
      }
  *count = found;
}
constexpr unsigned blocks = 128, threads = 256;
} // namespace

PropagationResult propagate(Tag, Grid2D grid, Slot slots,
                            std::span<const std::uint8_t> occupancy,
                            std::span<const Id> seeds, std::span<Id> output,
                            std::size_t max_sweeps) {
  const auto n = storage_size(grid, slots);
  check_span(occupancy, n);
  check_span(seeds, n);
  check_span(output, n);
  Buffer<Id> first(n), second(n);
  Buffer<unsigned> flags(1);
  GPU_LAUNCH_KERNEL(initialize, blocks, threads,
                    (occupancy.data(), seeds.data(), first.data, n), 0);
  Id *before = first.data, *after = second.data;
  PropagationResult result{Status::IterationLimit, 0};
  for (std::size_t step = 0; step < max_sweeps; ++step) {
    GPU_LAUNCH_KERNEL(zero<unsigned>, 1, 1, (flags.data, 1), 0);
    GPU_LAUNCH_KERNEL(sweep, blocks, threads,
                      (grid, n, occupancy.data(), before, after, flags.data), 0);
    unsigned state = 0;
    GPU_CHECK(gpuMemcpy(&state, flags.data, sizeof(state), gpuMemcpyDeviceToHost));
    ++result.sweeps;
    std::swap(before, after);
    if (!(state & 2u)) {
      result.status = Status::Success;
      break;
    }
    if (!(state & 1u)) {
      result.status = Status::Unseeded;
      break;
    }
  }
  if (result.status == Status::Success)
    GPU_CHECK(
        gpuMemcpy(output.data(), before, n * sizeof(Id), gpuMemcpyDeviceToDevice));
  GPU_CHECK(gpuDeviceSynchronize());
  return result;
}

AdjacencyResult adjacency(Tag, Grid2D grid, Slot slots, std::span<const Id> labels,
                          std::span<const Id> identities, std::span<Contact> output,
                          std::size_t contact_radius) {
  const auto n = storage_size(grid, slots);
  if (contact_radius >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument(
        "grain contact radius exceeds signed coordinate limit");
  check_span(labels, n);
  if (output.size() && !output.data())
    throw std::invalid_argument("null grain contact output");
  const auto grains = identities.size();
  // Reuse the independent canonical graph validation for the host registry.
  validate(ContactGraph{std::vector<Id>(identities.begin(), identities.end()), {}});
  if (grains && grains > std::numeric_limits<std::size_t>::max() / grains)
    throw std::overflow_error("grain adjacency matrix size overflow");
  Buffer<Id> ids(grains);
  Buffer<unsigned> matrix(grains * grains), active(grains), error(1);
  Buffer<Contact> edges(output.size());
  Buffer<std::size_t> count(1);
  if (grains)
    GPU_CHECK(gpuMemcpy(ids.data, identities.data(), grains * sizeof(Id),
                        gpuMemcpyHostToDevice));
  GPU_LAUNCH_KERNEL(zero<unsigned>, blocks, threads, (matrix.data, grains * grains),
                    0);
  GPU_LAUNCH_KERNEL(zero<unsigned>, blocks, threads, (active.data, grains), 0);
  GPU_LAUNCH_KERNEL(zero<unsigned>, 1, 1, (error.data, 1), 0);
  GPU_LAUNCH_KERNEL(contacts, blocks, threads,
                    (grid, slots, labels.data(), ids.data, grains, matrix.data,
                     active.data, error.data, static_cast<int>(contact_radius)),
                    0);
  unsigned bad = 0;
  GPU_CHECK(gpuMemcpy(&bad, error.data, sizeof(bad), gpuMemcpyDeviceToHost));
  if (bad) return {Status::UnknownIdentity, 0, {}};
  GPU_LAUNCH_KERNEL(
      pack, 1, 1,
      (matrix.data, ids.data, grains, edges.data, output.size(), count.data), 0);
  std::size_t found = 0;
  GPU_CHECK(gpuMemcpy(&found, count.data, sizeof(found), gpuMemcpyDeviceToHost));
  if (found > output.size()) return {Status::CapacityOverflow, found, {}};
  std::vector<unsigned> observed(grains);
  if (grains)
    GPU_CHECK(gpuMemcpy(observed.data(), active.data, grains * sizeof(unsigned),
                        gpuMemcpyDeviceToHost));
  AdjacencyResult result{Status::Success, found, {}};
  for (std::size_t i = 0; i < grains; ++i)
    if (observed[i]) result.active.push_back(identities[i]);
  if (found)
    GPU_CHECK(gpuMemcpy(output.data(), edges.data, found * sizeof(Contact),
                        gpuMemcpyDeviceToDevice));
  GPU_CHECK(gpuDeviceSynchronize());
  return result;
}
} // namespace pfc::grain::tracking

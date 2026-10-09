// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#include <openpfc/kernel/grain/label_contacts.hpp>
#include <openpfc/runtime/gpu/grain_transfer.hpp>

namespace pfc::grain::distributed {
namespace label_detail {
template <class T> __global__ void zero(T *matrix, std::size_t size) {
  for (auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
       i += std::size_t(gridDim.x) * blockDim.x)
    matrix[i] = 0;
}
struct DeviceVisitor {
  unsigned long long *samples;
  unsigned *matrix;
  std::size_t grains;
  __device__ void present(std::size_t a) { atomicAdd(samples + a, 1ull); }
  __device__ void edge(std::size_t a, std::size_t b) {
    if (a > b) {
      auto t = a;
      a = b;
      b = t;
    }
    atomicOr(matrix + a * grains + b, 1u);
  }
};
template <class T>
__global__ void inspect(LabelHalo halo, Slot slots, const T *labels,
                        const pfc::grain::detail::Assignment *a, std::size_t grains,
                        std::size_t cells, int radius, bool extract,
                        unsigned long long *samples, unsigned *matrix,
                        unsigned *flags) {
  for (auto c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; c < cells;
       c += std::size_t(gridDim.x) * blockDim.x) {
    const auto bad = visit(halo, slots, labels, a, grains, c, radius, extract,
                           DeviceVisitor{samples, matrix, grains});
    if (bad) atomicOr(flags, bad);
  }
}
static __global__ void pack(const pfc::grain::detail::Assignment *a,
                            std::size_t grains, const unsigned *matrix,
                            Contact *edges, std::size_t capacity,
                            unsigned long long *count) {
  if (threadIdx.x || blockIdx.x) return;
  unsigned long long used = 0;
  for (std::size_t i = 0; i < grains; ++i)
    for (std::size_t j = i + 1; j < grains; ++j)
      if (matrix[i * grains + j]) {
        if (used < capacity) edges[used] = {a[i].id, a[j].id};
        ++used;
      }
  *count = used;
}
} // namespace label_detail

/// Fixed-registry scratch owner. Construct outside an accepted-state loop;
/// reconstruct after any UID/slot change. No device allocation occurs in a
/// workspace-backed probe/extraction call. Host compact result vectors allocate.
/// Extraction reservation is optional and explicitly O(grains squared).
template <class Backend> class ContactWorkspace;
template <class Backend, class T>
ContactObservation
label_contacts(LabelHalo halo, const pfc::core::DataBuffer<Backend, T> &labels,
               ContactWorkspace<Backend> &work, std::size_t radius,
               std::uint64_t required_generation, bool extract_edges = false,
               pfc::gpuStream_t stream = nullptr);
template <class Backend> class ContactWorkspace {
  using Assignment = pfc::grain::detail::Assignment;
  Slot slots;
  std::vector<Assignment> registry;
  std::size_t capacity;
  bool extraction;
  pfc::core::DataBuffer<Backend, Assignment> assignments;
  pfc::core::DataBuffer<Backend, unsigned long long> samples, count;
  pfc::core::DataBuffer<Backend, unsigned> flags, matrix;
  pfc::core::DataBuffer<Backend, Contact> edges;
  template <class B, class T>
  friend ContactObservation label_contacts(LabelHalo,
                                           const pfc::core::DataBuffer<B, T> &,
                                           ContactWorkspace<B> &, std::size_t,
                                           std::uint64_t, bool, pfc::gpuStream_t);

public:
  ContactWorkspace(Slot k, std::span<const Grain> grains, bool reserve_edges = false,
                   std::size_t edge_capacity = 4096)
      : slots(k), capacity(edge_capacity), extraction(reserve_edges) {
    auto prepared = pfc::grain::detail::prepare_transfer(k, grains, {});
    if (!k || k == unassigned || prepared.status != TransferStatus::Success)
      throw std::invalid_argument("invalid UID contact registry");
    registry = std::move(prepared.assignments);
    const auto n = registry.size();
    if (reserve_edges &&
        (edge_capacity > SIZE_MAX / sizeof(Contact) ||
         (n && (n > SIZE_MAX / n || n * n > SIZE_MAX / sizeof(unsigned)))))
      throw std::overflow_error("UID adjacency workspace payload overflow");
    assignments = pfc::core::DataBuffer<Backend, Assignment>(n);
    assignments.copy_from_host(registry);
    samples = pfc::core::DataBuffer<Backend, unsigned long long>(n);
    count = pfc::core::DataBuffer<Backend, unsigned long long>(1);
    flags = pfc::core::DataBuffer<Backend, unsigned>(1);
    matrix = pfc::core::DataBuffer<Backend, unsigned>(reserve_edges ? n * n : 0);
    edges =
        pfc::core::DataBuffer<Backend, Contact>(reserve_edges ? edge_capacity : 0);
    GPU_CHECK(pfc::gpuStreamSynchronize(nullptr));
  }
};

/// Synchronous device realization: full UID images stay resident. Probe mode
/// returns only owned support counts/flags. Extraction additionally returns
/// bounded edges, uses the reserved O(grains squared) matrix and a serial pack.
/// Caller supplies a completed Full halo from required_generation, covering
/// radius, whose producer finishes on stream. No identity is propagated.
/// Workspace is tied to its immutable canonical registry; it is non-reentrant.
template <class Backend, class T>
ContactObservation
label_contacts(LabelHalo halo, const pfc::core::DataBuffer<Backend, T> &labels,
               ContactWorkspace<Backend> &work, std::size_t radius,
               std::uint64_t required_generation, bool extract_edges,
               pfc::gpuStream_t stream) {
  label_detail::validate(halo, work.slots, labels.size(), radius,
                         required_generation);
  if (extract_edges && !work.extraction)
    throw std::invalid_argument("UID contact extraction storage not reserved");
  const auto n = work.registry.size();
  if (work.assignments.size() != n || work.samples.size() != n ||
      work.flags.size() != 1 || work.count.size() != 1 ||
      (work.extraction &&
       (work.matrix.size() != n * n || work.edges.size() != work.capacity)))
    throw std::invalid_argument("moved-from or invalid UID contact scratch owner");
  GPU_LAUNCH_KERNEL((label_detail::zero<unsigned long long>), 1, 256,
                    (work.samples.data(), n), stream);
  GPU_LAUNCH_KERNEL((label_detail::zero<unsigned>), 1, 1, (work.flags.data(), 1),
                    stream);
  if (extract_edges && n)
    GPU_LAUNCH_KERNEL((label_detail::zero<unsigned>), 256, 256,
                      (work.matrix.data(), n * n), stream);
  const auto cells = cell_count(halo.partition.local());
  unsigned blocks = unsigned(std::min<std::size_t>((cells - 1) / 256 + 1, 65535));
  GPU_LAUNCH_KERNEL((label_detail::inspect<T>), blocks, 256,
                    (halo, work.slots, labels.data(), work.assignments.data(), n,
                     cells, int(radius), extract_edges, work.samples.data(),
                     work.matrix.data(), work.flags.data()),
                    stream);
  GPU_CHECK(pfc::gpuStreamSynchronize(stream));
  auto observed = work.samples.to_host();
  std::vector<std::uint64_t> support(observed.begin(), observed.end());
  auto out = label_detail::finish(halo, work.registry, support,
                                  work.flags.to_host().front(), radius,
                                  required_generation, work.slots, extract_edges);
  if (extract_edges) {
    GPU_LAUNCH_KERNEL(label_detail::pack, 1, 1,
                      (work.assignments.data(), n, work.matrix.data(),
                       work.edges.data(), work.capacity, work.count.data()),
                      stream);
    GPU_CHECK(pfc::gpuStreamSynchronize(stream));
    auto used = work.count.to_host().front();
    if (used > work.capacity) throw std::length_error("UID contact edge capacity");
    out.edges.resize(std::size_t(used));
    if (used)
      GPU_CHECK(pfc::gpuMemcpy(out.edges.data(), work.edges.data(),
                               used * sizeof(Contact), pfc::gpuMemcpyDeviceToHost));
  }
  return out;
}
/// Convenience allocating call; use ContactWorkspace for repeated probes.
template <class Backend, class T>
ContactObservation
label_contacts(LabelHalo halo, Slot slots,
               const pfc::core::DataBuffer<Backend, T> &labels,
               std::span<const Grain> grains, std::size_t radius,
               std::uint64_t required_generation, bool extract_edges = false,
               std::size_t edge_capacity = 4096, pfc::gpuStream_t stream = nullptr) {
  label_detail::validate(halo, slots, labels.size(), radius, required_generation);
  ContactWorkspace<Backend> work(slots, grains, extract_edges, edge_capacity);
  return label_contacts(halo, labels, work, radius, required_generation,
                        extract_edges, stream);
}
} // namespace pfc::grain::distributed
#endif

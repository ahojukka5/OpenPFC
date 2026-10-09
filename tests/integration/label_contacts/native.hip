// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "fixture.hpp"
#include <openpfc/runtime/gpu/distributed_grain.hpp>
#include <openpfc/runtime/gpu/full_padded_device_halo_gpu.hpp>
#include <openpfc/runtime/gpu/label_contacts.hpp>

__global__ void write_label(double *labels, std::size_t i, double value) {
  labels[i] = value;
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    int devices = 0;
    fixture::require(hipGetDeviceCount(&devices) == hipSuccess && devices,
                     "actual HIP hardware required");
    fixture::require(hipSetDevice(rank % devices) == hipSuccess, "HIP selection");
    hipDeviceProp_t prop{};
    fixture::require(hipGetDeviceProperties(&prop, rank % devices) == hipSuccess,
                     "device metadata");
    std::cerr << "label contacts HIP rank=" << rank << " visible=" << devices
              << " selected=" << rank % devices
              << " architecture=" << prop.gcnArchName << '\n';
    hipStream_t stream{};
    GPU_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    using Backend = pfc::backend::HIPTag;
    fixture::dg::LabelHalo integer_halo{
        {{8, 8, 8, false, false, false, pfc::grain::Connectivity::TwentySix},
         {0, 0, 0},
         {8, 8, 8}},
        3,
        7,
        true};
    std::vector<pfc::grain::Id> integer_input(2 * integer_halo.field_size());
    const auto high = std::numeric_limits<pfc::grain::Id>::max() - 1;
    integer_input[integer_halo.at(2, 2, 2)] = 101;
    integer_input[integer_halo.at(4, 2, 2)] = high;
    std::vector<pfc::grain::Grain> integer_registry{{101, 0, true}, {high, 0, true}};
    pfc::core::DataBuffer<Backend, pfc::grain::Id> integers(integer_input.size());
    integers.copy_from_host(integer_input);
    auto full_uid = fixture::dg::label_contacts(integer_halo, 2, integers,
                                                integer_registry, 2, 7, true);
    fixture::require(full_uid.proximity && full_uid.edges.size() == 1 &&
                         full_uid.edges.front().second == high,
                     "native uint64 UID truncated");
    bool oversized = false;
    try {
      fixture::dg::ContactWorkspace<Backend> impossible(2, integer_registry, true,
                                                        SIZE_MAX);
    } catch (const std::overflow_error &) {
      oversized = true;
    }
    fixture::require(oversized, "native edge-byte capacity overflow accepted");
    for (pfc::grain::Slot slots : {2u, 3u})
      for (auto connectivity :
           {pfc::grain::Connectivity::Six, pfc::grain::Connectivity::TwentySix})
        for (bool periodic : {false, true})
          for (int scenario = 0; scenario < 7; ++scenario) {
            fixture::Case c(scenario, connectivity, periodic);
            c.slots = slots;
            auto domain = pfc::domain::with_spacing({32, 16, 16}, {.13, .27, .41},
                                                    {periodic, periodic, periodic});
            auto decomp =
                pfc::decomposition::create(domain, fixture::processes(ranks));
            auto h = fixture::layout(decomp, rank, c.global);
            pfc::core::DataBuffer<Backend, double> labels(h.field_size() * c.slots);
            labels.copy_from_host(fixture::padded(c, h));
            std::vector<double *> pointers;
            for (pfc::grain::Slot s = 0; s < c.slots; ++s)
              pointers.push_back(labels.data() + s * h.field_size());
            pfc::gpu::DeviceFullHalo<pfc::hip::HIPHaloOps> halo(
                decomp, rank, 3, MPI_COMM_WORLD, c.slots);
            GPU_CHECK(hipStreamSynchronize(nullptr));
            halo.exchange(pointers.data(), stream);
            fixture::dg::ContactWorkspace<Backend> work(c.slots, c.registry, true,
                                                        4096);
            fixture::dg::ContactWorkspace<Backend> empty(c.slots, c.registry, true,
                                                         0);
            auto scan = [&](int radius, bool edges, std::size_t capacity,
                            auto layout) {
              return fixture::dg::label_contacts(
                  layout, labels, capacity ? work : empty, radius, 7, edges, stream);
            };
            fixture::verify(c, h, scan, rank, ranks);
            fixture::dg::ContactWorkspace<Backend> movable(c.slots, c.registry);
            auto survivor = std::move(movable);
            bool moved_rejected = false;
            try {
              fixture::dg::label_contacts(h, labels, movable, 1, 7, false, stream);
            } catch (const std::invalid_argument &) {
              moved_rejected = true;
            }
            fixture::require(moved_rejected,
                             "moved-from scratch owner launched kernels");
            auto survived = fixture::dg::label_contacts(h, labels, survivor, 1, 7,
                                                        false, stream);
            fixture::require(survived.slots == c.slots,
                             "moved-to scratch owner invalid");
            // Repeated probes reuse identical device storage and reset flags/counts.
            pfc::grain::Diagnostics allocations;
            pfc::grain::diagnostics::Scope scope(&allocations);
            auto a = scan(2, false, 4096, h), b = scan(2, false, 4096, h);
            fixture::require(
                allocations.allocations(pfc::grain::diagnostics::Space::Device)
                            .requests == 0 &&
                    a.support.size() == b.support.size() &&
                    a.proximity == b.proximity,
                "workspace probe allocated device buffers or failed reset");
            auto own = c.owned(h.partition);
            auto first = std::find_if(own.begin(), own.end(),
                                      [](auto id) { return id != 0; });
            const auto cells = pfc::grain::cell_count(h.partition.local());
            const auto index =
                first == own.end() ? 0 : std::size_t(first - own.begin());
            const auto cell = index % cells;
            const auto padded =
                (index / cells) * h.field_size() +
                h.at(cell % h.partition.extent[0],
                     (cell / h.partition.extent[0]) % h.partition.extent[1],
                     cell / (h.partition.extent[0] * h.partition.extent[1]));
            for (double invalid : {.5, std::numeric_limits<double>::quiet_NaN(),
                                   9007199254740994.0, 999.0, 307.0}) {
              if (rank == 0) {
                fixture::require(first != own.end(),
                                 "failure control rank lacks UID");
                GPU_LAUNCH_KERNEL(write_label, 1, 1,
                                  (labels.data(), padded, invalid), stream);
              }
              auto rejected = fixture::dg::contact_trigger(
                  [&] { return scan(1, false, 4096, h); }, MPI_COMM_WORLD);
              fixture::require(rejected.status ==
                                   (invalid == 999
                                        ? fixture::dg::Status::UnknownIdentity
                                        : fixture::dg::Status::InvalidInput),
                               "native invalid/unknown/source-slot UID accepted");
            }
            if (rank == 0)
              GPU_LAUNCH_KERNEL(write_label, 1, 1,
                                (labels.data(), padded, double(*first)), stream);
            auto n = pfc::grain::cell_count(h.partition.local());
            std::vector<double> values(own.size());
            for (std::size_t i = 0; i < values.size(); ++i)
              if (own[i]) values[i] = own[i] == 307 ? 0 : -.25;
            pfc::core::DataBuffer<Backend, double> resident(values.size());
            resident.copy_from_host(values);
            pfc::core::DataBuffer<Backend, pfc::grain::Id> identities(own.size());
            identities.copy_from_host(own);
            pfc::grain::remapping::Options options;
            options.epoch = 7;
            options.contact_radius = 2;
            auto plan = fixture::dg::prepare_contacts(
                [&] { return scan(2, true, 4096, h); }, c.registry, c.slots, options,
                MPI_COMM_WORLD);
            if (plan.status == fixture::dg::Status::Success) {
              auto apply = [&](auto grains, auto moves) {
                return pfc::grain::transfer_signed(h.partition.local(), c.slots,
                                                   resident, identities, grains,
                                                   moves, 0, stream);
              };
              auto staged = fixture::dg::stage(plan, MPI_COMM_WORLD, apply);
              fixture::require(staged.status == fixture::dg::Status::Success,
                               "native collective signed transfer");
              auto oracle = pfc::grain::transfer_signed(
                  h.partition.local(), c.slots, std::span<const double>(values),
                  std::span<const pfc::grain::Id>(own), plan.local_grains,
                  plan.local_moves);
              fixture::require(
                  staged.transaction->values.to_host() == oracle.values &&
                      staged.transaction->labels.to_host() == oracle.labels,
                  "native signed amplitude/UID oracle");
              auto failed = fixture::dg::stage(
                  plan, MPI_COMM_WORLD, [&](auto grains, auto moves) {
                    auto result = apply(grains, moves);
                    if (rank == 0)
                      result.status = pfc::grain::TransferStatus::InvalidSupport;
                    return result;
                  });
              fixture::require(failed.status == fixture::dg::Status::StageFailure &&
                                   !failed.transaction,
                               "one-rank device failure accepted");
              fixture::require(resident.to_host() == values &&
                                   identities.to_host() == own,
                               "input GPU fields mutated before publication");
            }
          }
    GPU_CHECK(hipStreamDestroy(stream));
  } catch (const std::exception &e) {
    std::cerr << "rank " << rank << ": " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
  return 0;
}

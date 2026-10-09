// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "fixture.hpp"
#include <iostream>
#include <openpfc/runtime/cpu/detail/grain_tracking.hpp>
#include <openpfc/runtime/gpu/distributed_grain.hpp>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    int devices = 0;
    require(hipGetDeviceCount(&devices) == hipSuccess && devices,
            "actual HIP hardware required");
    require(hipSetDevice(rank % devices) == hipSuccess, "HIP device selection");
    hipDeviceProp_t prop{};
    require(hipGetDeviceProperties(&prop, rank % devices) == hipSuccess,
            "HIP metadata");
    std::cerr << "distributed grain HIP rank=" << rank << " visible=" << devices
              << " selected=" << rank % devices << " device=" << prop.name
              << " architecture=" << prop.gcnArchName << '\n';
    hipStream_t stream{};
    require(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess,
            "nonblocking stream");
    using Backend = pfc::backend::HIPTag;
    for (auto connectivity : {Connectivity::Six, Connectivity::TwentySix}) {
      Fixture f(rank, ranks, connectivity);
      remapping::Options options;
      options.contact_radius = 3;
      options.epoch = 17;
      pfc::core::DataBuffer<Backend, double> values(f.values.size());
      values.copy_from_host(f.values);
      pfc::core::DataBuffer<Backend, Id> labels(f.labels.size());
      labels.copy_from_host(f.labels);
      auto producer = [&] {
        return dg::observe(f.part, f.slots, labels, f.grains, 3, 64, stream);
      };
      auto plan = dg::prepare(producer, f.grains, f.slots, options, MPI_COMM_WORLD);
      require(plan.status == dg::Status::Success,
              "actual HIP distributed summary/plan");
      auto oracle = tracking::reference::contact_graph(f.part.global, f.slots,
                                                       f.global_labels, 3);
      require(plan.snapshot.graph.vertices == oracle.vertices &&
                  plan.snapshot.graph.edges == oracle.edges,
              "HIP compact graph matches independent global CPU oracle");
      require(!plan.global_moves.empty(), "actual GPU slot change");
      auto apply = [&](std::span<const Grain> grains,
                       std::span<const Transfer> moves) {
        return transfer_signed(f.part.local(), f.slots, values, labels, grains,
                               moves, 0, stream);
      };
      auto transaction = dg::stage(plan, MPI_COMM_WORLD, apply);
      require(transaction.status == dg::Status::Success && transaction.transaction,
              "actual HIP collective signed transfer");
      auto result_values = transaction.transaction->values.to_host();
      auto result_labels = transaction.transaction->labels.to_host();
      auto cpu = transfer_signed(f.part.local(), f.slots, f.values, f.labels,
                                 plan.local_grains, plan.local_moves);
      require(result_values == cpu.values && result_labels == cpu.labels,
              "HIP signed amplitude/UID parity including negative and zero owned "
              "values");
      require(values.to_host() == f.values && labels.to_host() == f.labels,
              "device originals untouched before commit");
      auto failure = dg::stage(plan, MPI_COMM_WORLD, [&](auto grains, auto moves) {
        auto staged = apply(grains, moves);
        if (rank == 0) staged.status = TransferStatus::InvalidSupport;
        return staged;
      });
      require(failure.status == dg::Status::StageFailure && !failure.transaction,
              "one-rank failure collectively rejects GPU buffers");
      require(values.to_host() == f.values && labels.to_host() == f.labels,
              "failed device transaction preserves originals");
      auto disabled = options;
      disabled.check_now = false;
      require(dg::prepare(producer, f.grains, f.slots, disabled, MPI_COMM_WORLD)
                      .status == dg::Status::Deferred,
              "HIP disabled control");
      require(dg::prepare(
                  [&] {
                    return dg::observe(f.part, f.slots, labels, f.grains, 3, 64,
                                       stream, 0);
                  },
                  f.grains, f.slots, options, MPI_COMM_WORLD)
                      .status ==
                  (ranks == 1 ? dg::Status::Success : dg::Status::CapacityOverflow),
              "explicit device boundary capacity failure");
      if (rank == 0)
        std::cout << "HIP distributed grain connectivity=" << int(connectivity)
                  << " ranks=" << ranks << " contacts=" << oracle.edges.size()
                  << " moves=" << plan.global_moves.size() << " verified\n";
    }
    require(hipStreamDestroy(stream) == hipSuccess, "destroy stream");
  } catch (const std::exception &e) {
    std::cerr << "rank " << rank << ": " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
  return 0;
}

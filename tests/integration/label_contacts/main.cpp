// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "fixture.hpp"
#include <iostream>
#include <memory>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    fixture::dg::LabelHalo wide{
        {{8, 8, 8, false, false, false, pfc::grain::Connectivity::TwentySix},
         {0, 0, 0},
         {8, 8, 8}},
        3,
        7,
        true};
    std::vector<pfc::grain::Id> integer_labels(2 * wide.field_size());
    const auto high = std::numeric_limits<pfc::grain::Id>::max() - 1;
    integer_labels[wide.at(2, 2, 2)] = 101;
    integer_labels[wide.at(4, 2, 2)] = high;
    std::vector<pfc::grain::Grain> integer_registry{{101, 0, true}, {high, 0, true}};
    auto integer_result = fixture::dg::label_contacts(
        wide, 2, std::span<const pfc::grain::Id>(integer_labels), integer_registry,
        2, 7, true);
    fixture::require(integer_result.proximity && integer_result.edges.size() == 1 &&
                         integer_result.edges.front().second == high,
                     "uint64 UID truncated to historical packed range");
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
            using Field = pfc::data::Field<double, pfc::HostSpace>;
            std::vector<std::unique_ptr<Field>> fields;
            std::vector<Field *> pointers;
            auto input = fixture::padded(c, h);
            for (pfc::grain::Slot s = 0; s < c.slots; ++s) {
              auto field = std::make_unique<Field>(
                  domain, pfc::decomposition::local_box(decomp, rank), 3);
              std::copy_n(input.data() + s * h.field_size(), h.field_size(),
                          field->data());
              pointers.push_back(field.get());
              fields.push_back(std::move(field));
            }
            pfc::comm::HaloExchangeOptions options;
            options.connectivity = pfc::comm::HaloConnectivity::Full;
            pfc::comm::HaloExchange<pfc::HostSpace, double> halo(
                pointers, decomp, rank, MPI_COMM_WORLD, options);
            halo.exchange();
            for (std::size_t s = 0; s < fields.size(); ++s)
              std::copy_n(fields[s]->data(), h.field_size(),
                          input.data() + s * h.field_size());
            auto scan = [&](int radius, bool edges, std::size_t cap, auto layout) {
              return fixture::dg::label_contacts(layout, c.slots,
                                                 std::span<const double>(input),
                                                 c.registry, radius, 7, edges, cap);
            };
            fixture::verify(c, h, scan, rank, ranks);
            auto own = c.owned(h.partition);
            auto it = std::find_if(own.begin(), own.end(),
                                   [](auto id) { return id != 0; });
            for (double invalid : {.5, std::numeric_limits<double>::quiet_NaN(),
                                   9007199254740994.0, 999.0, 307.0}) {
              auto bad = input;
              if (rank == 0 && it != own.end()) {
                auto at = std::size_t(it - own.begin()),
                     n = pfc::grain::cell_count(h.partition.local());
                auto cell = at % n;
                auto p =
                    (at / n) * h.field_size() +
                    h.at(cell % h.partition.extent[0],
                         (cell / h.partition.extent[0]) % h.partition.extent[1],
                         cell / (h.partition.extent[0] * h.partition.extent[1]));
                bad[p] = invalid;
              }
              auto rejected = fixture::dg::contact_trigger(
                  [&] {
                    return fixture::dg::label_contacts(
                        h, c.slots, std::span<const double>(bad), c.registry, 1, 7);
                  },
                  MPI_COMM_WORLD);
              fixture::require(rejected.status ==
                                   (invalid == 999
                                        ? fixture::dg::Status::UnknownIdentity
                                        : fixture::dg::Status::InvalidInput),
                               "invalid/unknown/source-slot UID accepted");
            }
          }
  } catch (const std::exception &e) {
    std::cerr << "rank " << rank << ": " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
  return 0;
}

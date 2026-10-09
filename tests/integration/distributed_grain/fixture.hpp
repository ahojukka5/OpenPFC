// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <map>
#include <openpfc/runtime/common/distributed_grain.hpp>
namespace dg = pfc::grain::distributed;
using namespace pfc::grain;
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
struct Fixture {
  dg::Partition part;
  Slot slots = 4;
  std::vector<Grain> grains{
      {101, 0, true}, {202, 0, true}, {303, 1, true}, {404, 3, true}};
  std::vector<double> values;
  std::vector<Id> seeds, labels, global_labels;
  std::vector<std::uint8_t> occupied;
  Fixture(int rank, int ranks, Connectivity connectivity) {
    int dims[3]{0, 0, 0};
    MPI_Dims_create(ranks, 3, dims);
    int x = rank % dims[0], y = (rank / dims[0]) % dims[1],
        z = rank / (dims[0] * dims[1]);
    part = {{16, 16, 16, true, true, true, connectivity},
            {std::size_t(16 * x / dims[0]), std::size_t(16 * y / dims[1]),
             std::size_t(16 * z / dims[2])},
            {std::size_t(16 / dims[0]), std::size_t(16 / dims[1]),
             std::size_t(16 / dims[2])}};
    auto cells = cell_count(part.local());
    values.resize(cells * slots);
    seeds.resize(values.size());
    labels.resize(values.size());
    occupied.resize(values.size());
    global_labels.resize(4096 * slots);
    auto put = [&](Id uid, Slot slot, int xx, int yy, int zz, bool seed,
                   double value) {
      auto global = std::size_t(xx + 16 * (yy + 16 * zz));
      global_labels[4096 * slot + global] = uid;
      if (xx < int(part.lower[0]) || yy < int(part.lower[1]) ||
          zz < int(part.lower[2]) || xx >= int(part.lower[0] + part.extent[0]) ||
          yy >= int(part.lower[1] + part.extent[1]) ||
          zz >= int(part.lower[2] + part.extent[2]))
        return;
      auto local =
          std::size_t(xx - part.lower[0] +
                      part.extent[0] * (yy - part.lower[1] +
                                        part.extent[1] * (zz - part.lower[2])));
      auto at = cells * slot + local;
      labels[at] = uid;
      seeds[at] = seed ? uid : 0;
      values[at] = value;
      occupied[at] = 1;
    };
    for (int t = 6; t <= 9; ++t)
      put(101, 0, t, connectivity == Connectivity::Six ? 9 : t,
          connectivity == Connectivity::Six ? 9 : t, t == 6,
          t == 7 ? -.125 : .5 + .01 * t);
    put(202, 0, 11, 9, 9, true, .7);
    put(303, 1, 9, 9, 9, true,
        0); // UID-owned exact zero is meaningful signed support.
    put(404, 3, 0, connectivity == Connectivity::Six ? 2 : 0,
        connectivity == Connectivity::Six ? 2 : 0, true, -.25);
    put(404, 3, 15, connectivity == Connectivity::Six ? 2 : 15,
        connectivity == Connectivity::Six ? 2 : 15, false, -.3);
  }
};

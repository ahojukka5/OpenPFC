// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file scalar_field_file.hpp
 * @brief Write one owned scalar brick to an exact path.
 *
 * This is not a time series. A checkpoint or a named final field uses it
 * when the caller already chose the filename. The path is not numbered.
 */

#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/frontend/io/xdmf_binary_series.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/simulation/results_writer_domain.hpp>

namespace pfc::io {

namespace detail {

inline void create_parent(MPI_Comm comm, const std::filesystem::path &file) {
  const auto parent = file.parent_path();
  if (parent.empty()) return;
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  int failed = 0;
  if (rank == 0) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) failed = 1;
  }
  pfc::mpi::throw_on_mpi_error(MPI_Bcast(&failed, 1, MPI_INT, 0, comm),
                               "scalar field file: MPI_Bcast");
  pfc::mpi::throw_on_mpi_error(MPI_Barrier(comm), "scalar field file: MPI_Barrier");
  if (failed != 0) {
    throw std::runtime_error("scalar field file: cannot create '" + parent.string() +
                             "'");
  }
}

inline void agree(MPI_Comm comm, int local_ok, const std::string &error,
                  const std::string &peer) {
  int global_ok = 0;
  pfc::mpi::throw_on_mpi_error(
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm),
      "scalar field file: MPI_Allreduce");
  if (global_ok != 0) return;
  if (local_ok == 0 && !error.empty()) throw std::runtime_error(error);
  throw std::runtime_error(peer);
}

template <typename Space>
void pack_owned(data::Field<double, Space> &field, std::vector<double> &out) {
  const auto n = field.box().size;
  const int nx = n[0];
  const int ny = n[1];
  const int nz = n[2];
  const int halo = field.storage_halo();
  const auto npx = static_cast<std::size_t>(nx + 2 * halo);
  const auto npy = static_cast<std::size_t>(ny + 2 * halo);
  const auto hw = static_cast<std::size_t>(halo);
  out.resize(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
             static_cast<std::size_t>(nz));
  field.with_host_read([&](const double *data, std::size_t) {
    std::size_t q = 0;
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const std::size_t src = (static_cast<std::size_t>(i) + hw) +
                                  (static_cast<std::size_t>(j) + hw) * npx +
                                  (static_cast<std::size_t>(k) + hw) * npx * npy;
          out[q++] = data[src];
        }
      }
    }
  });
}

} // namespace detail

/// @p pack fills the owned brick, x-fastest, with no halo cells.
inline void write_scalar_brick(const Domain &domain, const Box3i &owned,
                               MPI_Comm comm, const std::filesystem::path &path,
                               std::function<void(std::vector<double> &)> pack) {
  if (comm == MPI_COMM_NULL) {
    throw std::invalid_argument("scalar field file: pass an explicit communicator");
  }
  if (!pack) {
    throw std::invalid_argument("scalar field file: no source");
  }
  detail::create_parent(comm, path);
  int local_ok = 1;
  std::string error;
  std::vector<double> packed;
  try {
    pack(packed);
  } catch (const std::exception &ex) {
    local_ok = 0;
    error = ex.what();
  } catch (...) {
    local_ok = 0;
    error = "scalar field file: pack failed";
  }
  detail::agree(comm, local_ok, error,
                "scalar field file: a peer failed while packing");
  BinaryWriter writer(path.string(), comm);
  if (!writer.writes_real()) {
    throw std::invalid_argument(
        "scalar field file: writer does not support real fields");
  }
  apply_writer_domain(writer, domain, owned);
  writer.write(0, pfc::field::FieldView<double>(packed));
}

/// Pack @p field's owned cells and write them to @p path. Collective on @p comm.
template <typename Space>
void write_scalar_brick(data::Field<double, Space> &field, MPI_Comm comm,
                        const std::filesystem::path &path) {
  write_scalar_brick(
      field.domain(), field.box(), comm, path,
      [&](std::vector<double> &out) { detail::pack_owned(field, out); });
}

/// One binary frame described for ParaView. Rank 0 writes; every rank reports
/// failure.
inline void write_scalar_xdmf(MPI_Comm comm, const std::filesystem::path &xdmf,
                              const BinarySeriesGeometry &geometry,
                              const std::string &attribute,
                              const std::string &binary_from_xdmf) {
  if (comm == MPI_COMM_NULL) {
    throw std::invalid_argument("scalar field file: pass an explicit communicator");
  }
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  std::string error;
  if (rank == 0) {
    try {
      write_xdmf_binary_series(xdmf.string(), geometry, {attribute},
                               {{binary_from_xdmf}}, {0.0});
    } catch (const std::exception &ex) {
      error = ex.what();
    } catch (...) {
      error = "scalar field file: XDMF write failed";
    }
  }
  int failed = error.empty() ? 0 : 1;
  pfc::mpi::throw_on_mpi_error(MPI_Bcast(&failed, 1, MPI_INT, 0, comm),
                               "scalar field file: MPI_Bcast");
  if (failed != 0) {
    if (!error.empty()) throw std::runtime_error(error);
    throw std::runtime_error("scalar field file: XDMF write failed on another rank");
  }
}

} // namespace pfc::io

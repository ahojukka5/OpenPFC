// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file diagnostics_series.hpp
 * @brief Rank-0 CSV of named scalar columns.
 *
 * The caller computes each column. This writer records step, time, and
 * those numbers. It does not know what a column means.
 *
 * Construction is collective on `options.comm`. Local checks (path,
 * column names, overwrite) are agreed before any rank opens the file.
 * A null communicator cannot be agreed, so that one check throws on the
 * rank that passed it. Every other mismatch waits for the peers.
 */

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

namespace pfc::io {

struct DiagnosticsSeriesOptions {
  std::filesystem::path path;
  MPI_Comm comm{MPI_COMM_NULL};
  /// Truncate an existing file. The default is to refuse one.
  bool overwrite{false};
};

class DiagnosticsSeries {
public:
  DiagnosticsSeries(std::vector<std::string> columns,
                    DiagnosticsSeriesOptions options)
      : m_columns(std::move(columns)), m_options(std::move(options)) {
    if (m_options.comm == MPI_COMM_NULL) {
      throw std::invalid_argument(
          "diagnostics series: pass an explicit communicator");
    }
    MPI_Comm_rank(m_options.comm, &m_rank);
    m_collective = true;
    std::string error;
    int local_ok = 1;
    try {
      validate_local_();
    } catch (const std::exception &ex) {
      local_ok = 0;
      error = ex.what();
    } catch (...) {
      local_ok = 0;
      error = "diagnostics series: invalid configuration";
    }
    agree_(local_ok, error, "diagnostics series: a peer rejected the configuration");
    agree_same_config_();
    open_();
  }

  DiagnosticsSeries(const DiagnosticsSeries &) = delete;
  DiagnosticsSeries &operator=(const DiagnosticsSeries &) = delete;
  DiagnosticsSeries(DiagnosticsSeries &&) = delete;
  DiagnosticsSeries &operator=(DiagnosticsSeries &&) = delete;

  ~DiagnosticsSeries() {
    try {
      close();
    } catch (...) {
    }
  }

  /// Every rank calls this. Rank 0 writes the row. A short row fails on
  /// every rank before the file is touched.
  void write(int step, double time, const std::vector<double> &values) {
    if (m_closed) {
      throw std::invalid_argument("diagnostics series: write after close");
    }
    int local_ok = values.size() == m_columns.size() ? 1 : 0;
    std::string error;
    if (local_ok == 0) {
      error = "diagnostics series: row has " + std::to_string(values.size()) +
              " values, expected " + std::to_string(m_columns.size());
    }
    agree_(local_ok, error, "diagnostics series: a peer rejected the row");
    if (m_rank == 0) {
      try {
        const std::string line = format_row_(step, time, values);
        if (std::fwrite(line.data(), 1, line.size(), m_file) != line.size() ||
            std::fflush(m_file) != 0) {
          throw std::runtime_error("diagnostics series: write failed for '" +
                                   m_options.path.string() + "'");
        }
      } catch (const std::exception &ex) {
        error = ex.what();
      } catch (...) {
        error = "diagnostics series: write failed";
      }
    }
    broadcast_failure_(error, "diagnostics series: write failed on another rank");
  }

  void close() {
    if (m_closed) return;
    m_closed = true;
    if (!m_collective) return;
    std::string error;
    if (m_rank == 0 && m_file != nullptr) {
      if (std::fclose(m_file) != 0) {
        error =
            "diagnostics series: close failed for '" + m_options.path.string() + "'";
      }
      m_file = nullptr;
    }
    broadcast_failure_(error, "diagnostics series: close failed on another rank");
  }

private:
  std::vector<std::string> m_columns;
  DiagnosticsSeriesOptions m_options;
  std::FILE *m_file{nullptr};
  int m_rank{0};
  bool m_closed{false};
  bool m_collective{false};

  void validate_local_() const {
    if (m_options.path.empty()) {
      throw std::invalid_argument("diagnostics series: path is empty");
    }
    if (m_columns.empty()) {
      throw std::invalid_argument("diagnostics series: need at least one column");
    }
    for (const std::string &name : m_columns) {
      if (name.empty() || name.find_first_of(",\r\n\"") != std::string::npos) {
        throw std::invalid_argument("diagnostics series: column name '" + name +
                                    "' is empty or not a single CSV field");
      }
    }
  }

  [[nodiscard]] std::uint64_t config_fingerprint_() const {
    std::uint64_t sig = 14695981039346656037ull;
    auto mix = [&](std::string_view text) {
      for (unsigned char c : text) {
        sig ^= c;
        sig *= 1099511628211ull;
      }
      sig ^= 0xff;
    };
    mix(m_options.path.generic_string());
    for (const std::string &name : m_columns) mix(name);
    sig ^= m_options.overwrite ? 1ull : 0ull;
    return sig;
  }

  void agree_same_config_() {
    const unsigned long long local = config_fingerprint_();
    unsigned long long lo = 0;
    unsigned long long hi = 0;
    pfc::mpi::throw_on_mpi_error(MPI_Allreduce(&local, &lo, 1,
                                               MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                                               m_options.comm),
                                 "diagnostics series: MPI_Allreduce");
    pfc::mpi::throw_on_mpi_error(MPI_Allreduce(&local, &hi, 1,
                                               MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                                               m_options.comm),
                                 "diagnostics series: MPI_Allreduce");
    if (lo != hi) {
      throw std::runtime_error(
          "diagnostics series: configuration differs across ranks");
    }
  }

  void open_() {
    std::string error;
    if (m_rank == 0) {
      try {
        const auto parent = m_options.path.parent_path();
        if (!parent.empty()) {
          std::filesystem::create_directories(parent);
        }
        if (!m_options.overwrite && std::filesystem::exists(m_options.path)) {
          throw std::runtime_error("diagnostics series: '" +
                                   m_options.path.string() + "' already exists");
        }
        m_file = std::fopen(m_options.path.string().c_str(), "w");
        if (m_file == nullptr) {
          throw std::runtime_error("diagnostics series: cannot open '" +
                                   m_options.path.string() + "'");
        }
        std::string header = "step,time";
        for (const std::string &name : m_columns) header += "," + name;
        header += "\n";
        if (std::fwrite(header.data(), 1, header.size(), m_file) != header.size()) {
          throw std::runtime_error(
              "diagnostics series: cannot write the header of '" +
              m_options.path.string() + "'");
        }
      } catch (const std::exception &ex) {
        error = ex.what();
        if (m_file != nullptr) {
          std::fclose(m_file);
          m_file = nullptr;
        }
      } catch (...) {
        error = "diagnostics series: open failed";
        if (m_file != nullptr) {
          std::fclose(m_file);
          m_file = nullptr;
        }
      }
    }
    broadcast_failure_(error, "diagnostics series: open failed on another rank");
  }

  void agree_(int local_ok, const std::string &error, const std::string &peer) {
    int global_ok = 0;
    pfc::mpi::throw_on_mpi_error(
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, m_options.comm),
        "diagnostics series: MPI_Allreduce");
    if (global_ok != 0) return;
    if (local_ok == 0 && !error.empty()) throw std::runtime_error(error);
    throw std::runtime_error(peer);
  }

  void broadcast_failure_(const std::string &error, const std::string &peer) {
    int failed = error.empty() ? 0 : 1;
    pfc::mpi::throw_on_mpi_error(MPI_Bcast(&failed, 1, MPI_INT, 0, m_options.comm),
                                 "diagnostics series: MPI_Bcast");
    if (failed == 0) return;
    if (!error.empty()) throw std::runtime_error(error);
    throw std::runtime_error(peer);
  }

  [[nodiscard]] static std::string format_row_(int step, double time,
                                               const std::vector<double> &values) {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    os << step << ',' << time;
    for (double v : values) os << ',' << v;
    os << '\n';
    return os.str();
  }
};

} // namespace pfc::io

// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file json_snapshot_fields.hpp
 * @brief Bind a `fields[]` JSON list onto a `SnapshotSeries`.
 *
 * The series itself has no JSON API. This helper reads `{name, data}`
 * entries and calls `add_field`. `.vti` selects VTK and `.bin` selects
 * raw binary.
 */

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include <openpfc/frontend/io/snapshot_series.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

namespace pfc::ui {

[[nodiscard]] inline std::vector<nlohmann::json>
snapshot_field_entries(const nlohmann::json &cfg) {
  const auto &fields = cfg.at("fields");
  if (!fields.is_array()) {
    throw std::invalid_argument("fields: must be an array of {name, data} objects");
  }
  std::vector<nlohmann::json> entries;
  entries.reserve(fields.size());
  for (const auto &entry : fields) {
    if (!entry.is_object() || !entry.contains("name") || !entry.contains("data") ||
        !entry.at("name").is_string() || !entry.at("data").is_string()) {
      throw std::invalid_argument(
          "fields: each entry needs string 'name' and 'data'");
    }
    entries.push_back(entry);
  }
  return entries;
}

/// Register @p field when `fields[]` asks for @p name. Missing `fields` is a no-op.
template <typename Space>
void bind_snapshot_field(io::SnapshotSeries &series, const nlohmann::json &cfg,
                         std::string_view name, data::Field<double, Space> &field) {
  if (!cfg.contains("fields")) return;
  for (const auto &entry : snapshot_field_entries(cfg)) {
    if (entry.at("name").get<std::string>() != name) continue;
    const std::string path = entry.at("data").get<std::string>();
    const auto ext = std::filesystem::path(path).extension().string();
    const std::string owned(name);
    if (ext == ".vti") {
      series.add_field(owned, field, path, io::SnapshotFormat::Vtk);
    } else if (ext == ".bin") {
      series.add_field(owned, field, path, io::SnapshotFormat::Binary);
    } else {
      throw std::invalid_argument("fields: '" + path +
                                  "' must be a .vti or .bin pattern");
    }
  }
}

/// Fail if `fields[]` names a source that was not registered on @p series.
inline void finish_snapshot_fields(const io::SnapshotSeries &series,
                                   const nlohmann::json &cfg) {
  if (!cfg.contains("fields")) return;
  for (const auto &entry : snapshot_field_entries(cfg)) {
    const std::string name = entry.at("name").get<std::string>();
    if (!series.has_field(name)) {
      throw std::invalid_argument("fields: no registered source named '" + name +
                                  "'");
    }
  }
}

} // namespace pfc::ui

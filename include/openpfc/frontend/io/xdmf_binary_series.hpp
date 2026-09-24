// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file xdmf_binary_series.hpp
 * @brief Temporal XDMF sidecar for Fortran-ordered float64 bricks.
 *
 * Headerless `BinaryWriter` dumps are not self-describing. This writes the
 * XDMF 2 collection ParaView opens (File → Open the `.xdmf`). Dimensions
 * are `nz ny nx` so the C-order XDMF layout matches Fortran `x`-fastest
 * files. Origin and spacing are stored as `z y x` in that same axis order.
 * Each binary path is written as supplied: the caller makes it valid from
 * the XDMF file, normally relative to `path`'s parent directory.
 * See `docs/reference/binary_field_io_spec.md` and `scripts/xdmfgen.py`.
 */

#include <cstdio>
#include <filesystem>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pfc::io {

/// Geometry of one raw-binary time series. Origin and spacing are physical.
struct BinarySeriesGeometry {
  int nx{0};
  int ny{0};
  int nz{0};
  double x0{0.0};
  double y0{0.0};
  double z0{0.0};
  double dx{1.0};
  double dy{1.0};
  double dz{1.0};
};

/// Replace XML markup characters. Paths and field names are caller data.
[[nodiscard]] inline std::string xml_text(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (const char c : text) {
    switch (c) {
    case '&': out += "&amp;"; break;
    case '<': out += "&lt;"; break;
    case '>': out += "&gt;"; break;
    case '"': out += "&quot;"; break;
    default: out += c; break;
    }
  }
  return out;
}

/// Write @p text. A failed open, write, or close throws.
inline void write_text_file(const std::filesystem::path &path,
                            std::string_view text) {
  if (path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      throw std::runtime_error("cannot create '" + path.parent_path().string() +
                               "': " + ec.message());
    }
  }
  std::FILE *fp = std::fopen(path.string().c_str(), "w");
  if (fp == nullptr) {
    throw std::runtime_error("cannot open '" + path.string() + "'");
  }
  const std::size_t wrote = std::fwrite(text.data(), 1, text.size(), fp);
  const int io_error = std::ferror(fp);
  const int closed = std::fclose(fp);
  if (wrote != text.size() || io_error != 0 || closed != 0) {
    throw std::runtime_error("cannot write '" + path.string() + "'");
  }
}

inline void write_xdmf_binary_series(
    const std::string &path, const BinarySeriesGeometry &geometry,
    const std::vector<std::string> &field_names,
    const std::vector<std::vector<std::string>> &paths_from_xdmf,
    const std::vector<double> &times) {
  if (field_names.empty() || times.empty()) {
    throw std::invalid_argument("XDMF series: need at least one field and time");
  }
  if (paths_from_xdmf.size() != field_names.size()) {
    throw std::invalid_argument("XDMF series: one path list per field");
  }
  for (const auto &frames : paths_from_xdmf) {
    if (frames.size() != times.size()) {
      throw std::invalid_argument("XDMF series: one path per time for every field");
    }
  }

  auto g = [](double value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", value);
    return std::string(buf);
  };

  std::ostringstream xml;
  xml.imbue(std::locale::classic());
  xml << "<?xml version=\"1.0\" ?>\n"
      << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
      << "<Xdmf Version=\"2.0\">\n"
      << "  <Domain>\n"
      << "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" "
         "CollectionType=\"Temporal\">\n";
  for (std::size_t i = 0; i < times.size(); ++i) {
    char step[32];
    std::snprintf(step, sizeof(step), "step_%04zu", i);
    xml << "      <Grid Name=\"" << step << "\" GridType=\"Uniform\">\n"
        << "        <Time Value=\"" << g(times[i]) << "\"/>\n"
        << "        <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\""
        << geometry.nz << " " << geometry.ny << " " << geometry.nx << "\"/>\n"
        << "        <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
        << "          <DataItem Dimensions=\"3\" NumberType=\"Float\" "
           "Precision=\"8\" Format=\"XML\">"
        << g(geometry.z0) << " " << g(geometry.y0) << " " << g(geometry.x0)
        << "</DataItem>\n"
        << "          <DataItem Dimensions=\"3\" NumberType=\"Float\" "
           "Precision=\"8\" Format=\"XML\">"
        << g(geometry.dz) << " " << g(geometry.dy) << " " << g(geometry.dx)
        << "</DataItem>\n"
        << "        </Geometry>\n";
    for (std::size_t f = 0; f < field_names.size(); ++f) {
      xml << "        <Attribute Name=\"" << xml_text(field_names[f])
          << "\" AttributeType=\"Scalar\" Center=\"Node\">\n"
          << "          <DataItem Dimensions=\"" << geometry.nz << " " << geometry.ny
          << " " << geometry.nx
          << "\" NumberType=\"Float\" Precision=\"8\" Format=\"Binary\" "
             "Endian=\"Little\">"
          << xml_text(paths_from_xdmf[f][i]) << "</DataItem>\n"
          << "        </Attribute>\n";
    }
    xml << "      </Grid>\n";
  }
  xml << "    </Grid>\n  </Domain>\n</Xdmf>\n";
  write_text_file(path, xml.str());
}

} // namespace pfc::io

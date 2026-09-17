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
 * files. See `docs/reference/binary_field_io_spec.md` and `scripts/xdmfgen.py`.
 */

#include <cstdio>
#include <string>
#include <vector>

namespace pfc::io {

inline void write_xdmf_binary_series(
    const std::string &path, int nx, int ny, int nz, double dx, double dy,
    double dz, const std::vector<std::string> &field_names,
    const std::vector<std::vector<std::string>> &rel_paths_per_field,
    const std::vector<double> &times) {
  if (field_names.empty() || times.empty()) return;
  std::FILE *fp = std::fopen(path.c_str(), "w");
  if (fp == nullptr) return;
  std::fprintf(fp,
               "<?xml version=\"1.0\" ?>\n"
               "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
               "<Xdmf Version=\"2.0\">\n"
               "  <Domain>\n"
               "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" "
               "CollectionType=\"Temporal\">\n");
  for (std::size_t i = 0; i < times.size(); ++i) {
    std::fprintf(fp, "      <Grid Name=\"step_%04zu\" GridType=\"Uniform\">\n",
                 i);
    std::fprintf(fp, "        <Time Value=\"%.17g\"/>\n", times[i]);
    std::fprintf(fp,
                 "        <Topology TopologyType=\"3DCORECTMesh\" "
                 "Dimensions=\"%d %d %d\"/>\n",
                 nz, ny, nx);
    std::fprintf(fp, "        <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    std::fprintf(fp,
                 "          <DataItem Dimensions=\"3\" NumberType=\"Float\" "
                 "Precision=\"8\" Format=\"XML\">0 0 0</DataItem>\n");
    std::fprintf(fp,
                 "          <DataItem Dimensions=\"3\" NumberType=\"Float\" "
                 "Precision=\"8\" Format=\"XML\">%.17g %.17g %.17g</DataItem>\n",
                 dz, dy, dx);
    std::fprintf(fp, "        </Geometry>\n");
    for (std::size_t f = 0; f < field_names.size(); ++f) {
      std::fprintf(fp,
                   "        <Attribute Name=\"%s\" AttributeType=\"Scalar\" "
                   "Center=\"Node\">\n",
                   field_names[f].c_str());
      std::fprintf(fp,
                   "          <DataItem Dimensions=\"%d %d %d\" "
                   "NumberType=\"Float\" Precision=\"8\" Format=\"Binary\" "
                   "Endian=\"Little\">%s</DataItem>\n",
                   nz, ny, nx, rel_paths_per_field[f][i].c_str());
      std::fprintf(fp, "        </Attribute>\n");
    }
    std::fprintf(fp, "      </Grid>\n");
  }
  std::fprintf(fp, "    </Grid>\n  </Domain>\n</Xdmf>\n");
  std::fclose(fp);
}

} // namespace pfc::io

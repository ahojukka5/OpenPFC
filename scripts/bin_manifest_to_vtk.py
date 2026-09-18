#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convert OpenPFC showcase .bin + manifest to ParaView VTK (.vti + .pvd).

2-D fields become ImageData with k-extent 0 0 (a plane, not a 1-cell volume).
3-D fields keep a cubic ImageData. Time series is a .pvd collection.
"""
from __future__ import print_function

import json
import os
import struct
import sys


def _load_manifest(path):
    with open(path) as fh:
        return json.load(fh)


def _nsteps(man):
    if man.get("times") is not None:
        return len(man["times"]), [float(t) for t in man["times"]]
    steps = man["steps"]
    return len(steps), [float(s) for s in steps]


def _brick(base, pattern, field, index):
    name = pattern.format(field=field, index=index)
    path = os.path.join(base, name)
    if not os.path.isfile(path):
        raise SystemExit("missing %s" % path)
    return path


def write_vti(path, nx, ny, nz, origin, spacing, arrays):
    """arrays: list of (name, raw_bytes) Fortran-ordered float64 bricks."""
    npts = nx * ny * nz
    header_size = 8
    offsets = []
    off = 0
    for name, blob in arrays:
        if len(blob) != npts * 8:
            raise SystemExit(
                "%s: %s payload %d, expected %d"
                % (path, name, len(blob), npts * 8)
            )
        offsets.append(off)
        off += header_size + len(blob)

    ox, oy, oz = origin
    dx, dy, dz = spacing
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
        '  <ImageData WholeExtent="0 %d 0 %d 0 %d" Origin="%.17g %.17g %.17g" Spacing="%.17g %.17g %.17g">'
        % (nx - 1, ny - 1, nz - 1, ox, oy, oz, dx, dy, dz),
        '    <Piece Extent="0 %d 0 %d 0 %d">' % (nx - 1, ny - 1, nz - 1),
        '      <PointData Scalars="%s">' % arrays[0][0],
    ]
    for (name, blob), offset in zip(arrays, offsets):
        lines.append(
            '        <DataArray type="Float64" Name="%s" format="appended" offset="%d"/>'
            % (name, offset)
        )
    lines.extend(
        [
            "      </PointData>",
            "      <CellData/>",
            "    </Piece>",
            "  </ImageData>",
            '  <AppendedData encoding="raw">',
            "_",
        ]
    )
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "wb") as fh:
        fh.write("\n".join(lines).encode("ascii"))
        for name, blob in arrays:
            fh.write(struct.pack("<Q", len(blob)))
            fh.write(blob)
        fh.write(b"\n  </AppendedData>\n</VTKFile>\n")


def write_pvd(path, rows):
    """rows: list of (time, relative_vti)."""
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for t, rel in rows:
        lines.append(
            '    <DataSet timestep="%.17g" group="" part="0" file="%s"/>' % (t, rel)
        )
    lines.extend(["  </Collection>", "</VTKFile>", ""])
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def convert_uniform(man_path, out_dir=None):
    man_path = os.path.abspath(man_path)
    base = os.path.dirname(man_path)
    man = _load_manifest(man_path)
    if "nvx" in man and "ny" not in man:
        return convert_weibel(man_path, out_dir)
    fields = list(man["fields"])
    pattern = man["pattern"]
    n, times = _nsteps(man)
    nx, ny, nz = int(man["nx"]), int(man["ny"]), int(man.get("nz", 1))
    dx = float(man.get("dx", 1.0))
    dy = float(man.get("dy", dx))
    dz = float(man.get("dz", dx if nz > 1 else 1.0))
    if nz == 1:
        dz = 1.0
    out_dir = os.path.abspath(out_dir or os.path.join(base, "paraview"))
    os.makedirs(out_dir, exist_ok=True)
    expected = 8 * nx * ny * nz
    rows = []
    for i, t in enumerate(times):
        arrays = []
        for field in fields:
            src = _brick(base, pattern, field, i)
            blob = open(src, "rb").read()
            if len(blob) != expected:
                raise SystemExit("%s size %d expected %d" % (src, len(blob), expected))
            arrays.append((field, blob))
        vti_name = "step_%04d.vti" % i
        write_vti(
            os.path.join(out_dir, vti_name),
            nx, ny, nz, (0.0, 0.0, 0.0), (dx, dy, dz), arrays,
        )
        rows.append((t, vti_name))
        if (i + 1) % 50 == 0 or i + 1 == n:
            print("  %s %d/%d" % (out_dir, i + 1, n))
    pvd = os.path.join(out_dir, "timeseries.pvd")
    write_pvd(pvd, rows)
    print("wrote", pvd, "(%d steps, fields: %s)" % (n, ", ".join(fields)))
    return pvd


def convert_weibel(man_path, out_dir=None):
    man_path = os.path.abspath(man_path)
    base = os.path.dirname(man_path)
    man = _load_manifest(man_path)
    pattern = man["pattern"]
    n, times = _nsteps(man)
    nx, nvx, nvy = int(man["nx"]), int(man["nvx"]), int(man["nvy"])
    dx = float(man.get("dx", 1.0))
    dvx = float(man.get("dvx", 1.0))
    dvy = float(man.get("dvy", 1.0))
    out_root = os.path.abspath(out_dir or os.path.join(base, "paraview"))
    specs = [
        ("f_xvx", nx, nvx, dx, dvx, (0.0, -0.5 * nvx * dvx, 0.0)),
        ("f_xvy", nx, nvy, dx, dvy, (0.0, -0.5 * nvy * dvy, 0.0)),
        ("Bz", nx, 1, dx, 1.0, (0.0, 0.0, 0.0)),
        ("Ey", nx, 1, dx, 1.0, (0.0, 0.0, 0.0)),
        ("Jy", nx, 1, dx, 1.0, (0.0, 0.0, 0.0)),
    ]
    written = []
    for field, nxi, nyi, dxi, dyi, origin in specs:
        out_dir = os.path.join(out_root, field)
        os.makedirs(out_dir, exist_ok=True)
        expected = 8 * nxi * nyi
        rows = []
        for i, t in enumerate(times):
            src = _brick(base, pattern, field, i)
            blob = open(src, "rb").read()
            if len(blob) != expected:
                raise SystemExit("%s size %d expected %d" % (src, len(blob), expected))
            vti_name = "step_%04d.vti" % i
            write_vti(
                os.path.join(out_dir, vti_name),
                nxi, nyi, 1, origin, (dxi, dyi, 1.0), [(field, blob)],
            )
            rows.append((t, vti_name))
        pvd = os.path.join(out_dir, "timeseries.pvd")
        write_pvd(pvd, rows)
        print("wrote", pvd, "(%d steps, %s %dx%d)" % (n, field, nxi, nyi))
        written.append(pvd)
    return written


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: bin_manifest_to_vtk.py MANIFEST.json [OUTDIR]")
    man = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    convert_uniform(man, out)


if __name__ == "__main__":
    main()

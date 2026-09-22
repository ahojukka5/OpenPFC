#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Issue #25 heat3d_fd_hip weak ladder: grids and CSV collection.

    python3 apps/heat3d/scripts/fd_weak_ladder.py --check
    python3 apps/heat3d/scripts/fd_weak_ladder.py --collect DIR --out CSV
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import statistics
import sys

# nodes, Nx, Ny, Nz, proc grid. Owned interior is always 256^3 / rank.
LADDER = (
    (1, 512, 512, 512, "2x2x2"),
    (2, 512, 512, 1024, "2x2x4"),
    (4, 512, 1024, 1024, "2x4x4"),
    (8, 1024, 1024, 1024, "4x4x4"),
    (16, 1024, 1024, 2048, "4x4x8"),
)

CSV_FIELDS = (
    "nodes",
    "gcds",
    "Nx",
    "Ny",
    "Nz",
    "cells",
    "cells_per_gcd",
    "proc_grid",
    "local_brick",
    "steps",
    "warmup",
    "job",
    "wall_step_s",
    "weak_eff",
    "gcell_s",
    "mcell_s_per_gcd",
    "halo_s",
    "rhs_s",
    "update_s",
    "halo_minmax",
    "rhs_minmax",
    "update_minmax",
    "gpu_aware",
    "contiguous",
    "revision",
    "dirty",
    "bin_sha256",
    "scratch",
    "notes",
)


def check():
    rc = 0
    print("nodes gcds   Nx   Ny   Nz  grid   local     cells/GCD")
    for nodes, nx, ny, nz, grid in LADDER:
        gcds = nodes * 8
        gx, gy, gz = (int(p) for p in grid.split("x"))
        if gx * gy * gz != gcds:
            print("FAIL grid product", grid, gcds)
            rc = 1
        loc = (nx // gx, ny // gy, nz // gz)
        if loc != (256, 256, 256):
            print("FAIL local", loc)
            rc = 1
        cells = nx * ny * nz
        per = cells / float(gcds)
        print(
            "%5d %4d %4d %4d %4d  %-7s %dx%dx%d  %.3e"
            % (nodes, gcds, nx, ny, nz, grid, loc[0], loc[1], loc[2], per)
        )
    return rc


def _meta_map(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("==="):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v
    return out


def _metric_index(names, key):
    try:
        return names.index(key)
    except ValueError:
        return -1


def wall_step_median(doc, warmup):
    if int(doc.get("schema_version") or 0) == 4 and "metrics" in doc:
        m = doc["metrics"].get("wall_step") or {}
        if "median" in m:
            return float(m["median"])
    names = list(doc.get("frame_metric_names") or [])
    idx = _metric_index(names, "wall_step")
    if idx < 0:
        return None
    values = []
    for rank in doc.get("ranks") or []:
        frames = rank.get("frames") or []
        for i, frame in enumerate(frames):
            if i < warmup:
                continue
            scalars = frame.get("scalars") or []
            if idx < len(scalars):
                values.append(float(scalars[idx]))
    if not values:
        return None
    return float(statistics.median(values))


def _parse_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _kv_line(path, prefix):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith(prefix):
                continue
            out = {}
            for tok in line.split()[1:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    out[k] = v
            return out
    return {}


def collect_run(run, warmup, t1):
    meta = _meta_map(os.path.join(run, "run_meta.txt"))
    prof = os.path.join(run, "timing_profile.json")
    if not os.path.isfile(prof):
        return None
    with open(prof) as f:
        doc = json.load(f)
    wall = wall_step_median(doc, warmup)
    if wall is None:
        return None
    nx = _parse_int(meta.get("Nx"))
    ny = _parse_int(meta.get("Ny"))
    nz = _parse_int(meta.get("Nz"))
    if not (nx and ny and nz):
        return None
    nodes = _parse_int(meta.get("nodes")) or 0
    gcds = _parse_int(meta.get("ntasks")) or (nodes * 8 if nodes else 0)
    cells = nx * ny * nz
    per = float(cells) / float(gcds) if gcds else 0.0
    gcell = (float(cells) / wall / 1.0e9) if wall else 0.0
    mcell_gcd = (float(cells) / float(gcds) / wall / 1.0e6) if wall and gcds else 0.0
    eff = (t1 / wall) if (t1 and wall) else ""
    notes = meta.get("HEAT3D_DIAG_TIMING", "unset")
    if notes == "1":
        notes = "diagnostic"
    else:
        notes = "clean"
    log = os.path.join(run, "run.log")
    decomp = _kv_line(log, "HEAT3D_FD_DECOMP")
    diag = _kv_line(log, "HEAT3D_DIAG")
    proc = decomp.get("proc_grid") or meta.get("OPENPFC_FD_PROC_GRID", "")
    local = decomp.get("local_min") or "256x256x256"
    return {
        "nodes": nodes,
        "gcds": gcds,
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "cells": cells,
        "cells_per_gcd": "%.6e" % per,
        "proc_grid": proc,
        "local_brick": local,
        "steps": meta.get("steps", ""),
        "warmup": warmup,
        "job": meta.get("job", os.path.basename(run)),
        "wall_step_s": "%.8f" % wall,
        "weak_eff": ("%.4f" % eff) if eff != "" else "",
        "gcell_s": "%.6f" % gcell,
        "mcell_s_per_gcd": "%.3f" % mcell_gcd,
        "halo_s": diag.get("halo_s", ""),
        "rhs_s": diag.get("rhs_s", ""),
        "update_s": diag.get("update_s", ""),
        "halo_minmax": diag.get("halo_minmax", ""),
        "rhs_minmax": diag.get("rhs_minmax", ""),
        "update_minmax": diag.get("update_minmax", ""),
        "gpu_aware": decomp.get("gpu_aware", ""),
        "contiguous": decomp.get("contiguous", ""),
        "revision": meta.get("revision", ""),
        "dirty": meta.get("dirty", ""),
        "bin_sha256": meta.get("bin_sha256", ""),
        "scratch": run,
        "notes": notes,
        "_wall": wall,
        "_nodes": nodes,
    }


def collect(root, out, warmup):
    runs = []
    root_runs = os.path.join(root, "runs") if os.path.isdir(os.path.join(root, "runs")) else root
    if os.path.isdir(root_runs):
        for name in sorted(os.listdir(root_runs)):
            path = os.path.join(root_runs, name)
            if os.path.isdir(path):
                runs.append(path)
    elif os.path.isdir(root):
        runs.append(root)
    rows = []
    for path in runs:
        try:
            row = collect_run(path, warmup, None)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            sys.stderr.write("skip %s: %s\n" % (path, exc))
            continue
        if row:
            rows.append(row)
    t1 = None
    for row in rows:
        if row["_nodes"] == 1 and row["notes"] == "clean":
            t1 = row["_wall"]
            break
    if t1 is None:
        for row in rows:
            if row["_nodes"] == 1:
                t1 = row["_wall"]
                break
    out_rows = []
    for row in rows:
        if t1:
            row["weak_eff"] = "%.4f" % (t1 / row["_wall"])
        row.pop("_wall", None)
        row.pop("_nodes", None)
        out_rows.append(row)
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, "w") as f:
        f.write("# Issue #25 heat3d_fd_hip LUMI-G weak scaling (256^3 interior/GCD).\n")
        f.write("# Collected from scratch run dirs. Do not interpolate.\n")
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print("wrote %s (%d rows)" % (out, len(out_rows)))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--collect")
    p.add_argument("--out", default="out/report/data/heat3d_fd_lumi_g_weak.csv")
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    if args.check:
        return check()
    if args.collect:
        return collect(args.collect, args.out, args.warmup)
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

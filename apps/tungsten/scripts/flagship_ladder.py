#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Issue #13 weak-scaling ladder: grids, FFT layout, and CSV collection.

Does not invent timings. --check is arithmetic and needs no binary.
--collect reads scratch run directories and writes the compact CSV from
timing_profile.json plus run_meta.txt. Compatible with Python 3.6.

    python3 apps/tungsten/scripts/flagship_ladder.py --check
    python3 apps/tungsten/scripts/flagship_ladder.py --collect DIR --out CSV
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import statistics
import sys
import tempfile

# nodes, N  (8 GCD / node). Issue #13 candidates; 1200^3 is kept because
# OPENPFC_FFT_NODE_GRID=1 is 1x8xnnodes (1200 % 8 == 0 and 1200 % 4 == 0).
LADDER = (
    (1, 768),
    (2, 960),
    (4, 1200),
    (8, 1536),
    (16, 1920),
    (32, 2400),
    (60, 3000),
)

CSV_FIELDS = (
    "nodes",
    "gcds",
    "N",
    "cells",
    "cells_per_gcd",
    "steps",
    "warmup",
    "job",
    "wall_step_s",
    "rss_per_rank_gib",
    "proc_grid",
    "revision",
    "dirty",
    "bin_sha256",
    "fft_node_grid",
    "fft_proc_grid",
    "local_brick",
    "use_pencils",
    "input",
    "notes",
)

CSV_HEADER_COMMENTS = (
    "# Issue #13 LUMI-G tungsten_hip weak-scaling ladder.\n"
    "# Written by flagship_ladder.py --collect from scratch run dirs.\n"
    "# Do not interpolate or invent timings.\n"
    "# Protocol: docs/lumi_slurm/tungsten_hip_scaling.toml (I/O off, dt=1).\n"
    "# Multi-node default: 1D slab when N divides nproc; 1x8xN if not.\n"
    "# Scratch only for bulky logs.\n"
    "# Efficiency is wall_step(1 node) / wall_step(N) (weak; similar cells/GCD).\n"
    "# rss_per_rank_gib is host RSS from the profiler, not GCD HBM.\n"
    "# Quote device memory from a HIP_MEM / hipMemGetInfo line, never RSS.\n"
    "# local_brick is the owned real-space brick Nx/gx x Ny/gy x Nz/gz.\n"
)


def is_5_smooth(n):
    x = n
    for p in (2, 3, 5):
        while x % p == 0:
            x //= p
    return x == 1


def node_aware_grid(n, nproc):
    if nproc < 9 or nproc % 8 != 0:
        return (0, 0, 0)
    nnodes = nproc // 8
    if n % 8 == 0 and n % nnodes == 0:
        return (1, 8, nnodes)
    if n % nnodes == 0 and n % 8 == 0:
        return (1, nnodes, 8)
    return (0, 0, 0)


def legal_proc_grids(n, nproc):
    """Cartesian grids that factor nproc and divide a cubic N^3."""
    grids = []
    gx = 1
    while gx <= nproc:
        if nproc % gx == 0 and n % gx == 0:
            rest = nproc // gx
            gy = 1
            while gy <= rest:
                if rest % gy == 0 and n % gy == 0:
                    gz = rest // gy
                    if gz >= 1 and n % gz == 0 and gx * gy * gz == nproc:
                        grids.append((gx, gy, gz))
                gy += 1
        gx += 1
    return grids


def print_legal_grids(n, nproc):
    grids = legal_proc_grids(n, nproc)
    node = node_aware_grid(n, nproc)
    print("N=%d nproc=%d legal=%d 5-smooth=%s slab=%s cells/GCD=%.3e" % (
        n, nproc, len(grids), "yes" if is_5_smooth(n) else "no",
        "yes" if n % nproc == 0 else "no", n ** 3 / float(nproc)))
    for gx, gy, gz in grids:
        tags = []
        if (gx, gy, gz) == node:
            tags.append("node-aware")
        if gx == 1 and gy == 1:
            tags.append("slab-z")
        if gx == 1 and gz == 1:
            tags.append("slab-y")
        if gy == 1 and gz == 1:
            tags.append("slab-x")
        if gx == 1 and gy > 1 and gz > 1:
            tags.append("r2c-pencil")
        if gx == gy == gz:
            tags.append("cube")
        split = sum(1 for v in (gx, gy, gz) if v > 1)
        if split == 3:
            tags.append("brick-3d")
        tag = (" " + " ".join(tags)) if tags else ""
        print(
            "  %dx%dx%d  local=%dx%dx%d%s"
            % (gx, gy, gz, n // gx, n // gy, n // gz, tag)
        )
    return 0 if grids else 1


def factor_list(n):
    x = n
    out = []
    p = 2
    while p * p <= x:
        while x % p == 0:
            out.append(p)
            x //= p
        p = 3 if p == 2 else p + 2
    if x > 1:
        out.append(x)
    return out


def print_candidates(nodes, lo_rel=0.84, hi_rel=1.16, window=160):
    """Nearby 5-smooth / slab-legal N for a weak-scaling point."""
    nproc = nodes * 8
    # Frozen 1-node 768^3 / 8 GCD band.
    ref = (768 ** 3) / 8.0
    target = int(round(ref * nproc) ** (1.0 / 3.0))
    print(
        "nodes=%d nproc=%d target~%d ref_cells/GCD=%.3e band=[%.0f%%, %.0f%%]"
        % (nodes, nproc, target, ref, 100 * lo_rel, 100 * hi_rel)
    )
    print(
        "%5s %3s %5s %10s %7s  %s"
        % ("N", "5s", "slab", "cells/GCD", "rel", "factors")
    )
    n0 = max(32, target - window)
    n1 = target + window
    n = n0 if n0 % 2 == 0 else n0 + 1
    while n <= n1:
        per = (n ** 3) / float(nproc)
        rel = per / ref
        slab = n % nproc == 0
        smooth = is_5_smooth(n)
        in_band = lo_rel <= rel <= hi_rel
        if in_band and (slab or smooth):
            print(
                "%5d %3s %5s %10.2e %+6.1f%%  %s"
                % (
                    n,
                    "yes" if smooth else "no",
                    "yes" if slab else "no",
                    per,
                    100.0 * (rel - 1.0),
                    "x".join(str(p) for p in factor_list(n)),
                )
            )
        n += 2
    return 0


def default_grid(n, nproc):
    """Match spectral_fft_proc_grid without OPENPFC_FFT_NODE_GRID."""
    if nproc < 9:
        return (2, 2, 2) if nproc == 8 and n % 2 == 0 else (1, 1, 1)
    if n % nproc == 0:
        return (1, 1, nproc)
    return node_aware_grid(n, nproc)


def check():
    rc = 0
    print("nodes  gcds     N   cells/GCD   5-smooth  default     1x8xN")
    for nodes, n in LADDER:
        gcds = nodes * 8
        cells = n ** 3
        per = cells / float(gcds)
        node = node_aware_grid(n, gcds) if gcds >= 9 else (0, 0, 0)
        grid = default_grid(n, gcds)
        ok_smooth = is_5_smooth(n)
        if gcds >= 9:
            ok_grid = grid != (0, 0, 0)
        else:
            ok_grid = n % 8 == 0
        mark = "ok" if ok_smooth and ok_grid else "FAIL"
        if mark != "ok":
            rc = 1
        print(
            "%5d %5d %5d %10.2e   %-8s %s  %s  %s"
            % (nodes, gcds, n, per, "yes" if ok_smooth else "no", grid, node, mark)
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


def _scalar_values(doc, name, warmup):
    names = list(doc.get("frame_metric_names") or [])
    idx = _metric_index(names, name)
    if idx < 0:
        return []
    values = []
    for rank in doc.get("ranks") or []:
        frames = rank.get("frames") or []
        for i, frame in enumerate(frames):
            if i < warmup:
                continue
            scalars = frame.get("scalars") or []
            if idx < len(scalars):
                values.append(float(scalars[idx]))
    return values


def wall_step_median(doc, warmup):
    if int(doc.get("schema_version") or 0) == 4 and "metrics" in doc:
        m = doc["metrics"].get("wall_step") or {}
        if "median" in m:
            return float(m["median"])
    vals = _scalar_values(doc, "wall_step", warmup)
    if not vals:
        return None
    return float(statistics.median(vals))


def rss_per_rank_gib(doc, warmup):
    if int(doc.get("schema_version") or 0) == 4 and "metrics" in doc:
        m = doc["metrics"].get("rss_bytes") or {}
        if "median" in m:
            return float(m["median"]) / (1024.0 ** 3)
    vals = _scalar_values(doc, "rss_bytes", warmup)
    if not vals:
        return None
    return float(statistics.median(vals)) / (1024.0 ** 3)


def _find_profile(run):
    for name in ("timing_profile.json", "timing_profile.json.gz"):
        p = os.path.join(run, name)
        if os.path.isfile(p):
            return p
    return None


def _parse_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def collect_run(run, warmup):
    try:
        return _collect_run_unchecked(run, warmup)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        sys.stderr.write("skip %s: %s\n" % (run, exc))
        return None


def _collect_run_unchecked(run, warmup):
    meta = _meta_map(os.path.join(run, "run_meta.txt"))
    prof_path = _find_profile(run)
    if prof_path is None:
        return None
    with open(prof_path) as f:
        doc = json.load(f)
    wall = wall_step_median(doc, warmup)
    if wall is None:
        return None
    n = _parse_int(meta.get("N")) or _parse_int(meta.get("Lx"))
    if not n:
        toml = os.path.join(run, "input.toml")
        if os.path.isfile(toml):
            with open(toml) as f:
                for line in f:
                    if line.strip().startswith("Lx"):
                        n = _parse_int(line.split("=", 1)[1].strip())
                        break
    if not n:
        return None
    nodes = _parse_int(meta.get("nodes")) or 0
    gcds = _parse_int(meta.get("ntasks")) or (nodes * 8 if nodes else 0)
    if nodes <= 0 and gcds:
        nodes = max(1, gcds // 8)
    cells = n ** 3 if n else 0
    per = (float(cells) / float(gcds)) if gcds else 0.0
    rss = rss_per_rank_gib(doc, warmup)
    row = {
        "nodes": nodes,
        "gcds": gcds,
        "N": n,
        "cells": cells,
        "cells_per_gcd": "%.6e" % per,
        "steps": meta.get("steps", ""),
        "warmup": warmup,
        "job": meta.get("job", os.path.basename(run)),
        "wall_step_s": "%.8g" % wall,
        "rss_per_rank_gib": "" if rss is None else "%.6g" % rss,
        "proc_grid": meta.get("proc_grid", ""),
        "revision": meta.get("revision", ""),
        "dirty": meta.get("dirty", ""),
        "bin_sha256": meta.get("bin_sha256", ""),
        "fft_node_grid": meta.get("OPENPFC_FFT_NODE_GRID", ""),
        "fft_proc_grid": meta.get("OPENPFC_FFT_PROC_GRID", ""),
        "local_brick": meta.get("local_brick", ""),
        "use_pencils": meta.get("use_pencils", ""),
        "input": meta.get("input", os.path.join(run, "input.toml")),
        "notes": meta.get("notes", os.path.basename(run)),
    }
    return row


def collect(root, out_path, warmup):
    runs = []
    if os.path.isdir(os.path.join(root, "runs")):
        root = os.path.join(root, "runs")
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        row = collect_run(path, warmup)
        if row:
            runs.append(row)
    runs.sort(key=lambda r: (int(r["nodes"]), int(r["N"])))
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("".join(CSV_HEADER_COMMENTS))
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        for row in runs:
            w.writerow(row)
    return len(runs)


def _fake_profile(wall_steps, rss_bytes):
    frames = []
    for i, w in enumerate(wall_steps):
        frames.append({"scalars": [float(i), 0.0, w, float(rss_bytes)], "regions": {}})
    return {
        "schema_version": 2,
        "n_mpi_ranks": 8,
        "total_frames": len(wall_steps),
        "frame_metric_names": ["step", "mpi_rank", "wall_step", "rss_bytes"],
        "ranks": [{"mpi_rank": 0, "n_frames": len(frames), "frames": frames}],
    }


def collect_self_test():
    tmp = tempfile.mkdtemp(prefix="flagship-collect-")
    run = os.path.join(tmp, "thip-flag-1n-768_1")
    os.makedirs(run)
    with open(os.path.join(run, "run_meta.txt"), "w") as f:
        f.write(
            "job=1\nnodes=1\nntasks=8\nLx=768\nsteps=20\n"
            "revision=deadbeef\ndirty=0\nbin_sha256=abc\n"
            "OPENPFC_FFT_NODE_GRID=1\nOPENPFC_FFT_PROC_GRID=unset\n"
            "proc_grid=min-surface(2x2x2)\nlocal_brick=384x384x384\n"
            "use_pencils=false\n"
            "input=%s/input.toml\n" % run
        )
    with open(os.path.join(run, "timing_profile.json"), "w") as f:
        json.dump(_fake_profile([0.9, 0.10, 0.12, 0.11], 8 * 1024 ** 3), f)
    out = os.path.join(tmp, "out.csv")
    n = collect(tmp, out, warmup=1)
    if n != 1:
        print("expected 1 collected row, got", n, file=sys.stderr)
        return 1
    with open(out) as f:
        rows = list(csv.DictReader((ln for ln in f if not ln.startswith("#"))))
    row = rows[0]
    if abs(float(row["wall_step_s"]) - 0.11) > 1e-12:
        print("median wall_step wrong:", row["wall_step_s"], file=sys.stderr)
        return 1
    if row["N"] != "768" or row["revision"] != "deadbeef":
        print("metadata not copied:", row, file=sys.stderr)
        return 1
    if row["local_brick"] != "384x384x384" or row["use_pencils"] != "false":
        print("layout metadata not copied:", row, file=sys.stderr)
        return 1
    junk = os.path.join(tmp, "heat3d-old")
    os.makedirs(junk)
    with open(os.path.join(junk, "run_meta.txt"), "w") as f:
        f.write("job=9\nnodes=1\nLx=12.56\nsteps=100\n")
    with open(os.path.join(junk, "timing_profile.json"), "w") as f:
        f.write("{not-json")
    n = collect(tmp, out, warmup=1)
    if n != 1:
        print("mixed root should keep the flagship row, got", n, file=sys.stderr)
        return 1
    print("collect-self-test ok", out)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--collect", metavar="DIR", default=None)
    p.add_argument(
        "--out",
        default="docs/report/data/tungsten_lumi_g_flagship.csv",
    )
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--collect-self-test", action="store_true")
    p.add_argument("--grids", nargs=2, type=int, metavar=("N", "NPROC"))
    p.add_argument(
        "--candidates",
        type=int,
        metavar="NODES",
        help="list nearby 5-smooth / slab-legal N for that node count",
    )
    args = p.parse_args()
    if args.collect_self_test:
        sys.exit(collect_self_test())
    if args.grids:
        sys.exit(print_legal_grids(args.grids[0], args.grids[1]))
    if args.candidates is not None:
        sys.exit(print_candidates(args.candidates))
    if args.collect:
        n = collect(args.collect, args.out, args.warmup)
        print("wrote %d rows to %s" % (n, args.out))
        sys.exit(0 if n >= 0 else 1)
    if args.check:
        sys.exit(check())
    check()


if __name__ == "__main__":
    main()

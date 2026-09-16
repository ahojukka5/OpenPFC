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
    "input",
    "notes",
)

CSV_HEADER_COMMENTS = (
    "# Issue #13 LUMI-G tungsten_hip weak-scaling ladder.\n"
    "# Written by flagship_ladder.py --collect from scratch run dirs.\n"
    "# Do not interpolate or invent timings.\n"
    "# Protocol: docs/lumi_slurm/tungsten_hip_scaling.toml (I/O off, dt=1).\n"
    "# Multi-node: OPENPFC_FFT_NODE_GRID=1. Scratch only for bulky logs.\n"
    "# Efficiency is wall_step(1 node) / wall_step(N) (weak; similar cells/GCD).\n"
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


def check():
    rc = 0
    print("nodes  gcds     N   cells/GCD   5-smooth  1x8xN grid")
    for nodes, n in LADDER:
        gcds = nodes * 8
        cells = n ** 3
        per = cells / float(gcds)
        grid = node_aware_grid(n, gcds) if gcds >= 9 else (1, 1, 1)
        ok_smooth = is_5_smooth(n)
        if gcds >= 9:
            ok_grid = grid != (0, 0, 0)
        else:
            ok_grid = n % 8 == 0
        mark = "ok" if ok_smooth and ok_grid else "FAIL"
        if mark != "ok":
            rc = 1
        print(
            "%5d %5d %5d %10.2e   %-8s %s  %s"
            % (nodes, gcds, n, per, "yes" if ok_smooth else "no", grid, mark)
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


def collect_run(run, warmup):
    meta = _meta_map(os.path.join(run, "run_meta.txt"))
    prof_path = _find_profile(run)
    if prof_path is None:
        return None
    with open(prof_path) as f:
        doc = json.load(f)
    wall = wall_step_median(doc, warmup)
    if wall is None:
        return None
    n = int(meta.get("Lx") or meta.get("N") or 0)
    if n <= 0:
        toml = os.path.join(run, "input.toml")
        if os.path.isfile(toml):
            with open(toml) as f:
                for line in f:
                    if line.strip().startswith("Lx"):
                        n = int(line.split("=")[1].strip())
                        break
    nodes = int(meta.get("nodes") or 0)
    gcds = int(meta.get("ntasks") or (nodes * 8 if nodes else 0))
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
        if os.path.isdir(path):
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
            "OPENPFC_FFT_NODE_GRID=1\nproc_grid=min-surface\n"
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
    args = p.parse_args()
    if args.collect_self_test:
        sys.exit(collect_self_test())
    if args.collect:
        n = collect(args.collect, args.out, args.warmup)
        print("wrote %d rows to %s" % (n, args.out))
        sys.exit(0 if n >= 0 else 1)
    if args.check:
        sys.exit(check())
    check()


if __name__ == "__main__":
    main()

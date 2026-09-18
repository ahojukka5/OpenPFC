#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Aggregate heFFTe per-rank trace logs into exclusive phase buckets.

Leaf events are summed; the outer compute_transform "reshape" wrapper is
kept as a check total and is not added to pack+mpi (it nests them).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

FFT = ("fft-1d", "fft-1d x3", "scale")
PACK = (
    "packing",
    "unpacking",
    "self packing",
    "self unpacking",
    "copy",
    "reshape/copy",
)
MPI = (
    "all2all",
    "all2allv",
    "waitany",
)
MPI_PREFIX = ("irecv ", "isend ", "send ")
OUTER = ("reshape",)


def classify(name: str) -> str:
    n = name.strip()
    if n in FFT:
        return "fft"
    if n in PACK:
        return "pack"
    if n in MPI or n.startswith(MPI_PREFIX):
        return "mpi"
    if n in OUTER:
        return "reshape_outer"
    return "other"


def parse_log(path: Path) -> list[tuple[str, float, float]]:
    events = []
    for line in path.read_text().splitlines():
        line = line.rstrip()
        if not line:
            continue
        # name is 40 chars, then start, then duration
        name = line[:40].strip()
        rest = line[40:].split()
        if len(rest) < 2:
            continue
        events.append((name, float(rest[0]), float(rest[1])))
    return events


def factorize(n: int) -> str:
    if n <= 1:
        return str(n)
    parts = []
    x = n
    p = 2
    while p * p <= x:
        k = 0
        while x % p == 0:
            x //= p
            k += 1
        if k:
            parts.append(f"{p}^{k}" if k > 1 else str(p))
        p += 1 if p == 2 else 2
    if x > 1:
        parts.append(str(x))
    return "*".join(parts)


def read_meta(run: Path) -> dict[str, str]:
    meta = {}
    p = run / "run_meta.txt"
    if not p.is_file():
        return meta
    for line in p.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            meta[k.strip()] = v.strip()
    return meta


def profile_wall(run: Path) -> float | None:
    # schema-v4 JSON next to the rundir, name timing_profile*.json
    cands = list(run.glob("timing_profile*.json")) + list(run.glob("*.json"))
    for p in cands:
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("wall_step", "wall_time_per_step", "mean_step"):
            if key in data:
                return float(data[key])
        timers = data.get("timers") or data.get("phases") or {}
        if isinstance(timers, dict) and "step" in timers:
            val = timers["step"]
            if isinstance(val, dict) and "mean" in val:
                return float(val["mean"])
            if isinstance(val, (int, float)):
                return float(val)
    return None


def summarize_rank(events: list[tuple[str, float, float]], skip_frac: float = 0.1):
    """Drop the first skip_frac of event time span (warmup), then sum leaves."""
    if not events:
        return {k: 0.0 for k in ("fft", "pack", "mpi", "reshape_outer", "other", "n_events")}
    t0 = events[0][1]
    t1 = events[-1][1] + events[-1][2]
    cut = t0 + skip_frac * (t1 - t0)
    buckets = defaultdict(float)
    n = 0
    for name, start, dur in events:
        if start < cut:
            continue
        buckets[classify(name)] += dur
        n += 1
    buckets["n_events"] = float(n)
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root", nargs="?")
    ap.add_argument("--out", default="-")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        fake = "fft-1d" + " " * 34 + "1.0                 0.25\n"
        fake += "packing" + " " * 33 + "1.25                0.10\n"
        fake += "all2all" + " " * 33 + "1.35                0.40\n"
        tmp = Path("/tmp/heffte_trace_self_test/fake_run")
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "heffte_trace_0.log").write_text(fake)
        (tmp / "run_meta.txt").write_text("job=0\nLx=8\nLy=8\nLz=8\nproc_grid=1x1x1\n")
        args.runs_root = str(tmp.parent)
        skip_frac = 0.0
    else:
        skip_frac = 0.05
    if not args.runs_root:
        ap.error("runs_root is required unless --self-test")
    root = Path(args.runs_root)
    rows = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        logs = sorted(run.glob("heffte_trace_*.log"))
        if not logs:
            continue
        meta = read_meta(run)
        rank_sums = []
        for log in logs:
            ev = parse_log(log)
            rank_sums.append(summarize_rank(ev, skip_frac=skip_frac))
        # critical path: max across ranks of each exclusive bucket
        def mx(key):
            return max(r[key] for r in rank_sums)

        fft = mx("fft")
        pack = mx("pack")
        mpi = mx("mpi")
        other = mx("other")
        outer = mx("reshape_outer")
        leaves = fft + pack + mpi + other
        nx = int(meta.get("Lx") or 0)
        ny = int(meta.get("Ly") or 0)
        nz = int(meta.get("Lz") or 0)
        wall = profile_wall(run)
        rows.append(
            {
                "run": run.name,
                "job": meta.get("job", ""),
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "proc_grid": meta.get("proc_grid", ""),
                "local_brick": meta.get("local_brick", ""),
                "factors_x": factorize(nx) if nx else "",
                "factors_y": factorize(ny) if ny else "",
                "factors_z": factorize(nz) if nz else "",
                "bin_sha256": meta.get("bin_sha256", ""),
                "revision": meta.get("revision", ""),
                "dirty": meta.get("dirty", ""),
                "heffte_root": meta.get("HEFFTE_ROOT", ""),
                "fft_s": f"{fft:.6f}",
                "pack_s": f"{pack:.6f}",
                "mpi_s": f"{mpi:.6f}",
                "other_s": f"{other:.6f}",
                "reshape_outer_s": f"{outer:.6f}",
                "leaf_sum_s": f"{leaves:.6f}",
                "wall_step_s": "" if wall is None else f"{wall:.6f}",
                "fft_frac_leaf": f"{fft / leaves:.4f}" if leaves else "",
                "pack_frac_leaf": f"{pack / leaves:.4f}" if leaves else "",
                "mpi_frac_leaf": f"{mpi / leaves:.4f}" if leaves else "",
                "n_logs": len(logs),
            }
        )
    if not rows:
        print("no heffte_trace_*.log files found", file=__import__("sys").stderr)
        return 1
    fields = list(rows[0].keys())
    if args.out == "-":
        w = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

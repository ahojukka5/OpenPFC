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
MPI_WAIT = ("waitany",)
MPI_COLLECTIVE = ("all2all", "all2allv")
MPI_PREFIX = ("irecv ", "isend ", "send ")
PACK_PREFIX = ("unpacking from ",)
OUTER = ("reshape",)


def classify(name: str) -> str:
    n = name.strip()
    if n in FFT:
        return "fft"
    if n in PACK or n.startswith(PACK_PREFIX):
        return "pack"
    if n in MPI_WAIT:
        return "mpi_wait"
    if n in MPI_COLLECTIVE:
        return "mpi_coll"
    if n.startswith(MPI_PREFIX):
        return "mpi_post"
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


def profile_wall(run: Path, skip_frac: float) -> tuple[float | None, int]:
    """Mean of per-step max-rank wall_step after skipping the first skip_frac.

    timing_profile.json stores one frame per accepted step; scalars follow
    frame_metric_names (wall_step is index 2 in the tungsten schema).
    """
    cands = [p for p in run.glob("timing_profile*.json") if p.is_file()]
    for p in cands:
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        names = data.get("frame_metric_names") or []
        if "wall_step" not in names:
            continue
        idx = names.index("wall_step")
        ranks = data.get("ranks") or []
        if not ranks:
            continue
        n_frames = min(len(r.get("frames") or []) for r in ranks)
        if n_frames == 0:
            continue
        cut = int(skip_frac * n_frames)
        kept = 0
        acc = 0.0
        for i in range(cut, n_frames):
            mx = 0.0
            for r in ranks:
                frames = r.get("frames") or []
                scalars = frames[i].get("scalars") or []
                if len(scalars) > idx:
                    mx = max(mx, float(scalars[idx]))
            acc += mx
            kept += 1
        if kept:
            return acc / kept, kept
    return None, 0


def summarize_rank(events: list[tuple[str, float, float]], skip_frac: float = 0.1):
    """Drop the first skip_frac of event time span (warmup), then sum leaves."""
    keys = (
        "fft",
        "pack",
        "mpi_wait",
        "mpi_post",
        "mpi_coll",
        "reshape_outer",
        "other",
        "n_events",
    )
    if not events:
        return {k: 0.0 for k in keys}
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
        fake += "waitany" + " " * 33 + "1.35                0.30\n"
        fake += "unpacking from 1" + " " * 24 + "1.65                0.05\n"
        fake += "isend 8 for 1" + " " * 27 + "1.70                0.02\n"
        tmp = Path("/tmp/heffte_trace_self_test/fake_run")
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "heffte_trace_0.log").write_text(fake)
        (tmp / "run_meta.txt").write_text(
            "job=0\nLx=8\nLy=8\nLz=8\nproc_grid=1x1x1\nlocal_brick=8x8x8\n"
        )
        (tmp / "timing_profile.json").write_text(
            json.dumps(
                {
                    "frame_metric_names": ["step", "mpi_rank", "wall_step"],
                    "ranks": [
                        {"frames": [{"scalars": [1.0, 0.0, 0.80]}]},
                    ],
                }
            )
        )
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
        def leaf_total(r):
            return (
                r["fft"]
                + r["pack"]
                + r["mpi_wait"]
                + r["mpi_post"]
                + r["mpi_coll"]
                + r["other"]
            )

        # One rank: the largest traced leaf sum. Independent per-bucket
        # maxima would mix ranks and overstate the remainder.
        crit = max(rank_sums, key=leaf_total)
        fft = crit["fft"]
        pack = crit["pack"]
        mpi_wait = crit["mpi_wait"]
        mpi_post = crit["mpi_post"]
        mpi_coll = crit["mpi_coll"]
        mpi = mpi_wait + mpi_post + mpi_coll
        other = crit["other"]
        outer = crit["reshape_outer"]
        leaves = leaf_total(crit)
        nx = int(meta.get("Lx") or 0)
        ny = int(meta.get("Ly") or 0)
        nz = int(meta.get("Lz") or 0)
        wall, n_kept = profile_wall(run, skip_frac)
        n_steps = n_kept if n_kept else 0
        # Per-step phases so they sit next to wall_step. Trace totals span
        # the kept window; do not force leaf_sum == n_steps * wall_step.
        def per_step(total: float) -> float:
            return total / n_steps if n_steps else total

        fft_ps = per_step(fft)
        pack_ps = per_step(pack)
        mpi_ps = per_step(mpi)
        other_ps = per_step(other)
        leaves_ps = per_step(leaves)
        denom = wall if wall else leaves_ps
        remainder = None if wall is None else wall - leaves_ps
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
                "n_steps_kept": n_steps,
                "wall_step_s": "" if wall is None else f"{wall:.6f}",
                "fft_s": f"{fft_ps:.6f}",
                "pack_s": f"{pack_ps:.6f}",
                "mpi_s": f"{mpi_ps:.6f}",
                "mpi_wait_s": f"{per_step(mpi_wait):.6f}",
                "mpi_post_s": f"{per_step(mpi_post):.6f}",
                "mpi_coll_s": f"{per_step(mpi_coll):.6f}",
                "other_s": f"{other_ps:.6f}",
                "reshape_outer_s": f"{per_step(outer):.6f}",
                "leaf_sum_s": f"{leaves_ps:.6f}",
                "remainder_s": "" if remainder is None else f"{remainder:.6f}",
                "fft_frac": f"{fft_ps / denom:.4f}" if denom else "",
                "pack_frac": f"{pack_ps / denom:.4f}" if denom else "",
                "mpi_frac": f"{mpi_ps / denom:.4f}" if denom else "",
                "other_frac": f"{other_ps / denom:.4f}" if denom else "",
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

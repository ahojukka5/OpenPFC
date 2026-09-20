#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Issue #108 FD-order × halo-overlap weak scaling: ladders, collect, analyze.

Constant 256³ owned interior cells / GCD. Production two-stream overlap.
Orders 2, 4, 8, 12, 20 unless the production path rejects one.

    python3 apps/heat3d/scripts/fd_order_scaling.py --check
    python3 apps/heat3d/scripts/fd_order_scaling.py --ladder
    python3 apps/heat3d/scripts/fd_order_scaling.py --collect DIR --out CSV
    python3 apps/heat3d/scripts/fd_order_scaling.py --analyze CSV --out DIR
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ORDERS = (2, 4, 8, 12, 20)
NODE_LADDER = (1, 8, 32, 128, 512, 1024)
DIAG_NODES = (8, 128, 1024)
REPEAT_NODES = (128, 512, 1024)
GCDS_PER_NODE = 8
INTERIOR = 256
ISSUE = "108"

RUN_FIELDS = (
    "issue",
    "job",
    "account",
    "nodes",
    "ranks",
    "partition",
    "fd_order",
    "halo_width",
    "halo_face_bytes",
    "mode",
    "repeat",
    "Nx",
    "Ny",
    "Nz",
    "proc_grid",
    "local_brick",
    "halo_overlap",
    "gpu_aware",
    "contiguous",
    "admit",
    "reason",
    "revision",
    "dirty",
    "bin_sha256",
    "wall_step_s",
    "weak_eff",
    "post_s",
    "exposed_wait_s",
    "inner_s",
    "border_s",
    "halo_s",
    "rhs_s",
    "update_s",
    "offnode_faces",
    "checksum",
    "scratch",
)

SCALE_FIELDS = (
    "fd_order",
    "halo_width",
    "halo_face_bytes",
    "nodes",
    "ranks",
    "proc_grid",
    "n_repeats",
    "wall_step_s_median",
    "wall_step_s_min",
    "wall_step_s_max",
    "weak_eff",
    "halo_bytes_per_owned",
    "jobs",
)


def halo_width(order: int) -> int:
    return order // 2


def halo_face_bytes(order: int, interior: int = INTERIOR) -> int:
    # Six packed faces of interior^2 * width * sizeof(double).
    return 6 * interior * interior * halo_width(order) * 8


def proc_grid_for_ranks(ranks: int) -> Tuple[int, int, int]:
    """Double the smallest axis, z then y then x on ties.

    Matches the admitted FD-2 grids: 2x2x2, 4x4x4, 4x8x8, 8x8x16,
    16x16x16, 16x16x32.
    """
    if ranks < 1 or (ranks & (ranks - 1)) != 0:
        raise ValueError("ranks must be a power of two, got %d" % ranks)
    gx, gy, gz = 1, 1, 1
    n = 1
    while n < ranks:
        if gz <= gy and gz <= gx:
            gz *= 2
        elif gy <= gx:
            gy *= 2
        else:
            gx *= 2
        n *= 2
    return gx, gy, gz


def ladder() -> List[Tuple[int, int, int, int, str, int]]:
    """(nodes, Nx, Ny, Nz, proc_grid, ranks)."""
    out = []
    for nodes in NODE_LADDER:
        ranks = nodes * GCDS_PER_NODE
        gx, gy, gz = proc_grid_for_ranks(ranks)
        nx, ny, nz = INTERIOR * gx, INTERIOR * gy, INTERIOR * gz
        # Use x separators: sbatch --export splits on commas.
        grid = "%dx%dx%d" % (gx, gy, gz)
        out.append((nodes, nx, ny, nz, grid, ranks))
    return out


def repeats_for_nodes(nodes: int, override: Optional[int] = None) -> int:
    if override is not None:
        return override
    if nodes in REPEAT_NODES:
        return 3
    return 1


def check() -> int:
    rc = 0
    known = {
        1: (2, 2, 2, 512, 512, 512),
        8: (4, 4, 4, 1024, 1024, 1024),
        32: (4, 8, 8, 1024, 2048, 2048),
        128: (8, 8, 16, 2048, 2048, 4096),
        512: (16, 16, 16, 4096, 4096, 4096),
        1024: (16, 16, 32, 4096, 4096, 8192),
    }
    print("nodes ranks grid     global           local     halo_bytes")
    for nodes, nx, ny, nz, grid, ranks in ladder():
        gx, gy, gz = (int(p) for p in grid.replace(",", "x").split("x"))
        loc = (nx // gx, ny // gy, nz // gz)
        ok = loc == (INTERIOR, INTERIOR, INTERIOR) and gx * gy * gz == ranks
        expect = known.get(nodes)
        if expect and (gx, gy, gz, nx, ny, nz) != expect:
            ok = False
        print(
            "%5d %5d %-8s %dx%dx%d  %dx%dx%d  %s  %d"
            % (
                nodes,
                ranks,
                grid.replace(",", "x"),
                nx,
                ny,
                nz,
                loc[0],
                loc[1],
                loc[2],
                "ok" if ok else "FAIL",
                halo_face_bytes(2),
            )
        )
        if not ok:
            rc = 1
    print("orders: %s" % ",".join(str(o) for o in ORDERS))
    for order in ORDERS:
        if order % 2 or order < 2 or order > 20:
            print("FAIL unsupported order %d" % order)
            rc = 1
    return rc


def print_ladder() -> int:
    print("nodes Nx Ny Nz grid ranks")
    for nodes, nx, ny, nz, grid, ranks in ladder():
        print("%d %d %d %d %s %d" % (nodes, nx, ny, nz, grid, ranks))
    return 0


def _meta_map(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("==="):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "" or value == "unset":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _kv_line(path: str, prefix: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith(prefix):
                continue
            out: Dict[str, str] = {}
            for tok in line.split()[1:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    out[k] = v
            return out
    return {}


def checksum_line(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path) as f:
        for line in f:
            if line.startswith("HEAT3D_HIP_CHECKSUM "):
                return line.strip()
    return ""


def wall_step_median(doc: Dict[str, Any], warmup: int) -> Optional[float]:
    names = list(doc.get("frame_metric_names") or [])
    try:
        idx = names.index("wall_step")
    except ValueError:
        if int(doc.get("schema_version") or 0) == 4:
            m = (doc.get("metrics") or {}).get("wall_step") or {}
            if "median" in m:
                return float(m["median"])
        return None
    step_idx = names.index("step") if "step" in names else None
    values: List[float] = []
    for rank in doc.get("ranks") or []:
        frames = rank.get("frames") or []
        for i, frame in enumerate(frames):
            scalars = frame.get("scalars") or []
            step = scalars[step_idx] if step_idx is not None else i
            if step < warmup:
                continue
            if idx < len(scalars):
                values.append(float(scalars[idx]))
    if not values:
        return None
    return float(statistics.median(values))


def collect_run(run: str, warmup: int) -> Optional[Dict[str, Any]]:
    meta = _meta_map(os.path.join(run, "run_meta.txt"))
    log = os.path.join(run, "run.log")
    admit = _meta_map(os.path.join(run, "admit.txt"))
    decomp = _kv_line(log, "HEAT3D_FD_DECOMP")
    diag = _kv_line(log, "HEAT3D_DIAG")
    overlap = _kv_line(log, "HEAT3D_OVERLAP")
    nx = _parse_int(meta.get("Nx"))
    ny = _parse_int(meta.get("Ny"))
    nz = _parse_int(meta.get("Nz"))
    nodes = _parse_int(meta.get("nodes")) or 0
    ranks = _parse_int(meta.get("ntasks")) or (nodes * GCDS_PER_NODE)
    order = _parse_int(meta.get("fd_order")) or _parse_int(decomp.get("fd_order"))
    mode = meta.get("mode") or ("diagnostic" if meta.get("HEAT3D_DIAG_TIMING") == "1" else "clean")
    admit_flag = admit.get("admit") or "ok"
    reason = admit.get("reason") or "none"
    wall = None
    prof = os.path.join(run, "timing_profile.json")
    if os.path.isfile(prof) and admit_flag == "ok":
        with open(prof) as f:
            doc = json.load(f)
        wall = wall_step_median(doc, warmup)
    local = decomp.get("local_min") or meta.get("require_interior") or ""
    expected = "%dx%dx%d" % (INTERIOR, INTERIOR, INTERIOR)
    if local and local != expected and "x" in local:
        # local_min printed as 256x256x256
        if local.replace(",", "x") != expected:
            admit_flag = "reject"
            reason = "interior_%s_expected_%s" % (local, expected)
            wall = None
    if order is None:
        admit_flag = "reject"
        reason = "missing_fd_order"
        wall = None
    width = halo_width(order) if order else ""
    face_bytes = halo_face_bytes(order) if order else ""
    checksum = checksum_line(log)
    if admit_flag == "ok" and checksum and (
        "nan" in checksum.lower() or "inf" in checksum.lower()
    ):
        admit_flag = "reject"
        reason = "checksum_nonfinite"
        wall = None
    offnode = ""
    place = os.path.join(run, "fd_placement.txt")
    if os.path.isfile(place):
        faces = []
        with open(place) as f:
            header = True
            for line in f:
                if header:
                    header = False
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        faces.append(int(parts[-1]))
                    except ValueError:
                        pass
        if faces:
            offnode = str(max(faces))
    return {
        "issue": ISSUE,
        "job": meta.get("job", os.path.basename(run)),
        "account": meta.get("account", ""),
        "nodes": nodes,
        "ranks": ranks,
        "partition": meta.get("partition", ""),
        "fd_order": order or "",
        "halo_width": width,
        "halo_face_bytes": face_bytes,
        "mode": mode,
        "repeat": meta.get("repeat", ""),
        "Nx": nx or "",
        "Ny": ny or "",
        "Nz": nz or "",
        "proc_grid": decomp.get("proc_grid") or meta.get("OPENPFC_FD_PROC_GRID", ""),
        "local_brick": local,
        "halo_overlap": decomp.get("halo_overlap", ""),
        "gpu_aware": decomp.get("gpu_aware", ""),
        "contiguous": decomp.get("contiguous", ""),
        "admit": admit_flag,
        "reason": reason,
        "revision": meta.get("revision", ""),
        "dirty": meta.get("dirty", ""),
        "bin_sha256": meta.get("bin_sha256", ""),
        "wall_step_s": "" if wall is None else "%.8f" % wall,
        "weak_eff": "",
        "post_s": overlap.get("post_s", ""),
        "exposed_wait_s": overlap.get("exposed_wait_s", ""),
        "inner_s": overlap.get("inner_s", ""),
        "border_s": overlap.get("border_s", ""),
        "halo_s": diag.get("halo_s", ""),
        "rhs_s": diag.get("rhs_s", ""),
        "update_s": diag.get("update_s", ""),
        "offnode_faces": offnode,
        "checksum": checksum,
        "scratch": run,
    }


def collect_root(root: str, warmup: int) -> List[Dict[str, Any]]:
    runs_dir = os.path.join(root, "runs") if os.path.isdir(
        os.path.join(root, "runs")
    ) else root
    rows: List[Dict[str, Any]] = []
    if not os.path.isdir(runs_dir):
        return rows
    for name in sorted(os.listdir(runs_dir)):
        run = os.path.join(runs_dir, name)
        if not os.path.isdir(run):
            continue
        if not os.path.isfile(os.path.join(run, "run_meta.txt")):
            continue
        row = collect_run(run, warmup)
        if row is not None:
            rows.append(row)
    return rows


def write_csv(path: str, fields: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def analyze(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("admit")) != "ok":
            continue
        if str(row.get("mode") or "clean") != "clean":
            continue
        if not row.get("wall_step_s"):
            continue
        order = int(row["fd_order"])
        nodes = int(row["nodes"])
        grouped[(order, nodes)].append(row)

    baseline: Dict[int, float] = {}
    medians: Dict[Tuple[int, int], float] = {}
    out: List[Dict[str, Any]] = []
    for (order, nodes), group in sorted(grouped.items()):
        walls = [float(r["wall_step_s"]) for r in group]
        med = float(statistics.median(walls))
        medians[(order, nodes)] = med
        if nodes == 1:
            baseline[order] = med
        owned = float(INTERIOR ** 3)
        face_b = halo_face_bytes(order)
        jobs = ",".join(sorted({str(r.get("job") or "") for r in group}))
        out.append(
            {
                "fd_order": order,
                "halo_width": halo_width(order),
                "halo_face_bytes": face_b,
                "nodes": nodes,
                "ranks": group[0].get("ranks", nodes * GCDS_PER_NODE),
                "proc_grid": group[0].get("proc_grid", ""),
                "n_repeats": len(walls),
                "wall_step_s_median": "%.8f" % med,
                "wall_step_s_min": "%.8f" % min(walls),
                "wall_step_s_max": "%.8f" % max(walls),
                "weak_eff": "",
                "halo_bytes_per_owned": "%.6e" % (face_b / owned),
                "jobs": jobs,
            }
        )
    for row in out:
        order = int(row["fd_order"])
        med = float(row["wall_step_s_median"])
        base = baseline.get(order)
        if base and med > 0:
            row["weak_eff"] = "%.4f" % (base / med)
    return out


def notes_for(scale_rows: Sequence[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    by_nodes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in scale_rows:
        by_nodes[int(row["nodes"])].append(row)
    for nodes, rows in sorted(by_nodes.items()):
        ranked = sorted(rows, key=lambda r: (float(r["wall_step_s_median"]), int(r["fd_order"])))
        if not ranked:
            continue
        notes.append(
            "%d nodes: fastest order FD-%s (%.6f s); slowest FD-%s (%.6f s)"
            % (
                nodes,
                ranked[0]["fd_order"],
                float(ranked[0]["wall_step_s_median"]),
                ranked[-1]["fd_order"],
                float(ranked[-1]["wall_step_s_median"]),
            )
        )
    large = by_nodes.get(1024) or by_nodes.get(512) or []
    if large:
        effs = [
            (int(r["fd_order"]), _parse_float(str(r.get("weak_eff") or "")))
            for r in large
        ]
        effs = [(o, e) for o, e in effs if e is not None]
        if len(effs) >= 2:
            monotone = all(effs[i][1] >= effs[i + 1][1] for i in range(len(effs) - 1))
            if monotone:
                notes.append(
                    "large-scale weak efficiency degrades monotonically with order"
                )
            else:
                notes.append(
                    "large-scale weak efficiency is not monotone in order"
                )
    notes.append(
        "wall/step uses the frozen dt=0.01 protocol; do not mix with "
        "time-to-solution if a later order requires a smaller timestep"
    )
    notes.append("this campaign is a scaling mechanism study, not an accuracy claim")
    return notes


def write_markdown(path: str, scale_rows: Sequence[Dict[str, Any]], notes: Sequence[str]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write("# FD-order halo-overlap scaling (issue #108)\n\n")
        f.write(
            "Clean production runs only. Weak efficiency is per order versus "
            "that order's 1-node median. Halo face bytes are geometric, not "
            "measured payload.\n\n"
        )
        f.write(
            "| order | width | face B | nodes | n | median s | min | max | "
            "weak eff | B/owned |\n"
        )
        f.write("|-----:|------:|-------:|------:|--:|---------:|----:|----:|--------:|--------:|\n")
        rows = sorted(
            scale_rows,
            key=lambda r: (int(r["fd_order"]), int(r["nodes"])),
        )
        for r in rows:
            f.write(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n"
                % (
                    r["fd_order"],
                    r["halo_width"],
                    r["halo_face_bytes"],
                    r["nodes"],
                    r["n_repeats"],
                    r["wall_step_s_median"],
                    r["wall_step_s_min"],
                    r["wall_step_s_max"],
                    r["weak_eff"],
                    r["halo_bytes_per_owned"],
                )
            )
        f.write("\n## Notes\n\n")
        for n in notes:
            f.write("- %s\n" % n)
        f.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--ladder", action="store_true")
    p.add_argument("--collect")
    p.add_argument("--analyze")
    p.add_argument("--out")
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args(argv)
    if args.check:
        return check()
    if args.ladder:
        return print_ladder()
    if args.collect:
        if not args.out:
            print("--collect requires --out CSV", file=sys.stderr)
            return 2
        rows = collect_root(args.collect, args.warmup)
        write_csv(args.out, RUN_FIELDS, rows)
        print("wrote %d rows to %s" % (len(rows), args.out))
        return 0
    if args.analyze:
        if not args.out:
            print("--analyze requires --out DIR", file=sys.stderr)
            return 2
        rows = read_csv(args.analyze)
        scale = analyze(rows)
        write_csv(os.path.join(args.out, "scaling.csv"), SCALE_FIELDS, scale)
        notes = notes_for(scale)
        write_markdown(os.path.join(args.out, "scaling.md"), scale, notes)
        print("wrote %s" % os.path.join(args.out, "scaling.csv"))
        for n in notes:
            print(n)
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

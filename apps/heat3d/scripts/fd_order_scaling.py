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
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ORDERS = (2, 4, 8, 12, 20)
NODE_LADDER = (1, 8, 32, 128, 512, 1024)
DIAG_NODES = (8, 128, 1024)
# Clean repeats at every scale: 1/8/32-node jobs finish in seconds, so
# three allocations are cheaper than arguing from a single millisecond
# median. 1024-node stays at three; do not add a fourth.
REPEAT_NODES = NODE_LADDER
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
    "steps",
    "warmup",
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

GEOMETRY_FIELDS = (
    "fd_order",
    "halo_width",
    "owned_cells",
    "interior_cells",
    "boundary_cells",
    "interior_fraction",
    "boundary_fraction",
    "elems_per_face",
    "bytes_per_face",
    "total_face_elems",
    "total_face_bytes",
    "halo_bytes_per_owned",
    "d2_half_width",
    "d2_unique_offsets",
    "d2_naive_loads_3axis",
    "d2_arith_ops_per_point",
    "relative_arith_vs_fd2",
    "interior_work_units",
    "interior_work_per_halo_byte",
)

# EvenCentralD2: per axis, M neighbour pairs + centre, then three axes summed.
# Naive loads count centre three times (independent apply_d2_along).
# Unique offsets = 1 centre + 6M axial neighbours (the fused read set).
# Arithmetic: per axis (2M adds of ±k, M+1 muls, 1 scale); 2 adds to sum axes.


def halo_width(order: int) -> int:
    return order // 2


def halo_face_bytes(order: int, interior: int = INTERIOR) -> int:
    # Six packed Faces of interior^2 * width * sizeof(double). Corners
    # travel on more than one face; that is the production pack.
    return 6 * interior * interior * halo_width(order) * 8


def geometry_row(order: int, interior: int = INTERIOR) -> Dict[str, Any]:
    w = halo_width(order)
    owned = interior ** 3
    inner_n = interior - 2 * w
    interior_cells = inner_n ** 3 if inner_n > 0 else 0
    boundary_cells = owned - interior_cells
    elems_face = interior * interior * w
    bytes_face = elems_face * 8
    total_elems = 6 * elems_face
    total_bytes = 6 * bytes_face
    naive_loads = 3 * (1 + 2 * w)
    unique_offsets = 1 + 6 * w
    # per axis: 2w adds, (w+1) muls, 1 scale; plus 2 adds across axes
    arith = 3 * (2 * w + (w + 1) + 1) + 2
    arith2 = 3 * (2 * 1 + (1 + 1) + 1) + 2
    return {
        "fd_order": order,
        "halo_width": w,
        "owned_cells": owned,
        "interior_cells": interior_cells,
        "boundary_cells": boundary_cells,
        "interior_fraction": "%.6f" % (interior_cells / float(owned)),
        "boundary_fraction": "%.6f" % (boundary_cells / float(owned)),
        "elems_per_face": elems_face,
        "bytes_per_face": bytes_face,
        "total_face_elems": total_elems,
        "total_face_bytes": total_bytes,
        "halo_bytes_per_owned": "%.6e" % (total_bytes / float(owned)),
        "d2_half_width": w,
        "d2_unique_offsets": unique_offsets,
        "d2_naive_loads_3axis": naive_loads,
        "d2_arith_ops_per_point": arith,
        "relative_arith_vs_fd2": "%.4f" % (arith / float(arith2)),
        "interior_work_units": interior_cells * arith,
        "interior_work_per_halo_byte": (
            "%.6e" % (interior_cells * arith / float(total_bytes))
            if total_bytes
            else ""
        ),
    }


def geometry_table() -> List[Dict[str, Any]]:
    return [geometry_row(o) for o in ORDERS]


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


def timed_protocol(nodes: int) -> Tuple[int, int]:
    """Return (steps, warmup) for a node count.

    Heat3D FD wall/step is 1--8 ms. Extra timed steps cost almost nothing
    next to queue and launch, but the schema-v4 JSON grows with ranks ×
    frames. Cap so a profile stays around the 1024-node 105-step size
    (~440 MB). Headline metric is still the median after warmup.
    """
    if nodes <= 8:
        return 5005, 50
    if nodes <= 32:
        return 3005, 50
    if nodes <= 128:
        return 805, 20
    if nodes <= 512:
        return 205, 10
    return 105, 5


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


_SCALAR_ARRAY = re.compile(r'"scalars"\s*:\s*\[([^\]]*)\]')


def _floats_from_csv(inner: str) -> List[float]:
    nums: List[float] = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nums.append(float(part))
        except ValueError:
            pass
    return nums


def wall_step_median_scan(path: str, warmup: int) -> Optional[float]:
    """Median wall_step without json.load. 1024-node profiles are ~440 MB."""
    values: List[float] = []
    buf: List[float] = []
    in_scalars = False

    def consume_array(nums: List[float]) -> None:
        if len(nums) >= 3 and nums[0] >= warmup:
            values.append(nums[2])

    with open(path, "r", buffering=1024 * 1024) as handle:
        for line in handle:
            inline = list(_SCALAR_ARRAY.finditer(line))
            if inline:
                for match in inline:
                    consume_array(_floats_from_csv(match.group(1)))
                continue
            stripped = line.strip().rstrip(",")
            if '"scalars"' in line:
                in_scalars = True
                buf = []
                continue
            if not in_scalars:
                continue
            if stripped.startswith("]"):
                consume_array(buf)
                in_scalars = False
                buf = []
                continue
            try:
                buf.append(float(stripped))
            except ValueError:
                pass
    if not values:
        return None
    return float(statistics.median(values))


def wall_step_from_profile(path: str, warmup: int) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    size = os.path.getsize(path)
    if size <= 32 * 1024 * 1024:
        with open(path) as handle:
            return wall_step_median(json.load(handle), warmup)
    return wall_step_median_scan(path, warmup)


def recover_missing_admit(
    admit_flag: str,
    reason: str,
    checksum: str,
    decomp: Dict[str, str],
    prof: str,
) -> Tuple[str, str]:
    """Rebuild admit=ok when srun finished but admit.txt was never written.

    The #108 batch used to `cp` fd_placement.txt onto itself under set -e,
    so a successful Heat3D run left checksum + profile and no admit file.
    """
    if admit_flag and admit_flag != "missing":
        return admit_flag, reason
    if not checksum or "nan" in checksum.lower() or "inf" in checksum.lower():
        return admit_flag or "missing", reason or "missing_admit"
    if not decomp.get("proc_grid"):
        return admit_flag or "missing", reason or "missing_admit"
    if not os.path.isfile(prof):
        return admit_flag or "missing", reason or "missing_admit"
    return "ok", "recovered_missing_admit"


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
    admit_flag = admit.get("admit") or ""
    reason = admit.get("reason") or ""
    if not admit_flag:
        admit_flag = "missing"
        reason = reason or "missing_admit"
    wall = None
    prof = os.path.join(run, "timing_profile.json")
    checksum = checksum_line(log)
    admit_flag, reason = recover_missing_admit(
        admit_flag, reason, checksum, decomp, prof
    )
    if os.path.isfile(prof) and admit_flag == "ok":
        run_warmup = _parse_int(meta.get("warmup")) or warmup
        wall = wall_step_from_profile(prof, run_warmup)
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
        "steps": meta.get("steps", ""),
        "warmup": meta.get("warmup", ""),
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


def timed_steps(row: Mapping[str, Any]) -> int:
    steps = _parse_int(row.get("steps")) or 0
    warm = _parse_int(row.get("warmup")) or 0
    return max(0, steps - warm)


def prefer_longest_protocol(
    group: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep the longest timed window when a cell mixed 105- and 5005-step jobs."""
    lengths = [timed_steps(r) for r in group]
    if not any(lengths):
        return list(group)
    want = max(lengths)
    return [r for r, n in zip(group, lengths) if n == want]


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
        group = prefer_longest_protocol(group)
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


OVERLAP_MODEL = (
    "T_step ≈ T_post + max(T_inner, T_network_progress) + "
    "T_exposed_wait + T_border + T_update"
)

OVERLAP_MAP = (
    ("T_post", "HEAT3D_OVERLAP post_s", "diag overlap jobs"),
    ("T_inner", "HEAT3D_OVERLAP inner_s", "diag overlap jobs"),
    ("T_exposed_wait", "HEAT3D_OVERLAP exposed_wait_s", "diag overlap jobs"),
    ("T_border", "HEAT3D_OVERLAP border_s", "diag overlap jobs"),
    ("T_update", "HEAT3D_DIAG update_s", "diag jobs"),
    ("T_network_progress", "not timed; hidden iff inner_s > exposed_wait_s", "inferred"),
    ("T_step", "timing_profile.json wall_step", "clean and diag"),
)


def expected_jobs() -> List[Dict[str, Any]]:
    out = []
    for nodes, nx, ny, nz, grid, ranks in ladder():
        nrep = repeats_for_nodes(nodes)
        for order in ORDERS:
            for repeat in range(1, nrep + 1):
                out.append(
                    {
                        "name": "h3d108c-fd%d-%dn-r%d" % (order, nodes, repeat),
                        "mode": "clean",
                        "order": order,
                        "nodes": nodes,
                        "repeat": repeat,
                        "grid": grid,
                        "ranks": ranks,
                    }
                )
        if nodes in DIAG_NODES:
            for order in ORDERS:
                out.append(
                    {
                        "name": "h3d108d-fd%d-%dn-r1" % (order, nodes),
                        "mode": "diagnostic",
                        "order": order,
                        "nodes": nodes,
                        "repeat": 1,
                        "grid": grid,
                        "ranks": ranks,
                    }
                )
    return out


def _slurm_table() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    user = os.environ.get("USER", "juaho")
    for cmd in (
        [
            "sacct", "-X", "-S", "2026-09-20", "-u", user,
            "-A", "project_462001245", "-n", "-P",
            "--format=JobID,JobName,State",
        ],
        [
            "squeue", "-u", user, "-A", "project_462001245", "-h",
            "-o", "%i|%j|%T",
        ],
    ):
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0:
            continue
        for line in proc.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            jobid, name, state = parts[0], parts[1], parts[2].split()[0]
            if not name.startswith("h3d108"):
                continue
            out[name] = {"job": jobid, "state": state}
    return out


def write_status_markdown(
    path: str,
    rows: Sequence[Dict[str, Any]],
    scale_rows: Sequence[Dict[str, Any]],
    slurm: Dict[str, Dict[str, str]],
) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    by_key: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            order = int(row.get("fd_order") or 0)
            nodes = int(row.get("nodes") or 0)
        except (TypeError, ValueError):
            continue
        mode = str(row.get("mode") or "clean")
        by_key[(order, nodes, mode)].append(row)
    scale_map = {
        (int(r["fd_order"]), int(r["nodes"])): r for r in scale_rows
    }
    with open(path, "w") as f:
        f.write("# Issue #108 campaign status\n\n")
        f.write(
            "Clean production wall/step only in the efficiency columns. "
            "Do not declare an order winner from one allocation.\n\n"
        )
        f.write(
            "| order | nodes | clean n | wall/step | weak eff | spread | "
            "diag | PENDING | FAILED |\n"
        )
        f.write("|-----:|------:|--------:|----------:|---------:|-------:|-----:|--------:|-------:|\n")
        for order in ORDERS:
            for nodes in NODE_LADDER:
                want = repeats_for_nodes(nodes)
                cleans = by_key.get((order, nodes, "clean"), [])
                ok = [
                    r for r in cleans
                    if str(r.get("admit")) == "ok" and r.get("wall_step_s")
                ]
                n_ok = len(ok)
                scale = scale_map.get((order, nodes))
                wall = scale["wall_step_s_median"] if scale else ""
                eff = scale["weak_eff"] if scale else ""
                spread = ""
                if scale and n_ok:
                    lo = float(scale["wall_step_s_min"])
                    hi = float(scale["wall_step_s_max"])
                    if lo > 0:
                        spread = "%.2f%%" % (100.0 * (hi - lo) / lo)
                diags = by_key.get((order, nodes, "diagnostic"), [])
                diag_n = sum(
                    1 for r in diags
                    if str(r.get("admit")) == "ok" and r.get("inner_s")
                )
                pending = failed = 0
                names = [
                    "h3d108c-fd%d-%dn-r%d" % (order, nodes, r)
                    for r in range(1, want + 1)
                ]
                if nodes in DIAG_NODES:
                    names.append("h3d108d-fd%d-%dn-r1" % (order, nodes))
                for name in names:
                    st = (slurm.get(name) or {}).get("state", "")
                    if st == "PENDING":
                        pending += 1
                    elif st in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                        failed += 1
                f.write(
                    "| %d | %d | %d/%d | %s | %s | %s | %d | %d | %d |\n"
                    % (
                        order, nodes, n_ok, want, wall, eff, spread,
                        diag_n, pending, failed,
                    )
                )
        f.write("\nRejected and missing runs stay in `runs.csv`.\n")


def harvest(root: str, out_dir: str, warmup: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    rows = collect_root(root, warmup)
    write_csv(os.path.join(out_dir, "runs.csv"), RUN_FIELDS, rows)
    scale = analyze(rows)
    write_csv(os.path.join(out_dir, "scaling.csv"), SCALE_FIELDS, scale)
    notes = notes_for(scale)
    write_markdown(os.path.join(out_dir, "scaling.md"), scale, notes)
    slurm = _slurm_table()
    write_status_markdown(os.path.join(out_dir, "status.md"), rows, scale, slurm)
    n_ok = sum(1 for r in rows if str(r.get("admit")) == "ok" and r.get("wall_step_s"))
    n_rej = sum(1 for r in rows if str(r.get("admit")) != "ok")
    print("harvest rows=%d valid=%d rejected_or_incomplete=%d" % (
        len(rows), n_ok, n_rej
    ))
    print("wrote %s" % os.path.join(out_dir, "status.md"))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--ladder", action="store_true")
    p.add_argument("--geometry", action="store_true")
    p.add_argument("--timed-protocol", type=int, metavar="NODES")
    p.add_argument("--repeats", type=int, metavar="NODES")
    p.add_argument("--collect")
    p.add_argument("--analyze")
    p.add_argument("--harvest")
    p.add_argument("--out")
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args(argv)
    if args.check:
        return check()
    if args.ladder:
        return print_ladder()
    if args.timed_protocol is not None:
        steps, warm = timed_protocol(args.timed_protocol)
        print("%d %d" % (steps, warm))
        return 0
    if args.repeats is not None:
        print(repeats_for_nodes(args.repeats))
        return 0
    if args.geometry:
        if not args.out:
            print("--geometry requires --out CSV", file=sys.stderr)
            return 2
        write_csv(args.out, GEOMETRY_FIELDS, geometry_table())
        print("wrote %s" % args.out)
        return 0
    if args.harvest:
        if not args.out:
            print("--harvest requires --out DIR", file=sys.stderr)
            return 2
        return harvest(args.harvest, args.out, args.warmup)
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

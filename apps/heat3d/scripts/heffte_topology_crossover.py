#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Issue #106 HeFFTe topology-crossover campaign: order, ladders, collect.

The #61 tournament used a fixed sequential protocol order on power-of-two
node counts. This helper randomizes protocol order with a recorded seed,
samples densely around the previously observed ranking changes, and keeps
per-allocation values.

    python3 apps/heat3d/scripts/heffte_topology_crossover.py --check
    python3 apps/heat3d/scripts/heffte_topology_crossover.py --wave1
    python3 apps/heat3d/scripts/heffte_topology_crossover.py \\
        --order --family 768 --nodes 128 --repeat 1
    python3 apps/heat3d/scripts/heffte_topology_crossover.py \\
        --collect DIR --out CSV
    python3 apps/heat3d/scripts/heffte_topology_crossover.py \\
        --analyze CSV --out DIR
    python3 apps/heat3d/scripts/heffte_topology_crossover.py \\
        --descriptors --out CSV
    python3 apps/heat3d/scripts/heffte_topology_crossover.py \\
        --harvest DIR --out DIR
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import random
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from heffte_comm_plan import (  # noqa: E402
    COMM_MODELS,
    DESCRIPTOR_FIELDS,
    descriptor_rows,
    write_csv as write_descriptor_csv,
)

PROTOCOLS = ("p2p_plined", "p2p", "alltoallv", "alltoall")
GCDS_PER_NODE = 8
FAMILY_PRIMARY = 768
# LUMI-G standard-g accepts 1..1024 nodes. 124 is the documented electrical
# group size; 248 and 496 are integer multiples. Those counts are candidate
# sampling points, not verified topology labels.
WAVE1_NODES = (
    112,
    124,
    128,
    136,
    240,
    248,
    256,
    264,
    480,
    496,
    512,
    528,
)
WAVE1_REPEATS = 3
SEED_SALT = 106768

RUN_FIELDS = (
    "issue",
    "family",
    "job",
    "account",
    "nodes",
    "ranks",
    "partition",
    "repeat",
    "protocol_seed",
    "protocol_order",
    "protocol_index",
    "Nx",
    "Ny",
    "Nz",
    "local_inbox",
    "inbox_xyz",
    "outbox_xyz",
    "real_grid",
    "complex_grid",
    "reshape",
    "use_pencils",
    "use_reorder",
    "use_gpu_aware",
    "admit",
    "reason",
    "revision",
    "dirty",
    "bin_sha256",
    "heffte_module",
    "mpi_module",
    "wall_step_s",
    "wall_step_spread_s",
    "checksum",
    "n_hosts",
    "nid_min",
    "nid_max",
    "xname_cabinets",
    "scratch",
)

SCALE_FIELDS = (
    "family",
    "reshape",
    "nodes",
    "ranks",
    "n_repeats",
    "wall_step_s_median",
    "wall_step_s_min",
    "wall_step_s_max",
    "ratio_to_best",
    "rank_at_scale",
    "local_slope",
    "neighbor_nodes",
    "outbox_elems",
    "complex_grid",
    "jobs",
)

ALLOC_FIELDS = (
    "job",
    "nodes",
    "repeat",
    "protocol_seed",
    "protocol_order",
    "winner",
    "second",
    "winner_s",
    "second_s",
    "winner_ratio",
    "spread_pct",
    "n_hosts",
    "nid_min",
    "nid_max",
    "xname_cabinets",
    "account",
)


def local_grid(family: int, nodes: int) -> Tuple[int, int, int, int, int]:
    ranks = nodes * GCDS_PER_NODE
    return nodes, family, family, family * ranks, ranks


def wave1() -> List[Tuple[int, int, int, int, int]]:
    return [local_grid(FAMILY_PRIMARY, n) for n in WAVE1_NODES]


def protocol_seed(family: int, nodes: int, repeat: int) -> int:
    return SEED_SALT * 100000 + family * 1000 + nodes * 10 + repeat


def protocol_order(family: int, nodes: int, repeat: int) -> Tuple[int, List[str]]:
    seed = protocol_seed(family, nodes, repeat)
    rng = random.Random(seed)
    algs = list(PROTOCOLS)
    rng.shuffle(algs)
    return seed, algs


def protocol_order_colon(family: int, nodes: int, repeat: int) -> str:
    seed, algs = protocol_order(family, nodes, repeat)
    return "seed=%d protocols=%s" % (seed, ":".join(algs))


def check() -> int:
    rc = 0
    seen = set()
    print("family nodes ranks   Nx   Ny      Nz  local order_seed")
    for nodes, nx, ny, nz, ranks in wave1():
        loc = (nx, ny, nz // ranks)
        ok = loc == (FAMILY_PRIMARY, FAMILY_PRIMARY, FAMILY_PRIMARY)
        seed, algs = protocol_order(FAMILY_PRIMARY, nodes, 1)
        key = (nodes, seed, tuple(algs))
        print(
            "%6d %5d %5d %4d %4d %7d  %dx%dx%d %s seed=%d %s"
            % (
                FAMILY_PRIMARY,
                nodes,
                ranks,
                nx,
                ny,
                nz,
                loc[0],
                loc[1],
                loc[2],
                "ok" if ok else "FAIL",
                seed,
                ":".join(algs),
            )
        )
        if not ok or nodes in seen:
            rc = 1
        seen.add(nodes)
        _ = key
    orders = [
        tuple(protocol_order(FAMILY_PRIMARY, n, r)[1])
        for n in WAVE1_NODES
        for r in range(1, WAVE1_REPEATS + 1)
    ]
    if len(set(orders)) < 2:
        print("FAIL protocol orders are not diversified")
        rc = 1
    else:
        print("distinct first-wave orders: %d" % len(set(orders)))
    return rc


def print_wave1() -> int:
    print("nodes Nx Ny Nz ranks")
    for nodes, nx, ny, nz, ranks in wave1():
        print("%d %d %d %d %d" % (nodes, nx, ny, nz, ranks))
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
    if value is None or value == "" or value == "unset" or value == "na":
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


def banner_map(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("HEAT3D_SPECTRAL_HIP"):
                continue
            if line.startswith("HEAT3D_SPECTRAL_HIP_CHECKSUM"):
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
            if line.startswith("HEAT3D_SPECTRAL_HIP_CHECKSUM "):
                return line.strip()
    return ""


def wall_step_stats(
    doc: Dict[str, Any], warmup: int
) -> Tuple[Optional[float], Optional[float]]:
    names = list(doc.get("frame_metric_names") or [])
    try:
        idx = names.index("wall_step")
    except ValueError:
        if int(doc.get("schema_version") or 0) == 4:
            m = (doc.get("metrics") or {}).get("wall_step") or {}
            if "median" in m:
                return float(m["median"]), None
        return None, None
    step_idx = names.index("step") if "step" in names else None
    values: List[float] = []
    for rank in doc.get("ranks") or []:
        frames = rank.get("frames") or []
        for i, frame in enumerate(frames):
            scalars = frame.get("scalars") or []
            if step_idx is not None and step_idx >= len(scalars):
                return None, None
            step = scalars[step_idx] if step_idx is not None else i
            if step < warmup:
                continue
            if idx < len(scalars):
                values.append(float(scalars[idx]))
    if not values:
        return None, None
    med = float(statistics.median(values))
    spread = float(max(values) - min(values)) if len(values) > 1 else 0.0
    return med, spread


def _xyz_product(text: str) -> Optional[int]:
    if not text or "x" not in text:
        return None
    parts = text.lower().split("x")
    if len(parts) != 3:
        return None
    try:
        vals = [int(p) for p in parts]
    except ValueError:
        return None
    prod = 1
    for v in vals:
        prod *= v
    return prod


def _placement_summary(run_dir: str) -> Dict[str, str]:
    hosts_path = os.path.join(run_dir, "placement", "hostnames.txt")
    facts_path = os.path.join(run_dir, "placement", "node_facts.txt")
    n_hosts = ""
    nid_min = ""
    nid_max = ""
    cabinets: List[str] = []
    if os.path.isfile(hosts_path):
        with open(hosts_path) as f:
            hosts = [ln.strip() for ln in f if ln.strip()]
        n_hosts = str(len(hosts))
    nids: List[int] = []
    if os.path.isfile(facts_path):
        with open(facts_path) as f:
            for line in f:
                meta = {}
                for tok in line.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta[k] = v
                nid = _parse_int(meta.get("nid"))
                if nid is not None:
                    nids.append(nid)
                xname = meta.get("xname") or meta.get("cname") or ""
                # Cray EX xname: x<cabinet>c<chassis>s<slot>b<board>n<node>
                m = re.match(r"x(\d+)c", xname)
                if m:
                    cabinets.append(m.group(1))
    if nids:
        nid_min = str(min(nids))
        nid_max = str(max(nids))
    cab = ",".join(sorted(set(cabinets))) if cabinets else ""
    return {
        "n_hosts": n_hosts,
        "nid_min": nid_min,
        "nid_max": nid_max,
        "xname_cabinets": cab,
    }


def collect_protocol(
    run_dir: str, proto_dir: str, proto: str, warmup: int
) -> Optional[Dict[str, Any]]:
    meta = _meta_map(os.path.join(run_dir, "run_meta.txt"))
    admit = _meta_map(os.path.join(proto_dir, "admit.txt"))
    log = os.path.join(proto_dir, "run.log")
    banner = banner_map(log)
    nx = _parse_int(meta.get("Nx"))
    ny = _parse_int(meta.get("Ny"))
    nz = _parse_int(meta.get("Nz"))
    nodes = _parse_int(meta.get("nodes")) or 0
    ranks = _parse_int(meta.get("ntasks")) or 0
    family = _parse_int(meta.get("family"))
    if family is None and nx in (512, 768):
        family = nx
    order = meta.get("protocols") or meta.get("protocol_order") or ""
    order = order.replace(",", ":")
    algs = [a for a in order.split(":") if a]
    proto_index = ""
    if proto in algs:
        proto_index = str(algs.index(proto))
    admit_flag = admit.get("admit", "")
    reason = admit.get("reason", "")
    wall = None
    spread = None
    prof = os.path.join(proto_dir, "timing_profile.json")
    if os.path.isfile(prof) and admit_flag == "ok":
        with open(prof) as f:
            doc = json.load(f)
        wall, spread = wall_step_stats(doc, warmup)
    inbox_xyz = banner.get("inbox_xyz", "")
    expected_local = ""
    if family and ranks:
        expected_local = "%dx%dx%d" % (family, family, family)
        if inbox_xyz and inbox_xyz != expected_local:
            admit_flag = "reject"
            reason = "inbox_%s_expected_%s" % (inbox_xyz, expected_local)
            wall = None
    banner_reshape = banner.get("reshape") or admit.get("banner_reshape") or ""
    if admit_flag == "ok" and banner_reshape and banner_reshape != proto:
        admit_flag = "reject"
        reason = "banner_reshape_%s_expected_%s" % (banner_reshape, proto)
        wall = None
    reshape = proto
    placement = _placement_summary(run_dir)
    return {
        "issue": "106",
        "family": family or "",
        "job": meta.get("job", os.path.basename(run_dir)),
        "account": meta.get("account", ""),
        "nodes": nodes,
        "ranks": ranks,
        "partition": meta.get("partition", ""),
        "repeat": meta.get("repeat", ""),
        "protocol_seed": meta.get("protocol_seed", ""),
        "protocol_order": order,
        "protocol_index": proto_index,
        "Nx": nx or "",
        "Ny": ny or "",
        "Nz": nz or "",
        "local_inbox": inbox_xyz or expected_local,
        "inbox_xyz": inbox_xyz,
        "outbox_xyz": banner.get("outbox_xyz", ""),
        "real_grid": banner.get("real_grid", ""),
        "complex_grid": banner.get("complex_grid", ""),
        "reshape": reshape,
        "use_pencils": banner.get("use_pencils", ""),
        "use_reorder": banner.get("use_reorder", ""),
        "use_gpu_aware": banner.get("gpu_aware", ""),
        "admit": admit_flag or "reject",
        "reason": reason or "missing_admit",
        "revision": meta.get("revision", ""),
        "dirty": meta.get("dirty", ""),
        "bin_sha256": meta.get("bin_sha256", ""),
        "heffte_module": meta.get("heffte_module", ""),
        "mpi_module": meta.get("mpi_module", ""),
        "wall_step_s": "" if wall is None else "%.8f" % wall,
        "wall_step_spread_s": "" if spread is None else "%.8f" % spread,
        "checksum": checksum_line(log),
        "n_hosts": placement["n_hosts"],
        "nid_min": placement["nid_min"],
        "nid_max": placement["nid_max"],
        "xname_cabinets": placement["xname_cabinets"],
        "scratch": proto_dir,
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
        for proto in PROTOCOLS:
            proto_dir = os.path.join(run, proto)
            if os.path.isdir(proto_dir):
                row = collect_protocol(run, proto_dir, proto, warmup)
                if row is not None:
                    rows.append(row)
                continue
            meta = _meta_map(os.path.join(run, "run_meta.txt"))
            admit = _meta_map(os.path.join(run, "admit.txt"))
            rows.append(
                {
                    "issue": "106",
                    "family": meta.get("family", FAMILY_PRIMARY),
                    "job": meta.get("job", name),
                    "account": meta.get("account", ""),
                    "nodes": meta.get("nodes", ""),
                    "ranks": meta.get("ntasks", ""),
                    "partition": meta.get("partition", ""),
                    "repeat": meta.get("repeat", ""),
                    "protocol_seed": meta.get("protocol_seed", ""),
                    "protocol_order": meta.get("protocols")
                    or meta.get("protocol_order")
                    or "",
                    "protocol_index": "",
                    "Nx": meta.get("Nx", ""),
                    "Ny": meta.get("Ny", ""),
                    "Nz": meta.get("Nz", ""),
                    "local_inbox": "",
                    "inbox_xyz": "",
                    "outbox_xyz": "",
                    "real_grid": "",
                    "complex_grid": "",
                    "reshape": proto,
                    "use_pencils": "",
                    "use_reorder": "",
                    "use_gpu_aware": "",
                    "admit": "missing",
                    "reason": admit.get("reason") or "protocol_dir_absent",
                    "revision": meta.get("revision", ""),
                    "dirty": meta.get("dirty", ""),
                    "bin_sha256": meta.get("bin_sha256", ""),
                    "heffte_module": meta.get("heffte_module", ""),
                    "mpi_module": meta.get("mpi_module", ""),
                    "wall_step_s": "",
                    "wall_step_spread_s": "",
                    "checksum": "",
                    "n_hosts": "",
                    "nid_min": "",
                    "nid_max": "",
                    "xname_cabinets": "",
                    "scratch": run,
                }
            )
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


def _key_scale(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("family") or ""),
        str(row.get("reshape") or ""),
        int(row.get("nodes") or 0),
        int(row.get("ranks") or 0),
    )


def analyze(
    rows: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_scale: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    by_job: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("admit")) != "ok":
            continue
        if not row.get("wall_step_s"):
            continue
        by_scale[_key_scale(row)].append(row)
        by_job[str(row.get("job") or "")].append(row)

    medians: Dict[Tuple[str, str, int], float] = {}
    out: List[Dict[str, Any]] = []
    for key, group in sorted(by_scale.items()):
        family, reshape, nodes, ranks = key
        walls = [float(r["wall_step_s"]) for r in group]
        med = float(statistics.median(walls))
        medians[(str(family), str(reshape), int(nodes))] = med
        jobs = ",".join(sorted({str(r.get("job") or "") for r in group}))
        outbox = group[0].get("outbox_xyz") or ""
        out.append(
            {
                "family": family,
                "reshape": reshape,
                "nodes": nodes,
                "ranks": ranks,
                "n_repeats": len(walls),
                "wall_step_s_median": "%.8f" % med,
                "wall_step_s_min": "%.8f" % min(walls),
                "wall_step_s_max": "%.8f" % max(walls),
                "ratio_to_best": "",
                "rank_at_scale": "",
                "local_slope": "",
                "neighbor_nodes": "",
                "outbox_elems": _xyz_product(str(outbox)) or "",
                "complex_grid": group[0].get("complex_grid", ""),
                "jobs": jobs,
            }
        )

    ranking_groups: Dict[Tuple[str, int], List[Tuple[str, float]]] = defaultdict(list)
    for row in out:
        family = str(row["family"])
        reshape = str(row["reshape"])
        nodes = int(row["nodes"])
        med = float(row["wall_step_s_median"])
        ranking_groups[(family, nodes)].append((reshape, med))

    best: Dict[Tuple[str, int], float] = {}
    rank_map: Dict[Tuple[str, int, str], int] = {}
    for (family, nodes), items in ranking_groups.items():
        items.sort(key=lambda x: (x[1], x[0]))
        best[(family, nodes)] = items[0][1]
        for i, (reshape, _) in enumerate(items, start=1):
            rank_map[(family, nodes, reshape)] = i

    node_lists: Dict[str, List[int]] = defaultdict(list)
    for family, reshape, nodes in medians:
        node_lists[family].append(nodes)
    for family in node_lists:
        node_lists[family] = sorted(set(node_lists[family]))

    for row in out:
        family = str(row["family"])
        reshape = str(row["reshape"])
        nodes = int(row["nodes"])
        med = float(row["wall_step_s_median"])
        b = best.get((family, nodes))
        if b and b > 0:
            row["ratio_to_best"] = "%.4f" % (med / b)
        rnk = rank_map.get((family, nodes, reshape))
        if rnk is not None:
            row["rank_at_scale"] = rnk
        neighbors = node_lists.get(family) or []
        prev_n = None
        next_n = None
        if nodes in neighbors:
            i = neighbors.index(nodes)
            if i > 0:
                prev_n = neighbors[i - 1]
            if i + 1 < len(neighbors):
                next_n = neighbors[i + 1]
        slope_bits = []
        if prev_n is not None:
            prev_t = medians.get((family, reshape, prev_n))
            if prev_t and prev_t > 0:
                slope_bits.append("%.4f" % (med / prev_t))
        if next_n is not None:
            next_t = medians.get((family, reshape, next_n))
            if next_t and med > 0:
                slope_bits.append("%.4f" % (next_t / med))
        row["local_slope"] = ",".join(slope_bits)
        neigh = []
        if prev_n is not None:
            neigh.append(str(prev_n))
        neigh.append(str(nodes))
        if next_n is not None:
            neigh.append(str(next_n))
        row["neighbor_nodes"] = "-".join(neigh)

    alloc_rows: List[Dict[str, Any]] = []
    for job, group in sorted(by_job.items()):
        if len(group) < len(PROTOCOLS):
            continue
        ranked = sorted(group, key=lambda r: (float(r["wall_step_s"]), str(r["reshape"])))
        winner = ranked[0]
        second = ranked[1] if len(ranked) > 1 else ranked[0]
        w = float(winner["wall_step_s"])
        s = float(second["wall_step_s"])
        walls = [float(r["wall_step_s"]) for r in ranked]
        spread = 0.0
        if min(walls) > 0:
            spread = 100.0 * (max(walls) - min(walls)) / min(walls)
        alloc_rows.append(
            {
                "job": job,
                "nodes": winner.get("nodes", ""),
                "repeat": winner.get("repeat", ""),
                "protocol_seed": winner.get("protocol_seed", ""),
                "protocol_order": winner.get("protocol_order", ""),
                "winner": winner.get("reshape", ""),
                "second": second.get("reshape", ""),
                "winner_s": "%.8f" % w,
                "second_s": "%.8f" % s,
                "winner_ratio": "%.4f" % (s / w if w > 0 else 0.0),
                "spread_pct": "%.2f" % spread,
                "n_hosts": winner.get("n_hosts", ""),
                "nid_min": winner.get("nid_min", ""),
                "nid_max": winner.get("nid_max", ""),
                "xname_cabinets": winner.get("xname_cabinets", ""),
                "account": winner.get("account", ""),
            }
        )
    return out, alloc_rows


def crossover_notes(
    scale_rows: Sequence[Dict[str, Any]], alloc_rows: Sequence[Dict[str, Any]]
) -> List[str]:
    notes: List[str] = []
    by_fam: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scale_rows:
        by_fam[str(row["family"])].append(row)
    for family, rows in sorted(by_fam.items()):
        scales = sorted({int(r["nodes"]) for r in rows})
        unique = []
        for nodes in scales:
            at = [r for r in rows if int(r["nodes"]) == nodes]
            at.sort(key=lambda r: (float(r["wall_step_s_median"]), str(r["reshape"])))
            if not at:
                continue
            alg = str(at[0]["reshape"])
            if not unique or unique[-1][1] != alg:
                unique.append((nodes, alg))
        if len(unique) <= 1 and unique:
            notes.append(
                "family %s: pooled median ranking is %s from %d to %d nodes"
                % (family, unique[0][1], scales[0], scales[-1])
            )
        elif unique:
            notes.append(
                "family %s: pooled ranking changes: " % family
                + "; ".join("%d nodes -> %s" % (n, a) for n, a in unique)
            )
    by_nodes: Dict[int, List[str]] = defaultdict(list)
    close = 0
    for row in alloc_rows:
        nodes = int(row["nodes"] or 0)
        by_nodes[nodes].append(str(row["winner"]))
        if _parse_float(str(row.get("winner_ratio") or "")) is not None:
            if float(row["winner_ratio"]) < 1.05:
                close += 1
    for nodes, winners in sorted(by_nodes.items()):
        uniq = sorted(set(winners))
        if len(uniq) == 1:
            notes.append(
                "%d nodes: all %d allocations elect %s"
                % (nodes, len(winners), uniq[0])
            )
        else:
            notes.append(
                "%d nodes: allocation winners disagree: %s"
                % (nodes, ", ".join("%s x%d" % (w, winners.count(w)) for w in uniq))
            )
    if close:
        notes.append(
            "%d allocations have winner/second within 5%%; do not treat those "
            "rankings as decisive" % close
        )
    notes.append(
        "xname cabinet fields are parsed from exposed Cray xname/cname strings; "
        "they are not LUMI-G electrical-group labels"
    )
    return notes


def write_markdown(
    path: str,
    scale_rows: Sequence[Dict[str, Any]],
    alloc_rows: Sequence[Dict[str, Any]],
    notes: Sequence[str],
) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write("# HeFFTe topology crossover (issue #106)\n\n")
        f.write(
            "Per-allocation values are retained. Pooled medians are a summary, "
            "not a selector. Protocol order is randomized per allocation. Do "
            "not change the production reshape default from these numbers.\n\n"
        )
        f.write("## Per-allocation winners\n\n")
        f.write(
            "| job | nodes | r | seed | order | winner | second | "
            "T_best | T_2/T_1 | hosts | nid range |\n"
        )
        f.write("|----:|-----:|--:|-----:|-------|--------|--------|------:|--------:|-----:|-----------|\n")
        for r in alloc_rows:
            f.write(
                "| %s | %s | %s | %s | `%s` | `%s` | `%s` | %s | %s | %s | %s-%s |\n"
                % (
                    r["job"],
                    r["nodes"],
                    r["repeat"],
                    r["protocol_seed"],
                    r["protocol_order"],
                    r["winner"],
                    r["second"],
                    r["winner_s"],
                    r["winner_ratio"],
                    r["n_hosts"],
                    r["nid_min"],
                    r["nid_max"],
                )
            )
        f.write("\n## Pooled medians\n\n")
        families = sorted({str(r["family"]) for r in scale_rows})
        for family in families:
            f.write("## Local %s³ / GCD\n\n" % family)
            f.write(
                "| nodes | ranks | reshape | n | median s | min | max | "
                "vs best | slope | rank |\n"
            )
            f.write("|------:|------:|---------|--:|---------:|----:|----:|--------:|------:|-----:|\n")
            fam_rows = [r for r in scale_rows if str(r["family"]) == family]
            fam_rows.sort(
                key=lambda r: (int(r["nodes"]), int(r["ranks"]), str(r["reshape"]))
            )
            for r in fam_rows:
                f.write(
                    "| %s | %s | `%s` | %s | %s | %s | %s | %s | %s | %s |\n"
                    % (
                        r["nodes"],
                        r["ranks"],
                        r["reshape"],
                        r["n_repeats"],
                        r["wall_step_s_median"],
                        r["wall_step_s_min"],
                        r["wall_step_s_max"],
                        r["ratio_to_best"],
                        r["local_slope"],
                        r["rank_at_scale"],
                    )
                )
            f.write("\n")
        f.write("## Notes\n\n")
        if notes:
            for n in notes:
                f.write("- %s\n" % n)
        else:
            f.write("- no admitted points\n")
        f.write("\n")


ORDER_FIELDS = (
    "job",
    "nodes",
    "repeat",
    "protocol_seed",
    "protocol_order",
    "protocol_index",
    "reshape",
    "admit",
    "reason",
    "wall_step_s",
    "checksum_ok",
)


def expected_wave1(family: int = FAMILY_PRIMARY) -> List[Dict[str, Any]]:
    rows = []
    for nodes in WAVE1_NODES:
        for repeat in range(1, WAVE1_REPEATS + 1):
            seed, algs = protocol_order(family, nodes, repeat)
            name = "h3d106-%d-%dn-r%d" % (family, nodes, repeat)
            rows.append(
                {
                    "name": name,
                    "nodes": nodes,
                    "repeat": repeat,
                    "ranks": nodes * GCDS_PER_NODE,
                    "seed": seed,
                    "protocols": algs,
                }
            )
    return rows


def _slurm_table(kind: str) -> Dict[str, Dict[str, str]]:
    """Map job name -> {jobid, state} from squeue or sacct. Empty if unavailable."""
    out: Dict[str, Dict[str, str]] = {}
    try:
        if kind == "squeue":
            cmd = [
                "squeue",
                "-u",
                os.environ.get("USER", "juaho"),
                "-A",
                "project_462001245",
                "-h",
                "-o",
                "%i|%j|%T",
            ]
        else:
            cmd = [
                "sacct",
                "-X",
                "-S",
                "2026-09-20",
                "-u",
                os.environ.get("USER", "juaho"),
                "-A",
                "project_462001245",
                "-n",
                "-P",
                "--format=JobID,JobName,State",
            ]
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return out
    if proc.returncode != 0:
        return out
    for line in proc.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        jobid, name, state = parts[0], parts[1], parts[2].split()[0]
        if not name.startswith("h3d106-"):
            continue
        prev = out.get(name)
        if prev is None or kind == "squeue":
            out[name] = {"job": jobid, "state": state}
    return out


def classify_row(row: Dict[str, Any]) -> str:
    admit = str(row.get("admit") or "")
    if admit == "ok" and row.get("wall_step_s"):
        return "valid_measurement"
    if admit == "ok":
        return "completed_no_wall"
    if admit == "missing":
        return "missing_protocol"
    if admit == "reject":
        reason = str(row.get("reason") or "")
        if "banner" in reason:
            return "invalid_protocol_banner"
        return "rejected"
    return "unknown"


def order_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        cs = str(row.get("checksum") or "")
        checksum_ok = ""
        if cs:
            checksum_ok = "no" if ("nan" in cs.lower() or "inf" in cs.lower()) else "yes"
        out.append(
            {
                "job": row.get("job", ""),
                "nodes": row.get("nodes", ""),
                "repeat": row.get("repeat", ""),
                "protocol_seed": row.get("protocol_seed", ""),
                "protocol_order": row.get("protocol_order", ""),
                "protocol_index": row.get("protocol_index", ""),
                "reshape": row.get("reshape", ""),
                "admit": row.get("admit", ""),
                "reason": row.get("reason", ""),
                "wall_step_s": row.get("wall_step_s", ""),
                "checksum_ok": checksum_ok,
            }
        )
    return out


def ranking_unresolved(alloc_at_nodes: Sequence[Dict[str, Any]], n_complete: int) -> str:
    if n_complete < WAVE1_REPEATS:
        return "yes_repeats_incomplete"
    close = 0
    winners = []
    for row in alloc_at_nodes:
        winners.append(str(row.get("winner") or ""))
        ratio = _parse_float(str(row.get("winner_ratio") or ""))
        if ratio is not None and ratio < 1.05:
            close += 1
    if len(set(winners)) > 1:
        return "yes_winners_disagree"
    if close:
        return "yes_winner_second_within_5pct"
    return "no"


def write_status_markdown(
    path: str,
    rows: Sequence[Dict[str, Any]],
    scale_rows: Sequence[Dict[str, Any]],
    alloc_rows: Sequence[Dict[str, Any]],
    slurm: Dict[str, Dict[str, str]],
) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    by_nodes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_nodes[int(row.get("nodes") or 0)].append(row)
    alloc_by_nodes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in alloc_rows:
        alloc_by_nodes[int(row.get("nodes") or 0)].append(row)
    with open(path, "w") as f:
        f.write("# Issue #106 campaign status\n\n")
        f.write(
            "Generated harvest. Do not declare a winner from one allocation "
            "when close competitors remain queued. This file is overwritten "
            "on harvest; raw run directories are never modified.\n\n"
        )
        f.write(
            "| nodes | completed/requested | best | T_best | spread | "
            "unresolved | PENDING | RUNNING | FAILED |\n"
        )
        f.write("|------:|--------------------:|------|-------:|-------:|-----------|--------:|--------:|-------:|\n")
        for spec in expected_wave1():
            _ = spec
        for nodes in WAVE1_NODES:
            group = [r for r in by_nodes.get(nodes, []) if str(r.get("reshape"))]
            ok_jobs = {
                str(r.get("job"))
                for r in group
                if classify_row(r) == "valid_measurement"
            }
            n_complete = 0
            for repeat in range(1, WAVE1_REPEATS + 1):
                name = "h3d106-%d-%dn-r%d" % (FAMILY_PRIMARY, nodes, repeat)
                live = slurm.get(name, {})
                # completed if 4 valid protocol measurements exist for that repeat
                n_ok = sum(
                    1
                    for r in group
                    if str(r.get("repeat")) == str(repeat)
                    and classify_row(r) == "valid_measurement"
                )
                if n_ok >= len(PROTOCOLS):
                    n_complete += 1
                _ = live
            pending = running = failed = 0
            for repeat in range(1, WAVE1_REPEATS + 1):
                name = "h3d106-%d-%dn-r%d" % (FAMILY_PRIMARY, nodes, repeat)
                st = (slurm.get(name) or {}).get("state", "")
                if st == "PENDING":
                    pending += 1
                elif st == "RUNNING":
                    running += 1
                elif st in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                    failed += 1
            at = [r for r in scale_rows if int(r.get("nodes") or 0) == nodes]
            at.sort(key=lambda r: (float(r["wall_step_s_median"]), str(r["reshape"])))
            best = at[0]["reshape"] if at else ""
            tbest = at[0]["wall_step_s_median"] if at else ""
            spread = ""
            if at:
                lo = float(at[0]["wall_step_s_median"])
                hi = max(float(r["wall_step_s_median"]) for r in at)
                if lo > 0:
                    spread = "%.2f%%" % (100.0 * (hi - lo) / lo)
            unresolved = ranking_unresolved(alloc_by_nodes.get(nodes, []), n_complete)
            f.write(
                "| %d | %d/%d | `%s` | %s | %s | %s | %d | %d | %d |\n"
                % (
                    nodes,
                    n_complete,
                    WAVE1_REPEATS,
                    best,
                    tbest,
                    spread,
                    unresolved,
                    pending,
                    running,
                    failed,
                )
            )
        f.write("\nCounts are allocation-level. Protocol rows that failed or are ")
        f.write("still missing stay in `runs.csv`; they are never dropped.\n")
        _ = ok_jobs


def harvest(root: str, out_dir: str, warmup: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    rows = collect_root(root, warmup)
    runs_csv = os.path.join(out_dir, "runs.csv")
    write_csv(runs_csv, RUN_FIELDS, rows)
    scale, alloc = analyze(rows)
    write_csv(os.path.join(out_dir, "scaling.csv"), SCALE_FIELDS, scale)
    write_csv(os.path.join(out_dir, "allocations.csv"), ALLOC_FIELDS, alloc)
    write_csv(os.path.join(out_dir, "order.csv"), ORDER_FIELDS, order_rows(rows))
    notes = crossover_notes(scale, alloc)
    write_markdown(os.path.join(out_dir, "scaling.md"), scale, alloc, notes)
    slurm = {}
    slurm.update(_slurm_table("sacct"))
    slurm.update(_slurm_table("squeue"))
    write_status_markdown(
        os.path.join(out_dir, "status.md"), rows, scale, alloc, slurm
    )
    n_ok = sum(1 for r in rows if classify_row(r) == "valid_measurement")
    n_fail = sum(1 for r in rows if classify_row(r) in ("rejected", "invalid_protocol_banner"))
    n_miss = sum(1 for r in rows if classify_row(r) == "missing_protocol")
    print("harvest rows=%d valid=%d rejected=%d missing=%d" % (
        len(rows), n_ok, n_fail, n_miss
    ))
    print("wrote %s" % os.path.join(out_dir, "status.md"))
    for n in notes:
        print(n)
    _ = COMM_MODELS
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--wave1", action="store_true")
    p.add_argument("--order", action="store_true")
    p.add_argument("--family", type=int, default=FAMILY_PRIMARY)
    p.add_argument("--nodes", type=int)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--collect")
    p.add_argument("--analyze")
    p.add_argument("--harvest")
    p.add_argument("--descriptors", action="store_true")
    p.add_argument("--out")
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args(argv)

    if args.check:
        return check()
    if args.wave1:
        return print_wave1()
    if args.order:
        if args.nodes is None:
            print("--order requires --nodes", file=sys.stderr)
            return 2
        print(protocol_order_colon(args.family, args.nodes, args.repeat))
        return 0
    if args.descriptors:
        if not args.out:
            print("--descriptors requires --out CSV", file=sys.stderr)
            return 2
        rows = descriptor_rows(WAVE1_NODES, args.family)
        write_descriptor_csv(args.out, rows)
        print("wrote %d descriptor rows to %s" % (len(rows), args.out))
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
        scale, alloc = analyze(rows)
        write_csv(os.path.join(args.out, "scaling.csv"), SCALE_FIELDS, scale)
        write_csv(os.path.join(args.out, "allocations.csv"), ALLOC_FIELDS, alloc)
        write_csv(os.path.join(args.out, "order.csv"), ORDER_FIELDS, order_rows(rows))
        notes = crossover_notes(scale, alloc)
        write_markdown(os.path.join(args.out, "scaling.md"), scale, alloc, notes)
        print("wrote %s" % os.path.join(args.out, "scaling.csv"))
        for n in notes:
            print(n)
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

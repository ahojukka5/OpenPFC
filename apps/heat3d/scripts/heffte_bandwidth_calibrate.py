#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Issue #119: HeFFTe-trace MPI bandwidth vs GPU OSU alltoall.

Issue #121: 32-node pencil vs slab A/B on the same diagnostic binary.
GPU-aware MPI is required (banner gpu_aware=1).

Diagnostic only. Walls from the trace binary are not production numbers.
Bytes are the admitted 768^3/GCD comm-plan replica (n_mpi=1, 2 FFTs/step)
for slabs. Pencil rows do not reuse those bytes.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

GCDS_PER_NODE = 8
FAMILY = 768
NODE_LADDER = (1, 2, 8, 32)
# Frozen #61 tournament winners at these rungs. p2p_plined is diagnostic:
# heffte-rocm-trace does not carry the production packall patch.
WINNER = {1: "p2p_plined", 2: "p2p_plined", 8: "alltoall", 32: "alltoall"}
# 32 nodes: one Heat3D protocol per srun (avoid sequential-srun 143).
HEAT3D_PROTOCOLS = {
    1: ("p2p_plined", "alltoall"),
    2: ("p2p_plined", "alltoall"),
    8: ("alltoall", "alltoallv"),
    32: ("alltoall",),
}
OSU_KINDS = {
    1: ("alltoall",),
    2: ("alltoall",),
    8: ("alltoall", "alltoallv"),
    32: ("alltoall",),
}
BYTES_STEP = {
    1: 7266631680,
    2: 7266631680,
    8: 7266631680,
    32: 7266631680,
}
BYTES_PEER = {
    1: 454164480,
    2: 227082240,
    8: 56770560,
    32: 14192640,
}
NIC_UNI_GCD = 12.5e9

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
MPI_COLL = ("all2all", "all2allv")
MPI_PREFIX = ("irecv ", "isend ", "send ")
PACK_PREFIX = ("unpacking from ",)

RUN_FIELDS = (
    "kind",
    "issue",
    "family",
    "nodes",
    "ranks",
    "reshape",
    "use_pencils",
    "gpu_aware",
    "job",
    "admit",
    "wall_step_s",
    "t_mpi_s",
    "t_pack_s",
    "t_fft_s",
    "n_mpi_coll_per_step",
    "real_grid",
    "complex_grid",
    "bytes_per_timestep",
    "bytes_per_peer",
    "gb_s_mpi",
    "gb_s_osu",
    "frac_nic_uni_gcd",
    "reason",
)


def classify(name: str) -> str:
    n = name.strip()
    if n in FFT:
        return "fft"
    if n in PACK or n.startswith(PACK_PREFIX):
        return "pack"
    if n in MPI_WAIT:
        return "mpi_wait"
    if n in MPI_COLL:
        return "mpi_coll"
    if n.startswith(MPI_PREFIX):
        return "mpi_post"
    return "other"


def parse_trace_log(path: str) -> List[Tuple[str, float, float]]:
    events: List[Tuple[str, float, float]] = []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip()
            if not line:
                continue
            name = line[:40].strip()
            rest = line[40:].split()
            if len(rest) < 2:
                continue
            events.append((name, float(rest[0]), float(rest[1])))
    return events


def sum_buckets(events: Sequence[Tuple[str, float, float]]) -> Dict[str, float]:
    out = {
        "fft": 0.0,
        "pack": 0.0,
        "mpi_wait": 0.0,
        "mpi_coll": 0.0,
        "mpi_post": 0.0,
    }
    for name, _, dur in events:
        key = classify(name)
        if key in out:
            out[key] += dur
    out["mpi"] = out["mpi_wait"] + out["mpi_coll"] + out["mpi_post"]
    return out


def find_trace_logs(run: str) -> List[str]:
    if not os.path.isdir(run):
        return []
    logs = []
    for name in os.listdir(run):
        if not name.startswith("heffte_trace_"):
            continue
        if name.endswith(".log") or name.endswith(".txt"):
            logs.append(os.path.join(run, name))
    logs.sort()
    return logs


def parse_osu_latency(path: str, size: int) -> Optional[float]:
    """Return average latency in seconds for an exact OSU size (bytes)."""
    if not os.path.isfile(path):
        return None
    found: List[float] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                msg = int(float(parts[0]))
                lat_us = float(parts[1])
            except ValueError:
                continue
            if msg == int(size):
                found.append(lat_us * 1e-6)
    if not found:
        return None
    return float(statistics.median(found))


def osu_gb_s(latency_s: float, bytes_peer: int, ranks: int) -> float:
    if latency_s <= 0.0:
        return 0.0
    payload = float(bytes_peer) * float(max(ranks - 1, 0))
    return payload / latency_s / 1e9


def critical_trace(run: str) -> Optional[Dict[str, float]]:
    logs = find_trace_logs(run)
    if not logs:
        return None
    ranked = []
    for log in logs:
        events = parse_trace_log(log)
        buckets = sum_buckets(events)
        buckets["n_mpi_coll"] = float(
            sum(1 for name, _, _ in events if name in MPI_COLL)
        )
        ranked.append(buckets)
    return max(ranked, key=lambda b: b["mpi"] + b["pack"] + b["fft"])


def _meta(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("==="):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def banner_field(path: str, key: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path) as handle:
        for line in handle:
            if "HEAT3D_SPECTRAL_HIP" not in line.split()[:1]:
                if not line.startswith("HEAT3D_SPECTRAL_HIP"):
                    continue
            for tok in line.split():
                if tok.startswith(key + "="):
                    return tok.split("=", 1)[1]
    return ""


def ladder() -> List[Tuple[int, int, int, int, int]]:
    """Return (nodes, Nx, Ny, Nz, ranks) with local inbox 768^3."""
    out = []
    for nodes in NODE_LADDER:
        ranks = nodes * GCDS_PER_NODE
        out.append((nodes, FAMILY, FAMILY, FAMILY * ranks, ranks))
    return out


def nz_for(nodes: int) -> int:
    return FAMILY * nodes * GCDS_PER_NODE


def heat3d_row(run: str) -> Optional[Dict[str, Any]]:
    meta = _meta(os.path.join(run, "run_meta.txt"))
    admit_map = _meta(os.path.join(run, "admit.txt"))
    nodes = int(meta.get("nodes") or 0)
    ranks = int(meta.get("ntasks") or nodes * GCDS_PER_NODE)
    proto = meta.get("reshape") or admit_map.get("expected_reshape") or ""
    expected_pencils = (
        meta.get("use_pencils") or admit_map.get("expected_use_pencils") or "0"
    )
    flag = admit_map.get("admit") or ""
    reason = admit_map.get("reason") or ""
    log = os.path.join(run, "run.log")
    banner = banner_field(log, "reshape")
    pencils = banner_field(log, "use_pencils") or expected_pencils
    aware = banner_field(log, "gpu_aware")
    real_grid = banner_field(log, "real_grid")
    complex_grid = banner_field(log, "complex_grid")
    if not flag:
        if not find_trace_logs(run):
            flag, reason = "reject", "missing_trace"
        elif proto and banner and banner != proto:
            flag, reason = "reject", "banner_reshape_%s" % banner
        elif pencils and pencils != expected_pencils:
            flag, reason = "reject", "banner_use_pencils_%s" % pencils
        elif aware and aware != "1":
            flag, reason = "reject", "banner_gpu_aware_%s" % aware
        else:
            flag, reason = "ok", "none"
            if not proto:
                proto = banner
    buckets = critical_trace(run)
    n_steps = int(meta.get("steps") or 20)
    t_mpi = t_pack = t_fft = n_coll = None
    if buckets and n_steps > 0:
        t_mpi = buckets["mpi"] / n_steps
        t_pack = buckets["pack"] / n_steps
        t_fft = buckets["fft"] / n_steps
        n_coll = buckets.get("n_mpi_coll", 0.0) / n_steps
    wall = None
    warmup = int(meta.get("warmup") or 1)
    prof = os.path.join(run, "timing_profile.json")
    if os.path.isfile(prof) and flag == "ok":
        try:
            with open(prof) as handle:
                data = json.load(handle)
            names = data.get("frame_metric_names") or []
            idx = names.index("wall_step") if "wall_step" in names else 2
            vals = []
            for rank in data.get("ranks") or []:
                for fr in rank.get("frames") or []:
                    sc = fr.get("scalars") or []
                    if len(sc) > idx and sc[0] >= warmup:
                        vals.append(float(sc[idx]))
            if vals:
                wall = float(statistics.median(vals))
        except (OSError, ValueError, TypeError, json.JSONDecodeError, IndexError):
            wall = None
    slab = pencils != "1"
    bstep = BYTES_STEP.get(nodes, 0) if slab else None
    bpeer = BYTES_PEER.get(nodes, 0) if slab else None
    gb = (bstep / t_mpi / 1e9) if slab and bstep and t_mpi else None
    frac = (gb * 1e9 / NIC_UNI_GCD) if gb and nodes > 1 else None
    issue = meta.get("issue") or ("121" if pencils == "1" else "119")
    return {
        "kind": "heat3d_trace",
        "issue": issue,
        "family": FAMILY,
        "nodes": nodes,
        "ranks": ranks,
        "reshape": proto,
        "use_pencils": pencils,
        "gpu_aware": aware or meta.get("gpu_aware") or "",
        "job": meta.get("job") or "",
        "admit": flag,
        "wall_step_s": wall,
        "t_mpi_s": t_mpi,
        "t_pack_s": t_pack,
        "t_fft_s": t_fft,
        "n_mpi_coll_per_step": n_coll,
        "real_grid": real_grid,
        "complex_grid": complex_grid,
        "bytes_per_timestep": bstep,
        "bytes_per_peer": bpeer,
        "gb_s_mpi": gb,
        "gb_s_osu": None,
        "frac_nic_uni_gcd": frac,
        "reason": reason,
    }


def osu_row(run: str) -> Optional[Dict[str, Any]]:
    meta = _meta(os.path.join(run, "run_meta.txt"))
    nodes = int(meta.get("nodes") or 0)
    ranks = int(meta.get("ntasks") or nodes * GCDS_PER_NODE)
    kind = meta.get("osu_kind") or meta.get("reshape") or "alltoall"
    peer = int(BYTES_PEER.get(nodes, 0))
    lat = parse_osu_latency(os.path.join(run, "osu.out"), peer)
    gb = osu_gb_s(lat, peer, ranks) if lat else None
    frac = (gb * 1e9 / NIC_UNI_GCD) if gb and nodes > 1 else None
    admit = "ok" if lat else "reject"
    return {
        "kind": "osu",
        "issue": "119",
        "family": FAMILY,
        "nodes": nodes,
        "ranks": ranks,
        "reshape": kind,
        "use_pencils": "",
        "gpu_aware": "1",
        "job": meta.get("job") or "",
        "admit": admit,
        "wall_step_s": None,
        "t_mpi_s": lat,
        "t_pack_s": None,
        "t_fft_s": None,
        "n_mpi_coll_per_step": None,
        "real_grid": "",
        "complex_grid": "",
        "bytes_per_timestep": BYTES_STEP.get(nodes, 0),
        "bytes_per_peer": peer,
        "gb_s_mpi": None,
        "gb_s_osu": gb,
        "frac_nic_uni_gcd": frac,
        "reason": "" if lat else "missing_osu_size",
    }


def collect(root: str) -> List[Dict[str, Any]]:
    runs = os.path.join(root, "runs")
    if not os.path.isdir(runs):
        return []
    rows: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(runs)):
        run = os.path.join(runs, name)
        if not os.path.isdir(run):
            continue
        meta = _meta(os.path.join(run, "run_meta.txt"))
        kind = meta.get("kind") or ""
        if (
            kind == "heat3d_trace"
            or name.startswith("h3dbw-")
            or find_trace_logs(run)
        ):
            row = heat3d_row(run)
            if row and row["nodes"]:
                rows.append(row)
        if kind == "osu" or name.startswith("osubw-") or os.path.isfile(
            os.path.join(run, "osu.out")
        ):
            row = osu_row(run)
            if row and row["nodes"]:
                rows.append(row)
    return rows


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(RUN_FIELDS), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {}
            for k in RUN_FIELDS:
                v = row.get(k)
                if v is None:
                    out[k] = ""
                elif isinstance(v, float):
                    out[k] = "%.8g" % v
                else:
                    out[k] = v
            w.writerow(out)


def check() -> int:
    assert NODE_LADDER == (1, 2, 8, 32)
    assert BYTES_STEP[8] == BYTES_STEP[1]
    assert HEAT3D_PROTOCOLS[32] == ("alltoall",)
    assert WINNER[8] == "alltoall"
    for nodes, nx, ny, nz, ranks in ladder():
        assert nx == FAMILY and ny == FAMILY
        assert nz == FAMILY * ranks
        assert ranks == nodes * GCDS_PER_NODE
        assert BYTES_PEER[nodes] > 0
    print("ok nodes=%s family=%d" % (NODE_LADDER, FAMILY))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--harvest")
    p.add_argument("--out")
    p.add_argument("--ladder", action="store_true")
    args = p.parse_args(argv)
    if args.check:
        return check()
    if args.ladder:
        print("nodes Nx Ny Nz ranks heat3d osu bytes_peer")
        for nodes, nx, ny, nz, ranks in ladder():
            print(
                "%d %d %d %d %d %s %s %d"
                % (
                    nodes,
                    nx,
                    ny,
                    nz,
                    ranks,
                    ",".join(HEAT3D_PROTOCOLS[nodes]),
                    ",".join(OSU_KINDS[nodes]),
                    BYTES_PEER[nodes],
                )
            )
        return 0
    if args.harvest:
        if not args.out:
            print("--harvest requires --out DIR", file=sys.stderr)
            return 2
        rows = collect(args.harvest)
        os.makedirs(args.out, exist_ok=True)
        write_csv(os.path.join(args.out, "bandwidth.csv"), rows)
        print("harvest rows=%d" % len(rows))
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

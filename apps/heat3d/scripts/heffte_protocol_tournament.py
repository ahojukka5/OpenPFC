#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Issue #61 HeFFTe reshape-algorithm tournament: ladders, collect, analyze.

    python3 apps/heat3d/scripts/heffte_protocol_tournament.py --check
    python3 apps/heat3d/scripts/heffte_protocol_tournament.py --ladder 768
    python3 apps/heat3d/scripts/heffte_protocol_tournament.py --collect DIR --out CSV
    python3 apps/heat3d/scripts/heffte_protocol_tournament.py --analyze CSV --out DIR
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROTOCOLS = ("p2p_plined", "p2p", "alltoallv", "alltoall")
NODE_LADDER_768 = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
NODE_LADDER_512 = (1, 2, 4, 8, 16, 32, 64, 128)
GCDS_PER_NODE = 8

RUN_FIELDS = (
    "family",
    "job",
    "account",
    "nodes",
    "ranks",
    "partition",
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
    "maxrss_kb",
    "scratch",
)

SCALE_FIELDS = (
    "family",
    "reshape",
    "nodes",
    "ranks",
    "Nx",
    "Ny",
    "Nz",
    "n_repeats",
    "wall_step_s_median",
    "wall_step_s_min",
    "wall_step_s_max",
    "weak_eff",
    "rel_p2p_plined",
    "t_double",
    "penalty_per_doubling",
    "rank_at_scale",
    "jobs",
)


def ladder(family: int) -> List[Tuple[int, int, int, int, int]]:
    """Return (nodes, Nx, Ny, Nz, ranks) with local inbox family^3."""
    if family == 768:
        nodes_list = NODE_LADDER_768
    elif family == 512:
        nodes_list = NODE_LADDER_512
    else:
        raise ValueError("family must be 768 or 512")
    out = []
    for nodes in nodes_list:
        ranks = nodes * GCDS_PER_NODE
        out.append((nodes, family, family, family * ranks, ranks))
    return out


def local_inbox_ok(nx: int, ny: int, nz: int, ranks: int, family: int) -> bool:
    if ranks < 1:
        return False
    return nx == family and ny == family and nz == family * ranks


def check() -> int:
    rc = 0
    print("family nodes ranks   Nx   Ny      Nz  local")
    for family in (768, 512):
        for nodes, nx, ny, nz, ranks in ladder(family):
            loc = (nx, ny, nz // ranks)
            ok = loc == (family, family, family) and local_inbox_ok(
                nx, ny, nz, ranks, family
            )
            print(
                "%6d %5d %5d %4d %4d %7d  %dx%dx%d %s"
                % (
                    family,
                    nodes,
                    ranks,
                    nx,
                    ny,
                    nz,
                    loc[0],
                    loc[1],
                    loc[2],
                    "ok" if ok else "FAIL",
                )
            )
            if not ok:
                rc = 1
    return rc


def print_ladder(family: int) -> int:
    print("nodes Nx Ny Nz ranks")
    for nodes, nx, ny, nz, ranks in ladder(family):
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


def banner_map(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("HEAT3D_SPECTRAL_HIP"):
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


def wall_step_stats(doc: Dict[str, Any], warmup: int) -> Tuple[Optional[float], Optional[float]]:
    names = list(doc.get("frame_metric_names") or [])
    try:
        idx = names.index("wall_step")
    except ValueError:
        if int(doc.get("schema_version") or 0) == 4:
            m = (doc.get("metrics") or {}).get("wall_step") or {}
            if "median" in m:
                return float(m["median"]), None
        return None, None
    values: List[float] = []
    for rank in doc.get("ranks") or []:
        frames = rank.get("frames") or []
        for i, frame in enumerate(frames):
            if i < warmup:
                continue
            scalars = frame.get("scalars") or []
            if idx < len(scalars):
                values.append(float(scalars[idx]))
    if not values:
        return None, None
    med = float(statistics.median(values))
    spread = float(max(values) - min(values)) if len(values) > 1 else 0.0
    return med, spread


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
    reshape = banner.get("reshape") or admit.get("banner_reshape") or proto
    if admit_flag == "ok" and reshape != proto:
        admit_flag = "reject"
        reason = "banner_reshape_%s_expected_%s" % (reshape, proto)
        wall = None
    return {
        "family": family or "",
        "job": meta.get("job", os.path.basename(run_dir)),
        "account": meta.get("account", ""),
        "nodes": nodes,
        "ranks": ranks,
        "partition": meta.get("partition", ""),
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
        "maxrss_kb": "",
        "scratch": proto_dir,
    }


def collect_root(root: str, warmup: int) -> List[Dict[str, Any]]:
    runs_dir = os.path.join(root, "runs") if os.path.isdir(os.path.join(root, "runs")) else root
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
            if not os.path.isdir(proto_dir):
                continue
            row = collect_protocol(run, proto_dir, proto, warmup)
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


def _key_scale(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("family") or ""),
        str(row.get("reshape") or ""),
        int(row.get("nodes") or 0),
        int(row.get("ranks") or 0),
    )


def admitted_times(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[Any, ...], List[float]]:
    grouped: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("admit")) != "ok":
            continue
        wall = _parse_float(str(row.get("wall_step_s") or ""))
        if wall is None:
            continue
        grouped[_key_scale(row)].append(wall)
        # keep a copy of a representative row elsewhere
    return grouped


def analyze(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_scale: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("admit")) != "ok":
            continue
        if not row.get("wall_step_s"):
            continue
        by_scale[_key_scale(row)].append(row)

    medians: Dict[Tuple[str, str, int, int], float] = {}
    out: List[Dict[str, Any]] = []
    for key, group in sorted(by_scale.items()):
        family, reshape, nodes, ranks = key
        walls = [float(r["wall_step_s"]) for r in group]
        med = float(statistics.median(walls))
        medians[(str(family), str(reshape), int(nodes), int(ranks))] = med
        jobs = ",".join(sorted({str(r.get("job") or "") for r in group}))
        nx = group[0].get("Nx")
        ny = group[0].get("Ny")
        nz = group[0].get("Nz")
        out.append(
            {
                "family": family,
                "reshape": reshape,
                "nodes": nodes,
                "ranks": ranks,
                "Nx": nx,
                "Ny": ny,
                "Nz": nz,
                "n_repeats": len(walls),
                "wall_step_s_median": "%.8f" % med,
                "wall_step_s_min": "%.8f" % min(walls),
                "wall_step_s_max": "%.8f" % max(walls),
                "weak_eff": "",
                "rel_p2p_plined": "",
                "t_double": "",
                "penalty_per_doubling": "",
                "rank_at_scale": "",
                "jobs": jobs,
            }
        )

    # Use the smallest admitted rank count within each family/protocol.
    # A one-GCD reference and a full node are distinct scales.
    baseline: Dict[Tuple[str, str], Tuple[Tuple[int, int], float]] = {}
    for (family, reshape, nodes, ranks), med in medians.items():
        cur = baseline.get((family, reshape))
        if cur is None or (ranks, nodes) < cur[0]:
            baseline[(family, reshape)] = ((ranks, nodes), med)

    ranking_groups: Dict[Tuple[str, int, int], List[Tuple[str, float]]] = defaultdict(list)
    for row in out:
        family = str(row["family"])
        reshape = str(row["reshape"])
        nodes = int(row["nodes"])
        ranks = int(row["ranks"])
        med = float(row["wall_step_s_median"])
        base = baseline.get((family, reshape))
        if base and med > 0:
            row["weak_eff"] = "%.4f" % (base[1] / med)
        plined = medians.get((family, "p2p_plined", nodes, ranks))
        if plined and plined > 0:
            row["rel_p2p_plined"] = "%.4f" % (med / plined)
        half = medians.get((family, reshape, nodes // 2, ranks // 2))
        if half and half > 0 and nodes >= 2 and nodes % 2 == 0 and ranks % 2 == 0:
            ratio = med / half
            row["t_double"] = "%.4f" % ratio
            row["penalty_per_doubling"] = "%.4f" % (ratio - 1.0)
        ranking_groups[(family, nodes, ranks)].append((reshape, med))

    rank_map: Dict[Tuple[str, int, int, str], int] = {}
    for (family, nodes, ranks), items in ranking_groups.items():
        items.sort(key=lambda x: (x[1], x[0]))
        for i, (reshape, _) in enumerate(items, start=1):
            rank_map[(family, nodes, ranks, reshape)] = i
    for row in out:
        rnk = rank_map.get(
            (str(row["family"]), int(row["nodes"]), int(row["ranks"]), str(row["reshape"]))
        )
        if rnk is not None:
            row["rank_at_scale"] = rnk
    return out


def crossover_notes(scale_rows: Sequence[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    by_fam: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scale_rows:
        by_fam[str(row["family"])].append(row)
    for family, rows in sorted(by_fam.items()):
        scales = sorted({(int(r["ranks"]), int(r["nodes"])) for r in rows})
        winners = []
        for ranks, nodes in scales:
            at = [r for r in rows if int(r["nodes"]) == nodes
                  and int(r["ranks"]) == ranks]
            at.sort(key=lambda r: (float(r["wall_step_s_median"]), str(r["reshape"])))
            if at:
                winners.append((nodes, ranks, str(at[0]["reshape"])))
        unique = []
        for nodes, ranks, alg in winners:
            if not unique or unique[-1][2] != alg:
                unique.append((nodes, ranks, alg))
        algs = [a for _, _, a in unique]
        if len(set(algs)) == 1:
            notes.append(
                "family %s: no crossover; %s remains fastest from %d to %d ranks"
                % (family, algs[0], winners[0][1], winners[-1][1])
            )
        else:
            notes.append(
                "family %s: ranking changes: " % family
                + "; ".join("%d nodes / %d ranks -> %s" % (n, r, a)
                            for n, r, a in unique)
            )
    return notes


def write_markdown(path: str, scale_rows: Sequence[Dict[str, Any]], notes: Sequence[str]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write("# HeFFTe protocol tournament (issue #61)\n\n")
        f.write(
            "Medians of admitted repeats. Families are not mixed. Weak efficiency uses the smallest admitted rank count per family/protocol. "
            "Do not treat the fastest repeat as the result.\n\n"
        )
        families = sorted({str(r["family"]) for r in scale_rows})
        for family in families:
            f.write("## Local %s³ / GCD\n\n" % family)
            f.write(
                "| nodes | ranks | reshape | n | median s | min | max | "
                "weak eff | vs p2p_plined | T(2P)/T(P) | rank |\n"
            )
            f.write("|------:|------:|---------|--:|---------:|----:|----:|--------:|-------------:|----------:|-----:|\n")
            fam_rows = [r for r in scale_rows if str(r["family"]) == family]
            fam_rows.sort(
                key=lambda r: (int(r["nodes"]), int(r["ranks"]), str(r["reshape"]))
            )
            for r in fam_rows:
                f.write(
                    "| %s | %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s |\n"
                    % (
                        r["nodes"],
                        r["ranks"],
                        r["reshape"],
                        r["n_repeats"],
                        r["wall_step_s_median"],
                        r["wall_step_s_min"],
                        r["wall_step_s_max"],
                        r["weak_eff"],
                        r["rel_p2p_plined"],
                        r["t_double"],
                        r["rank_at_scale"],
                    )
                )
            f.write("\n")
        f.write("## Slope / ranking notes\n\n")
        if notes:
            for n in notes:
                f.write("- %s\n" % n)
        else:
            f.write("- no admitted points\n")
        f.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--ladder", type=int, choices=(512, 768))
    p.add_argument("--collect")
    p.add_argument("--analyze")
    p.add_argument("--out")
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args(argv)

    if args.check:
        return check()
    if args.ladder:
        return print_ladder(args.ladder)
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
        notes = crossover_notes(scale)
        write_markdown(os.path.join(args.out, "scaling.md"), scale, notes)
        print("wrote %s" % os.path.join(args.out, "scaling.csv"))
        for n in notes:
            print(n)
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

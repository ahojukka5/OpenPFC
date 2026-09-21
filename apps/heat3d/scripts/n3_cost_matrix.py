#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frozen Heat3D 1-node GPU cost matrix for research #592 / OpenPFC #124.

Eight cells (N in {512,1024} x spectral, FD-2, FD-8, FD-12) times three
independent repeats. Does not splice onto the historical b91d2575 table.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ACCOUNT = "project_462001519"
PARTITION = "standard-g"
NODES = 1
NTASKS = 8
STEPS = 30
WARMUP = 5
ACCEPTED = 25
DT = 0.01
SIZES = (512, 1024)
OPERATORS = (
    ("spectral", 0),
    ("fd", 2),
    ("fd", 8),
    ("fd", 12),
)
REPEATS = (1, 2, 3)
N_CELLS = len(SIZES) * len(OPERATORS) * len(REPEATS)


def cells():
    for repeat in REPEATS:
        for n in SIZES:
            for method, order in OPERATORS:
                yield {
                    "repeat": repeat,
                    "N": n,
                    "method": method,
                    "fd_order": order,
                    "tag": f"n{n}_{method}{order or ''}_r{repeat}".replace(
                        "spectral0", "spectral"
                    ),
                }


def cell_tag(n, method, order, repeat):
    if method == "spectral":
        return f"n{n}_spectral_r{repeat}"
    return f"n{n}_fd{order}_r{repeat}"


def _meta_map(path: Path) -> Dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("===") or " " in key:
            continue
        out[key] = val.strip()
    return out


def _median_wall_s(profile: Path) -> Tuple[float, int]:
    data = json.loads(profile.read_text())
    names = data["frame_metric_names"]
    wi = names.index("wall_step")
    frames = data["ranks"][0]["frames"]
    walls = [float(f["scalars"][wi]) for f in frames]
    return statistics.median(walls), len(walls)


def _grep_kv(text: str, key: str) -> Optional[str]:
    token = key + "="
    for raw in text.replace("\n", " ").split():
        if raw.startswith(token):
            return raw[len(token) :]
    return None


def harvest_cell(run_dir: Path) -> Dict:
    meta_path = run_dir / "run_meta.txt"
    profile = run_dir / "timing_profile.json"
    log = run_dir / "cell.log"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing {meta_path}")
    if not profile.is_file():
        raise FileNotFoundError(f"missing {profile}")
    meta = _meta_map(meta_path)
    wall_s, n_frames = _median_wall_s(profile)
    log_text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    stdout = log_text
    gpu_aware = _grep_kv(stdout, "gpu_aware") or meta.get("gpu_aware", "")
    use_pencils = _grep_kv(stdout, "use_pencils")
    if use_pencils is None:
        use_pencils = meta.get("use_pencils", "")
    reasons = []
    if meta.get("account") != ACCOUNT:
        reasons.append(f"account={meta.get('account')}")
    if meta.get("partition") != PARTITION:
        reasons.append(f"partition={meta.get('partition')}")
    if meta.get("nodes") not in ("1", str(NODES)):
        reasons.append(f"nodes={meta.get('nodes')}")
    if meta.get("ntasks") not in ("8", str(NTASKS)):
        reasons.append(f"ntasks={meta.get('ntasks')}")
    if meta.get("steps") not in (str(STEPS),):
        reasons.append(f"steps={meta.get('steps')}")
    if meta.get("warmup") not in (str(WARMUP),):
        reasons.append(f"warmup={meta.get('warmup')}")
    if n_frames != ACCEPTED:
        reasons.append(f"n_accepted={n_frames}")
    if meta.get("dirty") not in ("0", "0\n"):
        reasons.append(f"dirty={meta.get('dirty')}")
    if gpu_aware not in ("1", "true", "True"):
        reasons.append(f"gpu_aware={gpu_aware}")
    method = meta.get("method", "")
    if method == "spectral" and use_pencils not in ("0", "false", "False"):
        reasons.append(f"use_pencils={use_pencils}")
    if meta.get("dt") not in (str(DT), "0.01"):
        reasons.append(f"dt={meta.get('dt')}")
    row = {
        "tag": meta.get("tag", run_dir.name),
        "repeat": int(meta.get("repeat", "0") or 0),
        "N": int(meta.get("N", "0") or 0),
        "method": method,
        "fd_order": int(meta.get("fd_order", "0") or 0),
        "job": meta.get("job", ""),
        "wall_step_ms": 1000.0 * wall_s,
        "n_accepted": n_frames,
        "gpu_aware": gpu_aware,
        "use_pencils": use_pencils,
        "revision": meta.get("revision", ""),
        "dirty": meta.get("dirty", ""),
        "bin_sha256": meta.get("bin_sha256", ""),
        "account": meta.get("account", ""),
        "partition": meta.get("partition", ""),
        "nodes": meta.get("nodes", ""),
        "ntasks": meta.get("ntasks", ""),
        "run_dir": str(run_dir),
        "admitted": "yes" if not reasons else "no",
        "reject_reason": ";".join(reasons),
    }
    return row


def harvest_root(root: Path) -> List[Dict]:
    runs = root / "runs"
    rows = []
    if not runs.is_dir():
        return rows
    for path in sorted(runs.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "run_meta.txt").is_file():
            continue
        try:
            rows.append(harvest_cell(path))
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "tag": path.name,
                    "repeat": 0,
                    "N": 0,
                    "method": "",
                    "fd_order": 0,
                    "job": "",
                    "wall_step_ms": math.nan,
                    "n_accepted": 0,
                    "gpu_aware": "",
                    "use_pencils": "",
                    "revision": "",
                    "dirty": "",
                    "bin_sha256": "",
                    "account": "",
                    "partition": "",
                    "nodes": "",
                    "ntasks": "",
                    "run_dir": str(path),
                    "admitted": "no",
                    "reject_reason": str(exc),
                }
            )
    return rows


def summarize(rows: List[Dict]) -> List[Dict]:
    groups = {}
    for row in rows:
        if row.get("admitted") != "yes":
            continue
        key = (row["N"], row["method"], row["fd_order"])
        groups.setdefault(key, []).append(row)
    out = []
    for (n, method, order), items in sorted(groups.items()):
        walls = [r["wall_step_ms"] for r in items]
        med = statistics.median(walls)
        mean = statistics.mean(walls)
        cv = 0.0
        if len(walls) > 1 and mean > 0:
            cv = 100.0 * statistics.stdev(walls) / mean
        out.append(
            {
                "N": n,
                "method": method,
                "fd_order": order,
                "n_repeats": len(walls),
                "wall_step_ms_median": med,
                "wall_step_ms_min": min(walls),
                "wall_step_ms_max": max(walls),
                "cv_percent": cv,
                "jobs": ",".join(r["job"] for r in items),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def check() -> int:
    got = list(cells())
    if len(got) != N_CELLS:
        print(f"expected {N_CELLS} cells, got {len(got)}", file=sys.stderr)
        return 1
    tags = [cell_tag(c["N"], c["method"], c["fd_order"], c["repeat"]) for c in got]
    if len(set(tags)) != N_CELLS:
        print("duplicate cell tags", file=sys.stderr)
        return 1
    print(f"n3 cost matrix: {N_CELLS} cells, account={ACCOUNT} partition={PARTITION}")
    print(f"protocol: steps={STEPS} warmup={WARMUP} accepted={ACCEPTED} dt={DT}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--harvest", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    if args.check:
        return check()
    if args.harvest is None:
        p.error("pass --check or --harvest ROOT")
    rows = harvest_root(args.harvest)
    out_dir = args.out or (args.harvest / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "heat3d_n3_cost_repeats.csv",
        rows,
        [
            "tag",
            "repeat",
            "N",
            "method",
            "fd_order",
            "job",
            "wall_step_ms",
            "n_accepted",
            "gpu_aware",
            "use_pencils",
            "revision",
            "dirty",
            "bin_sha256",
            "account",
            "partition",
            "nodes",
            "ntasks",
            "run_dir",
            "admitted",
            "reject_reason",
        ],
    )
    summary = summarize(rows)
    write_csv(
        out_dir / "heat3d_n3_cost_summary.csv",
        summary,
        [
            "N",
            "method",
            "fd_order",
            "n_repeats",
            "wall_step_ms_median",
            "wall_step_ms_min",
            "wall_step_ms_max",
            "cv_percent",
            "jobs",
        ],
    )
    admitted = sum(1 for r in rows if r.get("admitted") == "yes")
    print(f"harvested {len(rows)} runs, admitted {admitted}, summary {len(summary)}")
    return 0 if admitted == N_CELLS else 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Issue #107 selector regret scaffold. Blocked until #106 decides features.

Do not fit coefficients. Do not time held-out families. Do not change
production reshape policy from this module.

    python3 apps/heat3d/scripts/heffte_selector_regret.py --schema
    python3 apps/heat3d/scripts/heffte_selector_regret.py --regret CSV
"""

from __future__ import print_function

import argparse
import csv
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

PROTOCOLS = ("p2p_plined", "p2p", "alltoallv", "alltoall")
TRAIN_FAMILIES = (512, 768)
# Candidate held-out edges. Freeze only after #106, before any timing.
CANDIDATE_HELDOUT_FAMILIES = (384, 640, 896)

FEATURE_SCHEMA = (
    "family",
    "nodes",
    "ranks",
    "real_grid",
    "complex_grid",
    "inbox_xyz",
    "outbox_xyz",
    "n_mpi_reshapes",
    "comm_group_size",
    "n_peers",
    "bytes_per_rank_per_reshape",
    "bytes_per_timestep",
    "bytes_per_peer",
    "n_onnode_peers",
    "n_offnode_peers",
    "topology_feature",
)


def regret(t_selected: float, t_best: float) -> float:
    if t_best <= 0:
        raise ValueError("T_best must be positive")
    return t_selected / t_best - 1.0


def case_metrics(rows: Sequence[Dict[str, Any]], selected: str) -> Dict[str, Any]:
    walls = {}
    for row in rows:
        proto = str(row.get("reshape") or "")
        wall = row.get("wall_step_s")
        if proto in PROTOCOLS and wall not in ("", None):
            walls[proto] = float(wall)
    if selected not in walls:
        raise ValueError("selected protocol %s missing a wall time" % selected)
    if len(walls) < 2:
        raise ValueError("need at least two protocol times")
    best_proto = min(walls, key=lambda p: (walls[p], p))
    t_best = walls[best_proto]
    t_sel = walls[selected]
    return {
        "selected": selected,
        "best": best_proto,
        "hit": selected == best_proto,
        "regret": regret(t_sel, t_best),
        "t_selected": t_sel,
        "t_best": t_best,
        "n_protocols": len(walls),
    }


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    regrets = [float(c["regret"]) for c in cases]
    hits = [1 if c["hit"] else 0 for c in cases]
    return {
        "n_cases": len(cases),
        "median_regret": float(statistics.median(regrets)) if regrets else "",
        "worst_regret": float(max(regrets)) if regrets else "",
        "hit_rate": float(sum(hits) / len(hits)) if hits else "",
    }


def assign_split(family: int) -> str:
    if family in TRAIN_FAMILIES:
        return "train"
    if family in CANDIDATE_HELDOUT_FAMILIES:
        return "heldout_candidate"
    return "unknown"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schema", action="store_true")
    p.add_argument("--regret")
    p.add_argument("--selected", default="p2p_plined")
    args = p.parse_args(argv)
    if args.schema:
        print("split: train families %s" % ",".join(str(x) for x in TRAIN_FAMILIES))
        print(
            "held-out candidates (not frozen): %s"
            % ",".join(str(x) for x in CANDIDATE_HELDOUT_FAMILIES)
        )
        print("features: %s" % ",".join(FEATURE_SCHEMA))
        print("blocked: do not fit or time held-out cases until #106 reports")
        return 0
    if args.regret:
        with open(args.regret, newline="") as f:
            rows = list(csv.DictReader(f))
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            key = "%s/%s" % (row.get("family"), row.get("nodes"))
            grouped.setdefault(key, []).append(row)
        cases = []
        for key, group in sorted(grouped.items()):
            try:
                m = case_metrics(group, args.selected)
            except ValueError:
                continue
            m["case"] = key
            cases.append(m)
            print(
                "%s selected=%s best=%s regret=%.4f"
                % (key, m["selected"], m["best"], m["regret"])
            )
        s = summarize(cases)
        print(
            "n=%s median_regret=%s worst=%s hit_rate=%s"
            % (s["n_cases"], s["median_regret"], s["worst_regret"], s["hit_rate"])
        )
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

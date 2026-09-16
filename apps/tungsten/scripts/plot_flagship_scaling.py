#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot issue #13 weak-scaling efficiency from the compact CSV.

Does not interpolate missing node counts. If the CSV has no data rows,
exits 0 after printing that production points are not admitted yet.

    python3 apps/tungsten/scripts/plot_flagship_scaling.py \\
        --csv docs/report/data/tungsten_lumi_g_flagship.csv
"""

import argparse
import csv
from pathlib import Path


def load_rows(path):
    with path.open() as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    if len(lines) < 2:
        return []
    return list(csv.DictReader(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("docs/report/data/tungsten_lumi_g_flagship.csv"),
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    rows = load_rows(args.csv)
    if not rows:
        print("%s: no production rows; not plotting interpolated points" % args.csv)
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; CSV has %d rows" % len(rows))
        return
    nodes = [int(r["nodes"]) for r in rows]
    t1 = None
    for r in rows:
        if int(r["nodes"]) == 1:
            t1 = float(r["wall_step_s"])
            break
    if t1 is None or t1 <= 0.0:
        print("need a 1-node wall_step_s to compute efficiency")
        return
    eff = []
    for r in rows:
        t = float(r["wall_step_s"])
        # Weak scaling, constant cells/GCD: ideal is T(N)=T(1).
        eff.append(t1 / t)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(nodes, eff, "o-", color="#1f4e79")
    ax.set_xlabel("nodes")
    ax.set_ylabel("weak-scaling efficiency vs 1 node")
    ax.set_title("tungsten_hip LUMI-G flagship (issue #13)")
    ax.set_ylim(0.0, 1.15)
    ax.grid(True, alpha=0.3)
    out = args.out or (args.csv.parent / "tungsten_lumi_g_flagship.svg")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

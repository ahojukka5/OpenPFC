#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot J, grey, C11/C12 from an inverse3d history.csv (OpenPFC #9)."""
import csv
import sys
from pathlib import Path

def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    def col(name):
        return [float(r[name]) for r in rows if r.get(name) not in (None, "")]
    return rows, col

def main():
    if len(sys.argv) < 3:
        print("usage: plot_inverse3d_history.py history.csv out.png", file=sys.stderr)
        return 2
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    rows, col = load(src)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; wrote nothing", file=sys.stderr)
        return 1
    step = col("step")
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7.2, 8.0))
    ax[0].plot(step, col("J"), label="J")
    ax[0].plot(step, col("J_tensor"), label="J_tensor")
    ax[0].set_ylabel("objective")
    ax[0].legend()
    ax[1].plot(step, col("grey"), label="grey")
    ax[1].plot(step, col("volume"), label="volume")
    ax[1].set_ylabel("grey / vf")
    ax[1].legend()
    ax[2].plot(step, col("C11"), label="C11")
    ax[2].plot(step, col("C12"), label="C12")
    ax[2].set_xlabel("step")
    ax[2].set_ylabel("C")
    ax[2].legend()
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150)
    print("wrote", dest)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

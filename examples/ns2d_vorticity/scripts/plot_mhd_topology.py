#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""j + a-contours + tracked X/O overlay, plus Ez vs eta*j time series."""

from __future__ import print_function

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_mhd_fields import load_brick
import mhd_topology as mt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--inc", type=int, required=True)
    p.add_argument("--json", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("matplotlib required:", exc, file=sys.stderr)
        return 1
    n = args.n
    dx = 2.0 * math.pi / n
    a = load_brick(os.path.join(args.dir, "a_%04d.bin" % args.inc), n).T
    j = load_brick(os.path.join(args.dir, "j_%04d.bin" % args.inc), n).T
    w = load_brick(os.path.join(args.dir, "omega_%04d.bin" % args.inc), n).T
    pts = mt.locate_critical_points(
        load_brick(os.path.join(args.dir, "a_%04d.bin" % args.inc), n),
        load_brick(os.path.join(args.dir, "j_%04d.bin" % args.inc), n),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    jlim = np.max(np.abs(j))
    im = axes[0, 0].imshow(
        j, origin="lower", extent=(0, 2 * math.pi, 0, 2 * math.pi),
        cmap="RdBu_r", vmin=-jlim, vmax=jlim, interpolation="nearest")
    xg = (np.arange(n) + 0.5) * dx
    axes[0, 0].contour(xg, xg, a, levels=12, colors="k", linewidths=0.4)
    for pt in pts:
        mkr = {"X": "x", "O_max": "o", "O_min": "s", "degenerate": "+"}.get(pt["kind"], ".")
        axes[0, 0].plot(pt["x"], pt["y"], mkr, color="lime", markersize=7,
                        markeredgewidth=1.5)
    axes[0, 0].set_title("j, a-contours, X/O")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)
    wlim = np.max(np.abs(w))
    imw = axes[0, 1].imshow(
        w, origin="lower", extent=(0, 2 * math.pi, 0, 2 * math.pi),
        cmap="RdBu_r", vmin=-wlim, vmax=wlim, interpolation="nearest")
    axes[0, 1].set_title(r"$\omega$")
    fig.colorbar(imw, ax=axes[0, 1], fraction=0.046)

    if args.json and os.path.exists(args.json):
        with open(args.json) as fh:
            rep = json.load(fh)
        ez = rep.get("ez", [])
        if ez:
            t = [e["t"] for e in ez]
            axes[1, 0].plot(t, [e["Ez_X"] for e in ez], label=r"$E_z=-\dot a_X$")
            axes[1, 0].plot(t, [e["eta_j_X"] for e in ez], "--",
                            label=r"$\eta j_X$")
            axes[1, 0].legend(fontsize=8)
            axes[1, 0].set_xlabel("t")
            axes[1, 0].set_title("longest X-track electric field")
        ser = rep.get("series", [])
        if ser:
            axes[1, 1].plot([r["t"] for r in ser], [r["nX"] for r in ser],
                            label="nX")
            axes[1, 1].plot([r["t"] for r in ser], [r["nO"] for r in ser],
                            label="nO")
            axes[1, 1].plot([r["t"] for r in ser], [r["n_degen"] for r in ser],
                            label="degenerate")
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].set_xlabel("t")
            axes[1, 1].set_title("critical-point counts")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

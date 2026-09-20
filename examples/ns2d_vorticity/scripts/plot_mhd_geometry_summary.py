#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Geometry summary of admitted frozen-window means vs eta.

The figure is the mechanism statement: delta, L, L/delta, B_up and
j_X change little, so R ~ S_local^{-1} follows from Ez_X = eta j_X.

Example:
  python3 plot_mhd_geometry_summary.py \\
    --out figures/evidence/geometry_summary.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mhd_conference_catalog as cat
import mhd_conference_style as st


def _mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit("matplotlib is required: %s" % exc)
    return plt, np


PANELS = (
    ("delta", r"$\delta$", 0.0, 0.80),
    ("L", r"$L$", 0.0, 6.5),
    ("L_over_delta", r"$L/\delta$", 0.0, 12.0),
    ("B_up", r"$B_{\mathrm{up}}$", 0.0, 0.30),
    ("j_X", r"$j_X$", 0.0, 4.0),
    ("RS", r"$R\,S_{\mathrm{local}}$", 0.0, 55.0),
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    plt, np = _mpl()
    st.apply_paper_style(plt)
    means = sorted(
        cat.admitted_means_from_json(cat.data_root(args.data_root)),
        key=lambda m: m["eta"],
    )
    xs = list(range(len(means)))
    labels = [r"$%g$" % m["eta"] for m in means]

    fig, axes = plt.subplots(2, 3, figsize=st.FIG_GEOMETRY, sharex=True)
    for ax, (key, ylab, y0, y1) in zip(axes.ravel(), PANELS):
        y = [m[key] for m in means]
        ax.plot(xs, y, "o-", color=st.PAPER_EZ, ms=7, lw=1.4)
        ax.axhline(sum(y) / float(len(y)), color=st.PAPER_REF, ls=":", lw=1.0)
        ax.set_ylim(y0, y1)
        ax.set_ylabel(ylab)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    for ax in axes[1]:
        ax.set_xlabel(r"$\eta=\nu$")
    for ax in axes[1]:
        ax.set_xlabel(r"$\eta=\nu$")
    fig.suptitle(
        r"Orszag–Tang  $P_m=1$  admitted means  $t\in[0.314,0.70]$  "
        + cat.WINDOW_SCALING,
        fontsize=st.FONT_TITLE,
    )
    fig.text(
        0.5, 0.01,
        r"Geometry is nearly $\eta$-independent, so $E_{z,X}=\eta j_X$ "
        r"gives $R\sim S_{\mathrm{local}}^{-1}$. Not a Sweet–Parker layer.",
        ha="center", va="bottom", fontsize=st.FONT_ANNOT,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=st.DPI_STILL)
    plt.close(fig)
    cat.write_json(args.out + ".json", {
        "kind": "scaling",
        "window": cat.WINDOW_SCALING,
        "points": means,
        "output": os.path.abspath(args.out),
    })
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

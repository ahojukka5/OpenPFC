#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Log-log R vs S_local for the three admitted eta values.

Reference guides S^{-1/2} (Sweet–Parker) and S^{-1} are drawn through
the eta=0.005 point. They are not fits. Any OLS slope annotation labels
the ± as formal OLS SE only.

Example:
  python3 plot_mhd_scaling_summary.py \\
    --out figures/evidence/scaling_R_vs_S.png
"""

from __future__ import annotations

import argparse
import math
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    plt, np = _mpl()
    st.apply_paper_style(plt)
    means = cat.admitted_means_from_json(cat.data_root(args.data_root))
    S = np.array([m["S_local"] for m in means], dtype=float)
    R = np.array([m["R"] for m in means], dtype=float)
    slope, intercept, se = cat.loglog_ols(S, R)

    fig, ax = plt.subplots(figsize=st.FIG_SCALING)
    Sref = np.logspace(math.log10(S.min() / 1.25),
                       math.log10(S.max() * 1.25), 80)
    mid = means[1]
    c_sp = mid["R"] / mid["S_local"] ** (-0.5)
    c_inv = mid["R"] / mid["S_local"] ** (-1.0)
    ax.loglog(Sref, c_sp * Sref ** (-0.5), color=st.PAPER_SP, lw=1.2,
              ls="--", label=r"$S^{-1/2}$ reference (through $\eta=0.005$)")
    ax.loglog(Sref, c_inv * Sref ** (-1.0), color=st.PAPER_INV, lw=1.2,
              ls="-.", label=r"$S^{-1}$ reference (through $\eta=0.005$)")
    ax.loglog(S, R, "o", color=st.PAPER_EZ, ms=8, zorder=5,
              label="admitted window means")
    for m in means:
        ax.annotate(
            r"$\eta=%g$" % m["eta"],
            (m["S_local"], m["R"]),
            textcoords="offset points", xytext=(6, -10),
            fontsize=st.FONT_ANNOT, color=st.PAPER_TEXT,
        )
    ax.set_xlabel(r"$S_{\mathrm{local}}=L B_{\mathrm{up}}/\eta$")
    ax.set_ylabel(r"$R=|E_{z,X}|/B_{\mathrm{up}}^2$")
    ax.set_title(
        r"Orszag–Tang  $P_m=1$  %s  $t\in[0.314,0.70]$"
        % cat.WINDOW_SCALING,
        loc="left",
    )
    ax.legend(loc="upper right", fontsize=st.FONT_TICK)
    ax.text(
        0.03, 0.03,
        "formal OLS slope $%.2f\\pm%.3f$ (SE only)\n"
        "not a physical uncertainty; three points, not a law\n"
        r"Sweet–Parker $S^{-1/2}$ is not supported here"
        % (slope, se),
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=st.FONT_ANNOT,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f7f4ee", ec="#ddd6c6"),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=st.DPI_STILL)
    plt.close(fig)
    cat.write_json(args.out + ".json", {
        "kind": "scaling",
        "window": cat.WINDOW_SCALING,
        "t_lo": cat.T_SCALE_LO,
        "t_hi": cat.T_SCALE_HI,
        "points": means,
        "ols_slope": slope,
        "ols_intercept": intercept,
        "ols_se": se,
        "ols_se_meaning": "formal OLS standard error (n-2 dof); not physical",
        "output": os.path.abspath(args.out),
    })
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

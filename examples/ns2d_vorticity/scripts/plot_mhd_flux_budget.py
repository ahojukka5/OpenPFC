#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Publication flux-budget figure for accepted OT reconnection.

Plots Ez_X, eta j_X, dF/dt, and Ez_X-Ez_O on t in [0, 0.80] for the
admitted eta=0.005, N=512 run. Shades the frozen scaling window.

Example:
  python3 plot_mhd_flux_budget.py \\
    --out figures/evidence/flux_budget_eta0005.png
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
    except ImportError as exc:
        raise SystemExit("matplotlib is required: %s" % exc)
    return plt


def load_series(root):
    run = cat.case_dir(cat.FLUX_ID, root)
    rows = cat.window_rows(
        cat.load_tracked_rows(run), cat.T_RECONN_LO, cat.T_RECONN_HI)
    if len(rows) < 3:
        raise SystemExit("need island/sheet JSON under %s" % run)
    t = [float(r["t"]) for r in rows]
    ez = [float(r["Ez_X"]) for r in rows]
    etaj = [float(r["eta_j_X"]) for r in rows]
    dfdt = [float(r["dFdt"]) for r in rows]
    ezo = [float(r["Ez_X"]) - float(r["Ez_O"]) for r in rows]
    return t, ez, etaj, dfdt, ezo, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    plt = _mpl()
    st.apply_paper_style(plt)
    root = cat.data_root(args.data_root)
    t, ez, etaj, dfdt, ezo, rows = load_series(root)
    spec = cat.case_spec(cat.FLUX_ID)

    fig, ax = plt.subplots(figsize=st.FIG_FLUX)
    ax.axvspan(cat.T_SCALE_LO, cat.T_SCALE_HI, color="#f3ead2", lw=0,
               label="frozen scaling window")
    ax.plot(t, ez, color=st.PAPER_EZ, lw=2.0, label=r"$E_{z,X}$")
    ax.plot(t, etaj, color=st.PAPER_ETAJ, lw=1.5, ls="--",
            label=r"$\eta j_X$")
    ax.plot(t, dfdt, color=st.PAPER_DFDT, lw=2.0, label=r"$dF/dt$")
    ax.plot(t, ezo, color=st.PAPER_EZO, ls="none", marker="o", ms=4,
            markerfacecolor="none", markeredgewidth=1.1,
            label=r"$E_{z,X}-E_{z,O}$")
    ax.set_xlim(0.0, cat.T_RECONN_HI)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("electric field / flux derivative")
    ax.legend(loc="upper left", fontsize=st.FONT_TICK, ncol=2)
    mean_ez = sum(ez) / len(ez)
    mean_df = sum(dfdt) / len(dfdt)
    ax.set_title(
        r"Orszag–Tang  $\eta=\nu=%g$  $N=%d$  %s"
        % (spec["eta"], spec["N"], cat.WINDOW_RECONNECTION),
        loc="left",
    )
    ax.text(
        0.98, 0.05,
        r"$\langle E_{z,X}\rangle/\langle\dot F\rangle=%.2f$"
        "\nOhm check $E_{z,X}\\approx\\eta j_X$"
        "\nX-point share of $\\dot F$, not a plasmoid claim"
        % (mean_ez / mean_df),
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=st.FONT_ANNOT, color=st.PAPER_TEXT,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f7f4ee", ec="#ddd6c6"),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=st.DPI_STILL)
    plt.close(fig)
    cat.write_json(args.out + ".json", cat.provenance(
        cat.FLUX_ID, t=float(t[-1]), kind="evidence", extra={
            "t_first": t[0],
            "t_last": t[-1],
            "n": len(t),
            "mean_Ez_X": mean_ez,
            "mean_dFdt": mean_df,
            "run_dir": cat.case_dir(cat.FLUX_ID, root),
            "output": os.path.abspath(args.out),
        }))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

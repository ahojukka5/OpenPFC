#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot matched MHD vs hydro diagnostics from mhd2d CSV files."""

from __future__ import annotations

import argparse
import csv
import os
import sys


def load_csv(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty csv: {path}")
    out = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            try:
                out[k].append(float(v))
            except (TypeError, ValueError):
                out[k].append(v)
    return out


def cfl_key(data: dict) -> str:
    if "cfl_elsasser_sum" in data:
        return "cfl_elsasser_sum"
    if "cfl_elsasser" in data:
        return "cfl_elsasser"
    return "cfl"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mhd", required=True, help="MHD diagnostics.csv")
    p.add_argument("--hydro", required=True, help="hydro-control diagnostics.csv")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("matplotlib is required:", exc, file=sys.stderr)
        return 1

    mhd = load_csv(args.mhd)
    hydro = load_csv(args.hydro)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(mhd["time"], mhd["ke"], label="KE MHD")
    ax.plot(hydro["time"], hydro["ke"], label="KE hydro")
    ax.plot(mhd["time"], mhd["me"], label="ME MHD")
    ax.plot(mhd["time"], mhd["energy"], label="E MHD")
    ax.plot(hydro["time"], hydro["energy"], label="E hydro", linestyle="--")
    ax.set_xlabel("$t$")
    ax.set_ylabel("energy")
    ax.legend(fontsize=8)
    ax.set_title("energy partition")

    ax = axes[0, 1]
    ax.plot(mhd["time"], mhd["max_abs_j"], label=r"max$|j|$ MHD")
    ax.plot(hydro["time"], hydro["max_abs_j"], label=r"max$|j|$ hydro")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"max$|j|$")
    ax.legend(fontsize=8)
    ax.set_title("current")

    ax = axes[1, 0]
    ax.plot(mhd["time"], mhd["max_abs_omega"], label=r"max$|\omega|$ MHD")
    ax.plot(hydro["time"], hydro["max_abs_omega"], label=r"max$|\omega|$ hydro")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"max$|\omega|$")
    ax.legend(fontsize=8)
    ax.set_title("vorticity")

    ax = axes[1, 1]
    key = cfl_key(mhd)
    ax.plot(mhd["time"], mhd[key], label=key.replace("_", " "))
    if "cfl_nominal" in mhd:
        ax.axhline(mhd["cfl_nominal"][0], color="k", linestyle=":", label="nominal")
    ax.set_xlabel("$t$")
    ax.set_ylabel("CFL")
    ax.legend(fontsize=8)
    ax.set_title("timestep diagnostic")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

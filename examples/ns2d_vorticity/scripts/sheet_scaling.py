#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local sheet geometry + normalized rate on the #26 persistent-X island.

Proposed scalar R: time-average of R on t in [T_AVG_LO, T_AVG_HI] =
[0.10, 0.70] over frames with sheet_ok (uncapped FWHM/length and
B_up > B_UP_MIN). Peak R is secondary. If no common sheet_ok interval
exists, do not report a single scaling point.
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_mhd_reconnection as an
import island_flux_budget as ifb
import mhd_sheet_geometry as sg
import mhd_topology as mt

T_AVG_LO = 0.10
T_AVG_HI = 0.70


def attach_geometry(directory, n, eta, flux_rep):
    if "error" in flux_rep:
        return flux_rep
    table = an.load_csv(os.path.join(directory, "diagnostics.csv"))
    inc_of_t = {}
    for inc, row in table.items():
        inc_of_t[round(float(row["time"]), 6)] = inc
    for r in flux_rep["rows"]:
        inc = inc_of_t.get(round(r["t"], 6))
        if inc is None:
            # nearest
            inc = min(table, key=lambda i: abs(float(table[i]["time"]) - r["t"]))
        a = an.load_brick(os.path.join(directory, "a_%04d.bin" % inc), n)
        geo = sg.measure_sheet(a, r["x_X"], r["y_X"], eta, ez_x=r["Ez_X"])
        r.update({
            "delta": geo["delta"],
            "L": geo["L"],
            "aspect": geo["aspect"],
            "B_up": geo["B_up"],
            "S_local": geo["S_local"],
            "R": geo["R"],
            "n_hat": geo["n_hat"],
            "t_hat": geo["t_hat"],
            "delta_capped": geo["delta_capped"],
            "L_capped": geo["L_capped"],
            "sheet_ok": geo["sheet_ok"],
        })
    interior = [r for r in flux_rep["rows"]
                if T_AVG_LO - 1e-12 <= r["t"] <= T_AVG_HI + 1e-12]
    ok = [r for r in interior if r.get("sheet_ok")]
    Rs = [r["R"] for r in ok if r.get("R") is not None]
    Ss = [r["S_local"] for r in ok if r.get("S_local") is not None]
    flux_rep["scaling"] = {
        "T_AVG_LO": T_AVG_LO,
        "T_AVG_HI": T_AVG_HI,
        "n_avg": len(interior),
        "n_capped": int(sum(1 for r in interior
                            if r.get("delta_capped") or r.get("L_capped"))),
        "n_sheet_ok": len(ok),
        "common_thin_sheet_interval": bool(ok) and len(ok) == len(interior),
        "R_mean": float(np.mean(Rs)) if Rs else None,
        "R_std": float(np.std(Rs)) if Rs else None,
        "S_local_mean": float(np.mean(Ss)) if Ss else None,
        "delta_mean": float(np.mean([r["delta"] for r in ok])) if ok else None,
        "L_mean": float(np.mean([r["L"] for r in ok])) if ok else None,
        "B_up_mean": float(np.mean([r["B_up"] for r in ok])) if ok else None,
        "aspect_mean": (float(np.mean([r["aspect"] for r in ok
                                       if r["aspect"] is not None]))
                        if ok else None),
    }
    return flux_rep


def print_geo(rep, tag=""):
    ifb.print_report(rep, tag=tag)
    if "error" in rep or "scaling" not in rep:
        return
    s = rep["scaling"]
    print("  sheet_ok %d/%d frames in [%.2f, %.2f]; common_interval=%s" % (
        s["n_sheet_ok"], s["n_avg"], s["T_AVG_LO"], s["T_AVG_HI"],
        s["common_thin_sheet_interval"]))
    if s["n_sheet_ok"]:
        print("  geometry means on sheet_ok: delta=%.4g L=%.4g L/delta=%.3g "
              "B_up=%.4g S_local=%.4g R_mean=%.4e ± %.1e" % (
                  s["delta_mean"], s["L_mean"],
                  s["aspect_mean"], s["B_up_mean"], s["S_local_mean"],
                  s["R_mean"] or 0.0, s["R_std"] or 0.0))
    else:
        print("  no sheet_ok frames: do not extract a scalar R")
    rows = [r for r in rep["rows"] if r.get("R") is not None]
    for r in rows[:: max(1, len(rows) // 8)]:
        print("  t=%.4f delta=%.4g L=%.4g B_up=%.4g S=%.4g R=%.4e ok=%s" % (
            r["t"], r["delta"], r["L"], r["B_up"], r["S_local"], r["R"],
            r.get("sheet_ok")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--eta", type=float, required=True)
    p.add_argument("--t-max", type=float, default=0.80)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--json-out", default="")
    args = p.parse_args()
    flux = ifb.analyze_run(args.dir, args.n, args.eta, args.t_max, args.stride)
    rep = attach_geometry(args.dir, args.n, args.eta, flux)
    print_geo(rep)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rep, fh, indent=2, default=str)
            fh.write("\n")
        print("wrote", args.json_out)
    return 0 if "error" not in rep else 1


if __name__ == "__main__":
    sys.exit(main())

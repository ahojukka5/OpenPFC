#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local sheet geometry + normalized rate on the #26 persistent-X island.

First-stage window [T_AVG_LO, T_AVG_HI] = [0.10, 0.70] failed as a
common sheet_ok interval (early L-cap). It is kept as a
preregistered negative and is not retuned.

Second-stage window, frozen before eta=0.0025:
  T_SCALE_LO = 0.314
  T_SCALE_HI = 0.70
Do not move these endpoints after seeing a low-eta result.
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
# Prospective second-stage window. Frozen before eta=0.0025.
T_SCALE_LO = 0.314
T_SCALE_HI = 0.70


def _in_window(t, lo, hi):
    return lo - 1.0e-12 <= t <= hi + 1.0e-12


def _mean_std_minmax(vals):
    if not vals:
        return None, None, None, None
    a = np.asarray(vals, dtype=float)
    return (float(np.mean(a)), float(np.std(a)),
            float(np.min(a)), float(np.max(a)))


def _window_report(rows, lo, hi):
    interior = [r for r in rows if _in_window(r["t"], lo, hi)]
    ok = [r for r in interior if r.get("sheet_ok")]
    common = bool(interior) and len(ok) == len(interior)

    def col(key, src=None):
        src = interior if src is None else src
        return [r[key] for r in src if r.get(key) is not None]

    def pack(key, src=None):
        m, s, mn, mx = _mean_std_minmax(col(key, src))
        return {"mean": m, "std": s, "min": mn, "max": mx}

    # Scaling averages use every dump in the window only if it is
    # common sheet_ok; otherwise means are reported on sheet_ok dumps
    # and common_thin_sheet_interval is false.
    src = interior if common else ok
    return {
        "lo": lo,
        "hi": hi,
        "n_avg": len(interior),
        "n_capped": int(sum(1 for r in interior
                            if r.get("delta_capped") or r.get("L_capped"))),
        "n_sheet_ok": len(ok),
        "common_thin_sheet_interval": common,
        "Ez_X": pack("Ez_X", src),
        "eta_j_X": pack("eta_j_X", src),
        "dFdt": pack("dFdt", src),
        "delta": pack("delta", src),
        "L": pack("L", src),
        "aspect": pack("aspect", src),
        "B_up": pack("B_up", src),
        "S_local": pack("S_local", src),
        "R": pack("R", src),
        "R_mean": pack("R", src)["mean"],
        "R_std": pack("R", src)["std"],
        "S_local_mean": pack("S_local", src)["mean"],
        "delta_mean": pack("delta", src)["mean"],
        "L_mean": pack("L", src)["mean"],
        "B_up_mean": pack("B_up", src)["mean"],
        "aspect_mean": pack("aspect", src)["mean"],
    }


def attach_geometry(directory, n, eta, flux_rep, t_lo=None, t_hi=None):
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
            "orientation_ok": geo.get("orientation_ok", False),
        })
    scale_lo = T_SCALE_LO if t_lo is None else t_lo
    scale_hi = T_SCALE_HI if t_hi is None else t_hi
    flux_rep["scaling_stage1"] = _window_report(
        flux_rep["rows"], T_AVG_LO, T_AVG_HI)
    flux_rep["scaling"] = _window_report(
        flux_rep["rows"], scale_lo, scale_hi)
    flux_rep["scaling"]["T_AVG_LO"] = scale_lo
    flux_rep["scaling"]["T_AVG_HI"] = scale_hi
    flux_rep["scaling"]["T_SCALE_LO"] = scale_lo
    flux_rep["scaling"]["T_SCALE_HI"] = scale_hi
    flux_rep["scaling_stage1"]["T_AVG_LO"] = T_AVG_LO
    flux_rep["scaling_stage1"]["T_AVG_HI"] = T_AVG_HI
    return flux_rep


def print_geo(rep, tag=""):
    ifb.print_report(rep, tag=tag)
    if "error" in rep or "scaling" not in rep:
        return
    s1 = rep.get("scaling_stage1") or {}
    s = rep["scaling"]
    ot_window = (
        abs(float(s.get("T_SCALE_LO", T_SCALE_LO)) - T_SCALE_LO) < 1.0e-12
        and abs(float(s.get("T_SCALE_HI", T_SCALE_HI)) - T_SCALE_HI) < 1.0e-12)
    if s1 and ot_window:
        print("  stage1 [%.3f, %.2f]: sheet_ok %d/%d common=%s (failed first "
              "preregistration; not a scaling window)" % (
                  T_AVG_LO, T_AVG_HI, s1.get("n_sheet_ok", 0),
                  s1.get("n_avg", 0), s1.get("common_thin_sheet_interval")))
    print("  window [%.3f, %.3f]: sheet_ok %d/%d common=%s" % (
        s.get("T_SCALE_LO", T_SCALE_LO), s.get("T_SCALE_HI", T_SCALE_HI),
        s["n_sheet_ok"], s["n_avg"],
        s["common_thin_sheet_interval"]))
    if s["n_sheet_ok"]:
        print("  means: Ez_X=%.4g  eta*j_X=%.4g  dF/dt=%.4g  delta=%.4g  "
              "L=%.4g  L/delta=%.3g  B_up=%.4g  S=%.4g  R=%.4e ± %.1e" % (
                  (s["Ez_X"]["mean"] or 0.0), (s["eta_j_X"]["mean"] or 0.0),
                  (s["dFdt"]["mean"] or 0.0), s["delta_mean"], s["L_mean"],
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
    p.add_argument("--family", default="ot",
                   choices=("ot", "coalescence"))
    p.add_argument("--t-lo", type=float, default=None,
                   help="override the frozen scaling-window start")
    p.add_argument("--t-hi", type=float, default=None,
                   help="override the frozen scaling-window end")
    p.add_argument("--json-out", default="")
    p.add_argument("--degen-ratio", type=float, default=None,
                   help="X/O eigenvalue-ratio classification floor "
                        "(default 0.05). Lowering it extends tracking into "
                        "flatter saddles; a sensitivity study, not a new "
                        "default. Check the Ohm residual when you do.")
    args = p.parse_args()
    flux = ifb.analyze_run(args.dir, args.n, args.eta, args.t_max, args.stride,
                           family=args.family, degen_ratio=args.degen_ratio)
    rep = attach_geometry(args.dir, args.n, args.eta, flux,
                          t_lo=args.t_lo, t_hi=args.t_hi)
    print_geo(rep)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rep, fh, indent=2, default=str)
            fh.write("\n")
        print("wrote", args.json_out)
    return 0 if "error" not in rep else 1


if __name__ == "__main__":
    sys.exit(main())

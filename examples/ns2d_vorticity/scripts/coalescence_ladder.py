#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Emit the island-coalescence resistivity ladder as one compact CSV.

Reads the per-run flux/sheet JSON produced by island_flux_budget.py and
sheet_scaling.py and reduces each run to its window means. Produces no new
numerical result: every value is a mean over dumps already in those files.

The comparison window is an argument, not a constant, because the runs do
not all reach the same progress. The Stage-0 bracket is [0.30, 0.60]; the
three-resistivity ladder shares only [0.30, 0.38] because tracking
truncates earlier at lower eta.

Dumps whose Ohm residual |Ez_X - eta j_X| / |Ez_X| exceeds --ohm-max are
dropped, and the count is reported. That gate is an accuracy criterion on
the rate, unlike the X/O classification floor, which is geometric.
"""
import argparse
import csv
import json
import os
import sys


def window_rows(run_dir, sheet, flux, p_lo, p_hi, ohm_max):
    with open(os.path.join(run_dir, flux)) as fh:
        f = json.load(fh)
    with open(os.path.join(run_dir, sheet)) as fh:
        s = json.load(fh)
    f0 = f["rows"][0]["F"]
    pmap = {round(r["t"], 6): (f0 - r["F"]) / abs(f0) for r in f["rows"]}
    rows, dropped = [], 0
    for r in s["rows"]:
        key = round(r["t"], 6)
        if key not in pmap or not r.get("sheet_ok", True):
            continue
        p = pmap[key]
        if not (p_lo <= p <= p_hi):
            continue
        ez = r["Ez_X"]
        if ez and abs(ez - r["eta_j_X"]) / abs(ez) > ohm_max:
            dropped += 1
            continue
        rows.append(dict(r, p=p))
    return rows, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", action="append", required=True,
                    metavar="ETA:DIR:N:SHEET:FLUX",
                    help="repeatable, e.g. 0.005:ic256_e0005_cfl04:256:"
                         "sheet_degen001.json:flux_degen001.json")
    ap.add_argument("--p-lo", type=float, default=0.30)
    ap.add_argument("--p-hi", type=float, default=0.38)
    ap.add_argument("--ohm-max", type=float, default=1.0e-3)
    ap.add_argument("--csv-out", default="")
    a = ap.parse_args()

    out = []
    for spec in a.run:
        eta, d, n, sheet, flux = spec.split(":")
        rows, dropped = window_rows(os.path.join(a.root, d), sheet, flux,
                                    a.p_lo, a.p_hi, a.ohm_max)
        if not rows:
            print("no dumps in window for %s" % d, file=sys.stderr)
            return 1
        m = lambda k: sum(r[k] for r in rows) / len(rows)
        rec = {
            "eta": float(eta), "N": int(n), "n_dumps": len(rows),
            "n_dropped_ohm": dropped,
            "p_lo": a.p_lo, "p_hi": a.p_hi,
            "R_mean": m("R"), "S_local_mean": m("S_local"),
            "RS_mean": m("R") * m("S_local"),
            "delta_mean": m("delta"), "L_mean": m("L"),
            "aspect_mean": m("delta") / m("L"),
            "B_up_mean": m("B_up"), "j_X_mean": m("j_X"),
            "Ez_X_mean": m("Ez_X"),
            "cond_X_max": max(r["hess_cond_X"] for r in rows),
            "ohm_rel_max": max(abs(r["Ez_X"] - r["eta_j_X"]) / abs(r["Ez_X"])
                               for r in rows),
        }
        out.append(rec)
        print("eta=%-7s N=%-5s dumps=%-3d dropped=%d  R=%.5f S=%.2f R*S=%.2f"
              % (eta, n, rec["n_dumps"], dropped, rec["R_mean"],
                 rec["S_local_mean"], rec["RS_mean"]))

    if a.csv_out:
        with open(a.csv_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            for rec in out:
                w.writerow(rec)
        print("wrote", a.csv_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

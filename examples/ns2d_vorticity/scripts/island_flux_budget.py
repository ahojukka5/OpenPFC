#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Flux budget for one magnetically enclosed X/O island.

Induction: dt a + u·∇a = -η j. At a magnetic null ∇a=0, so
dt a = -η j and Ez = -dt a = η j.

Island flux F = a_O - a_X then satisfies
  dF/dt = Ez_X - Ez_O = η (j_X - j_O)
in this sign convention. A changing F is not by itself reconnection:
force-free eigenmode decay changes a_O with Ez_X=0.

Restricts to well-conditioned, consistently tracked X and O
(eig_ratio above the Hessian degeneracy cut).
"""

from __future__ import print_function

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mhd_topology as mt
import analyze_mhd_reconnection as an


DEGEN_CUT = 0.05  # eig_ratio; matches locate_critical_points default


def _well(p):
    if p is None:
        return False
    if p["kind"] == mt.KIND_DEGEN:
        return False
    return float(p.get("eig_ratio", 0.0)) >= DEGEN_CUT


def pick_island(pts, a, family="ot"):
    """Choose one X and an enclosed O from the t=0 magnetic faces.

    family="ot": prefer the a≈-0.5 OT X-family (O_min island, F=-1).
    family="coalescence": X nearest (π,π) with an enclosed O_max,
    the positive-island merger of the Ng flux.
    """
    mag, xs, os_ = mt.magnetic_connectivity(pts, a)
    if not mag:
        return None, None
    cands = []
    for node in mag:
        if not node["enclosed_O"]:
            continue
        xpt = xs[node["x_index"]]
        if not _well(xpt):
            continue
        enc = max(node["enclosed_O"], key=lambda e: abs(e["delta_a"]))
        opt = os_[enc["o_index"]]
        if not _well(opt):
            continue
        cands.append((xpt, opt, enc["delta_a"], node))
    if not cands:
        return None, None
    if family == "coalescence":
        xpt, _, _, node = min(
            cands,
            key=lambda c: mt.periodic_dist(c[0]["x"], c[0]["y"],
                                           math.pi, math.pi))
        omax = [e for e in node["enclosed_O"]
                if os_[e["o_index"]]["kind"] == mt.KIND_OMAX
                and _well(os_[e["o_index"]])]
        pool = omax if omax else node["enclosed_O"]
        enc = min(pool, key=lambda e: mt.periodic_dist(
            os_[e["o_index"]]["x"], os_[e["o_index"]]["y"],
            0.5 * math.pi, 0.5 * math.pi))
        return xpt, os_[enc["o_index"]]
    neg = [c for c in cands if c[0]["a"] < 0.0]
    pool = neg if neg else cands
    xpt, opt, _, _ = min(pool, key=lambda c: abs(abs(c[2]) - 1.0))
    return xpt, opt


def track_pair(frames, times, x0, o0):
    tracks, _ = mt.track_points(frames, times=times, max_speed=mt.DEFAULT_CP_SPEED)

    def nearest_track(p0, kinds):
        best, bd = None, 1.0e9
        for tr in tracks:
            p = tr["history"][0][1]
            if p["kind"] not in kinds and tr["kind"] not in kinds:
                continue
            d = mt.periodic_dist(p["x"], p["y"], p0["x"], p0["y"])
            if d < bd:
                bd, best = d, tr
        return best

    trx = nearest_track(x0, (mt.KIND_X, mt.KIND_DEGEN))
    tro = nearest_track(o0, (mt.KIND_OMAX, mt.KIND_OMIN, mt.KIND_DEGEN))
    return trx, tro


def series_from_tracks(trx, tro, times, eta, t_max):
    by_t_x = {ti: p for ti, p in trx["history"]}
    by_t_o = {ti: p for ti, p in tro["history"]}
    rows = []
    for ti, t in enumerate(times):
        if t > t_max + 1.0e-12:
            break
        px, po = by_t_x.get(ti), by_t_o.get(ti)
        if not _well(px) or not _well(po):
            break
        if px["kind"] != mt.KIND_X:
            break
        if po["kind"] not in (mt.KIND_OMAX, mt.KIND_OMIN):
            break
        rows.append({
            "t": float(t),
            "a_X": float(px["a"]),
            "a_O": float(po["a"]),
            "j_X": float(px["j"]),
            "j_O": float(po["j"]),
            "F": float(po["a"] - px["a"]),
            "hess_cond_X": float(px["hess_cond"]),
            "hess_cond_O": float(po["hess_cond"]),
            "eig_ratio_X": float(px["eig_ratio"]),
            "eig_ratio_O": float(po["eig_ratio"]),
            "x_X": float(px["x"]),
            "y_X": float(px["y"]),
            "x_O": float(po["x"]),
            "y_O": float(po["y"]),
        })
    if len(rows) < 3:
        return None
    t = np.array([r["t"] for r in rows])
    aX = np.array([r["a_X"] for r in rows])
    aO = np.array([r["a_O"] for r in rows])
    F = np.array([r["F"] for r in rows])
    jX = np.array([r["j_X"] for r in rows])
    jO = np.array([r["j_O"] for r in rows])
    Ez_X = -mt.central_diff(t, aX)
    Ez_O = -mt.central_diff(t, aO)
    dFdt = mt.central_diff(t, F)
    eta_j_X = eta * jX
    eta_j_O = eta * jO
    rhs_E = Ez_X - Ez_O
    rhs_j = eta * (jX - jO)
    for i, r in enumerate(rows):
        r["Ez_X"] = float(Ez_X[i])
        r["Ez_O"] = float(Ez_O[i])
        r["dFdt"] = float(dFdt[i])
        r["eta_j_X"] = float(eta_j_X[i])
        r["eta_j_O"] = float(eta_j_O[i])
        r["budget_E"] = float(dFdt[i] - rhs_E[i])
        r["budget_j"] = float(dFdt[i] - rhs_j[i])
        r["ohm_X"] = float(Ez_X[i] - eta_j_X[i])
        r["ohm_O"] = float(Ez_O[i] - eta_j_O[i])
        r["frac_Ez_X"] = float(Ez_X[i] / dFdt[i]) if abs(dFdt[i]) > 1.0e-12 else None
        r["frac_Ez_O"] = float(-Ez_O[i] / dFdt[i]) if abs(dFdt[i]) > 1.0e-12 else None
    scale = max(np.max(np.abs(dFdt)), np.max(np.abs(F)) * 1.0e-3, 1.0e-12)
    # Drop endpoints: one-sided differences.
    interior = rows[1:-1]
    def rms(key):
        v = np.array([r[key] for r in interior])
        return float(np.sqrt(np.mean(v * v)))
    summary = {
        "n_times": len(rows),
        "t_first": rows[0]["t"],
        "t_last": rows[-1]["t"],
        "F_first": rows[0]["F"],
        "F_last": rows[-1]["F"],
        "rms_budget_E": rms("budget_E"),
        "rms_budget_j": rms("budget_j"),
        "rms_ohm_X": rms("ohm_X"),
        "rms_ohm_O": rms("ohm_O"),
        "rms_dFdt": rms("dFdt"),
        "rel_budget_E": rms("budget_E") / scale,
        "rel_budget_j": rms("budget_j") / scale,
        "mean_frac_Ez_X": float(np.mean([r["frac_Ez_X"] for r in interior
                                         if r["frac_Ez_X"] is not None])),
        "mean_frac_Ez_O": float(np.mean([r["frac_Ez_O"] for r in interior
                                         if r["frac_Ez_O"] is not None])),
        "mean_Ez_X": float(np.mean([r["Ez_X"] for r in interior])),
        "mean_Ez_O": float(np.mean([r["Ez_O"] for r in interior])),
        "mean_dFdt": float(np.mean([r["dFdt"] for r in interior])),
    }
    return {"rows": rows, "summary": summary, "eta": eta}


def analyze_run(directory, n, eta, t_max=0.80, stride=1, family="ot"):
    incs = an.subsample(an.dump_increments(directory), stride)
    table = an.load_csv(os.path.join(directory, "diagnostics.csv"))
    times, frames = [], []
    a0 = None
    pts0 = None
    for inc in incs:
        a = an.load_brick(os.path.join(directory, "a_%04d.bin" % inc), n)
        pts = mt.locate_critical_points(a)
        t = float(table[inc]["time"]) if inc in table else float(inc)
        times.append(t)
        frames.append(pts)
        if a0 is None:
            a0 = a
            pts0 = pts
    x0, o0 = pick_island(pts0, a0, family=family)
    if x0 is None:
        return {"error": "no enclosed island at t=0", "dir": directory, "n": n}
    trx, tro = track_pair(frames, times, x0, o0)
    if trx is None or tro is None:
        return {"error": "could not track X/O", "dir": directory, "n": n}
    out = series_from_tracks(trx, tro, times, eta, t_max)
    if out is None:
        return {"error": "too few well-conditioned samples", "dir": directory}
    out["dir"] = directory
    out["n"] = n
    out["stride"] = stride
    out["t_max"] = t_max
    out["family"] = family
    out["x0"] = {"x": x0["x"], "y": x0["y"], "a": x0["a"], "j": x0["j"]}
    out["o0"] = {"x": o0["x"], "y": o0["y"], "a": o0["a"], "j": o0["j"]}
    return out


def print_report(rep, tag=""):
    if "error" in rep:
        print("%s ERROR %s" % (tag, rep["error"]))
        return
    s = rep["summary"]
    print("%s n=%s stride=%s  t=[%.4f, %.4f]  F: %.5f -> %.5f" % (
        tag, rep.get("n"), rep.get("stride"), s["t_first"], s["t_last"],
        s["F_first"], s["F_last"]))
    print("  rms(dF/dt - (Ez_X-Ez_O)) = %.3e  rel=%.3e  (identity if Ez=-da/dt)" % (
        s["rms_budget_E"], s["rel_budget_E"]))
    print("  rms(dF/dt - eta(j_X-j_O)) = %.3e  rel=%.3e" % (
        s["rms_budget_j"], s["rel_budget_j"]))
    print("  rms(Ez_X - eta j_X)=%.3e  rms(Ez_O - eta j_O)=%.3e" % (
        s["rms_ohm_X"], s["rms_ohm_O"]))
    print("  mean dF/dt=%.4e  Ez_X=%.4e  -Ez_O=%.4e  frac_X=%.3f  frac_O=%.3f" % (
        s["mean_dFdt"], s["mean_Ez_X"], -s["mean_Ez_O"],
        s["mean_frac_Ez_X"], s["mean_frac_Ez_O"]))
    rows = rep["rows"]
    for r in rows[:: max(1, len(rows) // 8)]:
        print("  t=%.4f F=%.5f dFdt=%+.4e Ez_X=%+.4e Ez_O=%+.4e "
              "ohmX=%+.2e ohmO=%+.2e budE=%+.2e condX=%.3g" % (
                  r["t"], r["F"], r["dFdt"], r["Ez_X"], r["Ez_O"],
                  r["ohm_X"], r["ohm_O"], r["budget_E"], r["hess_cond_X"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--eta", type=float, required=True)
    p.add_argument("--t-max", type=float, default=0.80)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--family", default="ot",
                   choices=("ot", "coalescence"))
    p.add_argument("--json-out", default="")
    args = p.parse_args()
    rep = analyze_run(args.dir, args.n, args.eta, args.t_max, args.stride,
                      family=args.family)
    print_report(rep)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rep, fh, indent=2, default=str)
            fh.write("\n")
        print("wrote", args.json_out)
    return 0 if "error" not in rep else 1


if __name__ == "__main__":
    sys.exit(main())

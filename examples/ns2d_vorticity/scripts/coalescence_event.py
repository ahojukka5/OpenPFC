#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Flux-progress event window for island coalescence (#113).

The comparison interval is NOT a copied Orszag–Tang clock window.
It is the first-crossing interval of a frozen fraction of the initial
island flux F = a_O - a_X, after a finite sheet exists and before the
first rebound.

Progress:
  p(t) = (F0 - F(t)) / F0     if the first sheet_ok transfer decreases F
  p(t) = (F(t) - F0) / F0     if it increases F

Stage-0 (eta=0.01) froze PROGRESS_LO=0.30 and PROGRESS_HI=0.60.
Later eta values use the same fractions, not the same times.
Do not retune them after seeing eta=0.005.
"""

from __future__ import print_function

import argparse
import json
import sys

import numpy as np

# Frozen from the eta=0.01 Stage-0 series only. Do not retune after
# seeing eta=0.005. See COALESCENCE.md.
PROGRESS_LO = 0.30
PROGRESS_HI = 0.60
PILEUP_RATIO = 1.25
REBOUND_STREAK = 2


def _mean(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=float)))


def progress_series(rows, f0, sign):
    out = []
    for r in rows:
        p = sign * (f0 - r["F"]) / f0
        q = dict(r)
        q["progress"] = float(p)
        out.append(q)
    return out


def first_sheet_ok(rows):
    for r in rows:
        if r.get("sheet_ok"):
            return r
    return None


def transfer_sign(rows, f0):
    """+1 if F decreases from F0 through the first sheet_ok dump."""
    hit = first_sheet_ok(rows)
    if hit is None:
        return 1.0
    return 1.0 if (f0 - hit["F"]) >= 0.0 else -1.0


def first_crossing(rows, key, target, after_t=-1.0e99):
    prev = None
    for r in rows:
        if r["t"] < after_t - 1.0e-12:
            prev = r
            continue
        if prev is not None:
            a, b = prev[key], r[key]
            if (a - target) * (b - target) <= 0.0:
                return r
        prev = r
    return None


def rebound_time(rows, sign, after_t):
    """First time dF/dt reverses against the transfer direction."""
    streak = 0
    for r in rows:
        if r["t"] < after_t - 1.0e-12:
            continue
        dFdt = r.get("dFdt")
        if dFdt is None:
            continue
        # Transfer that decreases F has sign=+1 and dFdt < 0.
        reversed_ = (sign * dFdt) > 0.0
        if reversed_:
            streak += 1
            if streak >= REBOUND_STREAK:
                return r["t"]
        else:
            streak = 0
    return None


def window_rows(rows, lo, hi, t_sheet, t_rebound):
    started = False
    out = []
    for r in rows:
        if t_sheet is not None and r["t"] < t_sheet - 1.0e-12:
            continue
        if t_rebound is not None and r["t"] > t_rebound + 1.0e-12:
            break
        p = r["progress"]
        if not started:
            if p + 1.0e-12 < lo:
                continue
            started = True
        if p > hi + 1.0e-12:
            break
        out.append(r)
    return out


def summarize(rows):
    ok = [r for r in rows if r.get("sheet_ok")]
    src = ok if ok else rows
    pile = None
    if src:
        b0 = src[0].get("B_up")
        bmax = max(r.get("B_up") or 0.0 for r in src)
        if b0 and b0 > 0.0:
            pile = float(bmax / b0)
    return {
        "n": len(src),
        "n_sheet_ok": len(ok),
        "t_first": src[0]["t"] if src else None,
        "t_last": src[-1]["t"] if src else None,
        "mean_R": _mean([r.get("R") for r in src]),
        "mean_S_local": _mean([r.get("S_local") for r in src]),
        "mean_delta_over_L": _mean([
            (r["delta"] / r["L"]) if r.get("delta") and r.get("L") else None
            for r in src]),
        "mean_j_X": _mean([r.get("j_X") for r in src]),
        "mean_L": _mean([r.get("L") for r in src]),
        "mean_B_up": _mean([r.get("B_up") for r in src]),
        "mean_delta": _mean([r.get("delta") for r in src]),
        "mean_Ez_X": _mean([r.get("Ez_X") for r in src]),
        "mean_eta_j_X": _mean([r.get("eta_j_X") for r in src]),
        "flux_pileup_ratio": pile,
        "flux_pileup": bool(pile is not None and pile >= PILEUP_RATIO),
    }


def analyze(rep, progress_lo, progress_hi):
    if "error" in rep:
        return {"error": rep["error"]}
    rows = list(rep.get("rows") or [])
    if len(rows) < 3:
        return {"error": "too few samples"}
    f0 = float(rep.get("summary", {}).get("F_first", rows[0]["F"]))
    if abs(f0) < 1.0e-12:
        return {"error": "F0 is zero"}
    sign = transfer_sign(rows, f0)
    rows = progress_series(rows, f0, sign)
    sheet = first_sheet_ok(rows)
    t_sheet = None if sheet is None else sheet["t"]
    t_rebound = rebound_time(rows, sign, t_sheet if t_sheet is not None
                             else rows[0]["t"])
    event = None
    if progress_lo is not None and progress_hi is not None:
        event_rows = window_rows(rows, progress_lo, progress_hi, t_sheet,
                                 t_rebound)
        event = summarize(event_rows)
        event["progress_lo"] = progress_lo
        event["progress_hi"] = progress_hi
        event["common_sheet_ok"] = bool(event_rows) and all(
            r.get("sheet_ok") for r in event_rows)
    return {
        "F0": f0,
        "transfer_sign": sign,
        "t_sheet_ok": t_sheet,
        "t_rebound": t_rebound,
        "sloshing_sentinel": t_rebound is not None,
        "progress_lo": progress_lo,
        "progress_hi": progress_hi,
        "p_last": rows[-1]["progress"],
        "F_last": rows[-1]["F"],
        "event": event,
        "rows": rows,
    }


def print_report(rep):
    if "error" in rep:
        print("ERROR", rep["error"])
        return
    print("F0=%.5f  sign=%+.0f  p_last=%.3f  t_sheet=%s  t_rebound=%s" % (
        rep["F0"], rep["transfer_sign"], rep["p_last"],
        None if rep["t_sheet_ok"] is None else "%.4f" % rep["t_sheet_ok"],
        None if rep["t_rebound"] is None else "%.4f" % rep["t_rebound"]))
    print("sloshing_sentinel=%s" % rep["sloshing_sentinel"])
    ev = rep.get("event")
    if not ev:
        print("event window not frozen (PROGRESS_LO/HI unset)")
        return
    print("event p in [%.3f, %.3f]  t=[%s, %s]  n=%s sheet_ok=%s common=%s" % (
        ev["progress_lo"], ev["progress_hi"],
        ev["t_first"], ev["t_last"], ev["n"], ev["n_sheet_ok"],
        ev["common_sheet_ok"]))
    print("  mean R=%.4e  S_local=%.4g  delta/L=%.4g  j_X=%.4g  L=%.4g  "
          "B_up=%.4g" % (
              ev["mean_R"] or 0.0, ev["mean_S_local"] or 0.0,
              ev["mean_delta_over_L"] or 0.0, ev["mean_j_X"] or 0.0,
              ev["mean_L"] or 0.0, ev["mean_B_up"] or 0.0))
    print("  flux_pileup_ratio=%.3g  flux_pileup=%s" % (
        ev["flux_pileup_ratio"] or 0.0, ev["flux_pileup"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", required=True,
                   help="sheet_scaling JSON (rows + F)")
    p.add_argument("--progress-lo", type=float, default=PROGRESS_LO)
    p.add_argument("--progress-hi", type=float, default=PROGRESS_HI)
    p.add_argument("--json-out", default="")
    args = p.parse_args()
    with open(args.json) as fh:
        src = json.load(fh)
    rep = analyze(src, args.progress_lo, args.progress_hi)
    print_report(rep)
    if args.json_out:
        slim = dict(rep)
        slim.pop("rows", None)
        with open(args.json_out, "w") as fh:
            json.dump(slim, fh, indent=2, default=str)
            fh.write("\n")
        print("wrote", args.json_out)
    return 0 if "error" not in rep else 1


if __name__ == "__main__":
    sys.exit(main())

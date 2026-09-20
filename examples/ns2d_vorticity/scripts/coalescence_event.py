#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""X-point reconnection-flux event window for island coalescence (#113).

A raw change of island flux F=a_O-a_X is NOT a suitable cross-eta event
coordinate for this benchmark: the unperturbed a=0.4 sin(x) sin(y) field is
itself a resistive eigenmode, so a_O and therefore F decay even with
E_z(X)=0.

Use instead the cumulative flux transferred through the tracked X-point,

  Psi_X(t) = integral_0^t E_z(X,t') dt' = a_X(0) - a_X(t),

and the dimensionless progress

  p_X(t) = transfer_sign * Psi_X(t) / |F(0)|.

The sign is frozen from E_z(X) at the first sheet_ok dump. A force-free /
pure-eigenmode decay with a_X=0 therefore has p_X=0 even while F changes.

After review of the eta=0.01 Stage-0 campaign, and before any eta=0.005 run,
the comparator is frozen to XFLUX_LO=0.05 and XFLUX_HI=0.10. The event starts
only after sheet_ok and stops before the first X-point-rate reversal.
Do not retune these constants after seeing eta=0.005.
"""

from __future__ import print_function

import argparse
import json
import sys

import numpy as np


# Frozen before any eta=0.005 run. These are fractions of |F(0)| transferred
# through the tracked X-point, not fractions of the total island-flux change.
XFLUX_LO = 0.05
XFLUX_HI = 0.10
PILEUP_RATIO = 1.25
REBOUND_STREAK = 2


def _mean(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=float)))


def first_sheet_ok(rows):
    for r in rows:
        if r.get("sheet_ok"):
            return r
    return None


def transfer_sign(rows):
    """Sign that makes cumulative X-point transfer positive.

    Ez_X=-d a_X/dt, so Psi_X=a_X(0)-a_X(t). Prefer Ez_X at the first
    sheet_ok sample; if it is numerically zero, use the first later non-zero
    Ez_X. This sign is a direction convention only, not an event detector.
    """
    hit = first_sheet_ok(rows)
    start = 0 if hit is None else rows.index(hit)
    for r in rows[start:]:
        ez = r.get("Ez_X")
        if ez is not None and abs(float(ez)) > 1.0e-14:
            return 1.0 if float(ez) > 0.0 else -1.0
    return 1.0


def attach_progress(rows, f0, sign):
    """Attach both accepted X-flux progress and diagnostic raw-F progress."""
    if abs(f0) < 1.0e-14:
        raise ValueError("F0 is zero")
    a_x0 = float(rows[0]["a_X"])
    out = []
    for r in rows:
        q = dict(r)
        psi_x = a_x0 - float(r["a_X"])
        q["Psi_X"] = float(psi_x)
        q["xflux_progress"] = float(sign * psi_x / abs(f0))
        # Diagnostic only. Never use this to align eta cases because O-point
        # eigenmode decay changes F even when the X-point transfer is zero.
        q["raw_F_progress"] = float((f0 - float(r["F"])) / abs(f0))
        out.append(q)
    return out


def rebound_time(rows, sign, after_t):
    """First persistent reversal of the X-point reconnection electric field."""
    streak = 0
    for r in rows:
        if r["t"] < after_t - 1.0e-12:
            continue
        ez = r.get("Ez_X")
        if ez is None:
            continue
        reversed_ = sign * float(ez) < -1.0e-14
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
        p = r["xflux_progress"]
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
        "xflux_first": src[0]["xflux_progress"] if src else None,
        "xflux_last": src[-1]["xflux_progress"] if src else None,
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


def analyze(rep, progress_lo=XFLUX_LO, progress_hi=XFLUX_HI):
    if "error" in rep:
        return {"error": rep["error"]}
    rows = list(rep.get("rows") or [])
    if len(rows) < 3:
        return {"error": "too few samples"}
    if any("a_X" not in r for r in rows):
        return {"error": "rows do not contain a_X; cannot form X-flux progress"}

    f0 = float(rep.get("summary", {}).get("F_first", rows[0]["F"]))
    if abs(f0) < 1.0e-12:
        return {"error": "F0 is zero"}

    sign = transfer_sign(rows)
    rows = attach_progress(rows, f0, sign)
    sheet = first_sheet_ok(rows)
    t_sheet = None if sheet is None else sheet["t"]
    t_rebound = rebound_time(
        rows, sign, t_sheet if t_sheet is not None else rows[0]["t"])

    event_rows = window_rows(
        rows, progress_lo, progress_hi, t_sheet, t_rebound)
    event = summarize(event_rows)
    event["progress_lo"] = progress_lo
    event["progress_hi"] = progress_hi
    event["common_sheet_ok"] = bool(event_rows) and all(
        r.get("sheet_ok") for r in event_rows)

    # Fail closed if the frozen X-flux bracket has already passed before a
    # finite sheet exists or is not reached in the available Stage-0 data.
    event["reached"] = bool(event_rows)
    event["complete"] = bool(
        event_rows
        and event_rows[0]["xflux_progress"] <= progress_hi + 1.0e-12
        and event_rows[-1]["xflux_progress"] >= progress_lo - 1.0e-12
    )

    return {
        "F0": f0,
        "a_X0": float(rows[0]["a_X"]),
        "transfer_sign": sign,
        "progress_definition": "sign*(a_X(0)-a_X(t))/abs(F(0))",
        "t_sheet_ok": t_sheet,
        "t_rebound": t_rebound,
        "sloshing_sentinel": t_rebound is not None,
        "progress_lo": progress_lo,
        "progress_hi": progress_hi,
        "p_X_last": rows[-1]["xflux_progress"],
        "raw_F_progress_last": rows[-1]["raw_F_progress"],
        "F_last": rows[-1]["F"],
        "event": event,
        "rows": rows,
    }


def print_report(rep):
    if "error" in rep:
        print("ERROR", rep["error"])
        return
    print("F0=%.5f a_X0=%+.5e sign=%+.0f p_X_last=%.3f raw_F_progress_last=%.3f "
          "t_sheet=%s t_rebound=%s" % (
              rep["F0"], rep["a_X0"], rep["transfer_sign"],
              rep["p_X_last"], rep["raw_F_progress_last"],
              None if rep["t_sheet_ok"] is None else "%.4f" % rep["t_sheet_ok"],
              None if rep["t_rebound"] is None else "%.4f" % rep["t_rebound"]))
    print("progress=%s" % rep["progress_definition"])
    print("sloshing_sentinel=%s" % rep["sloshing_sentinel"])
    ev = rep.get("event")
    print("event p_X in [%.3f, %.3f] reached=%s complete=%s "
          "t=[%s, %s] n=%s sheet_ok=%s common=%s" % (
              ev["progress_lo"], ev["progress_hi"], ev["reached"],
              ev["complete"], ev["t_first"], ev["t_last"], ev["n"],
              ev["n_sheet_ok"], ev["common_sheet_ok"]))
    if ev["n"]:
        print("  mean R=%.4e S_local=%.4g delta/L=%.4g j_X=%.4g L=%.4g "
              "B_up=%.4g" % (
                  ev["mean_R"] or 0.0, ev["mean_S_local"] or 0.0,
                  ev["mean_delta_over_L"] or 0.0, ev["mean_j_X"] or 0.0,
                  ev["mean_L"] or 0.0, ev["mean_B_up"] or 0.0))
        print("  flux_pileup_ratio=%.3g flux_pileup=%s" % (
            ev["flux_pileup_ratio"] or 0.0, ev["flux_pileup"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", required=True,
                   help="sheet_scaling JSON (rows include a_X and F)")
    p.add_argument("--progress-lo", type=float, default=XFLUX_LO)
    p.add_argument("--progress-hi", type=float, default=XFLUX_HI)
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
    if "error" in rep:
        return 1
    ev = rep["event"]
    return 0 if ev["reached"] and ev["complete"] and ev["common_sheet_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())

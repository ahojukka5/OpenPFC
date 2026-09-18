#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reconnection/topology analysis of mhd2d BinaryWriter dumps.

Computes tracked X/O points, Ez = -d(a_X)/dt vs eta*j_X, Morse-paired
Delta_a, and a conservative connectivity-change flag. Does not claim
reconnection; that requires agreement + convergence (issue #26).
"""

from __future__ import print_function

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mhd_topology as mt
from compare_mhd_fields import load_brick


def dump_increments(directory):
    incs = []
    for path in glob.glob(os.path.join(directory, "a_*.bin")):
        m = re.search(r"a_(\d+)\.bin$", os.path.basename(path))
        if m:
            incs.append(int(m.group(1)))
    return sorted(set(incs))


def load_csv(path):
    rows = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows[int(float(row["step"]))] = row
    return rows


def subsample(incs, stride):
    out = incs[::stride]
    if out[-1] != incs[-1]:
        out.append(incs[-1])
    return out


def sheet_metrics(j, x, y, length=mt.TWOPI):
    n = j.shape[0]
    dx = length / n
    jabs = np.abs(j)
    # Thickness: FWHM along the thinner of x/y through (x,y).
    i = int(round(x / dx - 0.5)) % n
    jj = int(round(y / dx - 0.5)) % n

    def fwhm(arr, ipeak):
        half = 0.5 * arr[ipeak]
        left = 0
        while left < n and arr[(ipeak - left - 1) % n] >= half:
            left += 1
        right = 0
        while right < n and arr[(ipeak + right + 1) % n] >= half:
            right += 1
        return min(n, left + right + 1) * dx

    th_x = fwhm(jabs[:, jj], i)
    th_y = fwhm(jabs[i, :], jj)
    thickness = min(th_x, th_y)
    # Length: sqrt of area with |j| >= 0.5 max in a neighbourhood, else
    # domain-wide half-max area^{1/2} as an operational upper bound.
    half = 0.5 * float(np.max(jabs))
    area = float(np.count_nonzero(jabs >= half)) * dx * dx
    length_op = math.sqrt(area)
    aspect = length_op / max(thickness, 1.0e-12)
    # Upstream |B| ~ |∇a| sampled one thickness away is not available here;
    # use rms |B| from j = k^2 a is heavy. Report peak |j| and local |j|.
    return {
        "thickness": thickness,
        "length_op": length_op,
        "aspect": aspect,
        "peak_j": float(np.max(jabs)),
        "j_at_point": float(jabs[i, jj]),
    }


def pairing_signature(pts):
    """Count-based fingerprint. Morse a-rounding is too noisy to be topology."""
    nX = sum(1 for p in pts if p["kind"] == mt.KIND_X)
    nO = sum(1 for p in pts if p["kind"] in (mt.KIND_OMAX, mt.KIND_OMIN))
    nD = sum(1 for p in pts if p["kind"] == mt.KIND_DEGEN)
    return (nX, nO, nD)


def analyze_dir(directory, n, eta, stride=1):
    incs = dump_increments(directory)
    incs = subsample(incs, stride)
    csv_path = os.path.join(directory, "diagnostics.csv")
    table = load_csv(csv_path) if os.path.exists(csv_path) else {}
    times = []
    frames = []
    fields = []
    for inc in incs:
        a = load_brick(os.path.join(directory, "a_%04d.bin" % inc), n)
        jpath = os.path.join(directory, "j_%04d.bin" % inc)
        j = load_brick(jpath, n) if os.path.exists(jpath) else None
        pts = mt.locate_critical_points(a, j)
        frames.append(pts)
        t = float(table[inc]["time"]) if inc in table else float(inc)
        times.append(t)
        fields.append({"inc": inc, "t": t, "a": a, "j": j, "pts": pts})
    dx = mt.TWOPI / n
    tracks, events = mt.track_points(frames, max_speed=4.0 * dx)
    series = []
    prev_sig = None
    for k, fld in enumerate(fields):
        pts = fld["pts"]
        pairs, xs, os_ = mt.pair_morse(pts, fld["a"])
        sig = pairing_signature(pts)
        rec = {
            "inc": fld["inc"],
            "t": fld["t"],
            "nX": sum(1 for p in pts if p["kind"] == mt.KIND_X),
            "nO": sum(1 for p in pts if p["kind"] in (mt.KIND_OMAX, mt.KIND_OMIN)),
            "n_degen": sum(1 for p in pts if p["kind"] == mt.KIND_DEGEN),
            "pairing": sig,
            "connectivity_changed": prev_sig is not None and sig != prev_sig,
        }
        prev_sig = sig
        if xs:
            xpt = max(xs, key=lambda p: abs(p["j"] or 0.0))
            rec["X"] = {kk: xpt[kk] for kk in ("x", "y", "a", "j", "hess_cond", "eig_ratio", "kind")}
            rec["eta_j_X"] = eta * (xpt["j"] or 0.0)
            if os_ and pairs:
                # Pair this X with linked O of largest |a|.
                xi = xs.index(xpt) if xpt in xs else 0
                links = []
                for pr in pairs:
                    if pr["x_index"] == xi:
                        links = pr["o_indices"]
                if links:
                    o = max((os_[i] for i in links), key=lambda p: abs(p["a"]))
                    rec["O"] = {kk: o[kk] for kk in ("x", "y", "a", "j", "kind")}
                    rec["Delta_a"] = o["a"] - xpt["a"]
            if fld["j"] is not None:
                rec["sheet"] = sheet_metrics(fld["j"], xpt["x"], xpt["y"])
                Bup = rec["sheet"]["thickness"]
                # Local Lundquist using operational length and V_A~1, eta given.
                rec["sheet"]["S_local"] = rec["sheet"]["length_op"] / max(eta, 1.0e-30)
        series.append(rec)

    # Ez along the longest X track.
    xtracks = [tr for tr in tracks if tr["kind"] == mt.KIND_X]
    ez_series = []
    if xtracks:
        tr = max(xtracks, key=lambda t: len(t["history"]))
        t_tr, a_tr, dadt = mt.da_dt_from_track(tr["history"], times)
        for m, (ti, p) in enumerate(tr["history"]):
            ez_series.append({
                "t": float(t_tr[m]),
                "a_X": float(a_tr[m]),
                "Ez_X": float(-dadt[m]),
                "eta_j_X": eta * (p["j"] or 0.0),
                "x": p["x"],
                "y": p["y"],
                "hess_cond": p["hess_cond"],
                "track_id": tr["id"],
            })
    return {
        "dir": directory,
        "n": n,
        "eta": eta,
        "stride": stride,
        "n_frames": len(fields),
        "times": times,
        "incs": incs,
        "n_tracks": len(tracks),
        "n_events": len(events),
        "events": [{"type": e["type"], "t_index": e.get("t_index")} for e in events],
        "series": series,
        "ez": ez_series,
        "connectivity_changes": sum(1 for r in series if r["connectivity_changed"]),
    }


def compare_resolutions(coarse, fine, times_of_interest):
    """Match nearest times and report diagnostic differences."""
    rows = []
    for t0 in times_of_interest:
        rc = min(coarse["series"], key=lambda r: abs(r["t"] - t0))
        rf = min(fine["series"], key=lambda r: abs(r["t"] - t0))
        def get(r, *keys):
            cur = r
            for k in keys:
                if cur is None or k not in cur:
                    return None
                cur = cur[k]
            return cur
        row = {"t_request": t0, "t_coarse": rc["t"], "t_fine": rf["t"]}
        for name, path in (
            ("nX", ("nX",)),
            ("nO", ("nO",)),
            ("a_X", ("X", "a")),
            ("a_O", ("O", "a")),
            ("j_X", ("X", "j")),
            ("x_X", ("X", "x")),
            ("y_X", ("X", "y")),
            ("Delta_a", ("Delta_a",)),
            ("eta_j_X", ("eta_j_X",)),
        ):
            vc, vf = get(rc, *path), get(rf, *path)
            row[name + "_c"] = vc
            row[name + "_f"] = vf
            if isinstance(vc, (int, float)) and isinstance(vf, (int, float)):
                row[name + "_diff"] = vf - vc
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--eta", type=float, required=True)
    p.add_argument("--stride", type=int, default=1,
                   help="keep every stride-th dump (cadence study)")
    p.add_argument("--fine-dir", default="")
    p.add_argument("--fine-n", type=int, default=0)
    p.add_argument("--json-out", default="")
    args = p.parse_args()
    report = analyze_dir(args.dir, args.n, args.eta, args.stride)
    print("frames=%d tracks=%d events=%d connectivity_changes=%d" % (
        report["n_frames"], report["n_tracks"], report["n_events"],
        report["connectivity_changes"]))
    for ez in report["ez"][:: max(1, len(report["ez"]) // 12)]:
        print("  t=%.4f Ez_X=%+.4e eta*j_X=%+.4e a_X=%.5f cond=%.2g" % (
            ez["t"], ez["Ez_X"], ez["eta_j_X"], ez["a_X"], ez["hess_cond"]))
    nchg = 0
    for rec in report["series"]:
        if rec["connectivity_changed"]:
            nchg += 1
            print("  connectivity change at t=%.4f nX=%d nO=%d degen=%d" % (
                rec["t"], rec["nX"], rec["nO"], rec["n_degen"]))
    if args.fine_dir:
        fine = analyze_dir(args.fine_dir, args.fine_n, args.eta, args.stride)
        report["fine"] = {k: fine[k] for k in fine if k != "series"}
        toi = [2.238, 1.924, 1.492, 0.0]
        report["resolution_compare"] = compare_resolutions(report, fine, toi)
        for row in report["resolution_compare"]:
            print("N-compare t~%.3f  nX %s vs %s  a_X %s vs %s  Delta_a %s vs %s" % (
                row["t_request"], row.get("nX_c"), row.get("nX_f"),
                row.get("a_X_c"), row.get("a_X_f"),
                row.get("Delta_a_c"), row.get("Delta_a_f")))
    if args.json_out:
        # Drop bulky pairing tuples to keep JSON small; stringify.
        out = dict(report)
        for rec in out["series"]:
            rec["pairing"] = repr(rec.get("pairing"))
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
            fh.write("\n")
        print("wrote", args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

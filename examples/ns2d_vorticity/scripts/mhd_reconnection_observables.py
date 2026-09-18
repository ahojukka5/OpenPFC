#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Candidate 2-D resistive-MHD reconnection observables from dumped a, j.

For this model's signs, Faraday gives E_z = -∂_t a. The induction equation is
∂_t a + u·∇a = η ∇² a = -η j. At a rest X-point (u=0) that reduces to
E_z = η j. Competing diagnostics (not a rate yet):

* a_O - a_X  (flux difference between a nearby O-point and X-point)
* -∂_t a at the strongest-current X-point
* η j at that X-point
* sheet FWHM / length aspect ratio

This script does not claim reconnection.
"""

from __future__ import annotations

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
from compare_mhd_fields import load_brick
from find_mhd_critical_points import find_critical_points


def dump_increments(directory: str) -> list[int]:
    incs = []
    for path in glob.glob(os.path.join(directory, "a_*.bin")):
        m = re.search(r"a_(\d+)\.bin$", os.path.basename(path))
        if m:
            incs.append(int(m.group(1)))
    return sorted(set(incs))


def load_times(csv_path: str) -> dict[int, dict]:
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[int(float(row["step"]))] = row
    return rows


def sheet_length_from_j(j: np.ndarray, frac: float = 0.5) -> float:
    """Operational length: number of cells with |j| >= frac * max|j|, as sqrt(area)."""
    jabs = np.abs(j)
    m = float(np.max(jabs))
    if m == 0.0:
        return 0.0
    n = j.shape[0]
    dx = 2.0 * math.pi / n
    area = float(np.count_nonzero(jabs >= frac * m)) * dx * dx
    return math.sqrt(area)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--eta", type=float, required=True)
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    incs = dump_increments(args.dir)
    csv_path = os.path.join(args.dir, "diagnostics.csv")
    times = load_times(csv_path) if os.path.exists(csv_path) else {}
    series = []
    prev_a = None
    prev_t = None
    for inc in incs:
        a = load_brick(os.path.join(args.dir, f"a_{inc:04d}.bin"), args.n)
        j = load_brick(os.path.join(args.dir, f"j_{inc:04d}.bin"), args.n)
        t = float(times[inc]["time"]) if inc in times else float("nan")
        pts = find_critical_points(a, j)
        xs = [q for q in pts if q["kind"] == "X"]
        os_ = [q for q in pts if q["kind"].startswith("O")]
        rec = {
            "inc": inc,
            "time": t,
            "nX": len(xs),
            "nO": len(os_),
        }
        if xs:
            xpt = max(xs, key=lambda q: abs(q["j_val"] or 0.0))
            rec["X"] = xpt
            rec["eta_j_X"] = args.eta * (xpt["j_val"] or 0.0)
        if os_:
            rec["a_Omax"] = max(q["a"] for q in os_)
            rec["a_Omin"] = min(q["a"] for q in os_)
            if xs:
                rec["aO_minus_aX"] = rec["a_Omax"] - rec["X"]["a"]
        jabs = np.abs(j)
        loc = np.unravel_index(int(np.argmax(jabs)), jabs.shape)
        rec["max_abs_j"] = float(jabs[loc])
        rec["eta_j_at_maxj"] = args.eta * float(j[loc])
        rec["sheet_length_op"] = sheet_length_from_j(j)
        if prev_a is not None and prev_t is not None and t > prev_t:
            ez = -(a - prev_a) / (t - prev_t)
            rec["Ez_at_maxj"] = float(ez[loc])
            rec["Ez_minus_eta_j_maxj"] = rec["Ez_at_maxj"] - rec["eta_j_at_maxj"]
            if "X" in rec:
                rec["Ez_X"] = float(ez[rec["X"]["ix"], rec["X"]["iy"]])
                rec["Ez_minus_eta_j"] = rec["Ez_X"] - rec["eta_j_X"]
        series.append(rec)
        prev_a, prev_t = a, t
        print(
            f"t={t:.4f} nX={rec['nX']} nO={rec['nO']} "
            f"max|j|={rec['max_abs_j']:.4g} "
            f"aO-aX={rec.get('aO_minus_aX')} "
            f"Ez_maxj={rec.get('Ez_at_maxj')} eta_j_maxj={rec.get('eta_j_at_maxj')}"
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"dir": args.dir, "n": args.n, "eta": args.eta, "series": series}, fh, indent=2)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

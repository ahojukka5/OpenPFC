#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic tests for the sub-grid periodic X/O tracker.

Fails if a known critical point is shifted, misclassified, or lost.
"""

from __future__ import print_function

import math
import os
import sys

try:
    import numpy as np
except ImportError:
    print("SKIP: numpy is not available")
    sys.exit(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mhd_topology as mt

TWOPI = 2.0 * math.pi
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)
    else:
        print(" ok ", msg)


def nearest(pts, x, y):
    return min(pts, key=lambda p: mt.periodic_dist(p["x"], p["y"], x, y))


def test_stationary_sinxsiny():
    print("\n== stationary a = sin x sin y ==")
    a = mt.sample_grid(lambda x, y: np.sin(x) * np.sin(y), 64)
    pts = mt.locate_critical_points(a)
    xs = [p for p in pts if p["kind"] == mt.KIND_X]
    omax = [p for p in pts if p["kind"] == mt.KIND_OMAX]
    omin = [p for p in pts if p["kind"] == mt.KIND_OMIN]
    check(len(xs) == 4, "exactly 4 X-points, got %d" % len(xs))
    check(len(omax) == 2, "exactly 2 O_max, got %d" % len(omax))
    check(len(omin) == 2, "exactly 2 O_min, got %d" % len(omin))
    saddles = [(0.0, 0.0), (math.pi, 0.0), (0.0, math.pi), (math.pi, math.pi)]
    maxima = [(0.5 * math.pi, 0.5 * math.pi), (1.5 * math.pi, 1.5 * math.pi)]
    minima = [(0.5 * math.pi, 1.5 * math.pi), (1.5 * math.pi, 0.5 * math.pi)]
    for x, y in saddles:
        p = nearest(xs, x, y) if xs else None
        check(p is not None and mt.periodic_dist(p["x"], p["y"], x, y) < 0.05,
              "X near (%.3f,%.3f)" % (x, y))
    for x, y in maxima:
        p = nearest(omax, x, y) if omax else None
        check(p is not None and mt.periodic_dist(p["x"], p["y"], x, y) < 0.05,
              "O_max near (%.3f,%.3f) a=%.4f" % (x, y, p["a"] if p else 0))
        if p:
            check(abs(p["a"] - 1.0) < 0.02, "O_max value ~ 1")
    for x, y in minima:
        p = nearest(omin, x, y) if omin else None
        check(p is not None and mt.periodic_dist(p["x"], p["y"], x, y) < 0.05,
              "O_min near (%.3f,%.3f)" % (x, y))
        if p:
            check(abs(p["a"] + 1.0) < 0.02, "O_min value ~ -1")
    # Shifted-grid failure: translating the field by 0.3 must move CPs by 0.3.
    a2 = mt.sample_grid(lambda x, y: np.sin(x - 0.3) * np.sin(y), 64)
    pts2 = mt.locate_critical_points(a2)
    xs2 = [p for p in pts2 if p["kind"] == mt.KIND_X]
    p0 = nearest(xs, 0.0, 0.0)
    p1 = nearest(xs2, 0.3, 0.0)
    check(p0 and p1 and abs(mt.periodic_delta(p1["x"], p0["x"]) - 0.3) < 0.05,
          "translated X follows the shift (fails if tracker stuck on nodes)")


def test_moving():
    print("\n== translating a = sin(x-vt) sin(y-wt) ==")
    v, w = 0.4, -0.25
    times = [0.05 * k for k in range(16)]
    frames = []
    for t in times:
        a = mt.sample_grid(
            lambda x, y, tt=t: np.sin(x - v * tt) * np.sin(y - w * tt), 64)
        frames.append(mt.locate_critical_points(a))
    tracks, events = mt.track_points(frames, max_speed=0.15)
    live = [tr for tr in tracks if len(tr["history"]) == len(times)]
    check(len(live) == 8, "8 tracks survive all frames, got %d" % len(live))
    births = [e for e in events if e["type"] == "birth" and e["t_index"] > 0]
    deaths = [e for e in events if e["type"] == "death"]
    check(len(births) == 0 and len(deaths) == 0,
          "no birth/death on a rigid translation (got %d/%d)" % (
              len(births), len(deaths)))
    # One X that starts near origin should travel (vt, wt).
    origin_tracks = []
    for tr in live:
        p0 = tr["history"][0][1]
        if tr["kind"] == mt.KIND_X and mt.periodic_dist(p0["x"], p0["y"], 0, 0) < 0.2:
            origin_tracks.append(tr)
    check(len(origin_tracks) == 1, "one X track starts at origin")
    if origin_tracks:
        pN = origin_tracks[0]["history"][-1][1]
        tN = times[-1]
        check(mt.periodic_dist(pN["x"], pN["y"], v * tN, w * tN) < 0.08,
              "origin X arrives at (vt,wt)")


def test_ot_initial():
    print("\n== OT flux a = 0.5 cos 2x + cos y ==")
    a = mt.sample_grid(lambda x, y: 0.5 * np.cos(2.0 * x) + np.cos(y), 64)
    pts = mt.locate_critical_points(a)
    xs = [p for p in pts if p["kind"] == mt.KIND_X]
    os_ = [p for p in pts if p["kind"] in (mt.KIND_OMAX, mt.KIND_OMIN)]
    check(len(pts) == 8, "OT has 8 critical points, got %d" % len(pts))
    check(len(xs) == 4 and len(os_) == 4, "OT 4 X and 4 O")


def test_creation_annihilation():
    print("\n== 1-D fold on a torus: pair creation/annihilation ==")
    # a = (sin x - alpha)^2 + 0.25 sin y. For |alpha|<1 there are extra
    # extrema in x; at |alpha|=1 they collide. Use alpha from 1.4 -> 0.4
    # (creation) then back (annihilation).
    alphas = [1.4, 1.2, 1.05, 0.9, 0.5, 0.9, 1.05, 1.2, 1.4]
    frames = []
    counts = []
    for alpha in alphas:
        a = mt.sample_grid(
            lambda x, y, al=alpha: (np.sin(x) - al) ** 2 + 0.25 * np.sin(y), 96)
        pts = mt.locate_critical_points(a)
        frames.append(pts)
        counts.append(len(pts))
        print("  alpha=%.2f nCP=%d kinds=%s" % (
            alpha, len(pts), [p["kind"] for p in pts]))
    tracks, events = mt.track_points(frames, max_speed=0.5)
    births = [e for e in events if e["type"] == "birth" and e["t_index"] > 0]
    deaths = [e for e in events if e["type"] == "death"]
    check(max(counts) > min(counts), "critical-point count changes with alpha")
    check(len(births) >= 1, "tracker records a birth during creation")
    check(len(deaths) >= 1, "tracker records a death during annihilation")


def test_decaying_eigenmode_keeps_counts():
    print("\n== decaying a = e^{-t} sin x sin y (no topology change) ==")
    frames = []
    for t in [0.0, 0.2, 0.5, 1.0]:
        a = mt.sample_grid(
            lambda x, y, tt=t: math.exp(-tt) * np.sin(x) * np.sin(y), 48)
        frames.append(mt.locate_critical_points(a))
        nX = sum(1 for p in frames[-1] if p["kind"] == mt.KIND_X)
        nO = sum(1 for p in frames[-1] if p["kind"] in (mt.KIND_OMAX, mt.KIND_OMIN))
        check(nX == 4 and nO == 4, "t=%.1f still 4X+4O (got %d+%d)" % (t, nX, nO))
    tracks, events = mt.track_points(frames, max_speed=0.2)
    deaths = [e for e in events if e["type"] == "death"]
    births = [e for e in events if e["type"] == "birth" and e["t_index"] > 0]
    check(len(deaths) == 0 and len(births) == 0,
          "decaying eigenmode has no birth/death")


def test_misclassify_fails():
    print("\n== classification must not swap X and O ==")
    a = mt.sample_grid(lambda x, y: np.sin(x) * np.sin(y), 48)
    pts = mt.locate_critical_points(a)
    p = nearest([q for q in pts if q["kind"] == mt.KIND_OMAX],
                0.5 * math.pi, 0.5 * math.pi)
    check(p["kind"] == mt.KIND_OMAX, "maximum is not labelled X")
    p = nearest([q for q in pts if q["kind"] == mt.KIND_X], 0.0, 0.0)
    check(p["kind"] == mt.KIND_X, "saddle is not labelled O")


def main():
    test_stationary_sinxsiny()
    test_moving()
    test_ot_initial()
    test_creation_annihilation()
    test_decaying_eigenmode_keeps_counts()
    test_misclassify_fails()
    print("\n%d failures" % len(FAILS))
    if FAILS:
        for f in FAILS:
            print(" -", f)
        return 1
    print("ALL TOPOLOGY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

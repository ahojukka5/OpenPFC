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
import island_flux_budget as ifb

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
    tracks, events = mt.track_points(frames, times=times, max_speed=2.0)
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


def test_moving_multi_n_and_cadence():
    print("\n== same moving topology at several N and cadences ==")
    v, w = 0.4, -0.25
    for n in (32, 64, 128):
        for dt in (0.025, 0.05, 0.10):
            nstep = int(round(0.4 / dt))
            times = [dt * k for k in range(nstep + 1)]
            frames = []
            for t in times:
                a = mt.sample_grid(
                    lambda x, y, tt=t: np.sin(x - v * tt) * np.sin(y - w * tt), n)
                frames.append(mt.locate_critical_points(a))
            tracks, events = mt.track_points(frames, times=times, max_speed=2.0)
            live = [tr for tr in tracks if len(tr["history"]) == len(times)]
            births = [e for e in events if e["type"] == "birth" and e["t_index"] > 0]
            deaths = [e for e in events if e["type"] == "death"]
            check(len(live) == 8 and len(births) == 0 and len(deaths) == 0,
                  "N=%d dt=%.3f: 8 live tracks, no birth/death (live=%d b=%d d=%d)"
                  % (n, dt, len(live), len(births), len(deaths)))


def test_separatrix_not_nearest_o():
    print("\n== Morse O is not the nearest O ==")
    # -cos x + cos y: X at (0,0) is Morse-linked to O_max at (π,0).
    # A compact diagonal bump adds a nearer O_max. A nearest-O fallback
    # would never report (π,0).
    def field(x, y):
        bump = 0.5 * np.exp(12.0 * (np.cos(x - 0.65) + np.cos(y - 0.65) - 2.0))
        return -np.cos(x) + np.cos(y) + bump

    a = mt.sample_grid(field, 128)
    pts = mt.locate_critical_points(a)
    xs = [p for p in pts if p["kind"] == mt.KIND_X]
    os_ = [p for p in pts if p["kind"] in (mt.KIND_OMAX, mt.KIND_OMIN)]
    xpt = nearest(xs, 0.0, 0.0)
    check(mt.periodic_dist(xpt["x"], xpt["y"], 0.0, 0.0) < 0.25,
          "origin X survives the bump")
    nearest_o = nearest(os_, xpt["x"], xpt["y"])
    far = nearest([p for p in os_ if p["kind"] == mt.KIND_OMAX], math.pi, 0.0)
    check(mt.periodic_dist(nearest_o["x"], nearest_o["y"], xpt["x"], xpt["y"]) + 1.0
          < mt.periodic_dist(far["x"], far["y"], xpt["x"], xpt["y"]),
          "decoy O is nearer than Morse O_max at (π,0)")
    pairs, xs2, os2 = mt.pair_morse(pts, a)
    xi = None
    for i, p in enumerate(xs2):
        if mt.periodic_dist(p["x"], p["y"], xpt["x"], xpt["y"]) < 1.0e-8:
            xi = i
            break
    check(xi is not None, "origin X is in the pairing list")
    linked = [os2[k] for k in pairs[xi]["o_indices"]] if xi is not None else []
    has_far = any(
        mt.periodic_dist(o["x"], o["y"], far["x"], far["y"]) < 0.3 for o in linked)
    check(has_far, "walk reaches Morse O at (π,0), not only the nearest decoy")
    print("  nearest O (%.3f,%.3f); Morse far O (%.3f,%.3f); linked %s" % (
        nearest_o["x"], nearest_o["y"], far["x"], far["y"],
        [(round(o["x"], 3), round(o["y"], 3), o["kind"]) for o in linked]))


def test_fourier_hessian_not_bilinear_artifact():
    print("\n== Fourier Hessian on sin x sin y is well-conditioned ==")
    for n in (32, 64, 128):
        a = mt.sample_grid(lambda x, y: np.sin(x) * np.sin(y), n)
        pts = mt.locate_critical_points(a)
        xs = [p for p in pts if p["kind"] == mt.KIND_X]
        check(len(xs) == 4, "N=%d still 4 X" % n)
        for p in xs:
            check(p["eig_ratio"] > 0.2, "N=%d X eig_ratio=%.3f not degenerate" % (
                n, p["eig_ratio"]))
            check(p["grad2"] < 1.0e-12, "N=%d Fourier |∇a|²=%.3e at X" % (n, p["grad2"]))


def test_magnetic_not_morse_sinxsiny():
    print("\n== magnetic a=a_X connectivity ≠ Morse ±∇a (sin x sin y) ==")
    for n in (32, 64, 128):
        a = mt.sample_grid(lambda x, y: np.sin(x) * np.sin(y), n)
        pts = mt.locate_critical_points(a)
        xs = [p for p in pts if p["kind"] == mt.KIND_X]
        os_ = [p for p in pts if p["kind"] in (mt.KIND_OMAX, mt.KIND_OMIN)]
        x0 = nearest(xs, 0.0, 0.0)
        rays = mt.levelset_rays(x0)
        check(len(rays) == 4, "N=%d four level-set rays, got %d" % (n, len(rays)))
        # Level-set rays are the axes, Morse eigenvectors the diagonals.
        axis_like = 0
        for vx, vy in rays:
            if abs(vx) < 0.25 or abs(vy) < 0.25:
                axis_like += 1
        check(axis_like == 4, "N=%d magnetic rays are axial, not diagonal" % n)
        mag, _, _ = mt.magnetic_connectivity(pts, a)
        node = min(mag, key=lambda g: mt.periodic_dist(g["x"], g["y"], 0, 0))
        hits = []
        for b in node["branches"]:
            if b["hit_x"] is None:
                continue
            hits.append(b["hit_x_pos"])
        check(len(hits) >= 2, "N=%d X(0,0) magnetic branches hit other X" % n)
        hit_o = False
        for hx, hy in hits:
            for o in os_:
                if mt.periodic_dist(hx, hy, o["x"], o["y"]) < 0.4:
                    hit_o = True
        check(not hit_o, "N=%d magnetic hits are X-points, not O-points" % n)
        # Morse goes to O; magnetic does not.
        pairs, xs2, os2 = mt.pair_morse(pts, a)
        xi = min(range(len(xs2)),
                 key=lambda i: mt.periodic_dist(xs2[i]["x"], xs2[i]["y"], 0, 0))
        morse_o = len(pairs[xi]["o_indices"]) >= 1
        check(morse_o, "N=%d Morse still reaches an O (graphs differ)" % n)
        # O_max at (π/2,π/2) enclosed by a=0 separatrix square.
        enclosed_max = False
        for g in mag:
            for e in g["enclosed_O"]:
                if e["kind"] == mt.KIND_OMAX and mt.periodic_dist(
                        e["x"], e["y"], 0.5 * math.pi, 0.5 * math.pi) < 0.4:
                    enclosed_max = True
        check(enclosed_max, "N=%d O_max at (π/2,π/2) enclosed by a=a_X" % n)


def test_ot_t0_multigraph_faces():
    print("\n== OT t=0 a=0.5 cos 2x + cos y is a separatrix multigraph ==")
    for n in (64, 128):
        a = mt.sample_grid(
            lambda x, y: 0.5 * np.cos(2.0 * x) + np.cos(y), n)
        pts = mt.locate_critical_points(a)
        xs = [p for p in pts if p["kind"] == mt.KIND_X]
        omax = [p for p in pts if p["kind"] == mt.KIND_OMAX]
        omin = [p for p in pts if p["kind"] == mt.KIND_OMIN]
        check(len(xs) == 4, "N=%d OT has 4 X, got %d" % (n, len(xs)))
        check(len(omax) == 2 and len(omin) == 2,
              "N=%d OT has 2 O_max and 2 O_min" % n)
        for p in omax:
            check(abs(p["a"] - 1.5) < 0.05, "N=%d O_max a=1.5 got %.4f" % (n, p["a"]))
        for p in omin:
            check(abs(p["a"] + 1.5) < 0.05, "N=%d O_min a=-1.5 got %.4f" % (n, p["a"]))
        avals = sorted(p["a"] for p in xs)
        check(abs(avals[0] + 0.5) < 0.05 and abs(avals[-1] - 0.5) < 0.05,
              "N=%d X critical values ±0.5" % n)
        mag, _, _ = mt.magnetic_connectivity(pts, a)
        # Parallel branches: some X pair must have n_edges > unique neighbours.
        parallel = False
        for g in mag:
            if g["n_edges"] > len(g["connects_x"]):
                parallel = True
        check(parallel,
              "N=%d OT has parallel X–X separatrix edges (simple graph would drop them)"
              % n)
        enclosed_max = 0
        enclosed_min = 0
        da_max = []
        da_min = []
        for g in mag:
            for e in g["enclosed_O"]:
                if e["kind"] == mt.KIND_OMAX:
                    enclosed_max += 1
                    da_max.append(e["delta_a"])
                elif e["kind"] == mt.KIND_OMIN:
                    enclosed_min += 1
                    da_min.append(e["delta_a"])
        check(enclosed_max >= 2, "N=%d O_max regions enclosed, got %d" % (n, enclosed_max))
        check(enclosed_min >= 2, "N=%d O_min regions enclosed, got %d" % (n, enclosed_min))
        if da_max:
            check(all(abs(d - 1.0) < 0.08 for d in da_max),
                  "N=%d Delta_a for O_max is 1.5-0.5=1, got %s" % (n, da_max))
        if da_min:
            check(all(abs(d + 1.0) < 0.08 for d in da_min),
                  "N=%d Delta_a for O_min is -1.5-(-0.5)=-1, got %s" % (n, da_min))


def test_force_free_budget_is_O_diffusion():
    print("\n== force-free eigenmode: dF/dt = -Ez_O, Ez_X≈0 ==")
    eta = 0.05
    times = [0.0, 0.08, 0.16, 0.24, 0.32, 0.40]
    frames = []
    for t in times:
        a = mt.sample_grid(
            lambda x, y, tt=t: math.exp(-2.0 * eta * tt) * np.sin(x) * np.sin(y),
            48)
        frames.append(mt.locate_critical_points(a))
    x0 = nearest([p for p in frames[0] if p["kind"] == mt.KIND_X], 0.0, 0.0)
    o0 = nearest([p for p in frames[0] if p["kind"] == mt.KIND_OMAX],
                 0.5 * math.pi, 0.5 * math.pi)
    trx, tro = ifb.track_pair(frames, times, x0, o0)
    out = ifb.series_from_tracks(trx, tro, times, eta, t_max=1.0)
    check(out is not None, "force-free island series exists")
    s = out["summary"]
    check(s["rel_budget_E"] < 0.05, "dF/dt ≈ Ez_X-Ez_O  rel=%.3e" % s["rel_budget_E"])
    check(s["rel_budget_j"] < 0.05, "dF/dt ≈ eta(j_X-j_O) rel=%.3e" % s["rel_budget_j"])
    check(abs(s["mean_Ez_X"]) < 0.2 * abs(s["mean_dFdt"]),
          "Ez_X is small vs dF/dt (diffusion of O, not X reconnection)")
    check(s["mean_frac_Ez_O"] > 0.8,
          "most of dF/dt is -Ez_O, frac=%.3f" % s["mean_frac_Ez_O"])


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
    test_moving_multi_n_and_cadence()
    test_separatrix_not_nearest_o()
    test_fourier_hessian_not_bilinear_artifact()
    test_magnetic_not_morse_sinxsiny()
    test_ot_t0_multigraph_faces()
    test_force_free_budget_is_O_diffusion()
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

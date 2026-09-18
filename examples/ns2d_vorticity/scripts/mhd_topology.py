#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sub-grid periodic X/O tracker for 2-D magnetic flux a(x,y).

B = (∂y a, −∂x a), so magnetic nulls are ∇a = 0. This module locates those
nulls by Newton refinement inside cells, classifies them from the Hessian,
tracks them in time, and pairs X/O points by Morse (eigendirection) walks.

It is analysis-only. Agreement of Ez = −∂t a with η j at an X-point is a
resistive Ohm diagnostic, not by itself a reconnection claim.
"""

from __future__ import print_function

import math

import numpy as np

TWOPI = 2.0 * math.pi
KIND_X = "X"
KIND_OMAX = "O_max"
KIND_OMIN = "O_min"
KIND_DEGEN = "degenerate"


def wrap(x, length=TWOPI):
    return np.mod(x, length)


def periodic_delta(a, b, length=TWOPI):
    d = wrap(a - b, length)
    return np.where(d > 0.5 * length, d - length, d)


def periodic_dist(x0, y0, x1, y1, length=TWOPI):
    dx = float(periodic_delta(x0, x1, length))
    dy = float(periodic_delta(y0, y1, length))
    return math.hypot(dx, dy)


def bilinear(field, x, y, dx):
    """Interpolate a cell-centered field; nodes at (i+1/2) dx."""
    n = field.shape[0]
    s = x / dx - 0.5
    t = y / dx - 0.5
    i0 = int(math.floor(s)) % n
    j0 = int(math.floor(t)) % n
    fx = s - math.floor(s)
    fy = t - math.floor(t)
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    v00 = field[i0, j0]
    v10 = field[i1, j0]
    v01 = field[i0, j1]
    v11 = field[i1, j1]
    return (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (
        1.0 - fx
    ) * fy * v01 + fx * fy * v11


def spectral_derivs(a, length=TWOPI):
    """Periodic spectral ∂x, ∂y, Hessian of a cell-centered field."""
    n = a.shape[0]
    kx = 2.0 * math.pi * np.fft.fftfreq(n, d=length / n)
    ky = kx.copy()
    hat = np.fft.fft2(a)
    ax = np.fft.ifft2(1j * kx[:, None] * hat).real
    ay = np.fft.ifft2(1j * ky[None, :] * hat).real
    axx = np.fft.ifft2(-(kx[:, None] ** 2) * hat).real
    ayy = np.fft.ifft2(-(ky[None, :] ** 2) * hat).real
    axy = np.fft.ifft2(-kx[:, None] * ky[None, :] * hat).real
    return ax, ay, axx, ayy, axy


def _classify(axx, ayy, axy, degen_ratio=0.05):
    det = axx * ayy - axy * axy
    tr = axx + ayy
    disc = tr * tr - 4.0 * det
    if disc < 0.0:
        disc = 0.0
    lam1 = 0.5 * (tr + math.sqrt(disc))
    lam2 = 0.5 * (tr - math.sqrt(disc))
    mag = max(abs(lam1), abs(lam2), 1.0e-30)
    ratio = min(abs(lam1), abs(lam2)) / mag
    cond = mag / max(min(abs(lam1), abs(lam2)), 1.0e-30)
    degenerate = ratio < degen_ratio
    if det < 0.0:
        kind = KIND_DEGEN if degenerate else KIND_X
    elif axx < 0.0 or tr < 0.0:
        kind = KIND_DEGEN if degenerate else KIND_OMAX
    else:
        kind = KIND_DEGEN if degenerate else KIND_OMIN
    return kind, det, ratio, cond, lam1, lam2


def _eigvecs(axx, ayy, axy, lam1, lam2):
    def vec(lam):
        # (H - lam I) v = 0
        if abs(axy) > 1.0e-14:
            vx, vy = axy, lam - axx
        elif abs(axx - lam) <= abs(ayy - lam):
            vx, vy = 1.0, 0.0
        else:
            vx, vy = 0.0, 1.0
        nrm = math.hypot(vx, vy)
        if nrm < 1.0e-30:
            return 1.0, 0.0
        return vx / nrm, vy / nrm

    return vec(lam1), vec(lam2)


def locate_critical_points(a, j=None, length=TWOPI, max_newton=12, degen_ratio=0.05):
    """Return sub-grid critical points of a on the periodic square."""
    n = a.shape[0]
    dx = length / n
    ax, ay, axx, ayy, axy = spectral_derivs(a, length)
    g2 = ax * ax + ay * ay
    local_min = np.ones((n, n), dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            local_min &= g2 <= np.roll(np.roll(g2, di, 0), dj, 1)
    # Seed Newton from every local |∇a|² minimum; duplicates collapse later.
    seeds = np.argwhere(local_min)
    pts = []
    for i0, j0 in seeds:
        x = (i0 + 0.5) * dx
        y = (j0 + 0.5) * dx
        ok = False
        for _ in range(max_newton):
            gx = bilinear(ax, x, y, dx)
            gy = bilinear(ay, x, y, dx)
            hxx = bilinear(axx, x, y, dx)
            hyy = bilinear(ayy, x, y, dx)
            hxy = bilinear(axy, x, y, dx)
            det = hxx * hyy - hxy * hxy
            if abs(det) < 1.0e-18:
                break
            dxn = (-gx * hyy + gy * hxy) / det
            dyn = (-gy * hxx + gx * hxy) / det
            x = wrap(x + dxn, length)
            y = wrap(y + dyn, length)
            if dxn * dxn + dyn * dyn < 1.0e-16 * dx * dx:
                ok = True
                break
        gx = bilinear(ax, x, y, dx)
        gy = bilinear(ay, x, y, dx)
        if (not ok) and (gx * gx + gy * gy > 1.0e-6):
            continue
        hxx = bilinear(axx, x, y, dx)
        hyy = bilinear(ayy, x, y, dx)
        hxy = bilinear(axy, x, y, dx)
        kind, det, ratio, cond, lam1, lam2 = _classify(hxx, hyy, hxy, degen_ratio)
        rec = {
            "kind": kind,
            "x": float(x),
            "y": float(y),
            "a": float(bilinear(a, x, y, dx)),
            "grad2": float(gx * gx + gy * gy),
            "hess_det": float(det),
            "eig_ratio": float(ratio),
            "hess_cond": float(cond),
            "lam1": float(lam1),
            "lam2": float(lam2),
            "j": None if j is None else float(bilinear(j, x, y, dx)),
        }
        rec["v1"], rec["v2"] = _eigvecs(hxx, hyy, hxy, lam1, lam2)
        pts.append(rec)
    # Collapse Newton duplicates (periodic).
    pts.sort(key=lambda p: p["grad2"])
    uniq = []
    for p in pts:
        if any(periodic_dist(p["x"], p["y"], q["x"], q["y"], length) < 0.35 * dx
               for q in uniq):
            continue
        uniq.append(p)
    uniq.sort(key=lambda p: (p["kind"], p["a"], p["x"], p["y"]))
    return uniq


def _walk_to_extremum(a, start, direction, extrema, length=TWOPI, nstep=80):
    n = a.shape[0]
    dx = length / n
    ax, ay, _, _, _ = spectral_derivs(a, length)
    x, y = start
    vx, vy = direction
    for _ in range(nstep):
        gx = bilinear(ax, x, y, dx)
        gy = bilinear(ay, x, y, dx)
        # Follow ±∇a aligned with the current direction.
        dot = gx * vx + gy * vy
        if abs(dot) < 1.0e-18:
            break
        if dot < 0.0:
            gx, gy = -gx, -gy
        gn = math.hypot(gx, gy)
        if gn < 1.0e-14:
            break
        x = wrap(x + 0.6 * dx * gx / gn, length)
        y = wrap(y + 0.6 * dx * gy / gn, length)
        vx, vy = gx / gn, gy / gn
        for k, o in enumerate(extrema):
            if periodic_dist(x, y, o["x"], o["y"], length) < 1.5 * dx:
                return k
    # Fallback: nearest extremum.
    if not extrema:
        return None
    return min(range(len(extrema)),
               key=lambda k: periodic_dist(x, y, extrema[k]["x"], extrema[k]["y"],
                                           length))


def pair_morse(points, a, length=TWOPI):
    """Pair each X-point with O-points reached along Hessian eigendirections."""
    xs = [p for p in points if p["kind"] == KIND_X]
    os_ = [p for p in points if p["kind"] in (KIND_OMAX, KIND_OMIN)]
    pairs = []
    for xi, xpt in enumerate(xs):
        ends = []
        for vec in (xpt["v1"], xpt["v2"]):
            for sign in (+1.0, -1.0):
                k = _walk_to_extremum(
                    a, (xpt["x"], xpt["y"]),
                    (sign * vec[0], sign * vec[1]), os_, length)
                if k is not None:
                    ends.append(k)
        uniq_ends = sorted(set(ends))
        pairs.append({
            "x_index": xi,
            "o_indices": uniq_ends,
            "n_links": len(uniq_ends),
        })
    return pairs, xs, os_


def track_points(frames, max_speed=None, length=TWOPI):
    """Greedy periodic tracking. frames is a list of point-lists in time.

    Returns tracks (list of dicts) and events (birth/death). Merge/split are
    reported only when two deaths/births coincide spatially with a degenerate
    Hessian; otherwise they stay birth/death.
    """
    tracks = []
    events = []
    next_id = 0

    def new_track(p, t_index):
        nonlocal next_id
        tid = next_id
        next_id += 1
        tracks.append({
            "id": tid,
            "kind": p["kind"],
            "history": [(t_index, dict(p))],
            "alive": True,
        })
        events.append({"type": "birth", "id": tid, "t_index": t_index, "p": p})
        return tid

    if not frames:
        return tracks, events
    for p in frames[0]:
        new_track(p, 0)

    for t in range(1, len(frames)):
        prev = [(tr, tr["history"][-1][1]) for tr in tracks if tr["alive"]]
        curr = list(frames[t])
        used_prev = set()
        used_curr = set()
        # Cost matrix: periodic distance; forbid kind changes except degenerate.
        cands = []
        for ip, (tr, pp) in enumerate(prev):
            for ic, pc in enumerate(curr):
                d = periodic_dist(pp["x"], pp["y"], pc["x"], pc["y"], length)
                if max_speed is not None and d > max_speed:
                    continue
                kind_ok = (pp["kind"] == pc["kind"]
                           or KIND_DEGEN in (pp["kind"], pc["kind"]))
                if not kind_ok:
                    continue
                cands.append((d, ip, ic))
        cands.sort()
        for d, ip, ic in cands:
            if ip in used_prev or ic in used_curr:
                continue
            used_prev.add(ip)
            used_curr.add(ic)
            tr, _ = prev[ip]
            pc = curr[ic]
            tr["history"].append((t, dict(pc)))
            if pc["kind"] != KIND_DEGEN:
                tr["kind"] = pc["kind"]
        for ip, (tr, pp) in enumerate(prev):
            if ip in used_prev:
                continue
            tr["alive"] = False
            events.append({
                "type": "death",
                "id": tr["id"],
                "t_index": t,
                "p": pp,
            })
        for ic, pc in enumerate(curr):
            if ic in used_curr:
                continue
            new_track(pc, t)

        deaths = [e for e in events if e["type"] == "death" and e["t_index"] == t]
        births = [e for e in events if e["type"] == "birth" and e["t_index"] == t]
        degen_now = [p for p in curr if p["kind"] == KIND_DEGEN]
        thresh = length / 32.0
        for dth in deaths:
            for bth in births:
                if periodic_dist(dth["p"]["x"], dth["p"]["y"],
                                 bth["p"]["x"], bth["p"]["y"], length) < thresh:
                    events.append({
                        "type": "possible_replace",
                        "t_index": t,
                        "death_id": dth["id"],
                        "birth_id": bth["id"],
                    })
        if degen_now and (deaths or births):
            events.append({
                "type": "degenerate_hessian",
                "t_index": t,
                "n_degen": len(degen_now),
                "n_death": len(deaths),
                "n_birth": len(births),
            })
    return tracks, events


def sample_grid(func, n, length=TWOPI):
    dx = length / n
    i = (np.arange(n) + 0.5) * dx
    x = i[:, None]
    y = i[None, :]
    return func(x, y)


def da_dt_from_track(history, times):
    """Central difference of a along a track; one-sided at endpoints."""
    a = np.array([h[1]["a"] for h in history], dtype=float)
    t_idx = [h[0] for h in history]
    t = np.array([times[i] for i in t_idx], dtype=float)
    dadt = np.zeros_like(a)
    if len(a) == 1:
        return t, a, dadt
    dadt[0] = (a[1] - a[0]) / max(t[1] - t[0], 1.0e-30)
    dadt[-1] = (a[-1] - a[-2]) / max(t[-1] - t[-2], 1.0e-30)
    for k in range(1, len(a) - 1):
        dadt[k] = (a[k + 1] - a[k - 1]) / max(t[k + 1] - t[k - 1], 1.0e-30)
    return t, a, dadt

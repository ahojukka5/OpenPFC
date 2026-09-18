#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline magnetic critical-point finder for 2-D flux a(x,y).

B = (∂y a, −∂x a), so B=0 where ∇a=0. Classification uses the Hessian of a:

* det H < 0: saddle (candidate X-point)
* det H > 0 and a_xx < 0: local max (candidate O-point)
* det H > 0 and a_xx > 0: local min (candidate O-point)

This is analysis only. It does not define a reconnection rate.

Example:
  python3 find_mhd_critical_points.py --self-test
  python3 find_mhd_critical_points.py --dir ... --n 256 --inc 228 --json-out pts.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np


def load_brick(path: str, n: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == n * n:
        return raw.reshape((n, n), order="F")
    if raw.size == n * n * 1:
        return raw.reshape((n, n, 1), order="F")[:, :, 0]
    raise ValueError(f"{path}: got {raw.size} doubles, expected {n * n}")


def periodic_grad(a: np.ndarray, dx: float):
    ax = (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / (2.0 * dx)
    ay = (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / (2.0 * dx)
    return ax, ay


def hessian(a: np.ndarray, dx: float):
    axx = (np.roll(a, -1, 0) - 2.0 * a + np.roll(a, 1, 0)) / (dx * dx)
    ayy = (np.roll(a, -1, 1) - 2.0 * a + np.roll(a, 1, 1)) / (dx * dx)
    axy = (
        np.roll(np.roll(a, -1, 0), -1, 1)
        - np.roll(np.roll(a, -1, 0), 1, 1)
        - np.roll(np.roll(a, 1, 0), -1, 1)
        + np.roll(np.roll(a, 1, 0), 1, 1)
    ) / (4.0 * dx * dx)
    return axx, ayy, axy


def find_critical_points(a: np.ndarray, j: np.ndarray | None = None) -> list[dict]:
    n = a.shape[0]
    dx = 2.0 * math.pi / n
    ax, ay = periodic_grad(a, dx)
    g2 = ax * ax + ay * ay
    axx, ayy, axy = hessian(a, dx)
    det = axx * ayy - axy * axy
    # Local minima of |∇a|^2, including periodic neighbors.
    local_min = np.ones(a.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            local_min &= g2 <= np.roll(np.roll(g2, di, 0), dj, 1)
    # Keep only the smallest |∇a| in the grid, plus a modest tail.
    # Exact CPs need not sit on nodes; |∇a|² at the nearest cell is O(dx²).
    # Keep the tight cluster of local minima, not a fraction of max|∇a|².
    if np.any(local_min):
        g_min = float(np.min(g2[local_min]))
    else:
        g_min = float(np.min(g2))
    g_cut = max(g_min * 8.0, 1.0e-12)
    mask = local_min & (g2 <= g_cut)
    pts = []
    seen = set()
    for i, jj in zip(*np.where(mask)):
        # Dedup 2-cell clusters: keep the smallest g2 in a 3x3.
        best_i, best_j = i, jj
        best = g2[i, jj]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii = (i + di) % n
                j2 = (jj + dj) % n
                if g2[ii, j2] < best:
                    best = g2[ii, j2]
                    best_i, best_j = ii, j2
        key = (int(best_i), int(best_j))
        if key in seen:
            continue
        seen.add(key)
        hdet = float(det[best_i, best_j])
        hxx = float(axx[best_i, best_j])
        if hdet < 0.0:
            kind = "X"
        elif hxx < 0.0:
            kind = "O_max"
        else:
            kind = "O_min"
        rec = {
            "kind": kind,
            "ix": key[0],
            "iy": key[1],
            "x": (key[0] + 0.5) * dx,
            "y": (key[1] + 0.5) * dx,
            "a": float(a[best_i, best_j]),
            "grad2": float(g2[best_i, best_j]),
            "hess_det": hdet,
            "j_val": float(j[best_i, best_j]) if j is not None else None,
        }
        pts.append(rec)
    pts.sort(key=lambda p: (p["kind"], p["a"]))
    return pts


def synthetic_a(n: int) -> np.ndarray:
    dx = 2.0 * math.pi / n
    i = (np.arange(n) + 0.5) * dx
    x = i[:, None]
    y = i[None, :]
    return np.sin(x) * np.sin(y)


def self_test() -> int:
    a = synthetic_a(64)
    pts = find_critical_points(a)
    xs = [p for p in pts if p["kind"] == "X"]
    os_ = [p for p in pts if p["kind"].startswith("O")]
    if len(xs) < 4 or len(os_) < 4:
        print("FAIL: expected >=4 X and >=4 O, got", len(xs), len(os_), file=sys.stderr)
        return 1
    # Exact saddles of sin x sin y: (n π, m π). Extrema: odd multiples of π/2.
    def near(p, x, y, tol=2.0 * math.pi / 64 * 2.5):
        dx = abs(p["x"] - x)
        dy = abs(p["y"] - y)
        dx = min(dx, 2.0 * math.pi - dx)
        dy = min(dy, 2.0 * math.pi - dy)
        return math.hypot(dx, dy) < tol

    saddles = [(0.0, 0.0), (math.pi, 0.0), (0.0, math.pi), (math.pi, math.pi)]
    extrema = [
        (0.5 * math.pi, 0.5 * math.pi),
        (1.5 * math.pi, 1.5 * math.pi),
        (0.5 * math.pi, 1.5 * math.pi),
        (1.5 * math.pi, 0.5 * math.pi),
    ]
    for x, y in saddles:
        if not any(near(p, x, y) for p in xs):
            print("FAIL: missing X near", x, y, file=sys.stderr)
            return 1
    for x, y in extrema:
        if not any(near(p, x, y) for p in os_):
            print("FAIL: missing O near", x, y, file=sys.stderr)
            return 1
    print(f"SELF-TEST PASS  X={len(xs)} O={len(os_)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir", default="")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--inc", type=int, default=-1)
    p.add_argument("--prev-inc", type=int, default=-1)
    p.add_argument("--dt-dump", type=float, default=0.0)
    p.add_argument("--json-out", default="")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not args.dir or args.n <= 0 or args.inc < 0:
        p.error("--dir --n --inc are required unless --self-test")
    a = load_brick(os.path.join(args.dir, f"a_{args.inc:04d}.bin"), args.n)
    jpath = os.path.join(args.dir, f"j_{args.inc:04d}.bin")
    j = load_brick(jpath, args.n) if os.path.exists(jpath) else None
    pts = find_critical_points(a, j)
    ez = None
    if args.prev_inc >= 0 and args.dt_dump > 0.0:
        a_prev = load_brick(os.path.join(args.dir, f"a_{args.prev_inc:04d}.bin"), args.n)
        ez = -(a - a_prev) / args.dt_dump
        for rec in pts:
            rec["Ez"] = float(ez[rec["ix"], rec["iy"]])
            if rec["j_val"] is not None:
                rec["eta_j_proxy"] = rec["j_val"]
    nX = sum(1 for r in pts if r["kind"] == "X")
    nO = sum(1 for r in pts if r["kind"].startswith("O"))
    print(f"inc={args.inc} n={args.n} X={nX} O={nO}")
    for rec in pts:
        extra = ""
        if rec.get("Ez") is not None:
            extra = f"  Ez={rec['Ez']:.4e}"
        print(
            f"  {rec['kind']:5s}  x={rec['x']:.4f} y={rec['y']:.4f}  "
            f"a={rec['a']:.6g}  j={rec['j_val']}  |grad|^2={rec['grad2']:.3e}{extra}"
        )
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"inc": args.inc, "n": args.n, "points": pts}, fh, indent=2)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

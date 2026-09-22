#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact constant-mobility forced bending+tension trajectory.

Evaluates the discrete Fourier solution of

    p = B nabla^4 u - gamma nabla^2 u + p_ext,
    d_t u = M0 nabla^2 p,

on the same periodic grid and sampled Gaussian load as ``ehd_film_nonlinear``.
Matches ``ehd_film/linear_oracle.hpp``. This is the paper comparator for
; it does not run OpenPFC.

    python3 apps/ehd_film/scripts/forced_linear_oracle.py \\
        --B 100 --out linear_B100.csv
"""

import argparse
import csv
import math
import sys

import numpy as np


def gaussian_load(n, dx, p0, a):
    x = np.arange(n, dtype=float) * dx
    cx = cy = 0.5 * n * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    pext = p0 * np.exp(-r2 / (2.0 * a * a))
    return X, Y, pext


def wavenumbers(n, dx):
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return kx ** 2 + ky ** 2


def deflection_field(p_hat, k2, B, gamma, M0, t, t_load):
    lam = M0 * (B * k2 ** 3 + gamma * k2 ** 2)
    stiff = k2 * (B * k2 + gamma)
    U = np.zeros_like(p_hat)
    mask = k2 > 0.0
    t_on = min(t, t_load) if t > 0.0 else 0.0
    if t_on > 0.0:
        U[mask] = -p_hat[mask] / stiff[mask] * (1.0 - np.exp(-lam[mask] * t_on))
        if t > t_load:
            U[mask] *= np.exp(-lam[mask] * (t - t_load))
    return np.real(np.fft.ifft2(U))


def observables(u, X, Y, h0, dx):
    h = h0 + u
    n = h.shape[0]
    ic = n // 2
    cx = cy = 0.5 * n * dx
    w = (h - h0) ** 2
    den = float(np.sum(w))
    if den > 0.0:
        spreading = math.sqrt(
            float(np.sum(w * ((X - cx) ** 2 + (Y - cy) ** 2))) / den
        )
    else:
        spreading = 0.0
    cell = dx * dx
    return {
        "h_center": float(h[ic, ic]),
        "deflection_center": float(h0 - h[ic, ic]),
        "spreading_radius": spreading,
        "displaced_volume": float(np.sum(np.maximum(0.0, h0 - h)) * cell),
        "volume": float(np.sum(h) * cell),
        "h_min": float(np.min(h)),
        "h_max": float(np.max(h)),
    }


def times(t1, saveat):
    n = int(round(t1 / saveat))
    for i in range(n + 1):
        yield i * saveat


def write_csv(path, rows, extra):
    fieldnames = [
        "time",
        "B",
        "gamma",
        "p0",
        "load_width",
        "t_load",
        "N",
        "h_center",
        "deflection_center",
        "spreading_radius",
        "displaced_volume",
        "volume",
        "volume_rel_drift",
        "h_min",
        "h_max",
    ]
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = dict(extra)
            out.update(row)
            w.writerow(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--dx", type=float, default=1.0)
    p.add_argument("--h0", type=float, default=1.0)
    p.add_argument("--M0", type=float, default=1.0)
    p.add_argument("--B", type=float, required=True)
    p.add_argument("--gamma", type=float, default=10.0)
    p.add_argument("--p0", type=float, default=0.5)
    p.add_argument("--a", type=float, default=8.0)
    p.add_argument("--t-load", type=float, default=60.0)
    p.add_argument("--t1", type=float, default=600.0)
    p.add_argument("--saveat", type=float, default=10.0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    X, Y, pext = gaussian_load(args.N, args.dx, args.p0, args.a)
    p_hat = np.fft.fft2(pext)
    k2 = wavenumbers(args.N, args.dx)
    volume0 = None
    rows = []
    for t in times(args.t1, args.saveat):
        u = deflection_field(
            p_hat, k2, args.B, args.gamma, args.M0, t, args.t_load
        )
        obs = observables(u, X, Y, args.h0, args.dx)
        if volume0 is None:
            volume0 = obs["volume"]
        obs["time"] = t
        obs["volume_rel_drift"] = (
            (obs["volume"] - volume0) / volume0 if volume0 else 0.0
        )
        rows.append(obs)

    extra = {
        "B": args.B,
        "gamma": args.gamma,
        "p0": args.p0,
        "load_width": args.a,
        "t_load": args.t_load,
        "N": args.N,
    }
    write_csv(args.out, rows, extra)
    t60 = next(r for r in rows if abs(r["time"] - args.t_load) < 1e-12)
    t1 = rows[-1]
    sys.stderr.write(
        "linear oracle B=%g p0=%g N=%d: t=%g defl=%.6f R=%.4f; "
        "t=%g defl=%.6f R=%.4f -> %s\n"
        % (
            args.B,
            args.p0,
            args.N,
            args.t_load,
            t60["deflection_center"],
            t60["spreading_radius"],
            args.t1,
            t1["deflection_center"],
            t1["spreading_radius"],
            args.out,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

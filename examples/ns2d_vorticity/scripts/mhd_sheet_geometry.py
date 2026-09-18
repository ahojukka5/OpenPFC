#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frozen local current-sheet geometry at a persistent reconnection X-point.

Definitions are locked before any eta ladder (issue #38). Do not retune
ALPHA, KAPPA or R_ST per resistivity.

Orientation
  n_hat : sheet normal = eigenvector of the most negative Hessian of j
          at the X-point (Fourier). Fallback: largest-eigenvalue
          direction of the |j| structure tensor in a disk of radius R_ST.
  t_hat : (-n_y, n_x)

Thickness delta
  Sub-grid FWHM of |j| along n_hat through the X-point. Physical length.

Length L
  Extent along t_hat where |j| >= ALPHA * |j|_X, ALPHA = 0.5.
  Not sqrt(half-max area).

Upstream field B_up
  Reconnecting component B·t_hat sampled at n = ± KAPPA * delta,
  KAPPA = 2. Mean of the two absolute values.

V_A = B_up (density 1)
S_local = L * V_A / eta
R = |Ez_X| / (B_up * V_A) = |Ez_X| / B_up^2
"""

from __future__ import print_function

import math

import numpy as np

import mhd_topology as mt

# Frozen. Do not change between eta values.
ALPHA_J = 0.5
KAPPA_UP = 2.0
R_ST = 0.5
N_LINE = 401
S_MAX_N = 0.5 * math.pi
S_MAX_T = math.pi


def _unit(vx, vy):
    nrm = math.hypot(vx, vy)
    if nrm < 1.0e-30:
        return 1.0, 0.0
    return vx / nrm, vy / nrm


def _hess_eigs(hxx, hxy, hyy):
    tr = hxx + hyy
    det = hxx * hyy - hxy * hxy
    disc = max(tr * tr - 4.0 * det, 0.0)
    l1 = 0.5 * (tr + math.sqrt(disc))
    l2 = 0.5 * (tr - math.sqrt(disc))
    v1, v2 = mt._eigvecs(hxx, hyy, hxy, l1, l2)
    return (l1, v1), (l2, v2)


def sheet_frame(j, x, y, length=mt.TWOPI):
    """Return n_hat, t_hat at (x,y) from the Hessian of j."""
    n = j.shape[0]
    dx = length / float(n)
    hat = np.fft.fft2(j)
    kx = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    _, _, _, jxx, jyy, jxy = mt.fourier_point(
        j, x, y, length, hat=hat, kx=kx, ky=kx)
    (l1, v1), (l2, v2) = _hess_eigs(jxx, jxy, jyy)
    # Most negative eigenvalue: thinner (normal) direction.
    if l2 <= l1:
        nx, ny = v2
        lam_n, lam_t = l2, l1
    else:
        nx, ny = v1
        lam_n, lam_t = l1, l2
    if abs(lam_n - lam_t) < 1.0e-8 * max(abs(lam_n), abs(lam_t), 1.0):
        nx, ny = _structure_normal(j, x, y, dx, length)
    nx, ny = _unit(nx, ny)
    tx, ty = -ny, nx
    return (nx, ny), (tx, ty)


def _structure_normal(j, x, y, dx, length):
    """Fallback: ∇|j| structure tensor in a physical disk of radius R_ST."""
    n = j.shape[0]
    jx, jy, _, _, _ = mt.spectral_derivs(j, length)
    accxx = accxy = accyy = 0.0
    wsum = 0.0
    i0 = int(round(x / dx - 0.5)) % n
    j0 = int(round(y / dx - 0.5)) % n
    rad = max(1, int(R_ST / dx))
    for di in range(-rad, rad + 1):
        for dj in range(-rad, rad + 1):
            xi = (i0 + di) % n
            yj = (j0 + dj) % n
            px = (xi + 0.5) * dx
            py = (yj + 0.5) * dx
            if mt.periodic_dist(px, py, x, y, length) > R_ST:
                continue
            gx = jx[xi, yj]
            gy = jy[xi, yj]
            accxx += gx * gx
            accxy += gx * gy
            accyy += gy * gy
            wsum += 1.0
    if wsum < 4.0:
        return 1.0, 0.0
    (l1, v1), (l2, v2) = _hess_eigs(accxx / wsum, accxy / wsum, accyy / wsum)
    # Largest eigenvalue of ∇j⊗∇j is the normal.
    if l1 >= l2:
        return v1
    return v2


def _sample_line(field, x0, y0, ux, uy, s_max, n_samp, dx, length,
                 hat=None, kx=None):
    s = np.linspace(-s_max, s_max, n_samp)
    vals = np.empty(n_samp)
    if hat is None:
        hat = np.fft.fft2(field)
        kx = 2.0 * math.pi * np.fft.fftfreq(field.shape[0], d=dx)
    for i, si in enumerate(s):
        x = mt.wrap(x0 + si * ux, length)
        y = mt.wrap(y0 + si * uy, length)
        val, _, _, _, _, _ = mt.fourier_point(
            field, x, y, length, hat=hat, kx=kx, ky=kx)
        vals[i] = val
    return s, vals


def _width_at_level(s, f, level):
    """Extent of {s : |f| >= level}, with linear edge interpolation."""
    g = np.abs(f)
    above = np.where(g >= level)[0]
    if above.size == 0:
        return 0.0
    i0 = int(above[0])
    i1 = int(above[-1])
    if i0 > 0 and g[i0] != g[i0 - 1]:
        t = (level - g[i0 - 1]) / (g[i0] - g[i0 - 1])
        sL = s[i0 - 1] + t * (s[i0] - s[i0 - 1])
    else:
        sL = s[i0]
    if i1 + 1 < g.size and g[i1] != g[i1 + 1]:
        t = (level - g[i1]) / (g[i1 + 1] - g[i1])
        sR = s[i1] + t * (s[i1 + 1] - s[i1])
    else:
        sR = s[i1]
    return float(max(0.0, sR - sL))


def fwhm_line(s, f):
    peak = float(np.max(np.abs(f)))
    if peak <= 0.0:
        return 0.0
    return _width_at_level(s, f, 0.5 * peak)


def measure_sheet(a, x, y, eta, ez_x=None, length=mt.TWOPI):
    """Local sheet geometry and normalized rate at a point (x,y)."""
    n = a.shape[0]
    dx = length / float(n)
    ax, ay, axx, ayy, axy = mt.spectral_derivs(a, length)
    j = -(axx + ayy)
    bx = ay
    by = -ax
    n_hat, t_hat = sheet_frame(np.abs(j), x, y, length)
    jhat = np.fft.fft2(j)
    kx = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    j_x, _, _, _, _, _ = mt.fourier_point(
        j, x, y, length, hat=jhat, kx=kx, ky=kx)
    j_x = float(j_x)
    s_n, j_n = _sample_line(
        j, x, y, n_hat[0], n_hat[1], S_MAX_N, N_LINE, dx, length,
        hat=jhat, kx=kx)
    s_t, j_t = _sample_line(
        j, x, y, t_hat[0], t_hat[1], S_MAX_T, N_LINE, dx, length,
        hat=jhat, kx=kx)
    delta = fwhm_line(s_n, j_n)
    jref = abs(j_x) if abs(j_x) > 1.0e-30 else float(np.max(np.abs(j_n)))
    L = _width_at_level(s_t, j_t, ALPHA_J * jref)
    # B_up at ± KAPPA * delta along n_hat.
    d_up = KAPPA_UP * max(delta, dx)
    sides = []
    for sgn in (+1.0, -1.0):
        px = mt.wrap(x + sgn * d_up * n_hat[0], length)
        py = mt.wrap(y + sgn * d_up * n_hat[1], length)
        bpx = float(mt.bilinear(bx, px, py, dx))
        bpy = float(mt.bilinear(by, px, py, dx))
        sides.append(abs(bpx * t_hat[0] + bpy * t_hat[1]))
    b_up = 0.5 * (sides[0] + sides[1])
    v_a = b_up
    s_local = (L * v_a / eta) if eta > 0.0 else float("inf")
    if ez_x is None:
        r = None
    elif b_up > 1.0e-12:
        r = abs(float(ez_x)) / (b_up * v_a)
    else:
        r = None
    return {
        "n_hat": n_hat,
        "t_hat": t_hat,
        "delta": delta,
        "L": L,
        "aspect": (L / delta) if delta > 1.0e-30 else None,
        "B_up": b_up,
        "B_up_plus": sides[0],
        "B_up_minus": sides[1],
        "sample_distance": d_up,
        "j_X": j_x,
        "V_A": v_a,
        "S_local": s_local,
        "R": r,
        "ALPHA_J": ALPHA_J,
        "KAPPA_UP": KAPPA_UP,
    }


def harris_a(x, y, theta, delta, b0=1.0):
    """Infinite Harris sheet: a = B0 δ log cosh(n/δ), n = -x sinθ + y cosθ."""
    nn = -x * math.sin(theta) + y * math.cos(theta)
    return b0 * delta * np.log(np.cosh(np.clip(nn / delta, -40.0, 40.0)))


def gaussian_j(x, y, theta, sig_n, sig_t, j0=1.0):
    nn = -x * np.sin(theta) + y * np.cos(theta)
    tt = x * np.cos(theta) + y * np.sin(theta)
    return j0 * np.exp(-0.5 * (nn / sig_n) ** 2 - 0.5 * (tt / sig_t) ** 2)

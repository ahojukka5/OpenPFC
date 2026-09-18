#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frozen local current-sheet geometry at a persistent reconnection X-point.

Definitions are locked before any eta ladder (issue #38). Do not retune
ALPHA, KAPPA or R_ST per resistivity.

Orientation
  q = sign(j_X) * j  (smooth signed ridge; never Fourier-differentiate |j|).
  n_hat : most negative Hessian eigenvector of q at the X-point.
          Fallback: largest-eigenvalue direction of the ∇q structure
          tensor in a disk of radius R_ST. If |j_X| is below J_X_MIN
          or both constructions are degenerate, fail closed.
  t_hat : (-n_y, n_x)

Thickness delta
  Sub-grid FWHM of the connected |j| >= 0.5 |j_X| component containing
  s=0 along n_hat. Remote superlevel lobes are ignored.

Length L
  Connected |j| >= ALPHA |j_X| component containing s=0 along t_hat,
  ALPHA = 0.5. Not sqrt(half-max area).

Upstream field B_up
  Reconnecting component B·t_hat at n = ± KAPPA * delta, KAPPA = 2,
  with B = (a_y, -a_x) from Fourier derivatives of a (not bilinear).
  Mean of the two absolute values.

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
B_UP_MIN = 1.0e-3
J_X_MIN = 1.0e-12
# Search-window length 2*S_MAX_*; local FWHM saturates only if the
# connected s=0 component itself reaches an endpoint.
DELTA_CAP = 2.0 * S_MAX_N
L_CAP = 2.0 * S_MAX_T


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


def signed_ridge(j, j_x):
    """Smooth signed ridge q = sign(j_X) * j. None if |j_X| is too small."""
    if abs(float(j_x)) < J_X_MIN:
        return None
    sgn = 1.0 if float(j_x) >= 0.0 else -1.0
    return sgn * j


def sheet_frame(q, x, y, length=mt.TWOPI):
    """Return n_hat, t_hat at (x,y) from the Hessian of the signed ridge q.

    q must be a smooth spectral field (sign(j_X)*j), never |j|. Returns
    (None, None) if the Hessian is degenerate and the structure-tensor
    fallback cannot be formed. Does not guess an axis.
    """
    if q is None:
        return None, None
    n = q.shape[0]
    dx = length / float(n)
    hat = np.fft.fft2(q)
    kx = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    _, _, _, qxx, qyy, qxy = mt.fourier_point(
        q, x, y, length, hat=hat, kx=kx, ky=kx)
    (l1, v1), (l2, v2) = _hess_eigs(qxx, qxy, qyy)
    # Most negative eigenvalue: thinner (normal) direction of a ridge.
    if l2 <= l1:
        nx, ny = v2
        lam_n, lam_t = l2, l1
    else:
        nx, ny = v1
        lam_n, lam_t = l1, l2
    if abs(lam_n - lam_t) < 1.0e-8 * max(abs(lam_n), abs(lam_t), 1.0):
        st = _structure_normal(q, x, y, dx, length)
        if st is None:
            return None, None
        nx, ny = st
    nx, ny = _unit(nx, ny)
    tx, ty = -ny, nx
    return (nx, ny), (tx, ty)


def _structure_normal(q, x, y, dx, length):
    """Fallback: ∇q structure tensor in a physical disk of radius R_ST."""
    n = q.shape[0]
    qx, qy, _, _, _ = mt.spectral_derivs(q, length)
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
            gx = qx[xi, yj]
            gy = qy[xi, yj]
            accxx += gx * gx
            accxy += gx * gy
            accyy += gy * gy
            wsum += 1.0
    if wsum < 4.0:
        return None
    (l1, v1), (l2, v2) = _hess_eigs(accxx / wsum, accxy / wsum, accyy / wsum)
    # Largest eigenvalue of ∇q⊗∇q is the normal.
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


def _linear_crossing(s0, g0, s1, g1, level):
    if g1 == g0:
        return float(s0)
    t = (level - g0) / (g1 - g0)
    return float(s0 + t * (s1 - s0))


def _width_at_level(s, f, level):
    """Width of the connected |f| >= level component containing s=0.

    Walks left/right from the sample nearest s=0 until the first
    threshold crossings and interpolates them. Separated superlevel
    lobes are ignored. capped is True only if *this* component reaches
    a search-window endpoint.
    """
    s = np.asarray(s, dtype=float)
    g = np.abs(np.asarray(f, dtype=float))
    if g.size == 0 or level <= 0.0:
        return 0.0, False
    ic = int(np.argmin(np.abs(s)))
    if g[ic] < level:
        return 0.0, False
    iL = ic
    while iL > 0 and g[iL - 1] >= level:
        iL -= 1
    iR = ic
    n = g.size
    while iR + 1 < n and g[iR + 1] >= level:
        iR += 1
    capped_L = (iL == 0)
    capped_R = (iR + 1 >= n)
    if capped_L:
        sL = float(s[0])
    else:
        sL = _linear_crossing(s[iL - 1], g[iL - 1], s[iL], g[iL], level)
    if capped_R:
        sR = float(s[-1])
    else:
        sR = _linear_crossing(s[iR], g[iR], s[iR + 1], g[iR + 1], level)
    return float(max(0.0, sR - sL)), bool(capped_L or capped_R)


def fwhm_line(s, f):
    """Connected FWHM about s=0 using the local amplitude |f(s=0)|."""
    s = np.asarray(s, dtype=float)
    f = np.asarray(f, dtype=float)
    ic = int(np.argmin(np.abs(s)))
    peak = abs(float(f[ic]))
    if peak <= 0.0:
        return 0.0
    width, _ = _width_at_level(s, f, 0.5 * peak)
    return width


def _fail_closed(j_x=None):
    return {
        "n_hat": None,
        "t_hat": None,
        "delta": None,
        "L": None,
        "aspect": None,
        "B_up": None,
        "B_up_plus": None,
        "B_up_minus": None,
        "sample_distance": None,
        "j_X": j_x,
        "V_A": None,
        "S_local": None,
        "R": None,
        "delta_capped": False,
        "L_capped": False,
        "sheet_ok": False,
        "orientation_ok": False,
        "ALPHA_J": ALPHA_J,
        "KAPPA_UP": KAPPA_UP,
    }


def measure_sheet(a, x, y, eta, ez_x=None, length=mt.TWOPI):
    """Local sheet geometry and normalized rate at a point (x,y)."""
    n = a.shape[0]
    dx = length / float(n)
    ax, ay, axx, ayy, axy = mt.spectral_derivs(a, length)
    j = -(axx + ayy)
    jhat = np.fft.fft2(j)
    kx = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    j_x, _, _, _, _, _ = mt.fourier_point(
        j, x, y, length, hat=jhat, kx=kx, ky=kx)
    j_x = float(j_x)
    q = signed_ridge(j, j_x)
    n_hat, t_hat = sheet_frame(q, x, y, length)
    if n_hat is None or t_hat is None:
        return _fail_closed(j_x)
    s_n, j_n = _sample_line(
        j, x, y, n_hat[0], n_hat[1], S_MAX_N, N_LINE, dx, length,
        hat=jhat, kx=kx)
    s_t, j_t = _sample_line(
        j, x, y, t_hat[0], t_hat[1], S_MAX_T, N_LINE, dx, length,
        hat=jhat, kx=kx)
    jref = abs(j_x)
    level = ALPHA_J * jref
    delta, delta_capped = _width_at_level(s_n, j_n, level)
    L, L_capped = _width_at_level(s_t, j_t, level)
    # B_up at ± KAPPA * delta along n_hat, from Fourier derivatives of a.
    d_up = KAPPA_UP * max(delta, dx)
    ahat = np.fft.fft2(a)
    sides = []
    for sgn in (+1.0, -1.0):
        px = mt.wrap(x + sgn * d_up * n_hat[0], length)
        py = mt.wrap(y + sgn * d_up * n_hat[1], length)
        _, apx, apy, _, _, _ = mt.fourier_point(
            a, px, py, length, hat=ahat, kx=kx, ky=kx)
        bpx = float(apy)
        bpy = float(-apx)
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
    sheet_ok = (not delta_capped) and (not L_capped) and (b_up > B_UP_MIN)
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
        "delta_capped": delta_capped,
        "L_capped": L_capped,
        "sheet_ok": sheet_ok,
        "orientation_ok": True,
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

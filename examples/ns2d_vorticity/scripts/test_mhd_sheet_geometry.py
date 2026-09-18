#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic tests for frozen local current-sheet geometry (#38)."""

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
import mhd_sheet_geometry as sg

FAILS = []
FWHM_FACT = 2.0 * math.sqrt(2.0 * math.log(2.0))


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)
    else:
        print(" ok ", msg)


def test_rotated_gaussian_orientation_and_sizes():
    print("\n== rotated Gaussian |j| ridge: orientation, width, length ==")
    sig_n, sig_t = 0.08, 0.40
    delta_true = FWHM_FACT * sig_n
    L_true = FWHM_FACT * sig_t
    cx = cy = math.pi
    for theta in (0.0, 0.4, 1.1, 1.47):
        n_true = (-math.sin(theta), math.cos(theta))
        for n in (64, 128, 256):
            j = mt.sample_grid(
                lambda x, y, th=theta: sg.gaussian_j(
                    x - cx, y - cy, th, sig_n, sig_t), n)
            nh, th = sg.sheet_frame(j, cx, cy)
            align = abs(nh[0] * n_true[0] + nh[1] * n_true[1])
            check(align > 0.97,
                  "theta=%.2f N=%d n_hat align=%.4f (catches swapped axes if <cos 45)"
                  % (theta, n, align))
            dx = 2.0 * math.pi / n
            s_n, j_n = sg._sample_line(
                j, cx, cy, nh[0], nh[1], sg.S_MAX_N, sg.N_LINE, dx, mt.TWOPI)
            s_t, j_t = sg._sample_line(
                j, cx, cy, th[0], th[1], sg.S_MAX_T, sg.N_LINE, dx, mt.TWOPI)
            delta = sg.fwhm_line(s_n, j_n)
            L = sg.fwhm_line(s_t, j_t)
            check(abs(delta - delta_true) / delta_true < 0.12,
                  "theta=%.2f N=%d delta=%.4f true=%.4f" % (
                      theta, n, delta, delta_true))
            check(abs(L - L_true) / L_true < 0.12,
                  "theta=%.2f N=%d L=%.4f true=%.4f" % (theta, n, L, L_true))
            check(L > 2.5 * delta,
                  "theta=%.2f N=%d L/delta=%.2f (length ≠ thickness)" % (
                      theta, n, L / max(delta, 1e-12)))


def test_harris_b_up():
    print("\n== Harris sheet B_up at ±KAPPA delta ==")
    delta, b0 = 0.10, 1.3
    cx = cy = math.pi
    for theta in (0.0, 0.7):
        for n in (64, 128, 256):
            a = mt.sample_grid(
                lambda x, y, th=theta: sg.harris_a(
                    x - cx, y - cy, th, delta, b0), n)
            geo = sg.measure_sheet(a, cx, cy, eta=0.01)
            n_true = (-math.sin(theta), math.cos(theta))
            align = abs(geo["n_hat"][0] * n_true[0] + geo["n_hat"][1] * n_true[1])
            check(align > 0.95, "Harris theta=%.2f N=%d n_hat align=%.4f" % (
                theta, n, align))
            check(abs(geo["sample_distance"] - sg.KAPPA_UP * geo["delta"]) < 1e-9,
                  "sample distance is KAPPA*delta, not retuned")
            if abs(theta) < 1e-12:
                # Grid-aligned Harris: B_up saturates at B0.
                check(abs(geo["B_up"] - b0) / b0 < 0.08,
                      "Harris theta=0 N=%d B_up=%.4f expect ≈B0=%.4f" % (
                          n, geo["B_up"], b0))
            else:
                # Rotated Harris is not band-limited; require both sides
                # same sign-magnitude to within 25% (no one-sided sampling).
                check(min(geo["B_up_plus"], geo["B_up_minus"]) > 0.2 * b0,
                      "Harris theta=%.2f N=%d both sides sampled (%.3f, %.3f)" % (
                          theta, n, geo["B_up_plus"], geo["B_up_minus"]))


def test_wide_sheet_is_capped():
    print("\n== search-window saturation is flagged, not a physical width ==")
    cx = cy = math.pi
    # FWHM ~ 2.355 * sig_n. sig_n=1.6 => FWHM ~ 3.8 > DELTA_CAP=pi.
    j = mt.sample_grid(
        lambda x, y: sg.gaussian_j(x - cx, y - cy, 0.0, 1.6, 2.5), 128)
    nh, th = sg.sheet_frame(j, cx, cy)
    dx = 2.0 * math.pi / 128
    s_n, j_n = sg._sample_line(
        j, cx, cy, nh[0], nh[1], sg.S_MAX_N, sg.N_LINE, dx, mt.TWOPI)
    width, capped = sg._width_at_level(
        s_n, j_n, 0.5 * float(np.max(np.abs(j_n))))
    check(capped, "wide Gaussian FWHM hits S_MAX_N (width=%.3f, cap=%.3f)" % (
        width, sg.DELTA_CAP))
    a = mt.sample_grid(
        lambda x, y: sg.harris_a(x - cx, y - cy, 0.0, 0.10, 1.0), 128)
    geo = sg.measure_sheet(a, cx, cy, eta=0.01)
    check(not geo["delta_capped"],
          "thin Harris delta not capped (delta=%.4f)" % geo["delta"])
    check(geo["L_capped"] and not geo["sheet_ok"],
          "infinite Harris L is window-capped, not a finite sheet")
    # Finite ridge: Gaussian |j| generated from a localized flux.
    def loc_a(x, y):
        nn = -(x - cx)
        tt = (y - cy)
        return 0.08 * np.exp(-0.5 * (nn / 0.08) ** 2 - 0.5 * (tt / 0.40) ** 2)
    a2 = mt.sample_grid(loc_a, 128)
    geo2 = sg.measure_sheet(a2, cx, cy, eta=0.01)
    check(geo2["sheet_ok"] and not geo2["delta_capped"] and not geo2["L_capped"],
          "localized flux sheet_ok (delta=%.4f L=%.4f)" % (
              geo2["delta"], geo2["L"]))


def test_constants_frozen():
    print("\n== frozen constants ==")
    check(abs(sg.ALPHA_J - 0.5) < 1e-15, "ALPHA_J=0.5")
    check(abs(sg.KAPPA_UP - 2.0) < 1e-15, "KAPPA_UP=2")
    check(abs(sg.R_ST - 0.5) < 1e-15, "R_ST=0.5")
    check(abs(sg.B_UP_MIN - 1.0e-3) < 1e-15, "B_UP_MIN=1e-3")


def main():
    test_constants_frozen()
    test_rotated_gaussian_orientation_and_sizes()
    test_harris_b_up()
    test_wide_sheet_is_capped()
    print("\n%d failures" % len(FAILS))
    if FAILS:
        for f in FAILS:
            print(" -", f)
        return 1
    print("ALL SHEET GEOMETRY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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


def test_disconnected_lobe_is_ignored():
    print("\n== connected s=0 superlevel ignores a remote ridge ==")
    sig, j_c, j_r = 0.08, 1.0, 3.0
    s_off = 1.50
    s = np.linspace(-sg.S_MAX_N, sg.S_MAX_N, sg.N_LINE)
    f = (j_c * np.exp(-0.5 * (s / sig) ** 2)
         + j_r * np.exp(-0.5 * ((s - s_off) / sig) ** 2))
    level = sg.ALPHA_J * j_c
    width, capped = sg._width_at_level(s, f, level)
    true_w = FWHM_FACT * sig
    check(abs(width - true_w) / true_w < 0.12,
          "1-D central FWHM=%.4f true=%.4f (remote ignored)" % (width, true_w))
    check(not capped,
          "remote lobe at window edge does not cap the local sheet")
    # First-to-last union (the bug) would span the remote endpoint.
    g = np.abs(f)
    above = np.where(g >= level)[0]
    union = float(s[int(above[-1])] - s[int(above[0])])
    check(union > 2.0 * width,
          "union width=%.3f > 2*local (test would fail on first-to-last)" % union)
    check(int(above[-1]) + 1 >= g.size,
          "remote component reaches the search-window endpoint")

    cx = cy = math.pi
    n_grid = 128
    # Remote along +n (y) so it sits on the delta line, near S_MAX_N.
    j = mt.sample_grid(
        lambda x, y: (
            sg.gaussian_j(x - cx, y - cy, 0.0, sig, 0.40, j_c)
            + sg.gaussian_j(x - cx, y - (cy + s_off), 0.0, sig, 0.40, j_r)),
        n_grid)
    nh, th = sg.sheet_frame(j, cx, cy)
    dx = 2.0 * math.pi / n_grid
    s_n, j_n = sg._sample_line(
        j, cx, cy, nh[0], nh[1], sg.S_MAX_N, sg.N_LINE, dx, mt.TWOPI)
    j_x = float(j_n[int(np.argmin(np.abs(s_n)))])
    dlt, cap = sg._width_at_level(s_n, j_n, sg.ALPHA_J * abs(j_x))
    check(abs(dlt - true_w) / true_w < 0.15,
          "2-D delta=%.4f true=%.4f with remote n-ridge" % (dlt, true_w))
    check(not cap, "2-D local delta not capped by remote n-ridge")

    # Remote along +t (x) on the L line.
    j2 = mt.sample_grid(
        lambda x, y: (
            sg.gaussian_j(x - cx, y - cy, 0.0, sig, 0.40, j_c)
            + sg.gaussian_j(x - (cx + 2.2), y - cy, 0.0, sig, 0.40, j_r)),
        n_grid)
    nh2, th2 = sg.sheet_frame(j2, cx, cy)
    s_t, j_t = sg._sample_line(
        j2, cx, cy, th2[0], th2[1], sg.S_MAX_T, sg.N_LINE, dx, mt.TWOPI)
    j_x2 = float(j_t[int(np.argmin(np.abs(s_t)))])
    L, Lcap = sg._width_at_level(s_t, j_t, sg.ALPHA_J * abs(j_x2))
    L_true = FWHM_FACT * 0.40
    check(abs(L - L_true) / L_true < 0.15,
          "2-D L=%.4f true=%.4f with remote t-ridge" % (L, L_true))
    check(not Lcap, "2-D local L not capped by remote t-ridge")


def test_sign_invariance():
    print("\n== j and -j give the same local sheet (up to eigenvector sign) ==")
    cx = cy = math.pi
    for theta in (0.0, 0.7):
        for n in (64, 128, 256):
            a = mt.sample_grid(
                lambda x, y, th=theta: sg.harris_a(
                    x - cx, y - cy, th, 0.10, 1.3), n)
            gp = sg.measure_sheet(a, cx, cy, eta=0.01)
            gm = sg.measure_sheet(-a, cx, cy, eta=0.01)
            check(gp["orientation_ok"] and gm["orientation_ok"],
                  "Harris theta=%.2f N=%d both orientations ok" % (theta, n))
            align_n = abs(gp["n_hat"][0] * gm["n_hat"][0]
                          + gp["n_hat"][1] * gm["n_hat"][1])
            align_t = abs(gp["t_hat"][0] * gm["t_hat"][0]
                          + gp["t_hat"][1] * gm["t_hat"][1])
            check(align_n > 0.97,
                  "theta=%.2f N=%d n_hat align=%.4f under j -> -j" % (
                      theta, n, align_n))
            check(align_t > 0.97,
                  "theta=%.2f N=%d t_hat align=%.4f under j -> -j" % (
                      theta, n, align_t))
            check(abs(gp["delta"] - gm["delta"]) / max(gp["delta"], 1e-12) < 0.02,
                  "theta=%.2f N=%d delta +j=%.4f -j=%.4f" % (
                      theta, n, gp["delta"], gm["delta"]))
            check(abs(gp["L"] - gm["L"]) / max(gp["L"], 1e-12) < 0.02,
                  "theta=%.2f N=%d L +j=%.4f -j=%.4f" % (
                      theta, n, gp["L"], gm["L"]))


def test_constants_frozen():
    print("\n== frozen constants ==")
    check(abs(sg.ALPHA_J - 0.5) < 1e-15, "ALPHA_J=0.5")
    check(abs(sg.KAPPA_UP - 2.0) < 1e-15, "KAPPA_UP=2")
    check(abs(sg.R_ST - 0.5) < 1e-15, "R_ST=0.5")
    check(abs(sg.B_UP_MIN - 1.0e-3) < 1e-15, "B_UP_MIN=1e-3")
    check(abs(sg.J_X_MIN - 1.0e-12) < 1e-20, "J_X_MIN=1e-12")
    try:
        import sheet_scaling as sc
        check(abs(sc.T_AVG_LO - 0.10) < 1e-15, "stage1 T_AVG_LO=0.10 (failed)")
        check(abs(sc.T_AVG_HI - 0.70) < 1e-15, "stage1 T_AVG_HI=0.70 (failed)")
        check(abs(sc.T_SCALE_LO - 0.314) < 1e-15, "stage2 T_SCALE_LO=0.314")
        check(abs(sc.T_SCALE_HI - 0.70) < 1e-15, "stage2 T_SCALE_HI=0.70")
    except Exception as exc:
        check(False, "sheet_scaling constants: %s" % exc)


def main():
    test_constants_frozen()
    test_rotated_gaussian_orientation_and_sizes()
    test_harris_b_up()
    test_wide_sheet_is_capped()
    test_disconnected_lobe_is_ignored()
    test_sign_invariance()
    print("\n%d failures" % len(FAILS))
    if FAILS:
        for f in FAILS:
            print(" -", f)
        return 1
    print("ALL SHEET GEOMETRY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

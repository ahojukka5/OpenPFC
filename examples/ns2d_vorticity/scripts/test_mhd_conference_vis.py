#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the MHD conference visualization helpers (#71).

Does not require scratch field dumps or matplotlib. Scripts are invoked
with --help so CI exercises the new entry points.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

try:
    import numpy as np
except ImportError:
    print("SKIP: numpy is not available")
    sys.exit(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mhd_conference_catalog as cat
import mhd_conference_style as st

FAILS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)
    else:
        print(" ok ", msg)


def test_catalog_contract():
    print("\n== catalog contract ==")
    for cid, spec in cat.CASES.items():
        for key in ("subdir", "N", "nu", "eta", "label", "issues"):
            check(key in spec, "%s has %s" % (cid, key))
        check(spec["nu"] == spec["eta"], "%s Pm=1" % cid)
    check(cat.HERO_ID in cat.CASES, "hero case exists")
    check(cat.COMPARISON_IDS[1] == cat.HERO_ID, "hero is the comparison middle")
    check(len(cat.COMPARISON_IDS) == 3, "three comparison cases")
    etas = [cat.CASES[c]["eta"] for c in cat.COMPARISON_IDS]
    check(etas == [0.01, 0.005, 0.0025], "comparison eta order")
    ns = [cat.CASES[c]["N"] for c in cat.COMPARISON_IDS]
    check(ns == [256, 512, 1024], "accepted resolutions")


def test_windows_do_not_mix():
    print("\n== evidence vs showcase windows ==")
    check(cat.in_reconnection_window(0.80), "t=0.80 is reconnection")
    check(not cat.in_reconnection_window(0.81), "t=0.81 is not reconnection")
    check(cat.in_scaling_window(0.314), "stage-2 start")
    check(cat.in_scaling_window(0.70), "stage-2 end")
    check(not cat.in_scaling_window(0.10), "stage-1 leftover is not scaling")
    check(not cat.in_scaling_window(0.785), "t=0.785 outside scaling")
    check(cat.window_class_for_time(2.2, "showcase") == cat.WINDOW_SHOWCASE,
          "showcase label")
    check(cat.window_class_for_time(0.5, "evidence") == cat.WINDOW_RECONNECTION,
          "evidence label")
    line, tline = cat.banner_lines(cat.HERO_ID, 0.628)
    check("Orszag" in line and "0.005" in line and "512" in line,
          "banner has case/eta/N")
    check("accepted reconnection window" in line, "banner window class")
    check("0.628" in tline, "banner time")


def test_env_data_root():
    print("\n== data root override ==")
    old = os.environ.get(cat.DATA_ROOT_ENV)
    os.environ[cat.DATA_ROOT_ENV] = "/tmp/mhd-vis-root"
    try:
        check(cat.data_root().endswith("mhd-vis-root"), "env override")
        check(cat.data_root("/explicit").endswith("explicit"),
              "explicit wins")
    finally:
        if old is None:
            os.environ.pop(cat.DATA_ROOT_ENV, None)
        else:
            os.environ[cat.DATA_ROOT_ENV] = old
    check(cat.data_root() == cat.DEFAULT_DATA_ROOT
          or cat.DATA_ROOT_ENV in os.environ,
          "default scratch root when env unset")


def test_dump_matching(tmp):
    print("\n== dump time matching ==")
    # Synthetic diagnostics + empty bins are enough for the CSV path;
    # dump_records also requires j_*.bin files.
    with open(os.path.join(tmp, "diagnostics.csv"), "w") as fh:
        fh.write("step,time,max_abs_j\n")
        fh.write("0,0.0,1\n")
        fh.write("4,%.12f,2\n" % cat.PI80)
        fh.write("8,%.12f,3\n" % (2.0 * cat.PI80))
        fh.write("12,2.2,9\n")
    for inc in (0, 4, 8, 12):
        open(os.path.join(tmp, "j_%04d.bin" % inc), "wb").close()
    recs = cat.dump_records(tmp, t_min=0.0, t_max=cat.T_RECONN_HI)
    check(len(recs) == 3, "peak-current dump excluded by t<=0.80")
    check(abs(recs[1]["t"] - cat.PI80) < 1.0e-12, "matched dump time")
    near = cat.nearest_dump(recs, 0.04)
    check(near["inc"] == 4, "nearest dump to 0.04 is inc 4")


def test_ols():
    print("\n== formal OLS SE ==")
    x = [10.0, 20.0, 40.0]
    y = [4.0 / v for v in x]
    slope, intercept, se = cat.loglog_ols(x, y)
    check(abs(slope + 1.0) < 1.0e-12, "exact S^{-1} slope")
    check(se < 1.0e-12, "zero SE on an exact line")
    S = [m["S_local"] for m in cat.ADMITTED_MEANS]
    R = [m["R"] for m in cat.ADMITTED_MEANS]
    slope, _, se = cat.loglog_ols(S, R)
    check(abs(slope + 1.10) < 0.02, "admitted slope near -1.10")
    check(0.01 < se < 0.04, "formal SE is O(0.02), not a physical bar")
    check(abs(slope + 0.5) > 0.3, "not Sweet-Parker -1/2")


def test_fixed_visual_constants():
    print("\n== frozen visual constants ==")
    check(len(cat.A_LEVELS) == 15, "fixed contour count")
    check(cat.A_LEVELS[0] < 0 < cat.A_LEVELS[-1], "signed a levels")
    check(cat.J_CLIM_EVIDENCE == 6.5, "frozen evidence j clim")
    check(cat.J_CLIM_SHOWCASE > cat.J_CLIM_EVIDENCE,
          "showcase clim is separate")
    check(st.J_CMAP == "RdBu_r", "consistent colormap")
    check(st.FIG_HERO[0] == st.FIG_HERO[0], "hero figure size exists")


def test_entry_points():
    print("\n== script entry points ==")
    scripts = [
        "render_mhd_conference_frames.py",
        "make_mhd_conference_animation.py",
        "plot_mhd_flux_budget.py",
        "plot_mhd_scaling_summary.py",
        "plot_mhd_geometry_summary.py",
        "make_mhd_conference_package.py",
    ]
    for name in scripts:
        path = os.path.join(HERE, name)
        proc = subprocess.run(
            [sys.executable, path, "--help"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        check(proc.returncode == 0, "%s --help" % name)


def test_refuse_showcase_in_evidence_window():
    print("\n== refuse mixed window labelling ==")
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "render_mhd_conference_frames.py"),
         "--asset", "hero", "--kind", "showcase", "--t", "0.5",
         "--out", "/tmp/should_not_write.png"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    check(proc.returncode != 0, "showcase+t=0.5 is rejected")
    err = (proc.stderr or b"").decode("utf-8", "replace")
    check("showcase" in err.lower() or "refusing" in err.lower(),
          "error names the window mix")


def main():
    test_catalog_contract()
    test_windows_do_not_mix()
    test_env_data_root()
    tmp = tempfile.mkdtemp(prefix="mhdvis-")
    try:
        test_dump_matching(tmp)
    finally:
        for name in os.listdir(tmp):
            os.remove(os.path.join(tmp, name))
        os.rmdir(tmp)
    test_ols()
    test_fixed_visual_constants()
    test_entry_points()
    test_refuse_showcase_in_evidence_window()
    print()
    if FAILS:
        print("%d FAIL" % len(FAILS))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

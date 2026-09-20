#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the full 2-D MHD conference visualization package.

Evidence assets use only accepted windows. The optional late-time movie
is written under figures/showcase/ and labelled as a dynamics showcase.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mhd_conference_catalog as cat
import mhd_conference_style as st
import make_mhd_conference_animation as enc
import plot_mhd_flux_budget as flux
import plot_mhd_geometry_summary as geom
import plot_mhd_scaling_summary as scaling
import render_mhd_conference_frames as rend

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def fig_root(out_root):
    return out_root or os.path.join(PKG, "figures")


def run_plot(fn, out, data_root=None):
    argv = ["--out", out]
    if data_root:
        argv = ["--data-root", data_root] + argv
    return fn(argv)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-root", default="")
    p.add_argument("--skip-animations", action="store_true")
    p.add_argument("--skip-showcase", action="store_true")
    p.add_argument("--stills-only", action="store_true")
    args = p.parse_args(argv)

    root = fig_root(args.out_root)
    evidence = os.path.join(root, "evidence")
    stills = os.path.join(evidence, "stills")
    showcase = os.path.join(root, "showcase")
    generated = os.path.join(root, "generated")
    for d in (evidence, stills, showcase, generated):
        os.makedirs(d, exist_ok=True)

    data = ["--data-root", args.data_root] if args.data_root else []
    droot = args.data_root

    flux_png = os.path.join(evidence, "flux_budget_eta0005.png")
    scaling_png = os.path.join(evidence, "scaling_R_vs_S.png")
    geom_png = os.path.join(evidence, "geometry_summary.png")
    rc = 0
    rc |= run_plot(flux.main, flux_png, droot) or 0
    rc |= run_plot(scaling.main, scaling_png, droot) or 0
    rc |= run_plot(geom.main, geom_png, droot) or 0
    still_scaling = os.path.join(stills, "scaling_R_vs_S.png")
    rc |= run_plot(scaling.main, still_scaling, droot) or 0

    hero_still = os.path.join(stills, "hero_sheet_eta0005_n512.png")
    topo_still = os.path.join(stills, "topology_overlay_eta0005.png")
    cmp_still = os.path.join(stills, "comparison_three_eta.png")
    rc |= rend.main(data + [
        "--asset", "hero", "--kind", "evidence",
        "--t", str(cat.STILL_SHEET_T), "--dpi", str(st.DPI_STILL),
        "--out", hero_still,
    ]) or 0
    rc |= rend.main(data + [
        "--asset", "hero", "--kind", "evidence",
        "--t", str(cat.STILL_TOPOLOGY_T), "--dpi", str(st.DPI_STILL),
        "--out", topo_still,
    ]) or 0
    rc |= rend.main(data + [
        "--asset", "comparison", "--kind", "evidence",
        "--t", str(cat.STILL_SHEET_T), "--dpi", str(st.DPI_STILL),
        "--out", cmp_still,
    ]) or 0

    if args.stills_only:
        return rc

    if not args.skip_animations:
        hero_dir = os.path.join(generated, "hero_evidence")
        cmp_dir = os.path.join(generated, "comparison_evidence")
        rc |= rend.main(data + [
            "--asset", "hero", "--kind", "evidence", "--out-dir", hero_dir,
        ]) or 0
        rc |= rend.main(data + [
            "--asset", "comparison", "--kind", "evidence", "--out-dir", cmp_dir,
        ]) or 0
        rc |= enc.main([
            "--frames-dir", hero_dir, "--kind", "evidence", "--asset", "hero",
            "--mp4", os.path.join(evidence, "hero_reconnection_eta0005_n512.mp4"),
        ]) or 0
        rc |= enc.main([
            "--frames-dir", cmp_dir, "--kind", "evidence",
            "--asset", "comparison",
            "--mp4", os.path.join(evidence, "comparison_three_eta.mp4"),
        ]) or 0

    if not args.skip_showcase and not args.skip_animations:
        show_dir = os.path.join(generated, "hero_showcase")
        rc |= rend.main(data + [
            "--asset", "hero", "--kind", "showcase", "--out-dir", show_dir,
        ]) or 0
        rc |= enc.main([
            "--frames-dir", show_dir, "--kind", "showcase", "--asset", "hero",
            "--mp4", os.path.join(showcase, "late_dynamics_eta0005_n512.mp4"),
        ]) or 0

    cat.write_json(os.path.join(root, "provenance.json"), {
        "data_root": cat.data_root(args.data_root),
        "evidence_window": [cat.T_RECONN_LO, cat.T_RECONN_HI],
        "scaling_window": [cat.T_SCALE_LO, cat.T_SCALE_HI],
        "j_clim_evidence": cat.J_CLIM_EVIDENCE,
        "j_clim_showcase": cat.J_CLIM_SHOWCASE,
        "cases": {k: dict(v) for k, v in cat.CASES.items()},
        "outputs": {
            "flux": flux_png,
            "scaling": scaling_png,
            "geometry": geom_png,
            "stills": [hero_still, cmp_still, topo_still, still_scaling],
        },
    })
    return rc


if __name__ == "__main__":
    sys.exit(main())

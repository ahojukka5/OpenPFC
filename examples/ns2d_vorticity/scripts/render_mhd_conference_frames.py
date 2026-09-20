#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render conference-grade MHD field frames (j, a-contours, X/O, sheet).

Nearest-neighbour imshow; fixed colour limits; no per-frame autoscale.
Evidence frames use only t<=0.80. Late-time frames must pass --kind
showcase.

Examples:
  python3 render_mhd_conference_frames.py --asset hero \\
    --out-dir figures/generated/hero
  python3 render_mhd_conference_frames.py --asset comparison --t 0.628 \\
    --out figures/evidence/stills/comparison_three_eta.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_mhd_fields import load_brick
import mhd_conference_catalog as cat
import mhd_conference_style as st
import mhd_topology as mt

KIND_EVIDENCE = "evidence"
KIND_SHOWCASE = "showcase"


def _require_mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
    except ImportError as exc:
        raise SystemExit("matplotlib is required: %s" % exc)
    return plt, FancyArrowPatch


def grid_xy(n):
    dx = 2.0 * math.pi / n
    return (np.arange(n) + 0.5) * dx


def wrapped_segments(xs, ys, period=2.0 * math.pi):
    segs = []
    cx, cy = [xs[0]], [ys[0]]
    for i in range(1, len(xs)):
        jump = (abs(xs[i] - xs[i - 1]) > 0.5 * period
                or abs(ys[i] - ys[i - 1]) > 0.5 * period)
        if jump:
            segs.append((cx, cy))
            cx, cy = [xs[i]], [ys[i]]
        else:
            cx.append(xs[i])
            cy.append(ys[i])
    segs.append((cx, cy))
    return segs


def draw_periodic_segment(ax, x0, y0, x1, y1, **kw):
    n = 48
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xs = np.mod(xs, 2.0 * math.pi)
    ys = np.mod(ys, 2.0 * math.pi)
    for sx, sy in wrapped_segments(xs.tolist(), ys.tolist()):
        ax.plot(sx, sy, **kw)


def draw_points(ax, pts, tracked=None):
    for pt in pts:
        spec = st.marker_spec(pt.get("kind", "."))
        spec["clip_on"] = False
        ax.scatter([pt["x"]], [pt["y"]], **spec)
    if tracked is not None:
        ax.scatter(
            [tracked["x_X"]], [tracked["y_X"]],
            s=110, facecolors="none", edgecolors=st.TRACK_RING,
            linewidths=1.2, zorder=7, clip_on=False,
        )
        ax.scatter(
            [tracked["x_O"]], [tracked["y_O"]],
            s=92, facecolors="none", edgecolors=st.TRACK_RING,
            linewidths=1.0, linestyle="--", zorder=7, clip_on=False,
        )


def draw_sheet_overlay(ax, row, FancyArrowPatch):
    if not row:
        return
    if not row.get("sheet_ok"):
        return
    x = float(row["x_X"])
    y = float(row["y_X"])
    tx, ty = row["t_hat"]
    nx, ny = row["n_hat"]
    L = float(row["L"])
    delta = float(row["delta"])
    draw_periodic_segment(
        ax, x - 0.5 * L * tx, y - 0.5 * L * ty,
        x + 0.5 * L * tx, y + 0.5 * L * ty,
        color=st.SHEET_T_COLOR, lw=st.LW_SHEET, zorder=5, solid_capstyle="round",
    )
    draw_periodic_segment(
        ax, x - 0.5 * delta * nx, y - 0.5 * delta * ny,
        x + 0.5 * delta * nx, y + 0.5 * delta * ny,
        color=st.SHEET_N_COLOR, lw=st.LW_SHEET + 0.2, zorder=5,
        solid_capstyle="round",
    )
    arr_len = 0.42
    for vx, vy, col, lab in (
        (tx, ty, st.SHEET_T_COLOR, r"$\hat t$"),
        (nx, ny, st.SHEET_N_COLOR, r"$\hat n$"),
    ):
        ax.annotate(
            lab, (x, y),
            xytext=(18 * vx, 18 * vy),
            textcoords="offset points",
            color=col, fontsize=st.FONT_ANNOT, zorder=8,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1),
        )
    ax.text(
        0.03, 0.04,
        r"$L=%.2f$   $\delta=%.2f$" % (L, delta),
        transform=ax.transAxes, color=st.TALK_TEXT,
        fontsize=st.FONT_ANNOT,
        bbox=dict(boxstyle="round,pad=0.25", fc="#00000088", ec="none"),
    )


def load_native_fields(rec, n):
    a = load_brick(rec["a_path"], n)
    j = load_brick(rec["j_path"], n)
    return a, j


def panel_fields(ax, a, j, n, jlim, title, FancyArrowPatch,
                 overlay_sheet=False, tracked=None):
    x = grid_xy(n)
    im = st.field_imshow(ax, j.T, vmin=-jlim, vmax=jlim)
    st.flux_contours(ax, x, a.T)
    pts = mt.locate_critical_points(a, j)
    draw_points(ax, pts, tracked=tracked)
    if overlay_sheet:
        draw_sheet_overlay(ax, tracked, FancyArrowPatch)
    st.style_field_axes(ax, title=title)
    return im


def _kind_jlim(kind):
    if kind == KIND_SHOWCASE:
        return cat.J_CLIM_SHOWCASE
    return cat.J_CLIM_EVIDENCE


def render_hero_frame(case_id, rec, out, root, kind, dpi, overlay_scale=True):
    plt, FancyArrowPatch = _require_mpl()
    st.apply_talk_style(plt)
    spec = cat.case_spec(case_id)
    run = cat.case_dir(case_id, root)
    a, j = load_native_fields(rec, spec["N"])
    tracked = cat.nearest_row(cat.load_tracked_rows(run), rec["t"])
    overlay = (
        overlay_scale
        and kind == KIND_EVIDENCE
        and tracked is not None
        and cat.in_scaling_window(rec["t"])
        and bool(tracked.get("sheet_ok"))
    )
    fig, ax = plt.subplots(figsize=st.FIG_HERO, dpi=dpi)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.88)
    im = panel_fields(
        ax, a, j, spec["N"], _kind_jlim(kind),
        r"current $j$  /  flux $a$",
        FancyArrowPatch, overlay_sheet=overlay, tracked=tracked,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$j$  (fixed $\pm %.1f$)" % _kind_jlim(kind),
                   color=st.TALK_TEXT)
    cbar.ax.yaxis.set_tick_params(color=st.TALK_TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=st.TALK_TEXT)
    line1, line2 = cat.banner_lines(case_id, rec["t"], kind)
    st.draw_banner(fig, line1, line2)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches=None, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def render_comparison_frame(recs_by_id, t, out, root, dpi):
    plt, FancyArrowPatch = _require_mpl()
    st.apply_talk_style(plt)
    fig, axes = plt.subplots(1, 3, figsize=st.FIG_COMPARISON, dpi=dpi)
    fig.subplots_adjust(left=0.04, right=0.92, bottom=0.12, top=0.82,
                        wspace=0.12)
    im = None
    for ax, cid in zip(axes, cat.COMPARISON_IDS):
        spec = cat.case_spec(cid)
        rec = recs_by_id[cid]
        a, j = load_native_fields(rec, spec["N"])
        tracked = cat.nearest_row(
            cat.load_tracked_rows(cat.case_dir(cid, root)), rec["t"])
        title = r"$\eta=\nu=%g$   $N=%d$" % (spec["eta"], spec["N"])
        im = panel_fields(
            ax, a, j, spec["N"], cat.J_CLIM_EVIDENCE, title,
            FancyArrowPatch, overlay_sheet=False, tracked=tracked,
        )
        ax.set_ylabel(r"$y$" if ax is axes[0] else "")
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012)
    cbar.set_label(r"$j$  (shared $\pm %.1f$)" % cat.J_CLIM_EVIDENCE,
                   color=st.TALK_TEXT)
    cbar.ax.yaxis.set_tick_params(color=st.TALK_TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=st.TALK_TEXT)
    fig.text(
        0.5, 0.97,
        "Orszag–Tang   $P_m=1$   %s   $t=%.3f$"
        % (cat.WINDOW_RECONNECTION, t),
        ha="center", va="top", color=st.TALK_TEXT, fontsize=st.FONT_BANNER,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches=None, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def dumps_for_asset(asset, kind, root, t_single=None):
    if asset == "comparison":
        t_max = cat.T_RECONN_HI
        times = cat.common_dump_times(cat.COMPARISON_IDS, root, t_max)
        if t_single is not None:
            times = [min(times, key=lambda t: abs(t - t_single))]
        by_id = {}
        for cid in cat.COMPARISON_IDS:
            recs = cat.dump_records(
                cat.case_dir(cid, root), t_min=0.0, t_max=t_max)
            keyed = {cat.time_key(r["t"]): r for r in recs}
            by_id[cid] = keyed
        frames = []
        for t in times:
            recs = {cid: by_id[cid][t] for cid in cat.COMPARISON_IDS}
            frames.append((t, recs))
        return frames
    cid = cat.HERO_ID
    if kind == KIND_SHOWCASE:
        recs = cat.dump_records(
            cat.case_dir(cid, root),
            t_min=cat.T_RECONN_HI + 1.0e-9, t_max=None)
    else:
        recs = cat.dump_records(
            cat.case_dir(cid, root), t_min=0.0, t_max=cat.T_RECONN_HI)
    if t_single is not None:
        recs = [cat.nearest_dump(recs, t_single)]
    return [(r["t"], r) for r in recs]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", choices=("hero", "comparison"), default="hero")
    p.add_argument("--kind", choices=(KIND_EVIDENCE, KIND_SHOWCASE),
                   default=KIND_EVIDENCE)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-dir", default="")
    p.add_argument("--out", default="")
    p.add_argument("--t", type=float, default=None,
                   help="render the dump nearest this time only")
    p.add_argument("--dpi", type=int, default=st.DPI_FRAME)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    if args.kind == KIND_SHOWCASE and args.asset != "hero":
        print("showcase is defined only for the hero OT eta=0.005 run",
              file=sys.stderr)
        return 2
    if args.kind == KIND_SHOWCASE and args.t is not None:
        if cat.in_reconnection_window(args.t) and args.t <= cat.T_RECONN_HI:
            print("refusing to label t<=0.80 as a late-time showcase",
                  file=sys.stderr)
            return 2

    root = cat.data_root(args.data_root)
    frames = dumps_for_asset(args.asset, args.kind, root, args.t)
    if args.limit:
        frames = frames[args.start:args.start + args.limit]
    elif args.start:
        frames = frames[args.start:]
    if not frames:
        print("no dumps for asset=%s kind=%s under %s"
              % (args.asset, args.kind, root), file=sys.stderr)
        return 1

    written = []
    if args.out and (args.t is not None or len(frames) == 1):
        t, payload = frames[0]
        if args.asset == "comparison":
            path = render_comparison_frame(
                payload, t, args.out, root, args.dpi)
        else:
            path = render_hero_frame(
                cat.HERO_ID, payload, args.out, root, args.kind, args.dpi)
        written.append(path)
    else:
        out_dir = args.out_dir or os.path.join("figures", "generated",
                                               "%s_%s" % (args.asset, args.kind))
        os.makedirs(out_dir, exist_ok=True)
        for i, (t, payload) in enumerate(frames):
            name = "frame_%04d.png" % i
            dest = os.path.join(out_dir, name)
            if args.asset == "comparison":
                render_comparison_frame(payload, t, dest, root, args.dpi)
            else:
                render_hero_frame(
                    cat.HERO_ID, payload, dest, root, args.kind, args.dpi)
            written.append(dest)
            print("wrote", dest, "t=%.5f" % t)

    sidecar = (args.out or written[0]) + ".json"
    extra = {
        "asset": args.asset,
        "n_frames": len(written),
        "j_clim": _kind_jlim(args.kind),
        "a_levels": list(cat.A_LEVELS),
        "data_root": root,
        "outputs": written,
        "t_first": frames[0][0],
        "t_last": frames[-1][0],
    }
    cid = cat.HERO_ID if args.asset == "hero" else cat.COMPARISON_IDS[1]
    cat.write_json(sidecar, cat.provenance(
        cid, t=frames[0][0], kind=args.kind, extra=extra))
    print("wrote", len(written), "frame(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

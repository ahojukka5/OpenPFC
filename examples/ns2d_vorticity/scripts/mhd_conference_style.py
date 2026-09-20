#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared visual language for the 2-D MHD conference package.

Field plots use a dark talk theme. XY evidence figures use a light
paper theme. Colour maps, contour levels, markers, and type sizes are
fixed; drivers must not autoscale j between frames of one animation.
"""

from __future__ import annotations

import math

from mhd_conference_catalog import A_LEVELS, J_CLIM_EVIDENCE

# Talk / field figures
TALK_FACE = "#0e131c"
TALK_AXES = "#141b27"
TALK_TEXT = "#f4f0e6"
TALK_MUTED = "#b7b1a4"
TALK_GRID = "#2a3344"
TALK_ACCENT = "#e8c872"

# Paper / XY figures
PAPER_FACE = "#ffffff"
PAPER_TEXT = "#1b1f24"
PAPER_GRID = "#d9d5cc"
PAPER_EZ = "#1f4e79"
PAPER_ETAJ = "#c0392b"
PAPER_DFDT = "#1e8449"
PAPER_EZO = "#6c3483"
PAPER_REF = "#7f8c8d"
PAPER_SP = "#b9770e"
PAPER_INV = "#1a5276"

J_CMAP = "RdBu_r"
X_COLOR = "#ffe566"
O_COLOR = "#7ee0ff"
DEGEN_COLOR = "#ff8a8a"
SHEET_T_COLOR = "#ffcc66"
SHEET_N_COLOR = "#f7f7f7"
TRACK_RING = "#ffffff"

FONT_TITLE = 11
FONT_LABEL = 10
FONT_TICK = 8
FONT_BANNER = 9
FONT_ANNOT = 8

LW_CONTOUR = 0.55
LW_SHEET = 1.6
LW_AXIS = 0.8

FIG_HERO = (8.0, 8.4)
FIG_COMPARISON = (15.6, 5.6)
FIG_FLUX = (8.6, 5.15)
FIG_SCALING = (6.5, 6.15)
FIG_GEOMETRY = (10.6, 6.7)
DPI_FRAME = 140
DPI_STILL = 220

HERO_FPS = 4
HOLD_LAST = 8


def apply_talk_style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FONT_LABEL,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "axes.linewidth": LW_AXIS,
        "axes.facecolor": TALK_AXES,
        "figure.facecolor": TALK_FACE,
        "savefig.facecolor": TALK_FACE,
        "text.color": TALK_TEXT,
        "axes.labelcolor": TALK_TEXT,
        "axes.edgecolor": TALK_MUTED,
        "xtick.color": TALK_TEXT,
        "ytick.color": TALK_TEXT,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "mathtext.fontset": "dejavusans",
        "savefig.pad_inches": 0.0,
        "savefig.dpi": DPI_STILL,
    })


def apply_paper_style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FONT_LABEL,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "axes.linewidth": LW_AXIS,
        "axes.facecolor": PAPER_FACE,
        "figure.facecolor": PAPER_FACE,
        "savefig.facecolor": PAPER_FACE,
        "text.color": PAPER_TEXT,
        "axes.labelcolor": PAPER_TEXT,
        "axes.edgecolor": "#333333",
        "xtick.color": PAPER_TEXT,
        "ytick.color": PAPER_TEXT,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.color": PAPER_GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "mathtext.fontset": "dejavusans",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "savefig.dpi": DPI_STILL,
    })


def field_imshow(ax, j, vmin=-J_CLIM_EVIDENCE, vmax=J_CLIM_EVIDENCE):
    return ax.imshow(
        j,
        origin="lower",
        extent=(0.0, 2.0 * math.pi, 0.0, 2.0 * math.pi),
        cmap=J_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )


def flux_contours(ax, x, a):
    return ax.contour(
        x,
        x,
        a,
        levels=list(A_LEVELS),
        colors="k",
        linewidths=LW_CONTOUR,
        origin="lower",
    )


def style_field_axes(ax, title=""):
    ax.set_xlim(0.0, 2.0 * math.pi)
    ax.set_ylim(0.0, 2.0 * math.pi)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks([0.0, math.pi, 2.0 * math.pi])
    ax.set_yticks([0.0, math.pi, 2.0 * math.pi])
    ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    if title:
        ax.set_title(title, color=TALK_TEXT, pad=6)


def draw_banner(fig, line1, line2=""):
    text = line1 if not line2 else "%s\n%s" % (line1, line2)
    fig.text(
        0.5, 0.975, text, ha="center", va="top",
        color=TALK_TEXT, fontsize=FONT_BANNER,
    )


def marker_spec(kind):
    if kind == "X":
        return dict(marker="x", color=X_COLOR, s=46, linewidths=1.7, zorder=6)
    if kind in ("O_max", "O_min"):
        shape = "o" if kind == "O_max" else "s"
        return dict(
            marker=shape,
            facecolors="none",
            edgecolors=O_COLOR,
            s=38,
            linewidths=1.35,
            zorder=6,
        )
    return dict(marker="+", color=DEGEN_COLOR, s=30, linewidths=1.2, zorder=6)

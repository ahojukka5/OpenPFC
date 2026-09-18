#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render mhd2d BinaryWriter frames: j with a-contours, plus omega.

Example:
  python3 render_mhd_frames.py \\
    --dir /scratch/.../ot256_nu0005 --n 256 --inc 153 \\
    --out /scratch/.../ot256_nu0005/frame_0153.png
"""

from __future__ import annotations

import argparse
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
    raise ValueError(f"{path}: got {raw.size} doubles, expected {n*n}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--inc", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    args = p.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("matplotlib is required:", exc, file=sys.stderr)
        return 1

    n = args.n
    dx = 2.0 * math.pi / n
    x = (np.arange(n) + 0.5) * dx
    y = (np.arange(n) + 0.5) * dx
    # Field layout is a[i,j] = a(x_i, y_j). imshow wants [y, x].
    a = load_brick(os.path.join(args.dir, f"a_{args.inc:04d}.bin"), n).T
    j = load_brick(os.path.join(args.dir, f"j_{args.inc:04d}.bin"), n).T
    w = load_brick(os.path.join(args.dir, f"omega_{args.inc:04d}.bin"), n).T

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), constrained_layout=True)
    jlim = np.max(np.abs(j))
    wlim = np.max(np.abs(w))
    im0 = axes[0].imshow(
        j,
        origin="lower",
        extent=(0.0, 2.0 * math.pi, 0.0, 2.0 * math.pi),
        cmap="RdBu_r",
        vmin=-jlim,
        vmax=jlim,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0].contour(
        x,
        y,
        a,
        levels=12,
        colors="k",
        linewidths=0.5,
        origin="lower",
    )
    axes[0].set_title("current density $j$ with flux $a$ contours")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="$j$")

    im1 = axes[1].imshow(
        w,
        origin="lower",
        extent=(0.0, 2.0 * math.pi, 0.0, 2.0 * math.pi),
        cmap="RdBu_r",
        vmin=-wlim,
        vmax=wlim,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title(r"vorticity $\omega$")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$y$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=r"$\omega$")

    if args.title:
        fig.suptitle(args.title)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

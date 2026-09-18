#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Midplane slices of a dumped rotating-cube unit cell (OpenPFC #31)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bin_path", type=Path)
    p.add_argument("--nx", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    n = args.nx
    h = np.fromfile(args.bin_path, dtype=np.float64)
    if h.size != n * n * n:
        raise SystemExit(f"expected {n}^3={n ** 3} values, got {h.size}")
    vol = h.reshape((n, n, n))  # k, j, i as written i-fastest → C order (k,j,i)
    # dense_from_field writes i-fastest, then j, then k, so reshape (nz,ny,nx)
    xy = vol[n // 2]
    xz = vol[:, n // 2, :]
    yz = vol[:, :, n // 2]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; wrote no PNG", file=sys.stderr)
        return 1
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), dpi=120)
    for ax, sl, title in (
        (axes[0], xy, "xy mid"),
        (axes[1], xz, "xz mid"),
        (axes[2], yz, "yz mid"),
    ):
        ax.imshow(sl, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"rotating-cubes {n}^3")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

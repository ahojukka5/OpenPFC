#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render issue #9 3-D inverse-homogenization frames.

Orthogonal midplanes are the audit view. The hero still is surface voxels
from a downsampled solver `h`. Claims use homogenizer CSV: directional
Poisson ratios from S=C^{-1}, not the isotropic shortcut nu_eff.

`--init=spinodal` on this branch is a Fourier-mode seed, not a
Cahn-Hilliard-evolved process microstructure.

    python3 apps/inverse_homogenization/scripts/render_inverse3d_showcase.py \\
        --run /scratch/.../inverse3d_JOB
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "docs" / "report" / "figures"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

from field_io import GridSpec, read_bin  # noqa: E402


def load_history(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        for k, v in row.items():
            try:
                out.setdefault(k, []).append(float(v))
            except ValueError:
                out.setdefault(k, []).append(float("nan"))
    return out


def load_cube(fields, man, idx):
    name = man["pattern"].replace("{field}", "h").replace("{index:04d}", "%04d" % idx)
    g = GridSpec(nx=man["nx"], ny=man["ny"], nz=man["nz"], dx=man["dx"], dy=man["dx"])
    return read_bin(fields / name, g)


def z_solid_std(cube):
    solid = (cube > 0.5).astype(float)
    frac = solid.mean(axis=(1, 2))
    return float(np.std(frac))


def downsample_mask(cube, nmax=32):
    step = max(1, int(np.ceil(max(cube.shape) / float(nmax))))
    return (cube[::step, ::step, ::step] > 0.5)


def write_voxels(mask, path, title):
    fig = plt.figure(figsize=(5.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(mask, facecolors="#4c78a8", edgecolor="none")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def last(hist, key):
    vals = hist.get(key) or [float("nan")]
    return vals[-1]


def render(run, out, fps):
    fields = run / "fields"
    man = json.loads(next(fields.glob("*_manifest.json")).read_text())
    if int(man["nz"]) < 2:
        raise SystemExit("nz < 2: this renderer is for issue #9 3-D cells")
    hist = load_history(run / "history.csv") if (run / "history.csv").exists() else {}
    n = len(man.get("steps", []))
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    last_h = load_cube(fields, man, n - 1)
    zstd = z_solid_std(last_h)
    mask = downsample_mask(last_h)
    write_voxels(mask, out / "inverse3d_voxels.png",
                 "h>0.5 surface voxels (downsampled)")

    for i in range(n):
        h = load_cube(fields, man, i)
        fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.4))
        mid = (
            (axes[0], h[h.shape[0] // 2], "xy mid"),
            (axes[1], h[:, h.shape[1] // 2, :], "xz mid"),
            (axes[2], h[:, :, h.shape[2] // 2], "yz mid"),
        )
        for ax, img, title in mid:
            ax.imshow(img, origin="lower", cmap="cividis", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = axes[3]
        if hist:
            ax.plot(hist.get("step", []), hist.get("J", []), color="0.2", label="J")
            if "nu12" in hist:
                ax2 = ax.twinx()
                ax2.plot(hist.get("step", []), hist.get("nu12", []),
                         color="#b04830", label=r"$\nu_{12}$")
                ax2.plot(hist.get("step", []), hist.get("nu13", []),
                         color="#4c78a8", label=r"$\nu_{13}$")
                ax2.set_ylabel(r"$\nu$ from $S=C^{-1}$")
            ax.set_xlabel("step")
            ax.set_ylabel("J")
        fig.suptitle("t-index %d  z-solid-std=%.4g" % (i, z_solid_std(h)))
        fig.tight_layout()
        fig.savefig(frames_dir / ("frame_%04d.png" % i), dpi=110)
        plt.close(fig)

    still = out / "final_slices.png"
    shutil.copy(frames_dir / ("frame_%04d.png" % (n - 1)), still)
    (out / "summary.md").write_text(
        "# Inverse 3-D showcase\n\n"
        "- run `%s`\n- grid %d^3\n- frames %d\n"
        "- z-solid-fraction std (final): %.6g (0 is an extrusion)\n"
        "- nu12 (S=C^{-1}): %s\n- nu13: %s\n- nu23: %s\n"
        "- nu_eff in the CSV is the isotropic shortcut C12/(C11+C12); "
        "do not quote it for an orthotropic C_H\n"
        "- --init=spinodal is a Fourier-mode seed, not Cahn-Hilliard\n"
        "- hero still: inverse3d_voxels.png; slices are audit panels\n"
        % (run, man["nx"], n, zstd, last(hist, "nu12"), last(hist, "nu13"),
           last(hist, "nu23"))
    )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.check_call(
            [
                ffmpeg, "-y", "-framerate", str(fps), "-i",
                str(frames_dir / "frame_%04d.png"), "-pix_fmt", "yuv420p",
                str(out / "inverse3d.mp4"),
            ]
        )
        print("wrote", out / "inverse3d.mp4")
    else:
        print("ffmpeg not found; frames in", frames_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=int, default=6)
    args = p.parse_args()
    out = args.out or (args.run / "present")
    out.mkdir(parents=True, exist_ok=True)
    render(args.run, out, args.fps)


if __name__ == "__main__":
    main()

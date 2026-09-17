#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render issue #12 3-D dendrite frames (orthogonal midplanes).

Spherical-seed solver bricks. Not a 2-D extrusion.

    python3 apps/alloy_dendrite_elastic/scripts/render_dendrite3d_showcase.py \\
        --run /scratch/.../dendrite3d_JOB
"""

from __future__ import annotations

import argparse
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


def load_cube(d, field, idx):
    man = json.loads(next(d.glob("*_manifest.json")).read_text())
    g = GridSpec(nx=man["nx"], ny=man["ny"], nz=man["nz"], dx=man["dx"], dy=man["dx"])
    name = man["pattern"].replace("{field}", field).replace("{index:04d}", "%04d" % idx)
    return read_bin(d / name, g), man


def render(run, out, fps):
    fields = run / "fields"
    phi0, man = load_cube(fields, "phi", 0)
    if int(man["nz"]) < 2:
        raise SystemExit("nz < 2: issue #12 is a 3-D brick")
    n = len(man["times"])
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    last, _ = load_cube(fields, "phi", n - 1)
    solid_z = (last > 0.0).mean(axis=(1, 2))
    zstd = float(np.std(solid_z))

    for i in range(n):
        phi, _ = load_cube(fields, "phi", i)
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
        views = (
            (axes[0], phi[phi.shape[0] // 2], "xy"),
            (axes[1], phi[:, phi.shape[1] // 2, :], "xz"),
            (axes[2], phi[:, :, phi.shape[2] // 2], "yz"),
        )
        for ax, img, title in views:
            ax.imshow(img, origin="lower", cmap="coolwarm", vmin=-1, vmax=1,
                      interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("t=%g  z-solid-std=%.4g" % (man["times"][i], float(np.std((phi > 0).mean(axis=(1, 2))))))
        fig.tight_layout()
        fig.savefig(frames / ("frame_%04d.png" % i), dpi=110)
        plt.close(fig)

    shutil.copy(frames / ("frame_%04d.png" % (n - 1)), out / "dendrite3d_still.png")
    step = max(1, int(man["nx"]) // 48)
    np.savez_compressed(
        out / "phi_downsampled.npz",
        phi=last[::step, ::step, ::step],
        dx=float(man["dx"]) * step,
        threshold_mask=(last[::step, ::step, ::step] > 0).astype(np.uint8),
    )
    write_rotate(last, out, fps)
    mask = last[:: max(1, int(man["nx"]) // 32),
                :: max(1, int(man["ny"]) // 32),
                :: max(1, int(man["nz"]) // 32)] > 0
    fig = plt.figure(figsize=(5.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(mask, facecolors="#b04830", edgecolor="none")
    ax.set_title("phi>0 surface voxels (downsampled)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(out / "dendrite3d_voxels.png", dpi=120)
    plt.close(fig)
    (out / "summary.md").write_text(
        "# Dendrite 3-D showcase\n\n- run `%s`\n- grid %d^3\n- frames %d\n"
        "- z-solid-fraction std: %.6g (0 is an extrusion)\n"
        "- isothermal; elasticity off; cubic <100> along the brick, eps4=0.05\n"
        "- hero still: dendrite3d_voxels.png; slices/MIP are audit panels\n"
        "- size later grids from HIP_MEM bytes_per_cell, not sacct MaxRSS\n"
        "- 8 doubles/cell estimates are provisional host-side counting\n"
        % (run, man["nx"], n, zstd)
    )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.check_call(
            [
                ffmpeg, "-y", "-framerate", str(fps), "-i",
                str(frames / "frame_%04d.png"), "-pix_fmt", "yuv420p",
                str(out / "dendrite3d.mp4"),
            ]
        )
        print("wrote", out / "dendrite3d.mp4")
    else:
        print("ffmpeg not found; frames in", frames)


def rotate_z(vol, deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    nz, ny, nx = vol.shape
    cy, cx = (ny - 1) * 0.5, (nx - 1) * 0.5
    yy, xx = np.mgrid[0:ny, 0:nx]
    xr = c * (xx - cx) - s * (yy - cy) + cx
    yr = s * (xx - cx) + c * (yy - cy) + cy
    xi = np.clip(np.rint(xr).astype(int), 0, nx - 1)
    yi = np.clip(np.rint(yr).astype(int), 0, ny - 1)
    return vol[:, yi, xi]


def write_rotate(vol, out, fps):
    rot = out / "rotate"
    rot.mkdir(parents=True, exist_ok=True)
    solid = (vol > 0).astype(np.float64)
    for i, deg in enumerate(range(0, 360, 10)):
        mip = rotate_z(solid, deg).max(axis=1)
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.imshow(mip, origin="lower", cmap="gray", interpolation="nearest")
        ax.set_title("MIP az=%d" % deg)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(rot / ("frame_%04d.png" % i), dpi=90)
        plt.close(fig)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.check_call(
            [
                ffmpeg, "-y", "-framerate", str(fps), "-i",
                str(rot / "frame_%04d.png"), "-pix_fmt", "yuv420p",
                str(out / "dendrite3d_rotate.mp4"),
            ]
        )
        print("wrote", out / "dendrite3d_rotate.mp4")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=int, default=60)
    args = p.parse_args()
    out = args.out or (args.run / "present")
    out.mkdir(parents=True, exist_ok=True)
    render(args.run, out, args.fps)


if __name__ == "__main__":
    main()

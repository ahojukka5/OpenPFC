#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render issue #8 inverse-homogenization showcase frames.

Reads MPI-IO `h` bricks plus `history.csv` from a scratch run directory.
Writes PNG stills, a 3x3 periodic tile, an optional affine-stretch still,
and an MP4/WebM when ffmpeg is on PATH. Solver frames are used as-is;
nothing is interpolated between iterations.

    python3 apps/inverse_homogenization/scripts/render_inverse_showcase.py \\
        --run /scratch/project_462001519/juaho/openpfc-showcase/inverse2d_JOB
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

from field_io import GridSpec, read_bin  # noqa: E402


def load_history(path: Path) -> dict[str, list[float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out: dict[str, list[float]] = {k: [] for k in rows[0]} if rows else {}
    for row in rows:
        for k, v in row.items():
            try:
                out[k].append(float(v))
            except ValueError:
                out[k].append(float("nan"))
    return out


def load_h(fields: Path, man: dict, idx: int) -> np.ndarray:
    name = man["pattern"].replace("{field}", "h").replace("{index:04d}", f"{idx:04d}")
    g = GridSpec(nx=man["nx"], ny=man["ny"], nz=man["nz"], dx=man["dx"], dy=man["dx"])
    cube = read_bin(fields / name, g)
    return cube[g.nz // 2]


def tile3(a: np.ndarray) -> np.ndarray:
    return np.tile(a, (3, 3))


def warp_auxetic(a: np.ndarray, ey: float = 0.12, nu: float = -0.1) -> np.ndarray:
    """Pull along y; lateral stretch uses the computed nu, not appearance."""
    ny, nx = a.shape
    ex = -nu * ey
    out_ny = max(1, int(round(ny * (1.0 + ey))))
    out_nx = max(1, int(round(nx * (1.0 + ex))))
    yy, xx = np.mgrid[0:out_ny, 0:out_nx]
    y0 = (yy / max(1.0 + ey, 1e-12)) % ny
    x0 = (xx / max(1.0 + ex, 1e-12)) % nx
    return a[y0.astype(int) % ny, x0.astype(int) % nx]


def render(run: Path, out: Path, fps: int) -> None:
    fields = run / "fields"
    mans = sorted(fields.glob("*_manifest.json"))
    if not mans:
        raise SystemExit(f"{fields}: no manifest")
    man = json.loads(mans[0].read_text())
    hist_path = run / "history.csv"
    hist = load_history(hist_path) if hist_path.exists() else {}
    n = len(man.get("steps", []))
    if n == 0:
        raise SystemExit("manifest has no steps")

    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    last = load_h(fields, man, n - 1)
    vmin, vmax = 0.0, 1.0
    nu_final = float(hist.get("nu_eff", [0.0])[-1]) if hist else 0.0

    for i in range(n):
        h = load_h(fields, man, i)
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2),
                                  gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
        ax = axes[0]
        ax.imshow(h, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        step = man["steps"][i] if i < len(man["steps"]) else i
        ax.set_title(f"h, step {step}")
        ax.set_xticks([]); ax.set_yticks([])
        ax = axes[1]
        ax.imshow(tile3(h > 0.5), origin="lower", cmap="cividis", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title("thresholded 3×3 tile")
        ax.set_xticks([]); ax.set_yticks([])
        ax = axes[2]
        if hist:
            ax.plot(hist.get("step", []), hist.get("J", []), color="0.2", label="J")
            ax.set_ylabel("J")
            ax2 = ax.twinx()
            ax2.plot(hist.get("step", []), hist.get("nu_eff", []), color="#b04830",
                     label=r"$\nu_{\mathrm{eff}}$")
            ax2.axhline(0.0, color="0.5", lw=0.6)
            ax2.set_ylabel(r"$\nu_{\mathrm{eff}}$")
            ax.axvline(step, color="0.4", lw=0.8)
        ax.set_xlabel("step")
        fig.tight_layout()
        fig.savefig(frames_dir / f"frame_{i:04d}.png", dpi=120)
        plt.close(fig)

    still = out / "final_cell.png"
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(last, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.set_title(f"final h, $\\nu_{{eff}}$={nu_final:.3f}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(still, dpi=160)
    plt.close(fig)

    tile_path = out / "final_tile.png"
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(tile3(last > 0.5), origin="lower", cmap="cividis", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title("final thresholded 3×3")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(tile_path, dpi=160)
    plt.close(fig)

    stretch = out / "final_stretch.png"
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(warp_auxetic(last > 0.5, ey=0.15, nu=nu_final), origin="lower",
              cmap="cividis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"affine stretch using $\\nu_{{eff}}$={nu_final:.3f}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(stretch, dpi=160)
    plt.close(fig)

    summary = out / "summary.md"
    summary.write_text(
        f"# Inverse 2-D showcase\n\n"
        f"- run: `{run}`\n"
        f"- grid: {man['nx']}×{man['ny']}×{man['nz']}\n"
        f"- frames: {n}\n"
        f"- final nu_eff (from homogenizer CSV): {nu_final:.6g}\n"
        f"- color scale: fixed h in [0,1]; morphology is solver output\n"
    )
    print(f"wrote {still}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found; PNG frames are in", frames_dir)
        return
    mp4 = out / "inverse2d.mp4"
    webm = out / "inverse2d.webm"
    subprocess.check_call(
        [ffmpeg, "-y", "-framerate", str(fps), "-i",
         str(frames_dir / "frame_%04d.png"), "-pix_fmt", "yuv420p", str(mp4)]
    )
    subprocess.check_call(
        [ffmpeg, "-y", "-framerate", str(fps), "-i",
         str(frames_dir / "frame_%04d.png"), "-c:v", "libvpx-vp9", "-crf", "32",
         "-b:v", "0", str(webm)]
    )
    print(f"wrote {mp4} {webm}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()
    out = args.out or (args.run / "present")
    out.mkdir(parents=True, exist_ok=True)
    render(args.run, out, args.fps)


if __name__ == "__main__":
    main()

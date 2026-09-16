#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Split-screen elastic-off vs on dendrite animation (issue #11).

Color scales are fixed across frames. Fields are solver bricks, not
interpolated. Does not treat historical 16.7% as a target.

    python3 apps/alloy_dendrite_elastic/scripts/render_dendrite_showcase.py \\
        --run /scratch/.../dendrite2d_JOB
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


def load_csv(path: Path) -> dict[str, list[float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out: dict[str, list[float]] = {k: [] for k in rows[0]} if rows else {}
    for row in rows:
        for k, v in row.items():
            try:
                out.setdefault(k, []).append(float(v))
            except ValueError:
                pass
    return out


def load_plane(d: Path, field: str, idx: int) -> np.ndarray:
    man = json.loads(next(d.glob("*_manifest.json")).read_text())
    g = GridSpec(nx=man["nx"], ny=man["ny"], nz=man["nz"], dx=man["dx"], dy=man["dx"])
    name = man["pattern"].replace("{field}", field).replace("{index:04d}", f"{idx:04d}")
    return read_bin(d / name, g)[g.nz // 2]


def render(run: Path, out: Path, fps: int) -> None:
    off_d = run / "fields_off"
    on_d = run / "fields_on"
    man = json.loads(next(off_d.glob("*_manifest.json")).read_text())
    n = len(man["times"])
    off_hist = load_csv(run / "off.csv") if (run / "off.csv").exists() else {}
    on_hist = load_csv(run / "on.csv") if (run / "on.csv").exists() else {}
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)

    phis_off = [load_plane(off_d, "phi", i) for i in range(n)]
    phis_on = [load_plane(on_d, "phi", i) for i in range(n)]
    us_on = [load_plane(on_d, "U", i) for i in range(n)]
    vmin_u = min(float(a.min()) for a in us_on)
    vmax_u = max(float(a.max()) for a in us_on)

    for i in range(n):
        fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.5))
        for ax, img, title, vmin, vmax, cmap in (
            (axes[0, 0], phis_off[i], "phi elastic off", -1, 1, "coolwarm"),
            (axes[0, 1], phis_on[i], "phi elastic on", -1, 1, "coolwarm"),
            (axes[0, 2], phis_on[i] - phis_off[i], r"$\Delta\phi$", -0.2, 0.2, "coolwarm"),
            (axes[1, 0], us_on[i], "U (on)", vmin_u, vmax_u, "viridis"),
        ):
            ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
        try:
            fel = load_plane(on_d, "f_el", i)
            axes[1, 1].imshow(fel, origin="lower", cmap="magma", interpolation="nearest")
            axes[1, 1].set_title(r"$f_{\mathrm{el}}$ (on)")
        except Exception:
            axes[1, 1].set_title("no f_el")
        axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])
        ax = axes[1, 2]
        if off_hist and on_hist:
            ax.plot(off_hist.get("t", []), off_hist.get("v_tip", []), label="off")
            ax.plot(on_hist.get("t", []), on_hist.get("v_tip", []), label="on")
            ax.legend(fontsize=8)
        ax.set_title("tip velocity")
        ax.set_xlabel("t")
        fig.suptitle(f"t={man['times'][i]:g}  (device Green, isothermal)")
        fig.tight_layout()
        fig.savefig(frames / f"frame_{i:04d}.png", dpi=110)
        plt.close(fig)

    shutil.copy(frames / f"frame_{n-1:04d}.png", out / "dendrite_still.png")
    (out / "summary.md").write_text(
        f"# Dendrite 2-D showcase\n\n- run `{run}`\n- grid {man['nx']}²\n"
        f"- frames {n}\n- color scales fixed; device Green path\n"
        f"- not a 16.7% science target\n"
    )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.check_call(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i",
             str(frames / "frame_%04d.png"), "-pix_fmt", "yuv420p",
             str(out / "dendrite2d.mp4")]
        )
        print("wrote", out / "dendrite2d.mp4")
    else:
        print("ffmpeg not found; frames in", frames)


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

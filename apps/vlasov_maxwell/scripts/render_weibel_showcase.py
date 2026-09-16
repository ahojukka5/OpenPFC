#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render issue #10 Weibel reduced frames.

Uses solver projections and the ledger CSV. Does not interpolate plasma
states. Nonlinear saturation is labelled as demonstration unless the
linear window in the summary matches the oracle.

    python3 apps/vlasov_maxwell/scripts/render_weibel_showcase.py --run DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: Path) -> dict[str, list[float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out: dict[str, list[float]] = {k: [] for k in rows[0]} if rows else {}
    for row in rows:
        for k, v in row.items():
            try:
                out[k].append(float(v))
            except ValueError:
                pass
    return out


def load_plane(d: Path, man: dict, field: str, idx: int, nx: int, ny: int) -> np.ndarray:
    name = man["pattern"].replace("{field}", field).replace("{index:04d}", f"{idx:04d}")
    raw = np.fromfile(d / name, dtype="<f8")
    return raw.reshape((nx, ny), order="F").T


def load_line(d: Path, man: dict, field: str, idx: int, nx: int) -> np.ndarray:
    name = man["pattern"].replace("{field}", field).replace("{index:04d}", f"{idx:04d}")
    return np.fromfile(d / name, dtype="<f8")[:nx]


def render(run: Path, out: Path, fps: int) -> None:
    red = run / "reduced"
    man = json.loads(next(red.glob("*_reduced_manifest.json")).read_text())
    times = man["times"]
    nx, nvx = man["nx"], man["nvx"]
    ledger = load_csv(run / "ledger.csv") if (run / "ledger.csv").exists() else {}
    summary = (run / "summary.csv").read_text() if (run / "summary.csv").exists() else ""
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    planes = [load_plane(red, man, "f_xvx", i, nx, nvx) for i in range(len(times))]
    vmax = max(float(p.max()) for p in planes)

    for i, t in enumerate(times):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
        ax = axes[0]
        ax.imshow(planes[i], origin="lower", aspect="auto", cmap="magma",
                  vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"$f(x,v_x)$  t={t:g}")
        ax.set_xlabel("x"); ax.set_ylabel(r"$v_x$")
        ax = axes[1]
        bz = load_line(red, man, "Bz", i, nx)
        ey = load_line(red, man, "Ey", i, nx)
        ax.plot(bz, label=r"$B_z$")
        ax.plot(ey, label=r"$E_y$")
        ax.set_title("fields")
        ax.legend(fontsize=8)
        ax = axes[2]
        if ledger:
            tcol = ledger.get("t", [])
            em = ledger.get("energy_bz", ledger.get("energy_em", []))
            ax.semilogy(tcol, np.maximum(np.abs(em), 1e-30))
            ax.axvline(t, color="0.4", lw=0.8)
        ax.set_title(r"magnetic energy (solver)")
        ax.set_xlabel("t")
        fig.tight_layout()
        fig.savefig(frames / f"frame_{i:04d}.png", dpi=120)
        plt.close(fig)

    still = out / "weibel_still.png"
    shutil.copy(frames / f"frame_{len(times)-1:04d}.png", still)
    (out / "summary.md").write_text(
        f"# Weibel showcase\n\n- run `{run}`\n- grid {nx}×{nvx}×{man['nvy']}\n"
        f"- frames {len(times)}\n- color scale fixed across frames\n"
        f"- post-linear interpretation is a demonstration unless the "
        f"summary linear window matches the oracle\n\n```\n{summary}\n```\n"
    )
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found; frames in", frames)
        return
    subprocess.check_call(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i",
         str(frames / "frame_%04d.png"), "-pix_fmt", "yuv420p",
         str(out / "weibel.mp4")]
    )
    print("wrote", out / "weibel.mp4")


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

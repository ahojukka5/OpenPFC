#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Encode MHD conference frame sequences to H.264 MP4.

Does not interpolate fields. Repeats the last frame HOLD_LAST times so
talk playback is not a two-second flash.

Example:
  python3 make_mhd_conference_animation.py \\
    --frames-dir figures/generated/hero_evidence \\
    --mp4 figures/evidence/hero_reconnection_eta0005_n512.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mhd_conference_catalog as cat
import mhd_conference_style as st


def list_frames(directory):
    names = sorted(
        n for n in os.listdir(directory)
        if n.startswith("frame_") and n.endswith(".png")
    )
    return [os.path.join(directory, n) for n in names]


def encode_mp4(frames, mp4, fps, ffmpeg):
    if not frames:
        raise ValueError("no frames")
    os.makedirs(os.path.dirname(os.path.abspath(mp4)) or ".", exist_ok=True)
    staging = mp4 + ".staging"
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    try:
        i = 0
        for path in frames:
            os.symlink(os.path.abspath(path),
                       os.path.join(staging, "f_%05d.png" % i))
            i += 1
        last = frames[-1]
        for _ in range(st.HOLD_LAST):
            os.symlink(os.path.abspath(last),
                       os.path.join(staging, "f_%05d.png" % i))
            i += 1
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(staging, "f_%05d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-movflags", "+faststart",
            mp4,
        ]
        subprocess.check_call(cmd)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return mp4


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--mp4", required=True)
    p.add_argument("--fps", type=int, default=st.HERO_FPS)
    p.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    p.add_argument("--kind", choices=("evidence", "showcase"),
                   default="evidence")
    p.add_argument("--asset", default="hero")
    args = p.parse_args(argv)

    frames = list_frames(args.frames_dir)
    if not frames:
        print("no frame_*.png in", args.frames_dir, file=sys.stderr)
        return 1
    ffmpeg = args.ffmpeg
    if shutil.which(ffmpeg) is None:
        print("ffmpeg not found:", ffmpeg, file=sys.stderr)
        return 1
    encode_mp4(frames, args.mp4, args.fps, ffmpeg)
    cat.write_json(args.mp4 + ".json", {
        "kind": args.kind,
        "asset": args.asset,
        "window": cat.window_class_for_time(None, args.kind),
        "fps": args.fps,
        "n_unique_frames": len(frames),
        "hold_last": st.HOLD_LAST,
        "frames_dir": os.path.abspath(args.frames_dir),
        "mp4": os.path.abspath(args.mp4),
        "note": "No field interpolation. Last frame is held for talk pacing.",
    })
    print("wrote", args.mp4)
    return 0


if __name__ == "__main__":
    sys.exit(main())

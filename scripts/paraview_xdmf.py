#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Open an OpenPFC XDMF time series in ParaView.

Run with ParaView's Python, not system Python:

    pvpython scripts/paraview_xdmf.py path/to/fields/inverse2d.xdmf
    pvbatch  scripts/paraview_xdmf.py path/to/fields/off.xdmf --animation out.avi

File → Open of the same ``.xdmf`` in the ParaView GUI is the usual path.
This script only wraps ``paraview.simple``.
"""

from __future__ import print_function

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Load an OpenPFC XDMF series with paraview.simple."
    )
    parser.add_argument("xdmf", help="XDMF collection written next to .bin dumps")
    parser.add_argument(
        "--animation",
        help="optional animation path (pvbatch SaveAnimation)",
    )
    parser.add_argument("--array", default=None, help="point array to color by")
    args = parser.parse_args()
    xdmf = os.path.abspath(args.xdmf)
    if not os.path.isfile(xdmf):
        sys.exit("missing %s" % xdmf)

    from paraview.simple import (  # noqa: E402
        ColorBy,
        GetAnimationScene,
        GetColorTransferFunction,
        OpenDataFile,
        Render,
        ResetCamera,
        SaveAnimation,
        Show,
        UpdatePipeline,
    )

    src = OpenDataFile(xdmf)
    UpdatePipeline()
    rep = Show(src)
    arrays = list(src.PointData.keys()) if hasattr(src, "PointData") else []
    name = args.array or (arrays[0] if arrays else None)
    if name:
        ColorBy(rep, ("POINTS", name))
        lut = GetColorTransferFunction(name)
        if lut is not None:
            lut.RescaleTransferFunctionToDataRange(True)
    ResetCamera()
    Render()
    if args.animation:
        scene = GetAnimationScene()
        scene.UpdateAnimationUsingDataTimeSteps()
        SaveAnimation(args.animation, ImageResolution=[1920, 1080], FrameRate=60)
        print("wrote", args.animation)
    else:
        print("loaded", xdmf, "arrays:", ", ".join(arrays) if arrays else "(none)")


if __name__ == "__main__":
    main()

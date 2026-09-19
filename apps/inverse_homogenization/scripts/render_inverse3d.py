#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ParaView stills from inverse3d XDMF bricks (solver snapshots only).

Usage:
  pvpython render_inverse3d.py /path/to/fields /path/to/stills
Expects h_init.xdmf, h_final.xdmf, h_thresh.xdmf from openpfc_inverse_homogenize.
"""
import os
import sys

def render_one(xdmf, png, iso=0.5):
    from paraview.simple import (
        XMLXdmfReader,
        Contour,
        GetActiveViewOrCreate,
        Show,
        ColorBy,
        SaveScreenshot,
        ResetCamera,
        Hide,
    )
    if not os.path.isfile(xdmf):
        print("missing", xdmf)
        return
    r = XMLXdmfReader(FileName=[xdmf])
    r.UpdatePipeline()
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [1920, 1080]
    view.Background = [1, 1, 1]
    disp = Show(r, view)
    ColorBy(disp, ("POINTS", "h"))
    try:
        c = Contour(Input=r)
        c.ContourBy = ["POINTS", "h"]
        c.Isosurfaces = [iso]
        Show(c, view)
        Hide(r, view)
    except Exception:
        pass
    ResetCamera(view)
    view.CameraViewUp = [0.2, 0.3, 0.9]
    os.makedirs(os.path.dirname(png), exist_ok=True)
    SaveScreenshot(png, view, ImageResolution=[1920, 1080])
    print("wrote", png)

def main():
    if len(sys.argv) < 3:
        print("usage: pvpython render_inverse3d.py FIELDS_DIR STILLS_DIR",
              file=sys.stderr)
        return 2
    src, dest = sys.argv[1], sys.argv[2]
    for name in ("h_init", "h_final", "h_thresh"):
        render_one(os.path.join(src, name + ".xdmf"),
                   os.path.join(dest, name + ".png"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

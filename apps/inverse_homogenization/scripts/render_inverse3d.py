#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""3-D ParaView isosurfaces from inverse3d XDMF (solver snapshots only).

Usage:
  pvpython render_inverse3d.py FIELDS_DIR STILLS_DIR [--iso 0.5] [--nu-xy -0.38]

Expects h_init.xdmf, h_final.xdmf, h_thresh.xdmf and optional h_%04d.bin
bricks (Fortran float64, matching the XDMF 3DCoRectMesh layout).
Does not invent material states. Affine deformation is labelled
HOMOGENIZED AFFINE, not local displacement physics.
"""
from __future__ import print_function

import argparse
import glob
import os
import sys


def _offscreen():
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
    os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")


def _write_xdmf(bin_path, xdmf_path, nx, ny, nz):
    name = os.path.basename(bin_path)
    with open(xdmf_path, "w") as f:
        f.write(
            '<?xml version="1.0"?>\n'
            '<Xdmf Version="2.0"><Domain><Grid Name="g" GridType="Uniform">\n'
            '<Topology TopologyType="3DCoRectMesh" Dimensions="%d %d %d"/>\n'
            '<Geometry Type="ORIGIN_DXDYDZ">\n'
            '<DataItem Format="XML" Dimensions="3">0 0 0</DataItem>\n'
            '<DataItem Format="XML" Dimensions="3">1 1 1</DataItem>\n'
            "</Geometry>\n"
            '<Attribute Name="h" Center="Node">\n'
            '<DataItem Format="Binary" DataType="Float" Precision="8" '
            'Endian="Little" Dimensions="%d %d %d">%s</DataItem>\n'
            "</Attribute>\n"
            "</Grid></Domain></Xdmf>\n" % (nz, ny, nx, nz, ny, nx, name)
        )


def _setup_view():
    from paraview.simple import GetActiveViewOrCreate

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [1920, 1080]
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 1
    return view


def _contour_from_xdmf(xdmf, iso):
    from paraview.simple import OpenDataFile, Contour

    r = OpenDataFile(xdmf)
    r.UpdatePipeline()
    c = Contour(Input=r)
    try:
        c.ContourBy = ["POINTS", "h"]
    except Exception:
        c.SelectInputScalars = ["POINTS", "h"]
    c.Isosurfaces = [iso]
    c.UpdatePipeline()
    return r, c


def _show_iso(c, view, color=(0.85, 0.45, 0.12)):
    from paraview.simple import Show, GetColorTransferFunction, Hide

    disp = Show(c, view)
    disp.Representation = "Surface"
    disp.DiffuseColor = list(color)
    disp.Specular = 0.2
    return disp


def _camera(view):
    from paraview.simple import ResetCamera

    ResetCamera(view)
    view.CameraPosition = [180, -160, 140]
    view.CameraFocalPoint = [32, 32, 60]
    view.CameraViewUp = [0.15, 0.25, 0.95]
    view.CameraViewAngle = 22
    view.Update()


def render_one(xdmf, png, iso, label):
    from paraview.simple import (
        SaveScreenshot,
        Hide,
        Text,
        Show,
        Delete,
        GetActiveViewOrCreate,
    )

    if not os.path.isfile(xdmf):
        print("missing", xdmf)
        return
    view = _setup_view()
    r, c = _contour_from_xdmf(xdmf, iso)
    _show_iso(c, view)
    txt = Text()
    txt.Text = label
    tdisp = Show(txt, view)
    tdisp.Color = [0.1, 0.1, 0.1]
    tdisp.FontSize = 18
    try:
        tdisp.WindowLocation = "Upper Left Corner"
    except Exception:
        pass
    _camera(view)
    os.makedirs(os.path.dirname(os.path.abspath(png)) or ".", exist_ok=True)
    SaveScreenshot(png, view, ImageResolution=[1920, 1080])
    print("wrote", png)
    Hide(c, view)
    Hide(txt, view)
    Delete(c)
    Delete(r)
    Delete(txt)


def render_affine(xdmf, dest_dir, iso, nu_xy, nframe=16):
    """HOMOGENIZED AFFINE box warp. Not local displacement physics."""
    from paraview.simple import (
        Box,
        Transform,
        Show,
        Hide,
        Text,
        SaveScreenshot,
        Delete,
        GetActiveViewOrCreate,
        ColorBy,
    )

    if not os.path.isfile(xdmf):
        return []
    view = _setup_view()
    r, c = _contour_from_xdmf(xdmf, iso)
    _show_iso(c, view, color=(0.2, 0.45, 0.75))
    box = Box()
    box.XLength = 64.0
    box.YLength = 64.0
    box.ZLength = 121.0
    box.Center = [32.0, 32.0, 60.5]
    bdisp = Show(box, view)
    bdisp.Representation = "Wireframe"
    bdisp.LineWidth = 2
    bdisp.AmbientColor = [0.1, 0.1, 0.1]
    bdisp.DiffuseColor = [0.1, 0.1, 0.1]
    txt = Text()
    tdisp = Show(txt, view)
    tdisp.Color = [0.05, 0.05, 0.05]
    tdisp.FontSize = 16
    try:
        tdisp.WindowLocation = "Upper Left Corner"
    except Exception:
        pass
    paths = []
    os.makedirs(dest_dir, exist_ok=True)
    emax = 0.12
    for i in range(nframe):
        e = emax * i / float(nframe - 1)
        ey = -float(nu_xy) * e
        # loading in x: eps_x=e, eps_y=-nu_xy*e; auxetic nu_xy<0 => ey>0
        tr = Transform(Input=c)
        tr.Transform.Scale = [1.0 + e, 1.0 + ey, 1.0]
        Hide(c, view)
        _show_iso(tr, view, color=(0.85, 0.35, 0.1))
        txt.Text = (
            "HOMOGENIZED AFFINE (not local displacement)\n"
            "uniaxial eps_x=%.3f  eps_y=%.3f  nu_xy=%.3f  "
            "transverse EXPANSION" % (e, ey, nu_xy)
        )
        _camera(view)
        png = os.path.join(dest_dir, "affine_%02d.png" % i)
        SaveScreenshot(png, view, ImageResolution=[1920, 1080])
        paths.append(png)
        print("wrote", png)
        Hide(tr, view)
        Delete(tr)
        Show(c, view)
    Hide(c, view)
    Hide(box, view)
    Hide(txt, view)
    Delete(c)
    Delete(r)
    Delete(box)
    Delete(txt)
    return paths


def main(argv=None):
    _offscreen()
    p = argparse.ArgumentParser()
    p.add_argument("fields_dir")
    p.add_argument("stills_dir")
    p.add_argument("--iso", type=float, default=0.5)
    p.add_argument("--nu-xy", type=float, default=None)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=121)
    p.add_argument("--animate", action="store_true")
    p.add_argument("--affine", action="store_true")
    args = p.parse_args(argv)

    src = args.fields_dir
    dest = args.stills_dir
    os.makedirs(dest, exist_ok=True)

    stills = [
        ("h_init.xdmf", "h_init_iso.png", "initial continuous"),
        ("h_final.xdmf", "h_final_iso.png", "final continuous"),
        ("h_thresh.xdmf", "h_thresh_iso.png", "final thresholded h>0.5"),
    ]
    mid = os.path.join(src, "h_0150.xdmf")
    if os.path.isfile(os.path.join(src, "h_0150.bin")):
        _write_xdmf(
            os.path.join(src, "h_0150.bin"),
            os.path.join(src, "h_0150.xdmf"),
            args.nx,
            args.ny,
            args.nz,
        )
        stills.insert(1, ("h_0150.xdmf", "h_mid_iso.png", "intermediate step 150"))

    for fname, png, label in stills:
        render_one(os.path.join(src, fname), os.path.join(dest, png), args.iso, label)

    if args.animate:
        bins = sorted(glob.glob(os.path.join(src, "h_[0-9][0-9][0-9][0-9].bin")))
        anim_dir = os.path.join(dest, "frames")
        os.makedirs(anim_dir, exist_ok=True)
        for b in bins:
            step = os.path.splitext(os.path.basename(b))[0]
            xd = os.path.join(src, step + ".xdmf")
            if not os.path.isfile(xd):
                _write_xdmf(b, xd, args.nx, args.ny, args.nz)
            render_one(
                xd,
                os.path.join(anim_dir, step + ".png"),
                args.iso,
                "optimization  " + step,
            )

    if args.affine and args.nu_xy is not None:
        render_affine(
            os.path.join(src, "h_thresh.xdmf"),
            os.path.join(dest, "affine"),
            args.iso,
            args.nu_xy,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render manifest-indexed fields ending at the exact certified material.

Requires NumPy; rendering additionally needs PyVista/VTK and ffmpeg. Physical
spacing and a fixed camera are shared by every frame. No interpolated time
states or affine deformations are manufactured.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admit(manifest_path):
    m = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    shape = [m[k] for k in ('nx','ny','nz')]
    if (m['order'] != 'fortran' or m['dtype'] != 'float64' or
            min(shape) < 1 or not np.isfinite(m['dx']) or m['dx'] <= 0):
        raise ValueError('unsupported field geometry')
    steps = m['steps']
    if not steps or steps != sorted(set(steps)) or steps[0] < 0:
        raise ValueError('snapshot steps must increase strictly')
    material = json.loads((root/'h_final_material.json').read_text())
    if not (material['inverse_converged'] and material['termination']=='CONVERGED'
            and material['elasticity_converged'] and material['diagnostics_valid']
            and material['accepted_step']==steps[-1]
            and material['grid']==shape and material['spacing']==m['dx']):
        raise ValueError('material is not the declared certified terminal state')
    frames = []
    for i,step in enumerate(steps):
        path = root/m['pattern'].format(field='h',index=i)
        if path.stat().st_size != 8*int(np.prod(shape)):
            raise ValueError('snapshot size does not match grid')
        values = np.fromfile(path,dtype='<f8')
        if not np.isfinite(values).all() or values.min()<0 or values.max()>1:
            raise ValueError('invalid material field')
        frames.append(dict(index=i,step=step,file=path.name,sha256=sha(path)))
    if frames[-1]['sha256'] != sha(root/'h_final.bin'):
        raise ValueError('terminal snapshot differs from certified final field')
    return m,dict(manifest_sha256=sha(manifest_path),material_sha256=sha(root/'h_final_material.json'),
                  final_sha256=sha(root/'h_final.bin'),accepted_step=steps[-1],frames=frames,
                  physical_extent=[n*m['dx'] for n in shape],iso=.5,
                  representation='periodic h=0.5 isosurface; no smoothing or time interpolation')


def render(manifest_path,m,record,output,ffmpeg):
    os.environ.setdefault('PYVISTA_OFF_SCREEN','true')
    import pyvista as pv
    extent = np.array(record['physical_extent'])
    center = extent/2
    plot = pv.Plotter(off_screen=True,window_size=(1280,960))
    plot.set_background('white')
    plot.camera_position = [center+np.array([2.1,-2.5,1.6])*max(extent),center,(0,0,1)]
    plot.enable_parallel_projection()
    plot.camera.parallel_scale = .7*max(extent)
    bounds = (0,extent[0],0,extent[1],0,extent[2])
    plot.add_mesh(pv.Box(bounds=bounds),style='wireframe',color='gray',line_width=1,reset_camera=False)
    plot.add_axes()
    dimensions = tuple(m[k]+1 for k in ('nx','ny','nz'))
    grid = pv.ImageData(dimensions=dimensions,spacing=(m['dx'],)*3)
    for frame in record['frames']:
        h = np.fromfile(manifest_path.parent/frame['file'],dtype='<f8').reshape((m['nz'],m['ny'],m['nx']))
        grid.point_data['h'] = np.pad(h,((0,1),(0,1),(0,1)),mode='wrap').ravel()
        contour = grid.contour([.5],scalars='h')
        if contour.n_points:
            plot.add_mesh(contour,color='#d68a30',name='material',smooth_shading=False,
                          reset_camera=False)
        else:
            plot.remove_actor('material',render=False)
        label = f"Accepted state {frame['step']}"
        if frame['step']==record['accepted_step']:
            label += ' | CONVERGED'
        plot.add_text(label,position='upper_left',font_size=18,color='black',name='label')
        plot.add_text(f"{m['nx']}-grid frozen inverse | h = 0.5\nPhysical cell {extent[0]:g} x {extent[1]:g} x {extent[2]:g}",
                      position='lower_left',font_size=12,color='black',name='caption')
        png = output/f"frame_{frame['index']:04d}.png"
        plot.screenshot(png)
        frame['png_sha256'] = sha(png)
        print('rendered',frame['step'],flush=True)
    plot.close()
    movie = output/'certified-trajectory.mp4'
    subprocess.run([ffmpeg,'-y','-framerate','5','-i',str(output/'frame_%04d.png'),
                    '-vf','tpad=stop_mode=clone:stop_duration=2,fps=30','-c:v','libx264',
                    '-pix_fmt','yuv420p','-crf','18',str(movie)],check=True)
    record.update(movie_sha256=sha(movie),pyvista=pv.__version__,vtk=pv.vtk_version_info,
                  camera=[(center+np.array([2.1,-2.5,1.6])*max(extent)).tolist(),center.tolist(),[0,0,1]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--ffmpeg',default='ffmpeg')
    args = p.parse_args()
    m,record = admit(args.manifest)
    args.output.mkdir(parents=True,exist_ok=False)
    record['renderer_sha256'] = sha(Path(__file__))
    if not args.validate_only:
        render(args.manifest,m,record,args.output,args.ffmpeg)
    (args.output/'frames.json').write_text(json.dumps(record,indent=2)+'\n')
    print('CERTIFIED_TRAJECTORY_OK')


if __name__=='__main__':
    main()

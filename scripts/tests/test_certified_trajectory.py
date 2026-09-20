#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exercise final-state admission through the actual visualization entrypoint."""
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'apps/inverse_homogenization/scripts/render_certified_trajectory.py'


def test_terminal_identity_and_physical_spacing(tmp_path):
    fields = tmp_path/'fields'
    fields.mkdir()
    data = struct.pack('<8d',*([.25,.75]*4))
    (fields/'h_0000.bin').write_bytes(data)
    (fields/'h_final.bin').write_bytes(data)
    (fields/'manifest.json').write_text(json.dumps(dict(nx=2,ny=2,nz=2,dx=.5,
        order='fortran',dtype='float64',steps=[120],pattern='h_{index:04d}.bin')))
    report = dict(inverse_converged=True,termination='CONVERGED',elasticity_converged=True,
                  diagnostics_valid=True,accepted_step=120,grid=[2,2,2],spacing=.5)
    (fields/'h_final_material.json').write_text(json.dumps(report))
    command = [sys.executable,str(SCRIPT),str(fields/'manifest.json')]
    subprocess.run(command+[str(tmp_path/'valid'),'--validate-only'],check=True)
    record = json.loads((tmp_path/'valid/frames.json').read_text())
    assert record['physical_extent']==[1,1,1]
    assert record['frames'][-1]['sha256']==record['final_sha256']
    (fields/'h_final.bin').write_bytes(struct.pack('<8d',*([.5]*8)))
    result = subprocess.run(command+[str(tmp_path/'invalid'),'--validate-only'],capture_output=True)
    assert result.returncode != 0
    assert not (tmp_path/'invalid').exists()

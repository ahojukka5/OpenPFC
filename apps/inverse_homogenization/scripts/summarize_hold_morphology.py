#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reduce native forward diagnostics at hold states 0, 300 and 1693.

Input is the committed raw text bundle, not live scratch. Native CPU matrices
are the unpenalized material response, distinct from optimization SIMP tensors.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def parse(text):
    lines = text.splitlines()
    matrix = lambda index: np.array([[float(v) for v in line.split()]
                                     for line in lines[index+1:index+7]])
    c = matrix(next(i for i,s in enumerate(lines) if s.startswith('C_H (')))
    compliance = matrix(lines.index('S = C_H^{-1}'))
    np.testing.assert_allclose(np.linalg.inv(c), compliance, rtol=2e-6, atol=1e-8)
    line = lambda start: next(s for s in lines if s.startswith(start)).split()
    tokens = line('nu_shortcut ')
    nus = {tokens[i]:float(tokens[i+1]) for i in range(0,len(tokens),2)}
    for name,i,j in [('nu_xy',0,1),('nu_xz',0,2),('nu_yx',1,0),
                     ('nu_yz',1,2),('nu_zx',2,0),('nu_zy',2,1)]:
        np.testing.assert_allclose(nus[name],-compliance[j,i]/compliance[i,i],rtol=2e-8)
    volume = line('volume_fraction ')
    assert volume[3] == 'yes' and volume[5] == 'yes'
    elastic = line('elasticity_converged ')
    assert elastic[1] == 'yes'
    topology, opening = line('solid_components '), line('opening_loss_r1 ')
    return dict(stiffness=c.tolist(),compliance=compliance.tolist(),poisson=nus,
                volume_fraction=float(volume[1]),elasticity_iterations=int(elastic[3]),
                elasticity_residual=float(elastic[5]),solid_components=int(topology[1]),
                void_components=int(topology[3]),solid_percolation=list(map(int,topology[5:8])),
                island_solid_fraction=float(topology[9]),grey_fraction=float(topology[11]),
                opening_loss_r1=float(opening[1]),opening_loss_r2=float(opening[3]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=Path);p.add_argument('gpu_reports',type=Path)
    p.add_argument('output',type=Path);args=p.parse_args()
    raw=json.loads(gzip.decompress(args.raw.read_bytes()))
    result={}
    for step in (0,300,1693):
        for mode in ('continuous','threshold'):
            name=f'state{step}_{mode}'
            assert raw[name+'/exit_code.txt'].strip()=='0'
            result[name]=parse(raw[name+'/run.log'])
    reports=json.loads(args.gpu_reports.read_text())
    errors={}
    for mode,key in [('continuous','h_final'),('threshold','h_thresh')]:
        cpu=np.array(result[f'state1693_{mode}']['stiffness'])
        gpu=np.array(reports[key]['stiffness_symmetric'])
        error=float(np.linalg.norm(cpu-gpu)/np.linalg.norm(gpu))
        assert error < 1e-8
        errors[mode]=error
    changes={}
    for mode in ('continuous','threshold'):
        first=np.array(result[f'state300_{mode}']['stiffness'])
        final=np.array(result[f'state1693_{mode}']['stiffness'])
        changes[mode]=float(np.linalg.norm(final-first)/np.linalg.norm(final))
    record=dict(cases=result,cpu_gpu_relative_tensor_error=errors,
                state300_to_final_tensor_error_relative_to_final=changes,
                raw_sha256=hashlib.sha256(args.raw.read_bytes()).hexdigest(),
                gpu_reports_sha256=hashlib.sha256(args.gpu_reports.read_bytes()).hexdigest(),
                scope='Unpenalized forward tensors and native voxel diagnostics; small frozen hold only')
    args.output.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()

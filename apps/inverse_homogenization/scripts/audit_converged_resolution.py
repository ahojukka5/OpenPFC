#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit the frozen converged 64/128 campaign (issue #98).

Material parsing reuses the qualified small-hold reduction. The live audit
checks fields; --archive reduces preserved text without claiming to recheck
omitted fields. Large fields are reproducible from the pinned seed/protocol.
"""
import argparse
import csv
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


def history(text):
    rows = list(csv.DictReader(l for l in text.splitlines() if l and not l.startswith('#')))
    assert [int(r['step']) for r in rows] == list(range(len(rows)))
    quiet = 0
    for r in rows:
        assert r['elasticity'] == '1'
        step = int(r['step'])
        assert int(r['frozen']) == (step >= 300)
        if step >= 300:
            assert float(r['simp_p']) == 2 and float(r['lambda_reg']) == .2
        passes = (step >= 300 and float(r['design_rms']) < 1e-4
                  and float(r['dJ_rel']) < 1e-6 and float(r['dC_rel']) < 1e-4)
        quiet = quiet + 1 if passes else 0
        assert int(r['conv_window']) == quiet
        assert int(r['candidate']) == (quiet >= 20)
        assert int(r['verified']) == (quiet >= 120)
    last = rows[-1]
    assert last['termination'] == 'CONVERGED' and quiet == 120
    assert all(r['termination'] == 'RUNNING' for r in rows[:-1])
    return dict(last=last, state300=rows[300], rows=len(rows),
                first_candidate=next(int(r['step']) for r in rows if r['candidate']=='1'),
                metric_precision='CSV values; stored fields were not dumped every accepted step')


def reduce(raw):
    result = {}
    for n in (64,128):
        prefix = str(n)+'/'
        rows = history(raw[prefix+'producer/history.csv'])
        cases = {}
        for state in ('state300','final'):
            for mode in ('continuous','threshold'):
                key = f'{state}_{mode}'
                assert raw[prefix+'material/'+key+'/exit_code.txt'].strip() == '0'
                cases[key] = parse(raw[prefix+'material/'+key+'/run.log'])
        errors = {}
        for mode,field in [('continuous','h_final'),('threshold','h_thresh')]:
            report = json.loads(raw[prefix+'producer/fields/'+field+'_material.json'])
            assert report['inverse_converged'] and report['diagnostics_valid']
            assert report['accepted_step'] == int(rows['last']['step'])
            assert report['grid'] == [n,n,121*n//64]
            assert report['spacing'] == 64/n
            gpu = np.array(report['stiffness_symmetric'])
            cpu = np.array(cases['final_'+mode]['stiffness'])
            errors[mode] = float(np.linalg.norm(cpu-gpu)/np.linalg.norm(gpu))
            assert errors[mode] < 1e-8
        changes = {}
        for mode in ('continuous','threshold'):
            a = np.array(cases['state300_'+mode]['stiffness'])
            b = np.array(cases['final_'+mode]['stiffness'])
            changes[mode] = float(np.linalg.norm(a-b)/np.linalg.norm(b))
        result[str(n)] = dict(history=rows, material=cases,
                             cpu_gpu_relative_tensor_error=errors,
                             state300_relative_tensor_error=changes)
    comparison = {}
    for mode in ('continuous','threshold'):
        a = np.array(result['64']['material']['final_'+mode]['stiffness'])
        b = np.array(result['128']['material']['final_'+mode]['stiffness'])
        comparison[mode] = float(np.linalg.norm(a-b)/np.linalg.norm(b))
    return dict(resolutions=result, relative_resolution_tensor_difference=comparison,
                physical_periodic_extent=[64,64,121],
                opening_radius_one_physical={str(n):result[str(n)]['material']['final_continuous'][
                    'opening_loss_r1' if n==64 else 'opening_loss_r2'] for n in (64,128)},
                scope='Unpenalized material tensors; two-grid differences are not an asymptotic convergence proof')


def collect(base):
    raw, checks, final = {}, {}, {}
    for n,pjob,mjob in ((64,22180900,22182461),(128,22180901,22182462)):
        producer = base/f'n0fixed{n}_{pjob}'
        material = base/f'material{n}_{mjob}'
        for label,root in (('producer',producer),('material',material)):
            for p in root.rglob('*'):
                if p.is_file() and p.suffix in ('.csv','.json','.txt','.sh','.sbatch','.md','.log','.err','.xdmf'):
                    raw[f'{n}/{label}/{p.relative_to(root)}'] = p.read_text(errors='replace')
        fields = producer/'fields'
        manifest = json.loads((fields/f'n0fixed{n}_manifest.json').read_text())
        assert manifest['dx'] == 64/n and manifest['order']=='fortran'
        step = int(history(raw[f'{n}/producer/history.csv'])['last']['step'])
        assert manifest['steps'][-1] == step
        index = manifest['steps'].index(step)
        snapshot = fields/manifest['pattern'].format(field='h',index=index)
        data = (fields/'h_final.bin').read_bytes()
        assert data == snapshot.read_bytes() == (material/'final.bin').read_bytes()
        h = np.frombuffer(data,dtype='<f8').reshape((121*n//64,n,n))
        assert np.isfinite(h).all() and h.min() >= 0 and h.max() <= 1
        threshold = np.fromfile(fields/'h_thresh.bin',dtype='<f8').reshape(h.shape)
        assert np.array_equal(threshold,(h>.5).astype(float))
        final[n] = h
        index300 = manifest['steps'].index(300)
        assert (fields/manifest['pattern'].format(field='h',index=index300)).read_bytes()==(material/'state300.bin').read_bytes()
        checks[str(n)] = dict(final_sha256=hashlib.sha256(data).hexdigest(),
                              final_step=step, final_exact_snapshot=True, strict_threshold=True,
                              initial_sha256=hashlib.sha256((producer/'initial.bin').read_bytes()).hexdigest(),
                              volume_fraction=float(h.mean()),grey_fraction=float(np.mean((h>.1)&(h<.9))),
                              dumped_steps=manifest['steps'],
                              snapshot_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(fields.glob(f'n0fixed{n}_h_*.bin'))})
    delta = final[64]-final[128][::2,::2,::2]
    checks['aligned_grid_comparison'] = dict(rms=float(np.sqrt(np.mean(delta*delta))),
        linf=float(np.max(np.abs(delta))),threshold_disagreement=float(np.mean((final[64]>.5)!=(final[128][::2,::2,::2]>.5))),
        definition='coincident physical grid nodes, no registration or smoothing')
    return raw,checks



def historical_comparison(raw, current):
    records = {}
    for mode in ('continuous','threshold'):
        assert raw[mode+'/exit_code.txt'].strip() == '0'
        old = parse(raw[mode+'/run.log'])
        new = current['resolutions']['64']['material']['final_'+mode]
        a,b = np.array(old['stiffness']),np.array(new['stiffness'])
        records[mode] = dict(historical=old,
            relative_difference_from_new_final=float(np.linalg.norm(a-b)/np.linalg.norm(b)))
    return dict(material=records,
                interpretation='Stored normalized post-budget field versus unnormalized converged64; iteration and force normalization both differ')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--live-root',type=Path)
    group.add_argument('--archive',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical',type=Path,help='optional historical native-consumer archive')
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.live_root:
        raw,checks = collect(args.live_root)
        (args.output/'raw.json.gz').write_bytes(gzip.compress(json.dumps(raw,sort_keys=True).encode(),mtime=0))
        (args.output/'field-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    else:
        raw = json.loads(gzip.decompress(args.archive.read_bytes()))
    summary = reduce(raw)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.historical:
        old = json.loads(gzip.decompress(args.historical.read_bytes()))
        (args.output/'historical-summary.json').write_text(json.dumps(historical_comparison(old,summary),indent=2)+'\n')
    print('CONVERGED_RESOLUTION_AUDIT_OK')


if __name__ == '__main__':
    main()

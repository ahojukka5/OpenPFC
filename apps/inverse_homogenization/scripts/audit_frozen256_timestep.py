#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit the frozen two-arm timestep probe; MAX_STEPS is not convergence."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def reduce(raw):
    result = {}
    for arm,dt,last in [('unchanged',.04,40),('scaled',.01,160)]:
        rows=list(csv.DictReader(l for l in raw[arm+'/history.csv'].splitlines()
                                 if l and not l.startswith('#')))
        assert [int(r['step']) for r in rows]==list(range(last+1))
        assert all(r['elasticity']=='1' and r['frozen']=='1' for r in rows)
        assert rows[-1]['termination']=='MAX_STEPS'
        assert not any(r['verified']=='1' for r in rows)
        assert raw[arm+'/exit_code.txt'].strip()=='0'
        assert f'# CERTIFIED_STEP {last} termination MAX_STEPS' in raw[arm+'/history.csv']
        j=[float(r['J']) for r in rows]
        result[arm]=dict(rows=len(rows),pseudo_time=dt*last,initial_J=j[0],final_J=j[-1],
            positive_csv_J_increments=sum(b>a for a,b in zip(j,j[1:])),last=rows[-1],
            median_ms=statistics.median(float(r['ms']) for r in rows))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--live-root',type=Path);g.add_argument('--archive',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.live_root:
        raw={str(f.relative_to(a.live_root)):f.read_text() for f in sorted(a.live_root.rglob('*'))
             if f.is_file() and f.suffix in ('.txt','.csv','.json','.log','.err','.sh','.sbatch','.md','.py')}
        fields={}
        for arm,last in [('unchanged',40),('scaled',160)]:
            d=a.live_root/arm/'fields'
            m=json.loads((d/(arm+'_manifest.json')).read_text())
            assert m['steps'][-1]==last
            f=d/'h_final.bin';h=np.fromfile(f,dtype='<f8')
            t=np.fromfile(d/'h_thresh.bin',dtype='<f8')
            assert h.size==256*256*484 and np.isfinite(h).all()
            assert h.min()>=0 and h.max()<=1
            assert np.array_equal(t,(h>.5).astype(float))
            assert f.read_bytes()==(d/f'{arm}_h_{len(m["steps"])-1:04d}.bin').read_bytes()
            fields[arm]=dict(final_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                terminal_snapshot_identical=True,strict_threshold_identical=True,
                minimum=float(h.min()),maximum=float(h.max()),mean=float(h.mean()),
                grey_fraction=float(np.mean((h>.1)&(h<.9))),threshold_fraction=float(t.mean()))
        (a.output/'field-checks.json').write_text(json.dumps(fields,indent=2)+'\n')
        (a.output/'raw.json.gz').write_bytes(gzip.compress(json.dumps(raw,sort_keys=True).encode(),mtime=0))
    else:
        raw=json.loads(gzip.decompress(a.archive.read_bytes()))
    (a.output/'summary.json').write_text(json.dumps(reduce(raw),indent=2)+'\n')
    print('FROZEN_TIMESTEP_PROBE_AUDIT_OK: no convergence admitted')


if __name__=='__main__': main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit the frozen 1694-state hold: RUN_DIRECTORY OUTPUT.json.

This is an experiment-specific verifier, not a general convergence validator.
The complete trajectory can be regenerated using the archived frozen command.
"""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
if len(sys.argv)!=3:
    raise SystemExit(__doc__)
root=Path(sys.argv[1])
d=root/'results/frozen_dt_0.04';f=d/'fields'
rows=list(csv.DictReader(s for s in (d/'history.csv').read_text().splitlines() if not s.startswith('#')))
assert len(rows)==1694 and all(int(r['step'])==i and len(r)==26 for i,r in enumerate(rows))
assert rows[-1]['termination']=='CONVERGED' and rows[-1]['verified']=='1'
assert all(r['termination']=='RUNNING' for r in rows[:-1])
assert all(r['elasticity']=='1' and r['frozen']=='1' and float(r['simp_p'])==2 and float(r['lambda_reg'])==.2 for r in rows)
cmd=json.loads((d/'command.json').read_text())
for flag in ['--normalize=0','--dt=0.04','--tol-design=0.0001','--tol-objective=0.000001','--tol-tensor=0.0001','--conv-window=20','--verify-convergence-steps=100','--continuation-steps=0']:
    assert flag in cmd,flag
m=json.loads((f/'frozen_dt_0.04_manifest.json').read_text())
assert m['steps']==list(range(1694))
path=lambda i:f/m['pattern'].format(field='h',index=i)
assert (f/'h_final.bin').read_bytes()==path(1693).read_bytes()
h=np.fromfile(f/'h_final.bin','<f8')
assert h.size==32*32*61 and h.min()>=0 and h.max()<=1
assert np.array_equal(np.fromfile(f/'h_thresh.bin','<f8'),(h>.5).astype(float))
assert path(0).read_bytes()==(root/'initial_h190.bin').read_bytes()
prev=np.fromfile(path(0),'<f8');rms=[]
for i,r in enumerate(rows[1:],1):
    now=np.fromfile(path(i),'<f8');dr=float(np.sqrt(np.mean((now-prev)**2)))
    np.testing.assert_allclose(dr,float(r['design_rms']),rtol=5e-6,atol=1e-12)
    np.testing.assert_allclose(np.mean((now>.5)!=(prev>.5)),float(r['morph_frac']),rtol=5e-6,atol=1e-12)
    rms.append(dr);prev=now
# Reconstruct the declared counter from printed metrics; design is also checked
# from exact fields above. Full C changes are source-owned, not reconstructed
# from the scalar CSV entries.
count=0;candidate=False;left=0;first_candidate=None
for i,r in enumerate(rows):
    quiet=i>0 and all(float(r[k])<tol for k,tol in [('design_rms',1e-4),('dJ_rel',1e-6),('dC_rel',1e-4)])
    if not quiet: count=0;candidate=False;left=0
    else:
        count+=1
        if candidate: left-=1
        elif count>=20: candidate=True;left=100;first_candidate=i if first_candidate is None else first_candidate
    assert count==int(r['conv_window']) and candidate==bool(int(r['candidate']))
    assert bool(int(r['verified']))==(candidate and left==0)
assert left==0
out={'job':22179764,'termination':rows[-1]['termination'],'accepted_step':1693,
'first_candidate':first_candidate,'window':20,'verification_hold':100,'numpy':np.__version__,
'final_sha256':hashlib.sha256((f/'h_final.bin').read_bytes()).hexdigest(),
'final_matches_accepted':True,'threshold_matches_strict_gt_half':True,'all_elasticity_passed':True,
'final_volume_fraction':float(h.mean()),'final_grey_fraction':float(np.mean((h>.1)&(h<.9))),
'final_row':rows[-1], 'last_120_maxima':{k:max(float(r[k]) for r in rows[-120:]) for k in ['design_rms','dJ_rel','dC_rel','morph_frac']},
'objective_increases_at_csv_precision':sum(float(b['J'])>float(a['J']) for a,b in zip(rows,rows[1:])),
'initial_and_terminal_rms':float(np.sqrt(np.mean((h-np.fromfile(path(0),'<f8'))**2))),
'limitation':'Full 6x6 tensor increments are computed by qualified source; scalar CSV cannot independently reconstruct them. Convergence is the frozen declared stopping certificate, not proof of a global or exact stationary optimum.'}
Path(sys.argv[2]).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

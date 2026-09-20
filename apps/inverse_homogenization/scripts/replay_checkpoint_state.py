#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPU restart qualification: moving fields and a restart inside a quiet hold."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def read_state(root):
    generation = (root / 'CURRENT').read_text().strip()
    lines = (root / generation / 'state.txt').read_text().splitlines()
    return dict(line.split(maxsplit=1) for line in lines)


def rows(path):
    return list(csv.DictReader(line for line in path.read_text().splitlines()
                               if line and not line.startswith('#')))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ranks', type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    result = {'binary_sha256': sha(args.binary), 'driver_sha256': sha(Path(__file__)),
              'target_sha256': sha(args.target), 'ranks': args.ranks, 'cases': []}
    common = ['--nx=8', '--ny=8', '--nz=8', '--dx=1', '--E-solid=1',
              '--nu-solid=0.3', '--E-void=0.002', '--nu-void=0.3',
              '--target=file', f'--C-target-file={args.target.resolve()}',
              '--lambda-volume=1', '--lambda-reg=0.2', '--lambda-reg-end=0.2',
              '--simp=2', '--simp-end=2', '--epsilon=2', '--normalize=0',
              '--dt=0.04', '--max-delta=0.04', '--project-volume=1',
              '--n-el-iter=400', '--continuation-steps=0', '--conv-window=20',
              '--verify-convergence-steps=100', '--tol-design=0.0001',
              '--tol-objective=0.000001', '--tol-tensor=0.0001', '--dump-every=1']
    for name, init, volume, maximum, cut in [('hold', 'uniform', 1, 500, 50),
                                           ('moving', 'noise', .2575865186, 9, 4)]:
        root = args.output / name
        root.mkdir()
        case_args = common + [f'--init={init}', f'--volume={volume}',
                              f'--init-volume={volume}', f'--max-steps={maximum}',
                              f'--run-id={name}']

        def run(folder, stage, extra=(), success=True):
            destination = root / folder
            destination.mkdir(exist_ok=True)
            command = ['srun', '--export=ALL', f'--ntasks={args.ranks}',
                       '--gpus-per-node=8', str(args.binary.resolve())] + case_args + [
                f'--checkpoint-dir={destination / "checkpoint"}',
                f'--dump-dir={destination / "fields"}',
                f'--csv={destination / (stage + ".csv")}'] + list(extra)
            (destination / (stage + '-command.json')).write_text(json.dumps(command, indent=2))
            with (destination / (stage + '.log')).open('w') as log:
                proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if success:
                assert proc.returncode == 0, (name, stage, proc.returncode)
            else:
                assert proc.returncode != 0, (name, 'terminal restart was accepted')
            return destination

        full = run('full', 'full')
        split = run('split', 'first', [f'--stop-after={cut}'])
        saved = read_state(split / 'checkpoint')
        assert int(saved['next_step']) == cut and int(saved['termination']) == 0
        if name == 'hold':
            assert int(saved['candidate']) == 1 and 0 < int(saved['verify_left']) < 100
        (root / 'interrupted-state.json').write_text(json.dumps(saved, indent=2))
        run('split', 'resume', [f'--restart={split / "checkpoint"}'])
        reference, resumed = rows(full / 'full.csv'), rows(split / 'resume.csv')
        assert reference[-1]['termination'] == ('CONVERGED' if name == 'hold' else 'MAX_STEPS')
        assert int(resumed[0]['step']) == cut
        lookup = {r['step']: r for r in reference}
        for row in resumed:
            assert {k: v for k, v in row.items() if k != 'ms'} == {
                k: v for k, v in lookup[row['step']].items() if k != 'ms'}
        assert read_state(full / 'checkpoint') == read_state(split / 'checkpoint')
        matched = []
        for file in sorted((full / 'fields').glob('*.bin')):
            assert file.read_bytes() == (split / 'fields' / file.name).read_bytes(), file.name
            matched.append({'file': file.name, 'sha256': sha(file)})
        assert matched
        terminal_before = read_state(split / 'checkpoint')
        run('rejected', 'terminal', [f'--restart={split / "checkpoint"}'], success=False)
        assert read_state(split / 'checkpoint') == terminal_before
        result['cases'].append({'name': name, 'cut': cut, 'saved_state': saved,
                                'last_row': reference[-1], 'overlap_rows': len(resumed),
                                'identical_fields': matched, 'terminal_rejected': True})
        (args.output / 'verified.json').write_text(json.dumps(result, indent=2)+'\n')
    print('CHECKPOINT_FIELD_AND_HOLD_EQUIVALENCE_OK')


if __name__ == '__main__':
    main()

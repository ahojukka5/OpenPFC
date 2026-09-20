#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit completed 160-state control: RUN_DIRECTORY BASELINE_DIRECTORY OUTPUT.

Replay the archived original160-command.txt first if the full trajectory is absent.
The baseline is normalized frozen74_22178459 with identical frozen inputs.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def audit(root, baseline):
    rows = list(csv.DictReader(line for line in (root / 'history.csv').read_text().splitlines()
                              if not line.startswith('#')))
    assert len(rows) == 160 and all(len(r) == 26 and r['elasticity'] == '1' for r in rows)
    assert all(math.isfinite(float(v)) for r in rows for k, v in r.items() if k != 'termination')
    manifest = json.loads((root / 'fields/frozen_n0_manifest.json').read_text())
    assert manifest['steps'] == list(range(160))
    final = (root / 'fields/h_final.bin').read_bytes()
    def field_path(i):
        return root / 'fields' / manifest['pattern'].format(field='h', index=i)
    assert final == field_path(159).read_bytes()
    field = np.frombuffer(final, dtype='<f8')
    assert field.size == 32*32*61 and field.min() >= 0 and field.max() <= 1
    assert np.array_equal(np.fromfile(root / 'fields/h_thresh.bin', '<f8'),
                          (field > .5).astype(float))
    fields = [np.fromfile(field_path(i), '<f8') for i in range(108, 160)]
    for name in ('inverse_hip', 'target.txt', 'initial_h190.bin'):
        assert hashlib.sha256((root / name).read_bytes()).digest() == hashlib.sha256(
            (baseline / name).read_bytes()).digest()
    return dict(job=22179446, termination=rows[-1]['termination'], rows=160,
        final_sha256=hashlib.sha256(final).hexdigest(), mean=float(field.mean()),
        tail={k: float(np.mean([float(r[k]) for r in rows[-50:]]))
              for k in ('J', 'design_rms', 'dJ_rel', 'dC_rel', 'grey')},
        tail_field_rms={str(lag): float(np.mean([np.sqrt(np.mean((b-a)**2))
            for a, b in zip(fields, fields[lag:])])) for lag in (1, 2)},
        objective_increases=sum(float(b['J']) > float(a['J']) for a, b in zip(rows, rows[1:])),
        all_elasticity_passed=True, final_matches_last_accepted=True,
        frozen_inputs_match_baseline=True)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    result = audit(Path(sys.argv[1]), Path(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Invoke the archive reducer and ensure an incomplete hold cannot be admitted."""
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'apps/inverse_homogenization/scripts/audit_converged_resolution.py'
ARCHIVE = ROOT/'apps/inverse_homogenization/evidence/converged-resolution'


def test_actual_reducer_and_rejected_certificate(tmp_path):
    subprocess.run([sys.executable,str(SCRIPT),'--archive',str(ARCHIVE/'raw.json.gz'),
                    '--output',str(tmp_path/'valid')],check=True)
    assert json.loads((tmp_path/'valid/summary.json').read_text()) == json.loads((ARCHIVE/'summary.json').read_text())
    raw = json.loads(gzip.decompress((ARCHIVE/'raw.json.gz').read_bytes()))
    # Remove the terminal state: successful elasticity on a late iterate must
    # never stand in for completing the full verification hold.
    key = '64/producer/history.csv'
    lines = raw[key].splitlines()
    data = [i for i,line in enumerate(lines) if line and not line.startswith('#')]
    del lines[data[-1]]
    raw[key] = '\n'.join(lines)+'\n'
    broken = tmp_path/'broken.json.gz'
    broken.write_bytes(gzip.compress(json.dumps(raw).encode()))
    result = subprocess.run([sys.executable,str(SCRIPT),'--archive',str(broken),
                             '--output',str(tmp_path/'invalid')],capture_output=True)
    assert result.returncode != 0
    assert not (tmp_path/'invalid/summary.json').exists()

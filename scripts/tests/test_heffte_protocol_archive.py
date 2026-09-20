#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The archived cohort must retain, but not mix, other accounts/protocols."""
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "apps/heat3d/scripts/archive_protocol_evidence.py"


def test_archive_replay_separates_frozen_cohort_and_checks_hashes(tmp_path):
    records = []
    for job, ranks, steps, warmup, account, wall in [
        (1, 1, 20, 1, "project_462001245", .1),
        (2, 8, 20, 1, "project_462001245", .2),
        (3, 1, 105, 5, "project_462001245", .01),
        (4, 1, 20, 1, "project_462001519", .001),
    ]:
        row = dict(family=768, job=job, nodes=1, ranks=ranks,
                   account=account, reshape="p2p_plined", admit="ok",
                   wall_step_s=f"{wall:.8f}")
        records.append(dict(row=row, steps=steps, warmup=warmup, dt="0.01",
                            timing=dict(central_order_statistics=[wall], median=wall)))
    artifact = "immutable metadata\n"
    sha = hashlib.sha256(artifact.encode()).hexdigest()
    bundle = dict(records=records, artifacts_by_sha256={sha: artifact})
    path = tmp_path / "records.json.gz"
    path.write_bytes(gzip.compress(json.dumps(bundle).encode(), mtime=0))
    subprocess.run([sys.executable, str(SCRIPT), "--replay", str(tmp_path)], check=True)
    with (tmp_path / "all-runs.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 4
    with (tmp_path / "campaign-runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["job"] for r in rows] == ["1", "2"]
    with (tmp_path / "scaling.csv").open() as stream:
        scales = list(csv.DictReader(stream))
    assert [float(r["weak_eff"]) for r in scales] == [1.0, .5]
    bundle["artifacts_by_sha256"][sha] = "changed metadata"
    path.write_bytes(gzip.compress(json.dumps(bundle).encode(), mtime=0))
    rejected = subprocess.run([sys.executable, str(SCRIPT), "--replay", str(tmp_path)],
                              capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "archive artifact hash mismatch" in rejected.stderr

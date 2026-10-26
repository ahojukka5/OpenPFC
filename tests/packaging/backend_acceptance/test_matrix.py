# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail-closed acceptance reporting, including the public runner entry point."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

import run_matrix


def receipt(backend="cuda", ranks=2):
    lines = []
    for rank in range(ranks):
        lines += ["IDENTITY " + json.dumps({
            "rank": rank, "backend": backend, "sdk_state": "clean", "host": "node",
            "device_uuid": f"device-{rank}", "sdk_revision": "a" * 40,
            "sdk_manifest_sha256": "b" * 64, "consumer_source_sha256": "c" * 64,
        }), "RESULT " + json.dumps({"rank": rank, "ranks": ranks, "backend": backend, "status": "PASS"})]
    return "\n".join(lines)


def test_complete_distinct_device_receipt():
    assert len(run_matrix.validate_receipt(receipt(), "cuda", 2)["results"]) == 2


@pytest.mark.parametrize("old,new", [
    ('"backend": "cuda"', '"backend": "cpu"'),
    ('"sdk_state": "clean"', '"sdk_state": "dirty"'),
    ('"device_uuid": "device-1"', '"device_uuid": "device-0"'),
    ('"sdk_revision": "' + "a" * 40 + '"', '"sdk_revision": "unknown"'),
    ('"status": "PASS"', '"status": "NOT_QUALIFIED"'),
])
def test_receipt_rejects_substitution_and_missing_evidence(old, new):
    with pytest.raises(ValueError):
        run_matrix.validate_receipt(receipt().replace(old, new), "cuda", 2)


@pytest.mark.parametrize("output", ["", "RESULT {}", receipt(ranks=1), receipt() + "\n" + receipt()])
def test_zero_exit_alone_cannot_qualify(output):
    with pytest.raises((ValueError, KeyError)):
        run_matrix.validate_receipt(output, "cuda", 2)


def test_runner_entrypoint_reports_missing_backends(tmp_path):
    output = tmp_path / "qualification"
    result = subprocess.run([sys.executable, str(Path(run_matrix.__file__)), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 77
    report = json.loads((output / "matrix.json").read_text())
    assert report["status"] == "NOT_QUALIFIED"
    assert len(report["rows"]) == 9
    assert all(row["status"] == "NOT_QUALIFIED" for row in report["rows"])

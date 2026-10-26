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
        }), "RESULT " + json.dumps({"rank": rank, "ranks": ranks, "backend": backend, "status": "PASS",
            "field_tolerance": 2e-12, "ledger_tolerance": 3e-11, "fft_tolerance": 2e-12,
            "field_max_error": 5e-16, "accepted_increment": 8 if ranks == 1 else 16,
            "restart": ranks > 1})]
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


def fragment(backend):
    rows = []
    for ranks in run_matrix.RANKS:
        rows.append(dict(backend=backend, ranks=ranks, status="PASS",
                         **run_matrix.validate_receipt(receipt(backend, ranks), backend, ranks)))
    return {"status": "NOT_QUALIFIED", "rows": rows}


def test_merge_requires_all_backends_and_preserves_failures(tmp_path):
    paths = []
    for backend in run_matrix.BACKENDS:
        path = tmp_path / (backend + ".json")
        path.write_text(json.dumps(fragment(backend)))
        paths.append(path)
    assert run_matrix.merge_reports(paths[:2], tmp_path / "partial")["status"] == "NOT_QUALIFIED"
    assert run_matrix.merge_reports(paths, tmp_path / "complete")["status"] == "PASS"
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"rows": [{"backend": "cuda", "ranks": 2, "status": "FAIL"}]}))
    assert run_matrix.merge_reports(paths + [failed], tmp_path / "failure")["status"] == "FAIL"


def test_merge_rejects_mixed_case_sources(tmp_path):
    paths = []
    for backend in ("cpu", "cuda"):
        path = tmp_path / (backend + ".json")
        text = json.dumps(fragment(backend))
        if backend == "cuda":
            text = text.replace("c" * 64, "d" * 64)
        path.write_text(text)
        paths.append(path)
    with pytest.raises(ValueError, match="different compiled"):
        run_matrix.merge_reports(paths, tmp_path / "mixed")


@pytest.mark.parametrize("old,new", [("2e-12", "2e-08"), ('"field_max_error": 5e-16', '"field_max_error": NaN')])
def test_receipt_rejects_weakened_or_nonfinite_results(old, new):
    with pytest.raises(ValueError):
        run_matrix.validate_receipt(receipt().replace(old, new), "cuda", 2)

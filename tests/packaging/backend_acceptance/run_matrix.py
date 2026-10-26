# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run installed-package acceptance on allocated CPU/CUDA/HIP resources.

Exit 0 requires all nine rows. Exit 77 means a partial qualification; exit 1
means a built, available case failed. Raw launcher output is retained.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess


BACKENDS = ("cpu", "cuda", "hip")
RANKS = (1, 2, 4)


def parse_lines(output, prefix):
    return [json.loads(line[len(prefix):]) for line in output.splitlines() if line.startswith(prefix)]


def validate_receipt(output, backend, ranks):
    """A zero exit without complete backend/provenance receipts is a failure."""
    results = parse_lines(output, "RESULT ")
    identities = parse_lines(output, "IDENTITY ")
    expected_ranks = set(range(ranks))
    if {r["rank"] for r in results} != expected_ranks or len(results) != ranks:
        raise ValueError("missing or duplicate result rank")
    if {r["rank"] for r in identities} != expected_ranks or len(identities) != ranks:
        raise ValueError("missing or duplicate source/device identity")
    if any(r["status"] != "PASS" or r["backend"] != backend or r["ranks"] != ranks for r in results):
        raise ValueError("backend substitution or failed result")
    for key in ("sdk_revision", "sdk_manifest_sha256", "consumer_source_sha256"):
        values = {r[key] for r in identities}
        length = 40 if key == "sdk_revision" else 64
        if len(values) != 1 or any(len(v) != length or any(c not in "0123456789abcdef" for c in v) for v in values):
            raise ValueError("missing or inconsistent source identity")
    if any(r["sdk_state"] != "clean" or r["backend"] != backend for r in identities):
        raise ValueError("dirty SDK or backend substitution")
    if backend != "cpu":
        devices = [(r["host"], r["device_uuid"]) for r in identities]
        if any(not uuid for _, uuid in devices) or len(set(devices)) != ranks:
            raise ValueError("multi-rank GPU case shared a device")
    return {"identities": identities, "results": results}


def execute(command, log, timeout):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr
        code = result.returncode
    except subprocess.TimeoutExpired as error:
        output = "TIMEOUT\n" + str(error)
        code = 124
    log.write_text(output)
    return code, output


def run_matrix(binaries, launcher, count_flag, output, timeout):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for backend in BACKENDS:
        binary = binaries.get(backend)
        baseline_ok = False
        for ranks in RANKS:
            row = {"backend": backend, "ranks": ranks, "status": "NOT_QUALIFIED"}
            rows.append(row)
            if not binary:
                row["reason"] = "no installed consumer binary supplied"
                continue
            if ranks > 1 and not baseline_ok:
                row["reason"] = "one-rank checkpoint prerequisite unavailable"
                continue
            prefix = launcher + [count_flag, str(ranks), str(binary)]
            bundle = output / (backend + "-checkpoint")
            if backend != "cpu":
                code, _ = execute(prefix + ["preflight", str(bundle)], output / f"{backend}-{ranks}-preflight.log", timeout)
                if code:
                    row["reason"] = "distinct assigned devices or device-buffer MPI preflight unavailable"
                    row["preflight_exit_code"] = code
                    continue
            command = prefix + ["run" if ranks == 1 else "resume", str(bundle)]
            code, text = execute(command, output / f"{backend}-{ranks}.log", timeout)
            row["command"] = command
            if code == 77:
                row["reason"] = "runtime capability unavailable"
                continue
            try:
                if code:
                    raise ValueError(f"acceptance process exited {code}")
                row.update(validate_receipt(text, backend, ranks))
                row["status"] = "PASS"
                if ranks == 1:
                    baseline_ok = True
            except (ValueError, KeyError, TypeError) as error:
                row["status"] = "FAIL"
                row["reason"] = str(error)
    status = "FAIL" if any(r["status"] == "FAIL" for r in rows) else (
        "PASS" if all(r["status"] == "PASS" for r in rows) else "NOT_QUALIFIED")
    report = {"status": status, "rows": rows}
    (output / "matrix.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for backend in BACKENDS:
        parser.add_argument("--" + backend, type=Path)
    parser.add_argument("--launcher", default="mpiexec")
    parser.add_argument("--count-flag", default="-n")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    binaries = {b: getattr(args, b).resolve(strict=True) for b in BACKENDS if getattr(args, b)}
    report = run_matrix(binaries, shlex.split(args.launcher), args.count_flag, args.output.resolve(), args.timeout)
    print(json.dumps(report, indent=2))
    return {"PASS": 0, "FAIL": 1, "NOT_QUALIFIED": 77}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())

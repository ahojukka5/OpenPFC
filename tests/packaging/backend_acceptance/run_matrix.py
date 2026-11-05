# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run installed-package acceptance on allocated CPU/CUDA/HIP resources.

Exit 0 requires all nine rows. Exit 77 means a partial qualification; exit 1
means a built, available case failed. Raw launcher output is retained.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
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
    for result in results:
        if (result["field_tolerance"] != 2e-12 or result["ledger_tolerance"] != 3e-11
                or result["fft_tolerance"] != 2e-12
                or not math.isfinite(result["field_max_error"])
                or not 0 <= result["field_max_error"] < 2e-12
                or result["accepted_increment"] != (8 if ranks == 1 else 16)
                or result["restart"] is not (ranks > 1)):
            raise ValueError("missing checks or changed acceptance criteria")
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
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   start_new_session=os.name == "posix")
    except OSError as error:
        output = "LAUNCH_ERROR\n" + str(error)
        code = 127
    else:
        try:
            output, _ = process.communicate(timeout=timeout)
            code = process.returncode
        except subprocess.TimeoutExpired:
            # Give MPI launchers a chance to stop their ranks. The group is
            # private to this invocation; unrelated jobs are never signaled.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            try:
                output, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                output, _ = process.communicate()
            output = "TIMEOUT\n" + output
            code = 124
    log.write_text(output)
    return code, output


def validate_builds(rows):
    """Compare actual passing receipts across rank counts and backends."""
    consumer_hashes = set()
    providers = {}
    for backend in BACKENDS:
        builds = set()
        for row in rows:
            if row["backend"] != backend or row["status"] != "PASS":
                continue
            identity = row["identities"][0]
            consumer_hashes.add(identity["consumer_source_sha256"])
            builds.add((identity["sdk_revision"], identity["sdk_manifest_sha256"]))
        if len(builds) > 1:
            raise ValueError(f"{backend} rows use different provider builds; select matching fragments")
        if builds:
            providers[backend] = next(iter(builds))[0]
    if len(consumer_hashes) > 1:
        raise ValueError("matrix fragments use different compiled acceptance cases")
    return providers


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
    try:
        report["provider_revisions"] = validate_builds(rows)
    except ValueError as error:
        report.update(status="FAIL", reason=str(error))
    (output / "matrix.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def merge_reports(paths, output):
    """Qualify exact backend builds only when all nine matching case rows pass."""
    output.mkdir(parents=True, exist_ok=False)
    inputs = [(str(path.resolve()), json.loads(path.read_text())) for path in paths]
    for _, report in inputs:
        for row in report["rows"]:
            if row["backend"] not in BACKENDS or row["ranks"] not in RANKS or row["status"] not in ("PASS", "FAIL", "NOT_QUALIFIED"):
                raise ValueError("invalid matrix row")
            if row["status"] == "PASS":
                text = "\n".join(
                    ["IDENTITY " + json.dumps(x) for x in row["identities"]]
                    + ["RESULT " + json.dumps(x) for x in row["results"]])
                validate_receipt(text, row["backend"], row["ranks"])
    rows = []
    for backend in BACKENDS:
        for ranks in RANKS:
            row = {"backend": backend, "ranks": ranks, "status": "NOT_QUALIFIED",
                   "reason": "no qualified input row"}
            candidates = [(path, candidate) for path, report in inputs for candidate in report["rows"]
                          if candidate["backend"] == backend and candidate["ranks"] == ranks]
            failures = [path for path, candidate in candidates if candidate["status"] == "FAIL"]
            if failures:
                row.update(status="FAIL", reason="an input row failed", input_reports=failures)
            else:
                for path, candidate in candidates:
                    if candidate["status"] != "PASS":
                        continue
                    if row["status"] != "PASS":
                        row = dict(candidate, input_report=path)
            rows.append(row)
    # Validate every supplied passing candidate, including duplicates that
    # were not chosen for the output row. Conflicting evidence cannot hide.
    passing = [row for _, report in inputs for row in report["rows"]
               if row["status"] == "PASS"]
    provider_revisions = validate_builds(passing)
    status = "FAIL" if any(r["status"] == "FAIL" for r in rows) else (
        "PASS" if all(r["status"] == "PASS" for r in rows) else "NOT_QUALIFIED")
    report = {"status": status, "provider_revisions": provider_revisions,
              "input_reports": [path for path, _ in inputs], "rows": rows}
    (output / "matrix.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for backend in BACKENDS:
        parser.add_argument("--" + backend, type=Path)
    parser.add_argument("--merge", type=Path, nargs="+", help="merge matrix fragments instead of launching cases")
    parser.add_argument("--launcher", default="mpiexec")
    parser.add_argument("--count-flag", default="-n")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    binaries = {b: getattr(args, b).resolve(strict=True) for b in BACKENDS if getattr(args, b)}
    if args.merge:
        if binaries:
            parser.error("--merge cannot be combined with consumer binaries")
        report = merge_reports(args.merge, args.output.resolve())
    else:
        report = run_matrix(binaries, shlex.split(args.launcher), args.count_flag, args.output.resolve(), args.timeout)
    print(json.dumps(report, indent=2))
    return {"PASS": 0, "FAIL": 1, "NOT_QUALIFIED": 77}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())

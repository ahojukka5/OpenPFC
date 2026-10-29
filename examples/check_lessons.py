#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the teaching executables and check fields against independent mode formulas."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile


def run(command: list[str], cwd: Path, success: bool = True) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=90)
    if (result.returncode == 0) != success:
        raise AssertionError(f"{command}\n{result.stdout}\n{result.stderr}")


def check_heat(executable: str, method: str, root: Path, launch: list[str]) -> None:
    directory = root / method
    directory.mkdir()
    parameters = {"n": 16, "steps": 20, "dt": 0.002, "diffusivity": 0.7}
    source = directory / "input.json"
    source.write_text(json.dumps(parameters))
    run([*launch, executable, str(source)], directory)

    # Analytic eigenvalue of the discrete stencil; no OpenPFC routines are used.
    n, dt, diffusion, steps = (parameters[key] for key in ("n", "dt", "diffusivity", "steps"))
    h = 2.0 * math.pi / n
    eigenvalue = -12.0 * math.sin(h / 2.0) ** 2 / h**2 if method in ("fd", "separated") else -3.0
    amplitude = 0.2 * (math.exp(eigenvalue * diffusion * dt * steps) if method == "exponential"
                       else (1.0 + dt * diffusion * eigenvalue) ** steps)
    fields = list((directory / "results").glob("field_rank_*.csv"))
    count = 0
    owned = set()
    for field in fields:
        for row in csv.DictReader(field.open()):
            x, y, z, value = (float(row[key]) for key in ("x", "y", "z", "u"))
            index = tuple(round(point/h) for point in (x, y, z))
            assert index not in owned, (method, index)
            owned.add(index)
            expected = 1.0 + amplitude * math.cos(x) * math.cos(y) * math.cos(z)
            assert math.isfinite(value) and abs(value - expected) < 2e-11, (method, row, expected)
            count += 1
    assert count == n**3, (method, count)
    assert owned == {(i,j,k) for k in range(n) for j in range(n) for i in range(n)}

    rows = list(csv.DictReader((directory / "results/diagnostics.csv").open()))
    assert len(rows) == steps + 1
    previous = math.inf
    for row in rows:
        mean, fluctuation, error = (float(row[key]) for key in ("mean", "rms_fluctuation", "rms_error"))
        assert math.isfinite(error) and abs(mean - 1.0) < 2e-12
        assert 0.0 <= fluctuation <= previous + 2e-13
        previous = fluctuation
    continuum_amplitude = 0.2 * math.exp(-3.0 * diffusion * dt * steps)
    expected_error = abs(amplitude - continuum_amplitude) / math.sqrt(8.0)
    assert abs(float(rows[-1]["rms_error"]) - expected_error) < 2e-12

    parameters["n"] = 7
    source.write_text(json.dumps(parameters))
    run([*launch, executable, str(source)], directory, success=False)
    if method != "exponential":
        parameters.update(n=16, dt=1.0)
        source.write_text(json.dumps(parameters))
        run([*launch, executable, str(source)], directory, success=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", required=True)
    parser.add_argument("--separated", required=True)
    parser.add_argument("--fourier", required=True)
    parser.add_argument("--exponential", required=True)
    parser.add_argument("--differentiation", required=True)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    launch = [args.launcher, args.rank_flag, str(args.ranks)] if args.launcher else []
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lessons-", dir=args.output) as temporary:
        root = Path(temporary).resolve()
        for method in ("fd", "fourier", "exponential", "separated"):
            check_heat(getattr(args, method), method, root, launch)
        directory = root / "differentiation"
        directory.mkdir()
        run([*launch, args.differentiation], directory)
        rows = list(csv.DictReader((directory / "results/convergence.csv").open()))
        assert [int(row["n"]) for row in rows] == [8, 16, 32, 64, 128]
        for key, lower, upper in (("fd2", 3.5, 4.5), ("fd4", 13.0, 19.0)):
            errors = [float(row[key]) for row in rows]
            assert all(math.isfinite(error) and error > 0.0 for error in errors)
            assert lower < errors[-2] / errors[-1] < upper, (key, errors)
        assert float(rows[0]["fourier"]) > 1e-4
        assert max(float(row["fourier"]) for row in rows[2:]) < 2e-11
    print(f"Lesson fields, convergence, conservation and failure controls passed ({args.ranks} rank(s)).")


if __name__ == "__main__":
    main()

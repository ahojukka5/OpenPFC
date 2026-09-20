#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qualify terminal HIP fields, then diagnose timestep sensitivity (#74).

Run inside an eight-rank GPU allocation with the normal LUMI modules loaded.
This is a fresh frozen-problem experiment, not a checkpoint/restart test.
No numerical target or convergence tolerance is changed between arms.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows(path):
    with path.open() as stream:
        result = list(csv.DictReader(line for line in stream
                                   if not line.startswith("#")))
    if not result or any(None in r or None in r.values() for r in result):
        raise RuntimeError("Malformed or empty history: " + str(path))
    for row in result:
        if len(row) != 26 or row["elasticity"] != "1":
            raise RuntimeError("Invalid iterate: " + repr(row))
        for key, value in row.items():
            if key != "termination" and not np.isfinite(float(value)):
                raise RuntimeError("Nonfinite iterate: " + repr(row))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--initial-field", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "qualification", "0.04", "0.02", "0.01"],
                        default="all")
    parser.add_argument("--normalize", type=int, choices=[0, 1], default=1)
    parser.add_argument("--frozen-steps", type=int,
                        help="separate fixed-step hold; numeric phase required")
    args = parser.parse_args()
    if args.frozen_steps is not None and (
            args.phase in ("all", "qualification") or args.frozen_steps < 2):
        parser.error("--frozen-steps requires a numeric phase and at least two states")
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "source_sha": args.source_sha, "binary": str(args.binary),
        "binary_sha256": digest(args.binary), "driver_sha256": digest(__file__),
        "target": str(args.target), "target_sha256": digest(args.target),
        "initial_field": str(args.initial_field),
        "initial_sha256": digest(args.initial_field),
        "python": platform.python_version(), "numpy": np.__version__,
        "job": os.environ.get("SLURM_JOB_ID"),
        "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "ranks": 8, "gcds": 8,
        "phase": args.phase, "normalize": args.normalize,
        "frozen_steps_override": args.frozen_steps,
        "hypothesis": ("The unnormalized total-objective update meets the original certificate"
                       if args.normalize == 0 else
                       "A smaller explicit step reduces the late oscillation"),
        "limitations": ["fresh frozen runs, not checkpoint resumes",
                        "smaller updates alone do not prove convergence",
                        "CSV metrics have six significant digits"],
    }
    save(args.output / "provenance.json", provenance)
    common = ["--nx=32", "--ny=32", "--nz=61", "--dx=1",
              "--E-solid=1", "--nu-solid=0.3", "--E-void=0.002",
              "--nu-void=0.3", "--target=file",
              "--C-target-file=" + str(args.target),
              "--volume=0.2575865186", "--init-volume=0.2575865186",
              "--normalize=" + str(args.normalize), "--max-delta=0.04", "--project-volume=1",
              "--n-el-iter=400", "--epsilon=2", "--lambda-volume=1",
              "--tol-design=0.0001", "--tol-objective=0.000001",
              "--tol-tensor=0.0001", "--conv-window=20",
              "--verify-convergence-steps=100", "--init=spinodal", "--seed=1"]
    frozen = ["--simp=2", "--simp-end=2", "--lambda-reg=0.2",
              "--lambda-reg-end=0.2", "--continuation-steps=0"]

    def run(name, options, base=common):
        folder = args.output / name
        folder.mkdir()
        command = ["srun", "--export=ALL", "--ntasks=8", "--gpus-per-node=8",
                   str(args.binary)] + base + options + [
                       "--run-id=" + name, "--csv=" + str(folder / "history.csv"),
                       "--dump-dir=" + str(folder / "fields")]
        save(folder / "command.json", command)
        started = time.monotonic()
        with (folder / "stdout.log").open("w") as out, \
                (folder / "stderr.log").open("w") as err:
            proc = subprocess.run(command, stdout=out, stderr=err, check=False)
        save(folder / "execution.json", {"returncode": proc.returncode,
                                          "seconds": time.monotonic() - started})
        if proc.returncode:
            raise RuntimeError("Failed run: " + name)
        history = rows(folder / "history.csv")
        manifest = json.loads((folder / "fields" / (name + "_manifest.json"))
                              .read_text())
        final = folder / "fields" / "h_final.bin"
        field = np.fromfile(str(final), dtype="<f8")
        n = manifest["nx"] * manifest["ny"] * manifest["nz"]
        assert field.size == n and np.isfinite(field).all()
        assert field.min() >= 0 and field.max() <= 1
        assert manifest["steps"] == sorted(set(manifest["steps"]))
        assert manifest["steps"][-1] == int(history[-1]["step"])
        last_snapshot = folder / "fields" / manifest["pattern"].format(
            field="h", index=len(manifest["steps"]) - 1)
        assert digest(final) == digest(last_snapshot)
        threshold = np.fromfile(str(folder / "fields" / "h_thresh.bin"), "<f8")
        assert np.array_equal(threshold, (field > 0.5).astype(float))
        assert np.isclose(field.mean(), float(history[-1]["volume"]), atol=5e-7)
        tail = history[-min(50, len(history)):]
        metrics = {key: {"min": min(float(r[key]) for r in tail),
                         "max": max(float(r[key]) for r in tail),
                         "mean": float(np.mean([float(r[key]) for r in tail]))}
                   for key in ["J", "design_rms", "dJ_rel", "dC_rel", "grey"]}
        summary = {"termination": history[-1]["termination"], "rows": len(history),
                   "final_sha256": digest(final), "tail": metrics}
        if len(manifest["steps"]) > 2 and all(
                b - a == 1 for a, b in zip(manifest["steps"], manifest["steps"][1:])):
            fields = [np.fromfile(str(folder / "fields" /
                      manifest["pattern"].format(field="h", index=i)), "<f8")
                      for i in range(max(0, len(history) - 52), len(history))]
            summary["tail_field_rms"] = {
                str(lag): float(np.mean([np.sqrt(np.mean((b - a) ** 2))
                    for a, b in zip(fields, fields[lag:])])) for lag in [1, 2]}
        save(folder / "summary.json", summary)
        print(name, summary["termination"], flush=True)
        return folder, history

    initial = "--load-bin=" + str(args.initial_field)
    qualification = ["--simp=1", "--simp-end=2", "--lambda-reg=0.05",
        "--lambda-reg-end=0.2", "--continuation-steps=40", "--max-steps=200",
        "--conv-window=10", "--verify-convergence-steps=20", "--dt=0.04",
        "--dump-every=5"]
    arms = [(0.04, 160), (0.02, 320), (0.01, 640)]
    if args.phase != "all":
        if args.phase == "qualification":
            run("qualification32", qualification)
        else:
            dt, steps = next((dt, steps) for dt, steps in arms if str(dt) == args.phase)
            steps = args.frozen_steps if args.frozen_steps is not None else steps
            run("frozen_dt_" + str(dt), frozen + [initial, "--dt=" + str(dt),
                "--max-steps=" + str(steps), "--dump-every=1"])
        return
    # Nonstationary field: terminal output must be the evaluated incoming state.
    short, short_rows = run("terminal_short", frozen + [initial, "--dt=0.04",
                            "--max-steps=3", "--dump-every=1"])
    longer, _ = run("terminal_long", frozen + [initial, "--dt=0.04",
                    "--max-steps=4", "--dump-every=1"])
    assert digest(short / "fields/h_final.bin") == digest(
        longer / "fields/terminal_long_h_0002.bin")
    assert digest(short / "fields/h_final.bin") != digest(longer / "fields/h_final.bin")
    _, evaluated = run("terminal_reevaluate", frozen + [
        "--load-bin=" + str(short / "fields/h_final.bin"), "--dt=0.04",
        "--max-steps=1", "--dump-every=1"])
    for key in ["J", "J_tensor", "J_volume", "J_reg", "C11", "C12", "C_fro"]:
        assert np.isclose(float(evaluated[0][key]), float(short_rows[-1][key]),
                          rtol=2e-5, atol=1e-10), key
    # An exactly homogeneous, projected state exercises CONVERGED at unchanged
    # tolerances. This is an implementation control, not scientific evidence.
    _, stationary = run("terminal_converged", ["--nx=8", "--ny=8", "--nz=8",
        "--init=uniform", "--init-volume=1", "--volume=1", "--project-volume=1",
        "--target=isotropic", "--E-target=1", "--nu-target=0.3",
        "--simp=2", "--lambda-reg=0", "--continuation-steps=0",
        "--max-steps=125", "--conv-window=20", "--verify-convergence-steps=100",
        "--dt=0.04", "--dump-every=10"], base=[])
    assert stationary[-1]["termination"] == "CONVERGED"
    save(args.output / "terminal_qualification.json", {"passed": True})
    run("qualification32", qualification)
    # 160/320/640 accepted states: compare trajectories on common elapsed
    # artificial time up to 6.36, accounting for the restored final state.
    for dt, steps in arms:
        run("frozen_dt_" + str(dt), frozen + [initial, "--dt=" + str(dt),
            "--max-steps=" + str(steps), "--dump-every=1"])


if __name__ == "__main__":
    main()

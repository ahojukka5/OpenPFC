#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The converged-resolution reducer rejects a hold that stops early.

Campaign archives are not package fixtures. This test builds a synthetic
two-grid text archive and checks the admission rule on that archive.
"""
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "apps/inverse_homogenization/scripts/audit_converged_resolution.py"


def _matrix_lines(matrix):
    return "\n".join(" ".join(f"{value:.16e}" for value in row) for row in matrix)


def _poisson(compliance):
    pairs = (
        ("nu_xy", 0, 1),
        ("nu_xz", 0, 2),
        ("nu_yx", 1, 0),
        ("nu_yz", 1, 2),
        ("nu_zx", 2, 0),
        ("nu_zy", 2, 1),
    )
    return {
        name: float(-compliance[j, i] / compliance[i, i]) for name, i, j in pairs
    }


def _log(stiffness, compliance):
    poisson = _poisson(compliance)
    shortcut = " ".join(f"{name} {poisson[name]:.16e}" for name in poisson)
    return (
        "\n".join(
            [
                "shape synthetic",
                "volume_fraction 0.25 spd yes converged yes",
                "C_H (engineering Voigt)",
                _matrix_lines(stiffness),
                "S = C_H^{-1}",
                _matrix_lines(compliance),
                f"nu_shortcut 0.0 {shortcut}",
                "elasticity_converged yes el_iters_max 2 el_residual_max 1.0e-12",
                "solid_components 1 void_components 1 perc_xyz 1 1 1 "
                "island_solid 0 grey 0",
                "opening_loss_r1 1.0e-3 opening_loss_r2 2.0e-3",
            ]
        )
        + "\n"
    )


def _history(last_step=419):
    header = (
        "step,elasticity,frozen,simp_p,lambda_reg,design_rms,dJ_rel,dC_rel,"
        "conv_window,candidate,verified,termination"
    )
    rows = [header]
    quiet = 0
    for step in range(last_step + 1):
        frozen = int(step >= 300)
        passes = step >= 300
        quiet = quiet + 1 if passes else 0
        termination = "CONVERGED" if step == last_step else "RUNNING"
        rows.append(
            ",".join(
                map(
                    str,
                    [
                        step,
                        1,
                        frozen,
                        2 if frozen else 1,
                        0.2 if frozen else 0.05,
                        1e-6 if passes else 1,
                        1e-8 if passes else 1,
                        1e-6 if passes else 1,
                        quiet,
                        int(quiet >= 20),
                        int(quiet >= 120),
                        termination,
                    ],
                )
            )
        )
    return "\n".join(rows) + "\n"


def _materials():
    final = np.eye(6)
    final[0, 1] = final[1, 0] = -0.25
    earlier = final * 1.01
    return final, np.linalg.inv(final), earlier, np.linalg.inv(earlier)


def _archive():
    final, final_compliance, earlier, earlier_compliance = _materials()
    raw = {}
    history = _history()
    for resolution in (64, 128):
        prefix = f"{resolution}/"
        raw[prefix + "producer/history.csv"] = history
        for state, stiffness, compliance in (
            ("state300", earlier, earlier_compliance),
            ("final", final, final_compliance),
        ):
            text = _log(stiffness, compliance)
            for mode in ("continuous", "threshold"):
                key = f"{prefix}material/{state}_{mode}/"
                raw[key + "exit_code.txt"] = "0\n"
                raw[key + "run.log"] = text
        for field in ("h_final", "h_thresh"):
            raw[prefix + f"producer/fields/{field}_material.json"] = json.dumps(
                {
                    "inverse_converged": True,
                    "diagnostics_valid": True,
                    "accepted_step": 419,
                    "grid": [resolution, resolution, 121 * resolution // 64],
                    "spacing": 64 / resolution,
                    "stiffness_symmetric": final.tolist(),
                }
            )
    return raw


def _historical():
    final, compliance, _, _ = _materials()
    text = _log(final, compliance)
    raw = {}
    for mode in ("continuous", "threshold"):
        raw[f"{mode}/exit_code.txt"] = "0\n"
        raw[f"{mode}/run.log"] = text
    return raw


def _write_gzip(path, payload):
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))


def test_incomplete_hold_is_rejected(tmp_path):
    archive = _archive()
    historical = _historical()
    _write_gzip(tmp_path / "raw.json.gz", archive)
    _write_gzip(tmp_path / "historical.json.gz", historical)
    valid = tmp_path / "valid"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--archive",
            str(tmp_path / "raw.json.gz"),
            "--output",
            str(valid),
            "--historical",
            str(tmp_path / "historical.json.gz"),
        ],
        check=True,
    )
    summary = json.loads((valid / "summary.json").read_text())
    historical_summary = json.loads((valid / "historical-summary.json").read_text())
    planted = _poisson(_materials()[1])
    reported = summary["resolutions"]["64"]["material"]["final_continuous"]["poisson"]
    assert {name: reported[name] for name in planted} == planted
    assert (
        historical_summary["material"]["threshold"]["historical"]["poisson"]["nu_xy"]
        < 0
    )

    lines = archive["64/producer/history.csv"].splitlines()
    data = [index for index, line in enumerate(lines) if line and not line.startswith("#")]
    del lines[data[-1]]
    archive["64/producer/history.csv"] = "\n".join(lines) + "\n"
    broken = tmp_path / "broken.json.gz"
    _write_gzip(broken, archive)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--archive",
            str(broken),
            "--output",
            str(tmp_path / "invalid"),
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "invalid" / "summary.json").exists()

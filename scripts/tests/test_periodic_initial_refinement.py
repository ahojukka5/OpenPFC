#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exercise the refinement CLI and its periodic physical-domain contract."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
DRIVER = (Path(__file__).resolve().parents[2] / "apps/inverse_homogenization"
          / "scripts/refine_periodic_initial.py")


def test_refinement_cli_preserves_samples_mean_and_periodic_seam(tmp_path):
    # Distinct axis lengths and values detect brick-order transposition.
    field = np.arange(24, dtype=float).reshape(2, 3, 4) / 24
    source, output = tmp_path / "coarse.bin", tmp_path / "fine.bin"
    field.astype("<f8").tofile(str(source))
    subprocess.run([sys.executable, str(DRIVER), str(source), str(output),
                    "--shape", "4", "3", "2", "--dx", "0.5"], check=True)
    fine = np.fromfile(str(output), "<f8").reshape(4, 6, 8)
    assert np.array_equal(fine[::2, ::2, ::2], field)
    assert fine.mean() == pytest.approx(field.mean(), abs=1e-15)
    assert fine.min() == field.min() and fine.max() == field.max()
    assert fine[0, 0, -1] == (field[0, 0, -1] + field[0, 0, 0]) / 2
    assert fine[-1, -1, -1] == pytest.approx(field[:, [0, 2]][:, :, [0, 3]].mean())
    meta = json.loads(output.with_suffix(".json").read_text())
    assert meta["periodic_cell_extent"] == [2, 1.5, 1]
    assert meta["output_dx"] == 0.25
    assert meta["output_shape_xyz"] == [8, 6, 4]

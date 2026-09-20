#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the blocked #107 regret scaffold."""

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "heat3d" / "scripts"))

import heffte_selector_regret as r  # noqa: E402


def test_regret_zero_on_hit():
    assert r.regret(1.5, 1.5) == pytest.approx(0.0)
    assert r.regret(1.65, 1.5) == pytest.approx(0.1)


def test_case_metrics_and_baselines():
    rows = [
        {"reshape": "p2p_plined", "wall_step_s": "2.0"},
        {"reshape": "alltoall", "wall_step_s": "1.0"},
        {"reshape": "p2p", "wall_step_s": "3.0"},
        {"reshape": "alltoallv", "wall_step_s": "1.1"},
    ]
    hit = r.case_metrics(rows, "alltoall")
    assert hit["hit"] is True
    assert hit["regret"] == pytest.approx(0.0)
    miss = r.case_metrics(rows, "p2p_plined")
    assert miss["hit"] is False
    assert miss["regret"] == pytest.approx(1.0)
    assert miss["best"] == "alltoall"


def test_splits_do_not_freeze_heldout():
    assert r.assign_split(768) == "train"
    assert r.assign_split(384) == "heldout_candidate"
    assert 384 in r.CANDIDATE_HELDOUT_FAMILIES


def test_summarize_hit_rate(tmp_path):
    path = tmp_path / "toy.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["family", "nodes", "reshape", "wall_step_s"]
        )
        w.writeheader()
        for proto, wall in (
            ("p2p_plined", 2.0),
            ("alltoall", 1.0),
            ("p2p", 3.0),
            ("alltoallv", 1.2),
        ):
            w.writerow(
                {
                    "family": "768",
                    "nodes": "8",
                    "reshape": proto,
                    "wall_step_s": str(wall),
                }
            )
    assert r.main(["--regret", str(path), "--selected", "p2p_plined"]) == 0
    assert r.main(["--schema"]) == 0

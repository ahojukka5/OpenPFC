#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the issue #61 HeFFTe protocol tournament tooling."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "heat3d" / "scripts"))

import heffte_protocol_tournament as t  # noqa: E402

SUBMIT = ROOT / "docs" / "lumi_slurm" / "submit_heffte_protocol_tournament.sh"


def _run(args, env, cwd):
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        universal_newlines=True,
    )
    out, err = proc.communicate()
    proc.stdout_text = out
    proc.stderr_text = err
    return proc


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _profile(wall):
    return {
        "schema_version": 2,
        "n_mpi_ranks": 1,
        "total_frames": 2,
        "frame_metric_names": ["step", "mpi_rank", "wall_step"],
        "region_paths": [],
        "ranks": [
            {
                "mpi_rank": 0,
                "n_frames": 2,
                "frames": [
                    {"scalars": [0, 0, wall], "regions": {}},
                    {"scalars": [1, 0, wall], "regions": {}},
                ],
            }
        ],
    }


def _run_tree(tmp, job, family, nodes, proto, wall, reshape=None):
    ranks = nodes * 8
    nx = family
    ny = family
    nz = family * ranks
    run = tmp / "runs" / f"{job}"
    proto_dir = run / proto
    reshape = reshape or proto
    _write(
        run / "run_meta.txt",
        "\n".join(
            [
                "=== heffte protocol tournament (issue #61) ===",
                "job=1001",
                f"name={job}",
                "account=project_462001245",
                f"nodes={nodes}",
                f"ntasks={ranks}",
                "partition=standard-g",
                f"Nx={nx}",
                f"Ny={ny}",
                f"Nz={nz}",
                "steps=20",
                "warmup=1",
                f"family={family}",
                "bin_sha256=abc",
                "revision=deadbeef",
                "dirty=0",
            ]
        )
        + "\n",
    )
    _write(
        proto_dir / "run.log",
        (
            "HEAT3D_SPECTRAL_HIP "
            f"N={nx}x{ny}x{nz} ranks={ranks} inbox={family**3} outbox=1 "
            f"inbox_xyz={family}x{family}x{family} outbox_xyz=385x{family}x{family} "
            f"real_grid=1x1x{ranks} complex_grid=1x1x{ranks} "
            f"use_pencils=0 use_reorder=1 reshape={reshape} gpu_aware=1\n"
            "HEAT3D_SPECTRAL_HIP_CHECKSUM sum_u=1.0 sumsq_u=1.0 l2=1.0\n"
        ),
    )
    _write(proto_dir / "admit.txt", "admit=ok\nreason=none\n")
    _write(proto_dir / "timing_profile.json", json.dumps(_profile(wall)))
    return proto_dir


def test_ladder_keeps_local_inbox():
    assert t.check() == 0
    for family in (768, 512):
        for nodes, nx, ny, nz, ranks in t.ladder(family):
            assert t.local_inbox_ok(nx, ny, nz, ranks, family)
            assert nz == family * ranks
            assert ranks == nodes * 8


def test_collect_and_analyze_separate_families(tmp_path):
    _run_tree(tmp_path, "a", 768, 1, "p2p_plined", 0.40)
    _run_tree(tmp_path, "a", 768, 1, "p2p", 0.50)
    _run_tree(tmp_path, "b", 768, 2, "p2p_plined", 0.80)
    _run_tree(tmp_path, "b", 768, 2, "p2p", 0.70)
    _run_tree(tmp_path, "c", 512, 1, "p2p_plined", 0.20)
    rows = t.collect_root(str(tmp_path), warmup=1)
    assert len(rows) == 5
    scale = t.analyze(rows)
    notes = t.crossover_notes(scale)
    fam768 = [r for r in scale if str(r["family"]) == "768"]
    fam512 = [r for r in scale if str(r["family"]) == "512"]
    assert fam768 and fam512
    # 2-node p2p (0.70) beats p2p_plined (0.80): crossover vs 1-node.
    assert any("ranking changes" in n for n in notes)
    plined_2 = next(
        r
        for r in fam768
        if r["reshape"] == "p2p_plined" and int(r["nodes"]) == 2
    )
    assert float(plined_2["t_double"]) == pytest.approx(2.0)
    assert float(plined_2["weak_eff"]) == pytest.approx(0.5)


def test_reject_wrong_banner(tmp_path):
    _run_tree(tmp_path, "bad", 768, 1, "alltoall", 0.4, reshape="p2p_plined")
    rows = t.collect_root(str(tmp_path), warmup=1)
    assert rows[0]["admit"] == "reject"
    assert rows[0]["wall_step_s"] == ""


def test_submit_refuses_old_account():
    env = os.environ.copy()
    env["ACCOUNT"] = "project_462001519"
    env["HEAT3D_SPECTRAL_HIP_BIN"] = "/bin/true"
    proc = _run(["bash", str(SUBMIT), "check"], env, str(ROOT))
    assert proc.returncode == 2
    assert "462001519" in proc.stderr_text


def test_submit_dry_run_768(tmp_path):
    env = os.environ.copy()
    env["ACCOUNT"] = "project_462001245"
    env["HEAT3D_SPECTRAL_HIP_BIN"] = "/bin/true"
    env["DRY_RUN"] = "1"
    env["NODES"] = "1,2"
    env["REPEATS"] = "1"
    env["OPENPFC_SRC"] = str(ROOT)
    env["OPENPFC_SCALING_ROOT"] = str(tmp_path)
    env["SBATCH_ACCOUNT"] = "project_462001519"
    proc = _run(["bash", str(SUBMIT), "768"], env, str(ROOT))
    assert proc.returncode == 0, proc.stderr_text + proc.stdout_text
    assert "--account=project_462001245" in proc.stdout_text
    assert "--export=NONE" in proc.stdout_text
    assert "HEAT3D_RESHAPE_ALG" not in proc.stdout_text
    assert "h3dpt-768-1n-r1" in proc.stdout_text
    assert "h3dpt-768-2n-r1" in proc.stdout_text

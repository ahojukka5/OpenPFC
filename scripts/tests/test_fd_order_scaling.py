#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the issue #108 FD-order scaling campaign tooling."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "heat3d" / "scripts"))

import fd_order_scaling as t  # noqa: E402

SUBMIT = ROOT / "docs" / "lumi_slurm" / "submit_fd_order_scaling.sh"
BATCH = ROOT / "docs" / "lumi_slurm" / "fd_order_scaling.sbatch"


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
                    {"scalars": [5, 0, wall], "regions": {}},
                    {"scalars": [6, 0, wall], "regions": {}},
                ],
            }
        ],
    }


def _run_tree(tmp, job, nodes, order, wall, mode="clean", grid="2x2x2"):
    ranks = nodes * 8
    gx, gy, gz = (int(p) for p in grid.split("x"))
    nx, ny, nz = 256 * gx, 256 * gy, 256 * gz
    run = tmp / "runs" / job
    _write(
        run / "run_meta.txt",
        "\n".join(
            [
                "=== fd-order scaling (issue #108) ===",
                "job=3001",
                f"name={job}",
                "account=project_462001245",
                f"nodes={nodes}",
                f"ntasks={ranks}",
                "partition=standard-g",
                f"Nx={nx}",
                f"Ny={ny}",
                f"Nz={nz}",
                "steps=105",
                "dt=0.01",
                f"fd_order={order}",
                f"mode={mode}",
                "repeat=1",
                "OPENPFC_FD_PROC_GRID=" + grid,
                "bin_sha256=abc",
                "revision=deadbeef",
                "dirty=0",
            ]
        )
        + "\n",
    )
    w = order // 2
    overlap = ""
    if mode == "diagnostic":
        overlap = (
            "HEAT3D_OVERLAP reduce=max_rank_median post_s=0.0001 "
            "exposed_wait_s=0.0002 inner_s=0.0003 border_s=0.0004 mode=1\n"
        )
    _write(
        run / "run.log",
        (
            f"HEAT3D_FD_DECOMP proc_grid={grid.replace('x', 'x')} "
            f"global={nx}x{ny}x{nz} local_min=256x256x256 "
            f"local_max=256x256x256 halo={w} gpu_aware=1 contiguous=1 "
            f"ranks={ranks} fd_order={order} halo_overlap=1\n"
            + overlap
            + "HEAT3D_HIP_CHECKSUM sum_u=1.0 sumsq_u=1.0 l2=1.0\n"
        ),
    )
    _write(run / "admit.txt", "admit=ok\nreason=none\n")
    _write(run / "timing_profile.json", json.dumps(_profile(wall)))
    _write(
        run / "fd_placement.txt",
        "rank host gpu rx ry rz offnode_faces\n0 nid1 0 0 0 0 3\n",
    )
    return run


def test_ladder_matches_admitted_fd2_grids():
    assert t.check() == 0
    by_nodes = {row[0]: row for row in t.ladder()}
    assert by_nodes[1][4] == "2x2x2"
    assert by_nodes[8][1:5] == (1024, 1024, 1024, "4x4x4")
    assert by_nodes[32][4] == "4x8x8"
    assert by_nodes[128][4] == "8x8x16"
    assert by_nodes[512][4] == "16x16x16"
    assert by_nodes[1024][4] == "16x16x32"
    for nodes, nx, ny, nz, grid, ranks in t.ladder():
        gx, gy, gz = (int(p) for p in grid.split("x"))
        assert gx * gy * gz == ranks
        assert (nx // gx, ny // gy, nz // gz) == (256, 256, 256)
        assert "," not in grid


def test_halo_bytes_scale_with_width():
    assert t.halo_width(2) == 1
    assert t.halo_width(20) == 10
    assert t.halo_face_bytes(4) == 2 * t.halo_face_bytes(2)
    assert t.halo_face_bytes(20) == 10 * t.halo_face_bytes(2)


def test_collect_and_analyze_separate_orders(tmp_path):
    _run_tree(tmp_path, "a", 1, 2, 0.001)
    _run_tree(tmp_path, "b", 128, 2, 0.0012, grid="8x8x16")
    _run_tree(tmp_path, "c", 1, 20, 0.002)
    _run_tree(tmp_path, "d", 128, 20, 0.0024, grid="8x8x16")
    _run_tree(tmp_path, "e", 8, 2, 0.0011, mode="diagnostic", grid="4x4x4")
    rows = t.collect_root(str(tmp_path), warmup=5)
    assert len(rows) == 5
    clean = [r for r in rows if r["mode"] == "clean"]
    diag = [r for r in rows if r["mode"] == "diagnostic"]
    assert len(clean) == 4
    assert diag[0]["exposed_wait_s"] == "0.0002"
    assert int(clean[0]["offnode_faces"]) == 3
    scale = t.analyze(rows)
    fd2 = [r for r in scale if int(r["fd_order"]) == 2]
    assert len(fd2) == 2
    one = next(r for r in fd2 if int(r["nodes"]) == 1)
    mid = next(r for r in fd2 if int(r["nodes"]) == 128)
    assert float(one["weak_eff"]) == pytest.approx(1.0)
    assert float(mid["weak_eff"]) == pytest.approx(0.001 / 0.0012, abs=1e-4)
    notes = t.notes_for(scale)
    assert any("not an accuracy claim" in n for n in notes)


def test_reject_nonfinite_checksum(tmp_path):
    run = _run_tree(tmp_path, "nan", 1, 20, 0.002)
    log = run / "run.log"
    log.write_text(log.read_text().replace("sum_u=1.0", "sum_u=nan"))
    rows = t.collect_root(str(tmp_path), warmup=5)
    assert rows[0]["admit"] == "reject"
    assert rows[0]["reason"] == "checksum_nonfinite"


@pytest.mark.parametrize("account", ["project_462001120", "project_462001519"])
def test_submit_refuses_unrelated_account(account):
    env = os.environ.copy()
    env["ACCOUNT"] = account
    env["HEAT3D_HIP_BIN"] = "/bin/true"
    proc = _run(["bash", str(SUBMIT), "check"], env, str(ROOT))
    assert proc.returncode == 2
    assert account in proc.stderr_text


def test_submit_dry_run_clean_and_diag(tmp_path):
    env = os.environ.copy()
    env["ACCOUNT"] = "project_462001245"
    env["HEAT3D_HIP_BIN"] = "/bin/true"
    env["DRY_RUN"] = "1"
    env["NODES"] = "1,8,128"
    env["ORDERS"] = "2,20"
    env["REPEATS"] = "1"
    env["OPENPFC_SRC"] = str(ROOT)
    env["OPENPFC_SCALING_ROOT"] = str(tmp_path)
    env["SBATCH_ACCOUNT"] = "project_462001519"
    proc = _run(["bash", str(SUBMIT), "clean"], env, str(ROOT))
    assert proc.returncode == 0, proc.stderr_text + proc.stdout_text
    assert "--account=project_462001245" in proc.stdout_text
    assert "--export=NONE" in proc.stdout_text
    assert "OPENPFC_FD_PROC_GRID=2x2x2" in proc.stdout_text
    assert "OPENPFC_FD_PROC_GRID=2,2,2" not in proc.stdout_text
    assert "h3d108c-fd2-1n-r1" in proc.stdout_text
    assert "h3d108c-fd20-128n-r1" in proc.stdout_text
    assert "HEAT3D_DIAG_TIMING" not in proc.stdout_text
    diag = _run(["bash", str(SUBMIT), "diag"], env, str(ROOT))
    assert diag.returncode == 0, diag.stderr_text + diag.stdout_text
    assert "h3d108d-fd2-8n-r1" in diag.stdout_text
    assert "HEAT3D_DIAG_TIMING=1" in diag.stdout_text
    assert "h3d108d-fd2-1n-r1" not in diag.stdout_text


@pytest.mark.parametrize("account", ["project_462001519", ""])
def test_batch_refuses_wrong_or_missing_account(account):
    env = os.environ.copy()
    env.update(
        SLURM_JOB_ACCOUNT=account,
        HEAT3D_HIP_BIN="/bin/true",
        HEAT3D_NX="512",
        HEAT3D_NY="512",
        HEAT3D_NZ="512",
        OPENPFC_FD_PROC_GRID="2x2x2",
    )
    proc = _run(["bash", str(BATCH)], env, str(ROOT))
    assert proc.returncode == 2
    assert "refusing #108 job billed" in proc.stderr_text

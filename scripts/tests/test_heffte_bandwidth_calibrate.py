#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the issue #119 HeFFTe-trace vs OSU bandwidth campaign."""

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "heat3d" / "scripts"))

import heffte_bandwidth_calibrate as c  # noqa: E402

SUBMIT = ROOT / "docs" / "lumi_slurm" / "submit_heffte_bandwidth_calibrate.sh"


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


def _clean_env():
    env = os.environ.copy()
    for key in (
        "NODES",
        "HEAT3D_DEPENDENCY",
        "OSU_DEPENDENCY",
        "OPENPFC_SRC",
        "OPENPFC_REVISION",
        "OPENPFC_DIRTY",
        "HEAT3D_SPECTRAL_HIP_BIN",
        "HEAT3D_USE_PENCILS",
        "HEAT3D_GPU_AWARE",
        "OPENPFC_ISSUE",
        "OSU_ALLTOALL_BIN",
        "OSU_ALLTOALLV_BIN",
        "SBATCH_ACCOUNT",
        "ACCOUNT",
        "PARTITION",
        "DRY_RUN",
    ):
        env.pop(key, None)
    return env


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _trace_line(name, start, dur):
    return "%-40s%20.12f%20.12f\n" % (name, start, dur)


def test_check_and_ladder():
    assert c.check() == 0
    rows = c.ladder()
    assert [r[0] for r in rows] == [1, 2, 8, 32]
    assert c.HEAT3D_PROTOCOLS[32] == ("alltoall",)
    assert c.BYTES_PEER[1] == 454164480
    assert c.nz_for(2) == 768 * 16


def test_osu_rate_excludes_self():
    # 8 ranks, 100 byte/peer, 1 us alltoall.
    assert c.osu_gb_s(1e-6, 100, 8) == pytest.approx(100 * 7 / 1e-6 / 1e9)


def test_harvest_trace_and_osu(tmp_path):
    hrun = tmp_path / "runs" / "h3dbw-768-8n-alltoall_1"
    _write(
        hrun / "run_meta.txt",
        "\n".join(
            [
                "=== heffte bandwidth calibrate (issue #119) ===",
                "kind=heat3d_trace",
                "job=9",
                "nodes=8",
                "ntasks=64",
                "reshape=alltoall",
                "steps=20",
                "warmup=1",
            ]
        )
        + "\n",
    )
    _write(
        hrun / "run.log",
        "HEAT3D_SPECTRAL_HIP N=768x768x49152 ranks=64 use_pencils=0 "
        "use_reorder=1 reshape=alltoall gpu_aware=1\n",
    )
    # 20 steps * 0.10 s MPI on the critical rank.
    body = ""
    t = 0.0
    for _ in range(20):
        body += _trace_line("all2all", t, 0.10)
        body += _trace_line("packing", t + 0.10, 0.02)
        body += _trace_line("fft-1d", t + 0.12, 0.03)
        t += 0.15
    _write(hrun / "heffte_trace_0.log", body)
    _write(hrun / "heffte_trace_1.log", _trace_line("all2all", 0.0, 0.01))
    _write(hrun / "admit.txt", "admit=ok\nreason=none\nexpected_reshape=alltoall\n")

    orun = tmp_path / "runs" / "osubw-768-8n-alltoall_1"
    _write(
        orun / "run_meta.txt",
        "\n".join(
            [
                "kind=osu",
                "job=10",
                "nodes=8",
                "ntasks=64",
                "osu_kind=alltoall",
            ]
        )
        + "\n",
    )
    peer = c.BYTES_PEER[8]
    # 1 ms alltoall at the campaign peer size.
    _write(
        orun / "osu.out",
        "# Size       Avg Latency(us)\n%d                 1000.0\n" % peer,
    )

    out = tmp_path / "results"
    assert (
        c.main(["--harvest", str(tmp_path), "--out", str(out)]) == 0
    )
    with (out / "bandwidth.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    kinds = {r["kind"]: r for r in rows}
    assert kinds["heat3d_trace"]["admit"] == "ok"
    assert kinds["heat3d_trace"]["use_pencils"] == "0"
    assert kinds["heat3d_trace"]["gpu_aware"] == "1"
    assert float(kinds["heat3d_trace"]["t_mpi_s"]) == pytest.approx(0.10)
    assert float(kinds["heat3d_trace"]["n_mpi_coll_per_step"]) == pytest.approx(1.0)
    assert float(kinds["heat3d_trace"]["gb_s_mpi"]) == pytest.approx(
        c.BYTES_STEP[8] / 0.10 / 1e9
    )
    assert kinds["osu"]["admit"] == "ok"
    assert float(kinds["osu"]["gb_s_osu"]) == pytest.approx(
        c.osu_gb_s(0.001, peer, 64)
    )


def test_reject_missing_trace(tmp_path):
    run = tmp_path / "runs" / "h3dbw-768-1n-alltoall_1"
    _write(
        run / "run_meta.txt",
        "kind=heat3d_trace\njob=1\nnodes=1\nntasks=8\nreshape=alltoall\nsteps=20\n",
    )
    _write(run / "run.log", "HEAT3D_SPECTRAL_HIP reshape=alltoall gpu_aware=1\n")
    rows = c.collect(str(tmp_path))
    assert rows[0]["admit"] == "reject"
    assert rows[0]["reason"] == "missing_trace"


@pytest.mark.parametrize("account", ["project_462001245", "project_462001120"])
def test_submit_refuses_campaign_account(account):
    env = _clean_env()
    env["ACCOUNT"] = account
    env["HEAT3D_SPECTRAL_HIP_BIN"] = "/bin/true"
    proc = _run(["bash", str(SUBMIT), "check"], env, str(ROOT))
    assert proc.returncode == 2
    assert account in proc.stderr_text


@pytest.mark.parametrize("account", ["project_462001245", ""])
def test_batch_refuses_wrong_or_missing_account(account):
    env = _clean_env()
    env.update(
        SLURM_JOB_ACCOUNT=account,
        SLURM_NTASKS="8",
        RUN_KIND="heat3d",
        HEAT3D_SPECTRAL_HIP_BIN="/bin/true",
        HEAT3D_NX="768",
        HEAT3D_NY="768",
        HEAT3D_NZ="6144",
        HEAT3D_RESHAPE_ALG="alltoall",
    )
    batch = ROOT / "docs/lumi_slurm/heffte_bandwidth_calibrate.sbatch"
    proc = _run(["bash", str(batch)], env, str(ROOT))
    assert proc.returncode == 2
    assert "refusing bandwidth job billed" in proc.stderr_text


def test_submit_dry_run_768(tmp_path):
    env = _clean_env()
    env["ACCOUNT"] = "project_462001519"
    env["HEAT3D_SPECTRAL_HIP_BIN"] = "/bin/true"
    env["OSU_ALLTOALL_BIN"] = "/bin/true"
    env["OSU_ALLTOALLV_BIN"] = "/bin/true"
    env["DRY_RUN"] = "1"
    env["OPENPFC_SRC"] = str(ROOT)
    env["OPENPFC_SCALING_ROOT"] = str(tmp_path)
    env["SBATCH_ACCOUNT"] = "project_462001245"
    proc = _run(["bash", str(SUBMIT), "768"], env, str(ROOT))
    assert proc.returncode == 0, proc.stderr_text + proc.stdout_text
    assert "--account=project_462001519" in proc.stdout_text
    assert "project_462001245" not in proc.stdout_text.split("--account=")[-1][:20]
    assert "h3dbw-768-1n-p2p_plined" in proc.stdout_text
    assert "h3dbw-768-32n-alltoall" in proc.stdout_text
    assert "osubw-768-8n-alltoallv" in proc.stdout_text
    assert "h3dbw-768-32n-alltoallv" not in proc.stdout_text
    assert "h3dbw-768-32n-alltoall-pencils" not in proc.stdout_text
    assert "--partition=standard-g" in proc.stdout_text
    assert "HEAT3D_RESHAPE_ALG=alltoall" in proc.stdout_text
    assert "HEAT3D_USE_PENCILS=0" in proc.stdout_text
    assert "HEAT3D_GPU_AWARE=1" in proc.stdout_text
    assert "HEAT3D_PROTOCOLS" not in proc.stdout_text


def test_submit_ignores_leaked_src(tmp_path):
    env = _clean_env()
    env["ACCOUNT"] = "project_462001519"
    env["HEAT3D_SPECTRAL_HIP_BIN"] = "/bin/true"
    env["OSU_ALLTOALL_BIN"] = "/bin/true"
    env["OSU_ALLTOALLV_BIN"] = "/bin/true"
    env["DRY_RUN"] = "1"
    env["NODES"] = "1"
    env["OPENPFC_SRC"] = str(tmp_path)
    env["OPENPFC_SCALING_ROOT"] = str(tmp_path)
    proc = _run(["bash", str(SUBMIT), "768"], env, str(ROOT))
    assert proc.returncode == 0, proc.stderr_text + proc.stdout_text
    assert "ignoring leaked OPENPFC_SRC" in proc.stderr_text


def test_submit_ignores_leaked_production_bin(tmp_path):
    env = _clean_env()
    env["ACCOUNT"] = "project_462001519"
    env["HEAT3D_SPECTRAL_HIP_BIN"] = (
        "/flash/project_462001245/juaho/build/openpfc-lumi-rocm-e98d8a87/"
        "apps/heat3d/heat3d_spectral_hip"
    )
    env["OSU_ALLTOALL_BIN"] = "/bin/true"
    env["OSU_ALLTOALLV_BIN"] = "/bin/true"
    env["DRY_RUN"] = "1"
    env["NODES"] = "1"
    env["OPENPFC_SRC"] = str(ROOT)
    env["OPENPFC_SCALING_ROOT"] = str(tmp_path)
    proc = _run(["bash", str(SUBMIT), "768"], env, str(ROOT))
    assert proc.returncode == 0, proc.stderr_text + proc.stdout_text
    assert "ignoring leaked HEAT3D_SPECTRAL_HIP_BIN" in proc.stderr_text
    assert "heffte-trace-heat3d" in proc.stdout_text
    assert "openpfc-lumi-rocm-e98d8a87" not in proc.stdout_text


def test_harvest_pencils_does_not_reuse_slab_bytes(tmp_path):
    hrun = tmp_path / "runs" / "h3dbw-768-32n-alltoall-pencils_1"
    _write(
        hrun / "run_meta.txt",
        "\n".join(
            [
                "kind=heat3d_trace",
                "issue=121",
                "job=11",
                "nodes=32",
                "ntasks=256",
                "reshape=alltoall",
                "use_pencils=1",
                "gpu_aware=1",
                "steps=20",
                "warmup=1",
            ]
        )
        + "\n",
    )
    _write(
        hrun / "run.log",
        "HEAT3D_SPECTRAL_HIP N=768x768x196608 ranks=256 "
        "real_grid=16x16x1 complex_grid=16x16x1 use_pencils=1 "
        "reshape=alltoall gpu_aware=1\n",
    )
    body = ""
    t = 0.0
    for _ in range(20):
        body += _trace_line("all2all", t, 0.20)
        body += _trace_line("all2all", t + 0.20, 0.20)
        body += _trace_line("packing", t + 0.40, 0.05)
        t += 0.50
    _write(hrun / "heffte_trace_0.log", body)
    _write(hrun / "admit.txt", "admit=ok\nreason=none\nexpected_use_pencils=1\n")
    rows = c.collect(str(tmp_path))
    assert len(rows) == 1
    row = rows[0]
    assert row["issue"] == "121"
    assert row["admit"] == "ok"
    assert row["use_pencils"] == "1"
    assert row["gpu_aware"] == "1"
    assert row["real_grid"] == "16x16x1"
    assert float(row["n_mpi_coll_per_step"]) == pytest.approx(2.0)
    assert row["gb_s_mpi"] is None
    assert row["bytes_per_timestep"] is None


def test_submit_dry_run_pencils(tmp_path):
    env = _clean_env()
    env["ACCOUNT"] = "project_462001519"
    env["HEAT3D_SPECTRAL_HIP_BIN"] = "/bin/true"
    env["DRY_RUN"] = "1"
    env["OPENPFC_SRC"] = str(ROOT)
    env["OPENPFC_SCALING_ROOT"] = str(tmp_path)
    proc = _run(["bash", str(SUBMIT), "pencils"], env, str(ROOT))
    assert proc.returncode == 0, proc.stderr_text + proc.stdout_text
    assert "h3dbw-768-32n-alltoall-pencils" in proc.stdout_text
    assert "HEAT3D_USE_PENCILS=1" in proc.stdout_text
    assert "HEAT3D_GPU_AWARE=1" in proc.stdout_text
    assert "OPENPFC_ISSUE=121" in proc.stdout_text
    assert "--nodes=32" in proc.stdout_text
    assert "--partition=standard-g" in proc.stdout_text
    assert "osubw-" not in proc.stdout_text
    assert "h3dbw-768-1n-" not in proc.stdout_text
    assert "HEAT3D_USE_PENCILS=0" not in proc.stdout_text

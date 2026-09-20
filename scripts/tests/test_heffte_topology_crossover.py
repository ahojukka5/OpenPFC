#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the issue #106 HeFFTe topology-crossover campaign tooling."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "heat3d" / "scripts"))

import heffte_topology_crossover as t  # noqa: E402

SUBMIT = ROOT / "docs" / "lumi_slurm" / "submit_heffte_topology_crossover.sh"
BATCH = ROOT / "docs" / "lumi_slurm" / "heffte_topology_crossover.sbatch"


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


def _run_tree(tmp, job, nodes, proto, wall, reshape=None, order=None, seed=1,
              repeat=1):
    family = 768
    ranks = nodes * 8
    nx = family
    ny = family
    nz = family * ranks
    run = tmp / "runs" / job
    proto_dir = run / proto
    reshape = reshape or proto
    order = order or "p2p_plined:p2p:alltoallv:alltoall"
    _write(
        run / "run_meta.txt",
        "\n".join(
            [
                "=== heffte topology crossover (issue #106) ===",
                f"job={job}",
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
                f"repeat={repeat}",
                f"protocol_seed={seed}",
                f"protocols={order}",
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
    _write(run / "placement" / "hostnames.txt", "nid005000\nnid005001\n")
    _write(
        run / "placement" / "node_facts.txt",
        "host=nid005000 nid=5000 cname=x1000c0s0b0n0 xname=x1000c0s0b0n0 cxi=cxi0\n"
        "host=nid005001 nid=5001 cname=x1000c0s0b1n0 xname=x1000c0s0b1n0 cxi=cxi0\n",
    )
    return proto_dir


def test_wave1_keeps_local_inbox_and_diversifies_order():
    assert t.check() == 0
    assert 124 in t.WAVE1_NODES
    assert 128 in t.WAVE1_NODES
    assert 256 in t.WAVE1_NODES
    assert 512 in t.WAVE1_NODES
    orders = set()
    for nodes, nx, ny, nz, ranks in t.wave1():
        assert t.local_grid(768, nodes)[4] == ranks
        assert nz == 768 * ranks
        assert nx == 768 and ny == 768
        for repeat in (1, 2, 3):
            seed, algs = t.protocol_order(768, nodes, repeat)
            assert seed == t.protocol_seed(768, nodes, repeat)
            assert sorted(algs) == sorted(t.PROTOCOLS)
            orders.add(tuple(algs))
    assert len(orders) > 1
    seed_a, order_a = t.protocol_order(768, 128, 1)
    seed_b, order_b = t.protocol_order(768, 128, 2)
    assert seed_a != seed_b
    # Deterministic: a second call matches the first.
    assert t.protocol_order(768, 128, 1) == (seed_a, order_a)
    assert order_a or order_b


def test_collect_keeps_per_allocation_order_and_placement(tmp_path):
    order = "alltoall:p2p_plined:p2p:alltoallv"
    _run_tree(tmp_path, "a", 128, "alltoall", 1.8, order=order, seed=11)
    _run_tree(tmp_path, "a", 128, "p2p_plined", 1.6, order=order, seed=11)
    rows = t.collect_root(str(tmp_path), warmup=1)
    assert len(rows) == 4
    by_proto = {r["reshape"]: r for r in rows}
    assert by_proto["p2p_plined"]["protocol_order"] == order
    assert by_proto["alltoall"]["protocol_index"] == "0"
    assert by_proto["p2p_plined"]["protocol_index"] == "1"
    assert by_proto["p2p_plined"]["admit"] == "ok"
    assert by_proto["p2p"]["admit"] == "missing"
    assert by_proto["p2p_plined"]["n_hosts"] == "2"
    assert by_proto["p2p_plined"]["nid_min"] == "5000"
    assert "1000" in by_proto["p2p_plined"]["xname_cabinets"]


def test_analyze_keeps_allocation_winners(tmp_path):
    order = "p2p:alltoallv:alltoall:p2p_plined"
    for proto, wall in (
        ("p2p", 2.4),
        ("alltoallv", 1.99),
        ("alltoall", 1.80),
        ("p2p_plined", 1.66),
    ):
        _run_tree(tmp_path, "j1", 128, proto, wall, order=order, seed=3)
    for proto, wall in (
        ("p2p", 2.5),
        ("alltoallv", 1.68),
        ("alltoall", 1.72),
        ("p2p_plined", 1.70),
    ):
        _run_tree(tmp_path, "j2", 136, proto, wall, order=order, seed=4)
    rows = t.collect_root(str(tmp_path), warmup=1)
    scale, alloc = t.analyze(rows)
    assert len(alloc) == 2
    winners = {int(r["nodes"]): r["winner"] for r in alloc}
    assert winners[128] == "p2p_plined"
    assert winners[136] == "alltoallv"
    notes = t.crossover_notes(scale, alloc)
    assert any("ranking changes" in n for n in notes)
    assert any("electrical-group" in n for n in notes)


def test_reject_wrong_banner(tmp_path):
    _run_tree(tmp_path, "bad", 128, "alltoall", 0.4, reshape="p2p_plined")
    rows = t.collect_root(str(tmp_path), warmup=1)
    by_proto = {r["reshape"]: r for r in rows}
    assert by_proto["alltoall"]["admit"] == "reject"
    assert by_proto["alltoall"]["wall_step_s"] == ""


@pytest.mark.parametrize("account", ["project_462001120", "project_462001519"])
def test_submit_refuses_unrelated_account(account):
    env = os.environ.copy()
    env["ACCOUNT"] = account
    env["HEAT3D_SPECTRAL_HIP_BIN"] = "/bin/true"
    proc = _run(["bash", str(SUBMIT), "check"], env, str(ROOT))
    assert proc.returncode == 2
    assert account in proc.stderr_text


def test_submit_dry_run_randomizes_order(tmp_path):
    env = os.environ.copy()
    env["ACCOUNT"] = "project_462001245"
    env["HEAT3D_SPECTRAL_HIP_BIN"] = "/bin/true"
    env["DRY_RUN"] = "1"
    env["NODES"] = "124,128"
    env["REPEATS"] = "2"
    env["OPENPFC_SRC"] = str(ROOT)
    env["OPENPFC_SCALING_ROOT"] = str(tmp_path)
    env["SBATCH_ACCOUNT"] = "project_462001519"
    proc = _run(["bash", str(SUBMIT), "wave1"], env, str(ROOT))
    assert proc.returncode == 0, proc.stderr_text + proc.stdout_text
    assert "--account=project_462001245" in proc.stdout_text
    assert "--export=NONE" in proc.stdout_text
    assert "HEAT3D_RESHAPE_ALG" not in proc.stdout_text
    assert "h3d106-768-124n-r1" in proc.stdout_text
    assert "h3d106-768-128n-r2" in proc.stdout_text
    assert "HEAT3D_PROTOCOL_SEED=" in proc.stdout_text
    assert "HEAT3D_PROTOCOLS=" in proc.stdout_text
    assert "HEAT3D_PROTOCOLS=p2p_plined,p2p" not in proc.stdout_text
    seeds = []
    for line in proc.stdout_text.splitlines():
        if "seed=" in line and "nodes=128" in line:
            for tok in line.split():
                if tok.startswith("seed="):
                    seeds.append(tok.split("=", 1)[1])
    assert len(set(seeds)) == 2


@pytest.mark.parametrize("account", ["project_462001519", ""])
def test_batch_refuses_wrong_or_missing_account(account):
    env = os.environ.copy()
    env.update(
        SLURM_JOB_ACCOUNT=account,
        HEAT3D_SPECTRAL_HIP_BIN="/bin/true",
        HEAT3D_NX="768",
        HEAT3D_NY="768",
        HEAT3D_NZ="768",
    )
    proc = _run(["bash", str(BATCH)], env, str(ROOT))
    assert proc.returncode == 2
    assert "refusing #106 job billed" in proc.stderr_text

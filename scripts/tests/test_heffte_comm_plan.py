#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-execution HeFFTe layout replica for issue #106."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "heat3d" / "scripts"))

import heffte_comm_plan as p  # noqa: E402
import heffte_topology_crossover as t  # noqa: E402


def test_split_matches_heffte_uneven_y_slab():
    parts = p.split_axis(1200, 32)
    assert parts[0] == 38
    assert parts[-1] == 37
    assert sum(parts) == 1200


def test_slab_grid_is_z_for_768_family():
    assert p.slab_proc_grid((768, 768, 768 * 1024), 1024) == (1, 1, 1024)


def test_frozen_banners_from_61_and_106():
    g16 = p.campaign_geometry(768, 16)
    assert p.fmt_grid(g16["complex_grid"]) == "1x128x1"
    g128 = p.campaign_geometry(768, 128)
    assert p.fmt_grid(g128["complex_grid"]) == "4x256x1"
    assert p.fmt_grid(g128["outbox_xyz"]) == "97x3x786432"
    g136 = p.campaign_geometry(768, 136)
    assert p.fmt_grid(g136["complex_grid"]) == "17x64x1"
    assert p.fmt_grid(g136["outbox_xyz"]) == "23x12x835584"
    assert g136["n_mpi_reshapes"] == 1


def test_descriptors_have_no_wall_times():
    rows = p.descriptor_rows(t.WAVE1_NODES)
    assert len(rows) == len(t.WAVE1_NODES) * len(p.PROTOCOLS)
    for row in rows:
        assert "wall" not in row
        assert row["plan_source"] == "openpfc_slab_r2c_min_reshape_replica"
        assert row["inbox_xyz"] == "768x768x768"
        assert int(row["n_mpi_reshapes"]) >= 1
        assert row["reshape"] in p.PROTOCOLS


def test_collect_keeps_missing_protocol_rows(tmp_path):
    run = tmp_path / "runs" / "h3d106-768-128n-r1"
    run.mkdir(parents=True)
    (run / "run_meta.txt").write_text(
        "\n".join(
            [
                "job=9",
                "account=project_462001245",
                "nodes=128",
                "ntasks=1024",
                "family=768",
                "repeat=1",
                "protocols=p2p:alltoall:p2p_plined:alltoallv",
                "Nx=768",
                "Ny=768",
                "Nz=786432",
            ]
        )
        + "\n"
    )
    (run / "p2p").mkdir()
    (run / "p2p" / "admit.txt").write_text("admit=reject\nreason=srun_exit_127\n")
    (run / "p2p" / "run.log").write_text("libfabric.so.1: cannot open\n")
    rows = t.collect_root(str(tmp_path), warmup=1)
    admits = {r["reshape"]: r["admit"] for r in rows}
    assert admits["p2p"] == "reject"
    assert admits["alltoall"] == "missing"
    assert admits["p2p_plined"] == "missing"
    assert all(r.get("wall_step_s") == "" or r["admit"] != "ok" for r in rows)


def test_status_marks_single_allocation_unresolved(tmp_path):
    family = 768
    nodes = 136
    ranks = nodes * 8
    run = tmp_path / "runs" / "h3d106-768-136n-r1"
    order = "p2p:alltoall:p2p_plined:alltoallv"
    (run).mkdir(parents=True)
    (run / "run_meta.txt").write_text(
        "job=22186611\naccount=project_462001245\nnodes=136\nntasks=%d\n"
        "partition=standard-g\nNx=768\nNy=768\nNz=%d\nfamily=768\nrepeat=1\n"
        "protocol_seed=1\nprotocols=%s\n" % (ranks, family * ranks, order)
    )
    for i, proto in enumerate(("p2p", "alltoall", "p2p_plined", "alltoallv")):
        proto_dir = run / proto
        proto_dir.mkdir()
        (proto_dir / "admit.txt").write_text("admit=ok\nreason=none\n")
        (proto_dir / "run.log").write_text(
            "HEAT3D_SPECTRAL_HIP N=768x768x%d ranks=%d inbox=1 outbox=1 "
            "inbox_xyz=768x768x768 outbox_xyz=23x12x835584 "
            "real_grid=1x1x%d complex_grid=17x64x1 use_pencils=0 "
            "use_reorder=1 reshape=%s gpu_aware=1\n"
            "HEAT3D_SPECTRAL_HIP_CHECKSUM sum_u=1.0 sumsq_u=1.0 l2=1.0\n"
            % (family * ranks, ranks, ranks, proto)
        )
        (proto_dir / "timing_profile.json").write_text(
            '{"schema_version":2,"n_mpi_ranks":1,"total_frames":2,'
            '"frame_metric_names":["step","mpi_rank","wall_step"],'
            '"ranks":[{"mpi_rank":0,"n_frames":2,"frames":['
            '{"scalars":[0,0,%.2f],"regions":{}},'
            '{"scalars":[1,0,%.2f],"regions":{}}]}]}' % (1.4 + 0.1 * i, 1.4 + 0.1 * i)
        )
    rc = t.harvest(str(tmp_path), str(tmp_path / "results"), warmup=1)
    assert rc == 0
    status = (tmp_path / "results" / "status.md").read_text()
    assert "136" in status
    assert "yes_repeats_incomplete" in status
    assert "p2p_plined" in status.splitlines()[4]
    assert "alltoall" in status
    order_csv = (tmp_path / "results" / "order.csv").read_text()
    assert "protocol_index" in order_csv
    assert "alltoall" in order_csv

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-execution HeFFTe reshape descriptors for issue #106.

Replicates the OpenPFC production path used by Heat3D spectral HIP:

* real process grid = ``slab_proc_grid`` with r2c in x (prefer z, then y);
* complex world = r2c box ``(Nx/2+1, Ny, Nz)``;
* complex grid = min-reshape selector among legal Cartesian splits
  (``complex_outbox.hpp`` ranking, with MPI-reshape count from the
  z-slab geometric classification that matches frozen #61/#106 banners).

``reshape_algorithm`` does not change the selected grid or MPI-reshape
count (pinned in ``test_complex_outbox.cpp``). Protocol rows therefore
share geometry and differ only in the MPI primitive.

This module emits **no wall times**.
"""

from __future__ import print_function

import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROTOCOLS = ("p2p_plined", "p2p", "alltoallv", "alltoall")
GCDS_PER_NODE = 8
FAMILY_PRIMARY = 768
COMPLEX_DTYPE_BYTES = 16  # complex double
FFTS_PER_STEP = 2  # Heat3D spectral HIP: forward + inverse
R2C_DIR = 0

DESCRIPTOR_FIELDS = (
    "issue",
    "family",
    "nodes",
    "ranks",
    "reshape",
    "real_grid",
    "complex_grid",
    "inbox_xyz",
    "outbox_xyz",
    "n_reshape_stages",
    "n_mpi_reshapes",
    "n_local_reshapes",
    "n_identity",
    "predicted_all_peer_rounds_per_cycle",
    "comm_group_size",
    "n_peers",
    "n_onnode_peers",
    "n_offnode_peers",
    "bytes_per_rank_per_reshape",
    "bytes_per_timestep",
    "bytes_per_peer",
    "node_local",
    "plan_source",
    "model_features",
)

# Candidate linear models. Coefficients are not fit here.
COMM_MODELS = (
    {
        "name": "messages_bytes_peers",
        "formula": "T_comm = a*n_messages + b*bytes + c*n_peers",
        "known_pre_exec": ("n_messages", "bytes", "n_peers", "n_mpi_reshapes"),
        "needs_measured": ("T_comm", "alpha", "beta", "gamma"),
    },
    {
        "name": "per_reshape_hockney",
        "formula": (
            "T_comm = sum_s (a_s + b_s * bytes_s / min(n_peers_s, 1)) "
            "with protocol-specific (a_s, b_s)"
        ),
        "known_pre_exec": ("n_mpi_reshapes", "bytes_per_reshape", "comm_group_size"),
        "needs_measured": ("per-protocol latency/bandwidth", "T_comm"),
    },
    {
        "name": "onnode_offnode",
        "formula": "T_comm = a*n_onnode_peers + b*n_offnode_peers + c*bytes",
        "known_pre_exec": ("n_onnode_peers", "n_offnode_peers", "bytes"),
        "needs_measured": (
            "allocation placement",
            "T_comm",
            "whether electrical-group locality is exposed",
        ),
    },
)


def fmt_grid(g: Sequence[int]) -> str:
    return "%dx%dx%d" % (int(g[0]), int(g[1]), int(g[2]))


def split_axis(size: int, nparts: int) -> List[int]:
    """HeFFTe/OpenPFC split: first ``size % nparts`` boxes get the extra cell."""
    if nparts < 1 or size < 1:
        raise ValueError("split_axis requires size>=1 and nparts>=1")
    base = size // nparts
    extra = size % nparts
    return [base + (1 if i < extra else 0) for i in range(nparts)]


def split_world(world: Sequence[int], grid: Sequence[int]) -> List[Tuple[int, int, int]]:
    xs = split_axis(world[0], grid[0])
    ys = split_axis(world[1], grid[1])
    zs = split_axis(world[2], grid[2])
    boxes = []
    for z in zs:
        for y in ys:
            for x in xs:
                boxes.append((x, y, z))
    return boxes


def r2c_complex_world(real: Sequence[int], r2c: int = R2C_DIR) -> Tuple[int, int, int]:
    nx, ny, nz = int(real[0]), int(real[1]), int(real[2])
    if r2c == 0:
        return (nx // 2 + 1, ny, nz)
    if r2c == 1:
        return (nx, ny // 2 + 1, nz)
    if r2c == 2:
        return (nx, ny, nz // 2 + 1)
    raise ValueError("r2c must be 0, 1, or 2")


def slab_proc_grid(size: Sequence[int], nproc: int, r2c: int = R2C_DIR) -> Tuple[int, int, int]:
    """Match ``pfc::decomposition::slab_proc_grid`` (r2c-in-plane preference)."""
    if nproc <= 1:
        return (1, 1, 1)
    pref = ((2, 1, 0), (2, 0, 1), (1, 0, 2))
    for d in pref[r2c]:
        if nproc <= size[d] and size[d] % nproc == 0:
            g = [1, 1, 1]
            g[d] = nproc
            return (g[0], g[1], g[2])
    raise ValueError("no even 1-D slab split for size=%s nproc=%d" % (size, nproc))


def legal_complex_proc_grids(
    world: Sequence[int], nproc: int
) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    nx, ny, nz = world
    for i in range(1, nproc + 1):
        if i > nx or nproc % i != 0:
            continue
        rest = nproc // i
        for j in range(1, rest + 1):
            if j > ny or rest % j != 0:
                continue
            k = rest // j
            if k < 1 or k > nz:
                continue
            out.append((i, j, k))
    return out


def estimate_mpi_reshapes(
    real_grid: Sequence[int], complex_grid: Sequence[int]
) -> Tuple[int, int, int, int]:
    """Geometric stand-in for ``heffte::plan_operations`` on this campaign.

    Heat3D #106 uses z-slabs (``1x1xP``) and r2c in x. Real and complex
    worlds have different extents, so a matching process grid is not an
    identity reshape. A complex outbox with ``gz==1`` needs one distributed
    reshape to gather z; ``gz>1`` needs a second. One rank is local.
    This reproduces the frozen 768³/GCD banners (16-node ``1x128x1``,
    128-node ``4x256x1``, 136-node ``17x64x1``).
    """
    nproc = int(real_grid[0]) * int(real_grid[1]) * int(real_grid[2])
    if nproc <= 1:
        return 0, 0, 4, 0
    n_mpi = 1 if int(complex_grid[2]) == 1 else 2
    n_ident = 4 - n_mpi
    return n_mpi, 0, n_ident, n_mpi


def volume_imbalance(world: Sequence[int], grid: Sequence[int]) -> float:
    boxes = split_world(world, grid)
    counts = [b[0] * b[1] * b[2] for b in boxes]
    return float(max(counts)) / float(min(counts))


def box_aspect(world: Sequence[int], grid: Sequence[int]) -> float:
    boxes = split_world(world, grid)
    mins = [min(b[d] for b in boxes) for d in range(3)]
    maxs = [max(b[d] for b in boxes) for d in range(3)]
    lo = min(mins)
    hi = max(maxs)
    if lo <= 0:
        return float("inf")
    return float(hi) / float(lo)


def select_min_reshape_complex_grid(
    real_world: Sequence[int],
    real_grid: Sequence[int],
    complex_world: Sequence[int],
) -> Tuple[int, int, int]:
    nproc = real_grid[0] * real_grid[1] * real_grid[2]
    grids = legal_complex_proc_grids(complex_world, nproc)
    if not grids:
        raise ValueError("no legal complex process grid")
    best = None
    winner = None
    for g in grids:
        n_mpi, n_local, n_ident, n_stages = estimate_mpi_reshapes(real_grid, g)
        rounds = 2 * n_mpi
        imb = volume_imbalance(complex_world, g)
        asp = box_aspect(complex_world, g)
        key = (n_mpi, rounds, imb, asp, g)
        if best is None or key < best:
            best = key
            winner = g
            _ = (n_local, n_ident, n_stages)
    assert winner is not None
    return winner


def campaign_geometry(family: int, nodes: int) -> Dict[str, Any]:
    ranks = nodes * GCDS_PER_NODE
    nx = ny = family
    nz = family * ranks
    real_world = (nx, ny, nz)
    real_grid = slab_proc_grid(real_world, ranks, R2C_DIR)
    complex_world = r2c_complex_world(real_world, R2C_DIR)
    complex_grid = select_min_reshape_complex_grid(
        real_world, real_grid, complex_world
    )
    inbox = (
        nx // real_grid[0],
        ny // real_grid[1],
        nz // real_grid[2],
    )
    outbox = split_world(complex_world, complex_grid)[0]
    n_mpi, n_local, n_ident, n_stages = estimate_mpi_reshapes(
        real_grid, complex_grid
    )
    local_elems = outbox[0] * outbox[1] * outbox[2]
    bytes_reshape = local_elems * COMPLEX_DTYPE_BYTES
    group = ranks
    n_peers = max(group - 1, 0)
    onnode = min(GCDS_PER_NODE - 1, n_peers) if nodes >= 1 else 0
    if nodes == 1:
        onnode = n_peers
    offnode = n_peers - onnode
    node_local = "yes" if nodes == 1 or n_mpi == 0 else "no"
    bytes_peer = bytes_reshape // group if group else 0
    return {
        "family": family,
        "nodes": nodes,
        "ranks": ranks,
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "real_grid": real_grid,
        "complex_grid": complex_grid,
        "inbox_xyz": inbox,
        "outbox_xyz": outbox,
        "n_mpi_reshapes": n_mpi,
        "n_local_reshapes": n_local,
        "n_identity": n_ident,
        "n_reshape_stages": n_stages,
        "predicted_all_peer_rounds_per_cycle": 2 * n_mpi,
        "comm_group_size": group,
        "n_peers": n_peers,
        "n_onnode_peers": onnode,
        "n_offnode_peers": offnode,
        "bytes_per_rank_per_reshape": bytes_reshape,
        "bytes_per_timestep": bytes_reshape * n_mpi * FFTS_PER_STEP,
        "bytes_per_peer": bytes_peer,
        "node_local": node_local,
        "plan_source": "openpfc_slab_r2c_min_reshape_replica",
    }


def descriptor_rows(
    nodes_list: Sequence[int],
    family: int = FAMILY_PRIMARY,
    protocols: Sequence[str] = PROTOCOLS,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for nodes in nodes_list:
        geo = campaign_geometry(family, nodes)
        for proto in protocols:
            n_msg = geo["n_peers"] * geo["n_mpi_reshapes"] * FFTS_PER_STEP
            if proto in ("alltoall", "alltoallv"):
                n_msg = geo["n_mpi_reshapes"] * FFTS_PER_STEP
            rows.append(
                {
                    "issue": "106",
                    "family": family,
                    "nodes": nodes,
                    "ranks": geo["ranks"],
                    "reshape": proto,
                    "real_grid": fmt_grid(geo["real_grid"]),
                    "complex_grid": fmt_grid(geo["complex_grid"]),
                    "inbox_xyz": fmt_grid(geo["inbox_xyz"]),
                    "outbox_xyz": fmt_grid(geo["outbox_xyz"]),
                    "n_reshape_stages": geo["n_reshape_stages"],
                    "n_mpi_reshapes": geo["n_mpi_reshapes"],
                    "n_local_reshapes": geo["n_local_reshapes"],
                    "n_identity": geo["n_identity"],
                    "predicted_all_peer_rounds_per_cycle": geo[
                        "predicted_all_peer_rounds_per_cycle"
                    ],
                    "comm_group_size": geo["comm_group_size"],
                    "n_peers": geo["n_peers"],
                    "n_onnode_peers": geo["n_onnode_peers"],
                    "n_offnode_peers": geo["n_offnode_peers"],
                    "bytes_per_rank_per_reshape": geo["bytes_per_rank_per_reshape"],
                    "bytes_per_timestep": geo["bytes_per_timestep"],
                    "bytes_per_peer": geo["bytes_per_peer"],
                    "node_local": geo["node_local"],
                    "plan_source": geo["plan_source"],
                    "model_features": "n_messages=%d;bytes=%d;n_peers=%d"
                    % (n_msg, geo["bytes_per_timestep"], geo["n_peers"]),
                }
            )
    return rows


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(DESCRIPTOR_FIELDS), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def model_notes() -> List[str]:
    notes = [
        "Coefficients were not fit. The #106 campaign stopped without a selector.",
        "reshape_algorithm is a timing mechanism, not a layout feature.",
        "n_mpi_reshapes is the geometric OpenPFC replica, not a live "
        "heffte::plan_operations dump (that dump is not available pre-job).",
    ]
    for m in COMM_MODELS:
        notes.append("%s: %s" % (m["name"], m["formula"]))
        notes.append(
            "  known pre-exec: %s; measured later: %s"
            % (", ".join(m["known_pre_exec"]), ", ".join(m["needs_measured"]))
        )
    return notes

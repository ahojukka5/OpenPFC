#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""XDMF sidecar for showcase BinaryWriter dumps."""

from __future__ import print_function

import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import xdmfgen  # noqa: E402


def test_manifest_xdmf_points_at_bricks():
    tmp = tempfile.mkdtemp(prefix="xdmfgen-")
    nx, ny, nz = 2, 2, 1
    payload = struct.pack("<4d", 1.0, 2.0, 3.0, 4.0)
    for i in range(2):
        path = os.path.join(tmp, "run_h_%04d.bin" % i)
        with open(path, "wb") as fh:
            fh.write(payload)
    man = {
        "run_id": "run",
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": 0.5,
        "fields": ["h"],
        "steps": [0, 1],
        "pattern": "run_{field}_{index:04d}.bin",
    }
    man_path = os.path.join(tmp, "run_manifest.json")
    with open(man_path, "w") as fh:
        json.dump(man, fh)
    written = xdmfgen.write_manifest_xdmf(man_path)
    assert len(written) == 1
    text = open(written[0]).read()
    assert "2DCoRectMesh" in text
    assert "run_h_0000.bin" in text
    assert "run_h_0001.bin" in text
    assert 'Name="h"' in text
    assert "Format=\"Binary\"" in text


if __name__ == "__main__":
    test_manifest_xdmf_points_at_bricks()
    print("ok")

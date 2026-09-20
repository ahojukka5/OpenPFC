#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sample one periodic trilinear initial field on a factor-two finer grid."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def refine_two(field):
    """Tensor-product midpoint interpolation, including the periodic seam."""
    result = field
    for axis in range(3):
        shape = list(result.shape)
        shape[axis] *= 2
        refined = np.empty(shape, dtype=np.float64)
        even = [slice(None)] * 3
        odd = [slice(None)] * 3
        even[axis] = slice(0, None, 2)
        odd[axis] = slice(1, None, 2)
        refined[tuple(even)] = result
        refined[tuple(odd)] = 0.5 * (result + np.roll(result, -1, axis=axis))
        result = refined
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shape", nargs=3, type=int, required=True,
                        metavar=("NX", "NY", "NZ"))
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--levels", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    if min(args.shape) <= 0 or not np.isfinite(args.dx) or args.dx <= 0:
        parser.error("shape and spacing must be positive and finite")
    data = np.fromfile(str(args.input), dtype="<f8")
    if data.size != int(np.prod(args.shape)) or not np.isfinite(data).all():
        parser.error("input is not the declared finite Float64 brick")
    if data.min() < 0 or data.max() > 1:
        parser.error("initial design must lie in [0,1]")
    field = data.reshape(tuple(reversed(args.shape)))
    result = field
    for _ in range(args.levels):
        result = refine_two(result)
    factor = 2 ** args.levels
    assert np.array_equal(result[::factor, ::factor, ::factor], field)
    assert np.isclose(result.mean(), field.mean(), rtol=0, atol=1e-14)
    assert result.min() >= field.min() and result.max() <= field.max()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("output or provenance already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.astype("<f8").tofile(str(args.output))
    record = {
        "method": "periodic tensor-product piecewise-linear interpolation",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_shape_xyz": args.shape,
        "output_shape_xyz": [n * factor for n in args.shape],
        "input_dx": args.dx, "output_dx": args.dx / factor,
        "periodic_cell_extent": [n * args.dx for n in args.shape],
        "mean": float(result.mean()), "min": float(result.min()),
        "max": float(result.max()), "order": "x-fastest Float64 little endian",
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

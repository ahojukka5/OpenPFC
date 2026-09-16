#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Issue #13 weak-scaling ladder: grids, cells/GCD, FFT layout check.

Does not invent timings. Run --check on a login node; it is arithmetic.
Compatible with the LUMI login Python 3.6.

    python3 apps/tungsten/scripts/flagship_ladder.py --check
"""

import argparse
import sys

# nodes, N  (8 GCD / node). Issue #13 candidates; 1200^3 is kept because
# OPENPFC_FFT_NODE_GRID=1 is 1x8xnnodes (1200 % 8 == 0 and 1200 % 4 == 0).
LADDER = (
    (1, 768),
    (2, 960),
    (4, 1200),
    (8, 1536),
    (16, 1920),
    (32, 2400),
    (60, 3000),
)


def is_5_smooth(n):
    x = n
    for p in (2, 3, 5):
        while x % p == 0:
            x //= p
    return x == 1


def node_aware_grid(n, nproc):
    if nproc < 9 or nproc % 8 != 0:
        return (0, 0, 0)
    nnodes = nproc // 8
    if n % 8 == 0 and n % nnodes == 0:
        return (1, 8, nnodes)
    if n % nnodes == 0 and n % 8 == 0:
        return (1, nnodes, 8)
    return (0, 0, 0)


def check():
    rc = 0
    print("nodes  gcds     N   cells/GCD   5-smooth  1x8xN grid")
    for nodes, n in LADDER:
        gcds = nodes * 8
        cells = n ** 3
        per = cells / float(gcds)
        grid = node_aware_grid(n, gcds) if gcds >= 9 else (1, 1, 1)
        ok_smooth = is_5_smooth(n)
        if gcds >= 9:
            ok_grid = grid != (0, 0, 0)
        else:
            ok_grid = n % 8 == 0
        mark = "ok" if ok_smooth and ok_grid else "FAIL"
        if mark != "ok":
            rc = 1
        print(
            "%5d %5d %5d %10.2e   %-8s %s  %s"
            % (nodes, gcds, n, per, "yes" if ok_smooth else "no", grid, mark)
        )
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.check:
        sys.exit(check())
    check()


if __name__ == "__main__":
    main()

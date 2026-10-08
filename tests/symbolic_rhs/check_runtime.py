#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent 80-digit decimal oracle against the compiled entry point."""
import decimal
import math
import random
import subprocess
import sys

decimal.getcontext().prec = 80
D = decimal.Decimal
rng = random.Random(353)
rows = [[D(rng.randint(-256,256))/16 for _ in range(16)] for _ in range(256)]
# Zero forcing explicitly verifies the bare homogeneous toy PDE.
for row in rows[:128]:
    row[4] = D(0)
# Perturb every input independently, including all three mixed derivatives.
base = [D(i+1)/16 for i in range(16)]
rows.append(base)
for i in range(16):
    row = list(base)
    row[i] += D(1)/1024
    rows.append(row)
request = "\n".join(" ".join(str(v) for v in row) for row in rows)+"\n"
result = subprocess.run([sys.argv[1], "--evaluate"], input=request,
                        text=True, capture_output=True, check=True)
records = []
for line in result.stdout.splitlines():
    # LUMI LibSci can print this loader diagnostic before numerical output.
    if line.startswith("[CRAYBLAS_WARNING]"):
        print(line, file=sys.stderr)
        continue
    records.append(line)
assert len(records) == len(rows), result.stdout
for row, record in zip(rows, records):
    t,x,y,z,rate,a,b,c,d,value,xx,yy,zz,xy,yz,xz = row
    force = rate*t*(x+y+z)
    expected = (value+a*xy+force, b*(xx+yy+zz)+c*xz+d*yz+force)
    actual = [float(v) for v in record.split()]
    assert len(actual) == 2
    for measured, exact in zip(actual, expected):
        assert math.isfinite(measured)
        assert abs(measured-float(exact)) <= 5e-15*max(1,abs(float(exact))), (row,measured,exact)
print(f"80-digit independent oracle: {len(rows)} cases, both outputs passed")

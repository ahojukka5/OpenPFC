#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail-closed common-band field gate for coalescence Stage 0 (#113).

Consumes one or more JSON reports written by compare_mhd_fields.py.
The gate is intentionally field-level: agreement of topology/scalar
diagnostics alone is not enough to admit the 128^2 -> 256^2 spatial step.
"""

from __future__ import annotations

import argparse
import json
import sys


# Frozen before any eta=0.005 run. Do not relax after seeing Stage-1 data.
FIELD_LIMITS = {
    "a": 1.0e-5,
    "j": 1.0e-3,
    "omega": 1.0e-3,
}
SPECTRUM_LIMITS = {
    "me_rel_l2": 1.0e-3,
    "j_rel_l2": 1.0e-3,
    "enstrophy_rel_l2": 1.0e-3,
}


def check_report(rep):
    failures = []
    fields = {f["field"]: f for f in rep.get("fields", [])}
    for name, limit in FIELD_LIMITS.items():
        if name not in fields:
            failures.append("missing field %s" % name)
            continue
        err = float(fields[name]["rel_l2_real"])
        if err > limit:
            failures.append("%s relL2 %.3e > %.3e" % (name, err, limit))
    spectra = rep.get("spectra", {})
    for key, limit in SPECTRUM_LIMITS.items():
        if key not in spectra:
            failures.append("missing spectrum %s" % key)
            continue
        err = float(spectra[key])
        if err > limit:
            failures.append("%s %.3e > %.3e" % (key, err, limit))
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json", nargs="+", help="compare_mhd_fields JSON reports")
    args = p.parse_args()

    bad = False
    for path in args.json:
        with open(path, encoding="utf-8") as fh:
            rep = json.load(fh)
        failures = check_report(rep)
        tag = "%s:%s vs %s:%s" % (
            rep.get("coarse_n"), rep.get("coarse_inc"),
            rep.get("fine_n"), rep.get("fine_inc"))
        if failures:
            bad = True
            print("FAIL", tag)
            for item in failures:
                print("  ", item)
        else:
            print("PASS", tag)
            for f in rep["fields"]:
                print("  %s relL2=%.3e" % (f["field"], f["rel_l2_real"]))
            sp = rep["spectra"]
            print("  spectra ME=%.3e j=%.3e Z=%.3e" % (
                sp["me_rel_l2"], sp["j_rel_l2"],
                sp["enstrophy_rel_l2"]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

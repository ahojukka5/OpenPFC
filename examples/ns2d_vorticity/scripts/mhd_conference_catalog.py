#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Accepted 2-D MHD run catalog for the conference visualization package.

Paths are not hardcoded in the figure drivers. Override the experiment
root with OPENPFC_MHD_DATA or --data-root. The numbers below are the
merged #27/#39 claims; scripts must not silently mix later peak-current
dumps into evidence assets.
"""

from __future__ import annotations

import csv
import json
import math
import os
from glob import glob

TWOPI = 2.0 * math.pi
PI80 = math.pi / 80.0

# #26 accepted reconnection window. Peak current at t=2.24 is outside.
T_RECONN_LO = 0.0
T_RECONN_HI = 0.80
# #38 frozen stage-2 scaling window. Do not move.
T_SCALE_LO = 0.314
T_SCALE_HI = 0.70

WINDOW_RECONNECTION = "accepted reconnection window"
WINDOW_SCALING = "scaling summary"
WINDOW_SHOWCASE = "late-time dynamics showcase"

# Frozen colour limits. Window max |j| on t<=0.80 across the three
# admitted runs is 6.136 (eta=0.0025 at t=0.785). Do not autoscale.
J_CLIM_EVIDENCE = 6.5
# Peak-current showcase only. Labelled on the figure; not evidence.
J_CLIM_SHOWCASE = 40.0

# Fixed flux contours. Analytic OT a is in [-1.5, 1.5]; do not relevel
# per frame.
A_LEVELS = tuple(-1.4 + 0.2 * i for i in range(15))

STILL_SHEET_T = 16.0 * PI80  # t = 0.628..., thin but still in stage 2
STILL_TOPOLOGY_T = 12.0 * PI80  # t = 0.471...

DEFAULT_DATA_ROOT = "/scratch/project_462001519/juaho/mhd2d-23"
DATA_ROOT_ENV = "OPENPFC_MHD_DATA"

# Printed #39 admitted means on [0.314, 0.70]. Fallback when run JSON
# is absent; JSON rows are preferred when the dumps are on disk.
ADMITTED_MEANS = (
    {
        "case_id": "ot256_eta01",
        "eta": 0.01,
        "N": 256,
        "R": 0.7736,
        "S_local": 66.75,
        "delta": 0.4554,
        "L": 3.549,
        "B_up": 0.1840,
        "j_X": 2.542,
        "RS": 43.72,
    },
    {
        "case_id": "ot512_eta0005",
        "eta": 0.005,
        "N": 512,
        "R": 0.3403,
        "S_local": 137.9,
        "delta": 0.4517,
        "L": 3.490,
        "B_up": 0.1971,
        "j_X": 2.648,
        "RS": 41.70,
    },
    {
        "case_id": "ot1024_eta00025",
        "eta": 0.0025,
        "N": 1024,
        "R": 0.1602,
        "S_local": 281.0,
        "delta": 0.4498,
        "L": 3.468,
        "B_up": 0.2042,
        "j_X": 2.704,
        "RS": 40.82,
    },
)

CASES = {
    "ot256_eta01": {
        "id": "ot256_eta01",
        "label": "Orszag–Tang",
        "subdir": "ot256_nu001_cfl04",
        "N": 256,
        "nu": 0.01,
        "eta": 0.01,
        "cfl": 0.4,
        "hero": False,
        "comparison": True,
        "spatial_convergence": "128 vs 256",
        "timestep_convergence": "CFL 0.4 vs 0.2 at 256",
        "issues": (38, 39),
        "notes": "Admitted eta=0.01 point. Not a peak-current run.",
    },
    "ot512_eta0005": {
        "id": "ot512_eta0005",
        "label": "Orszag–Tang",
        "subdir": "ot512_nu0005_peak",
        "N": 512,
        "nu": 0.005,
        "eta": 0.005,
        "cfl": 0.4,
        "hero": True,
        "comparison": True,
        "spatial_convergence": "256 vs 512",
        "timestep_convergence": "CFL 0.4 vs 0.2 at 256 (#26)",
        "issues": (23, 26, 38, 39),
        "notes": (
            "Hero and flux-budget case. Dumps continue to t=2.5; "
            "evidence assets must stop at t<=0.80."
        ),
    },
    "ot1024_eta00025": {
        "id": "ot1024_eta00025",
        "label": "Orszag–Tang",
        "subdir": "ot1024_nu00025_cfl02_t080",
        "N": 1024,
        "nu": 0.0025,
        "eta": 0.0025,
        "cfl": 0.2,
        "hero": False,
        "comparison": True,
        "spatial_convergence": "512 vs 1024",
        "timestep_convergence": (
            "geometry/F vs historical 512 CFL 0.4; Ez dump-cadence "
            "effect documented in SCALING.md"
        ),
        "issues": (38, 39),
        "notes": "Admitted eta=0.0025 spatial gate. t<=0.80 only.",
    },
}

HERO_ID = "ot512_eta0005"
COMPARISON_IDS = ("ot256_eta01", "ot512_eta0005", "ot1024_eta00025")
FLUX_ID = "ot512_eta0005"


def data_root(explicit=None):
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get(DATA_ROOT_ENV, "").strip()
    if env:
        return os.path.abspath(env)
    return DEFAULT_DATA_ROOT


def case_spec(case_id):
    if case_id not in CASES:
        raise KeyError("unknown MHD vis case %r" % (case_id,))
    return CASES[case_id]


def case_dir(case_id, root=None):
    spec = case_spec(case_id)
    return os.path.join(data_root(root), spec["subdir"])


def in_reconnection_window(t):
    return T_RECONN_LO - 1.0e-12 <= float(t) <= T_RECONN_HI + 1.0e-12


def in_scaling_window(t):
    return T_SCALE_LO - 1.0e-12 <= float(t) <= T_SCALE_HI + 1.0e-12


def time_key(t, ndigits=5):
    return round(float(t), ndigits)


def window_class_for_time(t, kind="evidence"):
    if kind == "showcase":
        return WINDOW_SHOWCASE
    if kind == "scaling":
        return WINDOW_SCALING
    return WINDOW_RECONNECTION


def provenance(case_id, t=None, kind="evidence", extra=None):
    spec = case_spec(case_id)
    rec = {
        "case": spec["label"],
        "case_id": spec["id"],
        "run_dir": spec["subdir"],
        "eta": spec["eta"],
        "nu": spec["nu"],
        "N": spec["N"],
        "cfl": spec["cfl"],
        "spatial_convergence": spec["spatial_convergence"],
        "timestep_convergence": spec["timestep_convergence"],
        "issues": list(spec["issues"]),
        "window": window_class_for_time(t, kind),
        "kind": kind,
    }
    if t is not None:
        rec["t"] = float(t)
    if extra:
        rec.update(extra)
    return rec


def banner_lines(case_id, t=None, kind="evidence"):
    spec = case_spec(case_id)
    window = window_class_for_time(t, kind)
    head = "%s   η = ν = %g   N = %d   %s" % (
        spec["label"], spec["eta"], spec["N"], window)
    if t is not None:
        return head, "t = %.3f" % float(t)
    return head, ""


def load_diagnostics(run_dir):
    path = os.path.join(run_dir, "diagnostics.csv")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for row in rows:
        step = int(float(row["step"]))
        out[step] = {
            "step": step,
            "t": float(row["time"]),
            "max_abs_j": float(row["max_abs_j"]),
        }
    return out


def list_dump_incs(run_dir, field="j"):
    paths = glob(os.path.join(run_dir, "%s_*.bin" % field))
    incs = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        incs.append(int(stem.split("_")[1]))
    return sorted(incs)


def dump_records(run_dir, t_min=None, t_max=None):
    diag = load_diagnostics(run_dir)
    recs = []
    for inc in list_dump_incs(run_dir):
        if inc not in diag:
            continue
        t = diag[inc]["t"]
        if t_min is not None and t < t_min - 1.0e-12:
            continue
        if t_max is not None and t > t_max + 1.0e-12:
            continue
        recs.append({
            "inc": inc,
            "t": t,
            "max_abs_j": diag[inc]["max_abs_j"],
            "j_path": os.path.join(run_dir, "j_%04d.bin" % inc),
            "a_path": os.path.join(run_dir, "a_%04d.bin" % inc),
        })
    recs.sort(key=lambda r: r["t"])
    return recs


def nearest_dump(recs, t):
    if not recs:
        raise ValueError("no dumps")
    return min(recs, key=lambda r: abs(r["t"] - float(t)))


def common_dump_times(case_ids, root=None, t_max=T_RECONN_HI):
    sets = []
    for cid in case_ids:
        recs = dump_records(case_dir(cid, root), t_min=0.0, t_max=t_max)
        sets.append({time_key(r["t"]) for r in recs})
    common = sorted(set.intersection(*sets)) if sets else []
    return common


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def sheet_json_path(run_dir):
    for name in ("sheet_scaling.json", "island_flux.json"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    return None


def load_tracked_rows(run_dir):
    path = sheet_json_path(run_dir)
    if path is None:
        return []
    data = load_json(path)
    return list(data.get("rows") or [])


def nearest_row(rows, t):
    if not rows:
        return None
    return min(rows, key=lambda r: abs(float(r["t"]) - float(t)))


def window_rows(rows, lo, hi):
    return [r for r in rows if lo - 1.0e-12 <= float(r["t"]) <= hi + 1.0e-12]


def mean_of(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def admitted_means_from_json(root=None):
    """Recompute frozen-window means from run JSON when present."""
    out = []
    for printed in ADMITTED_MEANS:
        cid = printed["case_id"]
        run = case_dir(cid, root)
        path = sheet_json_path(run)
        rec = dict(printed)
        rec["source"] = "SCALING.md printed table"
        if path and os.path.isdir(run):
            rows = window_rows(load_tracked_rows(run), T_SCALE_LO, T_SCALE_HI)
            if rows:
                rec["source"] = path
                rec["n_dumps"] = len(rows)
                rec["R"] = mean_of(rows, "R")
                rec["S_local"] = mean_of(rows, "S_local")
                rec["delta"] = mean_of(rows, "delta")
                rec["L"] = mean_of(rows, "L")
                rec["B_up"] = mean_of(rows, "B_up")
                rec["j_X"] = mean_of(rows, "j_X")
                rs = [float(r["R"]) * float(r["S_local"]) for r in rows
                      if r.get("R") is not None and r.get("S_local") is not None]
                rec["RS"] = sum(rs) / float(len(rs)) if rs else rec["RS"]
        rec["L_over_delta"] = rec["L"] / rec["delta"]
        out.append(rec)
    return out


def loglog_ols(x, y):
    """Return slope, intercept, formal OLS SE of log y vs log x.

    The SE is the ordinary 1-dof (n-2) standard error. It is not a
    physical uncertainty interval.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        raise ValueError("OLS slope requires at least three points")
    X = np.log(x)
    Y = np.log(y)
    xbar = float(X.mean())
    ybar = float(Y.mean())
    sxx = float(np.sum((X - xbar) ** 2))
    slope = float(np.sum((X - xbar) * (Y - ybar)) / sxx)
    intercept = ybar - slope * xbar
    pred = intercept + slope * X
    sse = float(np.sum((Y - pred) ** 2))
    dof = x.size - 2
    se = math.sqrt(sse / float(dof) / sxx)
    return slope, intercept, se


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path

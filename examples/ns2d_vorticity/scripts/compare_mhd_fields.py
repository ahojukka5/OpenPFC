#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Common-band field comparison for two mhd2d BinaryWriter dumps.

Loads Fortran-order double bricks (Nx x Ny x 1), takes an r2c FFT along x
and a full FFT along y, restricts the finer hat onto the coarser integer
wave-number lattice (HeFFTe-style 1/N^2 scaling), and reports relative L2
errors plus 1-D shell spectra.

Example:
  python3 compare_mhd_fields.py \\
    --coarse-dir /scratch/.../ot256_nu0005 --coarse-n 256 --coarse-inc 153 \\
    --fine-dir   /scratch/.../ot512_nu0005 --fine-n 512 --fine-inc 306
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np


def load_brick(path: str, n: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == n * n:
        return raw.reshape((n, n), order="F")
    if raw.size == n * n * 1:
        return raw.reshape((n, n, 1), order="F")[:, :, 0]
    raise ValueError(f"{path}: got {raw.size} doubles, expected {n*n}")


def rfft_xy(field: np.ndarray) -> np.ndarray:
    # r2c along x while the array is still real, then a full FFT along y.
    return np.fft.fft(np.fft.rfft(field, axis=0), axis=1)


def irfft_xy(hat: np.ndarray, n: int) -> np.ndarray:
    return np.fft.irfft(np.fft.ifft(hat, axis=1), n=n, axis=0).real


def restrict_hat(fine_hat: np.ndarray, n_coarse: int) -> np.ndarray:
    n_fine = fine_hat.shape[1]
    nkx = n_coarse // 2 + 1
    out = np.zeros((nkx, n_coarse), dtype=np.complex128)
    scale = (n_coarse / n_fine) ** 2
    for j in range(n_coarse):
        kj = j if j <= n_coarse // 2 else j - n_coarse
        jf = kj if kj >= 0 else n_fine + kj
        out[:, j] = fine_hat[:nkx, jf] * scale
    return out


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(b)
    if den == 0.0:
        return float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / den)


def shell_spectrum(hat: np.ndarray, n: int, weight: np.ndarray) -> np.ndarray:
    """weight is per-mode multiplier on |hat|^2 (already r2c)."""
    nkx, ny = hat.shape
    kmax = n // 2
    spec = np.zeros(kmax + 1)
    for j in range(ny):
        ky = j if j <= n // 2 else j - n
        for i in range(nkx):
            kx = i
            k = int(round(math.hypot(kx, ky)))
            if k <= kmax:
                spec[k] += weight[i, j] * (np.abs(hat[i, j]) ** 2)
    return spec


def k_arrays(n: int) -> tuple[np.ndarray, np.ndarray]:
    nkx = n // 2 + 1
    kx = np.arange(nkx)[:, None]
    ky = np.array([j if j <= n // 2 else j - n for j in range(n)])[None, :]
    return kx, ky


def field_path(directory: str, name: str, inc: int) -> str:
    return os.path.join(directory, f"{name}_{inc:04d}.bin")


def compare_one(name: str, coarse: np.ndarray, fine: np.ndarray) -> dict:
    n_c = coarse.shape[0]
    hat_c = rfft_xy(coarse)
    hat_f = restrict_hat(rfft_xy(fine), n_c)
    rec = irfft_xy(hat_f, n_c)
    return {
        "field": name,
        "rel_l2_real": rel_l2(rec, coarse),
        "rel_l2_hat": rel_l2(hat_f, hat_c),
        "coarse_linf": float(np.max(np.abs(coarse))),
        "fine_restricted_linf": float(np.max(np.abs(rec))),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coarse-dir", required=True)
    p.add_argument("--fine-dir", required=True)
    p.add_argument("--coarse-n", type=int, required=True)
    p.add_argument("--fine-n", type=int, required=True)
    p.add_argument("--coarse-inc", type=int, required=True)
    p.add_argument("--fine-inc", type=int, required=True)
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    report = {
        "coarse_dir": args.coarse_dir,
        "fine_dir": args.fine_dir,
        "coarse_n": args.coarse_n,
        "fine_n": args.fine_n,
        "coarse_inc": args.coarse_inc,
        "fine_inc": args.fine_inc,
        "fields": [],
        "spectra": {},
    }
    for name in ("a", "j", "omega"):
        c = load_brick(field_path(args.coarse_dir, name, args.coarse_inc), args.coarse_n)
        f = load_brick(field_path(args.fine_dir, name, args.fine_inc), args.fine_n)
        stats = compare_one(name, c, f)
        report["fields"].append(stats)
        print(
            f"{name:6s}  relL2_real={stats['rel_l2_real']:.4e}  "
            f"relL2_hat={stats['rel_l2_hat']:.4e}  "
            f"linf_coarse={stats['coarse_linf']:.6g}"
        )

    a_c = load_brick(field_path(args.coarse_dir, "a", args.coarse_inc), args.coarse_n)
    a_f = load_brick(field_path(args.fine_dir, "a", args.fine_inc), args.fine_n)
    j_c = load_brick(field_path(args.coarse_dir, "j", args.coarse_inc), args.coarse_n)
    w_c = load_brick(field_path(args.coarse_dir, "omega", args.coarse_inc), args.coarse_n)
    n = args.coarse_n
    kx, ky = k_arrays(n)
    k2 = kx * kx + ky * ky
    # Unnormalized hats; divide by N^4 so shells are comparable to 0.5<|B|^2>.
    scale = 1.0 / (n ** 4)
    r2c_w = np.where((kx == 0) | (kx == n // 2), 1.0, 2.0)
    r2c_w = np.broadcast_to(r2c_w, k2.shape).copy()
    hat_a_c = rfft_xy(a_c)
    hat_a_f = restrict_hat(rfft_xy(a_f), n)
    me_c = shell_spectrum(hat_a_c, n, 0.5 * k2 * r2c_w * scale)
    me_f = shell_spectrum(hat_a_f, n, 0.5 * k2 * r2c_w * scale)
    hat_j_c = rfft_xy(j_c)
    hat_j_f = restrict_hat(rfft_xy(load_brick(field_path(args.fine_dir, "j", args.fine_inc), args.fine_n)), n)
    jspec_c = shell_spectrum(hat_j_c, n, r2c_w * scale)
    jspec_f = shell_spectrum(hat_j_f, n, r2c_w * scale)
    hat_w_c = rfft_xy(w_c)
    hat_w_f = restrict_hat(
        rfft_xy(load_brick(field_path(args.fine_dir, "omega", args.fine_inc), args.fine_n)),
        n,
    )
    zspec_c = shell_spectrum(hat_w_c, n, r2c_w * scale)
    zspec_f = shell_spectrum(hat_w_f, n, r2c_w * scale)

    def spec_err(a, b):
        return rel_l2(a, b)

    report["spectra"] = {
        "me_rel_l2": spec_err(me_f, me_c),
        "j_rel_l2": spec_err(jspec_f, jspec_c),
        "enstrophy_rel_l2": spec_err(zspec_f, zspec_c),
        "me_coarse": me_c.tolist(),
        "me_fine_restricted": me_f.tolist(),
        "j_coarse": jspec_c.tolist(),
        "j_fine_restricted": jspec_f.tolist(),
    }
    print(
        f"spectra me_relL2={report['spectra']['me_rel_l2']:.4e}  "
        f"j_relL2={report['spectra']['j_rel_l2']:.4e}  "
        f"Z_relL2={report['spectra']['enstrophy_rel_l2']:.4e}"
    )

    jabs = np.abs(j_c)
    loc = np.unravel_index(np.argmax(jabs), jabs.shape)

    def periodic_fwhm(arr: np.ndarray, ipeak: int) -> int:
        n = arr.size
        half = 0.5 * arr[ipeak]
        left = 0
        while left < n and arr[(ipeak - left - 1) % n] >= half:
            left += 1
        right = 0
        while right < n and arr[(ipeak + right + 1) % n] >= half:
            right += 1
        return int(min(n, left + right + 1))

    fwhm_x = periodic_fwhm(jabs[:, loc[1]], loc[0])
    fwhm_y = periodic_fwhm(jabs[loc[0], :], loc[1])
    report["sheet"] = {
        "argmax_i": int(loc[0]),
        "argmax_j": int(loc[1]),
        "max_abs_j": float(jabs[loc]),
        "fwhm_cells_x": fwhm_x,
        "fwhm_cells_y": fwhm_y,
        "fwhm_cells_min": int(min(fwhm_x, fwhm_y)),
    }
    print(
        f"sheet argmax=({loc[0]},{loc[1]}) max|j|={jabs[loc]:.6g} "
        f"FWHM_x={fwhm_x} FWHM_y={fwhm_y} FWHM_min={min(fwhm_x, fwhm_y)}"
    )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

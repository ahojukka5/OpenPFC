#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001245
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --job-name=inv3d-restart-72
#SBATCH --output=/scratch/project_462001245/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001245/juaho/logs/%x-%j.err
#
# OpenPFC #72: Catch2 checkpoint tests plus HIP restart equivalence.
# Run 8 frozen-problem steps continuously, then 4+4 via --restart, and
# require overlapping iterate rows to match.

set -euo pipefail

SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/wt/openpfc-inv-ckpt}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-inv-ckpt72}"
OUT="${OUT_DIR:-/scratch/project_462001245/juaho/inverse-homogenization/restart_${SLURM_JOB_ID}}"
CT="${SRC}/apps/inverse_homogenization/evidence/2026-09-19-yang-a3-oracle-ev002.txt"
BIN="${BUILD}/apps/inverse_homogenization/openpfc_inverse_homogenize_hip"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export MPICH_GPU_SUPPORT_ENABLED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true
export CC=cc CXX=CC

mkdir -p "${OUT}/full/fields" "${OUT}/part/fields" /scratch/project_462001245/juaho/logs
{
  echo "job ${SLURM_JOB_ID} partition ${SLURM_JOB_PARTITION} host $(hostname)"
  echo "date $(date -Iseconds)"
  echo "account ${SLURM_JOB_ACCOUNT}"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
} | tee "${OUT}/provenance.txt"

if [[ ! -f "${BUILD}/CMakeCache.txt" ]]; then
  cmake -S "${SRC}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DHeffte_DIR=/users/juaho/opt/heffte/2.4.1-rocm/lib64/cmake/Heffte \
    -DOpenPFC_ENABLE_HIP=ON \
    -DOpenPFC_ENABLE_HIP_SPECTRAL=ON \
    -DOpenPFC_BUILD_TESTS=ON \
    -DOpenPFC_BUILD_APPS=ON \
    -DOpenPFC_BUILD_EXAMPLES=OFF
fi
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" \
  --target openpfc_inverse_homogenize_hip test_inverse_convergence

srun --ntasks=1 --gpus-per-node=1 \
  "${BUILD}/apps/inverse_homogenization/test_inverse_convergence"

COMMON=(
  --nx=32 --ny=32 --nz=61 --dx=1
  --E-solid=1.0 --nu-solid=0.3 --E-void=0.002 --nu-void=0.3
  --target=file --C-target-file="${CT}"
  --volume=0.2575865186 --init-volume=0.2575865186
  --normalize=1 --dt=0.04 --max-delta=0.04 --project-volume=1
  --lambda-reg=0.05 --lambda-reg-end=0.2 --simp=1 --simp-end=2
  --n-el-iter=200
  --init=spinodal --seed=1
  --continuation-steps=4 --conv-window=20
  --verify-convergence-steps=100 --dump-every=8
)

echo "=== continuous 8 steps ==="
srun --ntasks=8 --gpus-per-node=8 "${BIN}" \
  "${COMMON[@]}" --max-steps=8 \
  --run-id="inv3d_restart_full" \
  --csv="${OUT}/full/history.csv" \
  --dump-dir="${OUT}/full/fields" \
  --checkpoint-dir="${OUT}/full/checkpoint" \
  | tee "${OUT}/full/run.log"

echo "=== first 4 steps (walltime simulator) ==="
srun --ntasks=8 --gpus-per-node=8 "${BIN}" \
  "${COMMON[@]}" --max-steps=8 --stop-after=4 \
  --run-id="inv3d_restart_part" \
  --csv="${OUT}/part/history.csv" \
  --dump-dir="${OUT}/part/fields" \
  --checkpoint-dir="${OUT}/part/checkpoint" \
  | tee "${OUT}/part/run.log"

echo "=== restart steps 4-7 ==="
srun --ntasks=8 --gpus-per-node=8 "${BIN}" \
  "${COMMON[@]}" --max-steps=8 \
  --run-id="inv3d_restart_part" \
  --csv="${OUT}/part/history.csv" \
  --dump-dir="${OUT}/part/fields" \
  --checkpoint-dir="${OUT}/part/checkpoint" \
  --restart="${OUT}/part/checkpoint" \
  | tee "${OUT}/part/restart.log"

python3 - "${OUT}/full/history.csv" "${OUT}/part/history.csv" "${OUT}/compare.txt" <<'PY'
import csv, sys
full_path, part_path, out_path = sys.argv[1:4]

def rows(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            if line.startswith("step,"):
                continue
            r = next(csv.reader([line]))
            out[int(r[0])] = r
    return out

full, part = rows(full_path), rows(part_path)
need = [4, 5, 6, 7]
missing = [s for s in need if s not in full or s not in part]
if missing:
    raise SystemExit(f"missing overlapping steps {missing}")

def rel(a, b):
    a, b = float(a), float(b)
    den = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / den

cols = [(1, "J", 1e-10), (7, "C11", 1e-10), (11, "design_rms", 1e-12)]
fail = []
lines = ["step col rel"]
for s in need:
    for idx, name, tol in cols:
        e = rel(full[s][idx], part[s][idx])
        lines.append(f"{s} {name} {e:.3e}")
        if e > tol:
            fail.append(f"step {s} {name} rel={e:.3e} full={full[s][idx]} part={part[s][idx]}")
text = "\n".join(lines) + "\n"
open(out_path, "w").write(text)
print(text, end="")
if fail:
    raise SystemExit("restart mismatch:\n" + "\n".join(fail))
print("RESTART_EQUIVALENCE_OK")
PY

echo "done job ${SLURM_JOB_ID}" | tee -a "${OUT}/provenance.txt"

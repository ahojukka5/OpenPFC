#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=06:00:00
#SBATCH --job-name=weibel-show
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Issue #10: 1D2V Weibel on one LUMI-G node. Reduced frames only (no full f).
# Linear-validation arm (same grid/physics) first; the movie may saturate.
# Quote gamma only from the frozen linear window. SHOWCASE_MODE=pilot or prod.
# Movie dumps >=600 reduced frames (10 s at 60 fps); same t_end/k/vmax.

set -euo pipefail

SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-showcase-vm}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/openpfc-showcase/weibel_${SLURM_JOB_ID}}"
MODE="${SHOWCASE_MODE:-pilot}"

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

mkdir -p "${OUT}/reduced"
{
  echo "job ${SLURM_JOB_ID} partition ${SLURM_JOB_PARTITION} host $(hostname)"
  echo "date $(date -Iseconds)"
  echo "src ${SRC}"
  echo "revision $(git -C "${SRC}" rev-parse HEAD)"
  echo "dirty $(git -C "${SRC}" status --porcelain | wc -l)"
  echo "mode ${MODE}"
  module list
} | tee "${OUT}/provenance.txt"

if [[ ! -f "${BUILD}/build.ninja" && ! -f "${BUILD}/Makefile" ]]; then
  echo "No HIP build tree at ${BUILD}. Configure on the login node with"
  echo "  ./scripts/build.sh --machine=lumi --partition=standard-g --no-test --no-submit --build-dir=${BUILD}"
  exit 2
fi
cmake --build "${BUILD}" -j 8 --target vlasov_run
BIN="${BUILD}/apps/vlasov_maxwell/vlasov_run"
test -x "${BIN}"

CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"

if [[ "${MODE}" == "prod" ]]; then
  NX="${SHOWCASE_NX:-2048}"
  NV="${SHOWCASE_NV:-256}"
  TEND="${SHOWCASE_TEND:-150}"
  SAMPLES="${SHOWCASE_SAMPLES:-600}"
  if [[ -z "${SHOWCASE_NX:-}" ]]; then
    echo "SHOWCASE_NX unset: candidate ${NX} (FFT-friendly). After the pilot, grow toward 4096x512 if the node is under-filled." | tee -a "${OUT}/provenance.txt"
  fi
else
  NX="${SHOWCASE_NX:-256}"
  NV="${SHOWCASE_NV:-64}"
  TEND="${SHOWCASE_TEND:-80}"
  SAMPLES="${SHOWCASE_SAMPLES:-600}"
fi
# Auto-window needs ~5x mode_bz rise. Job 22111474 at t_end=40 peaked at
# 4.39x (NaN fit). Same k/vmax; only a longer linear arm.
LINEAR_TEND="${SHOWCASE_LINEAR_TEND:-80}"
LINEAR_SAMPLES="${SHOWCASE_LINEAR_SAMPLES:-80}"

COMMON=(
  "${BIN}" --case=weibel --device=hip
  --nx="${NX}" --nvx="${NV}" --nvy="${NV}"
)

echo "=== linear arm (no reduced dumps) ===" | tee -a "${OUT}/provenance.txt"
srun --ntasks=8 --gpus-per-node=8 --cpu-bind="${CPU_BIND}" \
  "${COMMON[@]}" --t-end="${LINEAR_TEND}" --samples="${LINEAR_SAMPLES}" \
  --run-id=weibel-linear --summary="${OUT}/linear.csv" \
  --csv="${OUT}/linear_ledger.csv" \
  | tee "${OUT}/linear.log"

if [[ -n "${SHOWCASE_FIT_T0:-}" && -n "${SHOWCASE_FIT_T1:-}" ]]; then
  FIT_T0="${SHOWCASE_FIT_T0}"
  FIT_T1="${SHOWCASE_FIT_T1}"
  FIT_OK=1
else
  read -r FIT_T0 FIT_T1 _ FIT_OK <<EOF
$(python3 -c 'import csv,math,sys
rows=list(csv.DictReader(open(sys.argv[1])))
if not rows:
    raise SystemExit("linear.csv is empty")
r=rows[-1]

def finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False

ok = int(finite(r["fit_t0"]) and finite(r["fit_t1"]))
print(r["fit_t0"], r["fit_t1"], r["gamma_fit"], ok)
' "${OUT}/linear.csv")
EOF
fi
echo "frozen linear window fit_t0=${FIT_T0} fit_t1=${FIT_T1} fit_ok=${FIT_OK}" \
  | tee -a "${OUT}/provenance.txt"

if [[ "${FIT_OK}" != "1" ]]; then
  echo "skip movie: auto-window is not finite; raise SHOWCASE_LINEAR_TEND" \
    | tee -a "${OUT}/provenance.txt"
else
  echo "=== nonlinear movie ===" | tee -a "${OUT}/provenance.txt"
  srun --ntasks=8 --gpus-per-node=8 --cpu-bind="${CPU_BIND}" \
    "${COMMON[@]}" --t-end="${TEND}" --samples="${SAMPLES}" \
    --fit-t0="${FIT_T0}" --fit-t1="${FIT_T1}" \
    --run-id=weibel --summary="${OUT}/summary.csv" \
    --csv="${OUT}/ledger.csv" \
    --reduced-dir="${OUT}/reduced" --reduced-every=1 \
    | tee "${OUT}/run.log"
fi

echo "=== done ${OUT} ==="
echo "Render: python3 ${SRC}/apps/vlasov_maxwell/scripts/render_weibel_showcase.py --run ${OUT}"

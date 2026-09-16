#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# Re-measure the Stage-1 planar ladder of PR #147 on a *compute* node.
#
# Research admission purpose (research#313): the original README tables were
# produced on a LUMI login node and the raw per-case output was not retained in
# the research evidence ledger. This is therefore the ONE bounded baseline
# rerun allowed if those original artifacts cannot be recovered. It is not a
# new science campaign and must not be extended into a shear/domain sweep.
#
# The cases below are the existing Stage-1 resolution/order/anti-trapping
# protocol. They run concurrently on distinct cores of one exclusive node;
# wall time is not an observable. The output bundle is deliberately
# provenance-complete: exact source SHA, source dirty state, binary SHA256,
# Slurm layout, module list, per-case summaries, consolidated CSV, and reports.
#
# Required in the job environment:
#   ALLOY_PLANAR_BIN    absolute path to alloy_dendrite_planar
#   OPENPFC_SOURCE_DIR  source checkout from which the binary was built
# Optional:
#   ALLOY_OUT           output directory (default under /scratch)
#
# Before submission, rebuild ALLOY_PLANAR_BIN from a clean
# OPENPFC_SOURCE_DIR checkout. The script refuses a dirty checkout. The binary
# hash is archived so the executed artifact remains identifiable even if a
# build directory is later removed.

set -euo pipefail

: "${ALLOY_PLANAR_BIN:?set ALLOY_PLANAR_BIN to alloy_dendrite_planar}"
: "${OPENPFC_SOURCE_DIR:?set OPENPFC_SOURCE_DIR to the clean source checkout used for the build}"

test -x "${ALLOY_PLANAR_BIN}"
test -d "${OPENPFC_SOURCE_DIR}/.git"

SOURCE_REV="$(git -C "${OPENPFC_SOURCE_DIR}" rev-parse HEAD)"
SOURCE_STATUS="$(git -C "${OPENPFC_SOURCE_DIR}" status --porcelain)"
if [[ -n "${SOURCE_STATUS}" ]]; then
  echo "ERROR: OPENPFC_SOURCE_DIR is dirty; baseline evidence requires an immutable clean source revision" >&2
  git -C "${OPENPFC_SOURCE_DIR}" status --short >&2
  exit 2
fi

OUT="${ALLOY_OUT:-/scratch/project_462001519/juaho/alloy-dendrite/planar_${SLURM_JOB_ID}}"
mkdir -p "${OUT}"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SUM="${OUT}/planar_summary.csv"
PROV="${OUT}/provenance.txt"

{
  echo "protocol=research#313 bounded planar quantitative-alloy baseline"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_partition=${SLURM_JOB_PARTITION}"
  echo "slurm_nodes=${SLURM_JOB_NUM_NODES:-1}"
  echo "slurm_ntasks=${SLURM_NTASKS:-32}"
  echo "source_dir=${OPENPFC_SOURCE_DIR}"
  echo "source_revision=${SOURCE_REV}"
  echo "source_dirty=no"
  echo "binary=${ALLOY_PLANAR_BIN}"
  echo "binary_sha256=$(sha256sum "${ALLOY_PLANAR_BIN}" | awk '{print $1}')"
  echo "hostname=$(hostname)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "--- modules ---"
  module -t list 2>&1 || true
  echo "--- binary dependencies ---"
  ldd "${ALLOY_PLANAR_BIN}" 2>&1 || true
} > "${PROV}"

# Preserve the exact protocol next to the numerical output.
cp "$0" "${OUT}/$(basename "$0")"

echo "=== alloy_dendrite planar baseline, job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} ==="
echo "source_revision=${SOURCE_REV}"
echo "binary_sha256=$(sha256sum "${ALLOY_PLANAR_BIN}" | awk '{print $1}')"
echo "out=${OUT}"

# One helper per case: a single-rank srun step pinned to its own core. Each
# writes its own summary CSV; they are consolidated after all steps complete,
# avoiding concurrent append races.
run_case() {
  local id="$1"; shift
  srun --exact --ntasks=1 --cpus-per-task=1 --mem=8G \
    "${ALLOY_PLANAR_BIN}" --summary="${OUT}/sum_${id}.csv" --run-id="${id}" "$@" \
    > "${OUT}/${id}.txt" 2>&1 &
}

# --- resolution ladder, 512 W0 box, t_end = 600, order 4 -------------------
run_case res_dx1.6 --nx=320  --dx=1.6 --velocity=0.1 --t-end=600
run_case res_dx1.2 --nx=427  --dx=1.2 --velocity=0.1 --t-end=600
run_case res_dx0.8 --nx=640  --dx=0.8 --velocity=0.1 --t-end=600
run_case res_dx0.6 --nx=854  --dx=0.6 --velocity=0.1 --t-end=600
run_case res_dx0.4 --nx=1280 --dx=0.4 --velocity=0.1 --t-end=600

# --- FD order at a fixed coarse dx ----------------------------------------
for o in 2 4 6 8; do
  run_case "order_${o}" --nx=427 --dx=1.2 --fd-order="${o}" --velocity=0.1 --t-end=600
done

# --- anti-trapping: k_eff against velocity, three models -------------------
# The V = 0.05 row needs a 1600 W0 box so the 40 W0 boundary layer has room.
for s in 1 0 m1; do
  case "${s}" in
    m1) sv=-1 ;;
    *)  sv="${s}" ;;
  esac
  run_case "at${s}_v0.05" --nx=2667 --dx=0.6 --velocity=0.05 --t-end=600 --at-scale="${sv}"
  run_case "at${s}_v0.10" --nx=854  --dx=0.6 --velocity=0.10 --t-end=600 --at-scale="${sv}"
  run_case "at${s}_v0.20" --nx=854  --dx=0.6 --velocity=0.20 --t-end=600 --at-scale="${sv}"
  run_case "at${s}_v0.40" --nx=854  --dx=0.6 --velocity=0.40 --t-end=600 --at-scale="${sv}"
done

wait

# Fail closed: every declared case must have produced exactly one non-empty
# summary before anything is admitted.
expected=$((5 + 4 + 3 * 4))
actual=$(find "${OUT}" -maxdepth 1 -name 'sum_*.csv' -type f -size +0c | wc -l)
if [[ "${actual}" -ne "${expected}" ]]; then
  echo "ERROR: expected ${expected} per-case summaries, found ${actual}" >&2
  exit 3
fi

# Deterministic machine-readable bundle. Header once, then lexicographic run_id
# order; run_id itself preserves which protocol arm each row belongs to.
first="$(find "${OUT}" -maxdepth 1 -name 'sum_*.csv' -type f | sort | head -1)"
head -1 "${first}" > "${SUM}"
while IFS= read -r f; do
  tail -n +2 "${f}" >> "${SUM}"
done < <(find "${OUT}" -maxdepth 1 -name 'sum_*.csv' -type f | sort)

{
  echo "source_revision=${SOURCE_REV}"
  echo "binary_sha256=$(sha256sum "${ALLOY_PLANAR_BIN}" | awk '{print $1}')"
  echo "summary_sha256=$(sha256sum "${SUM}" | awk '{print $1}')"
} >> "${PROV}"

echo "=== consolidated summary ==="
cat "${SUM}"
echo "=== per-case reports ==="
for f in "${OUT}"/*.txt; do
  echo "--- ${f} ---"
  cat "${f}"
done
echo "=== provenance ==="
cat "${PROV}"
echo "=== done ==="

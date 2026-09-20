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
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001245/juaho/logs/%x-%j.out
#
# Issue #106: randomized-order HeFFTe reshape comparison on one allocation.
# Submit with docs/lumi_slurm/submit_heffte_topology_crossover.sh.
#
# Fail-closed: refuse any billed account other than project_462001245.
# Drop leaked HEAT3D_* / OPENPFC_FFT_* knobs, then set only
# HEAT3D_RESHAPE_ALG for the current protocol.

set -euo pipefail

ALLOWED_ACCOUNTS="project_462001245"
ALLOWED_ALGS="p2p_plined p2p alltoallv alltoall"

: "${HEAT3D_SPECTRAL_HIP_BIN:?set HEAT3D_SPECTRAL_HIP_BIN to heat3d_spectral_hip}"
: "${HEAT3D_NX:?}"
: "${HEAT3D_NY:?}"
: "${HEAT3D_NZ:?}"

if [[ ! -x "${HEAT3D_SPECTRAL_HIP_BIN}" ]]; then
  echo "HEAT3D_SPECTRAL_HIP_BIN is not executable: ${HEAT3D_SPECTRAL_HIP_BIN}" >&2
  exit 1
fi

ACCOUNT_NOW="${SLURM_JOB_ACCOUNT:-}"
account_ok=0
for a in ${ALLOWED_ACCOUNTS}; do
  if [[ "${ACCOUNT_NOW}" == "${a}" ]]; then
    account_ok=1
    break
  fi
done
if [[ "${account_ok}" -ne 1 ]]; then
  echo "refusing #106 job billed to ${ACCOUNT_NOW:-missing} (allowed: ${ALLOWED_ACCOUNTS})" >&2
  exit 2
fi

drop_leaked_knobs() {
  unset HEAT3D_USE_PENCILS HEAT3D_RESHAPE_ALG HEAT3D_USE_REORDER \
    HEAT3D_GPU_AWARE HEAT3D_DIAG_TIMING HEAT3D_KEEP_OVERRIDES \
    OPENPFC_FFT_PROC_GRID OPENPFC_FFT_NODE_GRID OPENPFC_FFT_SLAB_AXIS \
    OPENPFC_FFT_COMPLEX_PROC_GRID OPENPFC_FFT_COMPLEX_OUTBOX || true
}

drop_leaked_knobs

NX="${HEAT3D_NX}"
NY="${HEAT3D_NY}"
NZ="${HEAT3D_NZ}"
STEPS="${HEAT3D_STEPS:-20}"
DT="${HEAT3D_DT:-0.01}"
WARMUP="${HEAT3D_WARMUP:-1}"
PROTOCOLS="${HEAT3D_PROTOCOLS:-p2p_plined:p2p:alltoallv:alltoall}"
PROTOCOLS="${PROTOCOLS//,/:}"
PROTOCOL_SEED="${HEAT3D_PROTOCOL_SEED:-unset}"
REPEAT="${OPENPFC_REPEAT:-unset}"
JOBROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001245/juaho/openpfc-scaling/heffte-topology-106}"
FAMILY="${OPENPFC_FAMILY:-768}"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true

NTASKS="${SLURM_NTASKS:?SLURM_NTASKS is not set}"
RUNDIR="${JOBROOT}/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${RUNDIR}"
cd "${RUNDIR}"
cp -f "${BASH_SOURCE[0]}" "${RUNDIR}/heffte_topology_crossover.sbatch"

PER_NODE="${SLURM_NTASKS_PER_NODE:-${NTASKS}}"
SRUN_BIND=()
CPU_BIND="none"
if (( PER_NODE == 8 )) && [[ "${SLURM_JOB_PARTITION}" == "standard-g" ]]; then
  CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"
  SRUN_BIND=(--cpu-bind="${CPU_BIND}")
elif (( PER_NODE > 8 )); then
  echo "at most 8 ranks per GCD; ntasks-per-node=${PER_NODE}" >&2
  exit 1
fi

collect_placement() {
  local dest="${RUNDIR}/placement"
  mkdir -p "${dest}"
  {
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "slurm_job_account=${ACCOUNT_NOW}"
    echo "slurm_job_partition=${SLURM_JOB_PARTITION}"
    echo "slurm_nnodes=${SLURM_NNODES}"
    echo "slurm_ntasks=${NTASKS}"
    echo "slurm_node_list=${SLURM_JOB_NODELIST}"
    echo "slurm_job_qos=${SLURM_JOB_QOS:-unset}"
    echo "slurm_cluster=${SLURM_CLUSTER_NAME:-unset}"
    echo "mpich_gpu_support=${MPICH_GPU_SUPPORT_ENABLED}"
    echo "mpich_ofi_nic_policy=${MPICH_OFI_NIC_POLICY:-unset}"
  } > "${dest}/slurm_env.txt"
  scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${dest}/hostnames.txt" || true
  scontrol show job "${SLURM_JOB_ID}" > "${dest}/scontrol_job.txt" || true
  set +e
  srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpu-bind=none \
    /bin/bash -c '
      host=$(hostname -s 2>/dev/null || hostname)
      nid="na"; cname="na"; xname="na"; cxi="na"
      [[ -r /proc/cray_xt/nid ]] && nid=$(tr -d "[:space:]" < /proc/cray_xt/nid)
      [[ -r /proc/cray_xt/cname ]] && cname=$(tr -d "[:space:]" < /proc/cray_xt/cname)
      for f in /etc/xname /proc/cray_xt/xname; do
        if [[ -r "$f" ]]; then
          xname=$(tr -d "[:space:]" < "$f")
          break
        fi
      done
      if ls -d /sys/class/cxi/cxi* >/dev/null 2>&1; then
        cxi=$(ls /sys/class/cxi | tr "\n" ",")
      fi
      printf "host=%s nid=%s cname=%s xname=%s cxi=%s slurm_nodeid=%s\n" \
        "$host" "$nid" "$cname" "$xname" "$cxi" "${SLURM_NODEID:-na}"
    ' > "${dest}/node_facts.txt"
  srun "${SRUN_BIND[@]}" /bin/bash -c '
      printf "rank=%s host=%s localid=%s slurm_gtid=%s\n" \
        "${SLURM_PROCID:-na}" "$(hostname -s 2>/dev/null || hostname)" \
        "${SLURM_LOCALID:-na}" "${SLURM_GTID:-na}"
    ' > "${dest}/rank_map.txt"
  set -e
}

BIN_SHA=""
if command -v sha256sum >/dev/null 2>&1; then
  BIN_SHA="$(sha256sum "${HEAT3D_SPECTRAL_HIP_BIN}" | awk '{print $1}')"
fi
REV="${OPENPFC_REVISION:-}"
DIRTY="${OPENPFC_DIRTY:-}"
SRC="${OPENPFC_SRC:-}"
if [[ -z "${REV}" ]] && command -v git >/dev/null 2>&1 && [[ -n "${SRC}" ]]; then
  if git -C "${SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    REV="$(git -C "${SRC}" rev-parse HEAD 2>/dev/null || true)"
    DIRTY="$(git -C "${SRC}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  fi
fi

{
  echo "=== heffte topology crossover (issue #106) ==="
  echo "job=${SLURM_JOB_ID}"
  echo "name=${SLURM_JOB_NAME}"
  echo "account=${ACCOUNT_NOW:-unset}"
  echo "sbatch_account=${ACCOUNT_NOW:-unset}"
  echo "allowed_accounts=${ALLOWED_ACCOUNTS}"
  echo "nodes=${SLURM_NNODES}"
  echo "ntasks=${NTASKS}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "bin=${HEAT3D_SPECTRAL_HIP_BIN}"
  echo "bin_sha256=${BIN_SHA}"
  echo "Nx=${NX}"
  echo "Ny=${NY}"
  echo "Nz=${NZ}"
  echo "N=${NX}x${NY}x${NZ}"
  echo "steps=${STEPS}"
  echo "dt=${DT}"
  echo "warmup=${WARMUP}"
  echo "family=${FAMILY}"
  echo "repeat=${REPEAT}"
  echo "protocol_seed=${PROTOCOL_SEED}"
  echo "protocols=${PROTOCOLS}"
  echo "protocol_order=${PROTOCOLS}"
  echo "cpu-bind=${CPU_BIND}"
  echo "mem_per_node=${SLURM_MEM_PER_NODE:-unset}"
  echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
  echo "MPICH_OFI_NIC_POLICY=${MPICH_OFI_NIC_POLICY:-unset}"
  echo "revision=${REV}"
  echo "dirty=${DIRTY}"
  echo "src=${SRC}"
  echo "cwd=${RUNDIR}"
  echo "heffte_module=$(module -t list 2>&1 | grep -i heffte | tr '\n' ' ' || true)"
  echo "mpi_module=$(module -t list 2>&1 | grep -iE 'cray-mpich|cpeGNU' | tr '\n' ' ' || true)"
  module list
} | tee "${RUNDIR}/run_meta.txt"

collect_placement

alg_allowed() {
  local a="$1"
  local x
  for x in ${ALLOWED_ALGS}; do
    if [[ "${x}" == "${a}" ]]; then
      return 0
    fi
  done
  return 1
}

banner_field() {
  local log="$1"
  local key="$2"
  awk -v k="${key}" '
    $1 == "HEAT3D_SPECTRAL_HIP" && $0 !~ /CHECKSUM/ {
      for (i = 2; i <= NF; i++) {
        if ($i ~ ("^" k "=")) {
          split($i, a, "=")
          print a[2]
          exit
        }
      }
    }
  ' "${log}"
}

IFS=':' read -ra ALGS <<< "${PROTOCOLS}"
JOB_RC=0
INDEX=0
for ALG in "${ALGS[@]}"; do
  ALG="$(echo "${ALG}" | tr -d '[:space:]')"
  if [[ -z "${ALG}" ]]; then
    continue
  fi
  if ! alg_allowed "${ALG}"; then
    echo "unknown reshape algorithm: ${ALG}" >&2
    JOB_RC=1
    continue
  fi

  PROTODIR="${RUNDIR}/${ALG}"
  mkdir -p "${PROTODIR}"
  drop_leaked_knobs
  export HEAT3D_RESHAPE_ALG="${ALG}"
  export HEAT3D_PROFILE_JSON="${PROTODIR}/timing_profile.json"
  export HEAT3D_WARMUP="${WARMUP}"

  {
    echo "=== protocol ${ALG} index=${INDEX} seed=${PROTOCOL_SEED} ==="
    echo "HEAT3D_RESHAPE_ALG=${HEAT3D_RESHAPE_ALG}"
    echo "HEAT3D_USE_PENCILS=${HEAT3D_USE_PENCILS:-unset}"
    echo "HEAT3D_USE_REORDER=${HEAT3D_USE_REORDER:-unset}"
    echo "HEAT3D_GPU_AWARE=${HEAT3D_GPU_AWARE:-unset}"
    echo "OPENPFC_FFT_PROC_GRID=${OPENPFC_FFT_PROC_GRID:-unset}"
    echo "OPENPFC_FFT_NODE_GRID=${OPENPFC_FFT_NODE_GRID:-unset}"
  } | tee "${PROTODIR}/env.txt"

  set +e
  srun --export=ALL "${SRUN_BIND[@]}" \
    "${HEAT3D_SPECTRAL_HIP_BIN}" "${NX}" "${NY}" "${NZ}" "${STEPS}" "${DT}" \
    2>&1 | tee "${PROTODIR}/run.log"
  SRUN_RC="${PIPESTATUS[0]}"
  set -e

  BANNER_ALG="$(banner_field "${PROTODIR}/run.log" reshape)"
  BANNER_PENCILS="$(banner_field "${PROTODIR}/run.log" use_pencils)"
  BANNER_REORDER="$(banner_field "${PROTODIR}/run.log" use_reorder)"
  BANNER_AWARE="$(banner_field "${PROTODIR}/run.log" gpu_aware)"
  ADMIT="ok"
  REASON=""
  if (( SRUN_RC != 0 )); then
    ADMIT="reject"
    REASON="srun_exit_${SRUN_RC}"
  elif [[ -z "${BANNER_ALG}" ]]; then
    ADMIT="reject"
    REASON="missing_reshape_banner"
  elif [[ "${BANNER_ALG}" != "${ALG}" ]]; then
    ADMIT="reject"
    REASON="banner_reshape_${BANNER_ALG}_expected_${ALG}"
  elif [[ "${BANNER_PENCILS}" != "0" ]]; then
    ADMIT="reject"
    REASON="banner_use_pencils_${BANNER_PENCILS}"
  elif [[ -n "${BANNER_REORDER}" && "${BANNER_REORDER}" != "1" ]]; then
    ADMIT="reject"
    REASON="banner_use_reorder_${BANNER_REORDER}"
  elif [[ "${BANNER_AWARE}" != "1" ]]; then
    ADMIT="reject"
    REASON="banner_gpu_aware_${BANNER_AWARE}"
  fi

  {
    echo "admit=${ADMIT}"
    echo "reason=${REASON:-none}"
    echo "srun_rc=${SRUN_RC}"
    echo "banner_reshape=${BANNER_ALG:-missing}"
    echo "banner_use_pencils=${BANNER_PENCILS:-missing}"
    echo "banner_use_reorder=${BANNER_REORDER:-missing}"
    echo "banner_gpu_aware=${BANNER_AWARE:-missing}"
    echo "expected_reshape=${ALG}"
    echo "protocol_index=${INDEX}"
    echo "protocol_seed=${PROTOCOL_SEED}"
  } | tee "${PROTODIR}/admit.txt"

  if [[ "${ADMIT}" != "ok" ]]; then
    echo "rejected ${ALG}: ${REASON}" >&2
    JOB_RC=1
  fi
  INDEX=$((INDEX + 1))
done

echo "=== topology crossover done rc=${JOB_RC} ==="
exit "${JOB_RC}"

#!/bin/bash
# Generic SLURM wrapper; resources are given on the sbatch command line by submit_all.sh.
#   sbatch [resources] experiments/job.sbatch <script in experiments/> [args...]
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
KIT=${PROPD_KIT:?PROPD_KIT not exported by submit_all.sh}
echo "[job] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-?} cmd=$*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
# Guard: SLURM can start a gres/gpu:N job with fewer GPUs visible (seen on a partially used node). The trainer
# cannot run on fewer, so requeue (up to 3 times, then hold) instead of hanging on Ray resource waits.
NG=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU " || true); WANT=${PROPD_GPUS:-0}
if [ "$WANT" -gt 0 ] && [ "$NG" -ne "$WANT" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "[job] ERROR: $NG GPUs visible, expected $WANT (restart count ${SLURM_RESTART_COUNT:-0}); requeueing"
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then scontrol requeue "$SLURM_JOB_ID"; else scontrol requeuehold "$SLURM_JOB_ID"; fi
  sleep 60; exit 1
fi
bash "$KIT/$1" "${@:2}"
echo "[job] $(date -Is) finished"

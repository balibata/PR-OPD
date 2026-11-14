#!/bin/bash
# Run the queue directly on THIS machine (no SLURM), one job after another, GPUS GPUs each.
# Use it inside an allocated node / interactive job, or on a machine without SLURM.
#
#   bash experiments/run_local.sh                                       # default settings / starts / arms / seeds
#   bash experiments/run_local.sh --settings 3b-ws --starts "100 140" --arms "grpo pr-opd" --seeds 1
#   # two allocations sharing one WORK directory (the two seeds in parallel):
#   bash experiments/run_local.sh --stage1 3b-alf --seeds 1             # allocation A
#   bash experiments/run_local.sh --stage1 3b-ws  --seeds 2             # allocation B
#   SMOKE=1 bash experiments/run_local.sh --settings 3b-ws              # smoke chain only
# Stage I of the settings in --stage1 (default: all selected settings) runs first. Then Stage II runs start as
# soon as their setting's Stage I is done; a Stage I that another allocation is running is waited for (polled
# every POLL_SEC=300 s, at most WAIT_HOURS=72 h). A failed run is logged and skipped; re-running continues.
set -uo pipefail
KIT=$(cd "$(dirname "$0")" && pwd); source "$KIT/config.sh"; source "$KIT/lib.sh"
SETTINGS=$SETTINGS_DEFAULT; SEEDS_SEL=$SEEDS; ARMS_SEL=$ARMS_DEFAULT; STARTS_SEL=""; STAGE1=""
while [ $# -gt 0 ]; do
  case "$1" in
    --settings) SETTINGS=$2; shift 2;; --seeds) SEEDS_SEL=$2; shift 2;; --arms) ARMS_SEL=$2; shift 2;;
    --starts) STARTS_SEL=$2; shift 2;; --stage1) STAGE1=$2; shift 2;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) die "unknown option $1";;
  esac
done
STAGE1=${STAGE1:-$SETTINGS}
export SMOKE=${SMOKE:-0}
[ "$SMOKE" = 1 ] && { ARMS_SEL=pr-opd; SEEDS_SEL=${SEEDS_SEL%% *}; STARTS_SEL=""; }
[ "$SMOKE" = 1 ] || check_protocol
mkdir -p "$WORK/slurm"
FAILED=()
run(){   # run <name> <script> args...   (console output -> $WORK/slurm/<name>-local.out)
  local name=$1; shift
  echo "[local] $(date -Is) start $name"
  if bash "$KIT/$1" "${@:2}" > "$WORK/slurm/$name-local.out" 2>&1; then echo "[local] $(date -Is) done  $name"; return 0; fi
  echo "[local] $(date -Is) FAILED $name (see $WORK/slurm/$name-local.out)"; FAILED+=("$name"); return 1
}

# ---- Stage I ----
for S in $STAGE1; do setting_vars "$S"; [ -f "$WORK/done/$TRUNK_NAME" ] || run "$TRUNK_NAME" grpo_trunk.sh "$S" || true; done

# ---- Stage II ----
PENDING=()
for SEED in $SEEDS_SEL; do for A in $ARMS_SEL; do for S in $SETTINGS; do
  setting_vars "$S"
  for P in $STARTS; do
    if [ -n "$STARTS_SEL" ]; then case " $STARTS_SEL " in *" $P "*) ;; *) continue;; esac; fi
    PENDING+=("$S|$P|$A|$SEED")
  done
done; done; done
deadline=$(( $(date +%s) + ${WAIT_HOURS:-72} * 3600 ))
while [ ${#PENDING[@]} -gt 0 ]; do
  REST=(); progressed=0
  for item in "${PENDING[@]}"; do
    IFS='|' read -r S P A SEED <<< "$item"
    setting_vars "$S"
    NAME=$(run_name "$S" "$P" "$A" "$SEED"); [ "$SMOKE" = 1 ] && NAME=smoke-$NAME
    [ -f "$WORK/done/$NAME" ] && continue
    if [ -f "$WORK/done/$TRUNK_NAME" ]; then run "$NAME" run_arm.sh "$S" "$P" "$A" "$SEED" || true; progressed=1; continue; fi
    case " $STAGE1 " in *" $S "*) echo "[local] skip $NAME (Stage I $TRUNK_NAME failed here)"; continue;; esac
    REST+=("$item")     # Stage I of $S belongs to another allocation: wait for it
  done
  PENDING=("${REST[@]+"${REST[@]}"}")
  [ ${#PENDING[@]} -eq 0 ] && break
  if [ "$progressed" = 0 ]; then
    [ "$(date +%s)" -lt "$deadline" ] || { echo "[local] gave up waiting for Stage I of: ${PENDING[*]}"; FAILED+=("${PENDING[@]}"); break; }
    echo "[local] $(date -Is) waiting for Stage I (${#PENDING[@]} runs pending)"; sleep "${POLL_SEC:-300}"
  fi
done
if [ ${#FAILED[@]} -gt 0 ]; then echo "[local] failed: ${FAILED[*]}"; exit 1; fi
echo "[local] all selected runs done"

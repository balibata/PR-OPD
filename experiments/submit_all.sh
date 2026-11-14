#!/bin/bash
# Queue the experiments on SLURM. Per setting: one Stage I job (GRPO trunk from the base model + evaluation of
# every start point), then one job per (start, arm, seed) that waits for it (afterok).
# At most LANES jobs run at once: jobs are named propd-lane<i> and carry --dependency=singleton; the run name is
# in the job comment (squeue -o %k). Finished runs ($WORK/done/<name>) and runs already queued are skipped, so the
# script is safe to re-run (also after failures).
#
#   bash experiments/submit_all.sh --dry-run                        # show what would be submitted
#   bash experiments/submit_all.sh                                  # default settings / starts / arms / seeds
#   bash experiments/submit_all.sh --settings "3b-ws" --starts "140" --arms "grpo pr-opd" --seeds "1"
#   SMOKE=1 bash experiments/submit_all.sh --settings 3b-ws         # smoke chain: 2 GRPO steps -> pr-opd 2 more steps + evals
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd); source "$KIT/config.sh"; source "$KIT/lib.sh"
SETTINGS=$SETTINGS_DEFAULT; SEEDS_SEL=$SEEDS; ARMS_SEL=$ARMS_DEFAULT; STARTS_SEL=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --settings) SETTINGS=$2; shift 2;; --seeds) SEEDS_SEL=$2; shift 2;; --arms) ARMS_SEL=$2; shift 2;;
    --starts) STARTS_SEL=$2; shift 2;; --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,13p' "$0"; exit 0;;
    *) die "unknown option $1";;
  esac
done
export PROPD_KIT=$KIT
mkdir -p "$WORK/slurm"
for n in 140 134 125 500; do
  [ -f "$WORK/data/v$n/text/test.parquet" ] || die "placeholder data missing ($WORK/data/v$n): run 'bash experiments/prep_data.sh' first"
done
[ "${SMOKE:-0}" = 1 ] && { ARMS_SEL=pr-opd; SEEDS_SEL=${SEEDS_SEL%% *}; STARTS_SEL=""; }
[ "${SMOKE:-0}" = 1 ] || check_protocol

# job id of a pending/running job whose comment is <name>
queued(){ { squeue -h -u "$USER" -o "%i %k" 2>/dev/null || true; } | awk -v n="$1" '$2==n{print $1; exit}'; }

LANE_I=0
sb(){   # sb <run name> <dependency job or ""> <script> args...   -> sets JOBID (no subshell: LANE_I must persist)
  local name=$1 dep=$2; shift 2
  local jname="propd-$name" deps=()
  if [ "$LANES" -gt 0 ]; then jname="propd-lane$(( LANE_I % LANES + 1 ))"; deps+=(singleton); LANE_I=$((LANE_I + 1)); fi
  [ -n "$dep" ] && deps+=("afterok:$dep")
  local args=(--job-name "$jname" --comment "$name" --nodes 1 --ntasks 1 --gres "$SB_GRES" --cpus-per-task "$SB_CPUS"
              --mem "$SB_MEM" --time "$SB_TIME" --output "$WORK/slurm/$name-%j.out"
              --export "ALL,PROPD_KIT=$KIT,SMOKE=${SMOKE:-0},PROPD_GPUS=$GPUS")
  [ -n "$SB_PARTITION" ] && args+=(--partition "$SB_PARTITION")
  [ -n "$SB_ACCOUNT" ] && args+=(--account "$SB_ACCOUNT")
  [ -n "$SB_QOS" ] && args+=(--qos "$SB_QOS")
  [ -n "$SB_EXCLUDE" ] && args+=(--exclude "$SB_EXCLUDE")
  [ ${#deps[@]} -gt 0 ] && args+=(--dependency "$(IFS=,; echo "${deps[*]}")")
  # shellcheck disable=SC2206
  [ -n "$SB_MAIL" ] && args+=($SB_MAIL)
  # shellcheck disable=SC2206
  [ -n "$SB_EXTRA" ] && args+=($SB_EXTRA)
  if [ "$DRY" = 1 ]; then echo "DRY sbatch ${args[*]} $KIT/job.sbatch $*" >&2; JOBID="DRY_$name"; return 0; fi
  JOBID=$(sbatch --parsable "${args[@]}" "$KIT/job.sbatch" "$@") || die "sbatch failed for $name"
  JOBID=${JOBID%%;*}
}

# ---- Stage I per setting ----
declare -A DEP
for S in $SETTINGS; do
  setting_vars "$S"
  DEP[$S]=""
  if [ -f "$WORK/done/$TRUNK_NAME" ]; then echo "[submit] $TRUNK_NAME done"; continue; fi
  J=$(queued "$TRUNK_NAME")
  if [ -n "$J" ]; then echo "[submit] $TRUNK_NAME already queued (job $J)"
  else sb "$TRUNK_NAME" "" grpo_trunk.sh "$S"; J=$JOBID; echo "[submit] $TRUNK_NAME -> job $J"; fi
  DEP[$S]=$J
done

# ---- Stage II: seed by seed, arm by arm, start by start, settings interleaved ----
n=0
for SEED in $SEEDS_SEL; do
  for A in $ARMS_SEL; do
    arm_vars "$A" >/dev/null
    for S in $SETTINGS; do
      setting_vars "$S"
      for P in $STARTS; do
        if [ -n "$STARTS_SEL" ]; then case " $STARTS_SEL " in *" $P "*) ;; *) continue;; esac; fi
        NAME=$(run_name "$S" "$P" "$A" "$SEED"); [ "${SMOKE:-0}" = 1 ] && NAME=smoke-$NAME
        if [ -f "$WORK/done/$NAME" ]; then echo "[submit] $NAME done, skip"; continue; fi
        if [ -n "$(queued "$NAME")" ]; then echo "[submit] $NAME already queued, skip"; continue; fi
        d=${DEP[$S]}; [[ "$d" == DRY_* ]] && d=1
        sb "$NAME" "$d" run_arm.sh "$S" "$P" "$A" "$SEED"; J=$JOBID
        echo "[submit] $NAME -> job $J${DEP[$S]:+ (after ${DEP[$S]})}"; n=$((n + 1))
      done
    done
  done
done
echo "[submit] $n Stage II job(s) submitted (settings: $SETTINGS; starts: ${STARTS_SEL:-all}; arms: $ARMS_SEL; seeds: $SEEDS_SEL; lanes=$LANES; dry=$DRY)"
echo "[submit] watch: squeue -u \$USER -o '%.9i %.14j %.40k %.8T %.10M %R'"

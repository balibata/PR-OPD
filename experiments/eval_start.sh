#!/bin/bash
# Evaluate one start point (trunk checkpoint global_step_<start>) of a setting once, on the same deterministic
# task sets as the Stage II runs. Called by grpo_trunk.sh.
#   bash experiments/eval_start.sh <setting> <start>
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd); source "$KIT/config.sh"; source "$KIT/lib.sh"
SETTING=${1:?usage: eval_start.sh <setting> <start>}; S=${2:?start step}
setting_vars "$SETTING"
NAME=$(start_name "$SETTING" "$S"); EV=$WORK/evals/$NAME; DONE=$WORK/done/$NAME; C=$(trunk_ckpt "$S")
mkdir -p "$EV" "$WORK/done"
[ -f "$DONE" ] && { echo "[eval_start] $NAME already done"; exit 0; }
[ -d "$C/actor" ] || die "start checkpoint missing: $C"
activate_env "$CONDA_ENV"; runtime_env; cd "$REPO"
UTIL=$(gpu_util "$ENV_KEY")
COMMON=(--ckpt "$C" --env "$ENV_KEY" --model "$BASE" "${NOTHINK_FLAG[@]}" --gpus "$GPUS" --micro "$MICRO"
        --lp_micro "$LP_MICRO" --tp "$TP" --gpu_util "$UTIL" --seed 1 --ray_cpus "$RAY_CPUS")
run_eval(){ local out=$1; shift; grep -qa 'val/success_rate' "$out" 2>/dev/null || "$@" --out "$out"; }
if [ "$ENV_KEY" = alfworld ]; then
  ALFWORLD_DET_EVAL=1 run_eval "$EV/seen140.log" bash scripts/eval.sh "${COMMON[@]}" --val_n 140 --data_dir "$WORK/data/v140/text"
  if [ "$EVAL_UNSEEN" = 1 ]; then
    ALFWORLD_DET_EVAL=1 run_eval "$EV/unseen134.log" bash scripts/eval.sh "${COMMON[@]}" --val_n 134 \
      --data_dir "$WORK/data/v134/text" --eval_split eval_out_of_distribution
  fi
else
  WEBSHOP_DET_EVAL=1 WEBSHOP_EVAL_OFFSET=0 run_eval "$EV/ws125.log" bash scripts/eval.sh "${COMMON[@]}" --val_n 125 \
    --data_dir "$WORK/data/v125/text"
  WEBSHOP_DET_EVAL=1 WEBSHOP_EVAL_OFFSET=0 WEBSHOP_EVAL_ADVANCE=1 run_eval "$EV/ws500.log" bash scripts/eval.sh "${COMMON[@]}" \
    --val_n 125 --data_dir "$WORK/data/v500/text"
fi
{ echo "name=$NAME"; echo "ckpt=$C"; echo "finished=$(date -Is)"; } > "$DONE"
echo "[eval_start] $NAME done"

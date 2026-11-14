#!/bin/bash
# Stage I of one setting: ONE GRPO run from the base model (env.seed GRPO_SEED) up to the last start step of the
# setting (ALFWorld 120, WebShop 140 by default). The checkpoints at every start step are protected from rotation
# and keep their optimizer state; afterwards each start point is evaluated once (eval_start.sh).
#
#   bash experiments/grpo_trunk.sh 3b-alf          # -> work/ckpts/grpo-3b-alf/global_step_{100,120}
#   bash experiments/grpo_trunk.sh 3b-ws           # -> work/ckpts/grpo-3b-ws/global_step_{100,120,140}
#   SMOKE=1 bash experiments/grpo_trunk.sh 3b-ws   # pipeline check: 2 GRPO steps -> smoke-grpo-3b-ws/global_step_2
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd); source "$KIT/config.sh"; source "$KIT/lib.sh"
SETTING=${1:?usage: grpo_trunk.sh <setting>}
setting_vars "$SETTING"
NAME=$TRUNK_NAME; STEPS=$TRUNK_STEPS; TF=$GRPO_TEST_FREQ; SAVE=$SAVE_FREQ
if [ "${SMOKE:-0}" = 1 ]; then TF=2; SAVE=2; fi
CK=$WORK/ckpts/$NAME; DONE=$WORK/done/$NAME
mkdir -p "$WORK/logs" "$WORK/done"
[ -f "$DONE" ] && { echo "[grpo_trunk] $NAME already done"; exit 0; }
for s in $STARTS; do [ $((s % SAVE)) -eq 0 ] || die "start step $s is not a multiple of SAVE_FREQ=$SAVE"; done

activate_env "$CONDA_ENV"; runtime_env; cd "$REPO"
UTIL=$(gpu_util "$ENV_KEY"); val_args "$ENV_KEY"
echo "[grpo_trunk] $NAME on $(hostname) GPUs=$(gpu_names) util=$UTIL micro=$MICRO steps=0->$STEPS keep=[$STARTS]"

LAST=$(cat "$CK/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [ "$LAST" -lt "$STEPS" ]; then
  if [ "$LAST" -gt 0 ]; then echo "[grpo_trunk] WARNING: resuming $NAME from step $LAST (env task sampling restarts)"; fi
  bash scripts/train.sh --method grpo --env "$ENV_KEY" --model "$BASE" "${NOTHINK_FLAG[@]}" \
    --name "$NAME" --steps "$STEPS" --gpus "$GPUS" --micro "$MICRO" --lp_micro "$LP_MICRO" --tp "$TP" \
    --gpu_util "$UTIL" --seed "$GRPO_SEED" --save_freq "$SAVE" --test_freq "$TF" --keep_steps "$STARTS" \
    --ray_cpus "$RAY_CPUS" --ckpt_root "$WORK/ckpts" --log_dir "$WORK/logs" "${VAL[@]}" \
    -- trainer.val_before_train=False trainer.max_actor_ckpt_to_keep=$KEEP_CKPT
fi
for s in $STARTS; do
  C=$(trunk_ckpt "$s")
  [ -d "$C/actor" ] || die "$NAME: no checkpoint at step $s (see $WORK/logs/$NAME.log)"
  ls "$C/actor"/optim_world_size_* >/dev/null 2>&1 || die "$NAME: step-$s checkpoint has no optimizer state"
done

# evaluate every start point once (the smoke run skips this; its arm smoke test runs the final evals)
if [ "${SMOKE:-0}" != 1 ]; then for s in $STARTS; do bash "$KIT/eval_start.sh" "$SETTING" "$s"; done; fi
{ echo "name=$NAME"; echo "host=$(hostname)"; echo "finished=$(date -Is)"; echo "starts=$STARTS"; } > "$DONE"
echo "[grpo_trunk] $NAME done -> $(for s in $STARTS; do trunk_ckpt "$s"; done | tr '\n' ' ')"

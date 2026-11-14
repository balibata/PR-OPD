#!/bin/bash
# Stage II run: <setting> x <start> x <arm> x <seed>. Resumes the trunk checkpoint global_step_<start> (weights,
# optimizer, step counter) and trains to the last END step (default 200); the checkpoints at every END step after
# <start> (default 150 and 200) are kept and evaluated.
#
#   bash experiments/run_arm.sh 3b-alf 100 pr-opd 1
#   bash experiments/run_arm.sh 3b-ws 140 grpo 2
#   SMOKE=1 bash experiments/run_arm.sh 3b-ws 2 pr-opd 1   # resumes smoke-grpo-3b-ws step 2, trains to 4, full evals at 3 and 4
#
# KL reference = base model (as in Stage I). env.seed = vLLM engine seed = <seed>. In-training validation every
# TEST_FREQ steps is deterministic:
#   ALFWorld: all 140 seen games, each exactly once (ALFWORLD_DET_EVAL=1)  -> the pass at an END step IS the seen result
#   WebShop : test goals 0..124 (WEBSHOP_DET_EVAL=1, offset 0)            -> curves only
# Final evaluations on every END-step checkpoint:
#   ALFWorld: (only if EVAL_UNSEEN=1, always in SMOKE) unseen split, 134 games -> evals/<name>/unseen_step<k>.log
#   WebShop : all 500 test goals (4 batches of 125)    -> evals/<name>/ws500_step<k>.log
# Afterwards the optimizer shards are deleted (model shards kept unless KEEP_FINAL_CKPTS=0).
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd); source "$KIT/config.sh"; source "$KIT/lib.sh"
SETTING=${1:?usage: run_arm.sh <setting> <start> <arm> <seed>}; S0=${2:?start}; ARM=${3:?arm}; SEED=${4:?seed}
setting_vars "$SETTING"; arm_vars "$ARM"
case " $STARTS " in *" $S0 "*) ;; *) die "start $S0 is not a start step of $SETTING ($STARTS)";; esac
NAME=$(run_name "$SETTING" "$S0" "$ARM" "$SEED"); START_CKPT=$(trunk_ckpt "$S0")
EVAL_STEPS=$(end_steps "$S0"); [ -n "$EVAL_STEPS" ] || die "no END step after start $S0 (END_STEPS=$END_STEPS)"
# shellcheck disable=SC2086
TOTAL=$(max_of $EVAL_STEPS); SAVE=$SAVE_FREQ; TF=$TEST_FREQ; KEEP=$KEEP_CKPT
if [ "${SMOKE:-0}" = 1 ]; then NAME=smoke-$NAME; SAVE=1; TF=1; else check_protocol; fi
CK=$WORK/ckpts/$NAME; LOG=$WORK/logs/$NAME.log; EV=$WORK/evals/$NAME; DONE=$WORK/done/$NAME
mkdir -p "$WORK/logs" "$WORK/done" "$EV"
[ -f "$DONE" ] && { echo "[run_arm] $NAME already done"; exit 0; }
[ -f "$WORK/done/$TRUNK_NAME" ] && [ -d "$START_CKPT/actor" ] || die "Stage I not finished: $START_CKPT (run grpo_trunk.sh $SETTING)"

activate_env "$CONDA_ENV"; runtime_env; cd "$REPO"
UTIL=$(gpu_util "$ENV_KEY"); val_args "$ENV_KEY"
echo "[run_arm] $NAME on $(hostname) GPUs=$(gpu_names) util=$UTIL micro=$MICRO steps=$S0->$TOTAL eval=[$EVAL_STEPS]"

# ---------------- training ----------------
LAST=$(cat "$CK/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [ "$LAST" -lt "$TOTAL" ]; then
  # A restarted job that never got past the start keeps the old log aside so the collector never mixes attempts.
  if [ "$LAST" -le "$S0" ] && [ -f "$LOG" ]; then mv "$LOG" "$LOG.prev.$(date +%s)"; fi
  if [ "$LAST" -gt "$S0" ]; then echo "[run_arm] WARNING: resuming $NAME from step $LAST (env task sampling restarts)"; fi
  bash scripts/train.sh "${ARM_ARGS[@]}" --env "$ENV_KEY" --model "$BASE" --init "$START_CKPT" "${NOTHINK_FLAG[@]}" \
    --name "$NAME" --steps "$TOTAL" --gpus "$GPUS" --micro "$MICRO" --lp_micro "$LP_MICRO" --tp "$TP" \
    --gpu_util "$UTIL" --seed "$SEED" --rollout_seed "$SEED" --save_freq "$SAVE" --test_freq "$TF" \
    --keep_steps "$EVAL_STEPS" --ray_cpus "$RAY_CPUS" --ckpt_root "$WORK/ckpts" --log_dir "$WORK/logs" "${VAL[@]}" \
    -- trainer.val_before_train=False trainer.max_actor_ckpt_to_keep=$KEEP
fi
LAST=$(cat "$CK/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
[ "$LAST" -ge "$TOTAL" ] || die "$NAME: training ended at step $LAST < $TOTAL (see $LOG)"

# ---------------- final evaluations ----------------
for k in $EVAL_STEPS; do
  C=$CK/global_step_$k; [ -d "$C/actor" ] || die "missing checkpoint $C"
  COMMON=(--ckpt "$C" --env "$ENV_KEY" --model "$BASE" "${NOTHINK_FLAG[@]}" --gpus "$GPUS" --micro "$MICRO"
          --lp_micro "$LP_MICRO" --tp "$TP" --gpu_util "$UTIL" --seed "$SEED" --ray_cpus "$RAY_CPUS")
  if [ "$ENV_KEY" = alfworld ]; then
    [ "$EVAL_UNSEEN" = 1 ] || [ "${SMOKE:-0}" = 1 ] || continue   # the seen-140 result is the in-training pass at step k
    O=$EV/unseen_step$k.log
    grep -qa 'val/success_rate' "$O" 2>/dev/null || \
      ALFWORLD_DET_EVAL=1 bash scripts/eval.sh "${COMMON[@]}" --val_n 134 --data_dir "$WORK/data/v134/text" \
        --eval_split eval_out_of_distribution --out "$O"
  else
    # one process, 4 batches of 125 -> all 500 test goals (WEBSHOP_EVAL_ADVANCE moves the chunk per batch)
    O=$EV/ws500_step$k.log
    grep -qa 'val/success_rate' "$O" 2>/dev/null || \
      WEBSHOP_DET_EVAL=1 WEBSHOP_EVAL_OFFSET=0 WEBSHOP_EVAL_ADVANCE=1 bash scripts/eval.sh "${COMMON[@]}" --val_n 125 \
        --data_dir "$WORK/data/v500/text" --out "$O"
  fi
done

# ---------------- disk: drop optimizer shards and the copied start checkpoint ----------------
rm -f "$CK"/global_step_*/actor/optim_world_size_* 2>/dev/null || true
rm -rf "${CK:?}/global_step_$S0"
if [ "$KEEP_FINAL_CKPTS" = 0 ]; then rm -rf "$CK"/global_step_*/actor 2>/dev/null || true; fi
{ echo "name=$NAME"; echo "host=$(hostname)"; echo "finished=$(date -Is)"; echo "steps=$S0->$TOTAL"; echo "eval=$EVAL_STEPS"; } > "$DONE"
echo "[run_arm] $NAME done"

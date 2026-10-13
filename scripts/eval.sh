#!/bin/bash
# Evaluate one checkpoint (validation only, no training) on ALFWorld or WebShop.
#
#   bash scripts/eval.sh --ckpt checkpoints/<run>/global_step_150 --env alfworld --model Qwen/Qwen2.5-3B-Instruct [--seed 1] [--gpus 8] [--out FILE]
#
# The checkpoint is loaded through verl's resume path; the policy weights are evaluated on the
# 128-task validation set (temperature 0.4) exactly as during training (trainer.val_only=True).
# --seed sets env.seed for the evaluation episodes (the paper protocol evaluates step 130/140/150
# with env.seed 1/2/3). The success rate is printed at the end and the full log is kept in --out.
set -euo pipefail
CKPT=""; ENV=""; MODEL=""; SEED=1; GPUS=8; OUT=""; MICRO=""; TP=2; DATA_DIR="${HOME}/data/verl-agent/text"; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt) CKPT=$2; shift 2;; --env) ENV=$2; shift 2;; --model) MODEL=$2; shift 2;; --seed) SEED=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;; --out) OUT=$2; shift 2;; --micro) MICRO=$2; shift 2;; --tp) TP=$2; shift 2;; --data_dir) DATA_DIR=$2; shift 2;;
    --) shift; EXTRA=("$@"); break;;
    -h|--help) sed -n '2,10p' "$0"; exit 0;;
    *) echo "unknown option: $1"; exit 1;;
  esac
done
[ -n "$CKPT" ] && [ -n "$ENV" ] && [ -n "$MODEL" ] || { echo "--ckpt, --env and --model are required"; exit 1; }
CKPT=${CKPT%/}; [ -d "$CKPT/actor" ] || { echo "not a checkpoint dir (needs actor/): $CKPT"; exit 1; }
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
RUN=$(basename "$(dirname "$CKPT")"); STEP=${CKPT##*global_step_}
OUT=${OUT:-"$(dirname "$CKPT")/eval_step${STEP}_seed${SEED}.log"}
case "$ENV" in
  alfworld) ENV_NAME="alfworld/AlfredTWEnv"; MAX_STEPS=50; MAX_PROMPT=2048; MINI=256; GPU_UTIL=0.6;  MICRO_DEFAULT=32;;
  webshop)  ENV_NAME="Webshop";             MAX_STEPS=15; MAX_PROMPT=4096; MINI=64;  GPU_UTIL=0.75; MICRO_DEFAULT=8;;
  *) echo "unknown --env $ENV"; exit 1;;
esac
MICRO=${MICRO:-$MICRO_DEFAULT}
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" VLLM_USE_V1="${VLLM_USE_V1:-1}" HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
MODEL_TAG=$(basename "${MODEL%/}" | tr '[:upper:]' '[:lower:]'); THINK_ARGS=()
if [ "${THINKING:-0}" = 0 ] && echo "$MODEL_TAG" | grep -q 'qwen3'; then THINK_ARGS=(+data.apply_chat_template_kwargs.enable_thinking=False); fi   # as in training (THINKING=1 to keep the template default)
TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/pr_opd_eval.XXXXXX")   # val_only never writes checkpoints; keep default_local_dir out of the run dir

echo "[eval.sh] $RUN step $STEP, env.seed=$SEED -> $OUT"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files=$DATA_DIR/train.parquet data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=16 data.val_batch_size=128 \
  data.max_prompt_length=$MAX_PROMPT data.max_response_length=512 \
  data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
  actor_rollout_ref.rollout.enable_chunked_prefill=False actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  env.env_name=$ENV_NAME env.seed=$SEED env.max_steps=$MAX_STEPS env.rollout.n=8 env.resources_per_worker.num_cpus=0.1 \
  trainer.critic_warmup=0 trainer.logger=[console] trainer.project_name=pr_opd_eval trainer.experiment_name=eval_${RUN}_step${STEP}_seed${SEED} \
  trainer.default_local_dir=$TMP_DIR trainer.n_gpus_per_node=$GPUS trainer.nnodes=1 trainer.ray_wait_register_center_timeout=600 \
  trainer.val_only=True trainer.resume_mode=resume_path trainer.resume_from_path=$CKPT \
  "${THINK_ARGS[@]}" "${EXTRA[@]}" > "$OUT" 2>&1 || { echo "[eval.sh] FAILED, see $OUT"; exit 1; }
rm -rf "$TMP_DIR"
echo "[eval.sh] $RUN step $STEP seed $SEED: $(grep -aoE 'val/[A-Za-z_]*success_rate[^ ]*:[0-9.]+' "$OUT" | sort -u | tr '\n' ' ')"
grep -aoE "val/webshop_task_score[^:]*:[0-9.]+" "$OUT" | tail -1 || true

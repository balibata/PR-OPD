#!/bin/bash
# Unified launcher for GRPO / SDAR / PR-OPD on ALFWorld and WebShop.
#
#   bash scripts/train.sh --method pr-opd --env alfworld --model Qwen/Qwen2.5-3B-Instruct \
#        [--init base | --init /path/to/grpo_run/global_step_100] [--proj none|mlp] [options] [-- extra hydra overrides]
#
# Options
#   --method   grpo | sdar | pr-opd                                   (required)
#              same-start comparison arms (all through the PR-OPD entry, same pipeline/RNG as pr-opd):
#                grpo-pr     GRPO only (both distillation coefficients 0)
#                token-opd   SDAR gated token objective (coef 0.01, beta 5) on self-generated hindsight skills
#                skillsd-self Skill-SD objective (k3 KL x importance ratio, lambda 0.001) on self-generated skills
#   --env      alfworld | webshop                                     (required)
#   --model    HF model id or local path                              (required)
#   --name     run name; checkpoints go to <ckpt_root>/<name>          (default <method>-<env>-<model basename>[-mlp][-from<N>])
#   --init     base (train from --model) or a checkpoint directory
#              ".../global_step_N" of a previous (e.g. GRPO) run to
#              continue from. The step counter continues from N.        (default base)
#   --proj     none | mlp   PR-OPD projection head                    (default none; pr-opd only)
#   --coef     PR-OPD representation-distillation coefficient         (default 0.01; pr-opd only)
#   --steps    total training steps                                   (default 150)
#   --gpus     GPUs on this node                                      (default 8)
#   --micro    actor micro batch size per GPU                         (default alfworld 32 / webshop 8; use 16 / 4 for 7B)
#   --tp       vLLM tensor parallel size                              (default 2)
#   --seed     env.seed                                               (default 0)
#   --save_freq / --test_freq                                         (default 10 / 5)
#   --thinking   keep the Qwen3 thinking template (default: enable_thinking=False for Qwen3 models)
#   --ckpt_root  checkpoint root                                      (default ./checkpoints)
#   --data_dir   parquet dir with train.parquet/test.parquet          (default $HOME/data/verl-agent/text)
#   --log_dir    where the console log goes                           (default ./logs)
#   --ref_model  KL reference model (default: same as --model). Use the base model when --model
#                is a merged checkpoint, so the KL anchor matches a resumed run.
#   --nothink    force the non-thinking chat template (needed when --model is a merged
#                checkpoint dir whose name no longer reveals the model family)
#   --val_n      validation episodes per pass = data.val_batch_size (default 128); the val
#                parquet in --val_dir must have exactly this many rows
#   --val_dir    parquet dir for validation (default: --data_dir)
#   --eval_split ALFWorld split: eval_in_distribution (default) | eval_out_of_distribution
#   --rollout_seed  vLLM sampling seed (default: config default)
#   --skill_source  pr-opd teacher context: self (default) | shuffled (other trajectory's skill)
#   --ckpt_contents comma list for actor checkpoints, e.g. model,extra (default model,optimizer,extra)
#   --gpu_util   vLLM gpu_memory_utilization (default alfworld 0.6 / webshop 0.75)
#   --ray_cpus   ray_init.num_cpus (set to the SLURM CPU allocation)
#   --lp_micro   log-prob micro batch per GPU for rollout/ref/teacher scoring (default 32)
#   --keep_steps "100 120"  global steps whose checkpoints are never rotated out (with
#                trainer.max_actor_ckpt_to_keep=1 only these plus the newest checkpoint stay on disk)
# Anything after "--" is passed verbatim to hydra (e.g. trainer.logger=['console','wandb']).
#
# All protocol hyper-parameters follow the SDAR paper scripts (examples/*_trainer/run_*.sh in
# ZJU-REAL/SDAR): batch 16 tasks x 8 rollouts, lr 1e-6, KL 0.01 (low_var_kl), 150 steps,
# ALFWorld max_steps 50 / WebShop max_steps 15, validation on 128 tasks every 5 steps.
set -euo pipefail

METHOD=""; ENV=""; MODEL=""; NAME=""; INIT="base"; PROJ="none"; COEF="0.01"; STEPS=150; GPUS=8
MICRO=""; TP=2; SEED=0; THINKING=0; NOTHINK=0; REF_MODEL=""; VAL_N=128; VAL_DIR=""; EVAL_SPLIT=""; ROLLOUT_SEED=""; SKILL_SOURCE=self; CKPT_CONTENTS=""; GPU_UTIL_OVR=""; RAY_CPUS=""; LP_MICRO=32; KEEP_STEPS=""; SAVE_FREQ=10; TEST_FREQ=5; CKPT_ROOT="./checkpoints"; DATA_DIR="${HOME}/data/verl-agent/text"; LOG_DIR="./logs"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --method) METHOD=$2; shift 2;;   --env) ENV=$2; shift 2;;       --model) MODEL=$2; shift 2;;
    --name) NAME=$2; shift 2;;       --init) INIT=$2; shift 2;;     --proj) PROJ=$2; shift 2;;
    --coef) COEF=$2; shift 2;;       --steps) STEPS=$2; shift 2;;   --gpus) GPUS=$2; shift 2;;
    --micro) MICRO=$2; shift 2;;     --tp) TP=$2; shift 2;;         --seed) SEED=$2; shift 2;;
    --save_freq) SAVE_FREQ=$2; shift 2;; --test_freq) TEST_FREQ=$2; shift 2;;
    --ckpt_root) CKPT_ROOT=$2; shift 2;; --data_dir) DATA_DIR=$2; shift 2;; --log_dir) LOG_DIR=$2; shift 2;;
    --thinking) THINKING=1; shift;;     --nothink) NOTHINK=1; shift;;
    --ref_model) REF_MODEL=$2; shift 2;; --val_n) VAL_N=$2; shift 2;; --val_dir) VAL_DIR=$2; shift 2;;
    --eval_split) EVAL_SPLIT=$2; shift 2;; --rollout_seed) ROLLOUT_SEED=$2; shift 2;; --skill_source) SKILL_SOURCE=$2; shift 2;;
    --ckpt_contents) CKPT_CONTENTS=$2; shift 2;; --gpu_util) GPU_UTIL_OVR=$2; shift 2;; --ray_cpus) RAY_CPUS=$2; shift 2;; --lp_micro) LP_MICRO=$2; shift 2;;
    --keep_steps) KEEP_STEPS=$2; shift 2;;
    --) shift; EXTRA=("$@"); break;;
    -h|--help) sed -n '2,50p' "$0"; exit 0;;
    *) echo "unknown option: $1 (use -- before hydra overrides)"; exit 1;;
  esac
done
[ -n "$METHOD" ] && [ -n "$ENV" ] && [ -n "$MODEL" ] || { echo "--method, --env and --model are required (see --help)"; exit 1; }
case "$METHOD" in grpo|sdar|pr-opd|grpo-pr|token-opd|skillsd-self) ;; *) echo "unknown --method $METHOD (grpo|sdar|pr-opd|grpo-pr|token-opd|skillsd-self)"; exit 1;; esac
case "$SKILL_SOURCE" in self|shuffled) ;; *) echo "unknown --skill_source $SKILL_SOURCE (self|shuffled)"; exit 1;; esac
case "$ENV" in alfworld|webshop) ;; *) echo "unknown --env $ENV (alfworld|webshop)"; exit 1;; esac
case "$PROJ" in none|mlp) ;; *) echo "unknown --proj $PROJ (none|mlp)"; exit 1;; esac

ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
MODEL_TAG=$(basename "${MODEL%/}" | tr '[:upper:]' '[:lower:]')
if [ -z "$NAME" ]; then
  NAME="${METHOD}-${ENV}-${MODEL_TAG}"
  [ "$METHOD" = pr-opd ] && [ "$PROJ" = mlp ] && NAME="${NAME}-mlp"
  [ "$INIT" != base ] && NAME="${NAME}-from$(basename "${INIT%/}" | sed 's/global_step_//')"
fi
CK="$CKPT_ROOT/$NAME"; mkdir -p "$CK" "$LOG_DIR"; RUNLOG="$LOG_DIR/$NAME.log"

# ---- start point ----------------------------------------------------------------------------
# A checkpoint directory global_step_N of a previous run is copied into this run's checkpoint
# dir; verl's resume_mode=auto then picks it up and continues from step N (optimizer state is
# restored when present, otherwise the optimizer starts fresh).
if [ "$INIT" != base ]; then
  SRC=${INIT%/}; STEP=${SRC##*global_step_}
  [ -d "$SRC/actor" ] || { echo "--init must point to a checkpoint dir .../global_step_N (with actor/): $SRC"; exit 1; }
  if [ ! -f "$CK/latest_checkpointed_iteration.txt" ]; then
    echo "[train.sh] initialising $NAME from $SRC (step $STEP)"
    cp -al "$SRC" "$CK/global_step_$STEP" 2>/dev/null || { rm -rf "$CK/global_step_$STEP"; cp -a "$SRC" "$CK/global_step_$STEP"; }
    echo "$STEP" > "$CK/latest_checkpointed_iteration.txt"
  else
    echo "[train.sh] $CK already has checkpoints; resuming from latest (ignoring --init)"
  fi
fi

# ---- protocol (identical to the SDAR paper scripts) -----------------------------------------
case "$ENV" in
  alfworld) ENV_NAME="alfworld/AlfredTWEnv"; MAX_STEPS=50; MAX_PROMPT=2048; MINI=256; GPU_UTIL=0.6;  MICRO_DEFAULT=32;;
  webshop)  ENV_NAME="Webshop";             MAX_STEPS=15; MAX_PROMPT=4096; MINI=64;  GPU_UTIL=0.75; MICRO_DEFAULT=8;;
esac
MICRO=${MICRO:-$MICRO_DEFAULT}
[ -n "$GPU_UTIL_OVR" ] && GPU_UTIL=$GPU_UTIL_OVR
VAL_DIR=${VAL_DIR:-$DATA_DIR}
case "$METHOD" in
  grpo)   ENTRY=verl.trainer.main_ppo;    METHOD_ARGS=();;
  sdar)   ENTRY=verl.trainer.main_sdar;   METHOD_ARGS=(+algorithm.sdar.sdar_coef=0.01 +algorithm.sdar.gate_beta=5.0
                                                       +algorithm.sdar.skills_dir=skills/$ENV +algorithm.sdar.skill_all=false);;
  pr-opd) ENTRY=verl.trainer.main_pr_opd; METHOD_ARGS=(+algorithm.pr_opd.oprd_coef=$COEF +algorithm.pr_opd.oprd_layers=all
                                                       +algorithm.pr_opd.oprd_normalize=True +algorithm.pr_opd.oprd_proj_mode=$PROJ)
          [ "$PROJ" = mlp ] && METHOD_ARGS+=(+algorithm.pr_opd.oprd_proj_hidden=512 +algorithm.pr_opd.oprd_proj_lr=1e-4)
          [ "$INIT" != base ] && METHOD_ARGS+=(+algorithm.pr_opd.proj_allow_missing=True)
          METHOD_ARGS+=(+algorithm.pr_opd.skill_source=$SKILL_SOURCE);;
  grpo-pr)      ENTRY=verl.trainer.main_pr_opd; METHOD_ARGS=(+algorithm.pr_opd.oprd_coef=0.0 +algorithm.pr_opd.coef=0.0 +algorithm.pr_opd.sdl_coef=0.0);;
  token-opd)    ENTRY=verl.trainer.main_pr_opd; METHOD_ARGS=(+algorithm.pr_opd.oprd_coef=0.0 +algorithm.pr_opd.coef=0.01 +algorithm.pr_opd.gate_beta=5.0
                                                             +algorithm.pr_opd.sdl_coef=0.0 +algorithm.pr_opd.skill_source=$SKILL_SOURCE);;
  skillsd-self) ENTRY=verl.trainer.main_pr_opd; METHOD_ARGS=(+algorithm.pr_opd.oprd_coef=0.0 +algorithm.pr_opd.coef=0.0
                                                             +algorithm.pr_opd.sdl_coef=0.001 +algorithm.pr_opd.skill_source=$SKILL_SOURCE);;
esac
[ -n "$REF_MODEL" ] && METHOD_ARGS+=(+actor_rollout_ref.ref.model_path=$REF_MODEL)
[ -n "$EVAL_SPLIT" ] && METHOD_ARGS+=(env.alfworld.eval_dataset=$EVAL_SPLIT)
[ -n "$ROLLOUT_SEED" ] && METHOD_ARGS+=(+actor_rollout_ref.rollout.seed=$ROLLOUT_SEED)
[ -n "$CKPT_CONTENTS" ] && METHOD_ARGS+=("actor_rollout_ref.actor.checkpoint.contents=[${CKPT_CONTENTS}]")
[ -n "$RAY_CPUS" ] && METHOD_ARGS+=(ray_init.num_cpus=$RAY_CPUS)

# Qwen3 models: the paper runs use the non-thinking chat template (the ALFWorld/WebShop
# prompts ask for <think>...</think> inside the response). Pass --thinking to keep the template default.
if [ "$NOTHINK" = 1 ] || { [ "$THINKING" = 0 ] && echo "$MODEL_TAG" | grep -q 'qwen3'; }; then
  METHOD_ARGS+=(+data.apply_chat_template_kwargs.enable_thinking=False)
fi

export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" VLLM_USE_V1="${VLLM_USE_V1:-1}" HYDRA_FULL_ERROR=1
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"   # same sampler kernel as the paper runs
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PR_OPD_KEEP_CKPT_STEPS="$KEEP_STEPS"   # read by verl/utils/checkpoint/checkpoint_manager.py

echo "[train.sh] $NAME: method=$METHOD env=$ENV model=$MODEL init=$INIT proj=$PROJ coef=$COEF steps=$STEPS micro=$MICRO -> $CK (log: $RUNLOG)"
python3 -m $ENTRY \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files=$DATA_DIR/train.parquet data.val_files=$VAL_DIR/test.parquet \
  data.train_batch_size=16 data.val_batch_size=$VAL_N \
  data.max_prompt_length=$MAX_PROMPT data.max_response_length=512 \
  data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LP_MICRO \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
  actor_rollout_ref.rollout.enable_chunked_prefill=False actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LP_MICRO actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  env.env_name=$ENV_NAME env.seed=$SEED env.max_steps=$MAX_STEPS env.rollout.n=8 env.resources_per_worker.num_cpus=0.1 \
  trainer.critic_warmup=0 trainer.logger=[console] \
  trainer.project_name=pr_opd_$ENV trainer.experiment_name=$NAME trainer.default_local_dir=$CK \
  trainer.n_gpus_per_node=$GPUS trainer.nnodes=1 trainer.ray_wait_register_center_timeout=600 \
  trainer.save_freq=$SAVE_FREQ trainer.test_freq=$TEST_FREQ trainer.max_actor_ckpt_to_keep=null \
  trainer.total_epochs=$STEPS trainer.total_training_steps=$STEPS trainer.val_before_train=True trainer.resume_mode=auto \
  "${METHOD_ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$RUNLOG"
echo "[train.sh] done: $NAME (checkpoints in $CK)"

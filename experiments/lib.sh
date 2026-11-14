# Shared helpers for the experiment scripts. Source after config.sh.
# shellcheck disable=SC2034

die(){ echo "[experiments] ERROR: $*" >&2; exit 1; }

max_of(){ local m=0 x; for x in "$@"; do [ "$x" -gt "$m" ] && m=$x; done; echo "$m"; }

# setting_vars <setting>: ENV_KEY BASE NOTHINK_FLAG MICRO LP_MICRO CONDA_ENV STARTS TRUNK_NAME TRUNK_STEPS
setting_vars(){
  case "$1" in
    # 3B Instruct: default chat template. 1.7B: non-thinking chat template (--nothink), as in the paper runs.
    3b-alf)  ENV_KEY=alfworld; BASE=$BASE_3B;  NOTHINK=0; MICRO=$MICRO_3B_ALF;  LP_MICRO=$LP_MICRO_3B;  CONDA_ENV=$CONDA_ENV_ALF;;
    3b-ws)   ENV_KEY=webshop;  BASE=$BASE_3B;  NOTHINK=0; MICRO=$MICRO_3B_WS;   LP_MICRO=$LP_MICRO_3B;  CONDA_ENV=$CONDA_ENV_WS;;
    17b-alf) ENV_KEY=alfworld; BASE=$BASE_17B; NOTHINK=1; MICRO=$MICRO_17B_ALF; LP_MICRO=$LP_MICRO_17B; CONDA_ENV=$CONDA_ENV_ALF;;
    17b-ws)  ENV_KEY=webshop;  BASE=$BASE_17B; NOTHINK=1; MICRO=$MICRO_17B_WS;  LP_MICRO=$LP_MICRO_17B; CONDA_ENV=$CONDA_ENV_WS;;
    *) die "unknown setting '$1' ($SETTINGS_ALL)";;
  esac
  NOTHINK_FLAG=(); if [ "$NOTHINK" = 1 ]; then NOTHINK_FLAG=(--nothink); fi
  if [ "$ENV_KEY" = alfworld ]; then STARTS=$START_STEPS_ALF; else STARTS=$START_STEPS_WS; fi
  # shellcheck disable=SC2086
  TRUNK_NAME=grpo-$1; TRUNK_STEPS=$(max_of $STARTS)
  if [ "${SMOKE:-0}" = 1 ]; then STARTS=2; TRUNK_NAME=smoke-grpo-$1; TRUNK_STEPS=2; fi
  return 0
}

trunk_ckpt(){ echo "$WORK/ckpts/$TRUNK_NAME/global_step_$1"; }          # trunk_ckpt <start>   (after setting_vars)
run_name(){ echo "$1-from$2-$3-s$4"; }                                    # <setting>-from<start>-<arm>-s<seed>
start_name(){ echo "start-$1-$2"; }                                       # start-<setting>-<start>

# end_steps <start>: the END steps after <start> (SMOKE: start+1 and start+2)
end_steps(){
  if [ "${SMOKE:-0}" = 1 ]; then echo "$(($1 + 1)) $(($1 + 2))"; return 0; fi
  local e out=""; for e in $END_STEPS; do [ "$e" -gt "$1" ] && out="$out $e"; done; echo "${out# }"
}

# check_protocol: END steps must be validated (multiples of TEST_FREQ) and saved (multiples of SAVE_FREQ)
check_protocol(){
  local e; for e in $END_STEPS; do
    [ $((e % TEST_FREQ)) -eq 0 ] || die "END step $e is not a multiple of TEST_FREQ=$TEST_FREQ (no validation pass there)"
    [ $((e % SAVE_FREQ)) -eq 0 ] || die "END step $e is not a multiple of SAVE_FREQ=$SAVE_FREQ (no checkpoint there)"
  done
}

# val_args <env_key>: deterministic in-training validation (exports the env switch, sets VAL)
val_args(){
  if [ "$1" = alfworld ]; then
    export ALFWORLD_DET_EVAL=1; VAL=(--val_n 140 --data_dir "$WORK/data/v140/text")
  else
    export WEBSHOP_DET_EVAL=1 WEBSHOP_EVAL_OFFSET=0; VAL=(--val_n 125 --data_dir "$WORK/data/v125/text")
  fi
}

# arm_vars <arm>: ARM_ARGS (train.sh options defining the objective)
# All arms use the PR-OPD entry point, so rollout, skill-free student pass, KL, entropy and
# logging are identical; only the distillation term differs.
arm_vars(){
  case "$1" in
    grpo)          ARM_ARGS=(--method grpo-pr);;                                                  # control: GRPO continued
    pr-opd)        ARM_ARGS=(--method pr-opd --proj mlp ${OPRD_COEF:+--coef "$OPRD_COEF"});;     # PR-OPD with the MLP projector
    pr-opd-noproj) ARM_ARGS=(--method pr-opd --proj none ${OPRD_COEF:+--coef "$OPRD_COEF"});;    # projector ablation
    token-opd)     ARM_ARGS=(--method token-opd);;          # SDAR gated token objective on the same self-generated skills
    skillsd-self)  ARM_ARGS=(--method skillsd-self);;       # Skill-SD objective on the same self-generated skills
    pr-opd-shuf)   ARM_ARGS=(--method pr-opd --proj mlp --skill_source shuffled);;                # content control
    *) die "unknown arm '$1'";;
  esac
  # OPRD_COEF (env, optional): representation-distillation coefficient for the pr-opd arms (default 0.01)
  return 0
}

activate_env(){
  if [ -n "${CONDA_MODULE:-}" ]; then module load "$CONDA_MODULE" || die "module load $CONDA_MODULE failed"; fi
  # shellcheck disable=SC1090
  source "$CONDA_SH" || die "cannot source $CONDA_SH"
  conda activate "$1" || die "cannot activate conda env $1"
  if [ "${ENV_KEY:-}" = webshop ]; then
    # WebShop's Lucene search runs in a JVM; without these the env actors die in __init__.
    # conda-forge openjdk puts libjvm either under $CONDA_PREFIX/lib/jvm/lib/server or directly under $CONDA_PREFIX/lib/server
    local jh=${JAVA_HOME_WS:-}
    if [ -z "$jh" ]; then jh=$CONDA_PREFIX/lib/jvm; [ -f "$jh/lib/server/libjvm.so" ] || jh=$CONDA_PREFIX; fi
    export JAVA_HOME=$jh
    export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
    [ -f "$JVM_PATH" ] || die "no JVM at $JVM_PATH (conda install -c conda-forge openjdk=11, or set JAVA_HOME_WS)"
    export PATH="$JAVA_HOME/bin:$PATH"
    # cap the per-worker heap (each search actor grows ~0.5 GB otherwise)
    export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms32m -Xmx512m}"
  fi
}

# Runtime hygiene on shared nodes: no core dumps, private Ray dir, sane thread counts.
runtime_env(){
  ulimit -c 0
  # some SLURM gres plugins also export ROCR/HIP_VISIBLE_DEVICES; Ray then refuses to start the CUDA workers
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
  local tag=${SLURM_JOB_ID:-$$}
  export RAY_TMPDIR=${RAY_TMPDIR_BASE:-${TMPDIR:-/tmp}}/ray_propd_$tag
  mkdir -p "$RAY_TMPDIR"
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} TOKENIZERS_PARALLELISM=false
  export WEBSHOP_LIGHT_WORKER=${WEBSHOP_LIGHT_WORKER:-1} WEBSHOP_SHARED_SEARCH=${WEBSHOP_SHARED_SEARCH:-1}
  # Ray CPUs are a logical scheduling budget. WebShop alone books ~28 of them (253 env workers x 0.1 + 4 Lucene
  # actors), so with a 32-CPU allocation the GPU workers could never be placed (job hangs before init).
  # Book at least RAY_CPUS_MIN (48) regardless of the allocation; real CPU use is unchanged.
  RAY_CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
  if [ "$RAY_CPUS" -lt "${RAY_CPUS_MIN:-48}" ]; then RAY_CPUS=${RAY_CPUS_MIN:-48}; fi
}

# gpu_util <env_key>: vLLM memory fraction from the card size of GPU 0.
gpu_util(){
  local mib; mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' ')
  mib=${mib:-81920}
  if [ "$mib" -ge 120000 ]; then
    [ "$1" = alfworld ] && echo "$GPU_UTIL_ALF_BIG" || echo "$GPU_UTIL_WS_BIG"
  elif [ "$mib" -ge 70000 ]; then
    [ "$1" = alfworld ] && echo "$GPU_UTIL_ALF_80G" || echo "$GPU_UTIL_WS_80G"
  elif [ "$mib" -ge 40000 ] && [ "$BASE" = "$BASE_17B" ]; then
    echo "[experiments] WARNING: ${mib} MiB cards; use MICRO_17B_ALF=8 MICRO_17B_WS=2 for ALL runs of a setting" >&2
    [ "$1" = alfworld ] && echo "$GPU_UTIL_ALF_48G" || echo "$GPU_UTIL_WS_48G"
  else
    die "GPU 0 has ${mib} MiB; the 3B runs need 80 GB cards, the 1.7B runs >= 40 GB"
  fi
}

gpu_names(){ nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | tr '\n' ' '; }

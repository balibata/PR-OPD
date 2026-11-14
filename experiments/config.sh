# Branched-start experiments (GRPO trunk -> every arm from the same checkpoint): configuration.
# Site-specific values (paths, conda, SLURM account/partition) live in experiments/site.sh, which
# setup.sh writes and which is loaded FIRST: anything set there overrides the defaults below.
# shellcheck disable=SC2034

# ============================== PATHS ==============================
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$KIT/.." && pwd)
# shellcheck disable=SC1091
[ -f "$KIT/site.sh" ] && source "$KIT/site.sh"
WORK=${WORK:-$REPO/work}                     # data/, ckpts/, logs/, evals/, done/, results/, slurm/

# Base models (local HF dirs). Stage I GRPO starts from them and they stay the KL reference.
BASE_3B=${BASE_3B:-/path/to/base_3b}         # 3B Instruct model of the paper (default chat template)
BASE_17B=${BASE_17B:-/path/to/base_17b}      # 1.7B model of the paper (non-thinking chat template)

# Software
CONDA_MODULE=${CONDA_MODULE:-}               # optional: a module to load before conda (e.g. anaconda3)
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-pr-opd}               # ONE python 3.10 env for ALFWorld and WebShop (as in the paper runs)
CONDA_ENV_ALF=${CONDA_ENV_ALF:-$CONDA_ENV}; CONDA_ENV_WS=${CONDA_ENV_WS:-$CONDA_ENV}
# WebShop search = pyserini -> pyjnius -> JVM (OpenJDK 11). Empty = the JDK inside the conda env.
JAVA_HOME_WS=${JAVA_HOME_WS:-}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}

# ============================== PROTOCOL ==============================
# Stage I  (grpo_trunk.sh): ONE GRPO run per setting from the base model (env.seed GRPO_SEED). Its checkpoints at
#          every start step of the setting are kept WITH optimizer state; each start point is evaluated once.
# Stage II (run_arm.sh)   : every (start, arm, seed) resumes the trunk checkpoint global_step_<start> (weights +
#          optimizer + step counter) and trains to the last END step. Only the distillation term differs per arm.
#          The checkpoints at all END steps are kept and evaluated.
START_STEPS_ALF=${START_STEPS_ALF:-"100 120"}        # ALFWorld: branch from GRPO@100 and GRPO@120
START_STEPS_WS=${START_STEPS_WS:-"100 120 140"}      # WebShop : branch from GRPO@100, GRPO@120 and GRPO@140
END_STEPS=${END_STEPS:-"150 200"}                    # evaluate every Stage II run at global steps 150 and 200
GRPO_SEED=${GRPO_SEED:-0}                    # Stage I env.seed (paper default)
GRPO_TEST_FREQ=${GRPO_TEST_FREQ:-25}         # Stage I deterministic validation (sanity curve)
SEEDS=${SEEDS:-"1 2"}                        # Stage II training seeds: env.seed (task sampling) + vLLM engine seed
TEST_FREQ=${TEST_FREQ:-25}                   # Stage II deterministic validation every 25 global steps (125, 150, 175, 200).
                                             # Every END step must be a multiple of it: on ALFWorld the in-training pass at
                                             # an END step IS the seen-140 test result.
SAVE_FREQ=${SAVE_FREQ:-10}                   # checkpoints (with optimizer) so a killed job can resume; besides the
KEEP_CKPT=${KEEP_CKPT:-1}                    # protected start/END steps only the newest one is kept on disk
KEEP_FINAL_CKPTS=${KEEP_FINAL_CKPTS:-1}      # 0 = delete the END-step model shards once their evaluations are done
EVAL_UNSEEN=${EVAL_UNSEEN:-0}                # 1 = also evaluate the ALFWorld unseen split (134 games) at every END step

SETTINGS_ALL="3b-alf 3b-ws 17b-alf 17b-ws"
SETTINGS_DEFAULT=${SETTINGS_DEFAULT:-"3b-alf 3b-ws"}
# Stage II arms (see lib.sh:arm_vars). grpo = GRPO continued from the same start (control).
ARMS_DEFAULT=${ARMS_DEFAULT:-"grpo pr-opd pr-opd-noproj skillsd-self token-opd"}
# Also available with --arms: pr-opd-shuf (teacher sees another trajectory's skill).

# ============================== TRAINING KNOBS ==============================
GPUS=${GPUS:-8}                  # GPUs per job (one node). Tested: 3B on 4 x A100 80 GB, 1.7B on 8 x H100 80 GB.
TP=${TP:-1}                      # vLLM tensor parallel (1 -> one generation replica per GPU)
# Micro batches change only the gradient-accumulation split, not the objective. They must be the SAME for
# every run of a setting (trunk included). Defaults fit 80 GB cards; the paper used 32/8 on 180 GB B200s.
MICRO_3B_ALF=${MICRO_3B_ALF:-8};   MICRO_3B_WS=${MICRO_3B_WS:-2};   LP_MICRO_3B=${LP_MICRO_3B:-8}
MICRO_17B_ALF=${MICRO_17B_ALF:-16}; MICRO_17B_WS=${MICRO_17B_WS:-4}; LP_MICRO_17B=${LP_MICRO_17B:-16}
# vLLM memory fraction, chosen at run time from the card size (does not change the math).
GPU_UTIL_ALF_BIG=${GPU_UTIL_ALF_BIG:-0.55}; GPU_UTIL_WS_BIG=${GPU_UTIL_WS_BIG:-0.60}    # >= 120 GB (H100 NVL/H200/B200)
GPU_UTIL_ALF_80G=${GPU_UTIL_ALF_80G:-0.40}; GPU_UTIL_WS_80G=${GPU_UTIL_WS_80G:-0.40}    # 80 GB (A100/H100)
GPU_UTIL_ALF_48G=${GPU_UTIL_ALF_48G:-0.30}; GPU_UTIL_WS_48G=${GPU_UTIL_WS_48G:-0.30}    # 40-48 GB, 1.7B only: also set
                                                                                         # MICRO_17B_ALF=8 MICRO_17B_WS=2

# ============================== SLURM ==============================
# Empty values are simply not passed to sbatch. Fill in what your cluster needs in site.sh.
SB_PARTITION=${SB_PARTITION:-}
SB_ACCOUNT=${SB_ACCOUNT:-}
SB_QOS=${SB_QOS:-}
SB_GRES=${SB_GRES:-gpu:$GPUS}
SB_CPUS=${SB_CPUS:-64}
SB_MEM=${SB_MEM:-400G}           # ALFWorld: 128 train + 140 val env workers (~0.4 GB each) + GPU workers
SB_TIME=${SB_TIME:-72:00:00}
SB_EXCLUDE=${SB_EXCLUDE:-}
# At most LANES jobs at the same time. Jobs are named propd-lane<i> and submitted with --dependency=singleton,
# so SLURM runs at most one job per lane. 0 = no limit.
LANES=${LANES:-2}
SB_MAIL=${SB_MAIL:-}             # e.g. "--mail-type=FAIL --mail-user=you@x"
SB_EXTRA=${SB_EXTRA:-}           # any extra sbatch flags (e.g. --constraint=a100)

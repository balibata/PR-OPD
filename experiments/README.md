# Branched-start experiments

This folder runs the full comparison protocol: **one GRPO trunk per setting, and every method branched from the
same trunk checkpoint**, so that all arms share the start weights, the optimizer state and the step counter, and
differ only in their distillation term. Each run is evaluated with the deterministic, full-coverage evaluation
described in [Evaluation protocol](#evaluation-protocol).

| environment | branch points (GRPO trunk step) | evaluated at global step | updates per run |
|---|---|---|---|
| ALFWorld | 100, 120 | 150 and 200 | 100 / 80 |
| WebShop  | 100, 120, 140 | 150 and 200 | 100 / 80 / 60 |

Settings: `3b-alf`, `3b-ws` (3B Instruct model, default chat template) and `17b-alf`, `17b-ws` (1.7B model,
non-thinking chat template). Arms (all through the PR-OPD entry point, see `lib.sh:arm_vars`):

| arm | objective |
|---|---|
| `grpo` | GRPO continued from the branch point (control) |
| `pr-opd` | PR-OPD with the per-layer MLP projector (`--proj mlp`, coefficient 0.01) |
| `pr-opd-noproj` | PR-OPD without projector (`--proj none`) |
| `token-opd` | SDAR's gated token-level objective (β = 5, coefficient 0.01) on the same self-generated hindsight skills |
| `skillsd-self` | Skill-SD objective (k3 KL × importance ratio, λ = 0.001) on the same self-generated hindsight skills |
| `pr-opd-shuf` | optional content control: the teacher sees another trajectory's skill |

With the defaults (5 arms × 2 training seeds) one model needs 2 trunk jobs + 20 ALFWorld + 30 WebShop runs.

## 1. Layout

```
experiments/
├── config.sh            protocol + defaults (branch points, END steps, seeds, micro batches, SLURM)
├── site.sh              written by setup.sh: paths, conda, SLURM account; overrides config.sh (not tracked)
├── lib.sh               settings / arms, env activation (JVM for WebShop), runtime hygiene
├── setup.sh             base models, ALFWorld data, JDK, WebShop data, site.sh, placeholder data, layout check
├── test_envs.sh         ALFWorld / WebShop (CPU) and GPU + vLLM checks
├── grpo_trunk.sh        Stage I: GRPO from the base model to the last branch point, keeps every branch point
├── eval_start.sh        evaluation of one branch point (called by grpo_trunk.sh)
├── run_arm.sh           Stage II: one (setting, start, arm, seed) run to step 200 + evaluations at 150 and 200
├── submit_all.sh        SLURM queue: trunk -> Stage II (afterok), at most LANES concurrent jobs
├── run_local.sh         the same queue without SLURM, one job after another
├── job.sbatch           SLURM wrapper
├── prep_data.sh         copies placeholder_data/ into $WORK/data (setup.sh calls it)
├── placeholder_data/    v140 / v134 / v125 / v500 parquets (they only fix the batch sizes)
├── collect_results.py   -> $WORK/results/{summary.md, runs.csv, curves.csv}
├── check_rollout_diversity.py   sanity check: GRPO groups must contain different rollouts
├── download_webshop.sh  WebShop small product set + Lucene indexes (called by setup.sh when missing)
└── requirements_env.txt pip freeze of the environment used for the paper runs
```

Run names: trunk `grpo-<setting>`; Stage II `<setting>-from<start>-<arm>-s<seed>` (e.g. `3b-ws-from140-pr-opd-s2`);
start-point evaluation `start-<setting>-<start>`. Everything is written under `$WORK` (default `<repo>/work`):
`ckpts/`, `logs/<run>.log` (training log, one `step:N - ...` line per update), `evals/<run>/`, `done/<run>`,
`slurm/`, `results/`.

## 2. Environment

One Python 3.10 environment runs both ALFWorld and WebShop (CUDA 12.8 driver). Key versions: `torch==2.8.0`
(cu128), `vllm==0.11.0`, `flash_attn==2.8.3`, `transformers==4.57.6`, `ray==2.51.1`, `alfworld==0.4.2`,
`textworld==1.7.0`, `pyserini==0.17.0`, `pyjnius==1.7.0`, `spacy==3.8.15` (+ `en_core_web_sm` 3.8.0).

```bash
conda create -n pr-opd python=3.10 -y && conda activate pr-opd
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.11.0
pip install flash-attn==2.8.3 --no-build-isolation        # or the matching pre-built wheel
pip install -r experiments/requirements_env.txt           # already-installed packages are skipped
pip install -e . --no-deps
```

## 3. Setup

```bash
bash experiments/setup.sh --base_3b <HF repo id or local dir> [--base_17b <HF repo id or local dir>] \
     [--assets /big/disk/assets] [--work /big/disk/work]
# edit experiments/site.sh: CONDA_SH / CONDA_ENV, GPUS, SB_PARTITION / SB_ACCOUNT, LANES
bash experiments/test_envs.sh alfworld        # 140 seen / 134 unseen games, one reset + step
bash experiments/test_envs.sh webshop         # JVM + search + one session
bash experiments/test_envs.sh gpu 3b          # inside a GPU job: vLLM generation with the base model
```

GPU defaults: `GPUS=8`, vLLM TP 1. Micro batches (`MICRO_3B_ALF=8 MICRO_3B_WS=2`, `MICRO_17B_ALF=16
MICRO_17B_WS=4`) fit 80 GB cards; they only change the gradient-accumulation split, but must be identical for all
runs of a setting (trunk included). The 3B runs were tested on 4 × A100 80 GB (`GPUS=4`), the 1.7B runs on
8 × H100 80 GB.

## 4. Smoke test (do this first)

```bash
SMOKE=1 bash experiments/submit_all.sh --settings "3b-alf 3b-ws"     # SLURM
SMOKE=1 bash experiments/run_local.sh  --settings "3b-alf 3b-ws"     # or on an allocated node
```
Per setting: a 2-step GRPO trunk, then `pr-opd` for 2 more steps with checkpoints and evaluations at steps 3 and 4
(WebShop 500 goals, ALFWorld unseen 134). Check `$WORK/slurm/*.out` and `$WORK/evals/smoke-*`, then delete
`$WORK/{ckpts,logs,evals,done}/smoke-*`.

## 5. Running

```bash
bash experiments/submit_all.sh --dry-run                              # what would be submitted
bash experiments/submit_all.sh                                        # SETTINGS_DEFAULT, all starts, ARMS_DEFAULT, SEEDS
bash experiments/submit_all.sh --settings "17b-alf 17b-ws"            # the 1.7B settings
bash experiments/submit_all.sh --settings 3b-ws --starts 140 --arms "grpo pr-opd" --seeds 1
bash experiments/run_local.sh  [same options]                          # without SLURM
bash experiments/run_local.sh --stage1 3b-alf --seeds 1                # two allocations sharing one $WORK:
bash experiments/run_local.sh --stage1 3b-ws  --seeds 2                #   seeds run in parallel
```
Both scripts skip finished runs (`$WORK/done/<run>`) and queued ones, so they can simply be re-run after a failure.
A killed job resumes from its newest checkpoint (every `SAVE_FREQ=10` steps); note that a resumed Stage II run
restarts the environment's task sampling, which weakens its pairing with the other arms of the same seed.

What one run does (`run_arm.sh 3b-ws 120 pr-opd 1`):
1. copies `ckpts/grpo-3b-ws/global_step_120` into the run and resumes it (`train.sh --init`, step counter 120,
   optimizer restored), env.seed = vLLM seed = 1, KL reference = base model;
2. trains to step 200; checkpoints at 150 and 200 are protected (`train.sh --keep_steps "150 200"`), other
   checkpoints are rotated (only the newest is kept);
3. evaluates the step-150 and step-200 checkpoints (WebShop 500 goals; ALFWorld seen 140 is the in-training pass);
4. deletes the optimizer shards and the copied start checkpoint (`KEEP_FINAL_CKPTS=0` also deletes the model shards).

Change the protocol in `site.sh`, not in the scripts: `START_STEPS_ALF`, `START_STEPS_WS`, `END_STEPS` (every END
step must be a multiple of `TEST_FREQ=25` and `SAVE_FREQ=10`), `SEEDS`, `ARMS_DEFAULT`, `EVAL_UNSEEN=1`,
`OPRD_COEF` (representation-loss coefficient of the pr-opd arms).

## 6. Results

```bash
python experiments/collect_results.py            # uses $WORK from site.sh / the environment
python experiments/check_rollout_diversity.py $WORK/logs/*-from*-s*.log
```
`summary.md` has one table per (setting, branch point, step): the branch point itself, then every arm as mean ±
std over seeds with the per-seed values and the paired difference to `grpo` (same setting, start and seed) with
a 95% t-interval. `runs.csv` holds every number, `curves.csv` the per-update training success, the in-training
validation and the distillation diagnostics (`oprd/loss`, `oprd/raw_loss` = distance of the policy's own,
unprojected states to the teacher's, `actor/sdl_loss`, ...).

## Evaluation protocol

All arms and seeds are evaluated on **identical task sets, each task exactly once**; decoding as in training
validation (sampling, temperature 0.4), one episode per task, the student never sees a skill.

- **ALFWorld seen (140)**: the `eval_in_distribution` split. With `ALFWORLD_DET_EVAL=1` the i-th of the 140
  validation slots always plays the i-th game of the sorted list (`agent_system/environments/env_package/alfworld/envs.py`),
  independent of the seed; at most 50 steps per episode. The in-training validation pass at step 150/200 is the
  reported number (overall success over the 140 games, plus the six task types Pick 35 / Look 13 / Clean 27 /
  Heat 16 / Cool 25 / Pick2 24). Without the switch the environment samples games with replacement (a 128-episode
  pass then covers only ~90 distinct games).
- **ALFWorld unseen (134)**: `eval_out_of_distribution`, optional (`EVAL_UNSEEN=1`).
- **WebShop (500)**: the 500 test goals (goal ids 0–499; training uses 500+). A separate evaluation process plays
  4 batches of 125 with `WEBSHOP_DET_EVAL=1 WEBSHOP_EVAL_ADVANCE=1` (goals 0–124, 125–249, 250–374, 375–499),
  at most 15 steps per episode. **Acc** = fraction of episodes whose purchase has reward 1.0; **Score** = mean
  WebShop reward (0–1, partial matches count; no purchase = 0) × 100. The in-training validation on goals 0–124
  is for curves only.

## Troubleshooting

| symptom | fix |
|---|---|
| CUDA OOM at vLLM init or in the actor update | lower `GPU_UTIL_*` by 0.05–0.1 in `site.sh`, then halve the `MICRO_*` of that setting (re-run its finished runs) |
| Stage II jobs pending with `DependencyNeverSatisfied` | the trunk failed: fix it, `scancel` the pending jobs, re-run `submit_all.sh` |
| `Stage I not finished` / `no optimizer state` | `done/grpo-<setting>` or a branch-point checkpoint is missing or incomplete (disk full while saving) |
| every Stage II arm below its start, success counts always multiples of 8 | identical rollouts inside GRPO groups: make sure `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` skips `seed` in the per-request SamplingParams; `check_rollout_diversity.py` flags such logs |
| job hangs before the workers start | Ray CPU budget too small for the WebShop env workers; `RAY_CPUS_MIN` (default 48) is booked regardless of the allocation |
| `ROCR_VISIBLE_DEVICES` error from Ray | `lib.sh` unsets it; otherwise unset it in `site.sh` |
| WebShop: JVM / `jnius` errors | `JAVA_HOME_WS` must contain `lib/server/libjvm.so` (empty = the conda env's openjdk) |
| WebShop: missing `items_shuffle_1000.json` or indexes | `bash experiments/download_webshop.sh` |
| ALFWorld: game files not found | `ALFWORLD_DATA` must contain `json_2.1.1/` and `logic/` (`setup.sh --check`) |
| textworld `GLIBC*` / `GLIBCXX*` errors from `libdownward.so` | rebuild on the target machine: `pip install --no-binary :all: --force-reinstall --no-deps fast-downward-textworld` |
| textworld `Directory not empty` | `$TMPDIR` is on a network filesystem: `export TMPDIR=/tmp` in `site.sh` |
| Ray AF_UNIX path longer than 107 bytes | `export RAY_TMPDIR_BASE=/tmp` in `site.sh` |
| `[job] ERROR: N GPUs visible, expected M` | SLURM gave fewer GPUs; the job requeues itself (3×, then hold); `SB_EXTRA=--exclusive` helps |
| redo one run | delete `done/<run>`, `ckpts/<run>`, `evals/<run>`, move `logs/<run>.log` aside, re-run the queue |

# PR-OPD: Privileged Representation On-Policy Distillation for Agentic RL

This repository contains the code for **PR-OPD**, a self-distillation method for
multi-turn LLM agents, together with the two baselines it is compared against
(**GRPO** and **SDAR**) on **ALFWorld** and **WebShop**.

The code is built on the [SDAR](https://github.com/ZJU-REAL/SDAR) codebase
(itself built on [verl-agent](https://github.com/langfengQ/verl-agent) / [verl](https://github.com/volcengine/verl)).
Everything that is not needed for GRPO, SDAR and PR-OPD on the two environments has been removed.

## Method

The policy is trained with GRPO (plus a KL term to the reference policy). On top of
that, at every training step the **current policy itself** reads each of its own
completed trajectories and writes a short *hindsight skill* (what worked, what went
wrong, what to do next time). That skill is prepended to the ordinary context to form a
**privileged teacher branch that shares weights with the student**. Instead of distilling
the teacher's next-token distribution (as SDAR does), PR-OPD aligns the student's
**hidden states** with the teacher's hidden states at every layer, on the response tokens
of the sampled trajectories. The optional **+MLP** variant passes the student hidden states
through a trainable per-layer MLP before the alignment.

Implementation map:

| component | file |
|---|---|
| entry point / config translation | `verl/trainer/main_pr_opd.py` |
| hindsight-skill generation, teacher context | `verl/trainer/ppo/pr_opd_ray_trainer.py`, `verl/trainer/ppo/pr_opd_utils.py` |
| representation loss, layer selection | `verl/trainer/ppo/oprd_utils.py` |
| per-layer MLP projection head (+MLP variant) | `verl/trainer/ppo/oprd_projector.py` |
| hidden-state extraction and the extra loss term in the actor | `verl/workers/actor/dp_actor.py` (`return_hidden_layers`, `update_policy`) |
| projector checkpointing | `verl/workers/fsdp_workers.py` (`oprd_proj_save/load`) |
| GRPO baseline | `verl/trainer/main_ppo.py` (upstream verl-agent) |
| SDAR baseline | `verl/trainer/main_sdar.py` (upstream SDAR; static skills in `skills/`) |

`verl/trainer/ppo/rlsd_ray_trainer.py` and `skillsd_ray_trainer.py` are the SDAR-family
trainer skeleton that both SDAR and PR-OPD inherit from; they are not separate baselines here.

## Installation

Follow the SDAR instructions (summarised here).

```bash
conda create -n pr-opd python==3.12 -y && conda activate pr-opd
pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

**ALFWorld**
```bash
pip3 install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
alfworld-download -f            # game files -> ~/.cache/alfworld (set ALFWORLD_DATA to override)
```

**WebShop** (needs Python 3.10 and a JVM for the Lucene search engine; see the
[SDAR README](https://github.com/ZJU-REAL/SDAR#2-webshop) for the full recipe)
```bash
conda create -n pr-opd-webshop python==3.10 -y && conda activate pr-opd-webshop
cd agent_system/environments/env_package/webshop/webshop && ./setup.sh -d all && cd -
pip3 install -e . && pip3 install vllm==0.11.0 flash-attn==2.7.4.post1 --no-build-isolation
```
WebShop env workers each start a JVM. If memory is tight, `WEBSHOP_SHARED_SEARCH=1 WEBSHOP_LIGHT_WORKER=1`
switches to one shared search actor and torch-free env workers (see `agent_system/environments/env_package/webshop/envs.py`).

**Data.** The trainers read a tiny parquet "dataset" that only fixes the batch size and modality
(the tasks come from the environments):
```bash
python examples/data_preprocess/prepare.py --mode text --train_data_size 16 --val_data_size 128
# -> ~/data/verl-agent/text/{train,test}.parquet   (override with --data_dir below)
```

## Training

One launcher covers all three methods and both environments:

```bash
bash scripts/train.sh --method {grpo|sdar|pr-opd} --env {alfworld|webshop} --model <hf-id-or-path> [options] [-- hydra overrides]
```

The two switches that define a PR-OPD run:

| switch | values | meaning |
|---|---|---|
| `--init` | `base` (default) or `path/to/run/global_step_N` | start from the base model, or branch from a checkpoint of a previous run (typically a GRPO run); the step counter continues from `N` |
| `--proj` | `none` (default) or `mlp` | PR-OPD without a projection head (main method) or with the per-layer MLP head |

Other options: `--coef` (representation-loss weight, default 0.01), `--steps` (default 150),
`--micro` (actor micro batch per GPU: default 32 on ALFWorld / 8 on WebShop; halve for 7B models),
`--gpus`, `--tp`, `--seed` (env seed, default 0), `--save_freq`/`--test_freq` (10 / 5),
`--ckpt_root` (default `./checkpoints`), `--name`. Run `bash scripts/train.sh --help` for the list.

Typical sequence on ALFWorld with Qwen2.5-3B-Instruct (8 GPUs):

```bash
# baselines
bash examples/grpo/run_alfworld_3b.sh                 # -> checkpoints/grpo-alfworld-qwen2.5-3b-instruct/
bash examples/sdar/run_alfworld_3b.sh

# PR-OPD from the base model
bash examples/pr_opd/run_alfworld_3b_from_base.sh                 # no projection head
bash examples/pr_opd/run_alfworld_3b_from_base.sh --proj mlp      # +MLP

# PR-OPD branched from the GRPO checkpoint at step 100 and trained to step 150
bash examples/pr_opd/run_alfworld_3b_from_grpo100.sh
bash examples/pr_opd/run_alfworld_3b_from_grpo100_mlp.sh
# the matching control: continue plain GRPO from the same checkpoint
bash scripts/train.sh --method grpo --env alfworld --model Qwen/Qwen2.5-3B-Instruct \
     --init checkpoints/grpo-alfworld-qwen2.5-3b-instruct/global_step_100
```

Replace `alfworld` by `webshop` for WebShop (`examples/*/run_webshop_3b*.sh`). Any model that
`transformers`/`vLLM` support can be passed to `--model` (Qwen2.5, Qwen3, OLMo-3, ...);
for Qwen3 models the launcher passes `enable_thinking=False` (the setting used in the paper);
`--thinking` keeps the template default.

Hyper-parameters follow the SDAR paper scripts: 16 tasks x 8 rollouts per step, lr 1e-6,
KL coefficient 0.01 (`low_var_kl`), invalid-action penalty 0.1, 150 steps, ALFWorld
`max_steps=50` / WebShop `max_steps=15`, validation on 128 tasks (temperature 0.4) every 5 steps.
PR-OPD defaults: all layers aligned, L2-normalised hidden states, coefficient 0.01;
+MLP uses a 512-wide hidden layer with its own AdamW at lr 1e-4.

All PR-OPD options are hydra overrides under `+algorithm.pr_opd.*`; the full list is in the
docstring of `verl/trainer/main_pr_opd.py` (including `skill_source=shuffled`, the mismatched-skill
control, and `oprd_pos_mask=success`, which restricts the representation loss to successful trajectories).

## Evaluation

Validation during training is logged every `--test_freq` steps (`val/success_rate`, plus per-task-type
rates on ALFWorld and `val/webshop_task_score` on WebShop). To evaluate a saved checkpoint:

```bash
bash scripts/eval.sh --ckpt checkpoints/<run>/global_step_150 --env alfworld --model Qwen/Qwen2.5-3B-Instruct --seed 1
bash examples/eval_130_140_150.sh checkpoints/<run> alfworld Qwen/Qwen2.5-3B-Instruct   # steps 130/140/150 with env.seed 1/2/3
```

Checkpoints are standard verl FSDP checkpoints (`global_step_N/actor/`); `scripts/model_merger.py`
merges them into a Hugging Face model. PR-OPD+MLP additionally stores `oprd_projectors.pt`
next to the actor shards; when branching from a checkpoint that has no projectors (a GRPO run)
the launcher passes `+algorithm.pr_opd.proj_allow_missing=True` so they start from scratch.



## Acknowledgements

Built on [SDAR](https://github.com/ZJU-REAL/SDAR), [verl-agent](https://github.com/langfengQ/verl-agent) and [verl](https://github.com/volcengine/verl).
The static skill libraries under `skills/` are from the SDAR release. License: Apache 2.0 (see `LICENSE`, `Notice.txt`).

#!/bin/bash
# Paper protocol: evaluate the step-130/140/150 checkpoints of a run with env.seed 1/2/3.
#   bash examples/eval_130_140_150.sh checkpoints/<run> alfworld Qwen/Qwen2.5-3B-Instruct
RUN=${1:?run dir}; ENV=${2:?env}; MODEL=${3:?model}
seed=0; for step in 130 140 150; do seed=$((seed+1)); bash scripts/eval.sh --ckpt "$RUN/global_step_$step" --env "$ENV" --model "$MODEL" --seed $seed; done

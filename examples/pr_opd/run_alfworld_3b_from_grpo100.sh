#!/bin/bash
# PR-OPD branched from the step-100 checkpoint of the GRPO run above and trained to step 150.
# 1) bash examples/grpo/run_alfworld_3b.sh                      -> checkpoints/grpo-alfworld-qwen2.5-3b-instruct/global_step_100
# 2) this script. Compare against "continue GRPO from the same checkpoint":
#       bash scripts/train.sh --method grpo --env alfworld --model Qwen/Qwen2.5-3B-Instruct \
#            --init checkpoints/grpo-alfworld-qwen2.5-3b-instruct/global_step_100
INIT=${INIT:-checkpoints/grpo-alfworld-qwen2.5-3b-instruct/global_step_100}
bash scripts/train.sh --method pr-opd --env alfworld --model Qwen/Qwen2.5-3B-Instruct --init "$INIT" --proj none --steps 150 "$@"

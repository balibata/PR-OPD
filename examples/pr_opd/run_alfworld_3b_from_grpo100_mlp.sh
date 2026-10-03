#!/bin/bash
# PR-OPD + per-layer MLP projection head, branched from GRPO step 100 (ALFWorld, Qwen2.5-3B-Instruct).
INIT=${INIT:-checkpoints/grpo-alfworld-qwen2.5-3b-instruct/global_step_100}
bash scripts/train.sh --method pr-opd --env alfworld --model Qwen/Qwen2.5-3B-Instruct --init "$INIT" --proj mlp --steps 150 "$@"

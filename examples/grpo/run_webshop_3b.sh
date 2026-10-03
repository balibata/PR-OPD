#!/bin/bash
# GRPO baseline, Qwen2.5-3B-Instruct, webshop (SDAR paper protocol, 150 steps, 8 GPUs).
bash scripts/train.sh --method grpo --env webshop --model Qwen/Qwen2.5-3B-Instruct "$@"

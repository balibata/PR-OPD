#!/bin/bash
# SDAR baseline (static skill library + gated token-level self-distillation), Qwen2.5-3B-Instruct, alfworld.
bash scripts/train.sh --method sdar --env alfworld --model Qwen/Qwen2.5-3B-Instruct "$@"

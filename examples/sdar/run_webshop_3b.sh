#!/bin/bash
# SDAR baseline (static skill library + gated token-level self-distillation), Qwen2.5-3B-Instruct, webshop.
bash scripts/train.sh --method sdar --env webshop --model Qwen/Qwen2.5-3B-Instruct "$@"

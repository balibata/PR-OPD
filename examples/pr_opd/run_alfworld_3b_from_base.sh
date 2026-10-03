#!/bin/bash
# PR-OPD (no projection head) trained from the base model, Qwen2.5-3B-Instruct, alfworld.
# Add "--proj mlp" for the +MLP variant.
bash scripts/train.sh --method pr-opd --env alfworld --model Qwen/Qwen2.5-3B-Instruct --proj none "$@"

#!/usr/bin/env python3
"""Detect identical rollouts inside GRPO groups from a training log.

Each step trains on B groups x n samples (here 16 x 8 = 128 episodes). If the n samples of a group are
identical (e.g. a fixed per-request vLLM seed), every group is all-success or all-fail, so the number of
successful episodes per step is always a multiple of n and GRPO gets no learning signal.
With diverse sampling only ~1 step in n has such a count by chance.

usage: python check_rollout_diversity.py <train log or .log.gz> [...] [--n 8] [--episodes 128]
exit 1 if any log looks collapsed (>= 60% of >= 5 steps are multiples of n).
"""
import gzip, re, sys

args = sys.argv[1:]
n, eps = 8, 128
if "--n" in args:
    i = args.index("--n"); n = int(args[i + 1]); del args[i:i + 2]
if "--episodes" in args:
    i = args.index("--episodes"); eps = int(args[i + 1]); del args[i:i + 2]
if not args:
    sys.exit(__doc__)
bad = 0
for path in args:
    op = gzip.open if path.endswith(".gz") else open
    counts = []
    with op(path, "rt", errors="replace") as f:
        for line in f:
            if "episode/success_rate:" not in line or not re.search(r"step:\d+ ", line):
                continue
            m = re.search(r"episode/success_rate:(-?[0-9.]+)", line)
            counts.append(round(float(m.group(1)) * eps))
    if len(counts) < 5:
        print(f"[diversity] {path}: only {len(counts)} steps logged, cannot judge"); continue
    frac = sum(c % n == 0 for c in counts) / len(counts)
    ok = frac < 0.6
    bad += not ok
    print(f"[diversity] {'OK ' if ok else 'BAD'} {path}: {frac:.0%} of {len(counts)} steps have a success count "
          f"divisible by {n} (expected ~{1/n:.0%} with diverse sampling); last counts {counts[-8:]}")
sys.exit(1 if bad else 0)

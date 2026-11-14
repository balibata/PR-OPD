#!/bin/bash
# Put the placeholder parquet "datasets" into $WORK/data (they only fix batch sizes; tasks come from the envs).
# Each validation dir must have EXACTLY val_n rows, otherwise the trainer runs several validation
# passes and the deterministic slots are replayed.
#   v140: ALFWorld seen split (eval_in_distribution, 140 games, every game once)
#   v134: ALFWorld unseen split (eval_out_of_distribution, 134 games)
#   v125: WebShop in-training validation (test goals 0..124)
#   v500: WebShop final evaluation: 500 rows, val_batch_size 125 -> 4 batches = all 500 goals in one process
# Pre-built copies ship in experiments/placeholder_data (no internet, no python needed). Only if they are
# missing is examples/data_preprocess/prepare.py used (downloads a tiny dataset once).
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd); source "$KIT/config.sh"; source "$KIT/lib.sh"
for n in 140 134 125 500; do
  out=$WORK/data/v$n
  if [ -f "$out/text/test.parquet" ]; then echo "[prep_data] $out exists"; continue; fi
  if [ -f "$KIT/placeholder_data/v$n/text/test.parquet" ]; then
    mkdir -p "$out"; cp -r "$KIT/placeholder_data/v$n/text" "$out/"; echo "[prep_data] $out <- placeholder_data/v$n"; continue
  fi
  activate_env "$CONDA_ENV_ALF"; cd "$REPO"
  python examples/data_preprocess/prepare.py --mode text --train_data_size 16 --val_data_size $n --local_dir "$out"
done
activate_env "$CONDA_ENV_ALF"
python - "$WORK/data" <<'PY'
import sys, pandas as pd
for n in (140, 134, 125, 500):
    d = f"{sys.argv[1]}/v{n}/text"
    tr, te = pd.read_parquet(f"{d}/train.parquet"), pd.read_parquet(f"{d}/test.parquet")
    assert len(tr) == 16 and len(te) == n, (d, len(tr), len(te))
    print(f"[prep_data] {d}: train {len(tr)} rows, test {len(te)} rows")
PY

#!/bin/bash
# One-time setup on a new cluster: fetch what the runs need, write experiments/site.sh, copy the placeholder
# data and check the layout. Run it on a login node AFTER the conda env exists (experiments/README.md), from anywhere:
#
#   bash experiments/setup.sh --base_3b <HF id | local dir> [--base_17b <HF id | local dir>]
#   bash experiments/setup.sh --base_3b ... --assets /big/disk/pr-opd-assets --work /big/disk/pr-opd-work
#   bash experiments/setup.sh --check                    # only re-check the layout
#
# Base models: a local dir is linked, an HF repo id is downloaded (huggingface_hub) into <assets>/models/.
# ALFWorld: official `alfworld-download` into <assets>/data/alfworld. JDK: openjdk=11 from conda-forge, installed
# into the conda env. WebShop small product set (1000 items) + Lucene indexes: download_webshop.sh if missing.
set -euo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$KIT/.." && pwd)
ASSETS=$REPO/assets; WORK_DIR=$REPO/work; B3=""; B17=""; CHECK_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --assets) ASSETS=$2; shift 2;; --work) WORK_DIR=$2; shift 2;;
    --base_3b) B3=$2; shift 2;; --base_17b) B17=$2; shift 2;; --check) CHECK_ONLY=1; shift;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown option $1"; exit 1;;
  esac
done
mkdir -p "$ASSETS" "$WORK_DIR"; ASSETS=$(cd "$ASSETS" && pwd); WORK_DIR=$(cd "$WORK_DIR" && pwd)
ALF_DIR=$ASSETS/data/alfworld
say(){ echo "[setup] $*"; }

# ---------------------------------------------------------------- conda / python
CONDA_SH_GUESS=""
if command -v conda >/dev/null 2>&1; then cb=$(conda info --base 2>/dev/null || true); [ -n "$cb" ] && CONDA_SH_GUESS="$cb/etc/profile.d/conda.sh"; fi
# shellcheck disable=SC1091
[ -f "$KIT/site.sh" ] && source "$KIT/site.sh"
CONDA_SH=${CONDA_SH:-$CONDA_SH_GUESS}; CONDA_ENV=${CONDA_ENV:-pr-opd}
if [ -n "${CONDA_MODULE:-}" ]; then module load "$CONDA_MODULE"; fi
if [ -n "$CONDA_SH" ] && [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"; conda activate "$CONDA_ENV" 2>/dev/null || say "WARNING: conda env '$CONDA_ENV' not found (create it first)"
fi
PY=$(command -v python || command -v python3)

# fetch_model <src> <dest>: link a local dir or download an HF repo id
fetch_model(){
  local src=$1 dest=$2
  mkdir -p "$(dirname "$dest")"
  if [ -d "$src" ]; then ln -sfn "$(cd "$src" && pwd)" "$dest"; say "base model $dest -> $src (linked)"
  elif [ -f "$dest/config.json" ]; then say "base model already in $dest"
  else say "downloading $src -> $dest"
    "$PY" -c "import sys; from huggingface_hub import snapshot_download as d; d(repo_id=sys.argv[1], local_dir=sys.argv[2])" "$src" "$dest"
  fi
}

if [ "$CHECK_ONLY" = 0 ]; then
  [ -n "$B3" ] || [ -n "$B17" ] || [ -f "$KIT/site.sh" ] || { echo "give --base_3b and/or --base_17b (HF repo id or local dir)"; exit 1; }
  [ -n "$B3" ] && fetch_model "$B3" "$ASSETS/models/base_3b"
  [ -n "$B17" ] && fetch_model "$B17" "$ASSETS/models/base_17b"
  if [ ! -d "$ALF_DIR/json_2.1.1" ]; then
    command -v alfworld-download >/dev/null || { echo "alfworld-download not found: activate the conda env (pip install alfworld==0.4.2)"; exit 1; }
    say "alfworld-download -> $ALF_DIR"; mkdir -p "$ALF_DIR"; ALFWORLD_DATA=$ALF_DIR alfworld-download
  fi
  if [ ! -f "${CONDA_PREFIX:-/nonexistent}/lib/jvm/lib/server/libjvm.so" ] && [ ! -f "${CONDA_PREFIX:-/nonexistent}/lib/server/libjvm.so" ]; then
    say "installing openjdk=11 into the conda env"; conda install -y -c conda-forge openjdk=11
  fi
  # ---------------------------------------------------------------- site.sh
  if [ ! -f "$KIT/site.sh" ]; then
    cat > "$KIT/site.sh" <<EOF
# Written by setup.sh on $(date -Is). Everything here overrides config.sh (see the SLURM block there).
WORK=$WORK_DIR
BASE_3B=$ASSETS/models/base_3b
BASE_17B=$ASSETS/models/base_17b
export ALFWORLD_DATA=$ALF_DIR
JAVA_HOME_WS=            # empty = the JDK inside the conda env
CONDA_MODULE=${CONDA_MODULE:-}
CONDA_SH=$CONDA_SH
CONDA_ENV=$CONDA_ENV
# GPUs per job and the settings to run by default
# GPUS=8
# SETTINGS_DEFAULT="3b-alf 3b-ws"
# SLURM (fill in what your cluster needs; empty = not passed to sbatch)
SB_PARTITION=
SB_ACCOUNT=
SB_QOS=
# SB_CPUS=64 SB_MEM=400G SB_TIME=72:00:00 SB_EXCLUDE= SB_EXTRA=
LANES=\${LANES:-2}   # at most this many jobs at the same time
EOF
    say "wrote $KIT/site.sh (check CONDA_SH / CONDA_ENV, GPUS and the SLURM lines)"
  else
    say "$KIT/site.sh exists, not overwritten"
  fi
  W=$REPO/agent_system/environments/env_package/webshop/webshop
  if [ ! -f "$W/data/items_shuffle_1000.json" ] || [ ! -d "$W/search_engine/indexes" ]; then
    say "WebShop product set / indexes missing -> download_webshop.sh"; bash "$KIT/download_webshop.sh"
  fi
  bash "$KIT/prep_data.sh" | grep -v '^\[prep_data\] .* exists$' || true
fi

# ---------------------------------------------------------------- checks
# shellcheck disable=SC1091
source "$KIT/config.sh"
ok=1
chk(){ if [ -e "$1" ]; then echo "  ok   $1"; else echo "  MISSING $1"; ok=0; fi; }
say "checking layout"
for b in "$BASE_3B" "$BASE_17B"; do
  if [ -f "$b/config.json" ]; then echo "  ok   $b"; else echo "  --   $b (not set up; only needed for its settings)"; fi
done
chk "$ALFWORLD_DATA/json_2.1.1/valid_seen"; chk "$ALFWORLD_DATA/json_2.1.1/valid_unseen"; chk "$ALFWORLD_DATA/logic"
J=${JAVA_HOME_WS:-${CONDA_PREFIX:-/nonexistent}/lib/jvm}; [ -n "${JAVA_HOME_WS:-}" ] || [ -f "$J/lib/server/libjvm.so" ] || J=${CONDA_PREFIX:-/nonexistent}
chk "$J/lib/server/libjvm.so"
W=$REPO/agent_system/environments/env_package/webshop/webshop
chk "$W/data/items_shuffle_1000.json"; chk "$W/data/items_ins_v2_1000.json"; chk "$W/data/items_human_ins.json"; chk "$W/search_engine/indexes"
for n in 140 134 125 500; do chk "$WORK/data/v$n/text/test.parquet"; done
n_seen=$(find "$ALFWORLD_DATA/json_2.1.1/valid_seen" -name 'game.tw-pddl' 2>/dev/null | wc -l)
echo "  info ALFWorld valid_seen games: $n_seen (expected 140)"; [ "$n_seen" = 140 ] || ok=0
if [ "$ok" = 1 ]; then say "layout OK"; else say "some files are missing (see above)"; exit 1; fi

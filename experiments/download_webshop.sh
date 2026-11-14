#!/bin/bash
# FALLBACK ONLY: the WebShop small product set (1000 items) and its Lucene indexes already ship inside the
# code. Run this only if setup.sh reports them missing. It follows the official WebShop setup.sh
# (agent_system/environments/env_package/webshop/webshop/setup.sh, "-d small"), without the pip/conda installs:
# downloads the 3 json files with gdown (Google Drive) and rebuilds the indexes with pyserini (needs the JVM).
#   bash experiments/download_webshop.sh
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd); source "$KIT/config.sh"; source "$KIT/lib.sh"
ENV_KEY=webshop; activate_env "$CONDA_ENV_WS"
W=$REPO/agent_system/environments/env_package/webshop/webshop
command -v gdown >/dev/null || pip install gdown
mkdir -p "$W/data"; cd "$W/data"
[ -f items_shuffle_1000.json ] || gdown "https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib"   # items_shuffle_1000
[ -f items_ins_v2_1000.json ]  || gdown "https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu"   # items_ins_v2_1000
[ -f items_human_ins.json ]    || gdown "https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O"   # items_human_ins
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || python -m spacy download en_core_web_sm
cd "$W/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k indexes
python convert_product_file_format.py
bash run_indexing.sh
ls -d indexes indexes_1k indexes_100 indexes_100k
echo "[download_webshop] done"

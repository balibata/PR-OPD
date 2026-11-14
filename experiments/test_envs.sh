#!/bin/bash
# Quick functional tests of the environments and the GPU stack, before any training.
#   bash experiments/test_envs.sh alfworld     # CPU: game counts (train / seen 140 / unseen 134) + reset/step one game
#   bash experiments/test_envs.sh webshop      # CPU: JVM + Lucene search + reset/step one WebShop session
#   bash experiments/test_envs.sh gpu [3b|17b] # GPU node: torch/vllm/flash-attn + 1 short vLLM generation with that base model
#   bash experiments/test_envs.sh all          # alfworld + webshop (+ gpu if a GPU is visible)
# alfworld/webshop are light and may run on a login node; gpu must run inside a GPU job.
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd); source "$KIT/config.sh"; source "$KIT/lib.sh"
WHAT=${1:-all}
cd "$REPO"
case "${2:-3b}" in
  3b)  BASE_MODEL_TEST=$BASE_3B;  GPU_MEM_NEED=75;;   # 3B: 80 GB cards
  17b) BASE_MODEL_TEST=$BASE_17B; GPU_MEM_NEED=38;;   # 1.7B: >= 40 GB cards (80 GB recommended)
  *) die "model must be 3b or 17b";;
esac

t_alfworld(){
  ENV_KEY=alfworld; activate_env "$CONDA_ENV_ALF"
  echo "[test] ALFWorld (ALFWORLD_DATA=$ALFWORLD_DATA)"
  python - <<'PY'
import os, yaml, time
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
cfg = yaml.safe_load(open("agent_system/environments/env_package/alfworld/configs/config_tw.yaml"))
expect = {"train": None, "eval_in_distribution": 140, "eval_out_of_distribution": 134}
for split, n in expect.items():
    env = get_environment(cfg["env"]["type"])(cfg, train_eval=split)
    print(f"  {split:26s} games = {env.num_games}" + ("" if n is None else f"  (expected {n})"))
    assert n is None or env.num_games == n, f"{split}: {env.num_games} != {n}"
env = get_environment(cfg["env"]["type"])(cfg, train_eval="eval_in_distribution")
env.game_files = sorted(env.game_files)[:1]; env.num_games = 1
e = env.init_env(batch_size=1); e.seed(0)
t = time.time(); obs, info = e.reset()
print("  reset ok:", obs[0].replace("\n", " ")[:160], "...")
obs, scores, dones, infos = e.step(["look"])
print("  step 'look' ok:", obs[0].replace("\n", " ")[:120], f"... ({time.time() - t:.1f}s)")
print("[test] ALFWorld OK")
PY
}

t_webshop(){
  ENV_KEY=webshop; activate_env "$CONDA_ENV_WS"
  echo "[test] WebShop (JAVA_HOME=$JAVA_HOME)"
  python - <<'PY'
import os, sys, time
root = os.path.abspath("agent_system/environments/env_package/webshop/webshop")
sys.path.append(root)
import gym
from web_agent_site.envs import WebAgentTextEnv  # noqa: F401  (registers WebAgentTextEnv-v0)
t = time.time()
env = gym.make("WebAgentTextEnv-v0", disable_env_checker=True, observation_mode="text", num_products=None,
               human_goals=False, file_path=os.path.join(root, "data/items_shuffle_1000.json"),
               attr_path=os.path.join(root, "data/items_ins_v2_1000.json"), seed=0)
goals = env.server.goals if hasattr(env, "server") else env.unwrapped.server.goals
print(f"  env built in {time.time() - t:.1f}s; goals = {len(goals)} (synthetic goals as in training; the first 500 are the test goals)")
assert len(goals) >= 500
obs, _ = env.reset(session=0)
print("  reset ok:", str(obs).replace("\n", " ")[:160], "...")
obs, reward, done, info = env.step("search[blue shirt]")
n = str(obs).count("[SEP]")
print("  search ok:", str(obs).replace("\n", " ")[:160], "...")
assert "Back to Search" in str(obs) or n > 5, "search returned no results page"
print("[test] WebShop OK")
PY
}

t_gpu(){
  ENV_KEY=alfworld; activate_env "$CONDA_ENV_ALF"
  echo "[test] GPU stack + vLLM generation with $BASE_MODEL_TEST"
  # run from a file, not stdin: vLLM's spawned engine process re-imports __main__ and cannot open "<stdin>"
  local T; T=$(mktemp "${TMPDIR:-/tmp}/propd_gputest_XXXX.py"); cat > "$T" <<'PY'
import sys, torch, vllm, flash_attn, transformers
# main guard: CUDA is initialised below, so vLLM spawns its engine process and re-imports this file
if __name__ == "__main__":
    print("  torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    print("  vllm", vllm.__version__, "flash_attn", flash_attn.__version__, "transformers", transformers.__version__)
    assert torch.cuda.is_available()
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    need = float(sys.argv[2])
    print(f"  GPU 0 memory: {mem:.0f} GiB" + ("" if mem >= need else f"  <-- these runs need >= {need:.0f} GiB cards"))
    from vllm import LLM, SamplingParams
    llm = LLM(model=sys.argv[1], gpu_memory_utilization=0.4, max_model_len=2048, enforce_eager=True)
    out = llm.generate(["You are in a kitchen. Your task is to put a clean mug on the table. What do you do first?"],
                       SamplingParams(temperature=0.4, max_tokens=48, seed=0))
    print("  generation ok:", out[0].outputs[0].text.replace("\n", " ")[:160])
    print("[test] GPU OK")
PY
  local rc=0; python "$T" "$BASE_MODEL_TEST" "$GPU_MEM_NEED" || rc=$?; rm -f "$T"; return $rc
}

case "$WHAT" in
  alfworld) t_alfworld;;
  webshop) t_webshop;;
  gpu) t_gpu;;
  all) t_alfworld; t_webshop
       if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then t_gpu; else echo "[test] no GPU visible: skipped the gpu test (run it in a GPU job)"; fi;;
  *) die "usage: test_envs.sh alfworld|webshop|gpu|all [3b|17b]";;
esac

# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lazy re-exports (PEP 562): importing this package must stay lightweight.
# Env worker actors (e.g. AlfworldWorker) live in submodules; when ray spawns
# a worker process, unpickling the actor class imports the parent package.
# An eager env_manager import here would drag torch (~0.4GB RSS) into every
# one of the potentially 100+ env worker processes.
def __getattr__(name):
    if name in ("EnvironmentManagerBase", "make_envs"):
        from agent_system.environments import env_manager
        return getattr(env_manager, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
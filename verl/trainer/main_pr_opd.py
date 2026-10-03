"""
Main entry point for PR-OPD (Privileged Representation On-Policy Distillation).

The policy is trained with GRPO (+ KL to the reference) and, on top of that, a
representation-space self-distillation term: at every step the current policy
writes a hindsight skill for each of its own trajectories, the skill is
prepended to the context to form a privileged teacher branch (same weights),
and the student's hidden states are aligned with the teacher's.

All PR-OPD options live under ``+algorithm.pr_opd.*`` (all optional):

    oprd_coef:          weight of the representation-distillation loss     (default 0.01;
                        0 disables PR-OPD and the run degenerates to GRPO)
    oprd_layers:        which hidden layers to align: "all" | "last" | "1,5,9" (default "all")
    oprd_last_k:        supervise only the last k response tokens (<=0: all) (default -1)
    oprd_normalize:     L2-normalise hidden states before the MSE           (default True)
    oprd_gate:          weight tokens by the SDAR sigmoid gate              (default False)
    oprd_proj_mode:     none | mlp  -- the "+MLP" variant                   (default none)
    oprd_proj_hidden:   MLP hidden width                                     (default 512)
    oprd_proj_lr:       MLP learning rate                                    (default 1e-4)
    proj_allow_missing: start MLP projectors from scratch when the resume
                        checkpoint has none (set this when initialising from
                        a GRPO checkpoint)                                   (default False)
    coef / gate_beta:   optional token-level (output-space) gated distillation
                        term from SDAR on the same teacher (default coef 0 = off)
    skill_source:       self | static | shuffled                             (default self)
    skills_dir / skill_all: static skill library (skill_source=static only)
    oprd_pos_mask:      none | success                                       (default none)
    max_obs_chars / max_action_chars / max_turns / max_skill_chars /
    skill_think / skill_strip_think / log_skill_examples: trajectory
                        serialisation and analyzer-output handling
"""

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_pr_opd(config)


def run_pr_opd(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = PROPDTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class PROPDTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        cfg = config.algorithm.get("pr_opd", {})
        with open_dict(config):
            # Optional output-space (token-level, SDAR-style gated) distillation on the
            # same privileged teacher. Off by default: PR-OPD distils representations.
            sdar_coef = cfg.get("coef", 0.0)
            config.actor_rollout_ref.actor.use_sdar_loss = sdar_coef > 0
            config.actor_rollout_ref.actor.sdar_loss_coef = sdar_coef
            config.actor_rollout_ref.actor.sdar_gate_beta = cfg.get("gate_beta", 5.0)

            # Representation-space distillation (PR-OPD loss on hidden states).
            oprd_coef = cfg.get("oprd_coef", 0.01)
            config.actor_rollout_ref.actor.use_oprd_loss = oprd_coef > 0
            config.actor_rollout_ref.actor.oprd_loss_coef = oprd_coef
            config.actor_rollout_ref.actor.oprd_layers = cfg.get("oprd_layers", "all")
            config.actor_rollout_ref.actor.oprd_last_k = cfg.get("oprd_last_k", -1)
            config.actor_rollout_ref.actor.oprd_normalize = cfg.get("oprd_normalize", True)
            config.actor_rollout_ref.actor.oprd_gate = cfg.get("oprd_gate", False)
            config.actor_rollout_ref.actor.oprd_gate_beta = cfg.get("gate_beta", 5.0)
            # Projection head: none (default, main method) | mlp (the "+MLP" variant)
            proj_mode = cfg.get("oprd_proj_mode", None)
            if proj_mode in ("", "none", "None", False):
                proj_mode = None
            config.actor_rollout_ref.actor.oprd_proj_mode = proj_mode
            config.actor_rollout_ref.actor.oprd_proj_hidden = cfg.get("oprd_proj_hidden", 512)
            config.actor_rollout_ref.actor.oprd_proj_lr = cfg.get("oprd_proj_lr", 1e-4)
            config.actor_rollout_ref.actor.oprd_proj_allow_missing = bool(cfg.get("proj_allow_missing", False))

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
            "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        from agent_system.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        a = config.actor_rollout_ref.actor
        print(f"[PR-OPD] representation distillation: coef={a.oprd_loss_coef} (enabled={a.use_oprd_loss}, "
              f"layers={a.oprd_layers}, last_k={a.oprd_last_k}, normalize={a.oprd_normalize}, "
              f"gate={a.oprd_gate}, proj={a.oprd_proj_mode})")
        print(f"[PR-OPD] token-level distillation: coef={a.sdar_loss_coef} (enabled={a.use_sdar_loss}, gate_beta={a.sdar_gate_beta})")

        from verl.trainer.ppo.pr_opd_ray_trainer import PROPDRayTrainer

        # Teacher context source: self (default) = the trajectory's own hindsight skill;
        # static = SDAR's external skill library (same retrieval as main_sdar).
        skill_source = str(cfg.get("skill_source", "self"))
        skill_provider = None
        if skill_source == "static":
            from verl.trainer.ppo.rlsd_utils import SkillProvider
            skill_provider = SkillProvider(
                skills_dir=cfg.get("skills_dir", "skills/alfworld"),
                skill_all=bool(cfg.get("skill_all", False)),
            )
            print(f"[PR-OPD] skill_source=static: teacher = current policy + static skills from "
                  f"{cfg.get('skills_dir')} ({list(skill_provider.skill_contents.keys())})")
        else:
            print(f"[PR-OPD] skill_source={skill_source}: teacher = current policy + self-generated hindsight skills")

        trainer = PROPDRayTrainer(
            skill_provider=skill_provider,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()

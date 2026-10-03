"""
PR-OPD (Privileged Representation On-Policy Distillation) trainer.

Extends SkillSDRayTrainer (the SDAR-family trainer skeleton) and replaces the
static-skill teacher with a self-generated one: at every training step the
CURRENT policy snapshot plays a second role as trajectory analyzer.  It reads
each completed on-policy trajectory, writes a hindsight skill, and the skill is
prepended to the ordinary context to form the privileged teacher branch.  The
teacher's hidden states (same weights, privileged context) are then distilled
into the student in representation space inside the actor update
(``dp_actor.update_policy`` -> ``oprd_utils.compute_oprd_loss``).

The inherited ``SkillSDRayTrainer.fit()`` drives the whole loop; the only
override needed is ``_compute_teacher_log_probs()``, which (a) builds the
teacher context and ships it to the actor, and (b) returns teacher log-probs
for the optional output-space (token-level, SDAR-style gated) distillation term.
"""

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.pr_opd_utils import (
    build_hindsight_teacher_batch,
    build_skill_gen_batch,
    extract_skill_texts,
    serialize_trajectories,
    shuffle_skill_assignment,
)
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer
from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch as build_static_teacher_batch


class PROPDRayTrainer(SkillSDRayTrainer):
    """PR-OPD trainer: on-policy hindsight-skill generation + representation distillation."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("skill_provider", None)  # no static skill library by default
        super().__init__(*args, **kwargs)
        cfg = self.config.algorithm.get("pr_opd", {})
        self.max_obs_chars = cfg.get("max_obs_chars", 500)
        self.max_action_chars = cfg.get("max_action_chars", 300)
        self.max_turns = cfg.get("max_turns", 12)
        self.max_skill_chars = cfg.get("max_skill_chars", 2000)
        self.skill_tag = cfg.get("skill_tag", "[Hindsight Skill Information]")
        self.log_skill_examples = cfg.get("log_skill_examples", 1)
        self.skill_think = str(cfg.get("skill_think", "template"))
        self.skill_strip_think = bool(cfg.get("skill_strip_think", False))
        # Teacher context source:
        #   self      (default) the trajectory's own self-generated hindsight skill
        #   static    SDAR's external static skill library (same retrieval as main_sdar)
        #   shuffled  mismatched control: skills generated as usual, then re-paired
        #             across task groups (see pr_opd_utils.shuffle_skill_assignment)
        self.skill_source = str(cfg.get("skill_source", "self"))
        self.static_skill_position = str(cfg.get("skill_position", "prefix"))
        # Per-token mask for the representation loss: none | success (only successful trajectories)
        self.oprd_pos_mask = str(cfg.get("oprd_pos_mask", "none")).lower()
        assert self.oprd_pos_mask in ("none", "", "success"), f"unknown oprd_pos_mask: {self.oprd_pos_mask}"
        if self.skill_source == "static":
            assert self.skill_provider is not None, "skill_source=static requires a skill_provider"
        # Representation distillation: the actor recomputes the teacher forward per
        # micro-batch, so it needs the teacher-context tensors in the batch.
        self.use_oprd = cfg.get("oprd_coef", 0.0) > 0
        # With both distillation coefficients at zero this reduces to plain GRPO;
        # skip skill generation and the teacher forward entirely so the run is a
        # true baseline (no wasted generation round, no extra sampling RNG draws).
        self.teacher_enabled = cfg.get("coef", 0.0) > 0 or self.use_oprd
        if not self.teacher_enabled:
            print("[PR-OPD] both distillation coefs are 0 -> pure GRPO baseline: "
                  "hindsight-skill generation and teacher forward DISABLED")

    def _generate_hindsight_skills(self, batch: DataProto) -> dict:
        """Use the current policy (analyzer role) to generate one hindsight skill
        per completed trajectory in the batch. Returns {traj_uid: skill_text}."""
        records = serialize_trajectories(
            batch,
            tokenizer=self.tokenizer,
            max_obs_chars=self.max_obs_chars,
            max_action_chars=self.max_action_chars,
            max_turns=self.max_turns,
        )

        gen_batch = build_skill_gen_batch(
            records,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            skill_think=self.skill_think,
        )

        gen_batch_padded, pad_size = pad_dataproto_to_divisor(
            gen_batch, self.actor_rollout_wg.world_size
        )
        gen_output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
        gen_output = unpad_dataproto(gen_output_padded, pad_size=pad_size)

        skills = extract_skill_texts(
            gen_output, tokenizer=self.tokenizer, max_skill_chars=self.max_skill_chars,
            skill_strip_think=self.skill_strip_think,
        )

        skill_by_traj = {uid: skill for (uid, _text, _r), skill in zip(records, skills)}

        # Console diagnostics (lightweight; full metrics go through the inherited fit()).
        n_success = sum(1 for (_u, _t, r) in records if r > 0)
        avg_len = sum(len(s) for s in skills) / max(len(skills), 1)
        avg_rec = sum(len(t) for (_u, t, _r) in records) / max(len(records), 1)
        print(
            f"[PR-OPD] step {getattr(self, 'global_steps', -1)}: "
            f"{len(records)} trajectories analyzed ({n_success} success), "
            f"avg skill length = {avg_len:.0f} chars, avg analyzer prompt = {avg_rec:.0f} chars"
        )
        for (uid, _text, reward), skill in list(zip(records, skills))[: self.log_skill_examples]:
            preview = skill[:400].replace("\n", " | ")
            print(f"[PR-OPD]   example (reward={reward:.1f}): {preview}")

        return skill_by_traj

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        """
        Privileged teacher forward pass:
        1) analyzer (= current policy) generates hindsight skills from the
           completed on-policy trajectories;
        2) skills are prepended per-turn to build the privileged teacher batch;
        3) the teacher context is attached to the batch so that the actor can run
           the teacher forward per micro-batch and align hidden states;
        4) the same policy re-scores the sampled actions under the augmented
           context, yielding teacher log-probs for the (optional) token-level term.
        """
        if not self.teacher_enabled:
            # Pure GRPO baseline: no teacher. Return the student's own log-probs so the
            # inherited fit() bookkeeping (teacher-student gap metrics) stays well-defined
            # and reports exactly zero.
            return batch.batch["old_log_probs"].clone()

        if self.skill_source == "static":
            # Control arm: teacher = same weights + SDAR static skills (retrieval identical
            # to main_sdar); no skill generation.
            teacher_batch = build_static_teacher_batch(
                batch=batch,
                skill_provider=self.skill_provider,
                tokenizer=self.tokenizer,
                max_prompt_length=self.config.data.max_prompt_length,
                truncation=self.config.data.get("truncation", "left"),
                skill_position=self.static_skill_position,
            )
        else:
            skill_by_traj = self._generate_hindsight_skills(batch)
            if self.skill_source == "shuffled":
                # Mismatched control: skills are generated as usual; only the
                # skill <-> trajectory pairing is broken (rotated across task groups).
                traj_uids = batch.non_tensor_batch["traj_uid"]
                group_ids = batch.non_tensor_batch.get("uid", None)
                group_by_traj = None
                if group_ids is not None:
                    group_by_traj = {}
                    for i in range(len(traj_uids)):
                        group_by_traj.setdefault(str(traj_uids[i]), str(group_ids[i]))
                skill_by_traj, sh = shuffle_skill_assignment(skill_by_traj, group_by_traj)
                warn = "  [WARN: fell back to a within-group shift -- not a valid mismatch control]" if sh["degraded"] else ""
                print(f"[PR-OPD] step {self.global_steps}: skill_source=shuffled -> "
                      f"{sh['n']} trajectories / {sh['n_groups']} task groups re-paired, "
                      f"same_traj={sh['same_traj']} same_group={sh['same_group']}{warn}")
            teacher_batch = build_hindsight_teacher_batch(
                batch=batch,
                skill_by_traj=skill_by_traj,
                tokenizer=self.tokenizer,
                max_prompt_length=self.config.data.max_prompt_length,
                skill_tag=self.skill_tag,
            )

        if self.use_oprd:
            # Ship the privileged context to the actor so update_policy() can run
            # the teacher forward per micro-batch and extract hidden states.
            batch.batch["teacher_input_ids"] = teacher_batch.batch["input_ids"]
            batch.batch["teacher_attention_mask"] = teacher_batch.batch["attention_mask"]
            batch.batch["teacher_position_ids"] = teacher_batch.batch["position_ids"]
            if self.oprd_pos_mask == "success":
                # Apply the representation loss only on rows of successful trajectories
                # (episode_rewards > 0); failed trajectories still get PG + KL.
                T = batch.batch["responses"].size(1)
                resp_mask_f = batch.batch["attention_mask"][:, -T:].float()
                ep_r = np.asarray(batch.non_tensor_batch["episode_rewards"], dtype=np.float64)
                keep = torch.as_tensor(ep_r > 0, device=resp_mask_f.device).float()
                pm = (resp_mask_f * keep.unsqueeze(-1)).contiguous()
                batch.batch["oprd_pos_mask"] = pm
                valid = resp_mask_f.sum().item()
                print(f"[PR-OPD] oprd_pos_mask=success: {keep.mean().item():.1%} successful rows, "
                      f"{pm.sum().item():.0f}/{valid:.0f} response tokens supervised "
                      f"({pm.sum().item()/max(valid,1):.1%})")

        teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        return teacher_output.batch["old_log_probs"]

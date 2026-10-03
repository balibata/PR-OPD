"""
PR-OPD (Privileged Representation On-Policy Distillation) utilities.

PR-OPD builds its privileged teacher on the fly: after every rollout phase the
CURRENT policy acts as a trajectory analyzer, reads each of its own completed
trajectories and writes a short natural-language *hindsight skill* (a reusable
workflow, the decisive observations, or failure-avoidance rules).  The skill is
prepended to the ordinary context to form the privileged teacher branch, which
shares weights with the student.  The teacher's hidden states are then distilled
into the student in representation space (see ``oprd_utils.compute_oprd_loss``).

Because analyzer and actor share the same (latest) weights, the hindsight
supervision evolves together with the policy: no external teacher, no static
skill bank, no SFT warm-up.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


# The instruction is appended AFTER the trajectory so that left-truncation
# (which keeps the tail of an overlong prompt) never cuts it off.
SKILL_ANALYSIS_INSTRUCTION = (
    "The interaction record above is now COMPLETE.\n"
    "You are reviewing it with hindsight. Write a short hindsight skill that would "
    "help an agent solve THIS task better on a fresh attempt.\n"
    "- If the attempt SUCCEEDED: summarize the reusable workflow and the decisive "
    "observations that led to success (e.g., where the target object was found, "
    "which actions mattered).\n"
    "- If the attempt FAILED: identify the critical mistakes and give concrete "
    "corrective rules to avoid them (e.g., locations searched in vain, invalid or "
    "repeated actions, wrong ordering of subgoals).\n"
    "Requirements: at most 120 words, imperative voice, numbered list of 2-5 rules, "
    "no restating of the full trajectory. Output ONLY the skill."
)


def _extract_user_content(text: str) -> str:
    """Strip chat-template remnants ('system\\n...', 'user\\n', trailing 'assistant')
    from a decoded (skip_special_tokens=True) prompt so the serialized trajectory
    stays compact."""
    idx = text.find("user\n")
    if idx != -1:
        text = text[idx + len("user\n"):]
    stripped = text.rstrip()
    if stripped.endswith("assistant"):
        text = stripped[: -len("assistant")]
    return text.strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + " ...[truncated]"
    return text


def serialize_trajectories(
    batch: DataProto,
    tokenizer,
    max_obs_chars: int = 500,
    max_action_chars: int = 300,
    max_turns: int = 12,
) -> List[Tuple[str, str, float]]:
    """
    Group the flat per-turn batch rows back into trajectories and serialize each
    into a compact text record followed by the analysis instruction.

    The batch produced by the multi-turn rollout is "one row per turn": the turns
    of one trajectory are scattered over the batch and are re-assembled here via
    ``non_tensor_batch['traj_uid']``.

    Returns a list of (traj_uid, serialized_text, episode_reward), one entry per
    trajectory, ordered by first appearance in the batch.
    """
    traj_uids = batch.non_tensor_batch["traj_uid"]
    turn_steps = batch.non_tensor_batch.get("turn_step", np.zeros(len(traj_uids)))
    episode_rewards = batch.non_tensor_batch.get("episode_rewards", np.zeros(len(traj_uids)))

    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]

    # traj_uid -> list of (turn_step, obs_text, action_text); remember order + reward
    turns_by_traj: Dict[str, List[Tuple[int, str, str]]] = {}
    order: List[str] = []
    reward_by_traj: Dict[str, float] = {}

    for i in range(len(traj_uids)):
        uid = str(traj_uids[i])
        obs_text = _extract_user_content(tokenizer.decode(prompts[i], skip_special_tokens=True))
        action_text = tokenizer.decode(responses[i], skip_special_tokens=True).strip()
        if uid not in turns_by_traj:
            turns_by_traj[uid] = []
            order.append(uid)
            reward_by_traj[uid] = float(episode_rewards[i])
        turns_by_traj[uid].append((int(turn_steps[i]), obs_text, action_text))

    records = []
    for uid in order:
        turns = sorted(turns_by_traj[uid], key=lambda t: t[0])
        # Keep the first turn (contains the task description) plus the most
        # recent turns, so overlong trajectories stay within budget.
        if len(turns) > max_turns:
            turns = [turns[0]] + turns[-(max_turns - 1):]

        parts = []
        for step, obs, act in turns:
            parts.append(
                f"[Turn {step}]\nObservation: {_truncate(obs, max_obs_chars)}\n"
                f"Action: {_truncate(act, max_action_chars)}"
            )
        reward = reward_by_traj[uid]
        outcome = "SUCCEEDED" if reward > 0 else "FAILED"
        header = "Below is the complete interaction record of an agent attempting a task.\n\n"
        footer = f"\n\n[Outcome] The attempt {outcome} (episode reward = {reward:.2f}).\n\n"
        records.append((uid, header + "\n\n".join(parts) + footer + SKILL_ANALYSIS_INSTRUCTION, reward))

    return records


def build_skill_gen_batch(
    records: List[Tuple[str, str, float]],
    tokenizer,
    max_prompt_length: int,
    skill_think: str = "template",
) -> DataProto:
    """
    Build a generation batch (one row per trajectory) for the analyzer role of the
    current policy. Mirrors TrajectoryCollector.preprocess_single_sample: chat
    template -> left-padded input_ids/attention_mask/position_ids + raw_prompt_ids.

    skill_think: "template" keeps the chat template's default (Qwen3 thinks first);
                 "off" passes enable_thinking=False (ignored by templates without it).
    """
    input_ids_list, attention_mask_list, raw_prompt_ids_list = [], [], []

    for _uid, analysis_text, _reward in records:
        chat = [{"role": "user", "content": analysis_text}]
        tmpl_kw = {"enable_thinking": False} if skill_think == "off" else {}
        prompt_with_template = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False, **tmpl_kw
        )
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_template,
            tokenizer=tokenizer,
            max_length=max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation="left",
        )
        raw_prompt_ids = tokenizer.encode(prompt_with_template, add_special_tokens=False)
        if len(raw_prompt_ids) > max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]

        input_ids_list.append(input_ids[0])
        attention_mask_list.append(attention_mask[0])
        raw_prompt_ids_list.append(raw_prompt_ids)

    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    position_ids = compute_position_id_with_mask(attention_mask)

    gen_batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        non_tensors={
            "raw_prompt_ids": np.array(raw_prompt_ids_list, dtype=object),
        },
    )
    return gen_batch


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    """Remove a <think>...</think> block (or an unclosed leading one) from the analyzer
    output so that only the final skill text is used as privileged context."""
    t = _THINK_RE.sub("", text)
    if "</think>" in t:
        t = t.split("</think>")[-1]
    return t.strip()


def extract_skill_texts(
    gen_output: DataProto,
    tokenizer,
    max_skill_chars: int = 2000,
    skill_strip_think: bool = False,
) -> List[str]:
    """Decode the analyzer generations into plain skill strings (row order preserved)."""
    responses = gen_output.batch["responses"]
    skills = []
    for i in range(responses.size(0)):
        skill = tokenizer.decode(responses[i], skip_special_tokens=True).strip()
        if skill_strip_think:
            skill = strip_think(skill)
        skills.append(_truncate(skill, max_skill_chars))
    return skills


def shuffle_skill_assignment(skill_by_traj: Dict[str, str],
                             group_by_traj: Optional[Dict[str, str]] = None):
    """Mismatched-skill control (``skill_source=shuffled``): re-pair {traj_uid: skill} across
    trajectories so that every trajectory receives, in its teacher context, the hindsight
    skill written for a DIFFERENT task.

    Why rotate whole task groups instead of shifting by one: GRPO samples ``env.rollout.n``
    rollouts per task and those trajectories are contiguous in the batch, so a shift by one
    would hand a trajectory the skill of another rollout of the SAME task (same room, same
    goal) -- hardly a mismatch.  Groups are keyed by ``non_tensor_batch['uid']`` (shared by
    all rollouts of a task) and rotated by one position.

    With no group information or a single group the function falls back to a trajectory-level
    rotation and sets ``stats['degraded']=True``; callers must log that, otherwise the arm
    silently turns into a same-task control.

    Returns (new_skill_by_traj, stats); the input is not modified.
    """
    uids = list(skill_by_traj.keys())
    n = len(uids)
    stats = {"n": n, "n_groups": 0, "degraded": False, "same_traj": 0, "same_group": 0}
    if n < 2:
        stats["degraded"] = True
        return dict(skill_by_traj), stats

    # bucket by group, keeping first-appearance order
    groups: Dict[str, List[str]] = {}
    order: List[str] = []
    for u in uids:
        g = str(group_by_traj.get(u, "__nogroup__")) if group_by_traj else "__nogroup__"
        if g not in groups:
            groups[g] = []
            order.append(g)
        groups[g].append(u)
    stats["n_groups"] = len(order)

    donor_of: Dict[str, str] = {}
    if len(order) < 2:
        # degraded path: a single task group (or no group info) -> trajectory-level rotation
        stats["degraded"] = True
        for i, u in enumerate(uids):
            donor_of[u] = uids[(i + 1) % n]
    else:
        for gi, g in enumerate(order):
            donors = groups[order[(gi + 1) % len(order)]]
            for j, u in enumerate(groups[g]):
                donor_of[u] = donors[j % len(donors)]

    new_map = {u: skill_by_traj[donor_of[u]] for u in uids}
    for u in uids:
        d = donor_of[u]
        if d == u:
            stats["same_traj"] += 1
        elif group_by_traj and str(group_by_traj.get(d)) == str(group_by_traj.get(u)):
            stats["same_group"] += 1
    return new_map, stats


def build_hindsight_teacher_batch(
    batch: DataProto,
    skill_by_traj: Dict[str, str],
    tokenizer,
    max_prompt_length: int,
    skill_tag: str = "[Hindsight Skill Information]",
) -> DataProto:
    """
    Build the privileged teacher batch by prepending each trajectory's generated
    hindsight skill to every turn of that trajectory. The sampled responses stay
    unchanged; only the prompt context differs from the student branch.

    Adapted from rlsd_ray_trainer.build_teacher_batch (static SkillProvider replaced
    by the on-policy skill_by_traj mapping).
    """
    bs = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)
    traj_uids = batch.non_tensor_batch["traj_uid"]

    teacher_input_ids_list, teacher_attention_mask_list, teacher_position_ids_list = [], [], []

    for i in range(bs):
        original_input_ids = batch.batch["input_ids"][i]
        original_attention_mask = batch.batch["attention_mask"][i]
        prompt_length = original_input_ids.size(0) - response_length

        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]

        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        valid_start = valid_start[0].item() if len(valid_start) > 0 else 0
        valid_prompt_ids = prompt_ids[valid_start:]
        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

        skill_text = skill_by_traj.get(str(traj_uids[i]), "")
        if skill_text:
            teacher_prompt_text = f"{skill_tag}\n{skill_text}\n\n" + prompt_text
        else:
            teacher_prompt_text = prompt_text

        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)
        if len(teacher_prompt_ids) > max_prompt_length:
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]

        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        actual_prompt_len = len(teacher_prompt_ids)

        pad_length = max_prompt_length - actual_prompt_len
        if pad_length > 0:
            pad_ids = torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
            teacher_prompt_ids = torch.cat([pad_ids, teacher_prompt_ids])
            t_prompt_mask = torch.cat([
                torch.zeros(pad_length, dtype=torch.long),
                torch.ones(actual_prompt_len, dtype=torch.long),
            ])
        else:
            t_prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)

        response_ids = batch.batch["responses"][i]
        response_mask = original_attention_mask[-response_length:]

        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids])
        teacher_full_mask = torch.cat([t_prompt_mask, response_mask])
        teacher_position_ids = compute_position_id_with_mask(teacher_full_mask.unsqueeze(0))[0]

        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        teacher_position_ids_list.append(teacher_position_ids)

    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(teacher_input_ids_list),
            "attention_mask": torch.stack(teacher_attention_mask_list),
            "position_ids": torch.stack(teacher_position_ids_list),
            "responses": batch.batch["responses"],
        },
    )
    return teacher_batch

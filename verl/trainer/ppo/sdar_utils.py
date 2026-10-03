"""
Confidence-Gated Teacher Distillation (SDAR) utilities.

Token-level gated distillation loss where the gate is derived from
the teacher-student log-probability gap, so tokens where the teacher
is more confident receive stronger distillation signal.
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_sdar_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gate_beta: float = 5.0,
    loss_agg_mode: str = "token-mean",
    loss_mask: torch.Tensor = None,
) -> tuple[torch.Tensor, dict]:
    """
    Confidence-Gated Teacher Distillation loss.

    L_SDAR = agg( g_t * (log pi_teacher - log pi_student) )
    where g_t = sigmoid(beta * delta_t), delta_t = log pi_teacher - log pi_student.

    The gate g_t is detached so gradients only flow through the student log-probs.

    Args:
        student_log_probs: (bs, response_length) - log pi_theta(y_t | x, y_<t).
            Current policy forward pass; retains gradients.
        teacher_log_probs: (bs, response_length) - log pi_teacher(y_t | x, r, y_<t).
            Frozen (no grad). Teacher sees skill-augmented input.
        response_mask: (bs, response_length) - mask for valid response tokens.
        gate_beta: temperature for the sigmoid gate. Higher = sharper gating.
        loss_agg_mode: aggregation mode passed to agg_loss.

    Returns:
        sdar_loss: scalar loss.
        metrics: dict with gating statistics.
    """
    teacher_log_probs = teacher_log_probs.detach()

    # Optional extra mask (e.g. only samples whose GRPO advantage is nonzero,
    # or only successful trajectories). Gate statistics are still reported on
    # the full response_mask so they stay comparable with unmasked runs.
    full_mask = response_mask
    if loss_mask is not None:
        response_mask = response_mask * loss_mask.to(response_mask.dtype)

    delta_t = teacher_log_probs - student_log_probs.detach()

    gate = torch.sigmoid(gate_beta * delta_t).detach()

    kl_per_token = teacher_log_probs - student_log_probs

    gated_kl = gate * kl_per_token

    loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = full_mask.sum().clamp(min=1)
        gate_mean = (gate * full_mask).sum() / mask_sum
        gate_active = ((gate > 0.5).float() * full_mask).sum() / mask_sum
        gap_mean = (delta_t * full_mask).sum() / mask_sum
        mask_ratio = response_mask.sum() / mask_sum

    metrics = {
        "sdar/gate_mean": gate_mean.item(),
        "sdar/gate_active_ratio": gate_active.item(),
        "sdar/teacher_gap_mean": gap_mean.item(),
        "sdar/loss": loss.detach().item(),
        "sdar/loss_mask_ratio": mask_ratio.item(),
    }

    return loss, metrics

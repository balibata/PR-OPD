"""
OPRD (On-Policy Representation Distillation) loss utilities for the
self-evolving OPD framework (SE-OPRD).

Instead of matching teacher/student token log-probabilities in the OUTPUT
space (SDAR gated loss), the OPRD loss aligns the student's intermediate
hidden states with the teacher branch's hidden states on the same sampled
response tokens — supervising in the REPRESENTATION space, before the LM head
(Yang et al., OPRD, arXiv 2606.06021).

In the self-evolving setting the teacher is the SAME model conditioned on a
privileged hindsight-skill context, so student and teacher representations
live in an identical, perfectly aligned space and no cross-architecture
bridge is needed (OPRD-Vanilla regime).

Loss (per selected layer l, per supervised position t):
    L = mean_l  masked_mean_t  w_t * s * || h_theta^(l,t) - sg(h_T^(l,t)) ||^2
where
    s   = 1/d                (paper Eq. 6)   if normalize=False
    s   = 1  on L2-normalized vectors        if normalize=True
                              (equivalent to 2 - 2*cos, bounded in [0,4])
    w_t = optional sigmoid confidence gate from the teacher-student
          log-prob gap (SDAR-style asymmetric trust), detached.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def last_k_position_mask(response_mask: torch.Tensor, last_k: int) -> torch.Tensor:
    """Restrict a (bs, response_len) validity mask to the last `last_k` valid
    tokens of each row (OPRD's LAST-k position selection). last_k <= 0 keeps all."""
    if last_k <= 0:
        return response_mask
    # count of valid tokens from position t to the end (inclusive)
    rev_cum = response_mask.flip(-1).cumsum(dim=-1).flip(-1)
    return response_mask * (rev_cum <= last_k)


def compute_oprd_loss(
    student_hidden: Dict[int, torch.Tensor],
    teacher_hidden: Dict[int, torch.Tensor],
    response_mask: torch.Tensor,
    last_k: int = -1,
    normalize: bool = True,
    gate: Optional[torch.Tensor] = None,
    layer_weights: Optional[Dict[int, float]] = None,
    topk_frac: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Args:
        student_hidden: {layer_idx: (bs, response_len, d)} — requires grad.
        teacher_hidden: {layer_idx: (bs, response_len, d)} — privileged branch,
            treated as constant (detached here regardless).
        response_mask:  (bs, response_len) valid-token mask.
        last_k:         supervise only the last k valid response tokens (<=0: all).
        normalize:      L2-normalize both vectors before the MSE (scale-invariant).
        gate:           optional (bs, response_len) per-token weight in [0,1],
                        detached by the caller.

    Returns:
        (scalar loss, metrics dict)
    """
    assert student_hidden.keys() == teacher_hidden.keys(), (
        f"layer mismatch: {sorted(student_hidden)} vs {sorted(teacher_hidden)}"
    )

    mask = last_k_position_mask(response_mask, last_k).float()
    weight = mask if gate is None else mask * gate.detach().float()
    denom = weight.sum().clamp(min=1.0)
    mask_denom = mask.sum().clamp(min=1.0)

    layer_losses = []
    sq_errs = []
    cos_sum = 0.0
    for layer in sorted(student_hidden.keys()):
        hs = student_hidden[layer].float()
        ht = teacher_hidden[layer].detach().float()

        if normalize:
            hs_n = F.normalize(hs, dim=-1)
            ht_n = F.normalize(ht, dim=-1)
            sq_err = (hs_n - ht_n).pow(2).sum(dim=-1)  # (bs, resp_len), == 2 - 2*cos
        else:
            d = hs.size(-1)
            sq_err = (hs - ht).pow(2).sum(dim=-1) / d

        sq_errs.append(sq_err)

        with torch.no_grad():
            cos = F.cosine_similarity(hs, ht, dim=-1)  # (bs, resp_len)
            cos_sum += (cos * mask).sum() / mask_denom

    # Optional: keep only the top-k fraction of valid tokens by representation gap
    if topk_frac and 0.0 < topk_frac < 1.0:
        with torch.no_grad():
            score = torch.stack(sq_errs).mean(0)                   # (bs, T) per-token mean squared error
            flat = score.reshape(-1)
            vmask = mask.reshape(-1) > 0
            k = max(int(vmask.sum().item() * topk_frac), 1)
            thr = torch.topk(flat[vmask], k, largest=True).values.min()
            sel = (score >= thr).float() * mask
        weight = sel if gate is None else sel * gate.detach().float()
        denom = weight.sum().clamp(min=1.0)

    for _se in sq_errs:
        layer_losses.append((_se * weight).sum() / denom)

    if layer_weights:
        # Optional per-layer weights; all-ones reduces to a plain mean over layers
        w = torch.tensor([float(layer_weights.get(l, 1.0)) for l in sorted(student_hidden.keys())],
                         device=layer_losses[0].device, dtype=layer_losses[0].dtype)
        loss = (torch.stack(layer_losses) * w).sum() / w.sum().clamp(min=1e-8)
    else:
        loss = torch.stack(layer_losses).mean()

    with torch.no_grad():
        metrics = {
            "oprd/loss": loss.detach().item(),
            "oprd/cos_sim": (cos_sum / max(len(layer_losses), 1)).item(),
            "oprd/num_layers": float(len(layer_losses)),
            "oprd/supervised_token_frac": (mask.sum() / response_mask.float().sum().clamp(min=1.0)).item(),
        }
        if topk_frac and 0.0 < topk_frac < 1.0:
            metrics["oprd/topk_frac"] = float(topk_frac)
            metrics["oprd/topk_kept"] = ((weight > 0).float().sum() / mask.sum().clamp(min=1.0)).item()
        if layer_weights:
            metrics["oprd/layer_weighted"] = 1.0
        if gate is not None:
            metrics["oprd/gate_mean"] = ((gate.float() * mask).sum() / mask_denom).item()

    return loss, metrics



def resolve_layer_spec(spec, num_layers: int) -> list:
    """
    Resolve a layer-selection spec into hidden_states tuple indices
    (1..num_layers; index 0 is the embedding output and is never selected).

    spec: "last" | "all" | "even" | int stride | "1,7,14,21,28"-style csv.
    """
    if spec is None:
        return [num_layers]
    if isinstance(spec, int):
        stride = max(spec, 1)
        layers = list(range(stride, num_layers + 1, stride))
        return layers or [num_layers]
    spec = str(spec).strip().lower()
    if spec == "last":
        return [num_layers]
    if spec == "all":
        return list(range(1, num_layers + 1))
    if spec == "even":
        return list(range(2, num_layers + 1, 2))
    layers = set()
    for tok in spec.split(","):            # supports ranges ("21-36") and mixes ("1-3,36")
        tok = tok.strip()
        if not tok: continue
        if "-" in tok:
            a, b = tok.split("-"); layers.update(range(int(a), int(b) + 1))
        else:
            layers.add(int(tok))
    layers = sorted(layers)
    for l in layers:
        assert 1 <= l <= num_layers, f"layer {l} out of range [1, {num_layers}]"
    return layers

"""
Per-layer MLP projection heads for the PR-OPD "+MLP" variant.

With ``oprd_proj_mode=mlp`` the student's hidden state at every supervised layer is
passed through a small two-layer MLP (d -> hidden -> d) before being aligned with the
privileged teacher's hidden state at the same layer; the teacher side is untouched.
The projectors are trained jointly with the policy by their own AdamW optimizer.

The projectors are intentionally NOT registered inside the FSDP-wrapped actor module:
verl feeds ``actor_module.state_dict()`` straight into vLLM's ``load_weights`` for the
rollout weight sync and unexpected keys would break that sync. They are therefore
checkpointed separately (``oprd_projectors.pt`` in the actor checkpoint directory).
"""
from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist


class _LayerMLP(nn.Module):
    """Two-layer MLP, d_in -> d_hidden -> d_out, GELU in between."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_out)
        nn.init.normal_(self.fc1.weight, std=0.02); nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class OPRDProjectors(nn.Module):
    """One MLP per supervised layer. ``layer_ids`` index HF ``hidden_states`` (1..L)."""

    def __init__(self, layer_ids: List[int], d: int, hidden: int = 512, dtype=torch.float32):
        super().__init__()
        self.layer_ids = list(layer_ids)
        self.proj = nn.ModuleDict({str(l): _LayerMLP(d, hidden, d) for l in self.layer_ids})
        self.to(dtype)

    def forward(self, layer: int, hs: torch.Tensor) -> torch.Tensor:
        m = self.proj[str(layer)]
        return m(hs.to(m.fc1.weight.dtype))   # bf16 hidden state -> projector dtype (fp32)

    def allreduce_grads(self):
        """The projectors are outside FSDP and see different data on every rank, so their
        gradients must be averaged across ranks by hand before the optimizer step."""
        if not dist.is_initialized():
            return
        world = dist.get_world_size()
        if world == 1:
            return
        for p in self.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world

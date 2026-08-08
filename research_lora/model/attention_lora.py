"""Optional second-stage attention adapter for baseline batch-first MultiheadAttention.

It is intentionally not part of DEFAULT_TARGETS.  The implementation only supports
self-attention because that is the validated ablation point in the current DiT.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class EgoQueryValueAttentionLoRA(nn.Module):
    def __init__(self, base: nn.MultiheadAttention, rank: int = 4, alpha: float | None = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if not base.batch_first or base._qkv_same_embed_dim is False:
            raise ValueError("Only batch_first, merged-QKV MultiheadAttention is supported")
        self.base, self.rank = base, int(rank)
        self.scaling = float(alpha if alpha is not None else rank) / rank
        self.dropout = nn.Dropout(dropout)
        d = base.embed_dim
        self.q_A, self.q_B = nn.Parameter(torch.empty(rank, d)), nn.Parameter(torch.zeros(d, rank))
        self.v_A, self.v_B = nn.Parameter(torch.empty(rank, d)), nn.Parameter(torch.zeros(d, rank))
        self.o_A, self.o_B = nn.Parameter(torch.empty(rank, d)), nn.Parameter(torch.zeros(d, rank))
        for parameter in (self.q_A, self.v_A, self.o_A):
            nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.enabled, self.strength = True, 1.0

    def _delta(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.dropout(x) @ a.t() @ b.t() * self.scaling

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, key_padding_mask=None,
                need_weights: bool = True, attn_mask=None, average_attn_weights: bool = True, is_causal: bool = False):
        zero_update = not bool(torch.count_nonzero(self.q_B) or torch.count_nonzero(self.v_B) or torch.count_nonzero(self.o_B))
        if not self.enabled or self.strength == 0.0 or zero_update:
            return self.base(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights,
                             attn_mask=attn_mask, average_attn_weights=average_attn_weights, is_causal=is_causal)
        if query is not key or key is not value:
            raise NotImplementedError("Attention LoRA ablation currently supports DiT self-attention only")
        x = query
        if x.ndim != 3:
            raise ValueError("Expected [B, P, D] batch-first tokens")
        d, h = self.base.embed_dim, self.base.num_heads
        q_weight, k_weight, v_weight = self.base.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = self.base.in_proj_bias.chunk(3, dim=0)
        q, k, v = F.linear(x, q_weight, q_bias), F.linear(x, k_weight, k_bias), F.linear(x, v_weight, v_bias)
        q = q.clone(); v = v.clone()
        q[:, 0] += self._delta(x[:, 0], self.q_A, self.q_B) * self.strength
        v[:, 0] += self._delta(x[:, 0], self.v_A, self.v_B) * self.strength
        bsz, tokens, _ = q.shape
        head_dim = d // h
        q, k, v = (z.reshape(bsz, tokens, h, head_dim).transpose(1, 2) for z in (q, k, v))
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :].bool(), float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                logits = logits.masked_fill(attn_mask, float("-inf"))
            else:
                logits = logits + attn_mask
        weights = torch.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=self.base.dropout, training=self.training)
        output = (weights @ v).transpose(1, 2).reshape(bsz, tokens, d)
        output = F.linear(output, self.base.out_proj.weight, self.base.out_proj.bias)
        output = output.clone()
        output[:, 0] += self._delta(x[:, 0], self.o_A, self.o_B) * self.strength
        returned_weights = weights.mean(dim=1) if average_attn_weights else weights
        return output, returned_weights if need_weights else None

    @torch.no_grad()
    def base_identity_error(self, x: torch.Tensor, **kwargs) -> float:
        """Return an exact-identity diagnostic for B=0 or disabled-adapter ablations."""
        adapted, _ = self(x, x, x, **kwargs)
        base, _ = self.base(x, x, x, **kwargs)
        return float((adapted - base).abs().max())

"""LoRA layers with exact baseline identity at initialisation and rho=0."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear can only wrap nn.Linear, got {type(base)!r}")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank, self.alpha = int(rank), float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.enabled = True
        self.strength = 1.0

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        # Returning the base result directly is necessary for bitwise rho=0 identity.
        if not self.enabled or self.strength == 0.0:
            return output
        return output + self.delta(x) * self.strength


class EgoMaskedLoRALinear(LoRALinear):
    """Applies the low-rank update only to token 0 of a [B, P, D] tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if not self.enabled or self.strength == 0.0:
            return output
        if x.ndim != 3:
            raise ValueError(f"EgoMaskedLoRALinear expects [B, P, D], got {tuple(x.shape)}")
        # clone avoids an in-place write to a tensor needed by autograd.
        result = output.clone()
        result[:, 0, :] = result[:, 0, :] + self.delta(x[:, 0, :]) * self.strength
        return result


class StyleEgoMaskedLoRALinear(nn.Module):
    """One frozen linear layer with mutually-exclusive aggressive/conservative adapters."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.aggressive = EgoMaskedLoRALinear(base, rank, alpha, dropout)
        self.conservative = EgoMaskedLoRALinear(base, rank, alpha, dropout)
        self.style: Literal["aggr", "cons", "normal"] = "normal"
        self.strength = 0.0
        self.enabled = True

    def set_router(self, style: str, strength: float, enabled: bool = True) -> None:
        aliases = {"aggressive": "aggr", "conservative": "cons"}
        style = aliases.get(style, style)
        if style not in {"aggr", "cons", "normal"}:
            raise ValueError(f"Unknown style {style!r}")
        self.style, self.strength, self.enabled = style, float(strength), bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if not self.enabled or self.style == "normal" or self.strength == 0.0:
            return output
        branch = self.aggressive if self.style == "aggr" else self.conservative
        branch.enabled, branch.strength = True, abs(self.strength)
        # Only the selected branch participates in autograd and receives gradients.
        return branch(x)

    def adapter_parameters(self):
        yield from self.aggressive.parameters(recurse=False)
        yield from self.conservative.parameters(recurse=False)

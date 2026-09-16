"""LoRA layers with exact baseline identity at initialisation and rho=0."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


def _strength_is_zero(strength: float | torch.Tensor) -> bool:
    """兼容原有标量强度和门控产生的逐样本强度。"""
    if torch.is_tensor(strength):
        # 不在每个 LoRA 层触发 GPU -> CPU 同步；逐样本零值由广播乘法自然处理。
        return strength.numel() == 0
    return float(strength) == 0.0


def _scale_delta(delta: torch.Tensor, strength: float | torch.Tensor) -> torch.Tensor:
    """把标量或 ``[B]`` 强度广播到 LoRA 增量。"""
    if not torch.is_tensor(strength):
        return delta * float(strength)
    value = strength.to(device=delta.device, dtype=delta.dtype)
    if value.ndim == 0:
        return delta * value
    if value.ndim != 1 or value.shape[0] != delta.shape[0]:
        raise ValueError(
            f"逐样本 LoRA 强度必须是 [{delta.shape[0]}]，实际为 {tuple(value.shape)}"
        )
    return delta * value.reshape((value.shape[0],) + (1,) * (delta.ndim - 1))


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
        if not self.enabled or _strength_is_zero(self.strength):
            return output
        return output + _scale_delta(self.delta(x), self.strength)


class EgoMaskedLoRALinear(LoRALinear):
    """Applies the low-rank update only to token 0 of a [B, P, D] tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if not self.enabled or _strength_is_zero(self.strength):
            return output
        if x.ndim != 3:
            raise ValueError(f"EgoMaskedLoRALinear expects [B, P, D], got {tuple(x.shape)}")
        # clone avoids an in-place write to a tensor needed by autograd.
        result = output.clone()
        result[:, 0, :] = result[:, 0, :] + _scale_delta(self.delta(x[:, 0, :]), self.strength)
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
        # 条件路由默认关闭；None 时完整保持原有 style + strength 路径。
        self._conditional_low: torch.Tensor | None = None
        self._conditional_high: torch.Tensor | None = None

    def set_router(self, style: str, strength: float | torch.Tensor, enabled: bool = True) -> None:
        aliases = {"aggressive": "aggr", "conservative": "cons"}
        style = aliases.get(style, style)
        if style not in {"aggr", "cons", "normal"}:
            raise ValueError(f"Unknown style {style!r}")
        routed_strength = strength if torch.is_tensor(strength) else float(strength)
        self.style, self.strength, self.enabled = style, routed_strength, bool(enabled)

    def set_conditional_coefficients(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        *,
        enabled: bool = True,
    ) -> None:
        """设置当前 batch 的 Low/High 逐样本系数。"""
        if low.ndim != 1 or high.ndim != 1 or low.shape != high.shape:
            raise ValueError("条件 LoRA 系数必须是形状相同的 [B] 张量")
        self._conditional_low = low
        self._conditional_high = high
        self.enabled = bool(enabled)

    def clear_conditional_coefficients(self) -> None:
        """关闭条件路由，恢复原有标量/逐样本 strength 路径。"""
        self._conditional_low = None
        self._conditional_high = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if not self.enabled:
            return output
        if self._conditional_low is not None or self._conditional_high is not None:
            if self._conditional_low is None or self._conditional_high is None:
                raise RuntimeError("Low/High 条件系数必须同时设置")
            if x.ndim != 3:
                raise ValueError(f"条件 Style LoRA 需要 [B,P,D]，实际为 {tuple(x.shape)}")
            result = output.clone()
            ego = x[:, 0, :]
            delta = _scale_delta(self.conservative.delta(ego), self._conditional_low)
            delta = delta + _scale_delta(self.aggressive.delta(ego), self._conditional_high)
            result[:, 0, :] = result[:, 0, :] + delta
            return result
        if self.style == "normal" or _strength_is_zero(self.strength):
            return output
        branch = self.aggressive if self.style == "aggr" else self.conservative
        branch.enabled, branch.strength = True, abs(self.strength)
        # Only the selected branch participates in autograd and receives gradients.
        return branch(x)

    def adapter_parameters(self):
        yield from self.aggressive.parameters(recurse=False)
        yield from self.conservative.parameters(recurse=False)

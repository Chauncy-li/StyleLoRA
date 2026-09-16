"""场景上下文条件的 LoRA 连续强度上限门控。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


class SceneStrengthGate(nn.Module):
    """用一个共享 MLP 从 ``h_c`` 预测 Low/High 两个连续强度上限。"""

    def __init__(self, hc_dim: int = 192, hidden_dim: int = 64) -> None:
        super().__init__()
        if hc_dim <= 0 or hidden_dim <= 0:
            raise ValueError("hc_dim 和 hidden_dim 必须为正")
        self.hc_dim = int(hc_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.hc_dim),
            nn.Linear(self.hc_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )

    def forward(self, h_c: torch.Tensor) -> torch.Tensor:
        """返回 ``[B,2]``，列顺序固定为 ``[c_low, c_high]``。"""
        if h_c.ndim != 2 or h_c.shape[-1] != self.hc_dim:
            raise ValueError(f"h_c 必须为 [B,{self.hc_dim}]，实际为 {tuple(h_c.shape)}")
        return torch.sigmoid(self.network(h_c))

    @staticmethod
    def effective_rho(caps: torch.Tensor, requested_rho: float | torch.Tensor) -> torch.Tensor:
        """保持符号并按对应方向上限截断，得到逐样本有效强度。"""
        if caps.ndim != 2 or caps.shape[1] != 2:
            raise ValueError(f"caps 必须是 [B,2]，实际为 {tuple(caps.shape)}")
        rho = torch.as_tensor(requested_rho, dtype=caps.dtype, device=caps.device)
        if rho.ndim == 0:
            rho = rho.expand(caps.shape[0])
        elif rho.ndim == 1 and rho.shape[0] == 1:
            rho = rho.expand(caps.shape[0])
        elif rho.ndim != 1 or rho.shape[0] != caps.shape[0]:
            raise ValueError("requested_rho 必须是标量或与 batch 等长的一维张量")
        selected_cap = torch.where(rho < 0, caps[:, 0], caps[:, 1])
        return torch.sign(rho) * torch.minimum(rho.abs(), selected_cap)


def save_scene_gate_checkpoint(
    path: str | Path,
    gate: SceneStrengthGate,
    *,
    training_config: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    """单独保存门控参数，不写入或覆盖 LoRA checkpoint。"""
    payload = {
        "format": "stylelora.scene_gate.v1",
        "model_config": {"hc_dim": gate.hc_dim, "hidden_dim": gate.hidden_dim},
        "model_state": {key: value.detach().cpu() for key, value in gate.state_dict().items()},
        "training_config": dict(training_config),
        "validation": dict(validation),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_scene_gate_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[SceneStrengthGate, dict[str, Any]]:
    """严格加载独立门控 checkpoint。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "stylelora.scene_gate.v1":
        raise ValueError(f"不支持的场景门控 checkpoint: {path}")
    config = payload.get("model_config", {})
    gate = SceneStrengthGate(hc_dim=int(config["hc_dim"]), hidden_dim=int(config["hidden_dim"]))
    gate.load_state_dict(payload["model_state"], strict=True)
    gate.to(device).eval()
    for parameter in gate.parameters():
        parameter.requires_grad_(False)
    return gate, payload

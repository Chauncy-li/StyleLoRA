"""由场景与目标风格共同决定的逐层 LoRA 路由。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


class ConditionalLoRARouter(nn.Module):
    """为每个 LoRA 层预测 High/Low 两个非负调制系数。

    最终层以零初始化，因此刚启用时 modulation 恒为 1，等价于原来的
    全层统一强度；训练后才会形成场景与目标风格相关的逐层差异。
    """

    def __init__(
        self,
        layer_names: Sequence[str],
        *,
        hc_dim: int = 192,
        z_dim: int = 8,
        hidden_dim: int = 128,
        max_log_scale: float = 1.3862943611198906,
        use_diffusion_time: bool = False,
    ) -> None:
        super().__init__()
        if not layer_names or len(set(layer_names)) != len(layer_names):
            raise ValueError("layer_names 必须是非空且不重复的层名序列")
        if min(hc_dim, z_dim, hidden_dim) <= 0 or max_log_scale <= 0:
            raise ValueError("路由网络维度和 max_log_scale 必须为正")
        self.layer_names = tuple(str(name) for name in layer_names)
        self.hc_dim = int(hc_dim)
        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_log_scale = float(max_log_scale)
        # 新方法不使用扩散时间。该开关仅用于读取已有 v1 checkpoint。
        self.use_diffusion_time = bool(use_diffusion_time)
        self.hc_norm = nn.LayerNorm(self.hc_dim)
        self.z_norm = nn.LayerNorm(self.z_dim)
        input_dim = self.hc_dim + self.z_dim + (3 if self.use_diffusion_time else 0)
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, len(self.layer_names) * 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def _batch_time(diffusion_time: torch.Tensor | float, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(diffusion_time, dtype=reference.dtype, device=reference.device)
        if value.ndim == 0:
            value = value.expand(batch_size)
        else:
            value = value.reshape(-1)
            if value.numel() == 1:
                value = value.expand(batch_size)
            elif value.numel() != batch_size:
                raise ValueError(f"diffusion_time 必须是标量或 [{batch_size}]，实际为 {tuple(value.shape)}")
        return value

    def forward(
        self,
        h_c: torch.Tensor,
        target_delta: torch.Tensor,
        diffusion_time: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """返回 ``[B,L,2]`` 调制量，最后一维顺序为 ``[Low, High]``。"""
        if h_c.ndim != 2 or h_c.shape[1] != self.hc_dim:
            raise ValueError(f"h_c 必须是 [B,{self.hc_dim}]，实际为 {tuple(h_c.shape)}")
        if target_delta.ndim != 2 or target_delta.shape != (h_c.shape[0], self.z_dim):
            raise ValueError(
                f"target_delta 必须是 [{h_c.shape[0]},{self.z_dim}]，实际为 {tuple(target_delta.shape)}"
            )
        features = [self.hc_norm(h_c), self.z_norm(target_delta)]
        if self.use_diffusion_time:
            if diffusion_time is None:
                diffusion_time = h_c.new_zeros(h_c.shape[0])
            time = self._batch_time(diffusion_time, h_c.shape[0], h_c)
            features.append(
                torch.stack((time, torch.sin(torch.pi * time), torch.cos(torch.pi * time)), dim=-1)
            )
        features = torch.cat(features, dim=-1)
        raw = self.network(features).reshape(h_c.shape[0], len(self.layer_names), 2)
        # 有界指数参数化允许增强或减弱各层，同时避免无界系数破坏训练稳定性。
        return torch.exp(self.max_log_scale * torch.tanh(raw))


def save_conditional_router_checkpoint(
    path: str | Path,
    router: ConditionalLoRARouter,
    *,
    prototypes: Mapping[str, torch.Tensor],
    training_config: Mapping[str, Any],
    best_validation: Mapping[str, Any] | None = None,
) -> None:
    """单独保存路由和风格原型，不改变现有 LoRA checkpoint 格式。"""
    required = {"low", "neutral", "high"}
    if set(prototypes) != required:
        raise ValueError(f"prototypes 必须且只能包含 {sorted(required)}")
    saved_prototypes = {}
    for name in sorted(required):
        value = torch.as_tensor(prototypes[name]).detach().cpu().float().reshape(-1)
        if value.numel() != router.z_dim:
            raise ValueError(f"prototype {name} 必须有 {router.z_dim} 维")
        saved_prototypes[name] = value
    payload = {
        "format": "stylelora.conditional_router.v2",
        "model_config": {
            "layer_names": list(router.layer_names),
            "hc_dim": router.hc_dim,
            "z_dim": router.z_dim,
            "hidden_dim": router.hidden_dim,
            "max_log_scale": router.max_log_scale,
            "use_diffusion_time": router.use_diffusion_time,
        },
        "model_state": {key: value.detach().cpu() for key, value in router.state_dict().items()},
        "prototypes": saved_prototypes,
        "training_config": dict(training_config),
        "best_validation": dict(best_validation or {}),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_conditional_router_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ConditionalLoRARouter, dict[str, torch.Tensor], dict[str, Any]]:
    """加载独立路由 checkpoint，并返回路由、风格原型和原始元数据。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_format = payload.get("format") if isinstance(payload, dict) else None
    if checkpoint_format not in {
        "stylelora.conditional_router.v1",
        "stylelora.conditional_router.v2",
    }:
        raise ValueError(f"不支持的条件路由 checkpoint: {path}")
    config = payload["model_config"]
    router = ConditionalLoRARouter(
        config["layer_names"],
        hc_dim=int(config["hc_dim"]),
        z_dim=int(config["z_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        max_log_scale=float(config["max_log_scale"]),
        use_diffusion_time=(
            True
            if checkpoint_format == "stylelora.conditional_router.v1"
            else bool(config.get("use_diffusion_time", False))
        ),
    )
    router.load_state_dict(payload["model_state"], strict=True)
    router.to(device)
    prototypes = {
        name: torch.as_tensor(value, device=device).float()
        for name, value in payload["prototypes"].items()
    }
    return router, prototypes, payload

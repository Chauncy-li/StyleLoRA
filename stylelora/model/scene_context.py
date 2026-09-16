"""统一提取冻结 DiffPlanner 的连续场景上下文表征。"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


def masked_mean_scene_context(encoding: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """按有效场景 token 对 encoder 输出做均值池化，得到 ``h_c``。"""
    if encoding.ndim != 3:
        raise ValueError(f"encoding 必须是 [B,P,D]，实际为 {tuple(encoding.shape)}")
    if padding_mask.shape != encoding.shape[:2]:
        raise ValueError(
            f"padding_mask {tuple(padding_mask.shape)} 与 encoding {tuple(encoding.shape)} 不匹配"
        )
    valid = ~padding_mask.bool()
    count = valid.sum(dim=1, keepdim=True).clamp_min(1)
    return (encoding * valid.unsqueeze(-1).to(encoding.dtype)).sum(dim=1) / count


def encode_scene_context(
    baseline: nn.Module,
    model_inputs: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """只运行一次冻结 encoder，同时返回原始 encoder 输出和 ``h_c``。

    padding mask 直接从 FusionEncoder 的实际调用参数捕获，避免在归一化输入上
    重复推断掩码而造成离线特征与在线门控不一致。hook 在本次前向结束后立即移除，
    不会进入 NuPlan planner 的序列化状态。
    """
    try:
        fusion = baseline.encoder.encoder.fusion
    except AttributeError as exc:
        raise RuntimeError("无法定位 DiffPlanner encoder.encoder.fusion") from exc

    captured: list[torch.Tensor] = []

    def _capture_mask(module, args) -> None:
        del module
        if len(args) >= 2:
            captured.append(args[1])

    handle = fusion.register_forward_pre_hook(_capture_mask)
    try:
        encoder_outputs = baseline.encoder(model_inputs)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("FusionEncoder hook 未捕获到 padding mask")
    encoding = encoder_outputs.get("encoding")
    if encoding is None:
        raise RuntimeError("DiffPlanner encoder 输出缺少 'encoding'")
    h_c = masked_mean_scene_context(encoding, captured[-1])
    return encoder_outputs, h_c

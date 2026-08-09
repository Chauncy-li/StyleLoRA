"""Explicit baseline construction and cache-batch conversion for all research scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from baseline.model.diff_planner.diffusion_planner import Diffusion_Planner
from baseline.utils.config import Config
from stylelora.lora.model.style_lora_planner import StyleLoRAPlanner


def _load_weights(model, checkpoint: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("ema_state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
    state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    matched = len(model.state_dict()) - len(missing)
    if matched / max(len(model.state_dict()), 1) < 0.98:
        raise RuntimeError(f"Baseline checkpoint coverage below 98%; missing={missing[:8]}, unexpected={unexpected[:8]}")


def load_plain_baseline(args_file: str, checkpoint: str, device: str) -> tuple[Diffusion_Planner, Config]:
    config = Config(args_file)
    model = Diffusion_Planner(config)
    _load_weights(model, checkpoint)
    return model.to(device).eval(), config


def load_baseline(args_file: str, checkpoint: str, device: str, *, rank: int = 4,
                  alpha: float | None = None, dropout: float = 0.0) -> tuple[StyleLoRAPlanner, Config]:
    model, config = load_plain_baseline(args_file, checkpoint, device)
    return StyleLoRAPlanner(model, rank=rank, alpha=alpha, dropout=dropout).to(device), config


def prepare_diffusion_batch(batch: Dict[str, Any], device: torch.device, observation_normalizer, *,
                            return_style_context: bool = False):
    """转换缓存 batch；可选保留训练三轴损失需要的物理场景上下文。

    默认返回值与既有脚本完全相同。``return_style_context=True`` 时额外返回未经过
    observation normalization 的真值、邻车和道路信息；这些张量只用于把模型预测
    反归一化后的物理轨迹映射到可微风格指标，不会改变 DiffPlanner 的模型输入。
    """
    metadata = batch.get("metadata") if isinstance(batch, dict) else None
    batch = batch.get("tensors", batch)
    required = ("ego_current_state", "ego_future_gt", "neighbor_agents_past", "neighbors_future_gt", "lanes", "lanes_speed_limit", "lanes_has_speed_limit", "route_lanes", "route_lanes_speed_limit", "route_lanes_has_speed_limit", "static_objects")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Cache misses required DiffPlanner keys: {missing}")
    inputs = {key: batch[key].to(device) for key in required if key not in {"ego_future_gt", "neighbors_future_gt"}}
    ego, neighbours = batch["ego_future_gt"].to(device), batch["neighbors_future_gt"].to(device)
    if "neighbor_agents_future_mask" in batch:
        # DataProcessor defines this boolean tensor as *valid* future frames.
        neighbour_mask = ~batch["neighbor_agents_future_mask"].to(device).bool()
    else:
        neighbour_mask = torch.sum(neighbours[..., :3] != 0, dim=-1) == 0
    ego = torch.cat((ego[..., :2], torch.stack((ego[..., 2].cos(), ego[..., 2].sin()), dim=-1)), dim=-1)
    neighbours = torch.cat((neighbours[..., :2], torch.stack((neighbours[..., 2].cos(), neighbours[..., 2].sin()), dim=-1)), dim=-1)
    neighbours[neighbour_mask] = 0
    model_inputs = observation_normalizer(inputs)
    if not return_style_context:
        return model_inputs, (ego, neighbours, neighbour_mask)
    if not isinstance(metadata, list) or len(metadata) != ego.shape[0]:
        raise ValueError("Style-prototype training requires one metadata record per batch sample")
    physical_context = {
        # 这里保留缓存中的物理单位状态；inputs 已被 observation_normalizer 处理，不能用于三轴计算。
        "ego_current_state": inputs["ego_current_state"],
        "ego_future_gt": batch["ego_future_gt"].to(device),
        "neighbor_agents_past": inputs["neighbor_agents_past"],
        "neighbors_future_gt": batch["neighbors_future_gt"].to(device),
        # 优先沿用缓存给出的权威有效帧掩码；缺失时才采用既有的全零回退逻辑。
        "neighbors_future_valid_mask": ~neighbour_mask,
        "route_lanes_speed_limit": inputs["route_lanes_speed_limit"],
        "route_lanes_has_speed_limit": inputs["route_lanes_has_speed_limit"],
        "lanes_speed_limit": inputs["lanes_speed_limit"],
        "lanes_has_speed_limit": inputs["lanes_has_speed_limit"],
        "scene_types": tuple(str(item["scene_type"]) for item in metadata),
    }
    return model_inputs, (ego, neighbours, neighbour_mask), physical_context


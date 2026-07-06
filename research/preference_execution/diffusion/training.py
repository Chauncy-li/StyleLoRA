"""Shared training helpers for preference-conditioned diffusion."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from research.preference_execution.diffusion.style_condition import (
    build_style_condition_feature,
    resolve_style_condition_feature_set,
    style_condition_valid_mask,
)


def apply_classifier_free_dropout(
    style_value_feature: torch.Tensor,
    style_feature_valid: torch.Tensor,
    dropout_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop style conditions for CFG-style training while keeping base planning active."""

    if style_value_feature.numel() == 0:
        return style_value_feature, style_feature_valid.bool()
    valid_mask = style_feature_valid.bool().view(-1)
    keep_mask = valid_mask.clone()
    if dropout_prob > 0:
        random_keep = torch.rand_like(valid_mask.float()) > float(dropout_prob)
        keep_mask = keep_mask & random_keep.bool()
    conditioned = torch.zeros_like(style_value_feature)
    if conditioned.ndim == 1:
        if bool(keep_mask[0]):
            conditioned = style_value_feature.clone()
    else:
        conditioned[keep_mask] = style_value_feature[keep_mask]
    return conditioned, keep_mask


def prepare_preference_conditioned_batch(
    batch: Dict[str, Any],
    args: Namespace,
    *,
    train: bool,
    aug: Any = None,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Build a diffusion-loss batch with explicit preference conditions."""

    device = torch.device(args.device)
    inputs: Dict[str, Any] = {
        "ego_current_state": batch["ego_current_state"].to(device),
        "neighbor_agents_past": batch["neighbor_agents_past"][:, : args.agent_num].to(device),
        "lanes": batch["lanes"].to(device),
        "lanes_speed_limit": batch["lanes_speed_limit"].to(device),
        "lanes_has_speed_limit": batch["lanes_has_speed_limit"].to(device),
        "route_lanes": batch["route_lanes"].to(device),
        "route_lanes_speed_limit": batch["route_lanes_speed_limit"].to(device),
        "route_lanes_has_speed_limit": batch["route_lanes_has_speed_limit"].to(device),
        "static_objects": batch["static_objects"].to(device),
        "ego_agent_past": batch["ego_agent_past"].to(device),
        "neighbor_agents_past_mask": batch["neighbor_agents_past_mask"][:, : args.agent_num].to(device),
        "neighbor_agents_future_mask": batch["neighbor_agents_future_mask"][:, : args.predicted_neighbor_num].to(device),
        "lanes_mask": batch["lanes_mask"].to(device),
        "route_lanes_mask": batch["route_lanes_mask"].to(device),
    }
    ego_future = batch["ego_future_gt"].to(device)
    neighbors_future = batch["neighbors_future_gt"][:, : args.predicted_neighbor_num].to(device)

    if aug is not None and train:
        inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    ego_future = torch.cat(
        [
            ego_future[..., :2],
            torch.stack([ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1),
        ],
        dim=-1,
    )

    neighbor_mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
    neighbors_future = torch.cat(
        [
            neighbors_future[..., :2],
            torch.stack([neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1),
        ],
        dim=-1,
    )
    neighbors_future[neighbor_mask] = 0.0

    inputs = args.observation_normalizer(inputs)

    target_preference_scene_vec = batch.get("target_preference_scene_vec")
    safe_preference_scene_vec = batch.get("safe_preference_scene_vec")
    effective_preference_scene_vec = batch.get("effective_preference_scene_vec")
    local_axis_gate_values = batch.get("local_axis_gate_values")
    scene_gate_values = batch.get("scene_gate_values")
    axis_gate_values = batch.get("axis_gate_values")

    if target_preference_scene_vec is not None:
        target_preference_scene_vec = target_preference_scene_vec.to(device)
    if safe_preference_scene_vec is not None:
        safe_preference_scene_vec = safe_preference_scene_vec.to(device)
    if effective_preference_scene_vec is not None:
        effective_preference_scene_vec = effective_preference_scene_vec.to(device)
    if local_axis_gate_values is not None:
        local_axis_gate_values = local_axis_gate_values.to(device)
    if scene_gate_values is not None:
        scene_gate_values = scene_gate_values.to(device)
    if axis_gate_values is not None:
        axis_gate_values = axis_gate_values.to(device)

    feature_set = resolve_style_condition_feature_set(args)
    style_value_condition = build_style_condition_feature(
        batch["style_value_condition"].to(device),
        feature_set=feature_set,
        target_scene_vec=target_preference_scene_vec,
        effective_scene_vec=effective_preference_scene_vec,
        local_axis_gate_values=local_axis_gate_values,
    )
    style_feature_valid = style_condition_valid_mask(style_value_condition).to(device)
    if train:
        style_value_condition, style_condition_used = apply_classifier_free_dropout(
            style_value_condition,
            style_feature_valid,
            args.cfg_dropout_prob,
        )
    else:
        style_condition_used = style_feature_valid.bool().view(-1)

    inputs["style_value_condition"] = style_value_condition
    inputs["style_feature_valid"] = style_feature_valid.float()
    inputs["style_condition_used"] = style_condition_used.float()
    inputs["cfg_guidance_scale"] = float(args.cfg_guidance_scale)

    optional_condition_tensors = {
        "scene_gate_values": scene_gate_values,
        "axis_gate_values": axis_gate_values,
        "local_axis_gate_values": local_axis_gate_values,
        "target_preference_scene_vec": target_preference_scene_vec,
        "safe_preference_scene_vec": safe_preference_scene_vec,
        "effective_preference_scene_vec": effective_preference_scene_vec,
    }
    for key, value in optional_condition_tensors.items():
        if value is not None:
            inputs[key] = value

    log_info = {
        "style_feature_valid_ratio": float(style_feature_valid.float().mean().item()),
        "style_condition_used_ratio": float(style_condition_used.float().mean().item()),
        "style_condition_l2": float(torch.linalg.norm(style_value_condition, dim=-1).mean().item()),
    }
    return inputs, ego_future, neighbors_future, neighbor_mask, log_info


def build_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create a timestamped experiment directory and return it."""

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = Path(base_dir) / experiment_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def serializable_args_dict(args: Namespace) -> Dict[str, Any]:
    """Convert an argparse namespace into a JSON-safe dict."""

    payload: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if hasattr(value, "to_dict"):
            payload[key] = value.to_dict()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        else:
            payload[key] = str(value)
    return payload


def write_json(path: str, payload: object) -> None:
    """Atomically write JSON to disk."""

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    os.replace(tmp_path, path_obj)

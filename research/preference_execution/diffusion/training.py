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
    build_style_phase_time_mask,
    build_style_condition_feature,
    phase_style_num_phases,
    resolve_style_condition_feature_set,
    style_condition_valid_mask,
)
from research.preference_execution.interaction_state.schema import SCENE_GATE_ORDER


def _optional_condition_tensor(value: Any, device: torch.device) -> torch.Tensor | None:
    """Move optional sidecar tensors while treating an empty export as absent.

    V6 deliberately uses only the explicit global condition in its baseline.
    Its legacy preference-execution auxiliary vectors are therefore empty.
    Passing a `[B, 0]` tensor through to logging produced NaN reductions and
    made an absent optional feature look present.
    """

    if value is None:
        return None
    tensor = value.to(device)
    return None if tensor.numel() == 0 else tensor


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
    target_preference_global_vec = batch.get("target_preference_global_vec")
    safe_preference_global_vec = batch.get("safe_preference_global_vec")
    effective_preference_global_vec = batch.get("effective_preference_global_vec")
    local_axis_gate_values = batch.get("local_axis_gate_values")
    scene_gate_values = batch.get("scene_gate_values")
    axis_gate_values = batch.get("axis_gate_values")
    scene_buckets = batch.get("scene_bucket")

    target_preference_scene_vec = _optional_condition_tensor(target_preference_scene_vec, device)
    safe_preference_scene_vec = _optional_condition_tensor(safe_preference_scene_vec, device)
    effective_preference_scene_vec = _optional_condition_tensor(effective_preference_scene_vec, device)
    target_preference_global_vec = _optional_condition_tensor(target_preference_global_vec, device)
    safe_preference_global_vec = _optional_condition_tensor(safe_preference_global_vec, device)
    effective_preference_global_vec = _optional_condition_tensor(effective_preference_global_vec, device)
    local_axis_gate_values = _optional_condition_tensor(local_axis_gate_values, device)
    scene_gate_values = _optional_condition_tensor(scene_gate_values, device)
    axis_gate_values = _optional_condition_tensor(axis_gate_values, device)

    feature_set = resolve_style_condition_feature_set(args)
    style_value_condition = build_style_condition_feature(
        batch["style_value_condition"].to(device),
        feature_set=feature_set,
        target_scene_vec=target_preference_scene_vec,
        safe_scene_vec=safe_preference_scene_vec,
        effective_scene_vec=effective_preference_scene_vec,
        local_axis_gate_values=local_axis_gate_values,
        target_global_vec=target_preference_global_vec,
        safe_global_vec=safe_preference_global_vec,
        scene_buckets=scene_buckets,
    )
    normal_anchor_style_value_condition = batch.get(
        "normal_anchor_style_value_condition"
    )
    if normal_anchor_style_value_condition is None:
        normal_anchor_style_value_condition = torch.zeros_like(
            style_value_condition
        )
    else:
        normal_anchor_style_value_condition = (
            normal_anchor_style_value_condition.to(device=device, dtype=torch.float32)
        )
        if tuple(normal_anchor_style_value_condition.shape) != tuple(
            style_value_condition.shape
        ):
            # Legacy expanded feature sets do not define the V6 semantic
            # normal anchor. They retain the old empty-condition behavior.
            normal_anchor_style_value_condition = torch.zeros_like(
                style_value_condition
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
    inputs[
        "normal_anchor_style_value_condition"
    ] = normal_anchor_style_value_condition
    inputs["style_feature_valid"] = style_feature_valid.float()
    inputs["style_condition_used"] = style_condition_used.float()
    inputs["cfg_guidance_scale"] = float(args.cfg_guidance_scale)
    inputs["two_stage_far_recovery_mix"] = float(getattr(args, "two_stage_far_recovery_mix", 0.5))
    inputs["temporal_gate_order_margin"] = float(getattr(args, "temporal_gate_order_margin", 0.02))
    if phase_style_num_phases(feature_set) > 0:
        if scene_buckets is None:
            raise ValueError("phase-wise style conditioning requires scene_bucket metadata in the batch.")
        phase_time_mask = build_style_phase_time_mask(
            feature_set,
            scene_buckets,
            future_len=int(args.future_len),
            include_current=True,
            device=device,
            dtype=style_value_condition.dtype,
            two_stage_split_ratio=float(getattr(args, "two_stage_split_ratio", 0.45)),
            two_stage_transition_ratio=float(getattr(args, "two_stage_transition_ratio", 0.18)),
        )
        if phase_time_mask is not None:
            inputs["phase_time_mask"] = phase_time_mask

    optional_condition_tensors = {
        "scene_gate_values": scene_gate_values,
        "axis_gate_values": axis_gate_values,
        "local_axis_gate_values": local_axis_gate_values,
        "target_preference_scene_vec": target_preference_scene_vec,
        "safe_preference_scene_vec": safe_preference_scene_vec,
        "effective_preference_scene_vec": effective_preference_scene_vec,
        "target_preference_global_vec": target_preference_global_vec,
        "safe_preference_global_vec": safe_preference_global_vec,
        "effective_preference_global_vec": effective_preference_global_vec,
        "preference_ego_current_xycs": batch["ego_current_state"][..., :4].to(device),
        "preference_ego_current_state_raw": batch["ego_current_state"].to(device),
        "preference_ego_agent_past_raw": batch["ego_agent_past"].to(device),
        "preference_neighbor_current_xycs": batch["neighbor_agents_past"][:, : args.predicted_neighbor_num, -1, :4].to(device),
        "preference_neighbor_agents_past_raw": batch["neighbor_agents_past"][
            :, : args.predicted_neighbor_num
        ].to(device),
        "preference_neighbor_agents_past_mask_raw": batch[
            "neighbor_agents_past_mask"
        ][:, : args.predicted_neighbor_num].to(device),
        "preference_lanes_raw": batch["lanes"].to(device),
        "preference_route_lanes_raw": batch["route_lanes"].to(device),
        "preference_route_lanes_speed_limit_raw": batch["route_lanes_speed_limit"].to(device),
        "preference_route_lanes_has_speed_limit_raw": batch["route_lanes_has_speed_limit"].to(device),
        "preference_route_lanes_mask_raw": batch["route_lanes_mask"].to(device),
        "preference_lanes_speed_limit_raw": batch["lanes_speed_limit"].to(device),
        "preference_lanes_has_speed_limit_raw": batch["lanes_has_speed_limit"].to(device),
        "preference_lanes_mask_raw": batch["lanes_mask"].to(device),
    }
    for key, value in optional_condition_tensors.items():
        if value is not None:
            inputs[key] = value
    inputs["scene_bucket"] = scene_buckets

    log_info = {
        "style_feature_valid_ratio": float(style_feature_valid.float().mean().item()),
        "style_condition_used_ratio": float(style_condition_used.float().mean().item()),
        "style_condition_l2": float(torch.linalg.norm(style_value_condition, dim=-1).mean().item()),
        "normal_anchor_active_ratio": float(
            style_condition_valid_mask(normal_anchor_style_value_condition)
            .float()
            .mean()
            .item()
        ),
    }
    if target_preference_scene_vec is not None:
        log_info["target_preference_l1"] = float(target_preference_scene_vec.abs().mean().item())
    if safe_preference_scene_vec is not None:
        log_info["safe_preference_l1"] = float(safe_preference_scene_vec.abs().mean().item())
    if effective_preference_scene_vec is not None:
        log_info["effective_preference_l1"] = float(effective_preference_scene_vec.abs().mean().item())
    if target_preference_scene_vec is not None and safe_preference_scene_vec is not None:
        log_info["target_safe_gap_l1"] = float(
            (target_preference_scene_vec - safe_preference_scene_vec).abs().mean().item()
        )
    if safe_preference_scene_vec is not None and effective_preference_scene_vec is not None:
        log_info["safe_effective_gap_l1"] = float(
            (safe_preference_scene_vec - effective_preference_scene_vec).abs().mean().item()
        )
    if target_preference_global_vec is not None:
        log_info["target_preference_global_l1"] = float(target_preference_global_vec.abs().mean().item())
    if safe_preference_global_vec is not None:
        log_info["safe_preference_global_l1"] = float(safe_preference_global_vec.abs().mean().item())
    if effective_preference_global_vec is not None:
        log_info["effective_preference_global_l1"] = float(effective_preference_global_vec.abs().mean().item())
    if "phase_time_mask" in inputs:
        phase_time_mask = torch.as_tensor(inputs["phase_time_mask"], device=device)
        log_info["phase_mask_mean"] = float(phase_time_mask.mean().item())
        if phase_time_mask.shape[1] >= 2:
            log_info["phase_mask_stage0_mean"] = float(phase_time_mask[:, 0].mean().item())
            log_info["phase_mask_stage1_mean"] = float(phase_time_mask[:, 1].mean().item())
    if local_axis_gate_values is not None:
        log_info["local_axis_gate_mean"] = float(local_axis_gate_values.mean().item())
        log_info["local_axis_gate_min"] = float(local_axis_gate_values.min().item())
    if scene_gate_values is not None:
        log_info["scene_gate_mean"] = float(scene_gate_values.mean().item())
        log_info["scene_gate_max"] = float(scene_gate_values.max().item())
        if scene_buckets is not None:
            normalized_scene_buckets = [scene_buckets] if isinstance(scene_buckets, str) else list(scene_buckets)
            scene_bucket_to_index = {scene_bucket: index for index, scene_bucket in enumerate(SCENE_GATE_ORDER)}
            target_scene_index = torch.as_tensor(
                [
                    scene_bucket_to_index.get(str(scene_bucket), -1)
                    for scene_bucket in normalized_scene_buckets
                ],
                device=device,
                dtype=torch.long,
            )
            valid_scene_mask = target_scene_index >= 0
            if bool(valid_scene_mask.any()):
                predicted_scene_index = torch.argmax(scene_gate_values, dim=-1)
                log_info["scene_gate_alignment_rate"] = float(
                    (predicted_scene_index[valid_scene_mask] == target_scene_index[valid_scene_mask]).float().mean().item()
                )
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

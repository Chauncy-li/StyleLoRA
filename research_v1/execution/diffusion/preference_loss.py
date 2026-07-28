"""Auxiliary preference-alignment losses for controllable execution."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

from baseline.model.style_planner.loss.diff_loss import _extract_diffusion_prediction
from research_v1.execution.diffusion.temporal_condition import (
    build_two_stage_gate_targets,
    build_two_stage_global_targets,
)


def _clamp01(value: torch.Tensor) -> torch.Tensor:
    return torch.clamp(value, min=0.0, max=1.0)


def _score_rising(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    scale = max(float(high) - float(low), 1e-6)
    return _clamp01((value - float(low)) / scale)


def _score_falling(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    scale = max(float(high) - float(low), 1e-6)
    return _clamp01((float(high) - value) / scale)


def _normalize_scene_bucket_batch(scene_buckets: Sequence[str] | str | None, batch_size: int) -> list[str]:
    if scene_buckets is None:
        return ["none"] * max(batch_size, 1)
    if isinstance(scene_buckets, str):
        return [scene_buckets] * max(batch_size, 1)
    normalized = [str(scene_bucket) for scene_bucket in scene_buckets]
    if not normalized:
        return ["none"] * max(batch_size, 1)
    if len(normalized) == 1 and batch_size > 1:
        return normalized * batch_size
    if len(normalized) != batch_size:
        raise ValueError(
            f"Expected {batch_size} scene buckets, got {len(normalized)}: {normalized!r}"
        )
    return normalized


def _route_speed_limit_mps(
    route_lanes_speed_limit: torch.Tensor,
    route_lanes_has_speed_limit: torch.Tensor,
    route_lanes_mask: torch.Tensor,
    lanes_speed_limit: torch.Tensor,
    lanes_has_speed_limit: torch.Tensor,
    lanes_mask: torch.Tensor,
) -> torch.Tensor:
    def _masked_mean(limit_tensor: torch.Tensor, has_limit: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        point_valid = torch.as_tensor(mask_tensor).bool()
        while point_valid.dim() > 1:
            point_valid = point_valid.any(dim=-1)
        valid = point_valid & torch.as_tensor(has_limit).bool().reshape(point_valid.shape)
        values = torch.as_tensor(limit_tensor, dtype=torch.float32).reshape(point_valid.shape)
        valid_values = torch.where(valid, values, torch.zeros_like(values))
        denom = valid.float().sum()
        if float(denom.item()) <= 0.0:
            return torch.zeros((), dtype=torch.float32, device=values.device)
        return valid_values.sum() / denom.clamp_min(1.0)

    route_speed = _masked_mean(route_lanes_speed_limit, route_lanes_has_speed_limit, route_lanes_mask)
    if float(route_speed.item()) > 1e-6:
        return route_speed
    return _masked_mean(lanes_speed_limit, lanes_has_speed_limit, lanes_mask)


def _diff_1d(values: torch.Tensor, dt: float) -> torch.Tensor:
    if values.numel() <= 1:
        return values.new_zeros((0,))
    return (values[1:] - values[:-1]) / max(float(dt), 1e-6)


def _speed_from_xy(xy: torch.Tensor, dt: float) -> torch.Tensor:
    if xy.shape[0] <= 1:
        return xy.new_zeros((0,))
    delta = xy[1:] - xy[:-1]
    return torch.linalg.norm(delta, dim=-1) / max(float(dt), 1e-6)


def _heading_change_from_xycs(xycs: torch.Tensor) -> torch.Tensor:
    if xycs.shape[0] <= 1:
        return xycs.new_zeros(())
    heading = torch.atan2(xycs[:, 3], xycs[:, 2])
    delta = heading[-1] - heading[0]
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs()


def _scene_proxy_from_prediction(
    scene_bucket: str,
    ego_current_xycs: torch.Tensor,
    ego_future_xycs: torch.Tensor,
    neighbors_current_xycs: torch.Tensor,
    neighbors_future_xycs: torch.Tensor,
    neighbor_future_mask: torch.Tensor,
    *,
    route_speed_limit_mps: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = ego_future_xycs.device
    dtype = ego_future_xycs.dtype
    ego_full = torch.cat([ego_current_xycs[None, :4], ego_future_xycs[..., :4]], dim=0)
    ego_xy = ego_full[:, :2]
    speed = _speed_from_xy(ego_xy, dt)
    if speed.numel() > 0:
        speed_with_current = torch.cat([speed[:1], speed], dim=0)
    else:
        speed_with_current = ego_full.new_zeros((ego_full.shape[0],))
    accel = _diff_1d(speed, dt)
    jerk = _diff_1d(accel, dt)
    accel_peak = torch.relu(accel).max() if accel.numel() > 0 else torch.zeros((), device=device, dtype=dtype)
    brake_peak = torch.relu(-accel).max() if accel.numel() > 0 else torch.zeros((), device=device, dtype=dtype)
    jerk_peak = jerk.abs().max() if jerk.numel() > 0 else torch.zeros((), device=device, dtype=dtype)

    if scene_bucket == "straight_free_drive":
        mean_speed = speed.mean() if speed.numel() > 0 else torch.zeros((), device=device, dtype=dtype)
        speed_ratio = mean_speed / route_speed_limit_mps.clamp_min(1e-3)
        speed_preference = _score_rising(speed_ratio, 0.55, 0.95)
        longitudinal_intensity = _clamp01(
            0.55 * _score_rising(accel_peak, 0.6, 1.8)
            + 0.45 * _score_rising(jerk_peak, 12.0, 45.0)
        )
        smoothness = _clamp01(
            0.60 * _score_falling(jerk_peak, 12.0, 45.0)
            + 0.40 * _score_falling(brake_peak, 0.6, 3.0)
        )
        return torch.stack([speed_preference, longitudinal_intensity, smoothness]), torch.ones(
            (3,), device=device, dtype=torch.bool
        )

    neighbors_full = torch.cat([neighbors_current_xycs[:, None, :4], neighbors_future_xycs[..., :4]], dim=1)
    current_valid = torch.linalg.norm(neighbors_current_xycs[:, :2], dim=-1) > 1e-4
    future_valid = ~neighbor_future_mask.bool()
    neighbor_valid = torch.cat([current_valid[:, None], future_valid], dim=1)
    rel_xy = neighbors_full[..., :2] - ego_full[None, :, :2]

    if scene_bucket == "straight_car_follow":
        lead_valid = neighbor_valid & (rel_xy[..., 0] > 0.0) & (rel_xy[..., 1].abs() < 4.0)
        if bool(lead_valid.any()):
            large = torch.full_like(rel_xy[..., 0], 1e6)
            lead_gap = torch.where(lead_valid, rel_xy[..., 0], large).amin(dim=0)
            step_has_lead = lead_valid.any(dim=0)
            min_gap = lead_gap[step_has_lead].amin()
            thw_series = lead_gap / speed_with_current.clamp_min(0.1)
            min_thw = thw_series[step_has_lead].amin()
            speed_drop_ratio = (
                (speed.max() - speed.min()) / speed.max().clamp_min(1e-3)
                if speed.numel() > 0
                else torch.zeros((), device=device, dtype=dtype)
            )
            headway_margin = _clamp01(
                0.55 * _score_rising(min_thw, 1.1, 3.0)
                + 0.45 * _score_rising(min_gap, 8.0, 26.0)
            )
            response_decisiveness = _clamp01(
                0.50 * _score_rising(brake_peak, 0.5, 2.4)
                + 0.50 * _score_rising(speed_drop_ratio, 0.03, 0.22)
            )
            response_smoothness = _clamp01(
                0.55 * _score_falling(brake_peak, 0.5, 2.4)
                + 0.45 * _score_falling(speed_drop_ratio, 0.03, 0.22)
            )
            return torch.stack(
                [headway_margin, response_decisiveness, response_smoothness]
            ), torch.ones((3,), device=device, dtype=torch.bool)
        return torch.zeros((3,), device=device, dtype=dtype), torch.zeros((3,), device=device, dtype=torch.bool)

    if scene_bucket == "straight_lane_change":
        lateral_offset = (ego_full[:, 1] - ego_full[0, 1]).abs()
        lateral_speed = _diff_1d(ego_full[:, 1], dt).abs()
        lateral_speed_peak = lateral_speed.max() if lateral_speed.numel() > 0 else torch.zeros((), device=device, dtype=dtype)
        onset_activation = torch.sigmoid((lateral_offset[1:] - 0.5) / 0.10) if ego_full.shape[0] > 1 else ego_full.new_zeros((0,))
        if onset_activation.numel() > 0 and float(onset_activation.sum().item()) > 1e-6:
            onset_index = torch.arange(
                1,
                ego_full.shape[0],
                device=device,
                dtype=dtype,
            )
            onset_step = (onset_activation * onset_index).sum() / onset_activation.sum().clamp_min(1e-6)
        else:
            onset_step = ego_full.new_tensor(35.0)
        heading_change = _heading_change_from_xycs(ego_full)

        active_steps = lateral_offset >= 0.5
        merge_valid = neighbor_valid & (rel_xy[..., 1].abs() < 4.5) & active_steps[None, :]
        if bool(merge_valid.any()):
            large = torch.full_like(rel_xy[..., 0], 1e6)
            merge_gap = torch.where(merge_valid, rel_xy[..., 0].abs(), large).amin()
            gap_acceptance = _score_falling(merge_gap, 10.0, 28.0)
            lateral_commitment = _clamp01(
                0.55 * _score_falling(onset_step, 8.0, 35.0)
                + 0.45 * _score_rising(lateral_speed_peak, 0.5, 1.8)
            )
            execution_smoothness = _clamp01(
                0.60 * _score_falling(lateral_speed_peak, 0.6, 1.8)
                + 0.40 * _score_falling(heading_change, 0.03, 0.18)
            )
            valid_mask = torch.tensor(
                [True, bool(onset_activation.max().item() > 0.25), bool(onset_activation.max().item() > 0.25)],
                device=device,
            )
            return torch.stack(
                [gap_acceptance, lateral_commitment, execution_smoothness]
            ), valid_mask.bool()
        return torch.zeros((3,), device=device, dtype=dtype), torch.zeros((3,), device=device, dtype=torch.bool)

    return torch.zeros((3,), device=device, dtype=dtype), torch.zeros((3,), device=device, dtype=torch.bool)


def compute_preference_aux_losses(
    *,
    decoder_output: Dict[str, torch.Tensor],
    inputs: Dict[str, Any],
    neighbors_future: torch.Tensor,
    neighbor_future_mask: torch.Tensor,
    state_normalizer: Any,
    model_type: str,
    dt: float,
) -> Dict[str, torch.Tensor]:
    reference_tensor = next(
        (value for value in decoder_output.values() if torch.is_tensor(value)),
        neighbors_future,
    )
    zero = reference_tensor.new_tensor(0.0)
    if model_type != "x_start":
        return {
            "preference_proxy_loss": zero,
            "preference_proxy_mae": zero,
            "temporal_near_condition_loss": zero,
            "temporal_far_condition_loss": zero,
            "temporal_near_gate_target_loss": zero,
            "temporal_far_gate_target_loss": zero,
            "temporal_gate_order_loss": zero,
            "temporal_near_gate_mean": zero,
            "temporal_far_gate_mean": zero,
            "temporal_stage_gap_mean": zero,
        }

    prediction = _extract_diffusion_prediction(
        decoder_output,
        model_type=model_type,
        future_steps=int(neighbors_future.shape[2]),
    )
    prediction_xycs = state_normalizer.inverse(prediction)[:, 0, :, :4]

    scene_buckets = _normalize_scene_bucket_batch(inputs.get("scene_bucket"), int(prediction.shape[0]))
    target_scene_vec = inputs.get("effective_preference_scene_vec")
    if target_scene_vec is None:
        return {
            "preference_proxy_loss": zero,
            "preference_proxy_mae": zero,
            "temporal_near_condition_loss": zero,
            "temporal_far_condition_loss": zero,
            "temporal_near_gate_target_loss": zero,
            "temporal_far_gate_target_loss": zero,
            "temporal_gate_order_loss": zero,
            "temporal_near_gate_mean": zero,
            "temporal_far_gate_mean": zero,
            "temporal_stage_gap_mean": zero,
        }
    target_scene_vec = torch.as_tensor(target_scene_vec, dtype=torch.float32, device=prediction.device)

    proxy_losses: list[torch.Tensor] = []
    proxy_vectors: list[torch.Tensor] = []
    valid_vectors: list[torch.Tensor] = []
    for batch_index, scene_bucket in enumerate(scene_buckets):
        route_speed_limit = _route_speed_limit_mps(
            inputs["preference_route_lanes_speed_limit_raw"][batch_index],
            inputs["preference_route_lanes_has_speed_limit_raw"][batch_index],
            inputs["preference_route_lanes_mask_raw"][batch_index],
            inputs["preference_lanes_speed_limit_raw"][batch_index],
            inputs["preference_lanes_has_speed_limit_raw"][batch_index],
            inputs["preference_lanes_mask_raw"][batch_index],
        ).to(device=prediction.device, dtype=torch.float32)
        route_speed_limit = route_speed_limit.clamp_min(1.0)
        proxy_vec, valid_mask = _scene_proxy_from_prediction(
            scene_bucket,
            inputs["preference_ego_current_xycs"][batch_index],
            prediction_xycs[batch_index],
            inputs["preference_neighbor_current_xycs"][batch_index],
            neighbors_future[batch_index],
            neighbor_future_mask[batch_index],
            route_speed_limit_mps=route_speed_limit,
            dt=dt,
        )
        proxy_vectors.append(proxy_vec)
        valid_vectors.append(valid_mask)
        if bool(valid_mask.any()):
            proxy_losses.append((proxy_vec[valid_mask] - target_scene_vec[batch_index][valid_mask]).abs().mean())

    preference_proxy_loss = torch.stack(proxy_losses).mean() if proxy_losses else zero

    near_condition = decoder_output.get("temporal_near_condition")
    far_condition = decoder_output.get("temporal_far_condition")
    near_gate = decoder_output.get("temporal_near_gate")
    far_gate = decoder_output.get("temporal_far_gate")
    safe_global = inputs.get("safe_preference_global_vec")
    effective_global = inputs.get("effective_preference_global_vec")
    temporal_near_condition_loss = zero
    temporal_far_condition_loss = zero
    temporal_near_gate_target_loss = zero
    temporal_far_gate_target_loss = zero
    temporal_gate_order_loss = zero
    temporal_stage_gap_mean = zero
    if safe_global is not None and effective_global is not None:
        safe_global = torch.as_tensor(safe_global, dtype=torch.float32, device=prediction.device)
        effective_global = torch.as_tensor(effective_global, dtype=torch.float32, device=prediction.device)
        near_target, far_target = build_two_stage_global_targets(
            safe_global,
            effective_global,
            far_recovery_mix=float(inputs.get("two_stage_far_recovery_mix", 0.5)),
        )
        if near_condition is not None:
            active_mask = (near_target.abs() > 1e-5) | (near_condition.abs() > 1e-5)
            if bool(active_mask.any()):
                temporal_near_condition_loss = (near_condition[active_mask] - near_target[active_mask]).abs().mean()
        if far_condition is not None:
            active_mask = (far_target.abs() > 1e-5) | (far_condition.abs() > 1e-5)
            if bool(active_mask.any()):
                temporal_far_condition_loss = (far_condition[active_mask] - far_target[active_mask]).abs().mean()
        near_gate_target, far_gate_target, gate_valid_mask = build_two_stage_gate_targets(
            safe_global,
            near_target,
            far_target,
        )
        if near_gate is not None and bool(gate_valid_mask.any()):
            temporal_near_gate_target_loss = (
                near_gate[gate_valid_mask] - near_gate_target[gate_valid_mask]
            ).abs().mean()
        if far_gate is not None and bool(gate_valid_mask.any()):
            temporal_far_gate_target_loss = (
                far_gate[gate_valid_mask] - far_gate_target[gate_valid_mask]
            ).abs().mean()
        if near_gate is not None and far_gate is not None and bool(gate_valid_mask.any()):
            order_margin = float(inputs.get("temporal_gate_order_margin", 0.02))
            order_mask = gate_valid_mask & ((far_gate_target - near_gate_target) > 1e-4)
            if bool(order_mask.any()):
                temporal_gate_order_loss = torch.relu(
                    near_gate[order_mask] - far_gate[order_mask] + order_margin
                ).mean()
    if near_condition is not None and far_condition is not None:
        temporal_stage_gap_mean = (far_condition - near_condition).mean()

    metrics: Dict[str, torch.Tensor] = {
        "preference_proxy_loss": preference_proxy_loss,
        "preference_proxy_mae": zero,
        "temporal_near_condition_loss": temporal_near_condition_loss,
        "temporal_far_condition_loss": temporal_far_condition_loss,
        "temporal_near_gate_target_loss": temporal_near_gate_target_loss,
        "temporal_far_gate_target_loss": temporal_far_gate_target_loss,
        "temporal_gate_order_loss": temporal_gate_order_loss,
        "temporal_near_gate_mean": zero,
        "temporal_far_gate_mean": zero,
        "temporal_stage_gap_mean": temporal_stage_gap_mean,
    }
    if proxy_vectors:
        stacked_proxy = torch.stack(proxy_vectors, dim=0)
        stacked_valid = torch.stack(valid_vectors, dim=0)
        if bool(stacked_valid.any()):
            metrics["preference_proxy_mae"] = (
                (stacked_proxy[stacked_valid] - target_scene_vec[stacked_valid]).abs().mean()
            )
    if "temporal_near_gate" in decoder_output:
        metrics["temporal_near_gate_mean"] = decoder_output["temporal_near_gate"].mean()
    if "temporal_far_gate" in decoder_output:
        metrics["temporal_far_gate_mean"] = decoder_output["temporal_far_gate"].mean()
    return metrics

"""Inference-only repair of a styled ego trajectory around a paired baseline path."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

from stylelora.lora.safety.trajectory_acceptance import (
    BaselineRelativeCandidateValidator,
    CandidateAcceptanceResult,
)


@dataclass(frozen=True)
class TrajectoryRepairResult:
    """Selected joint prediction and an auditable summary of the repair."""

    prediction: torch.Tensor
    applied: bool
    baseline_fallback: bool
    longitudinal_scale: float
    lateral_scale: float
    candidate_attempts: int
    baseline_hard_valid: bool
    failure_reasons: Tuple[str, ...]
    selected_metrics: Dict[str, float]
    collision_triggered: bool = False
    drivable_triggered: bool = False
    style_offroad_fraction: float = 0.0
    selected_offroad_fraction: float = 0.0
    drivable_check_time_ms: float = 0.0


def _normalized_heading(states: torch.Tensor) -> torch.Tensor:
    heading = states[..., 2:4]
    return heading / torch.linalg.vector_norm(heading, dim=-1, keepdim=True).clamp_min(1e-6)


def _headings_from_xy(xy: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    origin = torch.zeros_like(xy[..., :1, :])
    segment = torch.diff(torch.cat((origin, xy), dim=-2), dim=-2)
    norm = torch.linalg.vector_norm(segment, dim=-1, keepdim=True)
    return torch.where(norm > 1e-4, segment / norm.clamp_min(1e-6), fallback)


def compose_baseline_anchored_ego(
    styled_ego: torch.Tensor,
    baseline_ego: torch.Tensor,
    *,
    longitudinal_scale: float,
    lateral_scale: float,
) -> torch.Tensor:
    """Scale longitudinal/lateral styled residuals without changing the model weights or rho."""
    if styled_ego.shape != baseline_ego.shape or styled_ego.ndim < 2:
        raise ValueError("styled_ego and baseline_ego must have the same [..., T, D] shape")
    if styled_ego.shape[-1] < 4:
        raise ValueError("trajectory states must contain [x, y, cos, sin]")
    if not 0.0 <= float(longitudinal_scale) <= 1.0:
        raise ValueError("longitudinal_scale must lie in [0, 1]")
    if not 0.0 <= float(lateral_scale) <= 1.0:
        raise ValueError("lateral_scale must lie in [0, 1]")

    tangent = _normalized_heading(baseline_ego)
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
    displacement = styled_ego[..., :2] - baseline_ego[..., :2]
    longitudinal = (displacement * tangent).sum(dim=-1, keepdim=True)
    lateral = (displacement * normal).sum(dim=-1, keepdim=True)
    repaired_xy = (
        baseline_ego[..., :2]
        + float(longitudinal_scale) * longitudinal * tangent
        + float(lateral_scale) * lateral * normal
    )

    repaired = styled_ego.clone()
    repaired[..., :2] = repaired_xy
    repaired[..., 2:4] = _headings_from_xy(repaired_xy, tangent)
    return repaired


def compose_interpolated_ego_batch(
    styled_ego: torch.Tensor,
    baseline_ego: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Create all baseline-to-style interpolations in one broadcast operation."""
    if styled_ego.shape != baseline_ego.shape or styled_ego.ndim != 2:
        raise ValueError("styled_ego and baseline_ego must match [T, D]")
    if styled_ego.shape[-1] < 4:
        raise ValueError("trajectory states must contain [x, y, cos, sin]")
    if scales.ndim != 1 or scales.numel() == 0:
        raise ValueError("scales must be a non-empty one-dimensional tensor")
    if bool(((scales < 0.0) | (scales > 1.0)).any()):
        raise ValueError("scales must lie in [0, 1]")

    scale = scales.to(device=styled_ego.device, dtype=styled_ego.dtype)[:, None, None]
    result = styled_ego.unsqueeze(0).expand(scale.shape[0], -1, -1).clone()
    xy = baseline_ego[None, :, :2] + scale * (
        styled_ego[None, :, :2] - baseline_ego[None, :, :2]
    )
    fallback = _normalized_heading(baseline_ego)[None].expand(scale.shape[0], -1, -1)
    result[..., :2] = xy
    result[..., 2:4] = _headings_from_xy(xy, fallback)
    return result


def _vehicle_discs(
    states: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    *,
    origin_to_center: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized three-disc vehicle footprint for [..., T, 4] trajectories."""
    heading = states[..., 2:4]
    heading = heading / torch.linalg.vector_norm(
        heading, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    length = torch.as_tensor(length, dtype=states.dtype, device=states.device)
    width = torch.as_tensor(width, dtype=states.dtype, device=states.device)
    offset = (0.5 * (length - width)).clamp_min(0.0)
    offsets = torch.stack((-offset, torch.zeros_like(offset), offset), dim=-1)
    vehicle_center = states[..., :2] + heading * float(origin_to_center)
    # Insert the trajectory-time axis between per-vehicle sizes and disc offsets.
    offsets = offsets.unsqueeze(-2).unsqueeze(-1)
    centers = vehicle_center.unsqueeze(-2) + heading.unsqueeze(-2) * offsets
    return centers, 0.5 * width


def batched_three_disc_min_clearance(
    ego_candidates: torch.Tensor,
    neighbor_futures: torch.Tensor,
    neighbor_lengths: torch.Tensor,
    neighbor_widths: torch.Tensor,
    *,
    ego_length: float,
    ego_width: float,
    ego_origin_to_center: float,
    collision_margin_m: float,
) -> torch.Tensor:
    """Return one minimum clearance per ego candidate using fully batched geometry."""
    if ego_candidates.ndim != 3 or ego_candidates.shape[-1] < 4:
        raise ValueError("ego_candidates must be [K, T, 4]")
    if neighbor_futures.ndim != 3 or neighbor_futures.shape[-1] < 4:
        raise ValueError("neighbor_futures must be [N, T, 4]")
    if neighbor_futures.shape[1] != ego_candidates.shape[1]:
        raise ValueError("ego and neighbor horizons must match")
    if neighbor_futures.shape[0] == 0:
        return ego_candidates.new_full((ego_candidates.shape[0],), float("inf"))

    neighbor_lengths = neighbor_lengths.to(ego_candidates).clamp_min(0.5)
    neighbor_widths = neighbor_widths.to(ego_candidates).clamp_min(0.5)
    neighbor_lengths = torch.maximum(neighbor_lengths, neighbor_widths)
    ego_centers, ego_radius = _vehicle_discs(
        ego_candidates,
        ego_candidates.new_tensor(float(ego_length)),
        ego_candidates.new_tensor(float(ego_width)),
        origin_to_center=ego_origin_to_center,
    )
    neighbor_centers, neighbor_radius = _vehicle_discs(
        neighbor_futures.to(ego_candidates),
        neighbor_lengths,
        neighbor_widths,
    )
    delta = (
        ego_centers[:, None, :, :, None, :]
        - neighbor_centers[None, :, :, None, :, :]
    )
    distance = torch.linalg.vector_norm(delta, dim=-1)
    clearance = distance - (
        ego_radius
        + neighbor_radius[None, :, None, None, None]
        + float(collision_margin_m)
    )
    return clearance.flatten(1).amin(dim=1)


def _validated_scales(values: Sequence[float], name: str) -> Tuple[float, ...]:
    scales = tuple(float(value) for value in values)
    if not scales or any(value < 0.0 or value > 1.0 for value in scales):
        raise ValueError(f"{name} must contain values in [0, 1]")
    if any(right >= left for left, right in zip(scales, scales[1:])):
        raise ValueError(f"{name} must be strictly decreasing")
    if scales[0] != 1.0 or scales[-1] != 0.0:
        raise ValueError(f"{name} must start at 1.0 and end at 0.0")
    return scales


def candidate_scale_grid(
    longitudinal_scales: Sequence[float],
    lateral_scales: Sequence[float],
) -> Tuple[Tuple[float, float], ...]:
    """Prioritize preserving longitudinal style, then preserving lateral style."""
    longitudinal = _validated_scales(longitudinal_scales, "longitudinal_scales")
    lateral = _validated_scales(lateral_scales, "lateral_scales")
    return tuple((long_scale, lat_scale) for long_scale in longitudinal for lat_scale in lateral)


class BaselineAnchoredTrajectoryRepair:
    """Choose the least modified safe trajectory derived from styled and baseline outputs."""

    def __init__(self, config: Any, *, step_interval: float) -> None:
        self.validator = BaselineRelativeCandidateValidator(config, step_interval=step_interval)
        self.step_interval = float(step_interval)
        self.mode = str(getattr(config, "trajectory_repair_mode", "legacy"))
        if self.mode not in {"legacy", "collision_parallel"}:
            raise ValueError("trajectory_repair_mode must be legacy or collision_parallel")
        self.longitudinal_scales = _validated_scales(
            getattr(config, "trajectory_repair_longitudinal_scales", [1.0, 0.75, 0.5, 0.25, 0.0]),
            "trajectory_repair_longitudinal_scales",
        )
        self.lateral_scales = _validated_scales(
            getattr(config, "trajectory_repair_lateral_scales", [1.0, 0.75, 0.5, 0.25, 0.0]),
            "trajectory_repair_lateral_scales",
        )
        self.collision_scales = _validated_scales(
            getattr(
                config,
                "collision_repair_scales",
                [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0],
            ),
            "collision_repair_scales",
        )
        self.collision_horizon_s = float(
            getattr(config, "collision_repair_horizon_s", 2.0)
        )
        self.collision_min_clearance_m = float(
            getattr(config, "collision_repair_min_clearance_m", 0.5)
        )
        self.check_drivable_area = bool(
            getattr(config, "repair_check_drivable_area", False)
        )
        self.drivable_horizon_s = float(
            getattr(config, "repair_drivable_horizon_s", 2.0)
        )
        # Repair-specific strictness.  The older bounded-style validator keeps
        # its original defaults because this object owns its validator instance.
        self.validator.max_lateral_deviation_m = float(
            getattr(config, "trajectory_repair_max_mean_lateral_deviation_m", 0.5)
        )
        self.validator.selector.max_offroad_fraction = float(
            getattr(config, "trajectory_repair_max_offroad_fraction", 0.0)
        )
        self.validator.selector.max_wrong_way_fraction = float(
            getattr(config, "trajectory_repair_max_wrong_way_fraction", 0.05)
        )
        self.validator.selector.hard_min_clearance_m = float(
            getattr(config, "trajectory_repair_min_clearance_m", 0.5)
        )
        if self.validator.max_lateral_deviation_m < 0.0:
            raise ValueError("trajectory_repair_max_mean_lateral_deviation_m must be non-negative")
        if self.validator.selector.max_offroad_fraction < 0.0:
            raise ValueError("trajectory_repair_max_offroad_fraction must be non-negative")
        if self.validator.selector.max_wrong_way_fraction < 0.0:
            raise ValueError("trajectory_repair_max_wrong_way_fraction must be non-negative")
        if self.validator.selector.hard_min_clearance_m < 0.0:
            raise ValueError("trajectory_repair_min_clearance_m must be non-negative")
        if self.collision_horizon_s <= 0.0:
            raise ValueError("collision_repair_horizon_s must be positive")
        if self.collision_min_clearance_m < 0.0:
            raise ValueError("collision_repair_min_clearance_m must be non-negative")
        if self.drivable_horizon_s <= 0.0:
            raise ValueError("repair_drivable_horizon_s must be positive")

    def _short_offroad_fractions(
        self,
        ego_candidates: torch.Tensor,
        ego_state: Any,
        map_api: Any,
        point_cache: Dict[tuple[float, float], bool],
    ) -> tuple[torch.Tensor, float]:
        """Check short-horizon vehicle footprints, caching repeated map queries."""
        if not self.check_drivable_area:
            return ego_candidates.new_zeros((ego_candidates.shape[0],)), 0.0

        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        started = time.perf_counter()
        selector = self.validator.selector
        horizon = min(
            ego_candidates.shape[1],
            max(1, int(round(self.drivable_horizon_s / self.step_interval))),
        )
        states = ego_candidates[:, :horizon].detach().float().cpu().numpy()
        local_xy = states[..., :2]
        origin_heading = float(ego_state.rear_axle.heading)
        # The conversion itself is vectorized; map lookup remains a scalar NuPlan API.
        c, s = float(np.cos(origin_heading)), float(np.sin(origin_heading))
        origin = np.asarray(
            [ego_state.rear_axle.x, ego_state.rear_axle.y], dtype=np.float64
        )
        global_xy = np.stack(
            (
                local_xy[..., 0] * c - local_xy[..., 1] * s,
                local_xy[..., 0] * s + local_xy[..., 1] * c,
            ),
            axis=-1,
        ) + origin
        heading = np.arctan2(states[..., 3], states[..., 2]) + origin_heading
        forward = np.stack((np.cos(heading), np.sin(heading)), axis=-1)
        lateral = np.stack((-np.sin(heading), np.cos(heading)), axis=-1)
        vehicle = selector.ego_vehicle
        longitudinal_offsets = (
            -float(vehicle.rear_length),
            float(vehicle.rear_axle_to_center),
            float(vehicle.front_length),
        )
        lateral_offsets = (-0.5 * float(vehicle.width), 0.0, 0.5 * float(vehicle.width))
        points = np.stack(
            [
                global_xy + long_offset * forward + lat_offset * lateral
                for long_offset in longitudinal_offsets
                for lat_offset in lateral_offsets
            ],
            axis=-2,
        )
        indices = list(range(0, horizon, selector.drivable_check_stride))
        if indices[-1] != horizon - 1:
            indices.append(horizon - 1)
        points = points[:, indices]
        fractions = []
        try:
            for candidate_points in points:
                valid = []
                for xy in candidate_points.reshape(-1, 2):
                    key = (float(xy[0]), float(xy[1]))
                    if key not in point_cache:
                        point_cache[key] = bool(
                            map_api.is_in_layer(
                                Point2D(*key), SemanticMapLayer.DRIVABLE_AREA
                            )
                        )
                    valid.append(point_cache[key])
                fractions.append(1.0 - float(np.mean(valid)) if valid else 0.0)
        except Exception:
            # Preserve the selector's existing fail-open behavior for map API errors.
            fractions = [0.0] * ego_candidates.shape[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ego_candidates.new_tensor(fractions), elapsed_ms

    def _collision_clearance(
        self,
        ego_candidates: torch.Tensor,
        joint_prediction: torch.Tensor,
        raw_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compare K ego candidates to model and constant-velocity neighbor futures."""
        selector = self.validator.selector
        horizon = min(
            ego_candidates.shape[1],
            max(1, int(round(self.collision_horizon_s / self.step_interval))),
        )
        ego = ego_candidates[:, :horizon]
        neighbor_past = raw_inputs.get("neighbor_agents_past")
        if neighbor_past is None or joint_prediction.shape[1] <= 1:
            return ego.new_full((ego.shape[0],), float("inf"))
        current = neighbor_past[0, : joint_prediction.shape[1] - 1, -1].to(ego)
        valid = torch.any(torch.abs(current[:, :4]) > 1e-5, dim=-1)
        if not bool(valid.any()):
            return ego.new_full((ego.shape[0],), float("inf"))
        current = current[valid]
        model_neighbors = joint_prediction[0, 1 : 1 + valid.shape[0], :horizon][valid].to(ego)
        widths = current[:, 6]
        lengths = current[:, 7]
        vehicle = selector.ego_vehicle
        kwargs = {
            "ego_length": float(vehicle.length),
            "ego_width": float(vehicle.width),
            "ego_origin_to_center": float(vehicle.rear_axle_to_center),
            "collision_margin_m": float(selector.collision_margin_m),
        }
        if bool(torch.isfinite(model_neighbors).all()):
            model_clearance = batched_three_disc_min_clearance(
                ego, model_neighbors, lengths, widths, **kwargs
            )
        else:
            model_clearance = ego.new_full((ego.shape[0],), float("-inf"))

        time = torch.arange(1, horizon + 1, dtype=ego.dtype, device=ego.device)
        time = time * self.step_interval
        cv_xy = current[:, None, :2] + current[:, None, 4:6] * time[None, :, None]
        cv_heading = current[:, None, 2:4].expand(-1, horizon, -1)
        cv_neighbors = torch.cat((cv_xy, cv_heading), dim=-1)
        cv_clearance = batched_three_disc_min_clearance(
            ego, cv_neighbors, lengths, widths, **kwargs
        )
        return torch.minimum(model_clearance, cv_clearance)

    def _repair_collision_parallel(
        self,
        styled_prediction: torch.Tensor,
        baseline_prediction: torch.Tensor,
        *,
        raw_inputs: Dict[str, torch.Tensor],
        ego_state: Any,
        map_api: Any,
    ) -> TrajectoryRepairResult:
        styled_ego = styled_prediction[0, 0]
        baseline_ego = baseline_prediction[0, 0]
        point_cache: Dict[tuple[float, float], bool] = {}
        styled_clearance = self._collision_clearance(
            styled_ego[None], styled_prediction, raw_inputs
        )
        styled_offroad, drivable_time_ms = self._short_offroad_fractions(
            styled_ego[None], ego_state, map_api, point_cache
        )
        styled_finite = bool(torch.isfinite(styled_ego).all())
        collision_safe = bool(
            styled_clearance[0] >= self.collision_min_clearance_m
        )
        drivable_safe = bool(
            styled_offroad[0] <= self.validator.selector.max_offroad_fraction
        )
        styled_safe = styled_finite and collision_safe and drivable_safe
        if styled_safe:
            clearance = float(styled_clearance[0].item())
            offroad = float(styled_offroad[0].item())
            return TrajectoryRepairResult(
                prediction=styled_prediction.clone(),
                applied=False,
                baseline_fallback=False,
                longitudinal_scale=1.0,
                lateral_scale=1.0,
                candidate_attempts=1,
                baseline_hard_valid=False,
                failure_reasons=(),
                selected_metrics={
                    "min_predicted_clearance_m": clearance,
                    "short_offroad_fraction": offroad,
                },
                collision_triggered=False,
                drivable_triggered=False,
                style_offroad_fraction=offroad,
                selected_offroad_fraction=offroad,
                drivable_check_time_ms=drivable_time_ms,
            )

        scales = styled_ego.new_tensor(self.collision_scales)
        ego_candidates = compose_interpolated_ego_batch(styled_ego, baseline_ego, scales)
        clearances = self._collision_clearance(ego_candidates, styled_prediction, raw_inputs)
        offroad_fractions, candidate_drivable_time_ms = self._short_offroad_fractions(
            ego_candidates, ego_state, map_api, point_cache
        )
        drivable_time_ms += candidate_drivable_time_ms
        finite = torch.isfinite(ego_candidates).flatten(1).all(dim=1)
        safe = (
            finite
            & (clearances >= self.collision_min_clearance_m)
            & (
                offroad_fractions
                <= self.validator.selector.max_offroad_fraction
            )
        )
        initial_reasons = []
        if not collision_safe:
            initial_reasons.append("predicted_collision")
        if not drivable_safe:
            initial_reasons.append("short_horizon_offroad")
        if not styled_finite:
            initial_reasons.append("non_finite_trajectory")
        safe_indices = torch.nonzero(safe, as_tuple=False).flatten()
        if safe_indices.numel():
            selected_index = int(safe_indices[0].item())
            selected_scale = float(scales[selected_index].item())
            selected_clearance = float(clearances[selected_index].item())
            selected_offroad = float(offroad_fractions[selected_index].item())
            if selected_scale <= 1e-12:
                prediction = baseline_prediction.clone()
                fallback = True
            else:
                prediction = styled_prediction.clone()
                prediction[0, 0] = ego_candidates[selected_index]
                fallback = False
            return TrajectoryRepairResult(
                prediction=prediction,
                applied=True,
                baseline_fallback=fallback,
                longitudinal_scale=selected_scale,
                lateral_scale=selected_scale,
                candidate_attempts=len(self.collision_scales),
                baseline_hard_valid=bool(safe[-1].item()),
                failure_reasons=tuple(initial_reasons),
                selected_metrics={
                    "min_predicted_clearance_m": selected_clearance,
                    "short_offroad_fraction": selected_offroad,
                },
                collision_triggered=not collision_safe,
                drivable_triggered=not drivable_safe,
                style_offroad_fraction=float(styled_offroad[0].item()),
                selected_offroad_fraction=selected_offroad,
                drivable_check_time_ms=drivable_time_ms,
            )

        return TrajectoryRepairResult(
            prediction=baseline_prediction.clone(),
            applied=True,
            baseline_fallback=True,
            longitudinal_scale=0.0,
            lateral_scale=0.0,
            candidate_attempts=len(self.collision_scales),
            baseline_hard_valid=False,
            failure_reasons=tuple(initial_reasons + ["no_safe_interpolation"]),
            selected_metrics={},
            collision_triggered=not collision_safe,
            drivable_triggered=not drivable_safe,
            style_offroad_fraction=float(styled_offroad[0].item()),
            selected_offroad_fraction=float("nan"),
            drivable_check_time_ms=drivable_time_ms,
        )

    def repair(
        self,
        styled_prediction: torch.Tensor,
        baseline_prediction: torch.Tensor,
        *,
        raw_inputs: Dict[str, torch.Tensor],
        ego_state: Any,
        map_api: Any,
    ) -> TrajectoryRepairResult:
        if styled_prediction.shape != baseline_prediction.shape or styled_prediction.ndim != 4:
            raise ValueError("styled_prediction and baseline_prediction must match [B, P, T, 4]")
        if styled_prediction.shape[0] != 1:
            raise ValueError("closed-loop trajectory repair supports batch size 1")

        if self.mode == "collision_parallel":
            return self._repair_collision_parallel(
                styled_prediction,
                baseline_prediction,
                raw_inputs=raw_inputs,
                ego_state=ego_state,
                map_api=map_api,
            )

        all_reasons = []
        baseline_hard_valid = False
        attempts = 0
        for longitudinal_scale, lateral_scale in candidate_scale_grid(
            self.longitudinal_scales, self.lateral_scales
        ):
            candidate = styled_prediction.clone()
            if longitudinal_scale != 1.0 or lateral_scale != 1.0:
                candidate[0, 0] = compose_baseline_anchored_ego(
                    styled_prediction[0, 0],
                    baseline_prediction[0, 0],
                    longitudinal_scale=longitudinal_scale,
                    lateral_scale=lateral_scale,
                )
            evaluation: CandidateAcceptanceResult = self.validator.validate(
                candidate,
                baseline_prediction,
                raw_inputs=raw_inputs,
                ego_state=ego_state,
                map_api=map_api,
            )
            attempts += 1
            baseline_hard_valid = bool(evaluation.baseline_hard_valid)
            all_reasons.extend(evaluation.failure_reasons)
            if evaluation.accepted:
                return TrajectoryRepairResult(
                    prediction=candidate,
                    applied=longitudinal_scale != 1.0 or lateral_scale != 1.0,
                    baseline_fallback=False,
                    longitudinal_scale=longitudinal_scale,
                    lateral_scale=lateral_scale,
                    candidate_attempts=attempts,
                    baseline_hard_valid=baseline_hard_valid,
                    failure_reasons=tuple(dict.fromkeys(all_reasons)),
                    selected_metrics=dict(evaluation.candidate_metrics),
                )

        return TrajectoryRepairResult(
            prediction=baseline_prediction.clone(),
            applied=True,
            baseline_fallback=True,
            longitudinal_scale=0.0,
            lateral_scale=0.0,
            candidate_attempts=attempts,
            baseline_hard_valid=baseline_hard_valid,
            failure_reasons=tuple(dict.fromkeys(all_reasons)),
            selected_metrics={},
        )

"""Online preference execution for runtime style/intensity control."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from research_v1.execution.conditioning.builder import _make_global_vec
from research_v1.execution.diffusion.temporal_condition import build_two_stage_global_targets
from research_v1.execution.diffusion.temporal_condition import build_two_stage_gate_targets
from research_v1.execution.diffusion.style_condition import (
    build_style_phase_time_mask,
    build_style_condition_feature,
    resolve_style_condition_feature_set,
    style_condition_valid_mask,
    validate_style_condition_args,
)
from research_v1.execution.interaction.features import build_interaction_state_features
from research_v1.execution.interaction.gating import compute_scene_gates
from research_v1.execution.interaction.schema import AXIS_GATE_ORDER, SCENE_GATE_ORDER
from research_v1.execution.projection.schema import PROJECTION_LEVEL_ORDER, projection_output_dir, projection_stats_path
from research_v1.execution.projection.stats import VALID_STYLE_LABELS, _bucket_levels
from research_v1.scene_data.schema import style_axis_names_for_scene


def _as_numpy(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


class OnlinePreferenceConditioner:
    """Translate runtime style commands into planner-facing preference conditions."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._feature_set = resolve_style_condition_feature_set(config)
        self._condition_field = str(getattr(config, "condition_field", "effective_preference_global_vec"))
        validate_style_condition_args(self._condition_field, self._feature_set)
        self._future_len = int(getattr(config, "future_len", 80))
        self._min_bucket_size = int(getattr(config, "runtime_min_bucket_size", 128))
        self._stats_path = self._resolve_projection_stats_path(config)
        self._stats = self._load_stats(self._stats_path)

        default_style = getattr(config, "runtime_style_label", "normal")
        default_intensity = getattr(config, "runtime_style_intensity", 0.0)
        self._style_label = self._normalize_style_label(default_style)
        self._style_intensity = self._normalize_intensity(default_intensity)

    @property
    def stats_path(self) -> str:
        return self._stats_path

    @property
    def style_label(self) -> str:
        return self._style_label

    @property
    def style_intensity(self) -> float:
        return self._style_intensity

    def set_command(self, style_label: str | None = None, intensity: float | None = None) -> None:
        if style_label is not None:
            self._style_label = self._normalize_style_label(style_label)
        if intensity is not None:
            self._style_intensity = self._normalize_intensity(intensity)

    def apply(
        self,
        raw_inputs: Mapping[str, Any],
        normalized_inputs: Mapping[str, Any],
        *,
        device: torch.device | str,
    ) -> tuple[dict[str, Any], dict[str, object]]:
        runtime_record = self._build_runtime_record(raw_inputs)
        feature_bundle = build_interaction_state_features(runtime_record)
        gate_bundle = compute_scene_gates(feature_bundle)
        scene_bucket = str(gate_bundle.dominant_scene_gate)
        axis_names = list(style_axis_names_for_scene(scene_bucket))
        scene_axis_indices = [AXIS_GATE_ORDER.index(axis_name) for axis_name in axis_names]

        axis_gate_values = np.asarray(gate_bundle.axis_gate_values, dtype=np.float32)
        local_axis_gate_values = axis_gate_values[scene_axis_indices]
        scene_gate_values = np.asarray(gate_bundle.scene_gate_values, dtype=np.float32)

        scene_record = dict(runtime_record)
        scene_record["scene_bucket"] = scene_bucket

        scene_prototypes = self._resolve_scene_style_prototypes(scene_bucket)
        target_scene_vec = self._build_interpolated_target_vector(
            scene_prototypes,
            target_style_label=self._style_label,
            target_intensity_alpha=self._style_intensity,
        )
        lower_scene_vec, upper_scene_vec, selected_level, selected_bucket_key, selected_bucket_count = (
            self._select_bucket_bounds(scene_record)
        )
        safe_scene_vec = np.clip(target_scene_vec, lower_scene_vec, upper_scene_vec)
        effective_scene_vec = safe_scene_vec * local_axis_gate_values

        target_global_vec = _make_global_vec(axis_names, target_scene_vec.tolist())
        safe_global_vec = _make_global_vec(axis_names, safe_scene_vec.tolist())
        effective_global_vec = safe_global_vec * axis_gate_values
        temporal_near_target_global_vec, temporal_far_target_global_vec = build_two_stage_global_targets(
            torch.as_tensor(safe_global_vec, dtype=torch.float32),
            torch.as_tensor(effective_global_vec, dtype=torch.float32),
            far_recovery_mix=float(getattr(self._config, "two_stage_far_recovery_mix", 0.5)),
        )
        temporal_near_gate_target_vec, temporal_far_gate_target_vec, _ = build_two_stage_gate_targets(
            torch.as_tensor(safe_global_vec, dtype=torch.float32),
            temporal_near_target_global_vec,
            temporal_far_target_global_vec,
        )
        temporal_near_target_global_vec = temporal_near_target_global_vec.detach().cpu().numpy()
        temporal_far_target_global_vec = temporal_far_target_global_vec.detach().cpu().numpy()
        temporal_near_gate_target_vec = temporal_near_gate_target_vec.detach().cpu().numpy()
        temporal_far_gate_target_vec = temporal_far_gate_target_vec.detach().cpu().numpy()

        if self._condition_field == "target_preference_global_vec":
            base_global_condition = target_global_vec
        elif self._condition_field == "safe_preference_global_vec":
            base_global_condition = safe_global_vec
        else:
            base_global_condition = effective_global_vec

        model_inputs = dict(normalized_inputs)
        batch_style_condition = build_style_condition_feature(
            torch.as_tensor(base_global_condition[None, :], dtype=torch.float32, device=device),
            feature_set=self._feature_set,
            target_scene_vec=torch.as_tensor(target_scene_vec[None, :], dtype=torch.float32, device=device),
            safe_scene_vec=torch.as_tensor(safe_scene_vec[None, :], dtype=torch.float32, device=device),
            effective_scene_vec=torch.as_tensor(effective_scene_vec[None, :], dtype=torch.float32, device=device),
            local_axis_gate_values=torch.as_tensor(local_axis_gate_values[None, :], dtype=torch.float32, device=device),
            target_global_vec=torch.as_tensor(target_global_vec[None, :], dtype=torch.float32, device=device),
            safe_global_vec=torch.as_tensor(safe_global_vec[None, :], dtype=torch.float32, device=device),
            scene_buckets=[scene_bucket],
        )
        style_feature_valid = style_condition_valid_mask(batch_style_condition).to(device=device)

        model_inputs["style_value_condition"] = batch_style_condition
        model_inputs["style_feature_valid"] = style_feature_valid.float()
        model_inputs["style_condition_used"] = style_feature_valid.float()
        model_inputs["cfg_guidance_scale"] = float(getattr(self._config, "cfg_guidance_scale", 1.0))
        model_inputs["scene_gate_values"] = torch.as_tensor(scene_gate_values[None, :], dtype=torch.float32, device=device)
        model_inputs["axis_gate_values"] = torch.as_tensor(axis_gate_values[None, :], dtype=torch.float32, device=device)
        model_inputs["local_axis_gate_values"] = torch.as_tensor(
            local_axis_gate_values[None, :],
            dtype=torch.float32,
            device=device,
        )
        model_inputs["target_preference_scene_vec"] = torch.as_tensor(
            target_scene_vec[None, :],
            dtype=torch.float32,
            device=device,
        )
        model_inputs["safe_preference_scene_vec"] = torch.as_tensor(
            safe_scene_vec[None, :],
            dtype=torch.float32,
            device=device,
        )
        model_inputs["effective_preference_scene_vec"] = torch.as_tensor(
            effective_scene_vec[None, :],
            dtype=torch.float32,
            device=device,
        )
        model_inputs["target_preference_global_vec"] = torch.as_tensor(
            target_global_vec[None, :],
            dtype=torch.float32,
            device=device,
        )
        model_inputs["safe_preference_global_vec"] = torch.as_tensor(
            safe_global_vec[None, :],
            dtype=torch.float32,
            device=device,
        )
        model_inputs["effective_preference_global_vec"] = torch.as_tensor(
            effective_global_vec[None, :],
            dtype=torch.float32,
            device=device,
        )
        if bool(getattr(self._config, "use_phase_style_condition", False)):
            phase_time_mask = build_style_phase_time_mask(
                self._feature_set,
                [scene_bucket],
                future_len=self._future_len,
                include_current=True,
                device=torch.device(device),
                dtype=batch_style_condition.dtype,
                two_stage_split_ratio=float(getattr(self._config, "two_stage_split_ratio", 0.45)),
                two_stage_transition_ratio=float(getattr(self._config, "two_stage_transition_ratio", 0.18)),
            )
            if phase_time_mask is not None:
                model_inputs["phase_time_mask"] = phase_time_mask

        debug = {
            "style_label": self._style_label,
            "style_intensity": float(self._style_intensity),
            "scene_bucket": scene_bucket,
            "scene_axis_names": axis_names,
            "scene_gate_names": list(SCENE_GATE_ORDER),
            "axis_gate_names": list(AXIS_GATE_ORDER),
            "condition_field": self._condition_field,
            "condition_density_level": str(runtime_record["condition_density_level"]),
            "condition_speed_regime": str(runtime_record["condition_speed_regime"]),
            "condition_curvature_level": str(runtime_record["condition_curvature_level"]),
            "selected_bucket_level": selected_level,
            "selected_bucket_key": selected_bucket_key,
            "selected_bucket_count": int(selected_bucket_count),
            "dominant_scene_gate_score": float(gate_bundle.dominant_scene_gate_score),
            "scene_gate_values": [float(value) for value in scene_gate_values.tolist()],
            "axis_gate_values": [float(value) for value in axis_gate_values.tolist()],
            "local_axis_gate_values": [float(value) for value in local_axis_gate_values.tolist()],
            "target_preference_scene_vec": [float(value) for value in target_scene_vec.tolist()],
            "safe_preference_scene_vec": [float(value) for value in safe_scene_vec.tolist()],
            "effective_preference_scene_vec": [float(value) for value in effective_scene_vec.tolist()],
            "target_preference_global_vec": [float(value) for value in target_global_vec.tolist()],
            "safe_preference_global_vec": [float(value) for value in safe_global_vec.tolist()],
            "effective_preference_global_vec": [float(value) for value in effective_global_vec.tolist()],
            "temporal_near_target_global_vec": [
                float(value) for value in temporal_near_target_global_vec.tolist()
            ],
            "temporal_far_target_global_vec": [
                float(value) for value in temporal_far_target_global_vec.tolist()
            ],
            "temporal_near_gate_target_vec": [
                float(value) for value in temporal_near_gate_target_vec.tolist()
            ],
            "temporal_far_gate_target_vec": [
                float(value) for value in temporal_far_gate_target_vec.tolist()
            ],
            "runtime_context": {
                "lead_vehicle_present": bool(runtime_record["lead_vehicle_present"]),
                "following_min_gap": float(runtime_record["following_min_gap"]),
                "following_min_thw": runtime_record["following_min_thw"],
                "merge_min_gap": float(runtime_record["merge_min_gap"]),
                "merge_lateral_closure": float(runtime_record["merge_lateral_closure"]),
                "ego_speed_ratio_to_limit": runtime_record["ego_speed_ratio_to_limit"],
                "ego_mean_speed": float(runtime_record["ego_mean_speed"]),
                "event_speed_drop_ratio": float(runtime_record["event_speed_drop_ratio"]),
                "event_brake_peak": float(runtime_record["event_brake_peak"]),
                "ego_brake_peak": float(runtime_record["ego_brake_peak"]),
                "ego_lateral_disp": float(runtime_record["ego_lateral_disp"]),
                "ego_lateral_speed_peak": float(runtime_record["ego_lateral_speed_peak"]),
                "route_lane_count": int(runtime_record["route_lane_count"]),
                "nearby_agent_count": int(runtime_record["nearby_agent_count"]),
                "ego_progress": float(runtime_record["ego_progress"]),
                "ego_heading_change": float(runtime_record["ego_heading_change"]),
                "route_has_control": bool(runtime_record["route_has_control"]),
            },
        }
        return model_inputs, debug

    def _resolve_projection_stats_path(self, config: Any) -> str:
        override = str(getattr(config, "runtime_projection_stats_path", "") or "").strip()
        if override:
            return override
        train_split_root = str(getattr(config, "train_split_root", "") or "").strip()
        if train_split_root:
            return projection_stats_path(projection_output_dir(train_split_root))
        raise ValueError(
            "runtime_projection_stats_path is empty and train_split_root is unavailable; "
            "cannot resolve projection stats for online style control."
        )

    @staticmethod
    def _load_stats(stats_path: str) -> Mapping[str, object]:
        if not Path(stats_path).exists():
            raise FileNotFoundError(f"projection stats not found for runtime control: {stats_path}")
        with open(stats_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    @staticmethod
    def _normalize_style_label(style_label: object) -> str:
        label = str(style_label or "normal").strip().lower()
        if label not in VALID_STYLE_LABELS:
            raise ValueError(f"runtime_style_label must be one of {VALID_STYLE_LABELS}, got {style_label!r}")
        return label

    @staticmethod
    def _normalize_intensity(intensity: object) -> float:
        try:
            value = float(intensity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"runtime_style_intensity must be numeric, got {intensity!r}") from exc
        return _clip01(value)

    def _build_runtime_record(self, raw_inputs: Mapping[str, Any]) -> dict[str, object]:
        ego_current_state = _as_numpy(raw_inputs["ego_current_state"])[0].astype(np.float32)
        ego_agent_past = _as_numpy(raw_inputs["ego_agent_past"])[0].astype(np.float32)
        neighbor_agents_past = _as_numpy(raw_inputs["neighbor_agents_past"])[0].astype(np.float32)
        neighbor_agents_past_mask = _as_numpy(raw_inputs["neighbor_agents_past_mask"])[0].astype(bool)
        route_lanes = _as_numpy(raw_inputs["route_lanes"])[0].astype(np.float32)
        route_lanes_mask = _as_numpy(raw_inputs["route_lanes_mask"])[0].astype(bool)
        route_lanes_speed_limit = _as_numpy(raw_inputs["route_lanes_speed_limit"])[0].astype(np.float32)
        route_lanes_has_speed_limit = _as_numpy(raw_inputs["route_lanes_has_speed_limit"])[0].astype(bool)

        ego_speed = np.linalg.norm(ego_agent_past[:, 3:5], axis=-1)
        ego_mean_speed = float(np.mean(ego_speed)) if ego_speed.size > 0 else float(abs(ego_current_state[4]))
        ego_progress = self._path_length(ego_agent_past[:, :2])
        ego_heading_now = float(math.atan2(float(ego_current_state[3]), float(ego_current_state[2])))
        ego_heading_prev = float(ego_agent_past[0, 2]) if ego_agent_past.shape[0] > 0 else ego_heading_now
        ego_heading_change = abs(_wrap_angle(ego_heading_now - ego_heading_prev))

        longitudinal_accel = ego_agent_past[:, 5] if ego_agent_past.shape[1] > 5 else np.zeros((ego_agent_past.shape[0],), dtype=np.float32)
        ego_brake_peak = float(np.max(np.maximum(-longitudinal_accel, 0.0))) if longitudinal_accel.size > 0 else 0.0
        event_brake_peak = ego_brake_peak
        event_speed_drop_ratio = self._speed_drop_ratio(ego_speed)
        ego_lateral_disp = float(np.max(np.abs(ego_agent_past[:, 1]))) if ego_agent_past.shape[0] > 0 else 0.0
        ego_lateral_speed_peak = float(np.max(np.abs(ego_agent_past[:, 4]))) if ego_agent_past.shape[0] > 0 else abs(float(ego_current_state[5]))

        current_neighbor_state = neighbor_agents_past[:, -1, :]
        current_neighbor_valid = neighbor_agents_past_mask[:, -1].astype(bool)
        vehicle_mask = current_neighbor_state[:, 8] > 0.5 if current_neighbor_state.shape[1] >= 9 else current_neighbor_valid
        valid_vehicle_mask = current_neighbor_valid & vehicle_mask

        route_lane_count = int(np.sum(np.any(route_lanes_mask, axis=-1)))
        route_speed_limit_mps = self._route_speed_limit(route_lanes_speed_limit, route_lanes_has_speed_limit, route_lanes_mask)
        ego_speed_ratio_to_limit = None
        if route_speed_limit_mps is not None and route_speed_limit_mps > 1e-3:
            ego_speed_ratio_to_limit = float(abs(ego_current_state[4]) / route_speed_limit_mps)

        following_min_gap, following_min_thw, lead_vehicle_present = self._lead_follow_metrics(
            current_neighbor_state,
            valid_vehicle_mask,
            ego_speed_mps=max(float(abs(ego_current_state[4])), 0.0),
        )
        merge_min_gap = self._merge_gap_metric(current_neighbor_state, current_neighbor_valid)
        merge_lateral_closure = abs(float(ego_current_state[5]))
        nearby_agent_count = int(
            np.sum(
                current_neighbor_valid
                & (np.linalg.norm(current_neighbor_state[:, :2], axis=-1) <= 40.0)
            )
        )
        route_has_control = self._route_has_control(route_lanes, route_lanes_mask)

        density_level, speed_regime, curvature_level = self._infer_condition_tags(
            nearby_agent_count=nearby_agent_count,
            speed_ref=route_speed_limit_mps if route_speed_limit_mps is not None else ego_mean_speed,
            heading_change=ego_heading_change,
        )

        return {
            "scene_bucket": "none",
            "lead_vehicle_present": bool(lead_vehicle_present),
            "following_min_gap": float(following_min_gap),
            "following_min_thw": None if following_min_thw is None else float(following_min_thw),
            "merge_min_gap": float(merge_min_gap),
            "merge_lateral_closure": float(merge_lateral_closure),
            "ego_speed_ratio_to_limit": ego_speed_ratio_to_limit,
            "ego_mean_speed": float(ego_mean_speed),
            "event_speed_drop_ratio": float(event_speed_drop_ratio),
            "event_brake_peak": float(event_brake_peak),
            "ego_brake_peak": float(ego_brake_peak),
            "ego_lateral_disp": float(ego_lateral_disp),
            "ego_lateral_speed_peak": float(ego_lateral_speed_peak),
            "route_lane_count": int(route_lane_count),
            "nearby_agent_count": int(nearby_agent_count),
            "condition_density_level": density_level,
            "condition_speed_regime": speed_regime,
            "condition_curvature_level": curvature_level,
            "ego_progress": float(ego_progress),
            "ego_heading_change": float(ego_heading_change),
            "route_has_control": bool(route_has_control),
        }

    @staticmethod
    def _path_length(xy: np.ndarray) -> float:
        if xy.ndim != 2 or xy.shape[0] < 2:
            return 0.0
        deltas = xy[1:] - xy[:-1]
        return float(np.sum(np.linalg.norm(deltas, axis=-1)))

    @staticmethod
    def _speed_drop_ratio(speed: np.ndarray) -> float:
        if speed.size == 0:
            return 0.0
        peak = float(np.max(speed))
        if peak <= 1e-3:
            return 0.0
        return max((peak - float(speed[-1])) / peak, 0.0)

    @staticmethod
    def _route_speed_limit(
        route_lanes_speed_limit: np.ndarray,
        route_lanes_has_speed_limit: np.ndarray,
        route_lanes_mask: np.ndarray,
    ) -> float | None:
        valid_route = np.any(route_lanes_mask, axis=-1)
        valid_limit = valid_route & route_lanes_has_speed_limit.reshape(-1)
        if not np.any(valid_limit):
            return None
        values = route_lanes_speed_limit.reshape(-1)[valid_limit]
        if values.size == 0:
            return None
        return float(np.median(values))

    @staticmethod
    def _route_has_control(route_lanes: np.ndarray, route_lanes_mask: np.ndarray) -> bool:
        if route_lanes.shape[-1] < 12:
            return False
        valid_points = route_lanes_mask.astype(bool)
        if not np.any(valid_points):
            return False
        traffic_state = route_lanes[..., 8:12]
        yellow_or_red = (traffic_state[..., 1] > 0.5) | (traffic_state[..., 2] > 0.5)
        return bool(np.any(yellow_or_red & valid_points))

    @staticmethod
    def _lead_follow_metrics(
        current_neighbor_state: np.ndarray,
        valid_vehicle_mask: np.ndarray,
        *,
        ego_speed_mps: float,
    ) -> tuple[float, float | None, bool]:
        if current_neighbor_state.ndim != 2 or current_neighbor_state.shape[0] == 0:
            return 1e6, None, False
        ahead_mask = (
            valid_vehicle_mask
            & (current_neighbor_state[:, 0] > 0.0)
            & (np.abs(current_neighbor_state[:, 1]) < 2.8)
        )
        if not np.any(ahead_mask):
            return 1e6, None, False
        candidate_states = current_neighbor_state[ahead_mask]
        gap = np.maximum(candidate_states[:, 0] - 2.5 - 0.5 * candidate_states[:, 7], 0.0)
        min_gap = float(np.min(gap)) if gap.size > 0 else 1e6
        min_thw = None
        if ego_speed_mps > 0.5:
            min_thw = float(min_gap / max(ego_speed_mps, 1e-3))
        return min_gap, min_thw, True

    @staticmethod
    def _merge_gap_metric(current_neighbor_state: np.ndarray, current_neighbor_valid: np.ndarray) -> float:
        if current_neighbor_state.ndim != 2 or current_neighbor_state.shape[0] == 0:
            return 1e6
        lateral_mask = (
            current_neighbor_valid
            & (np.abs(current_neighbor_state[:, 1]) > 1.2)
            & (np.abs(current_neighbor_state[:, 1]) < 6.0)
        )
        if not np.any(lateral_mask):
            return 1e6
        candidates = current_neighbor_state[lateral_mask][:, :2]
        return float(np.min(np.linalg.norm(candidates, axis=-1)))

    @staticmethod
    def _infer_condition_tags(*, nearby_agent_count: int, speed_ref: float, heading_change: float) -> tuple[str, str, str]:
        if nearby_agent_count <= 2:
            density_level = "sparse"
        elif nearby_agent_count <= 7:
            density_level = "medium"
        else:
            density_level = "dense"

        speed_ref = float(speed_ref)
        if speed_ref < 6.0:
            speed_regime = "slow"
        elif speed_ref < 12.0:
            speed_regime = "urban"
        elif speed_ref < 20.0:
            speed_regime = "suburban"
        else:
            speed_regime = "high_speed"

        heading_change = abs(float(heading_change))
        if heading_change < 0.08:
            curvature_level = "low"
        elif heading_change < 0.18:
            curvature_level = "mild"
        else:
            curvature_level = "high"
        return density_level, speed_regime, curvature_level

    def _resolve_scene_style_prototypes(self, scene_bucket: str) -> dict[str, np.ndarray]:
        style_prototypes = self._stats.get("style_prototypes", {})
        scene_stats = style_prototypes.get(scene_bucket, {})
        scene_mean = scene_stats.get("scene_mean", None)
        styles = scene_stats.get("styles", {})
        normal_mean = styles.get("normal", {}).get("mean", scene_mean)
        if normal_mean is None:
            raise ValueError(f"Missing normal prototype for runtime scene bucket {scene_bucket!r}")

        prototypes: dict[str, np.ndarray] = {}
        for style_label in VALID_STYLE_LABELS:
            if style_label == "normal":
                candidate = normal_mean
            else:
                candidate = styles.get(style_label, {}).get("mean", normal_mean)
            candidate_vec = np.asarray(candidate, dtype=np.float32)
            if candidate_vec.shape[0] != 3:
                raise ValueError(
                    f"Expected 3-d prototype for {scene_bucket}/{style_label}, got {candidate_vec.shape}"
                )
            prototypes[style_label] = candidate_vec
        return prototypes

    @staticmethod
    def _build_interpolated_target_vector(
        scene_prototypes: Mapping[str, np.ndarray],
        *,
        target_style_label: str,
        target_intensity_alpha: float,
    ) -> np.ndarray:
        normal_vec = np.asarray(scene_prototypes["normal"], dtype=np.float32)
        if target_style_label == "normal" or target_intensity_alpha <= 1e-6:
            return normal_vec.copy()
        style_vec = np.asarray(scene_prototypes[target_style_label], dtype=np.float32)
        return normal_vec + float(target_intensity_alpha) * (style_vec - normal_vec)

    def _select_bucket_bounds(
        self,
        runtime_record: Mapping[str, object],
    ) -> tuple[np.ndarray, np.ndarray, str, str, int]:
        levels = self._stats.get("levels", {})
        bucket_levels = _bucket_levels(runtime_record)
        for level_name in PROJECTION_LEVEL_ORDER:
            bucket_key = bucket_levels[level_name]
            level_stats = levels.get(level_name, {})
            bucket_stats = level_stats.get(bucket_key, None)
            if bucket_stats is None:
                continue
            if int(bucket_stats.get("count", 0)) < self._min_bucket_size and level_name != "scene":
                continue
            return (
                np.asarray(bucket_stats["lower"], dtype=np.float32),
                np.asarray(bucket_stats["upper"], dtype=np.float32),
                level_name,
                bucket_key,
                int(bucket_stats.get("count", 0)),
            )

        scene_bucket = str(runtime_record.get("scene_bucket", "none"))
        scene_stats = levels.get("scene", {}).get(scene_bucket, None)
        if scene_stats is None:
            raise ValueError(f"Missing scene-level projection stats for runtime scene bucket {scene_bucket!r}")
        return (
            np.asarray(scene_stats["lower"], dtype=np.float32),
            np.asarray(scene_stats["upper"], dtype=np.float32),
            "scene",
            scene_bucket,
            int(scene_stats.get("count", 0)),
        )

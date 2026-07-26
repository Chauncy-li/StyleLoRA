"""Causal scene routing shared by offline continuous-style export and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping
from pathlib import Path

import numpy as np

from research.preference_execution.interaction_state.features import (
    InteractionStateFeatureBundle,
    build_interaction_state_features,
)
from research.preference_execution.interaction_state.gating import (
    SceneGateBundle,
    compute_scene_gates,
)


@dataclass(frozen=True)
class ContinuousSceneRoute:
    """Offline record of the causal scene router used later at runtime."""

    observed_scene_bucket: str
    routed_scene_bucket: str
    routed_scene_score: float
    observed_scene_score: float
    router_source: str
    router_note: str
    route_lane_change_intent: bool
    route_lane_change_intent_available: bool
    feature_bundle: InteractionStateFeatureBundle
    gate_bundle: SceneGateBundle

    @property
    def scene_consistent(self) -> bool:
        return self.observed_scene_bucket == self.routed_scene_bucket

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "observed_scene_bucket": self.observed_scene_bucket,
            "router_scene_bucket": self.routed_scene_bucket,
            "router_selected_scene_score": float(self.routed_scene_score),
            "router_observed_scene_score": float(self.observed_scene_score),
            "router_scene_consistent": bool(self.scene_consistent),
            "router_source": self.router_source,
            "router_note": self.router_note,
            "router_route_lane_change_intent": bool(self.route_lane_change_intent),
            "router_route_lane_change_intent_available": bool(self.route_lane_change_intent_available),
            "feature_names": list(self.feature_bundle.feature_names),
            "feature_values": [float(value) for value in self.feature_bundle.feature_values.tolist()],
            "feature_mask": [int(value) for value in self.feature_bundle.feature_mask.tolist()],
            "scene_gate_names": list(self.gate_bundle.scene_gate_names),
            "scene_gate_values": [float(value) for value in self.gate_bundle.scene_gate_values.tolist()],
            "axis_gate_names": list(self.gate_bundle.axis_gate_names),
            "axis_gate_values": [float(value) for value in self.gate_bundle.axis_gate_values.tolist()],
        }


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _path_length(xy: np.ndarray) -> float:
    if xy.ndim != 2 or xy.shape[0] < 2:
        return 0.0
    deltas = xy[1:] - xy[:-1]
    return float(np.sum(np.linalg.norm(deltas, axis=-1)))


def _speed_drop_ratio(speed: np.ndarray) -> float:
    if speed.size == 0:
        return 0.0
    peak = float(np.max(speed))
    if peak <= 1e-3:
        return 0.0
    return max((peak - float(speed[-1])) / peak, 0.0)


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


def _route_has_control(route_lanes: np.ndarray, route_lanes_mask: np.ndarray) -> bool:
    if route_lanes.shape[-1] < 12:
        return False
    valid_points = route_lanes_mask.astype(bool)
    if not np.any(valid_points):
        return False
    traffic_state = route_lanes[..., 8:12]
    yellow_or_red = (traffic_state[..., 1] > 0.5) | (traffic_state[..., 2] > 0.5)
    return bool(np.any(yellow_or_red & valid_points))


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


def _resolve_cache_path(record: Mapping[str, object]) -> str:
    for key in ("cache_path", "planner_cache_path", "style_cache_path"):
        value = str(record.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _optional_route_lane_change_intent(cache_data) -> tuple[bool, bool]:
    """Read an explicit upstream route-intent flag when a cache provides one.

    This deliberately does not infer intent from ego future motion.  Older
    caches have no such field and therefore retain exactly the old router
    behavior.  A route planner can later write one of these causal flags.
    """

    for key in (
        "route_lane_change_intent",
        "route_intent_lane_change",
        "route_requires_lane_change",
    ):
        if hasattr(cache_data, "files") and key in cache_data.files:
            values = np.asarray(cache_data[key]).reshape(-1)
            if values.size > 0:
                try:
                    return bool(float(values[0]) > 0.5), True
                except (TypeError, ValueError):
                    return bool(values[0]), True
    return False, False


def _apply_route_intent_to_gate_bundle(
    gate_bundle: SceneGateBundle,
    route_lane_change_intent: bool,
) -> SceneGateBundle:
    """Gently promote the lane-change gate only for explicit causal intent."""

    if not route_lane_change_intent:
        return gate_bundle
    values = np.asarray(gate_bundle.scene_gate_values, dtype=np.float32).copy()
    if values.shape[0] != 3:
        return gate_bundle
    # Multiplicative odds boost preserves the existing gate ordering when the
    # intent flag is absent and avoids turning intent into a hard override.
    values[2] *= float(np.exp(0.9))
    values /= max(float(np.sum(values)), 1e-6)
    dominant_index = int(np.argmax(values))
    repeat_count = max(int(gate_bundle.axis_gate_values.shape[0] / max(values.shape[0], 1)), 1)
    axis_values = np.repeat(values, repeat_count).astype(np.float32)
    return SceneGateBundle(
        scene_gate_names=gate_bundle.scene_gate_names,
        scene_gate_values=values.astype(np.float32),
        dominant_scene_gate=gate_bundle.scene_gate_names[dominant_index],
        dominant_scene_gate_score=float(values[dominant_index]),
        axis_gate_names=gate_bundle.axis_gate_names,
        axis_gate_values=axis_values,
    )


def _build_causal_record_from_cache(cache_path: str) -> Dict[str, object]:
    with np.load(cache_path, allow_pickle=False) as cache_data:
        ego_current_state = np.asarray(cache_data["ego_current_state"], dtype=np.float32)
        ego_agent_past = np.asarray(cache_data["ego_agent_past"], dtype=np.float32)
        neighbor_agents_past = np.asarray(cache_data["neighbor_agents_past"], dtype=np.float32)
        neighbor_agents_past_mask = np.asarray(cache_data["neighbor_agents_past_mask"], dtype=bool)
        route_lanes = np.asarray(cache_data["route_lanes"], dtype=np.float32)
        route_lanes_mask = np.asarray(cache_data["route_lanes_mask"], dtype=bool)
        route_lanes_speed_limit = np.asarray(cache_data["route_lanes_speed_limit"], dtype=np.float32)
        route_lanes_has_speed_limit = np.asarray(cache_data["route_lanes_has_speed_limit"], dtype=bool)
        route_lane_change_intent, route_lane_change_intent_available = _optional_route_lane_change_intent(cache_data)

    ego_speed = np.linalg.norm(ego_agent_past[:, 3:5], axis=-1)
    ego_mean_speed = float(np.mean(ego_speed)) if ego_speed.size > 0 else float(abs(ego_current_state[4]))
    ego_progress = _path_length(ego_agent_past[:, :2])
    ego_heading_now = float(np.arctan2(float(ego_current_state[3]), float(ego_current_state[2])))
    ego_heading_prev = float(ego_agent_past[0, 2]) if ego_agent_past.shape[0] > 0 else ego_heading_now
    ego_heading_change = abs(_wrap_angle(ego_heading_now - ego_heading_prev))

    longitudinal_accel = (
        ego_agent_past[:, 5]
        if ego_agent_past.ndim == 2 and ego_agent_past.shape[1] > 5
        else np.zeros((ego_agent_past.shape[0],), dtype=np.float32)
    )
    ego_brake_peak = float(np.max(np.maximum(-longitudinal_accel, 0.0))) if longitudinal_accel.size > 0 else 0.0
    event_brake_peak = ego_brake_peak
    event_speed_drop_ratio = _speed_drop_ratio(ego_speed)
    ego_lateral_disp = float(np.max(np.abs(ego_agent_past[:, 1]))) if ego_agent_past.shape[0] > 0 else 0.0
    ego_lateral_speed_peak = (
        float(np.max(np.abs(ego_agent_past[:, 4])))
        if ego_agent_past.shape[0] > 0
        else abs(float(ego_current_state[5]))
    )

    current_neighbor_state = neighbor_agents_past[:, -1, :] if neighbor_agents_past.ndim == 3 else np.zeros((0, 11), dtype=np.float32)
    current_neighbor_valid = neighbor_agents_past_mask[:, -1].astype(bool) if neighbor_agents_past_mask.ndim == 2 else np.zeros((0,), dtype=bool)
    vehicle_mask = current_neighbor_state[:, 8] > 0.5 if current_neighbor_state.shape[1] >= 9 else current_neighbor_valid
    valid_vehicle_mask = current_neighbor_valid & vehicle_mask

    route_lane_count = int(np.sum(np.any(route_lanes_mask, axis=-1))) if route_lanes_mask.ndim >= 2 else 0
    route_speed_limit_mps = _route_speed_limit(route_lanes_speed_limit, route_lanes_has_speed_limit, route_lanes_mask)
    ego_speed_ratio_to_limit = None
    if route_speed_limit_mps is not None and route_speed_limit_mps > 1e-3:
        ego_speed_ratio_to_limit = float(abs(ego_current_state[4]) / route_speed_limit_mps)

    following_min_gap, following_min_thw, lead_vehicle_present = _lead_follow_metrics(
        current_neighbor_state,
        valid_vehicle_mask,
        ego_speed_mps=max(float(abs(ego_current_state[4])), 0.0),
    )
    merge_min_gap = _merge_gap_metric(current_neighbor_state, current_neighbor_valid)
    merge_lateral_closure = abs(float(ego_current_state[5]))
    nearby_agent_count = int(
        np.sum(
            current_neighbor_valid
            & (np.linalg.norm(current_neighbor_state[:, :2], axis=-1) <= 40.0)
        )
    )
    route_has_control = _route_has_control(route_lanes, route_lanes_mask)

    density_level, speed_regime, curvature_level = _infer_condition_tags(
        nearby_agent_count=nearby_agent_count,
        speed_ref=route_speed_limit_mps if route_speed_limit_mps is not None else ego_mean_speed,
        heading_change=ego_heading_change,
    )

    return {
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
        "route_lane_change_intent": bool(route_lane_change_intent),
        "route_lane_change_intent_available": bool(route_lane_change_intent_available),
    }


def route_scene_from_record(record: Mapping[str, object]) -> ContinuousSceneRoute:
    """Route a normalized split-index record using only causal observable fields."""

    observed_scene_bucket = str(record.get("scene_bucket", record.get("scene_bucket_name", "none")))

    router_source = "record_fallback"
    router_note = "cache_missing_or_unavailable"
    causal_record = dict(record)
    cache_path = _resolve_cache_path(record)
    if cache_path and Path(cache_path).exists():
        try:
            cache_causal_record = _build_causal_record_from_cache(cache_path)
            if (
                not bool(cache_causal_record.get("route_lane_change_intent_available", False))
                and "route_lane_change_intent" in record
            ):
                # An explicit runtime/record intent is still causal and must
                # not be erased merely because an older cache lacks the field.
                cache_causal_record["route_lane_change_intent"] = bool(
                    record.get("route_lane_change_intent", False)
                )
                cache_causal_record["route_lane_change_intent_available"] = bool(
                    record.get("route_lane_change_intent_available", True)
                )
            causal_record = {
                **dict(record),
                **cache_causal_record,
            }
            router_source = "cache"
            router_note = str(cache_path)
        except (KeyError, ValueError, OSError, IndexError) as exc:
            router_source = "record_fallback"
            router_note = f"cache_read_failed:{type(exc).__name__}"

    route_lane_change_intent = bool(causal_record.get("route_lane_change_intent", False))
    route_lane_change_intent_available = bool(
        causal_record.get("route_lane_change_intent_available", "route_lane_change_intent" in causal_record)
    )

    feature_bundle = build_interaction_state_features(causal_record)
    gate_bundle = _apply_route_intent_to_gate_bundle(
        compute_scene_gates(feature_bundle),
        route_lane_change_intent=route_lane_change_intent,
    )
    scene_gate_values = np.asarray(gate_bundle.scene_gate_values, dtype=np.float32)

    observed_scene_score = 0.0
    if observed_scene_bucket in gate_bundle.scene_gate_names:
        observed_index = gate_bundle.scene_gate_names.index(observed_scene_bucket)
        observed_scene_score = float(scene_gate_values[observed_index])

    routed_scene_bucket = str(gate_bundle.dominant_scene_gate)
    routed_scene_score = float(gate_bundle.dominant_scene_gate_score)
    return ContinuousSceneRoute(
        observed_scene_bucket=observed_scene_bucket,
        routed_scene_bucket=routed_scene_bucket,
        routed_scene_score=routed_scene_score,
        observed_scene_score=observed_scene_score,
        router_source=router_source,
        router_note=router_note,
        route_lane_change_intent=route_lane_change_intent,
        route_lane_change_intent_available=route_lane_change_intent_available,
        feature_bundle=feature_bundle,
        gate_bundle=gate_bundle,
    )

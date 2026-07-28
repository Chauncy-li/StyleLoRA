"""High-precision straight-driving context splitter for style memory building."""
# High-precision straight-road context splitter for style-memory construction.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

# Import constants and result structures defined by the legacy schema.
from .schema import SCENE_BUCKET_ORDER, STYLE_SCORE_ORDER, StyleSceneSplitResult


# Road metadata containing these keywords is treated as a complex scene.
STRAIGHT_COMPLEX_SCENARIO_KEYWORDS = (
    "intersection", "roundabout", "crosswalk", "pedestrian",
    "protected", "unprotected", "stop_sign", "turn", "u_turn",
)


# Utility functions

def _clip01(value: float) -> float:
    """Clamp a confidence score to the [0, 1] interval."""
    return float(np.clip(value, 0.0, 1.0))

def _score_rising(value: float, low: float, high: float) -> float:
    """Increasing fuzzy score with linear interpolation between bounds."""
    if high <= low:
        return float(value >= high)
    return _clip01((float(value) - low) / (high - low))


def _score_falling(value: float, low: float, high: float) -> float:
    """Decreasing fuzzy score with linear interpolation between bounds."""
    if high <= low:
        return float(value <= low)
    return _clip01((high - float(value)) / (high - low))

def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Normalize an angle to the [-pi, pi] interval."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

# Safe statistics helpers that tolerate empty arrays.
def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0: return 0.0
    return float(np.mean(values))

def _safe_min(values: np.ndarray, default: float) -> float:
    if values.size == 0: return float(default)
    return float(np.min(values))

def _safe_max(values: np.ndarray, default: float = 0.0) -> float:
    if values.size == 0: return float(default)
    return float(np.max(values))

def _safe_percentile(values: np.ndarray, percentile: float, default: float = 0.0) -> float:
    if values.size == 0: return float(default)
    return float(np.percentile(values, percentile))

def _nonzero_xy_mask(traj: np.ndarray) -> np.ndarray:
    """Remove trajectory padding points at coordinate (0, 0)."""
    if traj.ndim != 3: return np.zeros((0, 0), dtype=bool)
    return np.linalg.norm(traj[:, :, :2], axis=-1) > 1e-4

def _resolve_neighbor_valid_mask(cache_data: Dict[str, np.ndarray], neighbors_future: np.ndarray) -> np.ndarray:
    """Resolve neighbor validity masks across NuPlan cache variants."""
    nonzero_mask = _nonzero_xy_mask(neighbors_future)
    raw_mask = cache_data.get("neighbor_agents_future_mask", None)
    if raw_mask is None: return nonzero_mask
    raw_mask = np.asarray(raw_mask, dtype=bool)
    if raw_mask.shape != nonzero_mask.shape: return nonzero_mask
    # Combine the supplied mask with the nonzero-trajectory mask.
    direct_overlap = int(np.logical_and(raw_mask, nonzero_mask).sum())
    inverse_overlap = int(np.logical_and(~raw_mask, nonzero_mask).sum())
    inferred_valid = raw_mask if direct_overlap >= inverse_overlap else ~raw_mask
    return np.logical_or(inferred_valid, nonzero_mask)


def _speed_from_xy(xy: np.ndarray, dt: float, anchor_xy: Optional[np.ndarray] = None) -> np.ndarray:
    if xy.ndim != 2 or xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if anchor_xy is None:
        anchor_xy = np.zeros((1, 2), dtype=np.float32)
    points = np.concatenate([anchor_xy.astype(np.float32), xy.astype(np.float32)], axis=0)
    delta = np.diff(points, axis=0)
    return (np.linalg.norm(delta, axis=-1) / max(dt, 1e-6)).astype(np.float32)


def _accel_from_speed(speed: np.ndarray, dt: float) -> np.ndarray:
    if speed.size < 2:
        return np.zeros((0,), dtype=np.float32)
    return (np.diff(speed) / max(dt, 1e-6)).astype(np.float32)


def _jerk_from_accel(accel: np.ndarray, dt: float) -> np.ndarray:
    if accel.size < 2:
        return np.zeros((0,), dtype=np.float32)
    return (np.diff(accel) / max(dt, 1e-6)).astype(np.float32)


def _heading_from_traj(traj: np.ndarray) -> np.ndarray:
    if traj.ndim != 2 or traj.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if traj.shape[1] >= 3:
        return traj[:, 2].astype(np.float32)
    if traj.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)
    delta = np.diff(traj[:, :2], axis=0)
    heading = np.arctan2(delta[:, 1], delta[:, 0]).astype(np.float32)
    return np.concatenate([heading[:1], heading], axis=0)


def _pairwise_distance(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    if a_xy.shape[0] == 0 or b_xy.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = a_xy[:, None, :] - b_xy[None, :, :]
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


def _scalar_cache_str(cache_data: Dict[str, np.ndarray], key: str, default: str = "") -> str:
    value = cache_data.get(key, None)
    if value is None:
        return default
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return default
    return str(arr[0])


def _masked_speed_limit_mps(
    speed_limit: np.ndarray,
    has_speed_limit: np.ndarray,
    lane_mask: np.ndarray,
) -> Optional[float]:
    speed_limit = np.asarray(speed_limit, dtype=np.float32)
    has_speed_limit = np.asarray(has_speed_limit, dtype=bool)
    lane_mask = np.asarray(lane_mask, dtype=bool)

    if speed_limit.ndim == 0:
        speed_limit = speed_limit.reshape(1, 1)
    if has_speed_limit.ndim == 0:
        has_speed_limit = has_speed_limit.reshape(1, 1)

    valid_lane_mask = has_speed_limit.reshape(-1).astype(bool)
    if lane_mask.ndim == 2 and lane_mask.shape[0] == valid_lane_mask.size:
        valid_lane_mask = np.logical_and(valid_lane_mask, lane_mask.any(axis=1))
    elif lane_mask.ndim == 1 and lane_mask.size == valid_lane_mask.size:
        valid_lane_mask = np.logical_and(valid_lane_mask, lane_mask.astype(bool))

    values = speed_limit.reshape(-1)
    values = values[valid_lane_mask[: values.size]]
    values = values[np.isfinite(values) & (values > 0.5)]
    if values.size == 0:
        return None
    return float(np.median(values))


def _speed_limit_reference_mps(cache_data: Dict[str, np.ndarray]) -> Optional[float]:
    route_speed_limit = _masked_speed_limit_mps(
        cache_data.get("route_lanes_speed_limit", np.zeros((0, 1), dtype=np.float32)),
        cache_data.get("route_lanes_has_speed_limit", np.zeros((0, 1), dtype=bool)),
        cache_data.get("route_lanes_mask", np.zeros((0, 0), dtype=bool)),
    )
    if route_speed_limit is not None:
        return route_speed_limit
    return _masked_speed_limit_mps(
        cache_data.get("lanes_speed_limit", np.zeros((0, 1), dtype=np.float32)),
        cache_data.get("lanes_has_speed_limit", np.zeros((0, 1), dtype=bool)),
        cache_data.get("lanes_mask", np.zeros((0, 0), dtype=bool)),
    )


def _route_has_restrictive_control(cache_data: Dict[str, np.ndarray]) -> bool:
    route_lanes = np.asarray(cache_data.get("route_lanes", np.zeros((0, 0, 0), dtype=np.float32)), dtype=np.float32)
    route_mask = np.asarray(cache_data.get("route_lanes_mask", np.zeros((0, 0), dtype=bool)), dtype=bool)
    if route_lanes.ndim != 3 or route_lanes.shape[-1] < 12:
        return False

    tl_state = route_lanes[..., 8:12]
    if route_mask.shape != tl_state.shape[:2]:
        route_mask = np.ones(tl_state.shape[:2], dtype=bool)
    route_xy = route_lanes[..., :2]
    near_route_mask = (
        (route_xy[..., 0] >= -3.0)
        & (route_xy[..., 0] <= 28.0)
        & (np.abs(route_xy[..., 1]) <= 6.0)
    )
    restrictive_control = ((tl_state[..., 1] > 0.5) | (tl_state[..., 2] > 0.5)) & route_mask & near_route_mask
    return bool(restrictive_control.any())


@dataclass(frozen=True)
class NeighborEvidence:
    agent_idx: int
    valid_count: int
    following_score: float
    crossing_score: float
    lateral_score: float
    same_direction_score: float
    heading_diff_mean: float
    min_same_time_distance: float
    min_pair_distance: float
    leading_min_gap: float
    leading_min_thw: Optional[float]
    leading_closing_speed_peak: float
    crossing_time_offset: float
    lateral_closure: float
    lateral_min_gap: float
    lateral_overlap_ratio: float
    merge_gap_at_onset: float
    following_conflict_step: int
    lateral_conflict_step: int


class StyleSceneSplitter:
    """Split samples into high-precision straight-driving contexts and styles."""

    def __init__(
        self,
        time_delta: float = 0.1,
        min_scene_score: float = 0.58,
        min_scene_margin: float = 0.10,
        min_split_confidence: float = 0.55,
    ) -> None:
        self.time_delta = float(time_delta)
        self.min_scene_score = float(min_scene_score)
        self.min_scene_margin = float(min_scene_margin)
        self.min_split_confidence = float(min_split_confidence)

    def split(self, cache_data: Dict[str, np.ndarray]) -> StyleSceneSplitResult:
        ego_future = np.asarray(cache_data["ego_agent_future"], dtype=np.float32)
        neighbors_future = np.asarray(cache_data["neighbor_agents_future"], dtype=np.float32)
        valid_mask = _resolve_neighbor_valid_mask(cache_data, neighbors_future)

        ego_xy = ego_future[:, :2].astype(np.float32)
        ego_heading = _heading_from_traj(ego_future)
        ego_speed = _speed_from_xy(ego_xy, self.time_delta)
        ego_accel = _accel_from_speed(ego_speed, self.time_delta)
        ego_jerk = _jerk_from_accel(ego_accel, self.time_delta)

        ego_mean_speed = _safe_mean(ego_speed)
        ego_accel_peak = _safe_max(np.maximum(ego_accel, 0.0))
        ego_brake_peak = _safe_max(np.maximum(-ego_accel, 0.0))
        ego_jerk_peak = _safe_max(np.abs(ego_jerk))
        ego_jerk_p90 = _safe_percentile(np.abs(ego_jerk), 90.0)
        ego_lateral_disp = float(np.max(np.abs(ego_xy[:, 1]))) if ego_xy.size > 0 else 0.0
        ego_progress = float(ego_xy[-1, 0]) if ego_xy.shape[0] > 0 else 0.0
        ego_heading_change = (
            float(abs(_wrap_angle(float(ego_heading[-1] - ego_heading[0]))))
            if ego_heading.size >= 2
            else 0.0
        )
        ego_lateral_speed_peak, ego_lateral_onset_step = self._ego_lateral_motion_metrics(ego_xy)

        scenario_type = _scalar_cache_str(cache_data, "scenario_type", default="").lower()
        route_has_control = _route_has_restrictive_control(cache_data)
        route_speed_limit_mps = _speed_limit_reference_mps(cache_data)
        ego_speed_ratio_to_limit = None
        if route_speed_limit_mps is not None and route_speed_limit_mps > 0.5:
            ego_speed_ratio_to_limit = float(ego_mean_speed / route_speed_limit_mps)

        topology_bucket, route_lane_count, straightness_score = self._infer_topology(
            cache_data=cache_data,
            ego_heading_change=ego_heading_change,
            ego_lateral_disp=ego_lateral_disp,
            route_has_control=route_has_control,
            scenario_type=scenario_type,
        )

        evidences = self._collect_neighbor_evidences(
            ego_xy=ego_xy,
            ego_heading=ego_heading,
            ego_speed=ego_speed,
            neighbors_future=neighbors_future,
            valid_mask=valid_mask,
            ego_lateral_onset_step=ego_lateral_onset_step,
        )

        scene_scores, scene_evidences, scene_reason = self._aggregate_scene_scores(
            evidences=evidences,
            topology_bucket=topology_bucket,
            straightness_score=straightness_score,
            route_has_control=route_has_control,
            route_speed_limit_mps=route_speed_limit_mps,
            scenario_type=scenario_type,
            ego_progress=ego_progress,
            ego_lateral_disp=ego_lateral_disp,
            ego_lateral_onset_step=ego_lateral_onset_step,
            ego_lateral_speed_peak=ego_lateral_speed_peak,
            ego_heading_change=ego_heading_change,
        )
        scene_bucket, primary_bucket, secondary_bucket, primary_score, secondary_score, select_reason = self._select_scene_bucket(
            scene_scores
        )
        if scene_reason:
            scene_reason = f"{scene_reason}; {select_reason}"
        else:
            scene_reason = select_reason

        dominant_evidence = scene_evidences.get(scene_bucket, None) if scene_bucket != "none" else None
        following_evidence = scene_evidences.get("straight_car_follow", None)
        crossing_evidence = self._strongest_evidence(evidences, key="crossing_score")
        lateral_evidence = scene_evidences.get("straight_lane_change", None)
        global_min_distance = min([e.min_same_time_distance for e in evidences], default=1e6)
        nearby_agent_count = int(sum(1 for evidence in evidences if evidence.min_same_time_distance <= 25.0))
        lead_vehicle_present = bool(
            following_evidence is not None
            and np.isfinite(following_evidence.leading_min_gap)
            and following_evidence.leading_min_gap < 60.0
        )

        conflict_step = -1
        if scene_bucket == "straight_car_follow" and following_evidence is not None:
            conflict_step = following_evidence.following_conflict_step
        response_metrics = self._event_response_metrics(ego_speed, ego_accel, conflict_step)

        style_scores, style_label, style_reason = self._classify_style(
            scene_bucket=scene_bucket,
            dominant_evidence=dominant_evidence,
            response_metrics=response_metrics,
            ego_mean_speed=ego_mean_speed,
            ego_accel_peak=ego_accel_peak,
            ego_jerk_peak=ego_jerk_peak,
            ego_jerk_p90=ego_jerk_p90,
            ego_lateral_speed_peak=ego_lateral_speed_peak,
            ego_lateral_onset_step=ego_lateral_onset_step,
            ego_speed_ratio_to_limit=ego_speed_ratio_to_limit,
        )

        scene_confidence = self._scene_confidence(primary_score, secondary_score, scene_bucket)
        style_confidence = self._style_confidence(style_scores, style_label)
        split_confidence = float(min(scene_confidence, style_confidence))
        split_valid = bool(
            scene_bucket != "none"
            and style_label != "unknown"
            and split_confidence >= self.min_split_confidence
        )
        subset_id = f"{scene_bucket}__{style_label}" if split_valid else "invalid"

        return StyleSceneSplitResult(
            scene_bucket=scene_bucket,
            style_label=style_label,
            subset_id=subset_id,
            topology_bucket=topology_bucket,
            primary_bucket=primary_bucket,
            secondary_bucket=secondary_bucket,
            scene_confidence=scene_confidence,
            style_confidence=style_confidence,
            split_confidence=split_confidence,
            split_valid=split_valid,
            primary_score=primary_score,
            secondary_score=secondary_score,
            dominant_neighbor_idx=-1 if dominant_evidence is None else int(dominant_evidence.agent_idx),
            scene_reason=scene_reason,
            style_reason=style_reason,
            scene_score_vec=np.array([scene_scores[name] for name in SCENE_BUCKET_ORDER], dtype=np.float32),
            style_score_vec=np.array([style_scores[name] for name in STYLE_SCORE_ORDER], dtype=np.float32),
            global_min_distance=float(global_min_distance),
            following_min_gap=float(1e6 if following_evidence is None else following_evidence.leading_min_gap),
            following_min_thw=None if following_evidence is None else following_evidence.leading_min_thw,
            crossing_min_distance=float(1e6 if crossing_evidence is None else crossing_evidence.min_pair_distance),
            crossing_time_offset=None if crossing_evidence is None else crossing_evidence.crossing_time_offset,
            merge_min_gap=float(1e6 if lateral_evidence is None else lateral_evidence.merge_gap_at_onset),
            merge_lateral_closure=float(0.0 if lateral_evidence is None else lateral_evidence.lateral_closure),
            ego_mean_speed=ego_mean_speed,
            ego_accel_peak=ego_accel_peak,
            ego_brake_peak=ego_brake_peak,
            ego_jerk_peak=ego_jerk_peak,
            ego_jerk_p90=ego_jerk_p90,
            ego_progress=ego_progress,
            ego_speed_ratio_to_limit=ego_speed_ratio_to_limit,
            route_speed_limit_mps=route_speed_limit_mps,
            event_speed_drop_ratio=response_metrics["speed_drop_ratio"],
            event_brake_peak=response_metrics["brake_peak"],
            ego_lateral_disp=ego_lateral_disp,
            ego_lateral_speed_peak=ego_lateral_speed_peak,
            ego_lateral_onset_step=ego_lateral_onset_step,
            ego_heading_change=ego_heading_change,
            route_lane_count=route_lane_count,
            nearby_agent_count=nearby_agent_count,
            lead_vehicle_present=lead_vehicle_present,
            route_has_control=route_has_control,
        )

    def _infer_topology(
        self,
        cache_data: Dict[str, np.ndarray],
        ego_heading_change: float,
        ego_lateral_disp: float,
        route_has_control: bool,
        scenario_type: str,
    ) -> Tuple[str, int, float]:
        route_lanes_mask = np.asarray(cache_data.get("route_lanes_mask", np.zeros((0, 0), dtype=bool)), dtype=bool)
        route_lane_count = 0
        if route_lanes_mask.ndim == 2:
            route_lane_count = int(route_lanes_mask.any(axis=1).sum())
        elif route_lanes_mask.ndim == 1:
            route_lane_count = int(route_lanes_mask.sum())

        straightness_score = _score_falling(ego_heading_change, 0.08, 0.26)
        scenario_is_complex = self._scenario_is_complex(scenario_type)

        if route_has_control or scenario_is_complex:
            return "junction_like", route_lane_count, straightness_score
        if ego_heading_change >= 0.35:
            return "turning", route_lane_count, straightness_score
        if ego_lateral_disp >= 1.2 and straightness_score >= 0.55:
            return "lane_change_like", route_lane_count, straightness_score
        if straightness_score >= 0.50 and route_lane_count > 0:
            return "straight", route_lane_count, straightness_score
        return "unknown", route_lane_count, straightness_score

    def _collect_neighbor_evidences(
        self,
        ego_xy: np.ndarray,
        ego_heading: np.ndarray,
        ego_speed: np.ndarray,
        neighbors_future: np.ndarray,
        valid_mask: np.ndarray,
        ego_lateral_onset_step: Optional[float],
    ) -> list[NeighborEvidence]:
        evidences: list[NeighborEvidence] = []
        if neighbors_future.ndim != 3 or valid_mask.ndim != 2:
            return evidences

        onset_step = None if ego_lateral_onset_step is None else int(ego_lateral_onset_step)

        for agent_idx in range(neighbors_future.shape[0]):
            agent_valid = valid_mask[agent_idx]
            valid_steps = np.where(agent_valid[: ego_xy.shape[0]])[0]
            if valid_steps.size < 3:
                continue

            nbr_traj = neighbors_future[agent_idx, valid_steps]
            nbr_xy = nbr_traj[:, :2].astype(np.float32)
            nbr_heading = _heading_from_traj(nbr_traj)
            ego_xy_valid = ego_xy[valid_steps]
            ego_heading_valid = ego_heading[valid_steps]
            ego_speed_valid = ego_speed[valid_steps]
            nbr_speed = _speed_from_xy(nbr_xy, self.time_delta)

            same_time_distance = np.linalg.norm(nbr_xy - ego_xy_valid, axis=-1).astype(np.float32)
            pairwise_distance = _pairwise_distance(ego_xy_valid, nbr_xy)
            min_same_time_distance = _safe_min(same_time_distance, default=1e6)
            min_pair_distance = _safe_min(pairwise_distance, default=1e6)

            heading_diff = np.abs(_wrap_angle(nbr_heading - ego_heading_valid))
            heading_diff_mean = _safe_mean(heading_diff)
            same_direction_score = _score_falling(heading_diff_mean, np.deg2rad(8.0), np.deg2rad(45.0))

            longitudinal_gap = nbr_xy[:, 0] - ego_xy_valid[:, 0]
            lateral_gap = nbr_xy[:, 1] - ego_xy_valid[:, 1]
            abs_lat = np.abs(lateral_gap)
            abs_lon = np.abs(longitudinal_gap)

            leading_mask = (longitudinal_gap > 0.8) & (abs_lat < 1.8)
            lead_persistence = _safe_mean(leading_mask.astype(np.float32))
            leading_min_gap = _safe_min(longitudinal_gap[leading_mask], default=1e6)
            thw_values = None
            if leading_mask.any():
                thw_values = longitudinal_gap[leading_mask] / np.maximum(ego_speed_valid[leading_mask], 0.8)
            leading_min_thw = None if thw_values is None or thw_values.size == 0 else float(np.min(thw_values))
            leading_closing_speed_peak = _safe_max(np.maximum(ego_speed_valid - nbr_speed, 0.0))
            same_lane_term = float(np.exp(-np.median(abs_lat) / 1.2))
            gap_term = float(np.exp(-min(leading_min_gap, 35.0) / 14.0)) if np.isfinite(leading_min_gap) else 0.0
            thw_term = 0.0 if leading_min_thw is None else _score_falling(leading_min_thw, 1.2, 2.8)
            following_score = _clip01(
                0.35 * lead_persistence
                + 0.30 * same_direction_score
                + 0.20 * same_lane_term
                + 0.10 * gap_term
                + 0.05 * thw_term
            )

            crossing_time_offset = 1e6
            if pairwise_distance.size > 0:
                min_pair_idx = np.unravel_index(np.argmin(pairwise_distance), pairwise_distance.shape)
                crossing_time_offset = abs(int(valid_steps[min_pair_idx[0]]) - int(valid_steps[min_pair_idx[1]])) * self.time_delta
            angle_term = _score_rising(heading_diff_mean, np.deg2rad(25.0), np.deg2rad(80.0))
            crossing_spatial_term = float(np.exp(-min_pair_distance / 5.0))
            crossing_time_term = _score_falling(crossing_time_offset, 0.2, 1.0)
            crossing_score = _clip01(
                0.45 * angle_term
                + 0.35 * crossing_spatial_term
                + 0.20 * crossing_time_term
            )

            adjacent_lane_term = float(np.exp(-abs(np.median(abs_lat) - 3.5) / 1.4))
            lateral_closure = max(float(abs_lat[0] - np.min(abs_lat)), 0.0)
            lateral_closure_term = _score_rising(lateral_closure, 0.6, 2.8)
            overlap_ratio = _safe_mean(((abs_lat < 6.0) & (abs_lon < 18.0)).astype(np.float32))
            lateral_min_gap = _safe_min(abs_lon[(abs_lat < 6.0)], default=1e6)
            ego_lat_speed_peak = _safe_max(np.abs(np.diff(np.concatenate([[0.0], ego_xy_valid[:, 1]])) / self.time_delta))
            nbr_lat_speed_peak = _safe_max(np.abs(np.diff(np.concatenate([[0.0], nbr_xy[:, 1]])) / self.time_delta))
            merge_motion_term = _score_rising(max(ego_lat_speed_peak, nbr_lat_speed_peak), 0.4, 1.4)
            lateral_score = _clip01(
                0.32 * adjacent_lane_term
                + 0.26 * lateral_closure_term
                + 0.22 * overlap_ratio
                + 0.10 * same_direction_score
                + 0.10 * merge_motion_term
            )

            if onset_step is not None:
                onset_local = int(np.argmin(np.abs(valid_steps - onset_step)))
            else:
                onset_local = int(np.argmin(abs_lat + 0.30 * abs_lon))
            lateral_conflict_local = int(np.argmin(abs_lat + 0.30 * abs_lon))
            lateral_conflict_step = int(valid_steps[lateral_conflict_local])
            merge_gap_at_onset = float(abs(longitudinal_gap[onset_local]))

            following_conflict_step = -1
            if leading_mask.any():
                lead_indices = np.where(leading_mask)[0]
                following_conflict_step = int(valid_steps[lead_indices[np.argmin(longitudinal_gap[leading_mask])]])

            evidences.append(
                NeighborEvidence(
                    agent_idx=int(agent_idx),
                    valid_count=int(valid_steps.size),
                    following_score=float(following_score),
                    crossing_score=float(crossing_score),
                    lateral_score=float(lateral_score),
                    same_direction_score=float(same_direction_score),
                    heading_diff_mean=float(heading_diff_mean),
                    min_same_time_distance=float(min_same_time_distance),
                    min_pair_distance=float(min_pair_distance),
                    leading_min_gap=float(leading_min_gap),
                    leading_min_thw=leading_min_thw,
                    leading_closing_speed_peak=float(leading_closing_speed_peak),
                    crossing_time_offset=float(crossing_time_offset),
                    lateral_closure=float(lateral_closure),
                    lateral_min_gap=float(lateral_min_gap),
                    lateral_overlap_ratio=float(overlap_ratio),
                    merge_gap_at_onset=float(merge_gap_at_onset),
                    following_conflict_step=int(following_conflict_step),
                    lateral_conflict_step=int(lateral_conflict_step),
                )
            )

        return evidences

    def _aggregate_scene_scores(
        self,
        evidences: Iterable[NeighborEvidence],
        topology_bucket: str,
        straightness_score: float,
        route_has_control: bool,
        route_speed_limit_mps: Optional[float],
        scenario_type: str,
        ego_progress: float,
        ego_lateral_disp: float,
        ego_lateral_onset_step: Optional[float],
        ego_lateral_speed_peak: float,
        ego_heading_change: float,
    ) -> Tuple[Dict[str, float], Dict[str, Optional[NeighborEvidence]], str]:
        scene_scores = {name: 0.0 for name in SCENE_BUCKET_ORDER}
        scene_evidences: Dict[str, Optional[NeighborEvidence]] = {name: None for name in SCENE_BUCKET_ORDER}

        evidences = list(evidences)
        following_best = self._strongest_evidence(evidences, key="following_score")
        lateral_best = self._strongest_evidence(evidences, key="lateral_score")
        crossing_best = self._strongest_evidence(evidences, key="crossing_score")

        scenario_is_complex = self._scenario_is_complex(scenario_type)
        if scenario_is_complex:
            return scene_scores, scene_evidences, "scenario_type_marks_complex_non_straight_task"
        if route_has_control:
            return scene_scores, scene_evidences, "nearby_restrictive_route_control_detected"
        if topology_bucket not in ("straight", "lane_change_like") and straightness_score < 0.72:
            return scene_scores, scene_evidences, "topology_is_not_confidently_straight"
        if crossing_best is not None and crossing_best.crossing_score >= 0.42 and crossing_best.min_pair_distance <= 4.0:
            return scene_scores, scene_evidences, "crossing_like_interaction_detected_so_sample_is_abstained"

        lateral_onset_score = 0.0 if ego_lateral_onset_step is None else _score_falling(ego_lateral_onset_step, 8.0, 40.0)
        lane_change_motion_score = _clip01(
            0.45 * _score_rising(ego_lateral_disp, 1.2, 3.0)
            + 0.25 * lateral_onset_score
            + 0.20 * _score_rising(ego_lateral_speed_peak, 0.6, 1.8)
            + 0.10 * _score_falling(ego_heading_change, 0.05, 0.22)
        )

        lane_change_motion_gate = bool(
            ego_lateral_disp >= 1.0
            and (ego_lateral_onset_step is not None or ego_lateral_speed_peak >= 0.65)
            and straightness_score >= 0.52
        )
        lane_change_interaction_gate = bool(
            lateral_best is not None
            and lateral_best.lateral_score >= 0.46
            and lateral_best.same_direction_score >= 0.42
            and (crossing_best is None or crossing_best.crossing_score < 0.36)
        )
        lane_change_gate = bool(
            lane_change_motion_gate
            and (
                lane_change_interaction_gate
                or ego_lateral_disp >= 1.6
                or ego_lateral_speed_peak >= 0.95
            )
        )
        if lane_change_gate:
            interaction_support = 0.0 if lateral_best is None else lateral_best.lateral_score
            score = _clip01(0.45 * straightness_score + 0.35 * lane_change_motion_score + 0.20 * interaction_support)
            scene_scores["straight_lane_change"] = score
            scene_evidences["straight_lane_change"] = lateral_best

        car_follow_gate = bool(
            following_best is not None
            and ego_lateral_disp < 0.95
            and following_best.following_score >= 0.56
            and following_best.same_direction_score >= 0.65
            and following_best.leading_min_gap < 45.0
            and (following_best.leading_min_thw is None or following_best.leading_min_thw < 4.5)
        )
        if car_follow_gate and following_best is not None:
            follow_support = max(
                following_best.following_score,
                _score_falling(following_best.leading_min_gap, 12.0, 32.0),
            )
            score = _clip01(0.45 * straightness_score + 0.55 * follow_support)
            scene_scores["straight_car_follow"] = score
            scene_evidences["straight_car_follow"] = following_best

        no_interaction_strength = 1.0 - max(
            0.0 if following_best is None else following_best.following_score,
            0.0 if lateral_best is None else lateral_best.lateral_score,
            0.0 if crossing_best is None else crossing_best.crossing_score,
        )
        free_drive_gate = bool(
            ego_lateral_disp < 0.90
            and ego_progress >= 9.0
            and route_speed_limit_mps is not None
            and no_interaction_strength >= 0.48
            and not lane_change_gate
            and not car_follow_gate
        )
        if free_drive_gate:
            score = _clip01(
                0.45 * straightness_score
                + 0.30 * _score_rising(ego_progress, 9.0, 28.0)
                + 0.25 * no_interaction_strength
            )
            scene_scores["straight_free_drive"] = score

        return scene_scores, scene_evidences, ""

    def _select_scene_bucket(
        self,
        scene_scores: Dict[str, float],
    ) -> Tuple[str, str, str, float, float, str]:
        ranked = sorted(scene_scores.items(), key=lambda item: item[1], reverse=True)
        primary_bucket, primary_score = ranked[0]
        secondary_bucket, secondary_score = ranked[1]

        if primary_score < self.min_scene_score:
            return "none", "none", "none", float(primary_score), float(secondary_score), "no_context_reaches_min_scene_score"
        if primary_score - secondary_score < self.min_scene_margin:
            return "none", primary_bucket, secondary_bucket, float(primary_score), float(secondary_score), "context_assignment_is_too_ambiguous"
        return primary_bucket, primary_bucket, secondary_bucket, float(primary_score), float(secondary_score), "single_high_precision_context_selected"

    def _event_response_metrics(
        self,
        ego_speed: np.ndarray,
        ego_accel: np.ndarray,
        conflict_step: int,
    ) -> Dict[str, float]:
        if ego_speed.size == 0:
            return {
                "speed_drop_ratio": 0.0,
                "brake_peak": 0.0,
            }

        if conflict_step < 0:
            anchor = min(2, ego_speed.shape[0] - 1)
        else:
            anchor = int(np.clip(conflict_step, 1, max(ego_speed.shape[0] - 1, 1)))

        pre_slice = slice(max(anchor - 3, 0), max(anchor, 1))
        post_slice = slice(anchor, min(anchor + 4, ego_speed.shape[0]))
        baseline_speed = _safe_mean(ego_speed[pre_slice])
        post_min_speed = _safe_min(ego_speed[post_slice], default=baseline_speed)
        speed_drop = max(baseline_speed - post_min_speed, 0.0)
        speed_drop_ratio = float(speed_drop / max(baseline_speed, 1.0))

        accel_slice = slice(max(anchor - 2, 0), min(anchor + 3, ego_accel.shape[0]))
        brake_peak = _safe_max(np.maximum(-ego_accel[accel_slice], 0.0))
        return {
            "speed_drop_ratio": float(speed_drop_ratio),
            "brake_peak": float(brake_peak),
        }

    def _ego_lateral_motion_metrics(self, ego_xy: np.ndarray) -> Tuple[float, Optional[float]]:
        if ego_xy.ndim != 2 or ego_xy.shape[0] == 0:
            return 0.0, None

        y = ego_xy[:, 1]
        lat_speed = np.diff(np.concatenate([[0.0], y])) / max(self.time_delta, 1e-6)
        lat_speed_peak = _safe_max(np.abs(lat_speed))

        onset_candidates = np.where((np.abs(y) >= 0.75) | (np.abs(lat_speed) >= 0.45))[0]
        onset_step = None if onset_candidates.size == 0 else float(int(onset_candidates[0]))
        return float(lat_speed_peak), onset_step

    def _classify_style(
        self,
        scene_bucket: str,
        dominant_evidence: Optional[NeighborEvidence],
        response_metrics: Dict[str, float],
        ego_mean_speed: float,
        ego_accel_peak: float,
        ego_jerk_peak: float,
        ego_jerk_p90: float,
        ego_lateral_speed_peak: float,
        ego_lateral_onset_step: Optional[float],
        ego_speed_ratio_to_limit: Optional[float],
    ) -> Tuple[Dict[str, float], str, str]:
        scores = {name: 0.0 for name in STYLE_SCORE_ORDER}
        if scene_bucket == "none":
            return scores, "unknown", "scene_bucket_none"

        if scene_bucket == "straight_car_follow":
            if dominant_evidence is None:
                return scores, "unknown", "car_follow_context_has_no_lead_evidence"
            min_thw = 3.5 if dominant_evidence.leading_min_thw is None else dominant_evidence.leading_min_thw
            min_gap = dominant_evidence.leading_min_gap
            closing_speed = dominant_evidence.leading_closing_speed_peak
            scores["aggressive"] = max(
                _score_falling(min_thw, 1.0, 1.7),
                _score_falling(min_gap, 8.0, 15.0),
                _score_rising(closing_speed, 0.8, 3.0),
            )
            scores["conservative"] = max(
                _score_rising(min_thw, 1.9, 2.8),
                _score_rising(min_gap, 15.0, 26.0),
                _score_rising(response_metrics["brake_peak"], 1.0, 2.4) * _score_rising(response_metrics["speed_drop_ratio"], 0.10, 0.28),
            )
            aggressive_gate = bool(
                (min_thw <= 1.85 and closing_speed >= 0.35)
                or (min_gap <= 10.5 and min_thw <= 2.6 and closing_speed >= 0.80)
            )
            conservative_gate = bool(
                (min_thw >= 2.35 and min_gap >= 14.0)
                or (response_metrics["speed_drop_ratio"] >= 0.14 and response_metrics["brake_peak"] >= 1.0)
            )
            reason = "straight_car_follow_style_uses_headway_gap_and_closing_response"
        elif scene_bucket == "straight_lane_change":
            merge_gap = None if dominant_evidence is None else dominant_evidence.merge_gap_at_onset
            onset_step = 80.0 if ego_lateral_onset_step is None else ego_lateral_onset_step
            aggressive_terms = [
                _score_falling(onset_step, 8.0, 24.0),
                _score_rising(ego_lateral_speed_peak, 0.8, 1.8),
            ]
            conservative_terms = [
                _score_rising(onset_step, 22.0, 40.0),
                _score_falling(ego_lateral_speed_peak, 0.45, 1.0),
            ]
            aggressive_gate = bool(onset_step <= 22.0 and ego_lateral_speed_peak >= 0.7)
            conservative_gate = bool(onset_step >= 30.0 and ego_lateral_speed_peak <= 1.0)
            if merge_gap is not None:
                aggressive_terms.append(_score_falling(merge_gap, 10.0, 18.0))
                conservative_terms.append(_score_rising(merge_gap, 18.0, 28.0))
                aggressive_gate = bool(aggressive_gate and (merge_gap <= 14.0 or ego_lateral_speed_peak >= 1.1))
                conservative_gate = bool(conservative_gate or merge_gap >= 20.0)
                reason = "straight_lane_change_style_uses_gap_acceptance_and_lateral_commitment"
            else:
                reason = "straight_lane_change_style_uses_onset_and_lateral_commitment_without_neighbor_gap"
            scores["aggressive"] = max(aggressive_terms)
            scores["conservative"] = max(conservative_terms)
        elif scene_bucket == "straight_free_drive":
            if ego_speed_ratio_to_limit is None:
                return scores, "unknown", "free_drive_has_no_speed_limit_reference"
            scores["aggressive"] = max(
                _score_rising(ego_speed_ratio_to_limit, 0.74, 0.92),
                0.55 * _score_rising(ego_accel_peak, 0.8, 1.7) + 0.45 * _score_rising(ego_jerk_p90, 18.0, 60.0),
            )
            scores["conservative"] = max(
                _score_falling(ego_speed_ratio_to_limit, 0.54, 0.74),
                0.70 * _score_falling(ego_accel_peak, 0.5, 1.3) + 0.30 * _score_falling(ego_jerk_p90, 12.0, 30.0),
            )
            aggressive_gate = bool(ego_speed_ratio_to_limit >= 0.82 and (ego_accel_peak >= 0.9 or ego_jerk_p90 >= 22.0))
            conservative_gate = bool(ego_speed_ratio_to_limit <= 0.68 and ego_accel_peak <= 1.5)
            reason = "straight_free_drive_style_uses_speed_ratio_to_limit_and_robust_longitudinal_dynamics"
        else:
            return scores, "unknown", "unsupported_context_bucket"

        scores["normal"] = _clip01(0.60 * (1.0 - max(scores["aggressive"], scores["conservative"])) + 0.40 * (1.0 - abs(scores["aggressive"] - scores["conservative"])))
        label = self._resolve_style_label(aggressive_gate, conservative_gate, scores)
        if label == "unknown":
            return scores, label, f"{reason}_but_extreme_rules_conflict"
        return scores, label, reason

    def _resolve_style_label(
        self,
        aggressive_gate: bool,
        conservative_gate: bool,
        scores: Dict[str, float],
    ) -> str:
        if aggressive_gate and conservative_gate:
            if scores["aggressive"] >= scores["conservative"] + 0.10:
                return "aggressive"
            if scores["conservative"] >= scores["aggressive"] + 0.10:
                return "conservative"
            return "unknown"
        if aggressive_gate:
            return "aggressive"
        if conservative_gate:
            return "conservative"
        return "normal"

    def _scene_confidence(self, primary_score: float, secondary_score: float, scene_bucket: str) -> float:
        if scene_bucket == "none":
            return 0.0
        margin_term = _clip01((primary_score - secondary_score) / max(self.min_scene_margin, 1e-6))
        return _clip01(0.70 * primary_score + 0.30 * margin_term)

    def _style_confidence(self, style_scores: Dict[str, float], style_label: str) -> float:
        if style_label == "unknown":
            return 0.0
        if style_label == "normal":
            aggressive = style_scores["aggressive"]
            conservative = style_scores["conservative"]
            return _clip01(
                0.45
                + 0.35 * (1.0 - max(aggressive, conservative))
                + 0.20 * (1.0 - abs(aggressive - conservative))
            )
        return _clip01(0.55 + 0.45 * style_scores[style_label])

    def _scenario_is_complex(self, scenario_type: str) -> bool:
        scenario_type = scenario_type.lower()
        return any(keyword in scenario_type for keyword in STRAIGHT_COMPLEX_SCENARIO_KEYWORDS)

    def _strongest_evidence(
        self,
        evidences: Iterable[NeighborEvidence],
        key: str,
    ) -> Optional[NeighborEvidence]:
        best = None
        best_score = -1.0
        for evidence in evidences:
            score = float(getattr(evidence, key))
            if score > best_score:
                best_score = score
                best = evidence
        return best

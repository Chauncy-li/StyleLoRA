"""Deterministic retrieval embeddings derived from style_scene_split outputs."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EXPLICIT_SIDECAR_FEATURE_NAMES: Sequence[str] = (
    "scene_is_free_drive",
    "scene_is_car_follow",
    "scene_is_lane_change",
    "split_confidence",
    "scene_confidence",
    "style_confidence",
    "global_min_distance_log1p",
    "following_min_gap_log1p",
    "following_min_thw",
    "crossing_min_distance_log1p",
    "crossing_time_offset",
    "merge_min_gap_log1p",
    "merge_lateral_closure",
    "ego_mean_speed",
    "ego_accel_peak",
    "ego_brake_peak",
    "ego_jerk_p90_log1p",
    "ego_progress",
    "ego_speed_ratio_to_limit",
    "event_speed_drop_ratio",
    "event_brake_peak",
    "ego_lateral_disp",
    "ego_lateral_speed_peak",
    "ego_lateral_onset_step",
    "ego_heading_change",
    "route_lane_count_log1p",
    "nearby_agent_count_log1p",
    "lead_vehicle_present",
    "route_has_control",
)

POOLED_SCENE_FEATURE_NAMES: Sequence[str] = (
    "ego_current_x",
    "ego_current_y",
    "ego_current_cos",
    "ego_current_sin",
    "ego_current_vx",
    "ego_current_vy",
    "ego_past_last_x",
    "ego_past_last_y",
    "ego_past_mean_speed",
    "ego_past_progress",
    "neighbor_count",
    "neighbor_min_dist",
    "neighbor_mean_dist",
    "neighbor_mean_x",
    "neighbor_mean_abs_y",
    "neighbor_mean_speed",
    "neighbor_nearest_1",
    "neighbor_nearest_2",
    "neighbor_nearest_3",
    "static_count",
    "static_min_dist",
    "static_mean_dist",
    "lane_count",
    "lane_mean_x",
    "lane_mean_y",
    "lane_mean_cos",
    "lane_mean_sin",
    "route_count",
    "route_mean_x",
    "route_mean_y",
    "route_mean_cos",
    "route_mean_sin",
    "route_speed_limit_median",
    "route_restrictive_control_ratio",
)


def _opt_float(value: object) -> float:
    if value is None:
        return np.nan
    value = float(value)
    if not np.isfinite(value):
        return np.nan
    return value


def build_explicit_sidecar_embedding(record: Dict[str, object]) -> Tuple[np.ndarray, List[str]]:
    scene_bucket = str(record.get("scene_bucket", "none"))

    def log1p_optional(value: object) -> float:
        value = _opt_float(value)
        if np.isnan(value):
            return np.nan
        return float(np.log1p(max(value, 0.0)))

    vector = np.asarray(
        [
            float(scene_bucket == "straight_free_drive"),
            float(scene_bucket == "straight_car_follow"),
            float(scene_bucket == "straight_lane_change"),
            _opt_float(record.get("split_confidence", 0.0)),
            _opt_float(record.get("scene_confidence", 0.0)),
            _opt_float(record.get("style_confidence", 0.0)),
            log1p_optional(record.get("global_min_distance", None)),
            log1p_optional(record.get("following_min_gap", None)),
            _opt_float(record.get("following_min_thw", None)),
            log1p_optional(record.get("crossing_min_distance", None)),
            _opt_float(record.get("crossing_time_offset", None)),
            log1p_optional(record.get("merge_min_gap", None)),
            _opt_float(record.get("merge_lateral_closure", None)),
            _opt_float(record.get("ego_mean_speed", 0.0)),
            _opt_float(record.get("ego_accel_peak", 0.0)),
            _opt_float(record.get("ego_brake_peak", 0.0)),
            log1p_optional(record.get("ego_jerk_p90", None)),
            _opt_float(record.get("ego_progress", 0.0)),
            _opt_float(record.get("ego_speed_ratio_to_limit", None)),
            _opt_float(record.get("event_speed_drop_ratio", 0.0)),
            _opt_float(record.get("event_brake_peak", 0.0)),
            _opt_float(record.get("ego_lateral_disp", 0.0)),
            _opt_float(record.get("ego_lateral_speed_peak", 0.0)),
            _opt_float(record.get("ego_lateral_onset_step", None)),
            _opt_float(record.get("ego_heading_change", 0.0)),
            float(np.log1p(max(int(record.get("route_lane_count", 0)), 0))),
            float(np.log1p(max(int(record.get("nearby_agent_count", 0)), 0))),
            float(bool(record.get("lead_vehicle_present", False))),
            float(bool(record.get("route_has_control", False))),
        ],
        dtype=np.float32,
    )
    return vector, list(EXPLICIT_SIDECAR_FEATURE_NAMES)


def _safe_array(cache_data, key: str, dims: int | None = None) -> np.ndarray:
    if hasattr(cache_data, "files") and key in cache_data.files:
        arr = np.asarray(cache_data[key])
    else:
        arr = np.zeros((0,), dtype=np.float32)
    if dims is not None and arr.ndim != dims:
        return np.zeros((0,) * max(dims, 1), dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def _safe_bool_array(cache_data, key: str, fallback_shape: Tuple[int, ...]) -> np.ndarray:
    if hasattr(cache_data, "files") and key in cache_data.files:
        arr = np.asarray(cache_data[key], dtype=bool)
        if arr.shape == fallback_shape:
            return arr
    return np.zeros(fallback_shape, dtype=bool)


def _nonzero_valid_mask(traj: np.ndarray) -> np.ndarray:
    if traj.ndim != 3:
        return np.zeros((0, 0), dtype=bool)
    return np.linalg.norm(traj[..., :2], axis=-1) > 1e-4


def _speed_from_xy(xy: np.ndarray) -> np.ndarray:
    if xy.ndim != 2 or xy.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    delta = np.diff(xy[:, :2], axis=0)
    return np.linalg.norm(delta, axis=-1).astype(np.float32)


def _heading_summary(polyline: np.ndarray) -> Tuple[float, float]:
    if polyline.ndim != 2 or polyline.shape[0] < 2:
        return 1.0, 0.0
    delta = polyline[-1, :2] - polyline[0, :2]
    heading = float(np.arctan2(delta[1], delta[0]))
    return float(np.cos(heading)), float(np.sin(heading))


def build_pooled_scene_embedding(cache_data) -> Tuple[np.ndarray, List[str]]:
    ego_current = _safe_array(cache_data, "ego_current_state")
    ego_past = _safe_array(cache_data, "ego_agent_past", dims=2)
    neighbor_past = _safe_array(cache_data, "neighbor_agents_past", dims=3)
    static_objects = _safe_array(cache_data, "static_objects", dims=2)
    lanes = _safe_array(cache_data, "lanes", dims=3)
    route_lanes = _safe_array(cache_data, "route_lanes", dims=3)
    route_speed_limit = _safe_array(cache_data, "route_lanes_speed_limit")
    neighbor_mask = _safe_bool_array(
        cache_data,
        "neighbor_agents_past_mask",
        fallback_shape=neighbor_past.shape[:2] if neighbor_past.ndim == 3 else (0, 0),
    )

    ego_current_slice = np.zeros((6,), dtype=np.float32)
    if ego_current.size > 0:
        ego_current_slice[: min(6, ego_current.shape[0])] = ego_current[:6]

    ego_past_last = np.zeros((2,), dtype=np.float32)
    ego_past_speed = 0.0
    ego_past_progress = 0.0
    if ego_past.ndim == 2 and ego_past.shape[0] > 0:
        ego_past_last = ego_past[-1, :2].astype(np.float32)
        ego_speed = _speed_from_xy(ego_past[:, :2])
        ego_past_speed = float(np.mean(ego_speed)) if ego_speed.size > 0 else 0.0
        ego_past_progress = float(ego_past[-1, 0] - ego_past[0, 0]) if ego_past.shape[0] > 1 else 0.0

    neighbor_positions: List[np.ndarray] = []
    neighbor_speeds: List[float] = []
    if neighbor_past.ndim == 3:
        fallback_mask = _nonzero_valid_mask(neighbor_past)
        if neighbor_mask.shape != fallback_mask.shape:
            neighbor_mask = fallback_mask
        else:
            neighbor_mask = np.logical_or(neighbor_mask, fallback_mask)

        for agent_idx in range(neighbor_past.shape[0]):
            valid_steps = np.where(neighbor_mask[agent_idx])[0]
            if valid_steps.size == 0:
                continue
            traj = neighbor_past[agent_idx, valid_steps]
            last_state = traj[-1]
            neighbor_positions.append(last_state[:2].astype(np.float32))
            traj_speed = _speed_from_xy(traj[:, :2])
            neighbor_speeds.append(float(np.mean(traj_speed)) if traj_speed.size > 0 else 0.0)

    neighbor_count = len(neighbor_positions)
    neighbor_min_dist = 0.0
    neighbor_mean_dist = 0.0
    neighbor_mean_x = 0.0
    neighbor_mean_abs_y = 0.0
    neighbor_mean_speed = 0.0
    nearest_three = np.zeros((3,), dtype=np.float32)
    if neighbor_positions:
        positions = np.stack(neighbor_positions, axis=0)
        distances = np.linalg.norm(positions, axis=-1)
        distances_sorted = np.sort(distances)
        nearest_three[: min(3, distances_sorted.shape[0])] = distances_sorted[:3]
        neighbor_min_dist = float(distances.min())
        neighbor_mean_dist = float(distances.mean())
        neighbor_mean_x = float(np.mean(positions[:, 0]))
        neighbor_mean_abs_y = float(np.mean(np.abs(positions[:, 1])))
        neighbor_mean_speed = float(np.mean(np.asarray(neighbor_speeds, dtype=np.float32))) if neighbor_speeds else 0.0

    static_count = 0
    static_min_dist = 0.0
    static_mean_dist = 0.0
    if static_objects.ndim == 2 and static_objects.shape[0] > 0:
        valid_mask = np.linalg.norm(static_objects[:, :2], axis=-1) > 1e-4
        valid_static = static_objects[valid_mask]
        static_count = int(valid_static.shape[0])
        if static_count > 0:
            static_dist = np.linalg.norm(valid_static[:, :2], axis=-1)
            static_min_dist = float(static_dist.min())
            static_mean_dist = float(static_dist.mean())

    lane_count = 0
    lane_mean_x = 0.0
    lane_mean_y = 0.0
    lane_mean_cos = 1.0
    lane_mean_sin = 0.0
    if lanes.ndim == 3 and lanes.shape[0] > 0:
        lane_valid = np.linalg.norm(lanes[..., :2], axis=-1) > 1e-4
        lane_keep = lane_valid.any(axis=1)
        valid_lanes = lanes[lane_keep]
        lane_count = int(valid_lanes.shape[0])
        if lane_count > 0:
            centers = valid_lanes[:, valid_lanes.shape[1] // 2, :2]
            lane_mean_x = float(np.mean(centers[:, 0]))
            lane_mean_y = float(np.mean(centers[:, 1]))
            headings = np.asarray([_heading_summary(polyline[:, :2]) for polyline in valid_lanes], dtype=np.float32)
            lane_mean_cos = float(np.mean(headings[:, 0]))
            lane_mean_sin = float(np.mean(headings[:, 1]))

    route_count = 0
    route_mean_x = 0.0
    route_mean_y = 0.0
    route_mean_cos = 1.0
    route_mean_sin = 0.0
    route_speed_limit_median = 0.0
    route_restrictive_control_ratio = 0.0
    if route_lanes.ndim == 3 and route_lanes.shape[0] > 0:
        route_valid = np.linalg.norm(route_lanes[..., :2], axis=-1) > 1e-4
        route_keep = route_valid.any(axis=1)
        valid_routes = route_lanes[route_keep]
        route_count = int(valid_routes.shape[0])
        if route_count > 0:
            centers = valid_routes[:, valid_routes.shape[1] // 2, :2]
            route_mean_x = float(np.mean(centers[:, 0]))
            route_mean_y = float(np.mean(centers[:, 1]))
            headings = np.asarray([_heading_summary(polyline[:, :2]) for polyline in valid_routes], dtype=np.float32)
            route_mean_cos = float(np.mean(headings[:, 0]))
            route_mean_sin = float(np.mean(headings[:, 1]))
            if route_speed_limit.size > 0:
                route_speed_values = route_speed_limit.reshape(-1)
                route_speed_values = route_speed_values[np.isfinite(route_speed_values) & (route_speed_values > 0.0)]
                if route_speed_values.size > 0:
                    route_speed_limit_median = float(np.median(route_speed_values))
            if valid_routes.shape[-1] >= 12:
                tl_state = valid_routes[..., 8:12]
                restrictive = np.logical_or(tl_state[..., 1] > 0.5, tl_state[..., 2] > 0.5)
                route_restrictive_control_ratio = float(np.mean(restrictive.astype(np.float32)))

    vector = np.asarray(
        [
            ego_current_slice[0],
            ego_current_slice[1],
            ego_current_slice[2],
            ego_current_slice[3],
            ego_current_slice[4],
            ego_current_slice[5],
            ego_past_last[0],
            ego_past_last[1],
            ego_past_speed,
            ego_past_progress,
            float(neighbor_count),
            neighbor_min_dist,
            neighbor_mean_dist,
            neighbor_mean_x,
            neighbor_mean_abs_y,
            neighbor_mean_speed,
            float(nearest_three[0]),
            float(nearest_three[1]),
            float(nearest_three[2]),
            float(static_count),
            static_min_dist,
            static_mean_dist,
            float(lane_count),
            lane_mean_x,
            lane_mean_y,
            lane_mean_cos,
            lane_mean_sin,
            float(route_count),
            route_mean_x,
            route_mean_y,
            route_mean_cos,
            route_mean_sin,
            route_speed_limit_median,
            route_restrictive_control_ratio,
        ],
        dtype=np.float32,
    )
    return vector, list(POOLED_SCENE_FEATURE_NAMES)


def robust_standardize(vectors: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """Robust standardization with NaN-safe median/IQR handling."""

    if vectors.ndim != 2:
        raise ValueError(f"Expected 2-D feature matrix, got shape={vectors.shape}")

    valid_columns = np.isfinite(vectors).any(axis=0)
    median = np.zeros((vectors.shape[1],), dtype=np.float32)
    q25 = np.zeros((vectors.shape[1],), dtype=np.float32)
    q75 = np.zeros((vectors.shape[1],), dtype=np.float32)
    if bool(valid_columns.any()):
        valid_vectors = vectors[:, valid_columns]
        median[valid_columns] = np.nanmedian(valid_vectors, axis=0).astype(np.float32)
        q25[valid_columns] = np.nanpercentile(valid_vectors, 25.0, axis=0).astype(np.float32)
        q75[valid_columns] = np.nanpercentile(valid_vectors, 75.0, axis=0).astype(np.float32)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(vectors), vectors, median[None, :])
    normalized = (filled - median[None, :]) / scale[None, :]
    norm = np.linalg.norm(normalized, axis=-1, keepdims=True)
    norm = np.where(norm > 1e-6, norm, 1.0)
    normalized = normalized / norm

    stats = {
        "median": [float(item) for item in median.tolist()],
        "q25": [float(item) for item in q25.tolist()],
        "q75": [float(item) for item in q75.tolist()],
        "scale": [float(item) for item in scale.tolist()],
    }
    return normalized.astype(np.float32), stats

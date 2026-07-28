"""Hard behavior-metric extraction for the continuous style representation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from .schema import CANONICAL_AXIS_BY_SCENE, RAW_BEHAVIOR_METRIC_BY_SCENE

DEFAULT_DT = 0.1
DEFAULT_LANE_WIDTH_M = 3.6
DEFAULT_SAME_LANE_HALF_WIDTH_M = 1.9
DEFAULT_TIME_GAP_VMIN_MPS = 1.5
DEFAULT_TTC_MAX_S = 10.0
DEFAULT_TRAJ_SMOOTH_WINDOW = 3
DEFAULT_NEIGHBOR_SMOOTH_WINDOW = 3
DEFAULT_ONSET_THRESHOLD_M = 0.75
DEFAULT_ONSET_LATERAL_SPEED_MPS = 0.45
DEFAULT_MIN_ONSET_SUSTAINED_STEPS = 2
DEFAULT_BRAKE_TRIGGER_MPS2 = 0.40
DEFAULT_CLOSING_SPEED_MIN_MPS = 0.25
DEFAULT_FOLLOW_PERSISTENCE_STEPS = 3
DEFAULT_BRAKE_SUSTAINED_STEPS = 3
# Following drivers often begin a safe response by a mild, sustained
# deceleration before a large brake event is visible.  The relief test below
# prevents a lead vehicle's own acceleration from being mistaken for ego
# response.
DEFAULT_FOLLOW_RESPONSE_DECEL_MPS2 = 0.15
DEFAULT_FOLLOW_RESPONSE_CLOSING_RELIEF_MPS = 0.20
# Free-driving axes are only meaningful when the vehicle is not being
# immediately constrained by a same-lane leader and has a usable speed target.
DEFAULT_FREE_LEAD_CLEARANCE_M = 30.0
DEFAULT_FREE_HEADWAY_CLEARANCE_S = 3.0
DEFAULT_CONTROL_MIN_LOOKAHEAD_M = 35.0
DEFAULT_CONTROL_BEHIND_TOLERANCE_M = 5.0
DEFAULT_CONTROL_LATERAL_TOLERANCE_M = 2.0 * DEFAULT_LANE_WIDTH_M
DEFAULT_FREE_MIN_VALID_STEPS = 5
DEFAULT_SPEED_OPPORTUNITY_MPS = 2.0
DEFAULT_RESPONSE_HORIZON_S = 2.0
DEFAULT_UPPER_TAIL_FRACTION = 0.20
DEFAULT_LOCAL_ROUTE_MATCH_DISTANCE_M = 8.0

EGO_LENGTH_M = float(get_pacifica_parameters().length)


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(float(value), float(low), float(high)))


def _safe_percentile(values: Sequence[float], percentile: float, default: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, percentile))


def _trimmed_mean(values: Sequence[float], trim_fraction: float = 0.10, default: float = 0.0) -> float:
    """Return a symmetric trimmed mean while remaining well-defined for short windows."""

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    arr.sort()
    trim = min(int(np.floor(float(trim_fraction) * arr.size)), max((arr.size - 1) // 2, 0))
    if trim > 0:
        arr = arr[trim:-trim]
    return float(np.mean(arr)) if arr.size > 0 else float(default)


def _upper_tail_mean(values: Sequence[float], fraction: float = DEFAULT_UPPER_TAIL_FRACTION, default: float = 0.0) -> float:
    """Mean the largest fraction of finite values instead of trusting one peak."""

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    count = max(int(np.ceil(float(fraction) * arr.size)), 1)
    return float(np.mean(np.partition(arr, arr.size - count)[-count:]))


def _trajectory_motion_stats(xy: np.ndarray, tangents: np.ndarray | None = None) -> Dict[str, float]:
    """Summarize actual forward movement for sample-quality gating and auditing."""

    points = np.asarray(xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        return {
            "ego_path_length_m": 0.0,
            "ego_net_displacement_m": 0.0,
            "ego_forward_progress_m": 0.0,
        }
    deltas = points[1:] - points[:-1]
    path_length = float(np.sum(np.linalg.norm(deltas, axis=-1))) if deltas.size > 0 else 0.0
    net_delta = points[-1] - points[0]
    net_displacement = float(np.linalg.norm(net_delta))
    if tangents is not None:
        tangent_arr = np.asarray(tangents, dtype=np.float32)
        tangent = tangent_arr[0] if tangent_arr.ndim == 2 and tangent_arr.shape[0] > 0 else None
    else:
        tangent = None
    if tangent is None or not np.all(np.isfinite(tangent)) or float(np.linalg.norm(tangent)) <= 1e-4:
        forward_progress = net_displacement
    else:
        forward_progress = float(np.dot(net_delta, tangent / max(float(np.linalg.norm(tangent)), 1e-6)))
    return {
        "ego_path_length_m": path_length,
        "ego_net_displacement_m": net_displacement,
        "ego_forward_progress_m": forward_progress,
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    if parsed >= 1e5:
        return None
    return float(parsed)


def _npz_array(cache_data, key: str, default: np.ndarray) -> np.ndarray:
    if hasattr(cache_data, "files") and key in cache_data.files:
        return np.asarray(cache_data[key])
    return np.asarray(default)


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    series = np.asarray(values, dtype=np.float32).reshape(-1)
    if series.size == 0 or int(window) <= 1:
        return series.astype(np.float32)
    window = min(int(window), int(series.size))
    if window <= 1:
        return series.astype(np.float32)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(series, (pad_left, pad_right), mode="edge")
    kernel = np.full((window,), 1.0 / float(window), dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _moving_average_matrix(values: np.ndarray, window: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or int(window) <= 1:
        return matrix.astype(np.float32)
    return np.stack(
        [_moving_average_1d(matrix[:, column], window) for column in range(matrix.shape[1])],
        axis=1,
    ).astype(np.float32)


def _smooth_xy(values: np.ndarray, window: int) -> np.ndarray:
    xy = np.asarray(values, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] == 0 or int(window) <= 1:
        return xy.astype(np.float32)
    smoothed = _moving_average_matrix(xy, window)
    smoothed[0] = xy[0]
    return smoothed.astype(np.float32)


def _finite_difference_1d(values: np.ndarray, dt: float) -> np.ndarray:
    series = np.asarray(values, dtype=np.float32).reshape(-1)
    if series.size <= 1:
        return np.zeros((0,), dtype=np.float32)
    return ((series[1:] - series[:-1]) / max(float(dt), 1e-6)).astype(np.float32)


def _velocity_from_xy(xy: np.ndarray, dt: float) -> np.ndarray:
    coords = np.asarray(xy, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if coords.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    delta = (coords[1:] - coords[:-1]) / max(float(dt), 1e-6)
    first = delta[:1]
    return np.concatenate([first, delta], axis=0).astype(np.float32)


def _speed_from_xy(xy: np.ndarray, dt: float) -> np.ndarray:
    velocity = _velocity_from_xy(xy, dt)
    return np.linalg.norm(velocity, axis=-1).astype(np.float32)


def _local_speed_limit_series(
    *,
    ego_xy: np.ndarray,
    route_lanes: np.ndarray,
    route_mask: np.ndarray,
    route_speed_limits: np.ndarray,
    route_has_speed_limits: np.ndarray,
    lanes: np.ndarray,
    lanes_mask: np.ndarray,
    lane_speed_limits: np.ndarray,
    lane_has_speed_limits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match each ego position to its nearest speed-limited route lane.

    A mean over every route-lane element confounds a current 25 mph lane with a
    downstream 35 mph lane.  The cache has one speed limit per lane element,
    so the behaviour metric uses a local nearest-lane value at every future
    frame and only falls back to nearby ordinary lanes when no route lane is
    available.
    """

    positions = np.asarray(ego_xy, dtype=np.float32)
    limits = np.full((positions.shape[0],), np.nan, dtype=np.float32)

    def _match(
        polylines: np.ndarray,
        mask: np.ndarray,
        speed_limits: np.ndarray,
        has_speed_limits: np.ndarray,
        unresolved: np.ndarray,
    ) -> None:
        if polylines.ndim != 3 or polylines.shape[-1] < 2 or positions.shape[0] == 0:
            return
        lane_limits = np.asarray(speed_limits, dtype=np.float32).reshape(-1)
        lane_has_limit = np.asarray(has_speed_limits, dtype=bool).reshape(-1)
        if mask.shape[:2] == polylines.shape[:2]:
            point_valid = np.any(mask, axis=1)
        else:
            point_valid = np.any(np.linalg.norm(polylines[..., :2], axis=-1) > 1e-4, axis=1)
            mask = np.linalg.norm(polylines[..., :2], axis=-1) > 1e-4
        usable_count = min(polylines.shape[0], lane_limits.size, lane_has_limit.size, point_valid.size)
        usable_lanes = np.zeros((polylines.shape[0],), dtype=bool)
        usable_lanes[:usable_count] = (
            point_valid[:usable_count]
            & lane_has_limit[:usable_count]
            & (lane_limits[:usable_count] > 0.5)
        )
        if not np.any(usable_lanes):
            return
        point_lane_index = np.repeat(np.arange(polylines.shape[0], dtype=np.int64), polylines.shape[1])
        valid_point = mask.reshape(-1) & usable_lanes[point_lane_index]
        lane_points = polylines[..., :2].reshape(-1, 2)[valid_point]
        lane_ids = point_lane_index[valid_point]
        if lane_points.shape[0] == 0:
            return
        unresolved_indices = np.where(unresolved)[0]
        distance_sq = np.sum(
            (positions[unresolved_indices, None, :2] - lane_points[None, :, :]) ** 2,
            axis=2,
        )
        closest = np.argmin(distance_sq, axis=1)
        closest_distance_sq = distance_sq[np.arange(closest.shape[0]), closest]
        accepted = closest_distance_sq <= DEFAULT_LOCAL_ROUTE_MATCH_DISTANCE_M**2
        if np.any(accepted):
            limits[unresolved_indices[accepted]] = lane_limits[lane_ids[closest[accepted]]]

    unresolved = np.ones((positions.shape[0],), dtype=bool)
    _match(route_lanes, route_mask, route_speed_limits, route_has_speed_limits, unresolved)
    unresolved = ~np.isfinite(limits)
    _match(lanes, lanes_mask, lane_speed_limits, lane_has_speed_limits, unresolved)
    valid = np.isfinite(limits) & (limits > 0.5)
    return limits.astype(np.float32), valid.astype(bool)


def _build_route_segment_reference(cache_data) -> tuple[np.ndarray, np.ndarray]:
    def _collect(polyline_key: str, mask_key: str) -> tuple[np.ndarray, np.ndarray]:
        polylines = np.asarray(_npz_array(cache_data, polyline_key, np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
        mask = np.asarray(_npz_array(cache_data, mask_key, np.zeros((0, 0), dtype=bool)), dtype=bool)
        if polylines.ndim != 3 or polylines.shape[-1] < 2:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

        segment_midpoints = []
        segment_tangents = []
        lane_count = polylines.shape[0]
        for lane_index in range(lane_count):
            lane_xy = polylines[lane_index, :, :2].astype(np.float32)
            if mask.shape[:2] == polylines.shape[:2]:
                valid_lane_xy = lane_xy[mask[lane_index]]
            else:
                valid_lane_xy = lane_xy[np.linalg.norm(lane_xy, axis=-1) > 1e-4]
            if valid_lane_xy.shape[0] < 2:
                continue
            segments = valid_lane_xy[1:] - valid_lane_xy[:-1]
            lengths = np.linalg.norm(segments, axis=-1)
            valid = lengths > 1e-4
            if not np.any(valid):
                continue
            segment_midpoints.append(0.5 * (valid_lane_xy[:-1][valid] + valid_lane_xy[1:][valid]))
            segment_tangents.append((segments[valid] / lengths[valid, None]).astype(np.float32))
        if not segment_midpoints:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        return (
            np.concatenate(segment_midpoints, axis=0).astype(np.float32),
            np.concatenate(segment_tangents, axis=0).astype(np.float32),
        )

    route_midpoints, route_tangents = _collect("route_lanes", "route_lanes_mask")
    if route_midpoints.shape[0] > 0:
        return route_midpoints, route_tangents
    return _collect("lanes", "lanes_mask")


def _estimate_tangents_for_xy(
    xy: np.ndarray,
    route_midpoints: np.ndarray,
    route_tangents: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(xy, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if route_midpoints.shape[0] > 0 and route_tangents.shape[0] == route_midpoints.shape[0]:
        # A purely pointwise nearest-segment lookup can jump between parallel
        # route lanes or branch segments.  Keep the map geometry as the primary
        # reference, but choose a locally coherent tangent sequence.
        diff = coords[:, None, :] - route_midpoints[None, :, :]
        distance_sq = np.sum(diff * diff, axis=-1)
        tangents = np.zeros((coords.shape[0], 2), dtype=np.float32)
        previous_tangent = None
        for step in range(coords.shape[0]):
            row = distance_sq[step]
            nearest_distance = float(np.min(row))
            candidate_indices = np.where(row <= nearest_distance + 0.75 * 0.75)[0]
            if candidate_indices.size == 0:
                candidate_indices = np.asarray([int(np.argmin(row))], dtype=np.int64)

            candidates = route_tangents[candidate_indices].astype(np.float32)
            if previous_tangent is None:
                selected_index = int(candidate_indices[np.argmin(row[candidate_indices])])
                selected = route_tangents[selected_index].astype(np.float32)
            else:
                alignment = np.sum(candidates * previous_tangent[None, :], axis=-1)
                # Prefer a geometrically nearby segment whose orientation is
                # consistent with the preceding route frame.
                score = row[candidate_indices] + 0.35 * (1.0 - alignment)
                selected_index = int(candidate_indices[int(np.argmin(score))])
                selected = route_tangents[selected_index].astype(np.float32)
                if float(np.dot(selected, previous_tangent)) < 0.0:
                    selected = -selected
            selected_norm = float(np.linalg.norm(selected))
            if selected_norm > 1e-6:
                selected = selected / selected_norm
                previous_tangent = selected
            tangents[step] = selected
    else:
        velocity = _velocity_from_xy(coords, DEFAULT_DT)
        speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
        tangents = np.divide(
            velocity,
            np.maximum(speed, 1e-6),
            out=np.zeros_like(velocity),
            where=speed > 1e-6,
        ).astype(np.float32)
        tangents[0] = np.asarray([1.0, 0.0], dtype=np.float32)

    tangent_norm = np.linalg.norm(tangents, axis=-1, keepdims=True)
    tangents = np.divide(
        tangents,
        np.maximum(tangent_norm, 1e-6),
        out=np.zeros_like(tangents),
        where=tangent_norm > 1e-6,
    ).astype(np.float32)
    zero_rows = np.linalg.norm(tangents, axis=-1) <= 1e-6
    if np.any(zero_rows):
        tangents[zero_rows] = np.asarray([1.0, 0.0], dtype=np.float32)
    return tangents.astype(np.float32)


def _project_relative(rel_xy: np.ndarray, tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.asarray(tangent, dtype=np.float32).reshape(2)
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-6)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    rel_xy = np.asarray(rel_xy, dtype=np.float32)
    delta_s = np.sum(rel_xy * tangent[None, :], axis=-1).astype(np.float32)
    delta_l = np.sum(rel_xy * normal[None, :], axis=-1).astype(np.float32)
    return delta_s, delta_l


def _build_neighbor_future_context(cache_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbor_agents_past = np.asarray(_npz_array(cache_data, "neighbor_agents_past", np.zeros((0, 0, 11), dtype=np.float32)), dtype=np.float32)
    neighbor_agents_future = np.asarray(_npz_array(cache_data, "neighbor_agents_future", np.zeros((0, 0, 3), dtype=np.float32)), dtype=np.float32)
    neighbor_agents_past_mask = np.asarray(_npz_array(cache_data, "neighbor_agents_past_mask", np.zeros((0, 0), dtype=bool)), dtype=bool)
    neighbor_agents_future_mask = np.asarray(_npz_array(cache_data, "neighbor_agents_future_mask", np.zeros((0, 0), dtype=bool)), dtype=bool)

    if neighbor_agents_past.ndim != 3 or neighbor_agents_future.ndim != 3:
        return (
            np.zeros((0, 0, 2), dtype=np.float32),
            np.zeros((0, 0), dtype=bool),
            np.zeros((0,), dtype=np.float32),
        )

    current_neighbor_state = neighbor_agents_past[:, -1, :] if neighbor_agents_past.shape[1] > 0 else np.zeros((neighbor_agents_past.shape[0], 11), dtype=np.float32)
    current_xy = current_neighbor_state[:, :2].astype(np.float32)
    future_xy = neighbor_agents_future[:, :, :2].astype(np.float32)
    full_xy = np.concatenate([current_xy[:, None, :], future_xy], axis=1)

    if neighbor_agents_past_mask.ndim == 2 and neighbor_agents_past_mask.shape[1] > 0:
        current_valid = neighbor_agents_past_mask[:, -1].astype(bool)
    else:
        current_valid = np.linalg.norm(current_xy, axis=-1) > 1e-4

    nonzero_future = np.linalg.norm(future_xy, axis=-1) > 1e-4
    if neighbor_agents_future_mask.shape == future_xy.shape[:2]:
        raw_future_mask = neighbor_agents_future_mask.astype(bool)
        direct_overlap = int(np.logical_and(raw_future_mask, nonzero_future).sum())
        inverse_overlap = int(np.logical_and(~raw_future_mask, nonzero_future).sum())
        future_valid = raw_future_mask if direct_overlap >= inverse_overlap else ~raw_future_mask
        future_valid = np.logical_or(future_valid, nonzero_future)
    else:
        future_valid = nonzero_future

    full_valid = np.concatenate([current_valid[:, None], future_valid], axis=1)
    if current_neighbor_state.shape[1] > 7:
        neighbor_length = np.asarray(current_neighbor_state[:, 7], dtype=np.float32)
    else:
        neighbor_length = np.full((current_xy.shape[0],), 4.5, dtype=np.float32)

    smoothed_xy = full_xy.astype(np.float32).copy()
    for agent_index in range(smoothed_xy.shape[0]):
        valid_steps = np.where(full_valid[agent_index])[0]
        if valid_steps.size < 2:
            continue
        smoothed_xy[agent_index, valid_steps] = _smooth_xy(
            smoothed_xy[agent_index, valid_steps],
            DEFAULT_NEIGHBOR_SMOOTH_WINDOW,
        )
    return smoothed_xy.astype(np.float32), full_valid.astype(bool), neighbor_length.astype(np.float32)


def _collect_route_lane_polylines(cache_data) -> list[np.ndarray]:
    """Return individual route-lane centerlines without flattening lane identity.

    The old tangent helper intentionally flattens all segments for fast local
    projection.  Lane-change gap evaluation needs stronger semantics: a target
    lane must be a concrete polyline, not an assumed +/- 3.6 m offset.
    """

    def _collect(polyline_key: str, mask_key: str) -> list[np.ndarray]:
        polylines = np.asarray(
            _npz_array(cache_data, polyline_key, np.zeros((0, 0, 2), dtype=np.float32)),
            dtype=np.float32,
        )
        mask = np.asarray(
            _npz_array(cache_data, mask_key, np.zeros((0, 0), dtype=bool)),
            dtype=bool,
        )
        if polylines.ndim != 3 or polylines.shape[-1] < 2:
            return []

        lane_polylines: list[np.ndarray] = []
        for lane_index in range(polylines.shape[0]):
            lane_xy = polylines[lane_index, :, :2].astype(np.float32)
            if mask.shape[:2] == polylines.shape[:2]:
                lane_xy = lane_xy[mask[lane_index]]
            else:
                lane_xy = lane_xy[np.linalg.norm(lane_xy, axis=-1) > 1e-4]
            if lane_xy.shape[0] >= 2:
                lane_polylines.append(lane_xy.astype(np.float32))
        return lane_polylines

    route_lanes = _collect("route_lanes", "route_lanes_mask")
    return route_lanes if route_lanes else _collect("lanes", "lanes_mask")


def _nearest_lane_frame(
    xy: np.ndarray,
    lane_polylines: Sequence[np.ndarray],
) -> tuple[int | None, float, np.ndarray, np.ndarray]:
    """Return nearest lane, squared distance, midpoint and tangent for one point."""

    point = np.asarray(xy, dtype=np.float32).reshape(2)
    best_lane_index = None
    best_distance_sq = float("inf")
    best_midpoint = np.zeros((2,), dtype=np.float32)
    best_tangent = np.asarray([1.0, 0.0], dtype=np.float32)
    for lane_index, lane_xy in enumerate(lane_polylines):
        if lane_xy.ndim != 2 or lane_xy.shape[0] < 2:
            continue
        segments = lane_xy[1:] - lane_xy[:-1]
        lengths = np.linalg.norm(segments, axis=-1)
        valid = lengths > 1e-4
        if not np.any(valid):
            continue
        starts = lane_xy[:-1][valid]
        segment_values = segments[valid]
        segment_lengths_sq = np.maximum(np.sum(segment_values * segment_values, axis=-1), 1e-6)
        alpha = np.clip(
            np.sum((point[None, :] - starts) * segment_values, axis=-1) / segment_lengths_sq,
            0.0,
            1.0,
        )
        projected = starts + alpha[:, None] * segment_values
        distance_sq = np.sum((point[None, :] - projected) ** 2, axis=-1)
        local_index = int(np.argmin(distance_sq))
        local_distance_sq = float(distance_sq[local_index])
        if local_distance_sq < best_distance_sq:
            tangent = segment_values[local_index] / max(float(np.linalg.norm(segment_values[local_index])), 1e-6)
            best_lane_index = int(lane_index)
            best_distance_sq = local_distance_sq
            best_midpoint = projected[local_index].astype(np.float32)
            best_tangent = tangent.astype(np.float32)
    return best_lane_index, best_distance_sq, best_midpoint, best_tangent


def _lane_relative_series(
    xy: np.ndarray,
    lane_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project positions into a single source-lane frame with continuous tangent sign."""

    coords = np.asarray(xy, dtype=np.float32)
    signed_lateral = np.zeros((coords.shape[0],), dtype=np.float32)
    tangents = np.zeros((coords.shape[0], 2), dtype=np.float32)
    previous_tangent = None
    for step, point in enumerate(coords):
        _, _, midpoint, tangent = _nearest_lane_frame(point, [lane_xy])
        if previous_tangent is not None and float(np.dot(tangent, previous_tangent)) < 0.0:
            tangent = -tangent
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
        signed_lateral[step] = float(np.dot(point - midpoint, normal))
        tangents[step] = tangent
        previous_tangent = tangent
    return signed_lateral.astype(np.float32), tangents.astype(np.float32)


def _first_sustained_true(mask: np.ndarray, min_steps: int) -> int | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    required = max(int(min_steps), 1)
    if values.size < required:
        return None
    run_length = 0
    for index, value in enumerate(values.tolist()):
        run_length = run_length + 1 if value else 0
        if run_length >= required:
            return int(index - required + 1)
    return None


def _infer_target_lane_index(
    ego_full_xy: np.ndarray,
    lane_polylines: Sequence[np.ndarray],
    source_lane_index: int,
    onset_step: int,
) -> int | None:
    """Infer the reached lane from the later ego trajectory, allowing multi-lane shifts."""

    if not lane_polylines or ego_full_xy.shape[0] == 0:
        return None
    start = min(max(int(onset_step) + 2, 0), ego_full_xy.shape[0] - 1)
    # The terminal eight frames are enough to identify the reached lane while
    # keeping this per-sample map association practical for large cache splits.
    start = max(start, max(0, ego_full_xy.shape[0] - 8))
    counts: Dict[int, int] = {}
    distance_sums: Dict[int, float] = {}
    for point in ego_full_xy[start:]:
        lane_index, distance_sq, _, _ = _nearest_lane_frame(point, lane_polylines)
        if lane_index is None or lane_index == source_lane_index:
            continue
        if distance_sq > (0.85 * DEFAULT_LANE_WIDTH_M) ** 2:
            continue
        counts[lane_index] = counts.get(lane_index, 0) + 1
        distance_sums[lane_index] = distance_sums.get(lane_index, 0.0) + float(distance_sq)
    if not counts:
        return None
    return min(
        counts,
        key=lambda lane_index: (-counts[lane_index], distance_sums[lane_index] / counts[lane_index]),
    )


@dataclass(frozen=True)
class BehaviorMetricBundle:
    """Raw behavior metrics and auxiliary data for later normalization."""

    scene_bucket: str
    raw_metric_names: tuple[str, str, str]
    raw_metric_values: np.ndarray
    canonical_axis_names: tuple[str, str, str]
    axis_valid_mask: np.ndarray
    axis_invalid_reasons: tuple[str, str, str]
    metric_source: str
    metric_note: str
    metric_aux: Dict[str, object]

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "behavior_metric_names": list(self.raw_metric_names),
            "behavior_metric_values": [float(value) for value in self.raw_metric_values.tolist()],
            "canonical_axis_names": list(self.canonical_axis_names),
            "behavior_metric_valid_mask": [bool(value) for value in self.axis_valid_mask.tolist()],
            "behavior_metric_invalid_reasons": list(self.axis_invalid_reasons),
            "behavior_metric_source": self.metric_source,
            "behavior_metric_note": self.metric_note,
            "behavior_metric_aux": dict(self.metric_aux),
        }


def _free_drive_raw_values(
    *,
    ego_mean_speed_mps: float,
    positive_accel_p90_mps2: float,
    positive_jerk_p90_mps3: float,
    route_speed_limit_mps: float | None,
    reference_speed_mps: float | None,
) -> np.ndarray:
    local_reference_speed = route_speed_limit_mps
    if local_reference_speed is None or local_reference_speed <= 0.5:
        local_reference_speed = reference_speed_mps if reference_speed_mps is not None and reference_speed_mps > 0.5 else max(float(ego_mean_speed_mps), 1.0)

    speed_ratio = _clip(float(ego_mean_speed_mps) / max(float(local_reference_speed), 1e-3), 0.0, 1.3)
    return np.asarray(
        [
            speed_ratio,
            max(float(positive_accel_p90_mps2), 0.0),
            # Directional positive jerk captures how strongly speed is built up,
            # rather than treating braking-induced stop-and-go variation as an
            # aggressive speed response.
            max(float(positive_jerk_p90_mps3), 0.0),
        ],
        dtype=np.float32,
    )


def _free_drive_clear_mask(
    *,
    ego_full_xy: np.ndarray,
    tangents: np.ndarray,
    ego_speed_mps: np.ndarray,
    neighbor_full_xy: np.ndarray,
    neighbor_valid: np.ndarray,
    neighbor_length: np.ndarray,
) -> np.ndarray:
    """Identify frames that are not materially constrained by a same-lane lead.

    Scene labels are assigned from a different temporal view.  This local check
    prevents a future lead that is already within following distance from
    silently contaminating a free-driving speed or acceleration measurement.
    """

    clear = np.ones((ego_full_xy.shape[0],), dtype=bool)
    speeds = np.asarray(ego_speed_mps, dtype=np.float32).reshape(-1)
    time_steps = min(ego_full_xy.shape[0], neighbor_full_xy.shape[1])
    for step in range(time_steps):
        rel_xy = neighbor_full_xy[:, step, :2] - ego_full_xy[step, :2]
        delta_s, delta_l = _project_relative(rel_xy, tangents[step])
        valid = neighbor_valid[:, step] & (delta_s > 0.0) & (np.abs(delta_l) < DEFAULT_SAME_LANE_HALF_WIDTH_M)
        if not np.any(valid):
            continue
        indices = np.where(valid)[0]
        effective_gap = np.maximum(
            delta_s[indices] - 0.5 * (EGO_LENGTH_M + neighbor_length[indices]),
            0.0,
        )
        dynamic_clearance = max(
            DEFAULT_FREE_LEAD_CLEARANCE_M,
            DEFAULT_FREE_HEADWAY_CLEARANCE_S * abs(float(speeds[min(step, max(speeds.size - 1, 0))])),
        )
        if effective_gap.size > 0 and float(np.min(effective_gap)) < dynamic_clearance:
            clear[step] = False
    return clear


def _traffic_control_influence_mask(
    *,
    route_lanes: np.ndarray,
    route_mask: np.ndarray,
    ego_full_xy: np.ndarray,
    tangents: np.ndarray,
    ego_speed_mps: np.ndarray,
) -> np.ndarray:
    """Mark frames approaching a route-relevant yellow/red traffic control.

    A free-flow speed deficit before a stop line is an environment constraint,
    not evidence that a driver is unwilling to accelerate.  The cache carries
    traffic-light state on route-lane points.  We only reject controls ahead of
    the ego and near its current route tangent, rather than rejecting an entire
    sample because a different branch contains a signal.
    """

    route_lanes = np.asarray(route_lanes, dtype=np.float32)
    route_mask = np.asarray(route_mask, dtype=bool)
    controlled = np.zeros((ego_full_xy.shape[0],), dtype=bool)
    if route_lanes.ndim != 3 or route_lanes.shape[-1] < 12:
        return controlled
    if route_mask.shape[:2] != route_lanes.shape[:2]:
        route_mask = np.linalg.norm(route_lanes[..., :2], axis=-1) > 1e-4
    traffic_state = route_lanes[..., 8:12]
    yellow_or_red = (traffic_state[..., 1] > 0.5) | (traffic_state[..., 2] > 0.5)
    control_points = route_lanes[..., :2][route_mask & yellow_or_red]
    if control_points.size == 0:
        return controlled

    speeds = np.asarray(ego_speed_mps, dtype=np.float32).reshape(-1)
    for step in range(ego_full_xy.shape[0]):
        tangent = tangents[min(step, tangents.shape[0] - 1)]
        rel_xy = control_points - ego_full_xy[step]
        delta_s, delta_l = _project_relative(rel_xy, tangent)
        lookahead = max(
            DEFAULT_CONTROL_MIN_LOOKAHEAD_M,
            DEFAULT_FREE_HEADWAY_CLEARANCE_S * abs(float(speeds[min(step, max(speeds.size - 1, 0))])),
        )
        relevant = (
            (delta_s >= -DEFAULT_CONTROL_BEHIND_TOLERANCE_M)
            & (delta_s <= lookahead)
            & (np.abs(delta_l) <= DEFAULT_CONTROL_LATERAL_TOLERANCE_M)
        )
        controlled[step] = bool(np.any(relevant))
    return controlled


def _free_drive_metrics_from_cache(
    cache_path: str,
    record: Mapping[str, object],
    reference_speed_mps: float | None,
) -> BehaviorMetricBundle:
    with np.load(cache_path, allow_pickle=False) as cache_data:
        ego_future = np.asarray(cache_data["ego_agent_future"], dtype=np.float32)
        neighbor_full_xy, neighbor_valid, neighbor_length = _build_neighbor_future_context(cache_data)
        route_midpoints, route_tangents = _build_route_segment_reference(cache_data)
        route_lanes = np.asarray(
            _npz_array(cache_data, "route_lanes", np.zeros((0, 0, 0), dtype=np.float32)),
            dtype=np.float32,
        )
        route_lanes_mask = np.asarray(
            _npz_array(cache_data, "route_lanes_mask", np.zeros((0, 0), dtype=bool)),
            dtype=bool,
        )
        route_lanes_speed_limit = np.asarray(
            _npz_array(cache_data, "route_lanes_speed_limit", np.zeros((0, 1), dtype=np.float32)),
            dtype=np.float32,
        )
        route_lanes_has_speed_limit = np.asarray(
            _npz_array(cache_data, "route_lanes_has_speed_limit", np.zeros((0, 1), dtype=bool)),
            dtype=bool,
        )
        lanes = np.asarray(
            _npz_array(cache_data, "lanes", np.zeros((0, 0, 0), dtype=np.float32)),
            dtype=np.float32,
        )
        lanes_mask = np.asarray(
            _npz_array(cache_data, "lanes_mask", np.zeros((0, 0), dtype=bool)),
            dtype=bool,
        )
        lanes_speed_limit = np.asarray(
            _npz_array(cache_data, "lanes_speed_limit", np.zeros((0, 1), dtype=np.float32)),
            dtype=np.float32,
        )
        lanes_has_speed_limit = np.asarray(
            _npz_array(cache_data, "lanes_has_speed_limit", np.zeros((0, 1), dtype=bool)),
            dtype=bool,
        )

    ego_future_xy = ego_future[:, :2].astype(np.float32) if ego_future.ndim == 2 else np.zeros((0, 2), dtype=np.float32)
    ego_full_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), ego_future_xy], axis=0)
    ego_full_xy = _smooth_xy(ego_full_xy, DEFAULT_TRAJ_SMOOTH_WINDOW)
    speed = _speed_from_xy(ego_full_xy, DEFAULT_DT)
    speed = _moving_average_1d(speed, DEFAULT_TRAJ_SMOOTH_WINDOW)
    tangents = _estimate_tangents_for_xy(ego_full_xy, route_midpoints, route_tangents)
    local_speed_limit_series, local_speed_limit_valid = _local_speed_limit_series(
        ego_xy=ego_full_xy,
        route_lanes=route_lanes,
        route_mask=route_lanes_mask,
        route_speed_limits=route_lanes_speed_limit,
        route_has_speed_limits=route_lanes_has_speed_limit,
        lanes=lanes,
        lanes_mask=lanes_mask,
        lane_speed_limits=lanes_speed_limit,
        lane_has_speed_limits=lanes_has_speed_limit,
    )
    free_clear_mask = _free_drive_clear_mask(
        ego_full_xy=ego_full_xy,
        tangents=tangents,
        ego_speed_mps=speed,
        neighbor_full_xy=neighbor_full_xy,
        neighbor_valid=neighbor_valid,
        neighbor_length=neighbor_length,
    )
    traffic_control_mask = _traffic_control_influence_mask(
        route_lanes=route_lanes,
        route_mask=route_lanes_mask,
        ego_full_xy=ego_full_xy,
        tangents=tangents,
        ego_speed_mps=speed,
    )
    accel_diff = _finite_difference_1d(speed, DEFAULT_DT)
    accel = np.zeros_like(speed, dtype=np.float32)
    if accel_diff.size > 0:
        accel[1:] = accel_diff
    positive_accel_p90 = _safe_percentile(np.maximum(accel, 0.0), 90.0, default=0.0)
    jerk = _finite_difference_1d(accel, DEFAULT_DT)
    positive_jerk_p90 = _safe_percentile(np.maximum(jerk, 0.0), 90.0, default=0.0)

    ego_mean_speed_mps = float(np.mean(speed)) if speed.size > 0 else max(float(record.get("ego_mean_speed", 0.0)), 0.0)
    speed_variation_mps = max(
        _safe_percentile(speed, 90.0, default=0.0) - _safe_percentile(speed, 10.0, default=0.0),
        0.0,
    )
    route_speed_limit_available = bool(np.any(local_speed_limit_valid))
    route_speed_limit_mps = (
        float(np.median(local_speed_limit_series[local_speed_limit_valid]))
        if route_speed_limit_available
        else None
    )
    usable_free_mask = free_clear_mask & ~traffic_control_mask & np.isfinite(speed) & local_speed_limit_valid
    free_step_count = int(np.sum(usable_free_mask))
    lead_clear_step_count = int(np.sum(free_clear_mask))
    lead_constrained_step_count = int(np.sum(~free_clear_mask))
    traffic_control_step_count = int(np.sum(traffic_control_mask))
    speed_ratio_values = (
        np.clip(speed[usable_free_mask] / np.maximum(local_speed_limit_series[usable_free_mask], 1e-3), 0.0, 1.3)
        if route_speed_limit_available
        else np.zeros((0,), dtype=np.float32)
    )
    speed_utilization = _trimmed_mean(speed_ratio_values, trim_fraction=0.10, default=0.0)

    speed_headroom = (
        np.where(local_speed_limit_valid, local_speed_limit_series - speed, 0.0)
        if route_speed_limit_available
        else np.zeros_like(speed)
    )
    opportunity_mask = usable_free_mask & (speed_headroom > DEFAULT_SPEED_OPPORTUNITY_MPS)
    opportunity_step_count = int(np.sum(opportunity_mask))
    accel_willingness = _upper_tail_mean(
        np.maximum(accel[opportunity_mask], 0.0),
        fraction=DEFAULT_UPPER_TAIL_FRACTION,
        default=0.0,
    )

    response_horizon_steps = max(int(round(DEFAULT_RESPONSE_HORIZON_S / DEFAULT_DT)), 1)
    opportunity_start_mask = opportunity_mask.copy()
    if opportunity_start_mask.size > 1:
        opportunity_start_mask[1:] &= ~opportunity_mask[:-1]
    response_values = []
    for start_step in np.where(opportunity_start_mask)[0].tolist():
        end_step = min(start_step + response_horizon_steps, speed.shape[0] - 1)
        if end_step - start_step < response_horizon_steps:
            continue
        response_window_mask = usable_free_mask[start_step : end_step + 1]
        # A masked-out red-light or close-lead interval must not be skipped and
        # bridged when measuring a "two-second response" event.
        if not bool(np.all(response_window_mask)):
            continue
        window_speed = speed[start_step : end_step + 1]
        recovery = max(float(np.max(window_speed)) - float(speed[start_step]), 0.0)
        available = max(float(speed_headroom[start_step]), DEFAULT_SPEED_OPPORTUNITY_MPS)
        response_values.append(float(np.clip(recovery / available, 0.0, 1.3)))
    speed_response = _upper_tail_mean(
        response_values,
        fraction=DEFAULT_UPPER_TAIL_FRACTION,
        default=0.0,
    )

    axis_valid_mask = np.asarray(
        [
            bool(route_speed_limit_available and free_step_count >= DEFAULT_FREE_MIN_VALID_STEPS),
            bool(route_speed_limit_available and opportunity_step_count >= DEFAULT_FREE_MIN_VALID_STEPS),
            bool(route_speed_limit_available and len(response_values) > 0),
        ],
        dtype=bool,
    )
    if not route_speed_limit_available:
        free_frame_invalid_reason = "route_speed_limit_unavailable"
    elif free_step_count < DEFAULT_FREE_MIN_VALID_STEPS and traffic_control_step_count > 0:
        free_frame_invalid_reason = "insufficient_free_frames_after_traffic_control_exclusion"
    else:
        free_frame_invalid_reason = "insufficient_unconstrained_free_frames"
    axis_invalid_reasons = (
        "" if axis_valid_mask[0] else free_frame_invalid_reason,
        "" if axis_valid_mask[1] else (
            "route_speed_limit_unavailable" if not route_speed_limit_available else (
                "insufficient_speed_opportunity_after_traffic_control_exclusion"
                if traffic_control_step_count > 0
                else "insufficient_speed_opportunity_frames"
            )
        ),
        "" if axis_valid_mask[2] else (
            "route_speed_limit_unavailable" if not route_speed_limit_available else (
                "no_speed_recovery_window_after_traffic_control_exclusion"
                if traffic_control_step_count > 0
                else "no_speed_recovery_opportunity_window"
            )
        ),
    )
    raw_values = np.asarray(
        [speed_utilization, accel_willingness, speed_response],
        dtype=np.float32,
    )
    motion_stats = _trajectory_motion_stats(ego_full_xy, tangents)
    return BehaviorMetricBundle(
        scene_bucket="straight_free_drive",
        raw_metric_names=RAW_BEHAVIOR_METRIC_BY_SCENE["straight_free_drive"],
        raw_metric_values=raw_values,
        canonical_axis_names=CANONICAL_AXIS_BY_SCENE["straight_free_drive"],
        axis_valid_mask=axis_valid_mask,
        axis_invalid_reasons=axis_invalid_reasons,
        metric_source="cache",
        metric_note=cache_path,
        metric_aux={
            "ego_mean_speed_mps": float(ego_mean_speed_mps),
            "route_speed_limit_mps": float(route_speed_limit_mps) if route_speed_limit_mps is not None else -1.0,
            "positive_accel_p90_mps2": float(positive_accel_p90),
            "positive_jerk_p90_mps3": float(positive_jerk_p90),
            "speed_variation_mps": float(speed_variation_mps),
            "free_metric_definition": "v5_opportunity_conditioned_local_speed_limit",
            "route_speed_limit_available": bool(route_speed_limit_available),
            "local_speed_limit_valid_step_count": int(np.sum(local_speed_limit_valid)),
            "free_lead_clearance_m": float(DEFAULT_FREE_LEAD_CLEARANCE_M),
            "free_lead_headway_clearance_s": float(DEFAULT_FREE_HEADWAY_CLEARANCE_S),
            "free_lead_constrained_step_count": int(lead_constrained_step_count),
            "traffic_control_excluded_step_count": int(traffic_control_step_count),
            "traffic_control_min_lookahead_m": float(DEFAULT_CONTROL_MIN_LOOKAHEAD_M),
            "free_lead_clear_step_count": int(lead_clear_step_count),
            "free_eligible_step_count": int(free_step_count),
            "speed_opportunity_step_count": int(opportunity_step_count),
            "speed_recovery_event_count": int(len(response_values)),
            "speed_recovery_horizon_s": float(DEFAULT_RESPONSE_HORIZON_S),
            "speed_opportunity_min_headroom_mps": float(DEFAULT_SPEED_OPPORTUNITY_MPS),
            **motion_stats,
        },
    )


def _car_follow_metrics_from_cache(
    cache_path: str,
    record: Mapping[str, object],
) -> BehaviorMetricBundle:
    with np.load(cache_path, allow_pickle=False) as cache_data:
        ego_future = np.asarray(cache_data["ego_agent_future"], dtype=np.float32)
        neighbor_full_xy, neighbor_valid, neighbor_length = _build_neighbor_future_context(cache_data)
        route_midpoints, route_tangents = _build_route_segment_reference(cache_data)

    ego_future_xy = ego_future[:, :2].astype(np.float32) if ego_future.ndim == 2 else np.zeros((0, 2), dtype=np.float32)
    ego_full_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), ego_future_xy], axis=0)
    ego_full_xy = _smooth_xy(ego_full_xy, DEFAULT_TRAJ_SMOOTH_WINDOW)
    tangents = _estimate_tangents_for_xy(ego_full_xy, route_midpoints, route_tangents)
    ego_velocity = _velocity_from_xy(ego_full_xy, DEFAULT_DT)
    ego_v_long = np.sum(ego_velocity * tangents, axis=-1).astype(np.float32)
    neighbor_velocity = np.zeros_like(neighbor_full_xy, dtype=np.float32)
    for agent_index in range(neighbor_full_xy.shape[0]):
        neighbor_velocity[agent_index] = _velocity_from_xy(neighbor_full_xy[agent_index], DEFAULT_DT)

    thw_values = []
    ttc_values = []
    first_brake_trigger_ttc = None
    first_response_type = None
    first_response_candidate_seen = False
    closing_event_count = 0
    response_start_ttc_unavailable_count = 0
    previous_lead_idx = None
    persistent_lead_steps = 0
    ego_accel_long = np.zeros_like(ego_v_long, dtype=np.float32)
    ego_accel_diff = _finite_difference_1d(ego_v_long, DEFAULT_DT)
    if ego_accel_diff.size > 0:
        # Acceleration at t uses the backward difference v_t - v_(t-1), so a
        # brake threshold is not shifted one future frame later.
        ego_accel_long[1:] = ego_accel_diff
    time_steps = min(ego_full_xy.shape[0], neighbor_full_xy.shape[1])
    closing_speed_by_step = np.full((time_steps,), np.nan, dtype=np.float32)
    ttc_by_step = np.full((time_steps,), np.nan, dtype=np.float32)
    closing_lead_idx_by_step = np.full((time_steps,), -1, dtype=np.int64)
    for step in range(time_steps):
        rel_xy = neighbor_full_xy[:, step, :2] - ego_full_xy[step, :2]
        delta_s, delta_l = _project_relative(rel_xy, tangents[step])
        valid = neighbor_valid[:, step] & (delta_s > 0.0) & (np.abs(delta_l) < DEFAULT_SAME_LANE_HALF_WIDTH_M)
        if not np.any(valid):
            previous_lead_idx = None
            persistent_lead_steps = 0
            continue

        valid_indices = np.where(valid)[0]
        effective_gap = np.maximum(
            delta_s[valid_indices] - 0.5 * (EGO_LENGTH_M + neighbor_length[valid_indices]),
            0.0,
        )
        if effective_gap.size == 0:
            previous_lead_idx = None
            persistent_lead_steps = 0
            continue

        local_lead = int(np.argmin(effective_gap))
        lead_idx = int(valid_indices[local_lead])
        lead_gap = float(effective_gap[local_lead])
        ego_speed_long = max(float(abs(ego_v_long[min(step, ego_v_long.shape[0] - 1)])), DEFAULT_TIME_GAP_VMIN_MPS)
        thw_values.append(lead_gap / ego_speed_long)

        # Keep the instantaneous closing TTC for the selected lead before the
        # persistence filter.  A sustained ego response becomes observable only
        # after several frames, whereas its physical start is at the beginning
        # of that window.  Recording only after persistence would force the
        # response-start lookup below to fall back to a TTC measured later.
        lead_speed_long = float(np.dot(neighbor_velocity[lead_idx, step], tangents[step]))
        closing_speed = max(float(ego_v_long[step]) - lead_speed_long, 0.0)
        if closing_speed >= DEFAULT_CLOSING_SPEED_MIN_MPS:
            current_ttc = min(lead_gap / max(closing_speed, 1e-3), DEFAULT_TTC_MAX_S)
            closing_speed_by_step[step] = float(closing_speed)
            ttc_by_step[step] = float(current_ttc)
            closing_lead_idx_by_step[step] = int(lead_idx)

        if previous_lead_idx == lead_idx:
            persistent_lead_steps += 1
        else:
            persistent_lead_steps = 1
        same_persistent_lead = persistent_lead_steps >= DEFAULT_FOLLOW_PERSISTENCE_STEPS
        if same_persistent_lead:
            if np.isfinite(ttc_by_step[step]):
                closing_event_count += 1
                current_ttc = float(ttc_by_step[step])
                ttc_values.append(current_ttc)
                # This is a reaction-threshold metric, not relative speed.  A
                # smaller TTC at the first sustained ego response means the
                # driver tolerated closing for longer and is more aggressive.
                brake_start = max(step - DEFAULT_BRAKE_SUSTAINED_STEPS + 1, 0)
                brake_window = ego_accel_long[brake_start : step + 1]
                ego_braking = bool(
                    brake_window.size >= DEFAULT_BRAKE_SUSTAINED_STEPS
                    and np.all(brake_window <= -DEFAULT_BRAKE_TRIGGER_MPS2)
                )
                ego_mild_decel = bool(
                    brake_window.size >= DEFAULT_BRAKE_SUSTAINED_STEPS
                    and np.all(brake_window <= -DEFAULT_FOLLOW_RESPONSE_DECEL_MPS2)
                )
                closing_window = closing_speed_by_step[brake_start : step + 1]
                same_lead_window = closing_lead_idx_by_step[brake_start : step + 1]
                closing_relief = bool(
                    closing_window.size >= DEFAULT_BRAKE_SUSTAINED_STEPS
                    and np.all(np.isfinite(closing_window))
                    and np.all(same_lead_window == lead_idx)
                    and float(closing_window[0] - closing_window[-1]) >= DEFAULT_FOLLOW_RESPONSE_CLOSING_RELIEF_MPS
                )
                response_type = (
                    "sustained_brake"
                    if ego_braking
                    else "sustained_decel_with_closing_relief"
                    if ego_mild_decel and closing_relief
                    else None
                )
                if response_type is not None and not first_response_candidate_seen:
                    first_response_candidate_seen = True
                    # Store TTC at the *start* of the sustained response, not
                    # 0.2 s later when the criterion becomes observable.
                    response_ttc = float(ttc_by_step[brake_start])
                    same_response_lead = bool(closing_lead_idx_by_step[brake_start] == lead_idx)
                    if math.isfinite(response_ttc) and same_response_lead:
                        first_brake_trigger_ttc = float(response_ttc)
                        first_response_type = response_type
                    else:
                        # Do not substitute the TTC at the end of the response
                        # window: that changes the metric's meaning.  Keep the
                        # axis unavailable when its response-start TTC was not
                        # physically observed for the same lead.
                        response_start_ttc_unavailable_count += 1

        previous_lead_idx = lead_idx

    fallback_headway = _optional_float(record.get("following_min_thw", None)) or 2.5
    headway_time = _safe_percentile(thw_values, 10.0, default=fallback_headway)
    ttc_margin = _safe_percentile(ttc_values, 10.0, default=DEFAULT_TTC_MAX_S)
    closing_tolerance_ttc = float(first_brake_trigger_ttc if first_brake_trigger_ttc is not None else DEFAULT_TTC_MAX_S)
    axis_valid_mask = np.asarray(
        [bool(thw_values), bool(ttc_values), first_brake_trigger_ttc is not None],
        dtype=bool,
    )
    axis_invalid_reasons = (
        "" if axis_valid_mask[0] else "lead_vehicle_not_observed",
        "" if axis_valid_mask[1] else "no_persistent_closing_event",
        "" if axis_valid_mask[2] else "no_persistent_closing_sustained_ego_response",
    )
    motion_stats = _trajectory_motion_stats(ego_full_xy, tangents)

    return BehaviorMetricBundle(
        scene_bucket="straight_car_follow",
        raw_metric_names=RAW_BEHAVIOR_METRIC_BY_SCENE["straight_car_follow"],
        raw_metric_values=np.asarray(
            [
                float(headway_time),
                float(ttc_margin),
                float(closing_tolerance_ttc),
            ],
            dtype=np.float32,
        ),
        canonical_axis_names=CANONICAL_AXIS_BY_SCENE["straight_car_follow"],
        axis_valid_mask=axis_valid_mask,
        axis_invalid_reasons=axis_invalid_reasons,
        metric_source="cache",
        metric_note=cache_path,
        metric_aux={
            "ego_mean_speed_mps": float(np.mean(np.abs(ego_v_long))) if ego_v_long.size > 0 else 0.0,
            "persistent_closing_event_count": int(closing_event_count),
            "brake_trigger_event_count": int(first_response_type == "sustained_brake"),
            "ego_response_event_count": int(first_brake_trigger_ttc is not None),
            "ego_response_type": str(first_response_type or ""),
            "response_start_ttc_unavailable_count": int(response_start_ttc_unavailable_count),
            "follow_lead_persistence_steps": int(DEFAULT_FOLLOW_PERSISTENCE_STEPS),
            "follow_brake_sustained_steps": int(DEFAULT_BRAKE_SUSTAINED_STEPS),
            "follow_response_decel_threshold_mps2": float(DEFAULT_FOLLOW_RESPONSE_DECEL_MPS2),
            "follow_response_closing_relief_mps": float(DEFAULT_FOLLOW_RESPONSE_CLOSING_RELIEF_MPS),
            "lead_present_step_count": int(len(thw_values)),
            "ttc_cap_s": float(DEFAULT_TTC_MAX_S),
            **motion_stats,
        },
    )


def _lane_change_metrics_from_cache(
    cache_path: str,
    record: Mapping[str, object],
) -> BehaviorMetricBundle:
    with np.load(cache_path, allow_pickle=False) as cache_data:
        ego_future = np.asarray(cache_data["ego_agent_future"], dtype=np.float32)
        neighbor_full_xy, neighbor_valid, neighbor_length = _build_neighbor_future_context(cache_data)
        route_midpoints, route_tangents = _build_route_segment_reference(cache_data)
        lane_polylines = _collect_route_lane_polylines(cache_data)

    ego_future_xy = ego_future[:, :2].astype(np.float32) if ego_future.ndim == 2 else np.zeros((0, 2), dtype=np.float32)
    ego_full_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), ego_future_xy], axis=0)
    ego_full_xy = _smooth_xy(ego_full_xy, DEFAULT_TRAJ_SMOOTH_WINDOW)
    source_lane_index, _, _, _ = _nearest_lane_frame(
        ego_full_xy[0] if ego_full_xy.shape[0] > 0 else np.zeros((2,), dtype=np.float32),
        lane_polylines,
    )
    if source_lane_index is not None:
        signed_lateral_disp, tangents = _lane_relative_series(
            ego_full_xy,
            lane_polylines[source_lane_index],
        )
    else:
        # Keep a map-based fallback for incomplete route geometry.  The samples
        # remain exportable, but target-lane gap will be marked unavailable.
        tangents = _estimate_tangents_for_xy(ego_full_xy, route_midpoints, route_tangents)
        initial_tangent = tangents[0] if tangents.shape[0] > 0 else np.asarray([1.0, 0.0], dtype=np.float32)
        initial_normal = np.asarray([-initial_tangent[1], initial_tangent[0]], dtype=np.float32)
        signed_lateral_disp = np.sum(
            (ego_full_xy - ego_full_xy[:1]) * initial_normal[None, :],
            axis=-1,
        ).astype(np.float32)

    ego_velocity = _velocity_from_xy(ego_full_xy, DEFAULT_DT)
    ego_v_long = np.sum(ego_velocity * tangents, axis=-1).astype(np.float32)
    lateral_speed_signed = np.zeros_like(signed_lateral_disp, dtype=np.float32)
    if signed_lateral_disp.size > 1:
        lateral_speed_signed[1:] = _finite_difference_1d(signed_lateral_disp, DEFAULT_DT).astype(np.float32)

    # The final reached side supplies the label-time target direction.  This is
    # not an input feature; it only prevents an opposite-direction correction
    # from inflating the execution axis.
    terminal_disp = signed_lateral_disp[max(signed_lateral_disp.size - 8, 0) :]
    target_direction = 0.0
    if source_lane_index is not None and terminal_disp.size > 0:
        terminal_median = float(np.median(terminal_disp))
        if abs(terminal_median) >= DEFAULT_ONSET_THRESHOLD_M:
            target_direction = float(np.sign(terminal_median))
    target_lateral_disp = target_direction * signed_lateral_disp
    target_lateral_speed = target_direction * lateral_speed_signed

    # Requiring material target-direction displacement *and* velocity avoids a
    # lane-centre offset or one-frame map association fluctuation becoming an
    # onset.  Source-lane association is required for all lane-change axes.
    onset_mask = (
        (source_lane_index is not None)
        & (target_direction != 0.0)
        & (target_lateral_disp >= DEFAULT_ONSET_THRESHOLD_M)
        & (target_lateral_speed >= DEFAULT_ONSET_LATERAL_SPEED_MPS)
    )
    if onset_mask.size > 0:
        onset_mask[0] = False
    onset_step = _first_sustained_true(onset_mask, DEFAULT_MIN_ONSET_SUSTAINED_STEPS)
    onset_time_s = float(
        (onset_step if onset_step is not None else max(ego_full_xy.shape[0] - 1, 0)) * DEFAULT_DT
    )
    observed_horizon_s = max(float(max(ego_full_xy.shape[0] - 1, 1)) * DEFAULT_DT, DEFAULT_DT)
    initiation_progress = float(np.clip(onset_time_s / observed_horizon_s, 0.0, 1.0))

    active_target_lateral_speed = (
        np.maximum(target_lateral_speed[max(int(onset_step) - 1, 0) :], 0.0)
        if onset_step is not None
        else np.zeros((0,), dtype=np.float32)
    )
    lateral_commit_raw = _safe_percentile(
        active_target_lateral_speed / max(DEFAULT_LANE_WIDTH_M, 1e-3),
        90.0,
        default=0.0,
    )
    target_lane_index = (
        _infer_target_lane_index(
            ego_full_xy=ego_full_xy,
            lane_polylines=lane_polylines,
            source_lane_index=int(source_lane_index),
            onset_step=int(onset_step),
        )
        if source_lane_index is not None and onset_step is not None
        else None
    )

    neighbor_velocity = np.zeros_like(neighbor_full_xy, dtype=np.float32)
    for agent_index in range(neighbor_full_xy.shape[0]):
        neighbor_velocity[agent_index] = _velocity_from_xy(neighbor_full_xy[agent_index], DEFAULT_DT)

    merge_gap_time_s = DEFAULT_TTC_MAX_S
    front_gap_time_s = None
    rear_ttc_s = None
    target_lane_agent_count = 0
    target_lane_tangent_alignment = None
    if target_lane_index is not None and onset_step is not None and neighbor_full_xy.shape[1] > 0:
        step = min(int(onset_step), neighbor_full_xy.shape[1] - 1)
        rel_xy = neighbor_full_xy[:, step, :2] - ego_full_xy[step, :2]
        valid = neighbor_valid[:, step]
        target_lane = lane_polylines[int(target_lane_index)]
        # m_gap is a target-lane quantity.  Source- and target-lane tangents
        # are almost parallel in the straight bucket, but using the target
        # tangent keeps front/rear assignment and relative speed semantically
        # correct near curvature, forks, and lane connectors.
        _, _, _, target_tangent = _nearest_lane_frame(ego_full_xy[step, :2], [target_lane])
        source_tangent = tangents[min(step, tangents.shape[0] - 1)]
        tangent_alignment = float(np.dot(target_tangent, source_tangent))
        if tangent_alignment < 0.0:
            target_tangent = -target_tangent
            tangent_alignment = -tangent_alignment
        target_lane_tangent_alignment = tangent_alignment
        delta_s, _ = _project_relative(rel_xy, target_tangent)
        ego_target_v_long = float(np.dot(ego_velocity[min(step, ego_velocity.shape[0] - 1)], target_tangent))
        target_lane_membership = np.zeros_like(valid, dtype=bool)
        for agent_index in np.where(valid)[0].tolist():
            _, distance_sq, _, _ = _nearest_lane_frame(
                neighbor_full_xy[agent_index, step, :2],
                [target_lane],
            )
            target_lane_membership[agent_index] = bool(
                distance_sq <= (0.55 * DEFAULT_LANE_WIDTH_M) ** 2
            )
        target_lane_mask = valid & target_lane_membership
        target_lane_agent_count = int(np.sum(target_lane_mask))

        if np.any(target_lane_mask):
            candidate_indices = np.where(target_lane_mask)[0]
            effective_long_gap = delta_s[candidate_indices]
            front_mask = effective_long_gap > 0.0
            rear_mask = effective_long_gap < 0.0

            if np.any(front_mask):
                front_indices = candidate_indices[front_mask]
                front_gap = np.maximum(
                    delta_s[front_indices] - 0.5 * (EGO_LENGTH_M + neighbor_length[front_indices]),
                    0.0,
                )
                if front_gap.size > 0:
                    front_idx = int(front_indices[np.argmin(front_gap)])
                    ego_speed_long = max(float(abs(ego_target_v_long)), DEFAULT_TIME_GAP_VMIN_MPS)
                    front_gap_time_s = float(np.min(front_gap) / ego_speed_long)

            if np.any(rear_mask):
                rear_indices = candidate_indices[rear_mask]
                rear_gap = np.maximum(
                    -delta_s[rear_indices] - 0.5 * (EGO_LENGTH_M + neighbor_length[rear_indices]),
                    0.0,
                )
                if rear_gap.size > 0:
                    rear_local = int(np.argmin(rear_gap))
                    rear_idx = int(rear_indices[rear_local])
                    rear_speed_long = float(np.dot(neighbor_velocity[rear_idx, step], target_tangent))
                    rear_closing_speed = max(rear_speed_long - ego_target_v_long, 0.0)
                    if rear_closing_speed >= DEFAULT_CLOSING_SPEED_MIN_MPS:
                        rear_ttc_s = float(min(rear_gap[rear_local] / rear_closing_speed, DEFAULT_TTC_MAX_S))

            valid_time_gaps = [value for value in (front_gap_time_s, rear_ttc_s) if value is not None and np.isfinite(value)]
            if valid_time_gaps:
                merge_gap_time_s = float(min(valid_time_gaps))

    axis_valid_mask = np.asarray(
        [
            source_lane_index is not None and target_direction != 0.0 and onset_step is not None,
            source_lane_index is not None and target_direction != 0.0 and onset_step is not None and active_target_lateral_speed.size > 0,
            bool(front_gap_time_s is not None or rear_ttc_s is not None),
        ],
        dtype=bool,
    )
    axis_invalid_reasons = (
        "" if axis_valid_mask[0] else (
            "source_lane_not_identified" if source_lane_index is None else "sustained_target_direction_lane_change_onset_not_detected"
        ),
        "" if axis_valid_mask[1] else "target_direction_lateral_execution_window_unavailable",
        "" if axis_valid_mask[2] else (
            "target_lane_not_identified" if target_lane_index is None else "target_lane_gap_unobserved"
        ),
    )
    motion_stats = _trajectory_motion_stats(ego_full_xy, tangents)

    return BehaviorMetricBundle(
        scene_bucket="straight_lane_change",
        raw_metric_names=RAW_BEHAVIOR_METRIC_BY_SCENE["straight_lane_change"],
        raw_metric_values=np.asarray(
            [
                float(initiation_progress),
                float(lateral_commit_raw),
                float(merge_gap_time_s),
            ],
            dtype=np.float32,
        ),
        canonical_axis_names=CANONICAL_AXIS_BY_SCENE["straight_lane_change"],
        axis_valid_mask=axis_valid_mask,
        axis_invalid_reasons=axis_invalid_reasons,
        metric_source="cache",
        metric_note=cache_path,
        metric_aux={
            "ego_mean_speed_mps": float(np.mean(np.abs(ego_v_long))) if ego_v_long.size > 0 else 0.0,
            "front_gap_time_s": float(front_gap_time_s) if front_gap_time_s is not None else -1.0,
            "rear_ttc_s": float(rear_ttc_s) if rear_ttc_s is not None else -1.0,
            "lane_onset_definition": "sustained(abs lateral displacement AND lateral speed)",
            "lane_gap_definition": "min(front_time_headway, rear_closing_ttc)",
            "lane_gap_projection_frame": "target_lane_tangent_at_onset",
            "target_lane_tangent_alignment_to_source": (
                float(target_lane_tangent_alignment) if target_lane_tangent_alignment is not None else -1.0
            ),
            "initiation_definition": "onset_time / observed_trajectory_horizon",
            "observed_horizon_s": float(observed_horizon_s),
            "target_direction": float(target_direction),
            "lateral_commitment_definition": "q90(target_direction_lateral_speed / nominal lane width)",
            "source_lane_index": int(source_lane_index) if source_lane_index is not None else -1,
            "target_lane_index": int(target_lane_index) if target_lane_index is not None else -1,
            "target_lane_agent_count": int(target_lane_agent_count),
            **motion_stats,
        },
    )


def _record_proxy_metric_bundle(
    scene_bucket: str,
    record: Mapping[str, object],
    reference_speed_mps: float | None,
) -> BehaviorMetricBundle:
    if scene_bucket == "straight_free_drive":
        ego_mean_speed_mps = _optional_float(record.get("ego_mean_speed", None)) or 0.0
        route_speed_limit_mps = _optional_float(record.get("route_speed_limit_mps", None))
        positive_accel_p90 = max(float(record.get("ego_accel_peak", 0.0)), 0.0)
        positive_jerk_p90 = max(float(record.get("ego_jerk_p90", 0.0)), 0.0)
        route_speed_limit_available = route_speed_limit_mps is not None and route_speed_limit_mps > 0.5
        raw_values = np.asarray(
            [
                _clip(ego_mean_speed_mps / max(float(route_speed_limit_mps or 0.0), 1e-3), 0.0, 1.3)
                if route_speed_limit_available
                else 0.0,
                positive_accel_p90,
                positive_jerk_p90,
            ],
            dtype=np.float32,
        )
        aux = {
            "ego_mean_speed_mps": float(ego_mean_speed_mps),
            "route_speed_limit_mps": float(route_speed_limit_mps) if route_speed_limit_mps is not None else -1.0,
            "positive_accel_p90_mps2": float(positive_accel_p90),
            "positive_jerk_p90_mps3": float(positive_jerk_p90),
            "free_metric_definition": "v3_proxy_speed_only",
            "route_speed_limit_available": bool(route_speed_limit_available),
        }
        axis_valid_mask = np.asarray([route_speed_limit_available, False, False], dtype=bool)
        axis_invalid_reasons = (
            "" if axis_valid_mask[0] else "route_speed_limit_unavailable_in_proxy",
            "requires_cache_speed_opportunity",
            "requires_cache_speed_recovery_window",
        )
    elif scene_bucket == "straight_car_follow":
        headway_time = _optional_float(record.get("following_min_thw", None)) or 2.5
        ttc_margin = DEFAULT_TTC_MAX_S
        closing_tolerance_ttc = DEFAULT_TTC_MAX_S
        raw_values = np.asarray([headway_time, ttc_margin, closing_tolerance_ttc], dtype=np.float32)
        aux = {
            "ego_mean_speed_mps": float(_optional_float(record.get("ego_mean_speed", None)) or 0.0),
            "persistent_closing_event_count": 0,
            "lead_present_step_count": 0,
            "ttc_cap_s": float(DEFAULT_TTC_MAX_S),
        }
        axis_valid_mask = np.asarray([record.get("following_min_thw", None) is not None, False, False], dtype=bool)
        axis_invalid_reasons = (
            "" if axis_valid_mask[0] else "lead_headway_unavailable_in_proxy",
            "requires_cache_persistent_closing",
            "requires_cache_brake_trigger",
        )
    elif scene_bucket == "straight_lane_change":
        onset_step = _optional_float(record.get("ego_lateral_onset_step", None))
        onset_time_s = float(onset_step * DEFAULT_DT) if onset_step is not None else 3.5
        lateral_commit = max(float(record.get("ego_lateral_speed_peak", 0.0)), 0.0)
        merge_gap_distance = _optional_float(record.get("merge_min_gap", None))
        ego_mean_speed_mps = float(_optional_float(record.get("ego_mean_speed", None)) or DEFAULT_TIME_GAP_VMIN_MPS)
        merge_gap_time_s = DEFAULT_TTC_MAX_S
        if merge_gap_distance is not None:
            merge_gap_time_s = max(float(merge_gap_distance) / max(ego_mean_speed_mps, DEFAULT_TIME_GAP_VMIN_MPS), 0.0)
        raw_values = np.asarray([onset_time_s, lateral_commit, merge_gap_time_s], dtype=np.float32)
        aux = {
            "ego_mean_speed_mps": float(ego_mean_speed_mps),
            "front_gap_time_s": -1.0,
            "rear_gap_time_s": -1.0,
            "source_lane_index": -1,
            "target_lane_index": -1,
        }
        onset_valid = onset_step is not None
        axis_valid_mask = np.asarray([onset_valid, onset_valid, False], dtype=bool)
        axis_invalid_reasons = (
            "" if onset_valid else "lateral_onset_unavailable_in_proxy",
            "" if onset_valid else "lateral_execution_unavailable_in_proxy",
            "requires_cache_target_lane_association",
        )
    else:
        raise ValueError(f"Unsupported scene bucket for behavior metrics: {scene_bucket!r}")

    return BehaviorMetricBundle(
        scene_bucket=scene_bucket,
        raw_metric_names=RAW_BEHAVIOR_METRIC_BY_SCENE[scene_bucket],
        raw_metric_values=np.asarray(raw_values, dtype=np.float32),
        canonical_axis_names=CANONICAL_AXIS_BY_SCENE[scene_bucket],
        axis_valid_mask=axis_valid_mask,
        axis_invalid_reasons=axis_invalid_reasons,
        metric_source="record_proxy",
        metric_note="cache_missing_or_unavailable",
        metric_aux=aux,
    )


def resolve_behavior_metric_values(
    *,
    scene_bucket: str,
    raw_metric_values: Sequence[float],
    metric_aux: Mapping[str, object] | None,
    reference_speed_mps: float | None,
) -> np.ndarray:
    raw_values = np.asarray(raw_metric_values, dtype=np.float32).reshape(-1)
    if raw_values.shape[0] != 3:
        raise ValueError(f"Expected 3 raw metric values for {scene_bucket!r}, got {raw_values.shape}")
    aux = metric_aux if isinstance(metric_aux, Mapping) else {}

    if scene_bucket != "straight_free_drive":
        return raw_values.astype(np.float32)

    # v3 free-driving measurements are already opportunity-conditioned at
    # extraction time.  Recomputing them from aggregate p90 values would erase
    # the frame-level eligibility decision, so stage two keeps them unchanged.
    if str(aux.get("free_metric_definition", "")).startswith("v3_"):
        return raw_values.astype(np.float32)

    ego_mean_speed_mps = _optional_float(aux.get("ego_mean_speed_mps", None)) if aux else None
    positive_accel_p90 = _optional_float(aux.get("positive_accel_p90_mps2", None)) if aux else None
    positive_jerk_p90 = _optional_float(aux.get("positive_jerk_p90_mps3", None)) if aux else None
    route_speed_limit_mps = _optional_float(aux.get("route_speed_limit_mps", None)) if aux else None
    if route_speed_limit_mps is not None and route_speed_limit_mps <= 0.5:
        route_speed_limit_mps = None

    return _free_drive_raw_values(
        ego_mean_speed_mps=float(ego_mean_speed_mps if ego_mean_speed_mps is not None else raw_values[0] * max(float(reference_speed_mps or 15.0), 1.0)),
        positive_accel_p90_mps2=float(positive_accel_p90 if positive_accel_p90 is not None else raw_values[1]),
        positive_jerk_p90_mps3=float(positive_jerk_p90 if positive_jerk_p90 is not None else raw_values[2]),
        route_speed_limit_mps=route_speed_limit_mps,
        reference_speed_mps=reference_speed_mps,
    )


def build_behavior_metric_bundle(
    record: Mapping[str, object],
    reference_speed_mps: float | None = None,
) -> BehaviorMetricBundle:
    """Build the final 3-d hard behavior metric bundle for one sample."""

    scene_bucket = str(record.get("scene_bucket", record.get("scene_bucket_name", "none")))
    if scene_bucket not in CANONICAL_AXIS_BY_SCENE:
        raise ValueError(f"Unsupported scene bucket for behavior metrics: {scene_bucket!r}")

    cache_path = ""
    for key in ("cache_path", "planner_cache_path", "style_cache_path"):
        candidate = str(record.get(key, "") or "").strip()
        if candidate:
            cache_path = candidate
            break

    if cache_path and Path(cache_path).exists():
        try:
            if scene_bucket == "straight_free_drive":
                return _free_drive_metrics_from_cache(cache_path, record, reference_speed_mps)
            if scene_bucket == "straight_car_follow":
                return _car_follow_metrics_from_cache(cache_path, record)
            if scene_bucket == "straight_lane_change":
                return _lane_change_metrics_from_cache(cache_path, record)
        except (KeyError, OSError, IndexError, ValueError) as exc:
            bundle = _record_proxy_metric_bundle(scene_bucket, record, reference_speed_mps)
            return BehaviorMetricBundle(
                scene_bucket=bundle.scene_bucket,
                raw_metric_names=bundle.raw_metric_names,
                raw_metric_values=bundle.raw_metric_values,
                canonical_axis_names=bundle.canonical_axis_names,
                axis_valid_mask=bundle.axis_valid_mask,
                axis_invalid_reasons=bundle.axis_invalid_reasons,
                metric_source="record_proxy",
                metric_note=f"cache_read_failed:{type(exc).__name__}",
                metric_aux=bundle.metric_aux,
            )

    return _record_proxy_metric_bundle(scene_bucket, record, reference_speed_mps)

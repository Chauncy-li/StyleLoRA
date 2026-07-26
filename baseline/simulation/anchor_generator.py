"""Map-feasible maneuver anchors for diffusion warm-start planning.

The generator deliberately uses NuPlan's lane graph instead of the tensorized
``route_lanes`` feature.  The latter is clipped and loses adjacency metadata,
while a warm-start maneuver must never be initialized on an opposite-direction
or disconnected lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from baseline.data_process.roadblock_utils import route_roadblock_correction


MANEUVER_ORDER: Tuple[str, ...] = ("keep", "left", "right")


@dataclass(frozen=True)
class AnchorCandidate:
    """One executable maneuver anchor in the current ego frame."""

    intent: str
    ego_future_local: np.ndarray  # [T, 4]: x, y, cos(heading), sin(heading)
    ego_future_global_xy: np.ndarray  # [T, 2]
    target_lane_id: Optional[str]
    diagnostics: Dict[str, object]


@dataclass(frozen=True)
class AnchorBundle:
    """All currently available maneuvers plus a deterministic brake fallback."""

    candidates: Tuple[AnchorCandidate, ...]
    fallback_ego_future_local: np.ndarray  # [T, 4]
    corrected_route_roadblock_ids: Tuple[str, ...]
    unavailable_reasons: Dict[str, str]


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _heading_difference(a: float, b: float) -> float:
    return float(abs(_wrap_angle(float(a) - float(b))))


def _polyline_arclength(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    segment = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(segment)])


def _deduplicate_polyline(xy: np.ndarray, min_spacing: float = 0.05) -> np.ndarray:
    if len(xy) < 2:
        return xy
    keep = np.concatenate(
        [np.ones((1,), dtype=bool), np.linalg.norm(np.diff(xy, axis=0), axis=1) > min_spacing]
    )
    return xy[keep]


class MapAnchorGenerator:
    """Construct keep/left/right anchors from the live NuPlan lane graph."""

    def __init__(self, config, *, horizon_s: float, step_interval: float) -> None:
        self.horizon_s = float(horizon_s)
        self.step_interval = float(step_interval)
        self.future_len = int(round(self.horizon_s / self.step_interval))
        self.query_radius_m = float(getattr(config, "anchor_query_radius_m", 12.0))
        self.max_path_length_m = float(getattr(config, "anchor_max_path_length_m", 160.0))
        self.max_heading_error_rad = float(
            np.deg2rad(float(getattr(config, "anchor_max_heading_error_deg", 60.0)))
        )
        self.lane_change_start_s = float(getattr(config, "anchor_lane_change_start_s", 0.5))
        self.lane_change_duration_s = float(getattr(config, "anchor_lane_change_duration_s", 3.5))
        self.min_lane_change_speed_mps = float(getattr(config, "anchor_min_lane_change_speed_mps", 1.0))
        self.min_reference_speed_mps = float(getattr(config, "anchor_min_reference_speed_mps", 2.0))
        self.max_reference_speed_mps = float(getattr(config, "anchor_max_reference_speed_mps", 18.0))
        self.speed_relaxation_s = float(getattr(config, "anchor_speed_relaxation_s", 1.5))
        self.fallback_deceleration_mps2 = float(getattr(config, "anchor_fallback_deceleration_mps2", 3.5))
        self.drivable_check_stride = max(1, int(getattr(config, "anchor_drivable_check_stride", 5)))
        self.max_anchor_offroad_fraction = float(getattr(config, "anchor_max_offroad_fraction", 0.05))

    def build(
        self,
        ego_state: EgoState,
        map_api: AbstractMap,
        route_roadblock_ids: Sequence[str],
    ) -> AnchorBundle:
        """Build every legal maneuver available at the current simulation step."""

        corrected_route_ids = self._correct_route(ego_state, map_api, route_roadblock_ids)
        route_set = set(corrected_route_ids)
        unavailable: Dict[str, str] = {}
        current_lane = self._select_current_lane(ego_state, map_api, route_set)

        if current_lane is None:
            stationary = self._stationary_local_anchor()
            candidate = AnchorCandidate(
                intent="keep",
                ego_future_local=stationary,
                ego_future_global_xy=self._local_to_global_xy(stationary[:, :2], ego_state),
                target_lane_id=None,
                diagnostics={"source": "stationary_no_lane", "map_feasible": False},
            )
            unavailable.update({"left": "current_lane_unavailable", "right": "current_lane_unavailable"})
            return AnchorBundle(
                candidates=(candidate,),
                fallback_ego_future_local=stationary.copy(),
                corrected_route_roadblock_ids=tuple(corrected_route_ids),
                unavailable_reasons=unavailable,
            )

        speed_mps = max(float(ego_state.dynamic_car_state.speed), 0.0)
        required_length = min(
            self.max_path_length_m,
            max(speed_mps, self.min_reference_speed_mps) * self.horizon_s + 25.0,
        )
        keep_path = self._build_lane_path(
            current_lane, ego_state, route_set, required_length, start_at_ego=True
        )
        if keep_path is None:
            stationary = self._stationary_local_anchor()
            candidate = AnchorCandidate(
                intent="keep",
                ego_future_local=stationary,
                ego_future_global_xy=self._local_to_global_xy(stationary[:, :2], ego_state),
                target_lane_id=str(current_lane.id),
                diagnostics={"source": "stationary_path_failure", "map_feasible": False},
            )
            unavailable.update({"left": "keep_path_unavailable", "right": "keep_path_unavailable"})
            return AnchorBundle(
                candidates=(candidate,),
                fallback_ego_future_local=stationary.copy(),
                corrected_route_roadblock_ids=tuple(corrected_route_ids),
                unavailable_reasons=unavailable,
            )

        progress = self._longitudinal_progress(current_lane, speed_mps)
        keep_xy = self._sample_polyline(keep_path, progress)
        keep_candidate = self._make_candidate(
            "keep", keep_xy, ego_state, map_api, target_lane_id=str(current_lane.id)
        )
        candidates: List[AnchorCandidate] = [keep_candidate]

        adjacent: Tuple[Optional[LaneGraphEdgeMapObject], Optional[LaneGraphEdgeMapObject]] = (None, None)
        try:
            adjacent = current_lane.adjacent_edges
        except (AttributeError, NotImplementedError):
            pass

        for intent, target_lane in zip(("left", "right"), adjacent):
            reason = self._validate_adjacent_lane(target_lane, ego_state, route_set, speed_mps)
            if reason is not None:
                unavailable[intent] = reason
                continue

            assert target_lane is not None
            target_path = self._build_lane_path(
                target_lane, ego_state, route_set, required_length, start_at_ego=False
            )
            if target_path is None:
                unavailable[intent] = "target_path_unavailable"
                continue
            target_xy = self._sample_polyline(target_path, progress)
            transition = self._lane_change_blend(keep_xy, target_xy)
            candidate = self._make_candidate(
                intent,
                transition,
                ego_state,
                map_api,
                target_lane_id=str(target_lane.id),
            )
            if not bool(candidate.diagnostics.get("map_feasible", False)):
                unavailable[intent] = "anchor_leaves_drivable_area"
                continue
            candidates.append(candidate)

        candidates.sort(key=lambda item: MANEUVER_ORDER.index(item.intent))
        fallback_progress = self._braking_progress(speed_mps)
        fallback_xy = self._sample_polyline(keep_path, fallback_progress)
        fallback_local = self._xy_to_local_state(fallback_xy, ego_state)
        return AnchorBundle(
            candidates=tuple(candidates),
            fallback_ego_future_local=fallback_local,
            corrected_route_roadblock_ids=tuple(corrected_route_ids),
            unavailable_reasons=unavailable,
        )

    def _correct_route(
        self, ego_state: EgoState, map_api: AbstractMap, route_roadblock_ids: Sequence[str]
    ) -> List[str]:
        route_ids = [str(item) for item in route_roadblock_ids if str(item)]
        if not route_ids:
            return []
        try:
            return list(route_roadblock_correction(ego_state, map_api, route_ids))
        except Exception:
            # Anchor generation must not take down closed-loop simulation.  The
            # original ordered route remains a useful successor preference.
            return route_ids

    def _select_current_lane(
        self, ego_state: EgoState, map_api: AbstractMap, route_set: set[str]
    ) -> Optional[LaneGraphEdgeMapObject]:
        point = ego_state.rear_axle.point
        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        nearby = map_api.get_proximal_map_objects(point, self.query_radius_m, layers)
        candidates = list(nearby.get(SemanticMapLayer.LANE, [])) + list(
            nearby.get(SemanticMapLayer.LANE_CONNECTOR, [])
        )
        ranked: List[Tuple[float, LaneGraphEdgeMapObject]] = []
        ego_xy = np.asarray([point.x, point.y], dtype=np.float64)
        for lane in candidates:
            path = self._lane_discrete_xy(lane)
            if path is None:
                continue
            nearest = int(np.argmin(np.linalg.norm(path - ego_xy[None, :], axis=1)))
            lane_heading = self._path_heading(path, nearest)
            heading_error = _heading_difference(lane_heading, ego_state.rear_axle.heading)
            if heading_error > self.max_heading_error_rad:
                continue
            distance = float(np.linalg.norm(path[nearest] - ego_xy))
            contains = False
            try:
                contains = bool(lane.contains_point(point))
            except Exception:
                pass
            on_route = not route_set or str(lane.get_roadblock_id()) in route_set
            score = distance + 4.0 * heading_error - (4.0 if contains else 0.0) - (2.0 if on_route else 0.0)
            ranked.append((score, lane))
        return min(ranked, key=lambda item: item[0])[1] if ranked else None

    def _validate_adjacent_lane(
        self,
        lane: Optional[LaneGraphEdgeMapObject],
        ego_state: EgoState,
        route_set: set[str],
        speed_mps: float,
    ) -> Optional[str]:
        if lane is None:
            return "adjacent_lane_unavailable"
        if speed_mps < self.min_lane_change_speed_mps:
            return "ego_speed_too_low"
        path = self._lane_discrete_xy(lane)
        if path is None:
            return "adjacent_lane_geometry_unavailable"
        ego_xy = np.asarray([ego_state.rear_axle.x, ego_state.rear_axle.y], dtype=np.float64)
        nearest = int(np.argmin(np.linalg.norm(path - ego_xy[None, :], axis=1)))
        heading = self._path_heading(path, nearest)
        if _heading_difference(heading, ego_state.rear_axle.heading) > self.max_heading_error_rad:
            return "opposite_direction_lane"
        if route_set and str(lane.get_roadblock_id()) not in route_set:
            return "adjacent_lane_off_route"
        return None

    def _build_lane_path(
        self,
        start_lane: LaneGraphEdgeMapObject,
        ego_state: EgoState,
        route_set: set[str],
        required_length: float,
        *,
        start_at_ego: bool,
    ) -> Optional[np.ndarray]:
        first = self._lane_discrete_xy(start_lane)
        if first is None:
            return None
        ego_xy = np.asarray([ego_state.rear_axle.x, ego_state.rear_axle.y], dtype=np.float64)
        nearest = int(np.argmin(np.linalg.norm(first - ego_xy[None, :], axis=1)))
        first_piece = first[nearest:]
        if start_at_ego:
            first_piece = np.concatenate([ego_xy[None, :], first_piece], axis=0)
        pieces: List[np.ndarray] = [first_piece]
        current = start_lane
        visited = {str(start_lane.id)}
        depth = 0
        while _polyline_arclength(_deduplicate_polyline(np.concatenate(pieces, axis=0)))[-1] < required_length:
            depth += 1
            if depth > 40:
                break
            next_lane = self._select_successor(current, route_set, visited)
            if next_lane is None:
                break
            next_path = self._lane_discrete_xy(next_lane)
            if next_path is None:
                break
            pieces.append(next_path)
            visited.add(str(next_lane.id))
            current = next_lane
        result = _deduplicate_polyline(np.concatenate(pieces, axis=0))
        return result if len(result) >= 2 and _polyline_arclength(result)[-1] > 1.0 else None

    def _select_successor(
        self,
        current: LaneGraphEdgeMapObject,
        route_set: set[str],
        visited: set[str],
    ) -> Optional[LaneGraphEdgeMapObject]:
        candidates = [edge for edge in current.outgoing_edges if str(edge.id) not in visited]
        if not candidates:
            return None
        current_path = self._lane_discrete_xy(current)
        if current_path is None:
            return None
        current_heading = self._path_heading(current_path, len(current_path) - 1)

        def score(edge: LaneGraphEdgeMapObject) -> Tuple[int, float, str]:
            path = self._lane_discrete_xy(edge)
            heading_error = np.pi if path is None else _heading_difference(self._path_heading(path, 0), current_heading)
            off_route = int(bool(route_set) and str(edge.get_roadblock_id()) not in route_set)
            return off_route, heading_error, str(edge.id)

        return min(candidates, key=score)

    @staticmethod
    def _lane_discrete_xy(lane: LaneGraphEdgeMapObject) -> Optional[np.ndarray]:
        try:
            xy = np.asarray([[pose.x, pose.y] for pose in lane.baseline_path.discrete_path], dtype=np.float64)
        except Exception:
            return None
        return _deduplicate_polyline(xy) if len(xy) >= 2 and np.isfinite(xy).all() else None

    @staticmethod
    def _path_heading(path: np.ndarray, index: int) -> float:
        if index <= 0:
            delta = path[1] - path[0]
        elif index >= len(path) - 1:
            delta = path[-1] - path[-2]
        else:
            delta = path[index + 1] - path[index - 1]
        return float(np.arctan2(delta[1], delta[0]))

    def _longitudinal_progress(self, lane: LaneGraphEdgeMapObject, speed_mps: float) -> np.ndarray:
        speed_limit = getattr(lane, "speed_limit_mps", None)
        if speed_limit is None or not np.isfinite(speed_limit):
            reference = max(speed_mps, self.min_reference_speed_mps)
        else:
            reference = min(max(speed_mps, self.min_reference_speed_mps), float(speed_limit))
        reference = float(np.clip(reference, 0.0, self.max_reference_speed_mps))
        time = np.arange(1, self.future_len + 1, dtype=np.float64) * self.step_interval
        tau = max(self.speed_relaxation_s, 1e-3)
        progress = speed_mps * time + (reference - speed_mps) * (
            time - tau * (1.0 - np.exp(-time / tau))
        )
        return np.maximum.accumulate(np.maximum(progress, 0.0))

    def _braking_progress(self, speed_mps: float) -> np.ndarray:
        time = np.arange(1, self.future_len + 1, dtype=np.float64) * self.step_interval
        deceleration = max(self.fallback_deceleration_mps2, 1e-3)
        stop_time = speed_mps / deceleration
        effective_time = np.minimum(time, stop_time)
        progress = speed_mps * effective_time - 0.5 * deceleration * effective_time**2
        return np.maximum.accumulate(np.maximum(progress, 0.0))

    @staticmethod
    def _sample_polyline(path: np.ndarray, progress: np.ndarray) -> np.ndarray:
        arclength = _polyline_arclength(path)
        clipped = np.clip(progress, 0.0, arclength[-1])
        x = np.interp(clipped, arclength, path[:, 0])
        y = np.interp(clipped, arclength, path[:, 1])
        return np.stack([x, y], axis=-1)

    def _lane_change_blend(self, keep_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
        time = np.arange(1, self.future_len + 1, dtype=np.float64) * self.step_interval
        phase = np.clip(
            (time - self.lane_change_start_s) / max(self.lane_change_duration_s, self.step_interval),
            0.0,
            1.0,
        )
        smooth = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
        return keep_xy + smooth[:, None] * (target_xy - keep_xy)

    def _make_candidate(
        self,
        intent: str,
        global_xy: np.ndarray,
        ego_state: EgoState,
        map_api: AbstractMap,
        *,
        target_lane_id: Optional[str],
    ) -> AnchorCandidate:
        local_state = self._xy_to_local_state(global_xy, ego_state)
        offroad_fraction = self._offroad_fraction(global_xy, map_api)
        diagnostics: Dict[str, object] = {
            "source": "nuplan_lane_graph",
            "map_feasible": bool(offroad_fraction <= self.max_anchor_offroad_fraction),
            "offroad_fraction": float(offroad_fraction),
            "target_lane_id": target_lane_id,
        }
        return AnchorCandidate(
            intent=intent,
            ego_future_local=local_state,
            ego_future_global_xy=global_xy.astype(np.float32),
            target_lane_id=target_lane_id,
            diagnostics=diagnostics,
        )

    def _offroad_fraction(self, global_xy: np.ndarray, map_api: AbstractMap) -> float:
        indices = list(range(0, len(global_xy), self.drivable_check_stride))
        if indices[-1] != len(global_xy) - 1:
            indices.append(len(global_xy) - 1)
        valid = []
        for index in indices:
            point = Point2D(float(global_xy[index, 0]), float(global_xy[index, 1]))
            try:
                valid.append(bool(map_api.is_in_layer(point, SemanticMapLayer.DRIVABLE_AREA)))
            except Exception:
                # Some map implementations do not expose DRIVABLE_AREA.  Lane
                # graph membership has already provided the primary guarantee.
                return 0.0
        return float(1.0 - np.mean(valid)) if valid else 0.0

    def _xy_to_local_state(self, global_xy: np.ndarray, ego_state: EgoState) -> np.ndarray:
        origin = np.asarray([ego_state.rear_axle.x, ego_state.rear_axle.y], dtype=np.float64)
        heading = float(ego_state.rear_axle.heading)
        c, s = np.cos(heading), np.sin(heading)
        delta = global_xy - origin[None, :]
        local_xy = np.stack(
            [delta[:, 0] * c + delta[:, 1] * s, -delta[:, 0] * s + delta[:, 1] * c],
            axis=-1,
        )
        points = np.concatenate([np.zeros((1, 2), dtype=np.float64), local_xy], axis=0)
        tangent = np.diff(points, axis=0)
        tangent_norm = np.linalg.norm(tangent, axis=-1)
        for index in range(len(tangent)):
            if tangent_norm[index] < 1e-4:
                tangent[index] = tangent[index - 1] if index > 0 else np.asarray([1.0, 0.0])
        local_heading = np.arctan2(tangent[:, 1], tangent[:, 0])
        return np.concatenate(
            [local_xy, np.cos(local_heading)[:, None], np.sin(local_heading)[:, None]], axis=-1
        ).astype(np.float32)

    @staticmethod
    def _local_to_global_xy(local_xy: np.ndarray, ego_state: EgoState) -> np.ndarray:
        heading = float(ego_state.rear_axle.heading)
        c, s = np.cos(heading), np.sin(heading)
        origin = np.asarray([ego_state.rear_axle.x, ego_state.rear_axle.y], dtype=np.float64)
        global_delta = np.stack(
            [local_xy[:, 0] * c - local_xy[:, 1] * s, local_xy[:, 0] * s + local_xy[:, 1] * c],
            axis=-1,
        )
        return global_delta + origin[None, :]

    def _stationary_local_anchor(self) -> np.ndarray:
        result = np.zeros((self.future_len, 4), dtype=np.float32)
        result[:, 2] = 1.0
        return result

"""Schema definitions for the straight-driving-first style split pipeline."""
# Foundational data structures for straight-road scene partitioning.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


STYLE_SCENE_SPLIT_SCHEMA_VERSION = 4

# Straight-road scene categories.
SCENE_BUCKET_ORDER = (
    "straight_free_drive",
    "straight_car_follow",
    "straight_lane_change",
)
# Three driving-style categories.
STYLE_SCORE_ORDER = ("aggressive", "normal", "conservative")

SCENE_BUCKET_NAME_TO_ID = {
    "none": 0,
    "straight_free_drive": 1,
    "straight_car_follow": 2,
    "straight_lane_change": 3,
}
SCENE_BUCKET_ID_TO_NAME = {value: key for key, value in SCENE_BUCKET_NAME_TO_ID.items()}

STYLE_LABEL_NAME_TO_ID = {
    "unknown": 0,
    "conservative": 1,
    "normal": 2,
    "aggressive": 3,
}
STYLE_LABEL_ID_TO_NAME = {value: key for key, value in STYLE_LABEL_NAME_TO_ID.items()}

# Road-topology categories.
TOPOLOGY_BUCKET_NAME_TO_ID = {
    "unknown": 0,
    "straight": 1,
    "turning": 2,
    "lane_change_like": 3,
    "junction_like": 4,
}
TOPOLOGY_BUCKET_ID_TO_NAME = {value: key for key, value in TOPOLOGY_BUCKET_NAME_TO_ID.items()}


def _float_or_default(value: float | None, default: float = -1.0) -> np.ndarray:
    if value is None:
        value = default
    return np.array(float(value), dtype=np.float32)


@dataclass(frozen=True)
class StyleSceneSplitResult:
    """Final per-sample split result."""

    scene_bucket: str
    style_label: str
    subset_id: str
    topology_bucket: str
    primary_bucket: str
    secondary_bucket: str
    scene_confidence: float
    style_confidence: float
    split_confidence: float
    split_valid: bool
    primary_score: float
    secondary_score: float
    dominant_neighbor_idx: int
    scene_reason: str
    style_reason: str
    scene_score_vec: np.ndarray
    style_score_vec: np.ndarray
    global_min_distance: float
    following_min_gap: float
    following_min_thw: float | None
    crossing_min_distance: float
    crossing_time_offset: float | None
    merge_min_gap: float
    merge_lateral_closure: float
    ego_mean_speed: float
    ego_accel_peak: float
    ego_brake_peak: float
    ego_jerk_peak: float
    ego_jerk_p90: float
    ego_progress: float
    ego_speed_ratio_to_limit: float | None
    route_speed_limit_mps: float | None
    event_speed_drop_ratio: float
    event_brake_peak: float
    ego_lateral_disp: float
    ego_lateral_speed_peak: float
    ego_lateral_onset_step: float | None
    ego_heading_change: float
    route_lane_count: int
    nearby_agent_count: int
    lead_vehicle_present: bool
    route_has_control: bool

    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        """Convert to a `.npz`-friendly payload."""

        return {
            "style_scene_split_schema_version": np.array(STYLE_SCENE_SPLIT_SCHEMA_VERSION, dtype=np.int64),
            "scene_bucket": np.array(SCENE_BUCKET_NAME_TO_ID[self.scene_bucket], dtype=np.int64),
            "style_label": np.array(STYLE_LABEL_NAME_TO_ID[self.style_label], dtype=np.int64),
            "subset_id": np.array(self.subset_id),
            "topology_bucket": np.array(TOPOLOGY_BUCKET_NAME_TO_ID[self.topology_bucket], dtype=np.int64),
            "primary_bucket": np.array(SCENE_BUCKET_NAME_TO_ID[self.primary_bucket], dtype=np.int64),
            "secondary_bucket": np.array(SCENE_BUCKET_NAME_TO_ID[self.secondary_bucket], dtype=np.int64),
            "scene_confidence": np.array(self.scene_confidence, dtype=np.float32),
            "style_confidence": np.array(self.style_confidence, dtype=np.float32),
            "split_confidence": np.array(self.split_confidence, dtype=np.float32),
            "split_valid": np.array(int(self.split_valid), dtype=np.int64),
            "primary_score": np.array(self.primary_score, dtype=np.float32),
            "secondary_score": np.array(self.secondary_score, dtype=np.float32),
            "dominant_neighbor_idx": np.array(self.dominant_neighbor_idx, dtype=np.int64),
            "scene_reason": np.array(self.scene_reason),
            "style_reason": np.array(self.style_reason),
            "scene_score_vec": np.asarray(self.scene_score_vec, dtype=np.float32),
            "style_score_vec": np.asarray(self.style_score_vec, dtype=np.float32),
            "global_min_distance": np.array(self.global_min_distance, dtype=np.float32),
            "following_min_gap": np.array(self.following_min_gap, dtype=np.float32),
            "following_min_thw": _float_or_default(self.following_min_thw),
            "crossing_min_distance": np.array(self.crossing_min_distance, dtype=np.float32),
            "crossing_time_offset": _float_or_default(self.crossing_time_offset),
            "merge_min_gap": np.array(self.merge_min_gap, dtype=np.float32),
            "merge_lateral_closure": np.array(self.merge_lateral_closure, dtype=np.float32),
            "ego_mean_speed": np.array(self.ego_mean_speed, dtype=np.float32),
            "ego_accel_peak": np.array(self.ego_accel_peak, dtype=np.float32),
            "ego_brake_peak": np.array(self.ego_brake_peak, dtype=np.float32),
            "ego_jerk_peak": np.array(self.ego_jerk_peak, dtype=np.float32),
            "ego_jerk_p90": np.array(self.ego_jerk_p90, dtype=np.float32),
            "ego_progress": np.array(self.ego_progress, dtype=np.float32),
            "ego_speed_ratio_to_limit": _float_or_default(self.ego_speed_ratio_to_limit),
            "route_speed_limit_mps": _float_or_default(self.route_speed_limit_mps),
            "event_speed_drop_ratio": np.array(self.event_speed_drop_ratio, dtype=np.float32),
            "event_brake_peak": np.array(self.event_brake_peak, dtype=np.float32),
            "ego_lateral_disp": np.array(self.ego_lateral_disp, dtype=np.float32),
            "ego_lateral_speed_peak": np.array(self.ego_lateral_speed_peak, dtype=np.float32),
            "ego_lateral_onset_step": _float_or_default(self.ego_lateral_onset_step),
            "ego_heading_change": np.array(self.ego_heading_change, dtype=np.float32),
            "route_lane_count": np.array(self.route_lane_count, dtype=np.int64),
            "nearby_agent_count": np.array(self.nearby_agent_count, dtype=np.int64),
            "lead_vehicle_present": np.array(int(self.lead_vehicle_present), dtype=np.int64),
            "route_has_control": np.array(int(self.route_has_control), dtype=np.int64),
        }



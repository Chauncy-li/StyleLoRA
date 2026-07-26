"""Shared scene and behavior-axis definitions for V5/V6."""

from __future__ import annotations

from typing import Dict, Tuple

from research.style_scene_split.defaults import PRIMARY_SCENE_BUCKETS

SCENE_BUCKET_ORDER: Tuple[str, ...] = tuple(PRIMARY_SCENE_BUCKETS)

CANONICAL_AXIS_BY_SCENE: Dict[str, Tuple[str, str, str]] = {
    "straight_free_drive": (
        "speed_utilization",
        "accel_willingness",
        "speed_response_intensity",
    ),
    "straight_car_follow": (
        "headway_tightness_from_h",
        "ttc_tightness",
        "closing_tolerance",
    ),
    "straight_lane_change": (
        "initiation_aggressiveness",
        "lateral_commitment",
        "small_gap_acceptance_from_m_gap",
    ),
}

RAW_BEHAVIOR_METRIC_BY_SCENE: Dict[str, Tuple[str, str, str]] = {
    "straight_free_drive": (
        "r_v",
        "r_a",
        "r_response",
    ),
    "straight_car_follow": (
        "h",
        "ttc_margin",
        "r_closing",
    ),
    "straight_lane_change": (
        "r_init",
        "r_commit",
        "m_gap",
    ),
}

RAW_METRIC_DIRECTION_BY_SCENE: Dict[str, Tuple[str, str, str]] = {
    "straight_free_drive": (
        "rising",
        "rising",
        "rising",
    ),
    "straight_car_follow": (
        "falling",
        "falling",
        # r_closing is the TTC at the first sustained ego response threshold:
        # accepting a smaller value means tolerating a tighter approach.
        "falling",
    ),
    "straight_lane_change": (
        "falling",
        "rising",
        "falling",
    ),
}


def canonical_axis_names_for_scene(scene_bucket: str) -> Tuple[str, str, str]:
    """Return aggressive-direction-aligned axis names for a scene bucket."""

    if scene_bucket not in CANONICAL_AXIS_BY_SCENE:
        raise KeyError(f"Unsupported scene bucket for continuous style: {scene_bucket!r}")
    return CANONICAL_AXIS_BY_SCENE[scene_bucket]

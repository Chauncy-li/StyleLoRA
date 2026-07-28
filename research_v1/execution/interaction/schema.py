"""Schema and file conventions for interaction-state proxy datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from research_v1.scene_data.paths import PRIMARY_SCENE_BUCKETS
from research_v1.scene_data.schema import style_axis_names_for_scene

INTERACTION_STATE_SCHEMA_VERSION = 1

SCENE_GATE_ORDER: Tuple[str, ...] = tuple(PRIMARY_SCENE_BUCKETS)
SCENE_GATE_TO_INDEX: Dict[str, int] = {
    scene_bucket: index for index, scene_bucket in enumerate(SCENE_GATE_ORDER)
}
AXIS_GATE_BY_SCENE: Dict[str, Tuple[str, str, str]] = {
    scene_bucket: style_axis_names_for_scene(scene_bucket) for scene_bucket in SCENE_GATE_ORDER
}
AXIS_GATE_ORDER: Tuple[str, ...] = tuple(
    axis_name
    for scene_bucket in SCENE_GATE_ORDER
    for axis_name in AXIS_GATE_BY_SCENE[scene_bucket]
)

FEATURE_NAME_ORDER: Tuple[str, ...] = (
    "lead_vehicle_present",
    "follow_gap_pressure",
    "follow_thw_pressure",
    "merge_gap_pressure",
    "merge_closure_pressure",
    "speed_ratio_norm",
    "speed_drop_pressure",
    "brake_pressure",
    "lateral_disp_norm",
    "lateral_speed_norm",
    "route_lane_count_norm",
    "nearby_agent_density_norm",
    "density_level_norm",
    "speed_regime_norm",
    "curvature_level_norm",
    "progress_norm",
    "heading_change_norm",
    "route_has_control",
)

DEFAULT_OUTPUT_SUBDIR = "interaction_state"
DEFAULT_INDEX_FILENAME = "interaction_state_index.jsonl"
DEFAULT_SUMMARY_FILENAME = "interaction_state_summary.json"
DEFAULT_VALIDATION_FILENAME = "validate_interaction_state.json"


def interaction_state_output_dir(split_root: str) -> str:
    """Return the canonical output directory for a split-root interaction-state export."""

    return str(Path(split_root) / DEFAULT_OUTPUT_SUBDIR)


def interaction_state_index_path(output_dir: str) -> str:
    """Return the canonical interaction-state JSONL path."""

    return str(Path(output_dir) / DEFAULT_INDEX_FILENAME)


def interaction_state_summary_path(output_dir: str) -> str:
    """Return the canonical interaction-state summary JSON path."""

    return str(Path(output_dir) / DEFAULT_SUMMARY_FILENAME)


def interaction_state_validation_path(output_dir: str) -> str:
    """Return the canonical interaction-state validation JSON path."""

    return str(Path(output_dir) / DEFAULT_VALIDATION_FILENAME)

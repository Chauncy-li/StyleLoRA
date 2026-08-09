"""Shared helpers for normalized split-index access and retrieval prep."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np


SPLIT_OPTIONAL_FLOAT_FIELDS: Sequence[str] = (
    "following_min_thw",
    "crossing_time_offset",
    "ego_speed_ratio_to_limit",
    "route_speed_limit_mps",
    "ego_lateral_onset_step",
)

SPLIT_REQUIRED_FLOAT_FIELDS: Sequence[str] = (
    "scene_confidence",
    "style_confidence",
    "split_confidence",
    "primary_score",
    "secondary_score",
    "global_min_distance",
    "following_min_gap",
    "crossing_min_distance",
    "merge_min_gap",
    "merge_lateral_closure",
    "ego_mean_speed",
    "ego_accel_peak",
    "ego_brake_peak",
    "ego_jerk_peak",
    "ego_jerk_p90",
    "ego_progress",
    "event_speed_drop_ratio",
    "event_brake_peak",
    "ego_lateral_disp",
    "ego_lateral_speed_peak",
    "ego_heading_change",
)

SPLIT_BOOLEAN_FIELDS: Sequence[str] = (
    "split_valid",
    "lead_vehicle_present",
    "route_has_control",
)

SPLIT_INTEGER_FIELDS: Sequence[str] = (
    "route_lane_count",
    "nearby_agent_count",
    "dominant_neighbor_idx",
)

RETRIEVAL_METRIC_KEYS_BY_BUCKET: Dict[str, Sequence[str]] = {
    "straight_car_follow": (
        "following_min_thw",
        "following_min_gap",
        "event_brake_peak",
        "event_speed_drop_ratio",
        "ego_brake_peak",
    ),
    "straight_lane_change": (
        "merge_min_gap",
        "ego_lateral_onset_step",
        "ego_lateral_speed_peak",
        "merge_lateral_closure",
        "ego_lateral_disp",
    ),
    "straight_free_drive": (
        "ego_speed_ratio_to_limit",
        "ego_accel_peak",
        "ego_jerk_p90",
        "ego_mean_speed",
        "ego_progress",
    ),
}


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, (float, np.floating)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return bool(default)


def _sanitize_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


def sanitize_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    if value >= 1e5 or value <= -0.5:
        return None
    return float(value)


def sanitize_float_list(values: object) -> List[float]:
    if values is None:
        return []
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return [float(item) for item in arr.tolist() if np.isfinite(item)]


def load_jsonl_records(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl_records(path: str, records: Iterable[Mapping[str, object]]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def normalize_split_index_record(record: Mapping[str, object]) -> Dict[str, object]:
    """Support both legacy and normalized split-index field names."""

    normalized = dict(record)

    source_mode = _as_str(record.get("source_mode"), default="")
    scene_bucket = _as_str(record.get("scene_bucket", record.get("scene_bucket_name")), default="none")
    style_label = _as_str(record.get("style_label", record.get("style_label_name")), default="unknown")
    topology_bucket = _as_str(record.get("topology_bucket", record.get("topology_bucket_name")), default="unknown")
    primary_bucket = _as_str(record.get("primary_bucket", record.get("primary_bucket_name")), default="none")
    secondary_bucket = _as_str(record.get("secondary_bucket", record.get("secondary_bucket_name")), default="none")

    sample_id = _as_str(record.get("sample_id"), default="")
    filename = _as_str(record.get("filename"), default=f"{sample_id}.npz" if sample_id else "")
    planner_cache_path = record.get("planner_cache_path", None)
    style_cache_path = record.get("style_cache_path", None)
    cache_path = record.get("cache_path", None)

    if style_cache_path is None and source_mode == "raw_scenario":
        style_cache_path = planner_cache_path or cache_path
    if cache_path is None:
        cache_path = style_cache_path or planner_cache_path

    normalized["source_mode"] = source_mode
    normalized["sample_id"] = sample_id
    normalized["filename"] = filename
    normalized["scene_bucket"] = scene_bucket
    normalized["scene_bucket_name"] = scene_bucket
    normalized["style_label"] = style_label
    normalized["style_label_name"] = style_label
    normalized["topology_bucket"] = topology_bucket
    normalized["topology_bucket_name"] = topology_bucket
    normalized["primary_bucket"] = primary_bucket
    normalized["primary_bucket_name"] = primary_bucket
    normalized["secondary_bucket"] = secondary_bucket
    normalized["secondary_bucket_name"] = secondary_bucket
    normalized["subset_id"] = _as_str(record.get("subset_id"), default="invalid")
    normalized["token"] = _as_str(record.get("token"), default=sample_id)
    normalized["map_name"] = _as_str(record.get("map_name"), default="unknown")
    normalized["log_name"] = _as_str(record.get("log_name"), default="")
    normalized["scenario_name"] = _as_str(record.get("scenario_name"), default="")
    normalized["scenario_type"] = _as_str(record.get("scenario_type"), default="")
    normalized["planner_cache_path"] = None if planner_cache_path in (None, "") else _as_str(planner_cache_path)
    normalized["style_cache_path"] = None if style_cache_path in (None, "") else _as_str(style_cache_path)
    normalized["cache_path"] = None if cache_path in (None, "") else _as_str(cache_path)
    normalized["sidecar_path"] = _as_str(record.get("sidecar_path"), default="")
    normalized["scene_reason"] = _as_str(record.get("scene_reason"), default="")
    normalized["style_reason"] = _as_str(record.get("style_reason"), default="")
    normalized["quality_reason"] = _as_str(record.get("quality_reason"), default="")

    for key in SPLIT_REQUIRED_FLOAT_FIELDS:
        normalized[key] = _sanitize_float(record.get(key, 0.0), default=0.0)
    for key in SPLIT_OPTIONAL_FLOAT_FIELDS:
        normalized[key] = sanitize_optional_float(record.get(key, None))
    for key in SPLIT_BOOLEAN_FIELDS:
        normalized[key] = _as_bool(record.get(key, False), default=False)
    for key in SPLIT_INTEGER_FIELDS:
        normalized[key] = int(_sanitize_float(record.get(key, 0), default=0.0))

    normalized["scene_score_vec"] = sanitize_float_list(record.get("scene_score_vec", []))
    normalized["style_score_vec"] = sanitize_float_list(record.get("style_score_vec", []))
    normalized["sample_quality_valid"] = _as_bool(record.get("sample_quality_valid", False), default=False)
    normalized["memory_eligible"] = _as_bool(
        record.get("memory_eligible", None),
        default=bool(normalized["split_valid"] and normalized["style_cache_path"]),
    )
    normalized["quality_score"] = _sanitize_float(record.get("quality_score", 0.0), default=0.0)
    normalized["style_performance_confidence"] = _sanitize_float(
        record.get("style_performance_confidence", 0.0),
        default=0.0,
    )
    normalized["quality_vec"] = sanitize_float_list(record.get("quality_vec", []))
    normalized["style_performance_vec"] = sanitize_float_list(record.get("style_performance_vec", []))
    normalized["style_axis_names"] = [str(item) for item in record.get("style_axis_names", [])]
    normalized["condition_density_level"] = _as_str(record.get("condition_density_level"), default="unknown")
    normalized["condition_speed_regime"] = _as_str(record.get("condition_speed_regime"), default="unknown")
    normalized["condition_curvature_level"] = _as_str(record.get("condition_curvature_level"), default="unknown")
    return normalized


def load_split_index(index_path: str, normalize: bool = True) -> List[Dict[str, object]]:
    records = load_jsonl_records(index_path)
    if not normalize:
        return records
    return [normalize_split_index_record(record) for record in records]


def iter_valid_split_records(index_path: str, require_style_cache: bool = False) -> Iterator[Dict[str, object]]:
    for record in load_split_index(index_path, normalize=True):
        if not bool(record["split_valid"]):
            continue
        if require_style_cache and not record.get("style_cache_path"):
            continue
        yield record


def bucket_metric_keys(scene_bucket: str) -> Sequence[str]:
    return RETRIEVAL_METRIC_KEYS_BY_BUCKET.get(str(scene_bucket), ())



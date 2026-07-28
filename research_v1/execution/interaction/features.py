"""Feature extraction for offline interaction-state proxy vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

from .schema import FEATURE_NAME_ORDER

DENSITY_LEVEL_TO_VALUE = {
    "unknown": 0.0,
    "sparse": 0.25,
    "medium": 0.60,
    "dense": 1.00,
}

SPEED_REGIME_TO_VALUE = {
    "unknown": 0.0,
    "slow": 0.20,
    "urban": 0.45,
    "suburban": 0.70,
    "high_speed": 1.00,
}

CURVATURE_LEVEL_TO_VALUE = {
    "unknown": 0.0,
    "low": 0.20,
    "mild": 0.60,
    "high": 1.00,
}


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return float(value)


@dataclass(frozen=True)
class InteractionStateFeatureBundle:
    """Normalized interaction-state proxy feature vector plus validity mask."""

    feature_names: Tuple[str, ...]
    feature_values: np.ndarray
    feature_mask: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(self.feature_names, self.feature_values.tolist())
        }

    def mask_dict(self) -> Dict[str, int]:
        return {
            name: int(value)
            for name, value in zip(self.feature_names, self.feature_mask.tolist())
        }

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "feature_values": [float(value) for value in self.feature_values.tolist()],
            "feature_mask": [int(value) for value in self.feature_mask.tolist()],
        }


def build_interaction_state_features(record: Mapping[str, object]) -> InteractionStateFeatureBundle:
    """Build a compact normalized proxy for the current interaction-state."""

    lead_vehicle_present = 1.0 if bool(record.get("lead_vehicle_present", False)) else 0.0

    following_min_gap = max(float(record.get("following_min_gap", 0.0)), 0.0)
    following_min_thw = _optional_float(record.get("following_min_thw", None))
    merge_min_gap = max(float(record.get("merge_min_gap", 0.0)), 0.0)
    merge_lateral_closure = abs(float(record.get("merge_lateral_closure", 0.0)))
    ego_speed_ratio_to_limit = _optional_float(record.get("ego_speed_ratio_to_limit", None))
    ego_mean_speed = max(float(record.get("ego_mean_speed", 0.0)), 0.0)
    event_speed_drop_ratio = max(float(record.get("event_speed_drop_ratio", 0.0)), 0.0)
    event_brake_peak = max(float(record.get("event_brake_peak", 0.0)), 0.0)
    ego_brake_peak = max(float(record.get("ego_brake_peak", 0.0)), 0.0)
    ego_lateral_disp = abs(float(record.get("ego_lateral_disp", 0.0)))
    ego_lateral_speed_peak = max(float(record.get("ego_lateral_speed_peak", 0.0)), 0.0)
    route_lane_count = max(int(record.get("route_lane_count", 0)), 0)
    nearby_agent_count = max(int(record.get("nearby_agent_count", 0)), 0)
    condition_density_level = str(record.get("condition_density_level", "unknown"))
    condition_speed_regime = str(record.get("condition_speed_regime", "unknown"))
    condition_curvature_level = str(record.get("condition_curvature_level", "unknown"))
    ego_progress = max(float(record.get("ego_progress", 0.0)), 0.0)
    ego_heading_change = abs(float(record.get("ego_heading_change", 0.0)))
    route_has_control = 1.0 if bool(record.get("route_has_control", False)) else 0.0

    speed_ratio_norm = (
        _clip01(ego_speed_ratio_to_limit)
        if ego_speed_ratio_to_limit is not None
        else _clip01(ego_mean_speed / 15.0)
    )

    values = {
        "lead_vehicle_present": lead_vehicle_present,
        "follow_gap_pressure": _clip01(1.0 - following_min_gap / 30.0),
        "follow_thw_pressure": 0.0 if following_min_thw is None else _clip01(1.0 - following_min_thw / 3.5),
        "merge_gap_pressure": _clip01(1.0 - merge_min_gap / 18.0),
        "merge_closure_pressure": _clip01(merge_lateral_closure / 2.0),
        "speed_ratio_norm": speed_ratio_norm,
        "speed_drop_pressure": _clip01(event_speed_drop_ratio),
        "brake_pressure": _clip01(max(event_brake_peak, ego_brake_peak) / 4.0),
        "lateral_disp_norm": _clip01(ego_lateral_disp / 3.5),
        "lateral_speed_norm": _clip01(ego_lateral_speed_peak / 1.5),
        "route_lane_count_norm": _clip01(max(route_lane_count - 1, 0) / 5.0),
        "nearby_agent_density_norm": _clip01(nearby_agent_count / 12.0),
        "density_level_norm": DENSITY_LEVEL_TO_VALUE.get(condition_density_level, 0.0),
        "speed_regime_norm": SPEED_REGIME_TO_VALUE.get(condition_speed_regime, 0.0),
        "curvature_level_norm": CURVATURE_LEVEL_TO_VALUE.get(condition_curvature_level, 0.0),
        "progress_norm": _clip01(ego_progress / 30.0),
        "heading_change_norm": _clip01(ego_heading_change / 0.35),
        "route_has_control": route_has_control,
    }

    masks = {
        "lead_vehicle_present": 1,
        "follow_gap_pressure": int(lead_vehicle_present > 0.0 or following_min_gap > 0.0),
        "follow_thw_pressure": int(following_min_thw is not None),
        "merge_gap_pressure": int(merge_min_gap > 0.0),
        "merge_closure_pressure": int(merge_lateral_closure > 0.0),
        "speed_ratio_norm": int(ego_speed_ratio_to_limit is not None or ego_mean_speed > 0.0),
        "speed_drop_pressure": 1,
        "brake_pressure": 1,
        "lateral_disp_norm": int(ego_lateral_disp > 0.0),
        "lateral_speed_norm": int(ego_lateral_speed_peak > 0.0),
        "route_lane_count_norm": int(route_lane_count > 0),
        "nearby_agent_density_norm": 1,
        "density_level_norm": int(condition_density_level != "unknown"),
        "speed_regime_norm": int(condition_speed_regime != "unknown"),
        "curvature_level_norm": int(condition_curvature_level != "unknown"),
        "progress_norm": 1,
        "heading_change_norm": 1,
        "route_has_control": 1,
    }

    feature_values = np.asarray(
        [values[name] for name in FEATURE_NAME_ORDER],
        dtype=np.float32,
    )
    feature_mask = np.asarray(
        [masks[name] for name in FEATURE_NAME_ORDER],
        dtype=np.int64,
    )
    return InteractionStateFeatureBundle(
        feature_names=FEATURE_NAME_ORDER,
        feature_values=feature_values,
        feature_mask=feature_mask,
    )

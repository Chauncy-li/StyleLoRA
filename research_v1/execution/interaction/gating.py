"""Soft scene-gating from normalized interaction-state proxy features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .features import InteractionStateFeatureBundle
from .schema import AXIS_GATE_BY_SCENE, AXIS_GATE_ORDER, SCENE_GATE_ORDER


def _softmax(logits: np.ndarray, temperature: float = 0.75) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32) / max(float(temperature), 1e-6)
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    return weights / np.clip(np.sum(weights), 1e-6, None)


@dataclass(frozen=True)
class SceneGateBundle:
    """Soft scene-gate summary used by later projection and conditioning stages."""

    scene_gate_names: Tuple[str, ...]
    scene_gate_values: np.ndarray
    dominant_scene_gate: str
    dominant_scene_gate_score: float
    axis_gate_names: Tuple[str, ...]
    axis_gate_values: np.ndarray

    def scene_gate_dict(self) -> Dict[str, float]:
        return {
            scene_bucket: float(value)
            for scene_bucket, value in zip(self.scene_gate_names, self.scene_gate_values.tolist())
        }

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "scene_gate_names": list(self.scene_gate_names),
            "scene_gate_values": [float(value) for value in self.scene_gate_values.tolist()],
            "dominant_scene_gate": self.dominant_scene_gate,
            "dominant_scene_gate_score": float(self.dominant_scene_gate_score),
            "axis_gate_names": list(self.axis_gate_names),
            "axis_gate_values": [float(value) for value in self.axis_gate_values.tolist()],
        }


def compute_scene_gates(feature_bundle: InteractionStateFeatureBundle) -> SceneGateBundle:
    """Map interaction-state proxy features to three soft straight-scene gates."""

    feature_dict = feature_bundle.as_dict()

    follow_pressure = max(
        feature_dict["follow_gap_pressure"],
        feature_dict["follow_thw_pressure"],
    )
    lane_change_pressure = max(
        feature_dict["merge_gap_pressure"],
        feature_dict["merge_closure_pressure"],
        feature_dict["lateral_disp_norm"],
        feature_dict["lateral_speed_norm"],
    )

    free_drive_logit = (
        0.95 * feature_dict["speed_ratio_norm"]
        + 0.60 * feature_dict["progress_norm"]
        + 0.25 * feature_dict["route_has_control"]
        + 0.20 * (1.0 - feature_dict["nearby_agent_density_norm"])
        + 0.35 * (1.0 - follow_pressure)
        + 0.45 * (1.0 - lane_change_pressure)
    )
    car_follow_logit = (
        1.40 * feature_dict["lead_vehicle_present"]
        + 1.10 * feature_dict["follow_gap_pressure"]
        + 1.00 * feature_dict["follow_thw_pressure"]
        + 0.55 * feature_dict["speed_drop_pressure"]
        + 0.50 * feature_dict["brake_pressure"]
        + 0.30 * feature_dict["nearby_agent_density_norm"]
        - 0.55 * feature_dict["lateral_disp_norm"]
        - 0.45 * feature_dict["lateral_speed_norm"]
    )
    lane_change_logit = (
        1.20 * feature_dict["lateral_disp_norm"]
        + 1.15 * feature_dict["lateral_speed_norm"]
        + 0.95 * feature_dict["merge_gap_pressure"]
        + 0.75 * feature_dict["merge_closure_pressure"]
        + 0.45 * feature_dict["heading_change_norm"]
        + 0.25 * feature_dict["route_lane_count_norm"]
        - 0.25 * feature_dict["lead_vehicle_present"]
    )

    scene_gate_values = _softmax(
        np.asarray(
            [
                free_drive_logit,
                car_follow_logit,
                lane_change_logit,
            ],
            dtype=np.float32,
        )
    )
    dominant_index = int(np.argmax(scene_gate_values))
    dominant_scene_gate = SCENE_GATE_ORDER[dominant_index]
    dominant_scene_gate_score = float(scene_gate_values[dominant_index])

    axis_gate_values = []
    for scene_bucket, scene_gate_value in zip(SCENE_GATE_ORDER, scene_gate_values.tolist()):
        axis_gate_values.extend([float(scene_gate_value)] * len(AXIS_GATE_BY_SCENE[scene_bucket]))

    return SceneGateBundle(
        scene_gate_names=SCENE_GATE_ORDER,
        scene_gate_values=scene_gate_values.astype(np.float32),
        dominant_scene_gate=dominant_scene_gate,
        dominant_scene_gate_score=dominant_scene_gate_score,
        axis_gate_names=AXIS_GATE_ORDER,
        axis_gate_values=np.asarray(axis_gate_values, dtype=np.float32),
    )

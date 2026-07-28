"""Causal interaction-state feature construction."""

from .builder import InteractionStateDatasetBuilder, build_interaction_state_record
from .features import InteractionStateFeatureBundle, build_interaction_state_features
from .gating import SceneGateBundle, compute_scene_gates
from .schema import (
    AXIS_GATE_ORDER,
    FEATURE_NAME_ORDER,
    INTERACTION_STATE_SCHEMA_VERSION,
    SCENE_GATE_ORDER,
    interaction_state_index_path,
    interaction_state_output_dir,
    interaction_state_summary_path,
)

__all__ = [
    "AXIS_GATE_ORDER",
    "FEATURE_NAME_ORDER",
    "INTERACTION_STATE_SCHEMA_VERSION",
    "InteractionStateDatasetBuilder",
    "InteractionStateFeatureBundle",
    "SCENE_GATE_ORDER",
    "SceneGateBundle",
    "build_interaction_state_features",
    "build_interaction_state_record",
    "compute_scene_gates",
    "interaction_state_index_path",
    "interaction_state_output_dir",
    "interaction_state_summary_path",
]

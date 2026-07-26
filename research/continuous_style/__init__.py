"""V5 data calibration and V6 continuous-preference conditioning."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEVKIT_ROOT = _REPO_ROOT / "nuplan-devkit"

for _path in (_DEVKIT_ROOT, _REPO_ROOT):
    _path_str = str(_path)
    if _path.exists() and _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from research._runtime import ensure_repo_on_path

ensure_repo_on_path()

from .metrics import build_behavior_metric_bundle, resolve_behavior_metric_values
from .router import route_scene_from_record
from .v6 import (
    STYLE_CONDITION_LAYOUT,
    StyleCommand,
    audit_v6_causal_router,
    audit_v6_command_support,
    build_rho_style_command,
    build_v6_direct_axis_conditions,
    causal_scene_gate_vector,
    evaluate_v6_rho_sweep,
    validate_v6_direct_axis_conditions,
)
from .schema import CANONICAL_AXIS_BY_SCENE
from .soft_metrics import soft_first_crossing_time, soft_high_quantile, soft_low_quantile, softmin

__all__ = [
    "CANONICAL_AXIS_BY_SCENE",
    "build_behavior_metric_bundle",
    "route_scene_from_record",
    "STYLE_CONDITION_LAYOUT",
    "StyleCommand",
    "audit_v6_causal_router",
    "audit_v6_command_support",
    "build_rho_style_command",
    "build_v6_direct_axis_conditions",
    "causal_scene_gate_vector",
    "evaluate_v6_rho_sweep",
    "resolve_behavior_metric_values",
    "soft_first_crossing_time",
    "soft_high_quantile",
    "soft_low_quantile",
    "softmin",
    "validate_v6_direct_axis_conditions",
]

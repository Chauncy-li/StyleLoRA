"""Shared style-condition builders for preference-conditioned diffusion."""

from __future__ import annotations

from typing import Any

import torch

STYLE_CONDITION_FEATURE_SET_GLOBAL_ONLY = "global_only"
STYLE_CONDITION_FEATURE_SET_EXEC_V2 = "exec_v2_effective_gap"
STYLE_CONDITION_FEATURE_SET_CHOICES = (
    STYLE_CONDITION_FEATURE_SET_GLOBAL_ONLY,
    STYLE_CONDITION_FEATURE_SET_EXEC_V2,
)

EXEC_V2_REQUIRED_CONDITION_FIELD = "effective_preference_global_vec"
GLOBAL_STYLE_DIM = 9
SCENE_STYLE_DIM = 3


def resolve_style_condition_feature_set(config: Any) -> str:
    feature_set = getattr(config, "style_condition_feature_set", STYLE_CONDITION_FEATURE_SET_GLOBAL_ONLY)
    if feature_set not in STYLE_CONDITION_FEATURE_SET_CHOICES:
        raise ValueError(
            f"Unsupported style_condition_feature_set={feature_set!r}. "
            f"Expected one of {STYLE_CONDITION_FEATURE_SET_CHOICES}."
        )
    return str(feature_set)


def validate_style_condition_args(condition_field: str, feature_set: str) -> None:
    if feature_set == STYLE_CONDITION_FEATURE_SET_EXEC_V2 and condition_field != EXEC_V2_REQUIRED_CONDITION_FIELD:
        raise ValueError(
            "style_condition_feature_set='exec_v2_effective_gap' currently requires "
            f"condition_field='{EXEC_V2_REQUIRED_CONDITION_FIELD}', got {condition_field!r}."
        )


def style_condition_dim(feature_set: str, *, base_global_dim: int = GLOBAL_STYLE_DIM) -> int:
    if feature_set == STYLE_CONDITION_FEATURE_SET_GLOBAL_ONLY:
        return int(base_global_dim)
    if feature_set == STYLE_CONDITION_FEATURE_SET_EXEC_V2:
        return int(base_global_dim + SCENE_STYLE_DIM + SCENE_STYLE_DIM)
    raise ValueError(f"Unsupported style_condition_feature_set={feature_set!r}.")


def style_condition_valid_mask(style_condition: torch.Tensor) -> torch.Tensor:
    style_condition = torch.as_tensor(style_condition, dtype=torch.float32)
    if style_condition.ndim == 1:
        return torch.any(style_condition.abs() > 1e-6).view(1)
    return torch.any(style_condition.abs() > 1e-6, dim=-1)


def build_style_condition_feature(
    base_global_condition: torch.Tensor,
    *,
    feature_set: str,
    target_scene_vec: torch.Tensor | None = None,
    effective_scene_vec: torch.Tensor | None = None,
    local_axis_gate_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the decoder-facing style condition while preserving old behavior by default."""

    base_global_condition = torch.as_tensor(base_global_condition, dtype=torch.float32)

    if feature_set == STYLE_CONDITION_FEATURE_SET_GLOBAL_ONLY:
        return base_global_condition

    if feature_set != STYLE_CONDITION_FEATURE_SET_EXEC_V2:
        raise ValueError(f"Unsupported style_condition_feature_set={feature_set!r}.")

    if target_scene_vec is None or effective_scene_vec is None or local_axis_gate_values is None:
        raise ValueError(
            "exec_v2_effective_gap requires target_scene_vec, effective_scene_vec, and local_axis_gate_values."
        )

    target_scene_vec = torch.as_tensor(target_scene_vec, dtype=torch.float32, device=base_global_condition.device)
    effective_scene_vec = torch.as_tensor(
        effective_scene_vec,
        dtype=torch.float32,
        device=base_global_condition.device,
    )
    local_axis_gate_values = torch.as_tensor(
        local_axis_gate_values,
        dtype=torch.float32,
        device=base_global_condition.device,
    )

    residual_scene_gap = target_scene_vec - effective_scene_vec
    return torch.cat([base_global_condition, residual_scene_gap, local_axis_gate_values], dim=-1)

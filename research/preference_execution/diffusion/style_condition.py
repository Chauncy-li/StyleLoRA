"""Shared style-condition builders for preference-conditioned diffusion."""

from __future__ import annotations

from typing import Any

import torch

from research.preference_execution.diffusion.phase_condition import (
    PHASE_STYLE_COUNT,
    PHASE_STYLE_GLOBAL_DIM,
    build_batch_phasewise_effective_global_condition,
    build_batch_phase_time_mask,
)
from research.preference_execution.diffusion.temporal_condition import (
    TEMPORAL_STAGE_COUNT,
    TEMPORAL_STYLE_GLOBAL_DIM,
    build_batch_two_stage_time_mask,
)

STYLE_CONDITION_FEATURE_SET_GLOBAL_ONLY = "global_only"
STYLE_CONDITION_FEATURE_SET_EXEC_V2 = "exec_v2_effective_gap"
STYLE_CONDITION_FEATURE_SET_PHASEWISE_EXEC_V1 = "phasewise_exec_v1"
STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1 = "two_stage_exec_v1"
STYLE_CONDITION_FEATURE_SET_CHOICES = (
    STYLE_CONDITION_FEATURE_SET_GLOBAL_ONLY,
    STYLE_CONDITION_FEATURE_SET_EXEC_V2,
    STYLE_CONDITION_FEATURE_SET_PHASEWISE_EXEC_V1,
    STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1,
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
    if condition_field == "style_value_condition" and feature_set != STYLE_CONDITION_FEATURE_SET_GLOBAL_ONLY:
        raise ValueError(
            "V6 direct-axis style_value_condition currently uses the explicit global-only layout. "
            "The Preference Axis Router consumes that 12D layout internally; it does not reuse "
            "the old expanded effective-preference feature sets."
        )
    if feature_set in (
        STYLE_CONDITION_FEATURE_SET_EXEC_V2,
        STYLE_CONDITION_FEATURE_SET_PHASEWISE_EXEC_V1,
        STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1,
    ) and condition_field != EXEC_V2_REQUIRED_CONDITION_FIELD:
        raise ValueError(
            "phase-aware effective-condition feature sets currently require "
            f"condition_field='{EXEC_V2_REQUIRED_CONDITION_FIELD}', got {condition_field!r}."
        )


def global_style_condition_dim(feature_set: str, *, base_global_dim: int = GLOBAL_STYLE_DIM) -> int:
    if feature_set == STYLE_CONDITION_FEATURE_SET_GLOBAL_ONLY:
        return int(base_global_dim)
    if feature_set in (
        STYLE_CONDITION_FEATURE_SET_EXEC_V2,
        STYLE_CONDITION_FEATURE_SET_PHASEWISE_EXEC_V1,
    ):
        return int(base_global_dim + SCENE_STYLE_DIM + SCENE_STYLE_DIM)
    if feature_set == STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1:
        return int(base_global_dim * 3 + SCENE_STYLE_DIM * 2)
    raise ValueError(f"Unsupported style_condition_feature_set={feature_set!r}.")


def phase_style_num_phases(feature_set: str) -> int:
    if feature_set == STYLE_CONDITION_FEATURE_SET_PHASEWISE_EXEC_V1:
        return int(PHASE_STYLE_COUNT)
    if feature_set == STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1:
        return int(TEMPORAL_STAGE_COUNT)
    return 0


def phase_style_condition_dim(feature_set: str) -> int:
    if feature_set == STYLE_CONDITION_FEATURE_SET_PHASEWISE_EXEC_V1:
        return int(PHASE_STYLE_GLOBAL_DIM)
    if feature_set == STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1:
        return int(TEMPORAL_STYLE_GLOBAL_DIM)
    return 0


def phase_style_flat_dim(feature_set: str) -> int:
    return int(phase_style_num_phases(feature_set) * phase_style_condition_dim(feature_set))


def style_condition_dim(feature_set: str, *, base_global_dim: int = GLOBAL_STYLE_DIM) -> int:
    global_dim = global_style_condition_dim(feature_set, base_global_dim=base_global_dim)
    if feature_set in (
        STYLE_CONDITION_FEATURE_SET_GLOBAL_ONLY,
        STYLE_CONDITION_FEATURE_SET_EXEC_V2,
    ):
        return int(global_dim)
    if feature_set in (
        STYLE_CONDITION_FEATURE_SET_PHASEWISE_EXEC_V1,
        STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1,
    ):
        return int(global_dim + phase_style_flat_dim(feature_set))
    raise ValueError(f"Unsupported style_condition_feature_set={feature_set!r}.")


def use_temporal_style_gate(feature_set: str) -> bool:
    return feature_set == STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1


def build_style_phase_time_mask(
    feature_set: str,
    scene_buckets: list[str] | tuple[str, ...] | str,
    future_len: int,
    *,
    include_current: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    two_stage_split_ratio: float = 0.45,
    two_stage_transition_ratio: float = 0.18,
) -> torch.Tensor | None:
    if feature_set == STYLE_CONDITION_FEATURE_SET_PHASEWISE_EXEC_V1:
        return build_batch_phase_time_mask(
            scene_buckets,
            future_len=future_len,
            include_current=include_current,
            device=device,
            dtype=dtype,
        )
    if feature_set == STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1:
        return build_batch_two_stage_time_mask(
            scene_buckets,
            future_len=future_len,
            include_current=include_current,
            device=device,
            dtype=dtype,
            split_ratio=two_stage_split_ratio,
            transition_ratio=two_stage_transition_ratio,
        )
    return None


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
    safe_scene_vec: torch.Tensor | None = None,
    effective_scene_vec: torch.Tensor | None = None,
    local_axis_gate_values: torch.Tensor | None = None,
    target_global_vec: torch.Tensor | None = None,
    safe_global_vec: torch.Tensor | None = None,
    scene_buckets: list[str] | tuple[str, ...] | str | None = None,
) -> torch.Tensor:
    """Build the decoder-facing style condition while preserving old behavior by default."""

    base_global_condition = torch.as_tensor(base_global_condition, dtype=torch.float32)

    if feature_set == STYLE_CONDITION_FEATURE_SET_GLOBAL_ONLY:
        return base_global_condition

    if feature_set not in (
        STYLE_CONDITION_FEATURE_SET_EXEC_V2,
        STYLE_CONDITION_FEATURE_SET_PHASEWISE_EXEC_V1,
        STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1,
    ):
        raise ValueError(f"Unsupported style_condition_feature_set={feature_set!r}.")

    if target_scene_vec is None or effective_scene_vec is None or local_axis_gate_values is None:
        raise ValueError(
            "phase-aware effective condition features require target_scene_vec, "
            "effective_scene_vec, and local_axis_gate_values."
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
    global_condition = torch.cat([base_global_condition, residual_scene_gap, local_axis_gate_values], dim=-1)
    if feature_set == STYLE_CONDITION_FEATURE_SET_EXEC_V2:
        return global_condition

    if feature_set == STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1:
        if safe_global_vec is None or target_global_vec is None:
            raise ValueError(
                "two_stage_exec_v1 requires safe_global_vec and target_global_vec."
            )
        safe_global_vec = torch.as_tensor(
            safe_global_vec,
            dtype=torch.float32,
            device=base_global_condition.device,
        )
        target_global_vec = torch.as_tensor(
            target_global_vec,
            dtype=torch.float32,
            device=base_global_condition.device,
        )
        global_condition = torch.cat(
            [
                base_global_condition,
                safe_global_vec,
                target_global_vec,
                residual_scene_gap,
                local_axis_gate_values,
            ],
            dim=-1,
        )
        stage_seed = torch.stack([safe_global_vec, safe_global_vec], dim=-2)
        stage_seed_flat = stage_seed.reshape(*stage_seed.shape[:-2], -1)
        return torch.cat([global_condition, stage_seed_flat], dim=-1)

    if scene_buckets is None:
        raise ValueError("phasewise_exec_v1 requires scene_buckets to build phase-aware conditions.")

    phasewise_global = build_batch_phasewise_effective_global_condition(
        scene_buckets,
        effective_scene_vec,
    ).to(device=global_condition.device, dtype=global_condition.dtype)
    phasewise_flat = phasewise_global.reshape(*phasewise_global.shape[:-2], -1)
    return torch.cat([global_condition, phasewise_flat], dim=-1)

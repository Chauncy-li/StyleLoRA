"""Two-stage temporal preference helpers for controllable execution."""

from __future__ import annotations

from typing import Sequence

import torch

from research_v1.execution.interaction.schema import AXIS_GATE_ORDER

TEMPORAL_STAGE_COUNT = 2
TEMPORAL_STYLE_GLOBAL_DIM = len(AXIS_GATE_ORDER)
DEFAULT_TWO_STAGE_SPLIT_RATIO = 0.45
DEFAULT_TWO_STAGE_TRANSITION_RATIO = 0.18
DEFAULT_TWO_STAGE_FAR_RECOVERY_MIX = 0.50


def _normalize_scene_bucket_batch(scene_buckets: Sequence[str] | str, batch_size: int) -> list[str]:
    if isinstance(scene_buckets, str):
        return [scene_buckets] * max(batch_size, 1)
    normalized = [str(scene_bucket) for scene_bucket in scene_buckets]
    if not normalized:
        return ["none"] * max(batch_size, 1)
    if len(normalized) == 1 and batch_size > 1:
        return normalized * batch_size
    if len(normalized) != batch_size:
        raise ValueError(
            f"Expected {batch_size} scene buckets, got {len(normalized)}: {normalized!r}"
        )
    return normalized


def build_batch_two_stage_time_mask(
    scene_buckets: Sequence[str] | str,
    future_len: int,
    *,
    include_current: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    split_ratio: float = DEFAULT_TWO_STAGE_SPLIT_RATIO,
    transition_ratio: float = DEFAULT_TWO_STAGE_TRANSITION_RATIO,
) -> torch.Tensor:
    """Build a soft near/far temporal decomposition over the prediction horizon."""

    batch_size = 1 if isinstance(scene_buckets, str) else len(scene_buckets)
    _normalize_scene_bucket_batch(scene_buckets, batch_size)

    future_len = int(max(future_len, 0))
    total_steps = future_len + int(include_current)
    masks = torch.zeros((batch_size, TEMPORAL_STAGE_COUNT, total_steps), device=device, dtype=dtype)
    if future_len <= 0:
        if include_current:
            masks[:, 0, 0] = 1.0
        return masks

    split_ratio = float(min(max(split_ratio, 0.05), 0.95))
    transition_ratio = float(min(max(transition_ratio, 1e-3), 0.90))
    transition_half = transition_ratio * 0.5

    timeline = torch.linspace(0.0, 1.0, steps=future_len, device=device, dtype=dtype)
    left = split_ratio - transition_half
    right = split_ratio + transition_half

    near_future = torch.ones_like(timeline)
    if right > left:
        falling = (right - timeline) / max(right - left, 1e-6)
        near_future = torch.where(
            timeline <= left,
            torch.ones_like(timeline),
            torch.where(
                timeline >= right,
                torch.zeros_like(timeline),
                torch.clamp(falling, min=0.0, max=1.0),
            ),
        )
    far_future = 1.0 - near_future

    future_offset = int(include_current)
    if include_current:
        masks[:, 0, 0] = 1.0
    masks[:, 0, future_offset:] = near_future
    masks[:, 1, future_offset:] = far_future
    return masks


def build_two_stage_global_targets(
    safe_global_vec: torch.Tensor,
    effective_global_vec: torch.Tensor,
    *,
    far_recovery_mix: float = DEFAULT_TWO_STAGE_FAR_RECOVERY_MIX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return explicit near/far supervision targets for two-stage execution.

    Near-stage target stays anchored to the immediately executable preference.
    Far-stage target recovers part of the clipped scene-realizable preference.
    """

    safe_global_vec = torch.as_tensor(safe_global_vec, dtype=torch.float32)
    effective_global_vec = torch.as_tensor(
        effective_global_vec,
        dtype=torch.float32,
        device=safe_global_vec.device,
    )
    far_recovery_mix = float(min(max(far_recovery_mix, 0.0), 1.0))
    near_target = effective_global_vec
    far_target = effective_global_vec + far_recovery_mix * (safe_global_vec - effective_global_vec)
    return near_target, far_target


def build_two_stage_gate_targets(
    safe_global_vec: torch.Tensor,
    near_target_global_vec: torch.Tensor,
    far_target_global_vec: torch.Tensor,
    *,
    min_safe_value: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project explicit near/far preference targets into gate-space targets.

    Because the temporal branch uses `safe_global_vec` as the shared seed and
    predicts stage-wise multiplicative gates, the ideal gate targets are the
    ratio between desired stage conditions and the safe seed.
    """

    safe_global_vec = torch.as_tensor(safe_global_vec, dtype=torch.float32)
    near_target_global_vec = torch.as_tensor(
        near_target_global_vec,
        dtype=torch.float32,
        device=safe_global_vec.device,
    )
    far_target_global_vec = torch.as_tensor(
        far_target_global_vec,
        dtype=torch.float32,
        device=safe_global_vec.device,
    )
    valid_mask = safe_global_vec.abs() > float(min_safe_value)
    denom = torch.where(
        valid_mask,
        safe_global_vec.abs().clamp_min(float(min_safe_value)),
        torch.ones_like(safe_global_vec),
    )
    near_gate_target = torch.where(
        valid_mask,
        torch.clamp(near_target_global_vec / denom, min=0.0, max=1.0),
        torch.zeros_like(safe_global_vec),
    )
    far_gate_target = torch.where(
        valid_mask,
        torch.clamp(far_target_global_vec / denom, min=0.0, max=1.0),
        torch.zeros_like(safe_global_vec),
    )
    return near_gate_target, far_gate_target, valid_mask

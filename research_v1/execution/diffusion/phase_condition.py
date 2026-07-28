"""Scene-specific phase-wise effective preference helpers."""

from __future__ import annotations

from typing import Sequence

import torch

from research_v1.execution.interaction.schema import AXIS_GATE_ORDER
from research_v1.scene_data.schema import style_axis_names_for_scene

PHASE_STYLE_COUNT = 3
PHASE_STYLE_GLOBAL_DIM = len(AXIS_GATE_ORDER)

SCENE_PHASE_NAMES = {
    "straight_free_drive": ("cruise", "accelerate", "stabilize"),
    "straight_car_follow": ("approach", "brake_response", "recover"),
    "straight_lane_change": ("prepare", "commit", "settle"),
}

# Each column keeps an average gain of 1.0 so the phase-wise branch preserves
# the global effective-preference scale while emphasizing different stages.
SCENE_PHASE_AXIS_TEMPLATES = {
    "straight_free_drive": (
        (0.90, 0.65, 1.20),
        (1.25, 1.55, 0.60),
        (0.85, 0.80, 1.20),
    ),
    "straight_car_follow": (
        (1.10, 0.65, 0.85),
        (1.15, 1.60, 0.60),
        (0.75, 0.75, 1.55),
    ),
    "straight_lane_change": (
        (1.20, 0.65, 0.90),
        (1.35, 1.55, 0.55),
        (0.45, 0.80, 1.55),
    ),
}

SCENE_PHASE_TIME_RATIOS = {
    "straight_free_drive": (0.34, 0.33, 0.33),
    "straight_car_follow": (0.32, 0.36, 0.32),
    "straight_lane_change": (0.35, 0.30, 0.35),
}


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


def _phase_template(scene_bucket: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = SCENE_PHASE_AXIS_TEMPLATES.get(scene_bucket)
    if values is None:
        return torch.ones((PHASE_STYLE_COUNT, 3), device=device, dtype=dtype)
    return torch.as_tensor(values, device=device, dtype=dtype)


def _phase_counts(scene_bucket: str, future_len: int) -> list[int]:
    future_len = int(max(future_len, 0))
    if future_len <= 0:
        return [0, 0, 0]

    ratios = SCENE_PHASE_TIME_RATIOS.get(scene_bucket, (1.0 / 3.0,) * PHASE_STYLE_COUNT)
    raw_counts = [int(round(float(ratio) * future_len)) for ratio in ratios]
    diff = future_len - sum(raw_counts)
    raw_counts[-1] += diff

    if raw_counts[-1] < 0:
        deficit = -raw_counts[-1]
        raw_counts[-1] = 0
        for phase_index in range(PHASE_STYLE_COUNT - 2, -1, -1):
            take = min(deficit, max(raw_counts[phase_index], 0))
            raw_counts[phase_index] -= take
            deficit -= take
            if deficit <= 0:
                break

    if sum(raw_counts) != future_len:
        raw_counts[-1] += future_len - sum(raw_counts)
    return [max(int(count), 0) for count in raw_counts]


def build_batch_phase_time_mask(
    scene_buckets: Sequence[str] | str,
    future_len: int,
    *,
    include_current: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    batch_size = 1 if isinstance(scene_buckets, str) else len(scene_buckets)
    normalized_scene_buckets = _normalize_scene_bucket_batch(scene_buckets, batch_size)
    total_steps = int(future_len) + int(include_current)
    phase_masks = torch.zeros(
        (batch_size, PHASE_STYLE_COUNT, total_steps),
        device=device,
        dtype=dtype,
    )
    start_offset = int(include_current)
    for batch_index, scene_bucket in enumerate(normalized_scene_buckets):
        counts = _phase_counts(scene_bucket, future_len)
        cursor = start_offset
        for phase_index, count in enumerate(counts):
            next_cursor = min(cursor + count, total_steps)
            if next_cursor > cursor:
                phase_masks[batch_index, phase_index, cursor:next_cursor] = 1.0
            cursor = next_cursor
    return phase_masks


def build_batch_phasewise_effective_global_condition(
    scene_buckets: Sequence[str] | str,
    effective_scene_vec: torch.Tensor,
) -> torch.Tensor:
    effective_scene_vec = torch.as_tensor(effective_scene_vec, dtype=torch.float32)
    squeeze_output = effective_scene_vec.ndim == 1
    if squeeze_output:
        effective_scene_vec = effective_scene_vec.unsqueeze(0)

    if effective_scene_vec.ndim != 2 or effective_scene_vec.shape[-1] != 3:
        raise ValueError(
            f"Expected effective_scene_vec with shape [B, 3], got {tuple(effective_scene_vec.shape)}"
        )

    batch_size = int(effective_scene_vec.shape[0])
    normalized_scene_buckets = _normalize_scene_bucket_batch(scene_buckets, batch_size)
    phase_global = torch.zeros(
        (batch_size, PHASE_STYLE_COUNT, PHASE_STYLE_GLOBAL_DIM),
        device=effective_scene_vec.device,
        dtype=effective_scene_vec.dtype,
    )

    for batch_index, scene_bucket in enumerate(normalized_scene_buckets):
        local_phase = effective_scene_vec[batch_index].unsqueeze(0) * _phase_template(
            scene_bucket,
            device=effective_scene_vec.device,
            dtype=effective_scene_vec.dtype,
        )
        axis_names = style_axis_names_for_scene(scene_bucket)
        for axis_index, axis_name in enumerate(axis_names):
            global_index = AXIS_GATE_ORDER.index(axis_name)
            phase_global[batch_index, :, global_index] = local_phase[:, axis_index]

    return phase_global.squeeze(0) if squeeze_output else phase_global

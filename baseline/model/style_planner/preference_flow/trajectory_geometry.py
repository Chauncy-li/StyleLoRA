"""Shared physical geometry for neutral-anchored Preference Flow edits."""

from __future__ import annotations

from typing import Tuple

import torch

from baseline.model.style_planner.preference_flow.contracts import (
    PreferenceFlowContractError,
)


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value) or not bool(torch.isfinite(value).all().item()):
        raise PreferenceFlowContractError(f"{name} must be a finite torch.Tensor")


def neutral_path_tangents(
    ego_current_state: torch.Tensor,
    neutral_future: torch.Tensor,
) -> torch.Tensor:
    """Return the physical neutral-path unit tangent at every future step.

    Both the longitudinal decoder and interaction supervision use this exact
    implementation.  ``ego_current_state`` is deliberately required in raw
    physical coordinates; solver/observation-normalized current states are not
    interchangeable with it.
    """

    if (
        ego_current_state.ndim != 2
        or neutral_future.ndim != 3
        or ego_current_state.shape[0] != neutral_future.shape[0]
        or ego_current_state.shape[-1] < 4
        or neutral_future.shape[-1] < 2
    ):
        raise PreferenceFlowContractError(
            "neutral path geometry requires current [B, >=4] and future [B, T, >=2]"
        )
    if (
        ego_current_state.device != neutral_future.device
        or ego_current_state.dtype != neutral_future.dtype
    ):
        raise PreferenceFlowContractError("current and future geometry must share device/dtype")
    _require_finite("physical ego current", ego_current_state)
    _require_finite("neutral future", neutral_future)
    positions = torch.cat((ego_current_state[:, None, :2], neutral_future[..., :2]), dim=1)
    deltas = positions[:, 1:] - positions[:, :-1]
    heading = ego_current_state[:, None, 2:4].expand_as(deltas)
    fallback = torch.zeros_like(deltas)
    fallback[..., 0] = 1.0
    heading_norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    fallback = torch.where(heading_norm > 1e-6, heading / heading_norm.clamp_min(1e-6), fallback)
    delta_norm = torch.linalg.vector_norm(deltas, dim=-1, keepdim=True)
    return torch.where(delta_norm > 1e-6, deltas / delta_norm.clamp_min(1e-6), fallback)


def project_relative_xy(
    relative_xy: torch.Tensor,
    tangent: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project XY offsets onto a tangent and its signed lateral normal."""

    if relative_xy.shape[-1] != 2 or tangent.shape[-1] != 2:
        raise PreferenceFlowContractError("relative_xy and tangent must end in XY channels")
    while tangent.ndim < relative_xy.ndim:
        tangent = tangent.unsqueeze(1)
    longitudinal = (relative_xy * tangent).sum(dim=-1)
    lateral = relative_xy[..., 0] * tangent[..., 1] - relative_xy[..., 1] * tangent[..., 0]
    return longitudinal, lateral


__all__ = ["neutral_path_tangents", "project_relative_xy"]

"""Neutral-anchored interaction attention for one shared Preference Flow."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn

from baseline.model.style_planner.preference_flow.contracts import (
    PreferenceFlowContractError,
)
from baseline.model.style_planner.preference_flow.trajectory_geometry import (
    neutral_path_tangents,
    project_relative_xy,
)


@dataclass(frozen=True)
class NeutralInteractionAttentionOutput:
    """Neutral-reference attention shared by all rho endpoints at one q."""

    interaction_context: torch.Tensor  # [B, C]
    neighbor_time_attention_weights: torch.Tensor  # [B, N, T]
    null_interaction_weight: torch.Tensor  # [B]
    interaction_confidence: torch.Tensor  # [B]
    agent_valid_mask: torch.Tensor  # [B, N]


class NeutralAnchoredInteractionAttention(nn.Module):
    """Use neutral ego planning to attend to neutral neighbor trajectories.

    The module has no ``rho`` or internal preference-coordinate input.  Its
    output is therefore fixed for every rho branch that shares a neutral clean
    prediction and diffusion phase.  Neighbor order is handled entirely by
    shared token projections and symmetric weighted pooling.
    """

    geometry_dim = 6

    def __init__(
        self,
        context_dim: int,
        *,
        hidden_dim: int = 32,
        dt: float = 0.1,
        ego_length_m: float = 5.2,
        default_neighbor_length_m: float = 4.5,
        min_speed_mps: float = 1.5,
        ttc_cap_s: float = 10.0,
    ) -> None:
        super().__init__()
        if any(isinstance(value, bool) or int(value) <= 0 for value in (context_dim, hidden_dim)):
            raise PreferenceFlowContractError("interaction context_dim and hidden_dim must be positive")
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise PreferenceFlowContractError("interaction dt must be finite and positive")
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.dt = float(dt)
        self.ego_length_m = float(ego_length_m)
        self.default_neighbor_length_m = float(default_neighbor_length_m)
        self.min_speed_mps = float(min_speed_mps)
        self.ttc_cap_s = float(ttc_cap_s)

        self.query_projection = nn.Linear(10, self.hidden_dim)
        self.key_projection = nn.Linear(4 + self.geometry_dim, self.hidden_dim)
        self.value_projection = nn.Linear(4 + self.geometry_dim, self.context_dim)
        self.geometry_bias = nn.Sequential(
            nn.Linear(self.geometry_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, 1)
        )
        nn.init.zeros_(self.geometry_bias[-1].weight)
        nn.init.zeros_(self.geometry_bias[-1].bias)
        self.null_key = nn.Parameter(torch.zeros(self.hidden_dim))
        self.null_value = nn.Parameter(torch.zeros(self.context_dim))
        self.null_bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _coordinate(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(value) or value.device != reference.device or value.dtype != reference.dtype:
            raise PreferenceFlowContractError(f"{name} must match attention device/dtype")
        batch_size = int(reference.shape[0])
        if value.ndim == 1 and int(value.shape[0]) == batch_size:
            result = value
        elif tuple(value.shape) == (batch_size, 1):
            result = value[:, 0]
        else:
            raise PreferenceFlowContractError(f"{name} must have shape [B] or [B, 1]")
        if not bool(torch.isfinite(result).all().item()):
            raise PreferenceFlowContractError(f"{name} must be finite")
        return result

    @staticmethod
    def _validate(
        neutral_ego_future: torch.Tensor,
        neutral_neighbor_future: torch.Tensor,
        ego_current_state: torch.Tensor,
        neighbor_current_state: torch.Tensor,
        agent_valid_mask: torch.Tensor,
    ) -> None:
        if (
            neutral_ego_future.ndim != 3
            or neutral_neighbor_future.ndim != 4
            or ego_current_state.ndim != 2
            or neighbor_current_state.ndim != 3
            or agent_valid_mask.ndim != 2
            or neutral_ego_future.shape[-1] < 4
            or neutral_neighbor_future.shape[-1] < 4
            or ego_current_state.shape[-1] < 4
            or neighbor_current_state.shape[-1] < 6
        ):
            raise PreferenceFlowContractError("invalid neutral interaction-attention tensor rank")
        batch_size, steps = neutral_ego_future.shape[:2]
        if int(steps) <= 0 or int(neutral_neighbor_future.shape[1]) <= 0:
            raise PreferenceFlowContractError("interaction attention requires non-empty neighbor/time tokens")
        if (
            neutral_neighbor_future.shape[:3] != (batch_size, neighbor_current_state.shape[1], steps)
            or ego_current_state.shape[0] != batch_size
            or tuple(agent_valid_mask.shape) != tuple(neighbor_current_state.shape[:2])
        ):
            raise PreferenceFlowContractError("neutral interaction-attention batch/agent/time shapes disagree")
        tensors = (neutral_ego_future, neutral_neighbor_future, ego_current_state, neighbor_current_state)
        if any(value.device != neutral_ego_future.device or value.dtype != neutral_ego_future.dtype for value in tensors):
            raise PreferenceFlowContractError("interaction tensors must share floating device/dtype")
        if agent_valid_mask.device != neutral_ego_future.device or agent_valid_mask.dtype != torch.bool:
            raise PreferenceFlowContractError("agent_valid_mask must be bool on the interaction device")
        if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
            raise PreferenceFlowContractError("interaction inputs must be finite")

    def _geometry(
        self,
        neutral_ego_future: torch.Tensor,
        neutral_neighbor_future: torch.Tensor,
        ego_current_state: torch.Tensor,
        neighbor_current_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tangent = neutral_path_tangents(ego_current_state, neutral_ego_future)
        ego_full = torch.cat((ego_current_state[:, None, :2], neutral_ego_future[..., :2]), dim=1)
        ego_velocity = (ego_full[:, 1:] - ego_full[:, :-1]) / self.dt
        ego_longitudinal_velocity = (ego_velocity * tangent).sum(dim=-1)

        neighbor_full = torch.cat(
            (neighbor_current_state[:, :, None, :2], neutral_neighbor_future[..., :2]), dim=2
        )
        neighbor_velocity = (neighbor_full[:, :, 1:] - neighbor_full[:, :, :-1]) / self.dt
        neighbor_longitudinal_velocity = (neighbor_velocity * tangent[:, None]).sum(dim=-1)
        relative_xy = neutral_neighbor_future[..., :2] - neutral_ego_future[:, None, :, :2]
        longitudinal, lateral = project_relative_xy(relative_xy, tangent)
        if neighbor_current_state.shape[-1] > 7:
            neighbor_length = neighbor_current_state[..., 7].abs()
            neighbor_length = torch.where(
                neighbor_length > 1e-3,
                neighbor_length,
                torch.full_like(neighbor_length, self.default_neighbor_length_m),
            )
        else:
            neighbor_length = torch.full_like(longitudinal[:, :, 0], self.default_neighbor_length_m)
        gap = (longitudinal - 0.5 * (self.ego_length_m + neighbor_length[:, :, None])).clamp_min(0.0)
        closing_speed = ego_longitudinal_velocity[:, None] - neighbor_longitudinal_velocity
        thw = gap / ego_longitudinal_velocity[:, None].abs().clamp_min(self.min_speed_mps)
        ttc = (gap / closing_speed.clamp_min(1e-3)).clamp(max=self.ttc_cap_s)
        geometry = torch.stack(
            (
                (longitudinal / 50.0).clamp(-2.0, 2.0),
                (lateral / 5.0).clamp(-2.0, 2.0),
                (closing_speed / 15.0).clamp(-2.0, 2.0),
                (thw / 5.0).clamp(0.0, 2.0),
                (ttc / self.ttc_cap_s).clamp(0.0, 1.0),
                (closing_speed >= 0.25).to(dtype=longitudinal.dtype),
            ),
            dim=-1,
        )
        # A neutral geometric prior makes null interaction preferable when no
        # same-lane/front relation is present; learned terms refine it.
        prior = (
            1.25 * (longitudinal > 0.0).to(dtype=longitudinal.dtype)
            - lateral.abs() / 1.9
            - longitudinal.abs() / 100.0
        )
        return geometry, prior

    def forward(
        self,
        neutral_ego_future: torch.Tensor,
        neutral_neighbor_future: torch.Tensor,
        *,
        ego_current_state: torch.Tensor,
        neighbor_current_state: torch.Tensor,
        agent_valid_mask: torch.Tensor,
        diffusion_time: torch.Tensor,
        log_snr: torch.Tensor,
    ) -> NeutralInteractionAttentionOutput:
        self._validate(
            neutral_ego_future, neutral_neighbor_future, ego_current_state,
            neighbor_current_state, agent_valid_mask,
        )
        time = self._coordinate("diffusion_time", diffusion_time, neutral_ego_future)
        signal_to_noise = self._coordinate("log_snr", log_snr, neutral_ego_future)
        batch_size, neighbors, steps = neutral_neighbor_future.shape[:3]
        geometry, prior = self._geometry(
            neutral_ego_future, neutral_neighbor_future, ego_current_state, neighbor_current_state
        )
        ego_summary = torch.cat(
            (neutral_ego_future.mean(dim=1), neutral_ego_future[:, -1], time[:, None], signal_to_noise[:, None]),
            dim=-1,
        )
        query = self.query_projection(ego_summary)
        token_features = torch.cat((neutral_neighbor_future, geometry), dim=-1)
        keys = self.key_projection(token_features)
        values = self.value_projection(token_features)
        logits = (keys * query[:, None, None]).sum(dim=-1) / math.sqrt(float(self.hidden_dim))
        logits = logits + prior + self.geometry_bias(geometry).squeeze(-1)
        valid_time = agent_valid_mask[:, :, None].expand(batch_size, neighbors, steps)
        flat_logits = logits.reshape(batch_size, neighbors * steps)
        flat_valid = valid_time.reshape(batch_size, neighbors * steps)
        flat_logits = flat_logits.masked_fill(~flat_valid, -torch.inf)
        null_logit = (query * self.null_key).sum(dim=-1) / math.sqrt(float(self.hidden_dim)) + self.null_bias
        weights = torch.softmax(torch.cat((flat_logits, null_logit[:, None]), dim=-1), dim=-1)
        neighbor_weights = weights[:, :-1].reshape(batch_size, neighbors, steps)
        neighbor_weights = torch.where(valid_time, neighbor_weights, torch.zeros_like(neighbor_weights))
        null_weight = weights[:, -1]
        interaction_context = (
            neighbor_weights[..., None] * values
        ).sum(dim=(1, 2)) + null_weight[:, None] * self.null_value
        confidence = neighbor_weights.sum(dim=(1, 2))
        if not bool(torch.isfinite(interaction_context).all().item()):
            raise PreferenceFlowContractError("interaction attention produced non-finite context")
        return NeutralInteractionAttentionOutput(
            interaction_context=interaction_context,
            neighbor_time_attention_weights=neighbor_weights,
            null_interaction_weight=null_weight,
            interaction_confidence=confidence,
            agent_valid_mask=agent_valid_mask.detach().clone(),
        )


__all__ = ["NeutralAnchoredInteractionAttention", "NeutralInteractionAttentionOutput"]

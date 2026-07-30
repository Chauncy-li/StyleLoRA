"""Low-dimensional preference vector field, independent of trajectories/DPM."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from baseline.model.style_planner.preference_flow.config import PreferenceFlowConfig
from baseline.model.style_planner.preference_flow.contracts import (
    PreferenceFlowContractError,
)


class PreferenceVectorField(nn.Module):
    """Map ``(z, c, q, r)`` to a velocity in the same latent space.

    The field intentionally has no ``rho`` argument.  ``r`` is the current
    integration coordinate and is provided by the integrator at every Euler or
    Heun evaluation.  With ``zero_initialize_output=True``, only the final
    affine layer is zeroed, making the initial velocity exactly zero while
    retaining a trainable state/condition/time/coordinate-dependent network.
    """

    def __init__(self, config: PreferenceFlowConfig) -> None:
        super().__init__()
        if not isinstance(config, PreferenceFlowConfig):
            raise TypeError(
                "config must be PreferenceFlowConfig, got "
                f"{type(config)!r}"
            )
        self.config = config
        self.input_dim = int(config.latent_dim + config.condition_dim + 2)

        layers = []
        in_features = self.input_dim
        for _ in range(int(config.num_layers)):
            layers.append(nn.Linear(in_features, config.hidden_dim))
            layers.append(nn.SiLU())
            in_features = int(config.hidden_dim)
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_features, config.latent_dim)
        if config.zero_initialize_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def _parameter_placement(self) -> Tuple[torch.device, torch.dtype]:
        parameter = next(self.parameters())
        return parameter.device, parameter.dtype

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not bool(torch.isfinite(value).all().item()):
            raise PreferenceFlowContractError(f"{name} must contain only finite values")

    def _coordinate(
        self,
        value: torch.Tensor,
        *,
        name: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise PreferenceFlowContractError(f"{name} must be a torch.Tensor")
        if value.device != device or value.dtype != dtype:
            raise PreferenceFlowContractError(
                f"{name} must match state device/dtype; got {value.device}/{value.dtype}, "
                f"expected {device}/{dtype}"
            )
        if value.ndim == 1 and int(value.shape[0]) == batch_size:
            normalized = value
        elif value.ndim == 2 and tuple(value.shape) == (batch_size, 1):
            normalized = value[:, 0]
        else:
            raise PreferenceFlowContractError(
                f"{name} must have shape [B] or [B, 1], got {tuple(value.shape)} "
                f"for B={batch_size}"
            )
        self._require_finite(name, normalized)
        return normalized

    def _validate_inputs(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        diffusion_time: torch.Tensor,
        preference_coordinate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(state):
            raise PreferenceFlowContractError("state must be a torch.Tensor")
        if state.ndim != 2 or int(state.shape[1]) != int(self.config.latent_dim):
            raise PreferenceFlowContractError(
                "state must have shape [B, latent_dim]=[B, "
                f"{self.config.latent_dim}], got {tuple(state.shape)}"
            )
        if not state.is_floating_point():
            raise PreferenceFlowContractError("state must have a floating dtype")
        model_device, model_dtype = self._parameter_placement()
        if state.device != model_device or state.dtype != model_dtype:
            raise PreferenceFlowContractError(
                "state must match vector-field parameter device/dtype; got "
                f"{state.device}/{state.dtype}, expected {model_device}/{model_dtype}"
            )
        self._require_finite("state", state)

        if not torch.is_tensor(condition):
            raise PreferenceFlowContractError("condition must be a torch.Tensor")
        expected_condition_shape = (int(state.shape[0]), int(self.config.condition_dim))
        if tuple(condition.shape) != expected_condition_shape:
            raise PreferenceFlowContractError(
                "condition must have shape [B, condition_dim]="
                f"{expected_condition_shape}, got {tuple(condition.shape)}"
            )
        if condition.device != state.device or condition.dtype != state.dtype:
            raise PreferenceFlowContractError(
                "condition must match state device/dtype; got "
                f"{condition.device}/{condition.dtype}, expected "
                f"{state.device}/{state.dtype}"
            )
        self._require_finite("condition", condition)
        batch_size = int(state.shape[0])
        time = self._coordinate(
            diffusion_time,
            name="diffusion_time",
            batch_size=batch_size,
            device=state.device,
            dtype=state.dtype,
        )
        coordinate = self._coordinate(
            preference_coordinate,
            name="preference_coordinate",
            batch_size=batch_size,
            device=state.device,
            dtype=state.dtype,
        )
        return state, condition, time, coordinate

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        diffusion_time: torch.Tensor,
        preference_coordinate: torch.Tensor,
    ) -> torch.Tensor:
        state, condition, time, coordinate = self._validate_inputs(
            state,
            condition,
            diffusion_time,
            preference_coordinate,
        )
        network_input = torch.cat(
            (state, condition, time.unsqueeze(-1), coordinate.unsqueeze(-1)),
            dim=-1,
        )
        velocity = self.output(self.hidden(network_input))
        if (
            tuple(velocity.shape) != tuple(state.shape)
            or velocity.device != state.device
            or velocity.dtype != state.dtype
        ):
            raise RuntimeError("PreferenceVectorField violated its output tensor contract")
        self._require_finite("vector-field velocity", velocity)
        return velocity


__all__ = ["PreferenceVectorField"]

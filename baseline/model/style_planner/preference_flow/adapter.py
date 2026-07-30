"""Preference Flow clean-prediction adapter for the preference DPM stream.

The adapter is deliberately small and has one job: bridge the verified
low-dimensional Preference Flow from Step 3 to the already-verified
clean-prediction callback from Steps 1--2.  It is called only by the
preference stream, reads the aligned neutral clean prediction plus the
preference stream's live ``x_q``, and writes a residual only to the ego future
portion of the solver-facing clean prediction.

It is not a feasibility or final-trajectory post-processing module.  The
legacy Step-4 residual decoder is zero-output; the Step-5 longitudinal decoder
instead has fixed nonzero geometry and remains an exact no-op because the
zero-initialized vector field produces zero latent displacement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as functional

from baseline.model.style_planner.preference_flow.clean_prediction_editor import (
    CleanPredictionEditContext,
)
from baseline.model.style_planner.preference_flow.config import PreferenceFlowConfig
from baseline.model.style_planner.preference_flow.contracts import (
    PreferenceFlowContractError,
)
from baseline.model.style_planner.preference_flow.integrator import (
    integrate_from_neutral,
)
from baseline.model.style_planner.preference_flow.trajectory_geometry import (
    neutral_path_tangents,
)
from baseline.model.style_planner.preference_flow.vector_field import (
    PreferenceVectorField,
)


RhoValue = Union[float, int, torch.Tensor]


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise PreferenceFlowContractError(f"{name} must contain only finite values")


def _max_abs(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().abs().max().cpu().item())


def _clone_record(record: "PreferenceFlowAdapterRecord") -> "PreferenceFlowAdapterRecord":
    return replace(
        record,
        rho=record.rho.detach().clone(),
        condition=record.condition.detach().clone(),
        latent_start=record.latent_start.detach().clone(),
        latent_end=record.latent_end.detach().clone(),
        ego_future_residual=record.ego_future_residual.detach().clone(),
    )


@dataclass(frozen=True)
class PreferenceFlowAdapterRecord:
    """Detached audit record for one preference clean-prediction edit."""

    model_evaluation_index: int
    diffusion_time: float
    rho: torch.Tensor
    condition: torch.Tensor
    latent_start: torch.Tensor
    latent_end: torch.Tensor
    ego_future_residual: torch.Tensor
    ego_current_residual_abs_max: float
    non_ego_residual_abs_max: float
    clean_prediction_device: str
    clean_prediction_dtype: str


class PreferenceFlowConditionEncoder(nn.Module):
    """Pool neutral x0, live xq, and optional causal metadata into ``c_q``.

    The Step-4 form uses only the two trajectory states.  Step 5 additionally
    reserves fixed slots for ``q``, log-SNR, and router-produced task metadata.
    No future-derived style target is accepted by this module.
    """

    def __init__(
        self,
        condition_dim: int,
        *,
        task_feature_dim: int = 0,
        include_diffusion_features: bool = False,
        interaction_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(condition_dim, bool) or int(condition_dim) <= 0:
            raise PreferenceFlowContractError(
                f"condition_dim must be a positive integer, got {condition_dim!r}"
            )
        if isinstance(task_feature_dim, bool) or int(task_feature_dim) < 0:
            raise PreferenceFlowContractError(
                "task_feature_dim must be a non-negative integer, got "
                f"{task_feature_dim!r}"
            )
        if isinstance(interaction_feature_dim, bool) or int(interaction_feature_dim) < 0:
            raise PreferenceFlowContractError(
                "interaction_feature_dim must be a non-negative integer, got "
                f"{interaction_feature_dim!r}"
            )
        self.condition_dim = int(condition_dim)
        self.task_feature_dim = int(task_feature_dim)
        self.include_diffusion_features = bool(include_diffusion_features)
        self.interaction_feature_dim = int(interaction_feature_dim)
        metadata_dim = self.task_feature_dim + (
            2 if self.include_diffusion_features else 0
        ) + self.interaction_feature_dim
        trajectory_dim = self.condition_dim - metadata_dim
        if trajectory_dim <= 0:
            raise PreferenceFlowContractError(
                "condition_dim must leave at least one trajectory feature after "
                f"metadata, got condition_dim={self.condition_dim}, "
                f"metadata_dim={metadata_dim}"
            )
        self._neutral_dim = max(1, trajectory_dim // 2)
        self._preference_dim = trajectory_dim - self._neutral_dim

    @staticmethod
    def _validate_joint_tensor(name: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise PreferenceFlowContractError(f"{name} must be a torch.Tensor")
        if value.ndim != 3 or int(value.shape[0]) <= 0 or int(value.shape[1]) <= 0:
            raise PreferenceFlowContractError(
                f"{name} must have non-empty shape [B, P, D], got {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise PreferenceFlowContractError(f"{name} must have a floating dtype")
        _require_finite(name, value)

    @staticmethod
    def _coordinate(
        name: str,
        value: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if value is None:
            raise PreferenceFlowContractError(f"{name} is required by this condition encoder")
        if not torch.is_tensor(value):
            raise PreferenceFlowContractError(f"{name} must be a torch.Tensor")
        batch_size = int(reference.shape[0])
        if value.device != reference.device or value.dtype != reference.dtype:
            raise PreferenceFlowContractError(f"{name} must match trajectory device/dtype")
        if value.ndim == 1 and int(value.shape[0]) == batch_size:
            coordinate = value
        elif tuple(value.shape) == (batch_size, 1):
            coordinate = value[:, 0]
        else:
            raise PreferenceFlowContractError(
                f"{name} must have shape [B] or [B, 1], got {tuple(value.shape)}"
            )
        _require_finite(name, coordinate)
        return coordinate

    def forward(
        self,
        neutral_clean_prediction: torch.Tensor,
        preference_current_state: torch.Tensor,
        *,
        diffusion_time: Optional[torch.Tensor] = None,
        log_snr: Optional[torch.Tensor] = None,
        task_features: Optional[torch.Tensor] = None,
        interaction_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_joint_tensor("neutral_clean_prediction", neutral_clean_prediction)
        self._validate_joint_tensor("preference_current_state", preference_current_state)
        if tuple(neutral_clean_prediction.shape) != tuple(preference_current_state.shape):
            raise PreferenceFlowContractError(
                "neutral clean prediction and preference current state must share "
                "shape; got "
                f"{tuple(neutral_clean_prediction.shape)} and "
                f"{tuple(preference_current_state.shape)}"
            )
        if (
            neutral_clean_prediction.device != preference_current_state.device
            or neutral_clean_prediction.dtype != preference_current_state.dtype
        ):
            raise PreferenceFlowContractError(
                "neutral clean prediction and preference current state must share "
                "device/dtype"
            )

        neutral_ego = neutral_clean_prediction[:, 0, :]
        preference_ego = preference_current_state[:, 0, :]
        neutral_features = functional.adaptive_avg_pool1d(
            neutral_ego.unsqueeze(1),
            self._neutral_dim,
        ).squeeze(1)
        parts = [neutral_features]
        if self._preference_dim > 0:
            preference_features = functional.adaptive_avg_pool1d(
                preference_ego.unsqueeze(1),
                self._preference_dim,
            ).squeeze(1)
            parts.append(preference_features)
        if self.include_diffusion_features:
            time = self._coordinate(
                "diffusion_time", diffusion_time, neutral_clean_prediction
            )
            signal_to_noise = self._coordinate(
                "log_snr", log_snr, neutral_clean_prediction
            )
            parts.append(torch.stack((time, signal_to_noise), dim=-1))
        if self.task_feature_dim > 0:
            if task_features is None or not torch.is_tensor(task_features):
                raise PreferenceFlowContractError(
                    "task_features is required when task_feature_dim is positive"
                )
            expected_shape = (int(neutral_clean_prediction.shape[0]), self.task_feature_dim)
            if tuple(task_features.shape) != expected_shape:
                raise PreferenceFlowContractError(
                    "task_features must have shape [B, task_feature_dim]="
                    f"{expected_shape}, got {tuple(task_features.shape)}"
                )
            if (
                task_features.device != neutral_clean_prediction.device
                or task_features.dtype != neutral_clean_prediction.dtype
            ):
                raise PreferenceFlowContractError(
                    "task_features must match trajectory device/dtype"
                )
            _require_finite("task_features", task_features)
            parts.append(task_features)
        elif task_features is not None:
            raise PreferenceFlowContractError(
                "task_features were supplied but this condition encoder has no task slots"
            )
        if self.interaction_feature_dim > 0:
            if interaction_context is None or not torch.is_tensor(interaction_context):
                raise PreferenceFlowContractError(
                    "interaction_context is required when interaction_feature_dim is positive"
                )
            expected_shape = (int(neutral_clean_prediction.shape[0]), self.interaction_feature_dim)
            if tuple(interaction_context.shape) != expected_shape:
                raise PreferenceFlowContractError(
                    "interaction_context must have shape [B, interaction_feature_dim]="
                    f"{expected_shape}, got {tuple(interaction_context.shape)}"
                )
            if (
                interaction_context.device != neutral_clean_prediction.device
                or interaction_context.dtype != neutral_clean_prediction.dtype
            ):
                raise PreferenceFlowContractError(
                    "interaction_context must match trajectory device/dtype"
                )
            _require_finite("interaction_context", interaction_context)
            parts.append(interaction_context)
        elif interaction_context is not None:
            raise PreferenceFlowContractError(
                "interaction_context was supplied but this condition encoder has no interaction slots"
            )
        condition = torch.cat(parts, dim=-1)
        if tuple(condition.shape) != (
            int(neutral_clean_prediction.shape[0]),
            self.condition_dim,
        ):
            raise RuntimeError("PreferenceFlowConditionEncoder violated its output shape")
        if (
            condition.device != neutral_clean_prediction.device
            or condition.dtype != neutral_clean_prediction.dtype
        ):
            raise RuntimeError("PreferenceFlowConditionEncoder changed device or dtype")
        _require_finite("PreferenceFlow condition", condition)
        return condition


class EgoTrajectoryResidualDecoder(nn.Module):
    """Decode a latent displacement into an ego-only future trajectory residual.

    A four-component amplitude is projected from the 8-D latent displacement
    and expanded over the future horizon with a deterministic zero-at-current
    temporal ramp.  The current pose is therefore never edited.  The final
    affine projection is zero-initialized by default, making the bridge safe
    before any future training stage.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        state_dim: int = 4,
        zero_initialize_output: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (("latent_dim", latent_dim), ("state_dim", state_dim)):
            if isinstance(value, bool) or int(value) <= 0:
                raise PreferenceFlowContractError(
                    f"{name} must be a positive integer, got {value!r}"
                )
        self.latent_dim = int(latent_dim)
        self.state_dim = int(state_dim)
        self.projection = nn.Linear(self.latent_dim, self.state_dim)
        if bool(zero_initialize_output):
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        latent_displacement: torch.Tensor,
        *,
        future_steps: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(latent_displacement):
            raise PreferenceFlowContractError("latent_displacement must be a torch.Tensor")
        if tuple(latent_displacement.shape[1:]) != (self.latent_dim,):
            raise PreferenceFlowContractError(
                "latent_displacement must have shape [B, latent_dim]=[B, "
                f"{self.latent_dim}], got {tuple(latent_displacement.shape)}"
            )
        if int(latent_displacement.shape[0]) <= 0:
            raise PreferenceFlowContractError("latent_displacement batch must be non-empty")
        if isinstance(future_steps, bool) or int(future_steps) <= 0:
            raise PreferenceFlowContractError(
                f"future_steps must be a positive integer, got {future_steps!r}"
            )
        parameter = self.projection.weight
        if (
            latent_displacement.device != parameter.device
            or latent_displacement.dtype != parameter.dtype
        ):
            raise PreferenceFlowContractError(
                "latent_displacement must match trajectory-decoder parameter "
                "device/dtype"
            )
        _require_finite("latent_displacement", latent_displacement)
        amplitude = self.projection(latent_displacement)
        ramp = torch.linspace(
            1.0 / float(future_steps),
            1.0,
            int(future_steps),
            device=amplitude.device,
            dtype=amplitude.dtype,
        )
        residual = amplitude[:, None, :] * ramp[None, :, None]
        _require_finite("ego future residual", residual)
        return residual


@dataclass(frozen=True)
class LongitudinalTrajectoryEdit:
    """Physical geometry actually used by a smooth longitudinal decode."""

    residual: torch.Tensor
    base_tangent: torch.Tensor
    physical_xy_residual: torch.Tensor


class SmoothLongitudinalTrajectoryResidualDecoder(nn.Module):
    """Map a latent displacement to a smooth, neutral-path longitudinal edit.

    The solver-facing joint state is normalized, whereas cached scene geometry
    is physical.  Callers that measure or train physical behavior pass the raw
    ego current state explicitly; the decoder then converts only the neutral
    future to physical units,
    shifts it along its own path tangent, and derives the edited heading from
    the edited path.  It has fixed nonzero geometry rather than a second
    zero-initialized learned projection: ``z=0`` is exactly identity, while
    the Jacobian from ``z`` to the residual is nonzero on the first backward
    pass.
    """

    state_dim = 4

    def __init__(
        self,
        latent_dim: int,
        *,
        ego_mean: torch.Tensor,
        ego_std: torch.Tensor,
        basis_count: int = 4,
        meters_per_latent: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(latent_dim, bool) or int(latent_dim) <= 0:
            raise PreferenceFlowContractError(
                f"latent_dim must be a positive integer, got {latent_dim!r}"
            )
        if isinstance(basis_count, bool) or int(basis_count) <= 0:
            raise PreferenceFlowContractError(
                f"basis_count must be a positive integer, got {basis_count!r}"
            )
        if not math.isfinite(float(meters_per_latent)) or float(meters_per_latent) <= 0.0:
            raise PreferenceFlowContractError("meters_per_latent must be finite and positive")
        mean = torch.as_tensor(ego_mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(ego_std, dtype=torch.float32).reshape(-1)
        if tuple(mean.shape) != (4,) or tuple(std.shape) != (4,):
            raise PreferenceFlowContractError(
                "ego_mean and ego_std must each contain the four StylePlanner state channels"
            )
        if not bool(torch.isfinite(mean).all().item()) or not bool(torch.isfinite(std).all().item()):
            raise PreferenceFlowContractError("trajectory normalization statistics must be finite")
        if bool(torch.any(std == 0).item()):
            raise PreferenceFlowContractError("trajectory normalization std must be nonzero")
        self.latent_dim = int(latent_dim)
        self.basis_count = int(basis_count)
        self.meters_per_latent = float(meters_per_latent)
        self.register_buffer("ego_mean", mean)
        self.register_buffer("ego_std", std)

        # A deterministic DCT-like matrix gives every latent coordinate a
        # nonzero path-edit Jacobian without creating a second trainable
        # zero-output bottleneck.
        latent_index = torch.arange(self.latent_dim, dtype=torch.float32)[:, None]
        basis_index = torch.arange(self.basis_count, dtype=torch.float32)[None, :]
        latent_to_basis = torch.cos(
            math.pi * (latent_index + 0.5) * basis_index / float(self.latent_dim)
        ) / math.sqrt(float(self.latent_dim))
        self.register_buffer("latent_to_basis", latent_to_basis)

    @staticmethod
    def _joint_layout(
        neutral_clean_prediction: torch.Tensor,
        current_state: torch.Tensor,
    ) -> Tuple[int, int, int]:
        if not torch.is_tensor(neutral_clean_prediction) or not torch.is_tensor(current_state):
            raise PreferenceFlowContractError(
                "neutral_clean_prediction and current_state must be torch.Tensor values"
            )
        if neutral_clean_prediction.ndim != 3 or tuple(current_state.shape) != tuple(neutral_clean_prediction.shape):
            raise PreferenceFlowContractError(
                "neutral_clean_prediction and current_state must share shape [B, P, D]"
            )
        if (
            neutral_clean_prediction.device != current_state.device
            or neutral_clean_prediction.dtype != current_state.dtype
            or not neutral_clean_prediction.is_floating_point()
        ):
            raise PreferenceFlowContractError(
                "neutral_clean_prediction and current_state must share floating device/dtype"
            )
        batch_size, agents, flattened_dim = (
            int(value) for value in neutral_clean_prediction.shape
        )
        if batch_size <= 0 or agents <= 0 or flattened_dim <= 4 or flattened_dim % 4 != 0:
            raise PreferenceFlowContractError(
                "joint state must contain current plus at least one future four-channel state"
            )
        _require_finite("neutral_clean_prediction", neutral_clean_prediction)
        _require_finite("current_state", current_state)
        return batch_size, agents, flattened_dim // 4

    @staticmethod
    def _safe_unit(vector: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        fallback_norm = torch.linalg.vector_norm(fallback, dim=-1, keepdim=True)
        default = torch.zeros_like(fallback)
        default[..., 0] = 1.0
        fallback_unit = torch.where(
            fallback_norm > 1e-6,
            fallback / fallback_norm.clamp_min(1e-6),
            default,
        )
        return torch.where(norm > 1e-6, vector / norm.clamp_min(1e-6), fallback_unit)

    @staticmethod
    def _physical_current(
        physical_ego_current_state: Optional[torch.Tensor],
        current_state: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if physical_ego_current_state is None:
            # Retain old editor behavior when an integration caller has not yet
            # supplied raw geometry. Step-5-S2 always supplies it explicitly.
            return current_state.reshape(batch_size, -1, current_state.shape[-1] // 4, 4)[:, 0, 0, :]
        if (
            not torch.is_tensor(physical_ego_current_state)
            or tuple(physical_ego_current_state.shape) != (batch_size, 4)
            or physical_ego_current_state.device != current_state.device
            or physical_ego_current_state.dtype != current_state.dtype
        ):
            raise PreferenceFlowContractError(
                "physical_ego_current_state must have shape [B, 4] and match decoder device/dtype"
            )
        _require_finite("physical_ego_current_state", physical_ego_current_state)
        return physical_ego_current_state

    def decode(
        self,
        latent_displacement: torch.Tensor,
        *,
        neutral_clean_prediction: torch.Tensor,
        current_state: torch.Tensor,
        physical_ego_current_state: Optional[torch.Tensor] = None,
    ) -> LongitudinalTrajectoryEdit:
        batch_size, _agents, state_count = self._joint_layout(
            neutral_clean_prediction, current_state
        )
        if tuple(latent_displacement.shape) != (batch_size, self.latent_dim):
            raise PreferenceFlowContractError(
                "latent_displacement must have shape [B, latent_dim]="
                f"[{batch_size}, {self.latent_dim}], got {tuple(latent_displacement.shape)}"
            )
        if (
            latent_displacement.device != neutral_clean_prediction.device
            or latent_displacement.dtype != neutral_clean_prediction.dtype
            or latent_displacement.device != self.ego_mean.device
            or latent_displacement.dtype != self.ego_mean.dtype
        ):
            raise PreferenceFlowContractError(
                "decoder inputs must match the decoder buffer device/dtype"
            )
        _require_finite("latent_displacement", latent_displacement)

        future_steps = state_count - 1
        neutral_joint = neutral_clean_prediction.reshape(batch_size, -1, state_count, 4)
        current_physical = self._physical_current(
            physical_ego_current_state, current_state, batch_size=batch_size
        )
        future_normalized = neutral_joint[:, 0, 1:, :]
        future_physical = future_normalized * self.ego_std + self.ego_mean

        time = torch.linspace(
            1.0 / float(future_steps),
            1.0,
            future_steps,
            device=latent_displacement.device,
            dtype=latent_displacement.dtype,
        )
        smooth_basis = torch.stack(
            (
                time,
                time.square(),
                time.square() * (3.0 - 2.0 * time),
                torch.sin(0.5 * math.pi * time),
            ),
            dim=0,
        )
        if self.basis_count != 4:
            # The fixed first four functions are sufficient for this research
            # stage.  Reject alternate dimensions instead of silently changing
            # the physical decoder contract.
            raise PreferenceFlowContractError("Smooth decoder currently requires basis_count=4")
        coefficients = latent_displacement @ self.latent_to_basis
        progress = (
            coefficients @ smooth_basis
        ) * float(self.meters_per_latent)

        base_xy = future_physical[..., :2]
        base_heading = future_physical[..., 2:4]
        base_tangent = neutral_path_tangents(current_physical, future_physical)
        edited_xy = base_xy + progress[..., None] * base_tangent
        edited_full_xy = torch.cat((current_physical[:, None, :2], edited_xy), dim=1)
        edited_tangent = self._safe_unit(
            edited_full_xy[:, 1:, :] - edited_full_xy[:, :-1, :], base_tangent
        )
        # Preserve the serialized heading bit-for-bit for an exact zero latent.
        # Once a longitudinal edit exists, heading follows the edited path.
        edited_heading = torch.where(
            progress.ne(0.0).unsqueeze(-1), edited_tangent, base_heading
        )
        physical_residual = torch.cat(
            (edited_xy - base_xy, edited_heading - base_heading), dim=-1
        )
        residual = physical_residual / self.ego_std
        _require_finite("smooth longitudinal ego future residual", residual)
        return LongitudinalTrajectoryEdit(
            residual=residual,
            base_tangent=base_tangent,
            physical_xy_residual=edited_xy - base_xy,
        )

    def forward(
        self,
        latent_displacement: torch.Tensor,
        *,
        neutral_clean_prediction: torch.Tensor,
        current_state: torch.Tensor,
        physical_ego_current_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decode(
            latent_displacement,
            neutral_clean_prediction=neutral_clean_prediction,
            current_state=current_state,
            physical_ego_current_state=physical_ego_current_state,
        ).residual


@dataclass(frozen=True)
class PreferenceFlowEditOutput:
    """Differentiable output of one single-phase Preference Flow edit."""

    clean_prediction: torch.Tensor
    condition: torch.Tensor
    latent_start: torch.Tensor
    latent_end: torch.Tensor
    ego_future_residual: torch.Tensor
    base_tangent: torch.Tensor
    physical_xy_residual: torch.Tensor


class PreferenceFlowTrainingAdapter(nn.Module):
    """Differentiable Step-5 bridge for one frozen-planner diffusion phase.

    It intentionally does not call the base planner or a DPM solver.  Its
    caller provides a frozen neutral x0 and the same live preference xq, then
    optimizes only this Flow-side module.  ``rho`` is used solely as the
    endpoint of :func:`integrate_from_neutral`.
    """

    def __init__(
        self,
        *,
        config: PreferenceFlowConfig,
        trajectory_decoder: SmoothLongitudinalTrajectoryResidualDecoder,
        vector_field: Optional[nn.Module] = None,
        condition_encoder: Optional[PreferenceFlowConditionEncoder] = None,
        integration_method: Optional[str] = None,
        integration_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, PreferenceFlowConfig):
            raise TypeError("config must be a PreferenceFlowConfig")
        if not isinstance(trajectory_decoder, SmoothLongitudinalTrajectoryResidualDecoder):
            raise TypeError("Step-5 training requires SmoothLongitudinalTrajectoryResidualDecoder")
        self.config = config
        self.vector_field = (
            PreferenceVectorField(config) if vector_field is None else vector_field
        )
        self.condition_encoder = (
            PreferenceFlowConditionEncoder(
                config.condition_dim,
                task_feature_dim=9,
                include_diffusion_features=True,
            )
            if condition_encoder is None
            else condition_encoder
        )
        self.trajectory_decoder = trajectory_decoder
        if not isinstance(self.vector_field, nn.Module):
            raise TypeError("vector_field must be an nn.Module")
        if not isinstance(self.condition_encoder, PreferenceFlowConditionEncoder):
            raise TypeError("condition_encoder must be a PreferenceFlowConditionEncoder")
        if int(self.condition_encoder.condition_dim) != int(config.condition_dim):
            raise PreferenceFlowContractError("condition encoder and flow config disagree")
        if int(self.trajectory_decoder.latent_dim) != int(config.latent_dim):
            raise PreferenceFlowContractError("trajectory decoder and flow config disagree")
        self._integration_method = integration_method
        self._integration_steps = integration_steps

    @staticmethod
    def _coordinate(
        name: str,
        value: torch.Tensor,
        reference: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise PreferenceFlowContractError(f"{name} must be a torch.Tensor")
        if value.device != reference.device or value.dtype != reference.dtype:
            raise PreferenceFlowContractError(f"{name} must match clean prediction device/dtype")
        if value.ndim == 1 and int(value.shape[0]) == batch_size:
            coordinate = value
        elif tuple(value.shape) == (batch_size, 1):
            coordinate = value[:, 0]
        else:
            raise PreferenceFlowContractError(
                f"{name} must have shape [B] or [B, 1], got {tuple(value.shape)}"
            )
        _require_finite(name, coordinate)
        return coordinate

    def forward(
        self,
        neutral_clean_prediction: torch.Tensor,
        preference_current_state: torch.Tensor,
        diffusion_time: torch.Tensor,
        log_snr: torch.Tensor,
        task_features: torch.Tensor,
        rho: torch.Tensor,
        interaction_context: Optional[torch.Tensor] = None,
        physical_ego_current_state: Optional[torch.Tensor] = None,
    ) -> PreferenceFlowEditOutput:
        batch_size, agents, state_count = SmoothLongitudinalTrajectoryResidualDecoder._joint_layout(
            neutral_clean_prediction, preference_current_state
        )
        time = self._coordinate(
            "diffusion_time", diffusion_time, neutral_clean_prediction, batch_size
        )
        signal_to_noise = self._coordinate(
            "log_snr", log_snr, neutral_clean_prediction, batch_size
        )
        endpoint = self._coordinate("rho", rho, neutral_clean_prediction, batch_size)
        condition = self.condition_encoder(
            neutral_clean_prediction,
            preference_current_state,
            diffusion_time=time,
            log_snr=signal_to_noise,
            task_features=task_features,
            interaction_context=interaction_context,
        )
        latent_start = torch.zeros(
            (batch_size, int(self.config.latent_dim)),
            device=neutral_clean_prediction.device,
            dtype=neutral_clean_prediction.dtype,
        )
        latent_end = integrate_from_neutral(
            self.vector_field,
            latent_start,
            condition,
            time,
            endpoint,
            num_steps=self._integration_steps,
            method=self._integration_method,
            config=self.config,
        )
        decoded = self.trajectory_decoder.decode(
            latent_end - latent_start,
            neutral_clean_prediction=neutral_clean_prediction,
            current_state=preference_current_state,
            physical_ego_current_state=physical_ego_current_state,
        )
        residual = decoded.residual
        if tuple(residual.shape) != (batch_size, state_count - 1, 4):
            raise RuntimeError("smooth trajectory decoder violated the ego-future contract")
        edited = neutral_clean_prediction.contiguous().clone()
        joint_trajectory = edited.reshape(batch_size, agents, state_count, 4)
        joint_trajectory[:, 0, 1:, :] = joint_trajectory[:, 0, 1:, :] + residual
        return PreferenceFlowEditOutput(
            clean_prediction=edited,
            condition=condition,
            latent_start=latent_start,
            latent_end=latent_end,
            ego_future_residual=residual,
            base_tangent=decoded.base_tangent,
            physical_xy_residual=decoded.physical_xy_residual,
        )


class PreferenceFlowCleanPredictionEditor(nn.Module):
    """Install Preference Flow as a preference-stream clean-prediction editor.

    ``rho`` is accepted only here as the integrator endpoint.  The vector field
    is called exclusively by :func:`integrate_from_neutral`, whose fourth input
    is the evolving internal coordinate ``r``.  The editor neither sees nor
    writes a final denormalized trajectory.
    """

    def __init__(
        self,
        *,
        rho: RhoValue,
        config: Optional[PreferenceFlowConfig] = None,
        vector_field: Optional[nn.Module] = None,
        condition_encoder: Optional[PreferenceFlowConditionEncoder] = None,
        trajectory_decoder: Optional[nn.Module] = None,
        integration_method: Optional[str] = None,
        integration_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = PreferenceFlowConfig() if config is None else config
        if not isinstance(self.config, PreferenceFlowConfig):
            raise TypeError(
                "config must be PreferenceFlowConfig or None, got "
                f"{type(self.config)!r}"
            )
        self.vector_field = (
            PreferenceVectorField(self.config) if vector_field is None else vector_field
        )
        if not isinstance(self.vector_field, nn.Module):
            raise TypeError("vector_field must be an nn.Module")
        field_config = getattr(self.vector_field, "config", None)
        if field_config is not None:
            if not isinstance(field_config, PreferenceFlowConfig):
                raise TypeError("vector_field.config must be a PreferenceFlowConfig")
            if (
                int(field_config.latent_dim) != int(self.config.latent_dim)
                or int(field_config.condition_dim) != int(self.config.condition_dim)
            ):
                raise PreferenceFlowContractError(
                    "vector_field configuration must match adapter latent/condition "
                    "dimensions"
                )
        self.condition_encoder = (
            PreferenceFlowConditionEncoder(self.config.condition_dim)
            if condition_encoder is None
            else condition_encoder
        )
        if not isinstance(self.condition_encoder, PreferenceFlowConditionEncoder):
            raise TypeError("condition_encoder must be a PreferenceFlowConditionEncoder")
        if int(self.condition_encoder.condition_dim) != int(self.config.condition_dim):
            raise PreferenceFlowContractError(
                "condition_encoder condition_dim must match PreferenceFlowConfig"
            )
        self.trajectory_decoder = (
            EgoTrajectoryResidualDecoder(
                self.config.latent_dim,
                zero_initialize_output=self.config.zero_initialize_output,
            )
            if trajectory_decoder is None
            else trajectory_decoder
        )
        if not isinstance(
            self.trajectory_decoder,
            (EgoTrajectoryResidualDecoder, SmoothLongitudinalTrajectoryResidualDecoder),
        ):
            raise TypeError(
                "trajectory_decoder must be an EgoTrajectoryResidualDecoder or "
                "SmoothLongitudinalTrajectoryResidualDecoder"
            )
        if int(self.trajectory_decoder.latent_dim) != int(self.config.latent_dim):
            raise PreferenceFlowContractError(
                "trajectory_decoder latent_dim must match PreferenceFlowConfig"
            )
        if int(self.trajectory_decoder.state_dim) != 4:
            raise PreferenceFlowContractError(
                "Step-4 StylePlanner adapter requires a four-component trajectory state"
            )
        if integration_method is not None and not isinstance(integration_method, str):
            raise PreferenceFlowContractError("integration_method must be a string or None")
        if integration_steps is not None and (
            isinstance(integration_steps, bool) or int(integration_steps) <= 0
        ):
            raise PreferenceFlowContractError(
                "integration_steps must be a positive integer or None"
            )
        self._rho = rho
        self._integration_method = integration_method
        self._integration_steps = integration_steps
        self._records: List[PreferenceFlowAdapterRecord] = []

    def reset(self) -> None:
        """Clear diagnostics before a fresh preference DPM rollout."""

        self._records.clear()

    @staticmethod
    def _joint_layout(clean_prediction: torch.Tensor) -> Tuple[int, int, int]:
        if not torch.is_tensor(clean_prediction):
            raise PreferenceFlowContractError("clean_prediction must be a torch.Tensor")
        if clean_prediction.ndim != 3:
            raise PreferenceFlowContractError(
                "clean_prediction must have shape [B, P, flattened_states], got "
                f"{tuple(clean_prediction.shape)}"
            )
        if not clean_prediction.is_floating_point():
            raise PreferenceFlowContractError("clean_prediction must have a floating dtype")
        batch_size, agents, flattened_dim = (int(value) for value in clean_prediction.shape)
        if batch_size <= 0 or agents <= 0 or flattened_dim <= 4 or flattened_dim % 4 != 0:
            raise PreferenceFlowContractError(
                "clean_prediction must contain ego/current plus at least one future "
                "four-component state; got "
                f"{tuple(clean_prediction.shape)}"
            )
        _require_finite("clean_prediction", clean_prediction)
        return batch_size, agents, flattened_dim // 4

    @staticmethod
    def _require_matching_tensor(
        name: str,
        value: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        if not torch.is_tensor(value):
            raise PreferenceFlowContractError(f"{name} must be a torch.Tensor")
        if tuple(value.shape) != tuple(reference.shape):
            raise PreferenceFlowContractError(
                f"{name} must match clean_prediction shape {tuple(reference.shape)}, "
                f"got {tuple(value.shape)}"
            )
        if value.device != reference.device or value.dtype != reference.dtype:
            raise PreferenceFlowContractError(
                f"{name} must match clean_prediction device/dtype"
            )
        _require_finite(name, value)

    @staticmethod
    def _require_matching_coordinate(
        name: str,
        value: torch.Tensor,
        reference: torch.Tensor,
        *,
        batch_size: int,
    ) -> None:
        if not torch.is_tensor(value):
            raise PreferenceFlowContractError(f"{name} must be a torch.Tensor")
        if value.device != reference.device or value.dtype != reference.dtype:
            raise PreferenceFlowContractError(
                f"{name} must match clean_prediction device/dtype"
            )
        if value.ndim != 1 or int(value.shape[0]) != batch_size:
            raise PreferenceFlowContractError(
                f"{name} must have shape [B]=[{batch_size}], got {tuple(value.shape)}"
            )
        _require_finite(name, value)

    def _validate_context(
        self,
        clean_prediction: torch.Tensor,
        context: CleanPredictionEditContext,
        *,
        batch_size: int,
    ) -> None:
        if not isinstance(context, CleanPredictionEditContext):
            raise TypeError(
                "context must be CleanPredictionEditContext, got "
                f"{type(context)!r}"
            )
        if context.stream_name != "preference":
            raise PreferenceFlowContractError(
                "PreferenceFlowCleanPredictionEditor is valid only for the "
                "preference stream"
            )
        if context.neutral_record is None:
            raise PreferenceFlowContractError(
                "preference Flow editor requires the aligned neutral reference"
            )
        if int(context.model_evaluation_index) < 0:
            raise PreferenceFlowContractError("model_evaluation_index must be non-negative")
        self._require_matching_tensor(
            "context.current_state",
            context.current_state,
            clean_prediction,
        )
        self._require_matching_tensor(
            "neutral_record.clean_prediction",
            context.neutral_record.clean_prediction,
            clean_prediction,
        )
        self._require_matching_coordinate(
            "context.diffusion_time",
            context.diffusion_time,
            clean_prediction,
            batch_size=batch_size,
        )
        self._require_matching_coordinate(
            "context.log_snr",
            context.log_snr,
            clean_prediction,
            batch_size=batch_size,
        )
        neutral = context.neutral_record
        if int(neutral.model_evaluation_index) != int(context.model_evaluation_index):
            raise PreferenceFlowContractError("neutral reference evaluation index mismatch")
        if bool(neutral.is_terminal_denoise) != bool(context.is_terminal_denoise):
            raise PreferenceFlowContractError("neutral reference terminal flag mismatch")
        if not torch.equal(neutral.diffusion_time, context.diffusion_time):
            raise PreferenceFlowContractError("neutral reference diffusion time mismatch")
        if not torch.equal(neutral.log_snr, context.log_snr):
            raise PreferenceFlowContractError("neutral reference log-SNR mismatch")

    def _rho_tensor(
        self,
        *,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if torch.is_tensor(self._rho):
            rho = self._rho
            if not rho.is_floating_point():
                raise PreferenceFlowContractError("rho tensor must have a floating dtype")
            if rho.device != reference.device or rho.dtype != reference.dtype:
                raise PreferenceFlowContractError(
                    "rho tensor must match clean_prediction device/dtype"
                )
            rho = rho.reshape(-1)
        elif isinstance(self._rho, Real) and not isinstance(self._rho, bool):
            rho_value = float(self._rho)
            if not math.isfinite(rho_value):
                raise PreferenceFlowContractError("rho must be finite")
            rho = torch.full(
                (batch_size,),
                rho_value,
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            raise PreferenceFlowContractError(
                "rho must be a finite scalar or a floating torch.Tensor"
            )
        if rho.numel() == 1 and batch_size != 1:
            rho = rho.expand(batch_size)
        if rho.ndim != 1 or int(rho.shape[0]) != batch_size:
            raise PreferenceFlowContractError(
                f"rho must be scalar or have shape [B]=[{batch_size}], got {tuple(rho.shape)}"
            )
        _require_finite("rho", rho)
        return rho

    def forward(
        self,
        clean_prediction: torch.Tensor,
        context: CleanPredictionEditContext,
    ) -> torch.Tensor:
        batch_size, agents, state_count = self._joint_layout(clean_prediction)
        self._validate_context(
            clean_prediction,
            context,
            batch_size=batch_size,
        )
        rho = self._rho_tensor(batch_size=batch_size, reference=clean_prediction)
        assert context.neutral_record is not None  # narrowed after validation
        condition = self.condition_encoder(
            context.neutral_record.clean_prediction,
            context.current_state,
        )
        initial_latent = torch.zeros(
            (batch_size, int(self.config.latent_dim)),
            device=clean_prediction.device,
            dtype=clean_prediction.dtype,
        )
        integrated_latent = integrate_from_neutral(
            self.vector_field,
            initial_latent,
            condition,
            context.diffusion_time,
            rho,
            num_steps=self._integration_steps,
            method=self._integration_method,
            config=self.config,
        )
        latent_displacement = integrated_latent - initial_latent
        if isinstance(
            self.trajectory_decoder, SmoothLongitudinalTrajectoryResidualDecoder
        ):
            ego_future_residual = self.trajectory_decoder(
                latent_displacement,
                neutral_clean_prediction=context.neutral_record.clean_prediction,
                current_state=context.current_state,
            )
        else:
            ego_future_residual = self.trajectory_decoder(
                latent_displacement,
                future_steps=state_count - 1,
            )
        if tuple(ego_future_residual.shape) != (batch_size, state_count - 1, 4):
            raise RuntimeError("trajectory decoder violated the ego-future shape contract")
        if (
            ego_future_residual.device != clean_prediction.device
            or ego_future_residual.dtype != clean_prediction.dtype
        ):
            raise PreferenceFlowContractError(
                "trajectory decoder changed clean-prediction device/dtype"
            )

        # Make the layout explicit before reshaping the joint state.  The DPM
        # path currently supplies contiguous x0 tensors, but an editor must not
        # silently lose its write if a future wrapper supplies a strided view.
        edited = clean_prediction.contiguous().clone()
        joint_trajectory = edited.reshape(batch_size, agents, state_count, 4)
        joint_trajectory[:, 0, 1:, :] = (
            joint_trajectory[:, 0, 1:, :] + ego_future_residual
        )
        direct_residual = edited - clean_prediction
        ego_current_residual = direct_residual[:, 0, :4]
        non_ego_residual = direct_residual[:, 1:, :] if agents > 1 else direct_residual[:, :0, :]
        self._records.append(
            PreferenceFlowAdapterRecord(
                model_evaluation_index=int(context.model_evaluation_index),
                diffusion_time=float(context.diffusion_time.detach()[0].cpu().item()),
                rho=rho.detach().clone(),
                condition=condition.detach().clone(),
                latent_start=initial_latent.detach().clone(),
                latent_end=integrated_latent.detach().clone(),
                ego_future_residual=ego_future_residual.detach().clone(),
                ego_current_residual_abs_max=_max_abs(ego_current_residual),
                non_ego_residual_abs_max=_max_abs(non_ego_residual),
                clean_prediction_device=str(clean_prediction.device),
                clean_prediction_dtype=str(clean_prediction.dtype),
            )
        )
        return edited

    def records(self) -> Tuple[PreferenceFlowAdapterRecord, ...]:
        """Return independent audit copies without exposing live DPM tensors."""

        return tuple(_clone_record(record) for record in self._records)

    def diagnostics(self) -> Dict[str, Any]:
        """Return compact JSON-safe diagnostics for one preference rollout."""

        return {
            "editor": "preference_flow_adapter",
            "model_evaluation_count": len(self._records),
            "calls": [
                {
                    "model_evaluation_index": int(record.model_evaluation_index),
                    "diffusion_time": float(record.diffusion_time),
                    "rho": [float(value) for value in record.rho.detach().cpu().tolist()],
                    "latent_displacement_abs_max": _max_abs(
                        record.latent_end - record.latent_start
                    ),
                    "ego_future_residual_abs_max": _max_abs(
                        record.ego_future_residual
                    ),
                    "ego_current_residual_abs_max": float(
                        record.ego_current_residual_abs_max
                    ),
                    "non_ego_residual_abs_max": float(
                        record.non_ego_residual_abs_max
                    ),
                    "clean_prediction_device": record.clean_prediction_device,
                    "clean_prediction_dtype": record.clean_prediction_dtype,
                }
                for record in self._records
            ],
        }


__all__ = [
    "EgoTrajectoryResidualDecoder",
    "LongitudinalTrajectoryEdit",
    "PreferenceFlowEditOutput",
    "PreferenceFlowAdapterRecord",
    "PreferenceFlowCleanPredictionEditor",
    "PreferenceFlowConditionEncoder",
    "PreferenceFlowTrainingAdapter",
    "SmoothLongitudinalTrajectoryResidualDecoder",
]

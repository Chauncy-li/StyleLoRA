"""Explicit Euler/Heun integration for standalone Preference Flow latents."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch

from baseline.model.style_planner.preference_flow.config import (
    DEFAULT_INTEGRATION_METHOD,
    DEFAULT_INTEGRATION_STEPS,
    INTEGRATION_METHODS,
    PreferenceFlowConfig,
)
from baseline.model.style_planner.preference_flow.contracts import (
    PreferenceFlowContractError,
)


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise PreferenceFlowContractError(f"{name} must contain only finite values")


def _coordinate(
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
            f"{name} must match initial_state device/dtype; got "
            f"{value.device}/{value.dtype}, expected {device}/{dtype}"
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
    _require_finite(name, normalized)
    return normalized


def _validate_state_and_condition(
    initial_state: torch.Tensor,
    condition: torch.Tensor,
    diffusion_time: torch.Tensor,
    *,
    config: Optional[PreferenceFlowConfig],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(initial_state):
        raise PreferenceFlowContractError("initial_state must be a torch.Tensor")
    if initial_state.ndim != 2 or int(initial_state.shape[0]) <= 0:
        raise PreferenceFlowContractError(
            "initial_state must have non-empty shape [B, K], got "
            f"{tuple(initial_state.shape)}"
        )
    if not initial_state.is_floating_point():
        raise PreferenceFlowContractError("initial_state must have a floating dtype")
    if config is not None and int(initial_state.shape[1]) != int(config.latent_dim):
        raise PreferenceFlowContractError(
            "initial_state latent dimension does not match config: "
            f"got {initial_state.shape[1]}, expected {config.latent_dim}"
        )
    _require_finite("initial_state", initial_state)
    if not torch.is_tensor(condition):
        raise PreferenceFlowContractError("condition must be a torch.Tensor")
    if condition.ndim != 2 or int(condition.shape[0]) != int(initial_state.shape[0]):
        raise PreferenceFlowContractError(
            "condition must have shape [B, C] with the same B as initial_state; "
            f"got {tuple(condition.shape)} for B={initial_state.shape[0]}"
        )
    if config is not None and int(condition.shape[1]) != int(config.condition_dim):
        raise PreferenceFlowContractError(
            "condition dimension does not match config: "
            f"got {condition.shape[1]}, expected {config.condition_dim}"
        )
    if condition.device != initial_state.device or condition.dtype != initial_state.dtype:
        raise PreferenceFlowContractError(
            "condition must match initial_state device/dtype; got "
            f"{condition.device}/{condition.dtype}, expected "
            f"{initial_state.device}/{initial_state.dtype}"
        )
    _require_finite("condition", condition)
    time = _coordinate(
        diffusion_time,
        name="diffusion_time",
        batch_size=int(initial_state.shape[0]),
        device=initial_state.device,
        dtype=initial_state.dtype,
    )
    return initial_state, condition, time


def _resolve_config(
    vector_field: Any,
    config: Optional[PreferenceFlowConfig],
) -> Optional[PreferenceFlowConfig]:
    if config is not None:
        if not isinstance(config, PreferenceFlowConfig):
            raise TypeError(
                "config must be PreferenceFlowConfig or None, got "
                f"{type(config)!r}"
            )
        return config
    inferred = getattr(vector_field, "config", None)
    if inferred is None:
        return None
    if not isinstance(inferred, PreferenceFlowConfig):
        raise TypeError(
            "vector_field.config must be PreferenceFlowConfig when present, got "
            f"{type(inferred)!r}"
        )
    return inferred


def _coordinate_bounds(
    *,
    config: Optional[PreferenceFlowConfig],
    coordinate_min: Optional[float],
    coordinate_max: Optional[float],
) -> Tuple[float, float]:
    if config is not None:
        if coordinate_min is not None or coordinate_max is not None:
            raise PreferenceFlowContractError(
                "coordinate bounds come from PreferenceFlowConfig; do not override them"
            )
        return float(config.rho_min), float(config.rho_max)
    lower = -1.0 if coordinate_min is None else float(coordinate_min)
    upper = 1.0 if coordinate_max is None else float(coordinate_max)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise PreferenceFlowContractError(
            "coordinate_min and coordinate_max must be finite with min < max"
        )
    return lower, upper


def _validate_method_and_steps(method: str, num_steps: int) -> str:
    if not isinstance(method, str):
        raise PreferenceFlowContractError(
            f"method must be one of {INTEGRATION_METHODS}, got {method!r}"
        )
    normalized_method = method.strip().lower()
    if normalized_method not in INTEGRATION_METHODS:
        raise PreferenceFlowContractError(
            f"method must be one of {INTEGRATION_METHODS}, got {method!r}"
        )
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise PreferenceFlowContractError(
            f"num_steps must be a positive integer, got {num_steps!r}"
        )
    return normalized_method


def _evaluate_vector_field(
    vector_field: Any,
    state: torch.Tensor,
    condition: torch.Tensor,
    diffusion_time: torch.Tensor,
    preference_coordinate: torch.Tensor,
) -> torch.Tensor:
    if not callable(vector_field):
        raise PreferenceFlowContractError("vector_field must be callable")
    velocity = vector_field(state, condition, diffusion_time, preference_coordinate)
    if not torch.is_tensor(velocity):
        raise PreferenceFlowContractError(
            "vector_field must return a torch.Tensor, got "
            f"{type(velocity)!r}"
        )
    if tuple(velocity.shape) != tuple(state.shape):
        raise PreferenceFlowContractError(
            "vector_field changed state shape: "
            f"expected {tuple(state.shape)}, got {tuple(velocity.shape)}"
        )
    if velocity.device != state.device or velocity.dtype != state.dtype:
        raise PreferenceFlowContractError(
            "vector_field output must match state device/dtype; got "
            f"{velocity.device}/{velocity.dtype}, expected "
            f"{state.device}/{state.dtype}"
        )
    _require_finite("vector_field output", velocity)
    return velocity


def integrate(
    vector_field: Any,
    initial_state: torch.Tensor,
    condition: torch.Tensor,
    diffusion_time: torch.Tensor,
    start_coordinate: torch.Tensor,
    end_coordinate: torch.Tensor,
    num_steps: int = DEFAULT_INTEGRATION_STEPS,
    method: str = DEFAULT_INTEGRATION_METHOD,
    *,
    config: Optional[PreferenceFlowConfig] = None,
    coordinate_min: Optional[float] = None,
    coordinate_max: Optional[float] = None,
) -> torch.Tensor:
    """Integrate ``dz/dr = V(z, c, q, r)`` over a general per-sample interval.

    The code convention follows the executable contract used by this project:
    a constant field ``V=c`` gives ``z_end = z_start + (end-start) * c``.
    A paper notation with a leading minus can equivalently define its vector
    field as the negative of this implementation's velocity.
    """

    resolved_config = _resolve_config(vector_field, config)
    method = _validate_method_and_steps(method, num_steps)
    initial_state, condition, time = _validate_state_and_condition(
        initial_state,
        condition,
        diffusion_time,
        config=resolved_config,
    )
    batch_size = int(initial_state.shape[0])
    start = _coordinate(
        start_coordinate,
        name="start_coordinate",
        batch_size=batch_size,
        device=initial_state.device,
        dtype=initial_state.dtype,
    )
    end = _coordinate(
        end_coordinate,
        name="end_coordinate",
        batch_size=batch_size,
        device=initial_state.device,
        dtype=initial_state.dtype,
    )
    lower, upper = _coordinate_bounds(
        config=resolved_config,
        coordinate_min=coordinate_min,
        coordinate_max=coordinate_max,
    )
    if bool(torch.any(start < lower).item()) or bool(torch.any(start > upper).item()):
        raise PreferenceFlowContractError(
            f"start_coordinate must lie in [{lower}, {upper}]"
        )
    if bool(torch.any(end < lower).item()) or bool(torch.any(end > upper).item()):
        raise PreferenceFlowContractError(
            f"end_coordinate must lie in [{lower}, {upper}]"
        )

    zero_interval = torch.eq(start, end)
    if bool(torch.all(zero_interval).item()):
        # Exact identity is a required semantic, not a numerical coincidence.
        return initial_state

    step_size = (end - start) / float(num_steps)
    state = initial_state
    coordinate = start
    zero_mask = zero_interval.unsqueeze(-1)
    for _ in range(num_steps):
        if method == "euler":
            velocity = _evaluate_vector_field(
                vector_field,
                state,
                condition,
                time,
                coordinate,
            )
            candidate = state + step_size.unsqueeze(-1) * velocity
        else:  # Heun / explicit trapezoidal rule.
            first_velocity = _evaluate_vector_field(
                vector_field,
                state,
                condition,
                time,
                coordinate,
            )
            next_coordinate = coordinate + step_size
            predictor = state + step_size.unsqueeze(-1) * first_velocity
            second_velocity = _evaluate_vector_field(
                vector_field,
                predictor,
                condition,
                time,
                next_coordinate,
            )
            candidate = state + 0.5 * step_size.unsqueeze(-1) * (
                first_velocity + second_velocity
            )
        _require_finite("integrated state", candidate)
        # For a mixed batch, exact-zero intervals remain bitwise initial values
        # even while other samples advance through the solver.
        state = torch.where(zero_mask, initial_state, candidate)
        coordinate = coordinate + step_size
    return state


def integrate_from_neutral(
    vector_field: Any,
    initial_state: torch.Tensor,
    condition: torch.Tensor,
    diffusion_time: torch.Tensor,
    rho: torch.Tensor,
    num_steps: Optional[int] = None,
    method: Optional[str] = None,
    *,
    config: Optional[PreferenceFlowConfig] = None,
) -> torch.Tensor:
    """Integrate from neutral coordinate zero to a user endpoint ``rho``.

    ``rho`` stays in this integrator wrapper.  The vector field receives only
    the evolving coordinate used for each numerical evaluation.
    """

    resolved_config = _resolve_config(vector_field, config)
    resolved_steps = (
        int(resolved_config.default_num_steps)
        if num_steps is None and resolved_config is not None
        else DEFAULT_INTEGRATION_STEPS if num_steps is None else num_steps
    )
    resolved_method = (
        str(resolved_config.default_method)
        if method is None and resolved_config is not None
        else DEFAULT_INTEGRATION_METHOD if method is None else method
    )
    if not torch.is_tensor(rho):
        raise PreferenceFlowContractError("rho must be a torch.Tensor")
    return integrate(
        vector_field,
        initial_state,
        condition,
        diffusion_time,
        start_coordinate=torch.zeros_like(rho),
        end_coordinate=rho,
        num_steps=resolved_steps,
        method=resolved_method,
        config=resolved_config,
    )


__all__ = ["integrate", "integrate_from_neutral"]

"""Configuration for the standalone Preference Flow latent dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import math

from baseline.model.style_planner.preference_flow.contracts import (
    PreferenceFlowContractError,
)


INTEGRATION_METHODS = ("euler", "heun")
DEFAULT_INTEGRATION_METHOD = "heun"
DEFAULT_INTEGRATION_STEPS = 2


@dataclass(frozen=True)
class PreferenceFlowConfig:
    """Static dimensions, coordinate bounds, and numerical defaults.

    ``rho_min`` and ``rho_max`` constrain integrator endpoints.  They are not
    vector-field inputs: the field only receives its current integration
    coordinate ``r``.
    """

    latent_dim: int = 8
    condition_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 3
    zero_initialize_output: bool = True
    rho_min: float = -1.0
    rho_max: float = 1.0
    default_method: str = DEFAULT_INTEGRATION_METHOD
    default_num_steps: int = DEFAULT_INTEGRATION_STEPS

    def __post_init__(self) -> None:
        for field_name in ("latent_dim", "condition_dim", "hidden_dim", "num_layers"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PreferenceFlowContractError(
                    f"{field_name} must be a positive integer, got {value!r}"
                )
        if not isinstance(self.zero_initialize_output, bool):
            raise PreferenceFlowContractError(
                "zero_initialize_output must be a bool, got "
                f"{self.zero_initialize_output!r}"
            )
        rho_min = float(self.rho_min)
        rho_max = float(self.rho_max)
        if not math.isfinite(rho_min) or not math.isfinite(rho_max) or rho_min >= rho_max:
            raise PreferenceFlowContractError(
                "rho_min and rho_max must be finite with rho_min < rho_max; "
                f"got {self.rho_min!r}, {self.rho_max!r}"
            )
        if not isinstance(self.default_method, str) or self.default_method.strip().lower() not in INTEGRATION_METHODS:
            raise PreferenceFlowContractError(
                "default_method must be one of "
                f"{INTEGRATION_METHODS}, got {self.default_method!r}"
            )
        if (
            isinstance(self.default_num_steps, bool)
            or not isinstance(self.default_num_steps, int)
            or self.default_num_steps <= 0
        ):
            raise PreferenceFlowContractError(
                "default_num_steps must be a positive integer, got "
                f"{self.default_num_steps!r}"
            )


__all__ = [
    "DEFAULT_INTEGRATION_METHOD",
    "DEFAULT_INTEGRATION_STEPS",
    "INTEGRATION_METHODS",
    "PreferenceFlowConfig",
]

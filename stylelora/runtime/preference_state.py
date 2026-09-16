"""Deterministic user-feedback updates for the continuous style coordinate."""

from __future__ import annotations

import math
from dataclasses import dataclass


_FEEDBACK_DELTAS = {
    "more_conservative": -1.0,
    "keep": 0.0,
    "more_aggressive": 1.0,
}


@dataclass
class PreferenceState:
    """Maintain one user's continuous preference coordinate in ``[-1, 1]``."""

    rho: float = 0.0
    step: float = 0.25

    def __post_init__(self) -> None:
        self.rho = float(self.rho)
        self.step = float(self.step)
        if not math.isfinite(self.rho) or not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be finite and lie in [-1, 1]")
        if not math.isfinite(self.step) or self.step <= 0.0:
            raise ValueError("feedback step must be a positive finite value")

    def update(self, feedback: str) -> dict[str, object]:
        """Apply relative feedback and return a serializable audit record."""
        command = str(feedback).strip().lower()
        if command not in _FEEDBACK_DELTAS:
            supported = ", ".join(_FEEDBACK_DELTAS)
            raise ValueError(f"unsupported feedback {feedback!r}; expected one of: {supported}")

        rho_before = self.rho
        requested_delta = self.step * _FEEDBACK_DELTAS[command]
        self.rho = max(-1.0, min(1.0, rho_before + requested_delta))
        return {
            "feedback": command,
            "rho_before": rho_before,
            "rho_after": self.rho,
            "requested_delta": requested_delta,
            "applied_delta": self.rho - rho_before,
        }

    def snapshot(self) -> dict[str, float]:
        return {"rho": self.rho, "step": self.step}

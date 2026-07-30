"""Differentiable soft metric helpers owned by the StylePlanner baseline.

These functions retain the existing numerical definitions used by
``preference_energy``.  They are colocated with their production consumer so
the baseline model does not depend on research execution packages.
"""

from __future__ import annotations

import torch


def softmin(values: torch.Tensor, temperature: float = 12.0) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.numel() == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    temperature = max(float(temperature), 1e-3)
    return -torch.logsumexp(-temperature * values, dim=0) / temperature


def soft_low_quantile(values: torch.Tensor, temperature: float = 12.0) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.numel() == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    weights = torch.softmax(-max(float(temperature), 1e-3) * values, dim=0)
    return torch.sum(weights * values)


def soft_high_quantile(values: torch.Tensor, temperature: float = 12.0) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.numel() == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    weights = torch.softmax(max(float(temperature), 1e-3) * values, dim=0)
    return torch.sum(weights * values)


def soft_first_crossing_time(
    values: torch.Tensor,
    *,
    threshold: float,
    dt: float,
    temperature: float = 20.0,
) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.numel() == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    step_index = torch.arange(values.shape[0], dtype=values.dtype, device=values.device)
    activation = torch.sigmoid(max(float(temperature), 1e-3) * (values - float(threshold)))
    if float(activation.sum().detach().cpu().item()) <= 1e-6:
        return values.new_tensor(float(max(values.shape[0] - 1, 0)) * float(dt))
    onset_step = torch.sum(activation * step_index) / activation.sum().clamp_min(1e-6)
    return onset_step * float(dt)


def safe_headway_series(
    gap_series: torch.Tensor,
    ego_speed_series: torch.Tensor,
    vmin_mps: float = 1.5,
) -> torch.Tensor:
    gap_series = torch.as_tensor(gap_series, dtype=torch.float32)
    ego_speed_series = torch.as_tensor(ego_speed_series, dtype=torch.float32)
    denom = torch.clamp(ego_speed_series.abs(), min=float(vmin_mps))
    return gap_series / denom


def safe_ttc_series(
    gap_series: torch.Tensor,
    closing_speed_series: torch.Tensor,
    ttc_cap_s: float = 10.0,
) -> torch.Tensor:
    gap_series = torch.as_tensor(gap_series, dtype=torch.float32)
    closing_speed_series = torch.as_tensor(closing_speed_series, dtype=torch.float32)
    positive_closing = torch.relu(closing_speed_series)
    ttc = gap_series / positive_closing.clamp_min(1e-3)
    return torch.clamp(ttc, min=0.0, max=float(ttc_cap_s))


__all__ = [
    "safe_headway_series",
    "safe_ttc_series",
    "soft_first_crossing_time",
    "soft_high_quantile",
    "soft_low_quantile",
    "softmin",
]

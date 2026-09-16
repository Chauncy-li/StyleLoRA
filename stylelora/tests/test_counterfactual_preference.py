from __future__ import annotations

import torch

from stylelora.lora.safety.trajectory_acceptance import cached_relative_feasibility_mask
from stylelora.scripts.build_counterfactual_preferences import (
    _continuous_prefix_masks,
    _target_magnitude,
)
from stylelora.training.conditional_preference_lora import (
    feasible_style_pair_objective,
)


def _straight(speed: float, *, steps: int = 20, dt: float = 0.1) -> torch.Tensor:
    time = torch.arange(1, steps + 1, dtype=torch.float32) * dt
    x = speed * time
    return torch.stack(
        (x, torch.zeros_like(x), torch.ones_like(x), torch.zeros_like(x)), dim=-1
    )[None]


def test_feasible_style_loss_stops_ranking_reward_after_margin() -> None:
    common = dict(
        preferred_error=torch.tensor([0.2, 0.2]),
        confidence=torch.ones(2),
        margin=0.1,
    )
    at_margin = feasible_style_pair_objective(
        rejected_error=torch.tensor([0.31, 0.31]), **common
    )
    far_beyond_margin = feasible_style_pair_objective(
        rejected_error=torch.tensor([30.0, 30.0]), **common
    )
    assert at_margin["ranking"].item() == 0.0
    assert far_beyond_margin["ranking"].item() == 0.0
    assert at_margin["anchor"].item() == far_beyond_margin["anchor"].item()


def test_feasible_style_loss_zero_confidence_is_finite() -> None:
    result = feasible_style_pair_objective(
        preferred_error=torch.ones(2),
        rejected_error=torch.ones(2),
        confidence=torch.zeros(2),
        margin=0.1,
    )
    assert torch.isfinite(result["anchor"])
    assert torch.isfinite(result["ranking"])
    assert result["anchor"].item() == 0.0
    assert result["ranking"].item() == 0.0


def test_relative_feasibility_accepts_baseline_equivalent_candidate() -> None:
    baseline = _straight(5.0)
    result = cached_relative_feasibility_mask(
        baseline.clone(), baseline, torch.zeros(1, 1, 2)
    )
    assert result.tolist() == [True]


def test_target_magnitude_is_stable_and_in_requested_grid() -> None:
    magnitudes = [0.25, 0.5, 0.75, 1.0]
    first = _target_magnitude("log:token", "high", magnitudes, 17)
    second = _target_magnitude("log:token", "high", magnitudes, 17)
    assert first == second
    assert first in magnitudes


def test_continuous_prefix_rejects_higher_strength_after_a_failure() -> None:
    magnitudes = [0.25, 0.5, 0.75, 1.0]
    masks = _continuous_prefix_masks(
        {
            0.25: torch.tensor([True, True]),
            0.5: torch.tensor([False, True]),
            0.75: torch.tensor([True, False]),
            1.0: torch.tensor([True, True]),
        },
        magnitudes,
    )
    assert masks[0.25].tolist() == [True, True]
    assert masks[0.5].tolist() == [False, True]
    assert masks[0.75].tolist() == [False, False]
    assert masks[1.0].tolist() == [False, False]

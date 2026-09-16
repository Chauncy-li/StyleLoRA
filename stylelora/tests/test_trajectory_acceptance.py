from __future__ import annotations

import torch

from stylelora.lora.safety.trajectory_acceptance import (
    cached_feasibility_mask,
    cached_hard_feasibility_mask,
    cached_relative_feasibility_mask,
    differentiable_relative_feasibility_penalty,
)
from stylelora.lora.safety.trajectory_repair import (
    batched_three_disc_min_clearance,
    candidate_scale_grid,
    compose_baseline_anchored_ego,
    compose_interpolated_ego_batch,
)


def _straight(speed: float, *, steps: int = 20, dt: float = 0.1) -> torch.Tensor:
    time = torch.arange(1, steps + 1, dtype=torch.float32) * dt
    x = speed * time
    return torch.stack((x, torch.zeros_like(x), torch.ones_like(x), torch.zeros_like(x)), dim=-1)[None]


def test_cached_feasibility_accepts_baseline_equivalent_candidate() -> None:
    baseline = _straight(5.0)
    mask = cached_feasibility_mask(baseline.clone(), baseline, torch.zeros(1, 1, 2))
    assert mask.tolist() == [True]


def test_cached_feasibility_rejects_excessive_progress_loss() -> None:
    baseline = _straight(5.0)
    stopped = _straight(0.0)
    mask = cached_feasibility_mask(
        stopped,
        baseline,
        torch.zeros(1, 1, 2),
        max_progress_loss_m=0.5,
    )
    assert mask.tolist() == [False]


def test_cached_hard_feasibility_does_not_treat_progress_loss_as_physical_failure() -> None:
    stopped = _straight(0.0)
    mask = cached_hard_feasibility_mask(stopped, torch.zeros(1, 1, 2))
    assert mask.tolist() == [True]


def test_cached_feasibility_rejects_non_finite_candidate() -> None:
    baseline = _straight(5.0)
    candidate = baseline.clone()
    candidate[:, 3, 0] = float("nan")
    mask = cached_feasibility_mask(candidate, baseline, torch.zeros(1, 1, 2))
    assert mask.tolist() == [False]


def test_relative_feasibility_does_not_treat_longitudinal_shift_as_lateral_drift() -> None:
    baseline = _straight(5.0)
    candidate = baseline.clone()
    candidate[..., 0] += 3.0
    mask = cached_relative_feasibility_mask(
        candidate,
        baseline,
        torch.zeros(1, 1, 2),
        max_mean_accel_degradation=1e6,
        max_mean_jerk_degradation=1e6,
        max_progress_loss_m=1e6,
        max_mean_lateral_deviation_m=0.01,
    )
    assert mask.tolist() == [True]


def test_relative_feasibility_rejects_excessive_lateral_drift() -> None:
    baseline = _straight(5.0)
    candidate = baseline.clone()
    candidate[..., 1] += 0.2
    mask = cached_relative_feasibility_mask(
        candidate,
        baseline,
        torch.zeros(1, 1, 2),
        max_mean_accel_degradation=1e6,
        max_mean_jerk_degradation=1e6,
        max_progress_loss_m=1e6,
        max_mean_lateral_deviation_m=0.1,
    )
    assert mask.tolist() == [False]


def test_differentiable_penalty_is_zero_inside_budget() -> None:
    baseline = _straight(5.0)
    candidate = baseline.clone().requires_grad_(True)
    result = differentiable_relative_feasibility_penalty(
        candidate, baseline, torch.zeros(1, 1, 2)
    )
    assert float(result["loss"]) == 0.0
    assert float(result["within_budget_rate"]) == 1.0


def test_differentiable_penalty_backpropagates_for_progress_loss() -> None:
    baseline = _straight(5.0)
    candidate = _straight(1.0).requires_grad_(True)
    result = differentiable_relative_feasibility_penalty(
        candidate,
        baseline,
        torch.zeros(1, 1, 2),
        max_mean_accel_degradation=1e6,
        max_mean_jerk_degradation=1e6,
        max_progress_loss_m=0.1,
        max_mean_lateral_deviation_m=1e6,
    )
    assert float(result["progress"]) > 0.0
    result["loss"].backward()
    assert candidate.grad is not None
    assert float(candidate.grad.abs().sum()) > 0.0


def test_differentiable_penalty_detects_lateral_excess() -> None:
    baseline = _straight(5.0)
    candidate = baseline.clone()
    candidate[..., 1] += 0.2
    result = differentiable_relative_feasibility_penalty(
        candidate,
        baseline,
        torch.zeros(1, 1, 2),
        max_mean_accel_degradation=1e6,
        max_mean_jerk_degradation=1e6,
        max_progress_loss_m=1e6,
        max_mean_lateral_deviation_m=0.1,
    )
    assert float(result["lateral"]) > 0.0


def test_repair_can_remove_lateral_residual_without_losing_longitudinal_style() -> None:
    baseline = _straight(5.0)[0]
    styled = baseline.clone()
    styled[..., 0] += 1.0
    styled[..., 1] += 0.4
    repaired = compose_baseline_anchored_ego(
        styled,
        baseline,
        longitudinal_scale=1.0,
        lateral_scale=0.0,
    )
    assert torch.allclose(repaired[..., 0], styled[..., 0])
    assert torch.allclose(repaired[..., 1], baseline[..., 1])


def test_repair_can_remove_longitudinal_residual_without_losing_lateral_style() -> None:
    baseline = _straight(5.0)[0]
    styled = baseline.clone()
    styled[..., 0] += 1.0
    styled[..., 1] += 0.4
    repaired = compose_baseline_anchored_ego(
        styled,
        baseline,
        longitudinal_scale=0.0,
        lateral_scale=1.0,
    )
    assert torch.allclose(repaired[..., 0], baseline[..., 0])
    assert torch.allclose(repaired[..., 1], styled[..., 1])


def test_repair_scale_grid_prioritizes_longitudinal_style() -> None:
    grid = candidate_scale_grid([1.0, 0.5, 0.0], [1.0, 0.5, 0.0])
    assert grid[:3] == ((1.0, 1.0), (1.0, 0.5), (1.0, 0.0))
    assert grid[-1] == (0.0, 0.0)


def test_interpolated_ego_batch_preserves_endpoints_and_order() -> None:
    baseline = _straight(3.0)[0]
    styled = _straight(7.0)[0]
    scales = torch.tensor([1.0, 0.5, 0.0])
    candidates = compose_interpolated_ego_batch(styled, baseline, scales)
    assert candidates.shape == (3, baseline.shape[0], baseline.shape[1])
    assert torch.allclose(candidates[0, :, :2], styled[:, :2])
    assert torch.allclose(candidates[-1, :, :2], baseline[:, :2])
    assert torch.allclose(
        candidates[1, :, :2], 0.5 * (styled[:, :2] + baseline[:, :2])
    )


def test_batched_clearance_distinguishes_parallel_candidates() -> None:
    ego_slow = _straight(2.0, steps=10)[0]
    ego_fast = _straight(8.0, steps=10)[0]
    neighbors = _straight(0.0, steps=10)[0][None]
    neighbors[..., 0] = 8.0
    clearances = batched_three_disc_min_clearance(
        torch.stack((ego_slow, ego_fast)),
        neighbors,
        torch.tensor([4.5]),
        torch.tensor([2.0]),
        ego_length=5.0,
        ego_width=2.0,
        ego_origin_to_center=1.5,
        collision_margin_m=0.25,
    )
    assert clearances.shape == (2,)
    assert float(clearances[0]) > float(clearances[1])

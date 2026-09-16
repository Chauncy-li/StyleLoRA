from __future__ import annotations

import torch

from stylelora.training.conditional_preference_lora import (
    _sample_ordered_rho_pairs,
    longitudinal_response_objective,
    ordered_pair_objective,
)
from stylelora.model.conditional_lora_router import (
    ConditionalLoRARouter,
    load_conditional_router_checkpoint,
    save_conditional_router_checkpoint,
)
from stylelora.tests.test_scene_gate import _constant_gate, _planner_and_inputs


def _router(planner) -> ConditionalLoRARouter:
    return ConditionalLoRARouter(planner.report.layers, hc_dim=8, z_dim=8, hidden_dim=16)


def _prototypes() -> dict[str, torch.Tensor]:
    return {
        "low": torch.tensor([-1.0, 0, 0, 0, 0, 0, 0, 0]),
        "neutral": torch.zeros(8),
        "high": torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]),
    }


def test_router_initialization_reproduces_legacy_strength() -> None:
    planner, inputs = _planner_and_inputs()
    planner.set_strength(0.7)
    _, legacy = planner(inputs)
    planner.attach_conditional_router(_router(planner), _prototypes(), enabled=True)
    _, conditional = planner(inputs)
    assert torch.allclose(legacy["prediction"], conditional["prediction"], atol=1e-6, rtol=1e-6)


def test_conditional_router_rho_zero_is_exact_baseline() -> None:
    planner, inputs = _planner_and_inputs()
    planner.attach_conditional_router(_router(planner), _prototypes(), enabled=True)
    planner.set_strength(0.0)
    _, conditional = planner(inputs)
    planner.disable_conditional_router().set_strength(0.0)
    _, baseline = planner(inputs)
    assert torch.equal(conditional["prediction"], baseline["prediction"])


def test_scene_gate_still_caps_conditional_router() -> None:
    planner, inputs = _planner_and_inputs()
    planner.attach_conditional_router(_router(planner), _prototypes(), enabled=True)
    planner.attach_scene_gate(_constant_gate(0.25, 0.6), enabled=True).set_strength(1.0)
    planner(inputs)
    assert torch.allclose(planner.last_scene_gate["effective_rho"], torch.full((3,), 0.6))
    assert planner.last_conditional_router is not None


def test_router_checkpoint_round_trip(tmp_path) -> None:
    planner, _ = _planner_and_inputs()
    router = _router(planner)
    path = tmp_path / "router.pt"
    save_conditional_router_checkpoint(
        path,
        router,
        prototypes=_prototypes(),
        training_config={"seed": 17},
        best_validation={"loss": 0.1},
    )
    loaded, prototypes, payload = load_conditional_router_checkpoint(path)
    h_c = torch.randn(3, 8)
    target = torch.randn(3, 8)
    assert torch.equal(router(h_c, target), loaded(h_c, target))
    assert set(prototypes) == {"low", "neutral", "high"}
    assert payload["format"] == "stylelora.conditional_router.v2"
    assert loaded.use_diffusion_time is False


def test_v2_requested_style_uses_prototypes_for_training_coordinate() -> None:
    planner, inputs = _planner_and_inputs()
    planner.attach_conditional_router(_router(planner), _prototypes(), enabled=True)
    rho = torch.tensor([-0.5, 0.0, 0.75])
    planner.set_conditional_coordinate(rho)
    _, h_c = planner.encode_context(inputs)
    target = planner.build_requested_style(rho, h_c)
    assert torch.allclose(target[:, 0], torch.tensor([-0.5, 0.0, 0.75]))


def test_ordered_pair_objective_normalizes_by_feasible_weight() -> None:
    low = torch.tensor([0.0, 0.0])
    high = torch.tensor([0.0, 1.0])
    confidence = torch.tensor([0.5, 0.5])
    feasible = torch.tensor([True, False])
    margin = torch.tensor([0.2, 0.2])
    result = ordered_pair_objective(low, high, confidence, feasible, margin)
    assert torch.allclose(result["loss"], torch.tensor(0.04))
    assert float(result["valid_weight"]) == 0.5
    assert float(result["valid_count"]) == 1.0


def test_ordered_pair_objective_handles_empty_feasible_set() -> None:
    result = ordered_pair_objective(
        torch.zeros(2),
        torch.ones(2),
        torch.ones(2),
        torch.zeros(2, dtype=torch.bool),
        torch.full((2,), 0.2),
    )
    assert float(result["loss"]) == 0.0
    assert float(result["violation_rate"]) == 0.0


def test_mixed_order_sampling_uses_adjacent_grid_when_ratio_is_one() -> None:
    torch.manual_seed(17)
    grid = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
    low, high, local = _sample_ordered_rho_pairs(
        torch.zeros(64),
        mode="mixed_adjacent",
        rho_grid=grid,
        local_order_ratio=1.0,
        min_rho_gap=0.25,
    )
    assert bool(local.all())
    assert torch.allclose(high - low, torch.full((64,), 0.25))
    assert set(low.tolist()).issubset(set(grid[:-1]))


def _straight_ego(final_x: float, steps: int = 8) -> torch.Tensor:
    x = torch.linspace(final_x / steps, final_x, steps)
    return torch.stack(
        (x, torch.zeros_like(x), torch.ones_like(x), torch.zeros_like(x)), dim=-1
    )[None]


def test_longitudinal_response_requires_visible_progress_gap() -> None:
    baseline = _straight_ego(8.0)
    result = longitudinal_response_objective(
        _straight_ego(7.9),
        _straight_ego(8.1),
        baseline,
        torch.zeros(1, 1, 2),
        torch.ones(1),
        torch.ones(1, dtype=torch.bool),
        torch.full((1,), 0.25),
        margin_m_per_rho=1.0,
        min_baseline_progress_m=3.0,
    )
    assert torch.allclose(result["mean_gap_m"], torch.tensor(0.2), atol=1e-5)
    assert torch.allclose(result["loss"], torch.tensor(0.0025), atol=1e-5)


def test_longitudinal_response_ignores_nonmoving_baseline() -> None:
    baseline = _straight_ego(1.0)
    result = longitudinal_response_objective(
        _straight_ego(0.5),
        _straight_ego(0.5),
        baseline,
        torch.zeros(1, 1, 2),
        torch.ones(1),
        torch.ones(1, dtype=torch.bool),
        torch.full((1,), 0.25),
        margin_m_per_rho=1.0,
        min_baseline_progress_m=3.0,
    )
    assert float(result["loss"]) == 0.0
    assert float(result["active_rate"]) == 0.0

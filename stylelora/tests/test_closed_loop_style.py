from __future__ import annotations

import numpy as np
import pytest

from stylelora.lora.evaluation.closed_loop_style import analyze_closed_loop_style_runs
from stylelora.scripts.evaluate_closed_loop import (
    OFFICIAL_SAFETY_METRICS,
    ROUTE_FAILURE_TOKENS,
    _challenge_result_dir,
    _paired_official_differences,
    _prepare_formal_tokens,
)
from stylelora.scripts.select_closed_loop_scenarios import _proportional_quotas


def _write_step(root, *, speed: float, gap: float, token: str = "") -> None:
    scenario = root / (token or "scenario_000001")
    scenario.mkdir(parents=True)
    dt = 0.1
    x = np.arange(1, 6, dtype=np.float32) * speed * dt
    ego_future = np.stack((x, np.zeros_like(x), np.ones_like(x), np.zeros_like(x)), axis=-1)
    neighbor_future = ego_future[None].copy()
    neighbor_future[0, :, 0] += gap
    neighbor_past = np.zeros((1, 2, 7), dtype=np.float32)
    neighbor_past[0, -1, 0] = gap
    ego_current = np.zeros(10, dtype=np.float32)
    ego_current[4] = speed
    payload = dict(
        iteration_index=np.asarray(0, dtype=np.int64),
        time_us=np.asarray(123456, dtype=np.int64),
        ego_current_state=ego_current,
        neighbor_agents_past=neighbor_past,
        generated_ego_future=ego_future,
        generated_neighbor_future=neighbor_future,
    )
    if token:
        payload["scenario_token"] = np.asarray(token)
    np.savez_compressed(scenario / "step_000000_iter_0000_123456.npz", **payload)


def test_closed_loop_style_uses_common_steps_and_correct_direction(tmp_path):
    roots = {}
    for rho, speed, gap in ((-1.0, 1.0, 20.0), (0.0, 2.0, 15.0), (1.0, 3.0, 10.0)):
        root = tmp_path / f"rho_{rho}"
        _write_step(root, speed=speed, gap=gap)
        roots[rho] = root

    report = analyze_closed_loop_style_runs(roots)

    assert report["available"] is True
    assert report["common_steps"] == 1
    assert report["common_current_lead_steps"] == 1
    assert report["trends"]["current_ego_speed"]["rho_spearman"] == pytest.approx(1.0)
    assert report["trends"]["current_min_front_gap"]["rho_spearman"] == pytest.approx(-1.0)
    assert report["trends"]["current_time_headway"]["direction_correct"] is True
    assert report["core_direction_correct_count"] == 3


def test_closed_loop_style_can_filter_exact_common_tokens(tmp_path):
    roots = {}
    for rho in (-1.0, 0.0, 1.0):
        root = tmp_path / f"rho_{rho}"
        _write_step(root, speed=2.0 + rho, gap=15.0, token="keep")
        _write_step(root, speed=20.0, gap=3.0, token="remove")
        roots[rho] = root

    report = analyze_closed_loop_style_runs(roots, allowed_tokens={"keep"})

    assert report["available"] is True
    assert report["common_steps"] == 1
    assert report["trends"]["current_ego_speed"]["rho_spearman"] == pytest.approx(1.0)


def test_formal_tokens_remove_fixed_route_failures_and_keep_45():
    source = [f"token_{index:02d}" for index in range(45)] + sorted(ROUTE_FAILURE_TOKENS)

    tokens, removed = _prepare_formal_tokens(source)

    assert tokens == [f"token_{index:02d}" for index in range(45)]
    assert set(removed) == ROUTE_FAILURE_TOKENS


def test_formal_tokens_support_explicit_200_scenarios():
    source = [f"token_{index:03d}" for index in range(200)] + sorted(ROUTE_FAILURE_TOKENS)

    tokens, removed = _prepare_formal_tokens(source, expected_count=200)

    assert tokens == [f"token_{index:03d}" for index in range(200)]
    assert set(removed) == ROUTE_FAILURE_TOKENS


def test_formal_tokens_support_explicit_300_candidates():
    source = [f"token_{index:03d}" for index in range(300)]

    tokens, removed = _prepare_formal_tokens(source, expected_count=300)

    assert tokens == source
    assert removed == []


def test_fixed_total_selection_allocates_exactly_200():
    quotas = _proportional_quotas(
        {"straight_free_drive": 80, "straight_car_follow": 720},
        total=200,
    )

    assert quotas == {"straight_free_drive": 20, "straight_car_follow": 180}
    assert sum(quotas.values()) == 200


def test_challenge_result_directory_contains_challenge(tmp_path):
    challenge = "closed_loop_nonreactive_agents"

    result_dir = _challenge_result_dir(tmp_path / "rho_minus1.00", challenge)

    assert result_dir.name == challenge


def test_paired_official_differences_use_rho_zero_per_scene():
    def official(score_a: float, score_b: float, metric_offset: float) -> dict:
        rows = []
        for token, score in (("a", score_a), ("b", score_b)):
            rows.append({
                "token": token,
                "log_name": f"log_{token}",
                "challenge_score": score,
                "safety_metrics": {
                    name: float(index) + metric_offset
                    for index, name in enumerate(OFFICIAL_SAFETY_METRICS)
                },
            })
        return {
            "available": True,
            "scenario_count": 2,
            "challenge_score": (score_a + score_b) / 2,
            "safety_metric_means": {},
            "per_scenario": rows,
        }

    records = [
        {"rho": 0.0, "official_aggregator": official(0.4, 0.6, 0.0)},
        {"rho": 1.0, "official_aggregator": official(0.7, 0.8, 0.25)},
    ]

    paired = _paired_official_differences(records)["rho_+1.00"]

    assert paired["paired_scenario_count"] == 2
    assert paired["mean_challenge_score_delta"] == pytest.approx(0.25)
    assert paired["mean_safety_metric_deltas"][OFFICIAL_SAFETY_METRICS[0]] == pytest.approx(0.25)
    assert {row["token"] for row in paired["per_scenario"]} == {"a", "b"}

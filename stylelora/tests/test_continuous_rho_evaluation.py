"""连续 rho 开环验证指标单测。"""

import pytest

from stylelora.scripts.evaluate_preference_lora import continuous_rho_metrics


def _records(curved: bool = False) -> list[dict]:
    rows = []
    for sample in range(2):
        for rho in (-1.0, -0.5, 0.0, 0.5, 1.0):
            value = rho + float(sample)
            if curved and abs(rho) == 0.5:
                value += 0.2
            rows.append({
                "batch": 0,
                "sample": sample,
                "rho": rho,
                "s": value,
                "z": [value, 2.0 * value],
            })
    return rows


def test_continuous_rho_metrics_zero_for_linear_path() -> None:
    report = continuous_rho_metrics(_records(), [-1.0, -0.5, 0.0, 0.5, 1.0])

    assert report["complete_sample_count"] == 2
    assert report["samplewise_s_monotonic_rate"] == 1.0
    assert report["style_span"]["mean"] == pytest.approx(2.0)
    assert report["average_slope"]["mean"] == pytest.approx(1.0)
    assert report["positive_slope_rate"] == 1.0
    assert report["nonzero_response_rate"] == 1.0
    assert report["low"]["latent_interpolation_mse"]["max"] == pytest.approx(0.0)
    assert report["high"]["latent_interpolation_mse"]["max"] == pytest.approx(0.0)
    assert report["high"]["scalar_s_interpolation_abs_error"]["max"] == pytest.approx(0.0)


def test_continuous_rho_metrics_detects_curved_path() -> None:
    report = continuous_rho_metrics(_records(curved=True), [-1.0, -0.5, 0.0, 0.5, 1.0])

    assert report["low"]["latent_interpolation_mse"]["mean"] > 0.0
    assert report["high"]["latent_interpolation_mse"]["mean"] > 0.0
    assert report["high"]["scalar_s_interpolation_abs_error"]["mean"] > 0.0


def test_continuous_rho_metrics_requires_zero() -> None:
    with pytest.raises(ValueError, match="rho=0"):
        continuous_rho_metrics([], [-1.0, -0.5, 0.5, 1.0])


def test_monotonic_rate_does_not_hide_a_constant_response() -> None:
    records = []
    for rho in (-1.0, 0.0, 1.0):
        records.append({"batch": 0, "sample": 0, "rho": rho, "s": 0.4, "z": [0.4]})

    report = continuous_rho_metrics(records, [-1.0, 0.0, 1.0])

    assert report["samplewise_s_monotonic_rate"] == 1.0
    assert report["style_span"]["mean"] == pytest.approx(0.0)
    assert report["average_slope"]["mean"] == pytest.approx(0.0)
    assert report["positive_slope_rate"] == 0.0
    assert report["nonzero_response_rate"] == 0.0


def test_effective_control_requires_response_and_performance_budget() -> None:
    records = []
    for rho in (-1.0, 0.0, 1.0):
        records.append({
            "batch": 0,
            "sample": 0,
            "rho": rho,
            "effective_rho": rho,
            "s": rho,
            "z": [rho],
            "ade": 1.7 if rho > 0 else 1.0,
            "fde": 2.0,
            "planned_abs_jerk_p90": 3.0,
        })

    report = continuous_rho_metrics(records, [-1.0, 0.0, 1.0])
    coverage = report["effective_control_coverage"]

    assert coverage["overall_response_rate"] == 1.0
    assert coverage["overall"] == pytest.approx(0.5)
    assert coverage["command_retention_ratio"]["mean"] == pytest.approx(1.0)
    assert report["performance_cost_vs_rho_zero"]["ade_delta"]["mean"] == pytest.approx(0.35)

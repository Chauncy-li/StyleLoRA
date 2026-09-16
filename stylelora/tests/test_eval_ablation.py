from __future__ import annotations

import pytest

from stylelora.eval.section_f_ablation_efficiency import _component_flags
from stylelora.eval.build_common_closed_loop_subset import (
    SAFETY_METRICS,
    _filter_official,
    _paired_differences,
)


def test_component_flags_parse_three_ablation_switches() -> None:
    flags = _component_flags([
        "Scalar-only alignment=0,0,0",
        "No lateral constraint=1,0,0",
        "Without adaptive gate=1,1,0",
        "Full CAST=1,1,1",
    ])

    assert flags["Scalar-only alignment"]["structured_alignment"] == 0
    assert flags["No lateral constraint"]["structured_alignment"] == 1
    assert flags["Without adaptive gate"]["lateral_constraint"] == 1
    assert flags["Full CAST"]["adaptive_gate"] == 1


def test_component_flags_reject_invalid_shape() -> None:
    with pytest.raises(ValueError):
        _component_flags(["Full CAST=1,1"])


def _official_rows(scores: dict[str, float]) -> dict:
    rows = [{
        "token": token,
        "log_name": f"log_{token}",
        "challenge_score": score,
        "safety_metrics": {metric: 1.0 for metric in SAFETY_METRICS},
    } for token, score in scores.items()]
    return {"available": True, "per_scenario": rows}


def test_common_subset_recomputes_means_and_paired_differences() -> None:
    zero = _filter_official(_official_rows({"a": 0.2, "b": 0.8}), ["b"])
    high = _filter_official(_official_rows({"a": 0.4, "b": 1.0}), ["b"])
    records = [
        {"rho": 0.0, "official_aggregator": zero},
        {"rho": 1.0, "official_aggregator": high},
    ]

    assert zero["scenario_count"] == 1
    assert zero["challenge_score"] == pytest.approx(0.8)
    paired = _paired_differences(records)["rho_+1.00"]
    assert paired["paired_scenario_count"] == 1
    assert paired["mean_challenge_score_delta"] == pytest.approx(0.2)

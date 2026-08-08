from research_lora.data.schema import StyleSample
from research_lora.scripts.evaluate_open_loop import (
    _SceneBalancedEvaluationSampler,
    _mark_common_style_validity,
    _parse_rhos,
)


def test_open_loop_sampler_is_equal_and_repeatable():
    samples = [StyleSample(f"free_{index}", "val", "straight_free_drive", "normal") for index in range(2)]
    samples += [StyleSample(f"follow_{index}", "val", "straight_car_follow", "normal") for index in range(5)]
    sampler = _SceneBalancedEvaluationSampler(samples, num_samples=20, seed=17)
    first, second = list(sampler), list(sampler)
    assert first == second
    report = sampler.report()
    assert report["straight_free_drive"] == report["straight_car_follow"] == 10


def test_open_loop_rho_parser_and_common_validity():
    assert _parse_rhos("-2,-1,0,1,2") == (-2.0, -1.0, 0.0, 1.0, 2.0)
    try:
        _parse_rhos("-1,1")
    except ValueError as error:
        assert "rho=0" in str(error)
    else:
        raise AssertionError("rho grid without zero must be rejected")

    records = [
        {"scene": "straight_car_follow", "sample_id": "car:0", "rho": rho, "style_valid": True}
        for rho in (-1.0, 0.0, 1.0)
    ]
    records += [
        {"scene": "straight_car_follow", "sample_id": "car:1", "rho": rho, "style_valid": rho != 1.0}
        for rho in (-1.0, 0.0, 1.0)
    ]
    audit = _mark_common_style_validity(records, (-1.0, 0.0, 1.0))
    assert audit["straight_car_follow"]["common_valid_samples"] == 1
    assert all(record["comparison_valid"] == (record["sample_id"] == "car:0") for record in records)

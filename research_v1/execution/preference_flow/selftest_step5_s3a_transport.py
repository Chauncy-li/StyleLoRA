"""Contract tests for Step 5-S3A-R longitudinal target and neutral audits."""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from research_v1.execution.preference_flow.longitudinal_transport_targets import (
    AxisCoherence,
    LongitudinalTransportError,
    axis_coherence_from_style,
    build_continuous_transport_target,
    classify_projection_backtracking,
    json_ready,
    project_expert_onto_neutral_path,
    raw_physical_feasibility_audit,
    trajectory_sanity_audit,
    transport_coordinate_is_identifiable,
    validate_explicit_audit_output_dir,
)
from research_v1.execution.preference_flow.run_step5_s3a_transport_audit import (
    INFERENCE_CACHE_KEYS,
    _FULL_DPM_SOURCE,
    _FIXED_Q_SOURCE,
    _all_valid_contract_outputs_finite,
    _full_dpm_prediction,
    _normalize_full_dpm_inputs_once,
    _source_report,
)


def _check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def _raises(callable_object, label: str) -> None:
    try:
        callable_object()
    except LongitudinalTransportError:
        return
    raise AssertionError(label)


def _script_constant(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing {name}")


def _function_source(source: str, function_name: str) -> str:
    lines = source.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(f"def {function_name}("))
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("def ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def run() -> dict[str, bool]:
    results: dict[str, bool] = {}

    straight = project_expert_onto_neutral_path(
        [0.0, 0.0], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    _check(straight.valid and np.array_equal(straight.expert_projected_progress, np.array([1.0, 2.0, 3.0])), "straight exact projection")
    results["straight_projection"] = True

    curved = project_expert_onto_neutral_path(
        [0.0, 0.0], [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]], [[0.5, 0.0], [1.0, 0.5], [1.5, 1.0]]
    )
    _check(curved.valid and np.allclose(curved.expert_projected_progress, [0.5, 1.5, 2.5]), "piecewise curved projection")
    results["curved_projection"] = True

    offset = project_expert_onto_neutral_path(
        [0.0, 0.0], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    )
    _check(offset.valid and np.array_equal(offset.expert_projected_progress, straight.neutral_progress), "lateral offset keeps progress")
    results["lateral_offset_progress"] = True

    different_route = project_expert_onto_neutral_path(
        [0.0, 0.0], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], [[1.0, 20.0], [2.0, 20.0], [3.0, 20.0]]
    )
    backtracking = project_expert_onto_neutral_path(
        [0.0, 0.0], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], [[2.0, 0.0], [1.0, 0.0], [3.0, 0.0]]
    )
    _check(not different_route.valid and not backtracking.valid and backtracking.monotonic_violation, "route mismatch or backtracking is abnormal")
    results["abnormal_projection"] = True

    _check(classify_projection_backtracking(0.01) == "numerical_or_local_projection_backtracking", "0.01m backtracking is numerical/local")
    _check(classify_projection_backtracking(0.10) == "semantic_backtracking", "0.10m backtracking is semantic")
    results["backtracking_classification"] = True

    sanity = trajectory_sanity_audit(
        [0.0, 0.0], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    _check(not sanity["neutral_generation_invalid"] and sanity["direct_ade_m"] == 0.0, "physical trajectory sanity")
    results["trajectory_sanity"] = True

    coherent = axis_coherence_from_style([0.60, 0.64, 0.62, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    incoherent = axis_coherence_from_style([0.10, 0.90, 0.50, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    _check(coherent.one_dimensional_coherent and coherent.target_rho_star is not None, "coherent axis label")
    _check(not incoherent.one_dimensional_coherent and incoherent.target_rho_star is None, "incoherent axis label excluded")
    results["axis_coherence"] = True

    _check(abs(float(coherent.rho_star) - 0.24) < 1e-12, "rho star is continuous")
    _check(all(abs(float(coherent.rho_star) - value) > 1e-9 for value in (-1.0, -0.5, 0.0, 0.5, 1.0)), "rho star was not quantized")
    results["continuous_rho"] = True

    neutral = np.array([1.0, 2.0, 3.0])
    expert = np.array([1.5, 3.0, 4.5])
    zero = build_continuous_transport_target(neutral, expert, 0.24, 0.0)
    one = build_continuous_transport_target(neutral, expert, 0.24, 1.0)
    _check(np.array_equal(zero.target_progress, neutral) and np.array_equal(zero.target_progress_residual, np.zeros_like(neutral)), "lambda zero exact neutral")
    _check(np.array_equal(one.target_progress, expert), "lambda one exact expert progress")
    results["transport_endpoints"] = True

    positive = build_continuous_transport_target(neutral, expert, 0.7, 0.5)
    negative = build_continuous_transport_target(neutral, expert, -0.7, 0.5)
    _check(positive.preference_coordinate_r > 0.0 and negative.preference_coordinate_r < 0.0, "positive and negative rho")
    _check(not transport_coordinate_is_identifiable(neutral, expert, 0.0), "non-neutral endpoint cannot share r zero")
    _check(transport_coordinate_is_identifiable(neutral, neutral, 0.0), "neutral r zero remains identifiable")
    results["rho_sign"] = True

    zero_length = project_expert_onto_neutral_path([0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    _check(
        not zero_length.valid
        and "neutral_path_has_no_nonzero_segment" in zero_length.invalid_reasons
        and np.array_equal(zero_length.zero_length_segment_indices, [0, 1]),
        "zero length path invalid",
    )
    _raises(lambda: build_continuous_transport_target([1.0], [1.0, 2.0], 0.0, 0.0), "invalid transport input rejected")
    results["invalid_inputs"] = True

    physics = raw_physical_feasibility_audit(
        [0.0, 0.0], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], straight.expert_projected_progress, np.zeros((1, 3, 4))
    )
    _check(json.loads(json.dumps(json_ready({"coherent": coherent, "projection": straight, "physics": physics}))) is not None, "JSON serializable")
    results["json_serializable"] = True

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = validate_explicit_audit_output_dir(root / "audit", root / "repo")
        _check(output.name == "audit", "explicit non-production output accepted")
        _raises(lambda: validate_explicit_audit_output_dir(root / "repo" / "baseline" / "results", root / "repo"), "baseline output rejected")
    results["explicit_output_only"] = True

    class CountingNormalizer:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, inputs):
            self.calls += 1
            return {key: value + 1.0 for key, value in inputs.items()}

    class FakeModel:
        def __init__(self) -> None:
            self.received = None

        def __call__(self, inputs):
            self.received = inputs
            prediction = torch.tensor([[[[3.0, 4.0, 0.0, 1.0], [7.0, 8.0, 0.0, 1.0]]]])
            return {}, {"prediction": prediction}

    normalizer = CountingNormalizer()
    raw_inputs = {"ego_current_state": torch.zeros((1, 1, 4), dtype=torch.float32)}
    normalized_inputs, normalization_meta = _normalize_full_dpm_inputs_once(raw_inputs, SimpleNamespace(observation_normalizer=normalizer))
    model = FakeModel()
    prediction_xy = _full_dpm_prediction(model, normalized_inputs, sampling_seed=7)
    _check(normalizer.calls == 1 and normalization_meta["observation_normalization_count"] == 1, "observation normalizer called exactly once")
    _check(torch.equal(raw_inputs["ego_current_state"], torch.zeros((1, 1, 4))) and torch.equal(model.received["ego_current_state"], torch.ones((1, 1, 4))), "raw and normalized full-DPM inputs remain distinct")
    _check(np.array_equal(prediction_xy, np.array([[3.0, 4.0], [7.0, 8.0]])), "full-DPM prediction is not inversed a second time")
    _check(
        normalization_meta["expert_future_in_full_dpm_input"] is False
        and normalization_meta["expert_future_used_in_input"] is False
        and "ego_agent_future" not in INFERENCE_CACHE_KEYS
        and "neighbor_agents_future" not in INFERENCE_CACHE_KEYS,
        "full-DPM input excludes expert future",
    )
    try:
        _normalize_full_dpm_inputs_once(
            {**raw_inputs, "ego_agent_future": torch.zeros((1, 2, 4))},
            SimpleNamespace(observation_normalizer=CountingNormalizer()),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("full-DPM normalizer must reject expert-future inputs")
    results["full_dpm_normalization_contract"] = True

    empty_report = _source_report("test", "test", [], {})
    _check(
        _all_valid_contract_outputs_finite([]) is None
        and empty_report["all_valid_contract_outputs_finite"] is None
        and empty_report["method_hypothesis_status"] == "insufficient_no_valid_contract",
        "empty valid-contract aggregate is null rather than a pass",
    )
    _check(_FIXED_Q_SOURCE != _FULL_DPM_SOURCE, "fixed-q and full-DPM reports are independent")
    results["empty_aggregate_and_dual_source"] = True

    script_path = Path(__file__).with_name("run_step5_s3a_transport_audit.py")
    script_source = script_path.read_text(encoding="utf-8")
    _check('parser.add_argument("--output-dir", required=True)' in script_source, "audit output has no production default")
    inference_keys = _script_constant(script_source, "INFERENCE_CACHE_KEYS")
    _check("ego_agent_future" not in inference_keys and "neighbor_agents_future" not in inference_keys, "expert futures absent from inference keys")
    _check("ego_agent_future" not in _function_source(script_source, "_load_inference_inputs"), "expert future absent from inference loader")
    full_source = _function_source(script_source, "_full_dpm_prediction")
    fixed_source = _function_source(script_source, "_fixed_q_prediction")
    _check("state_normalizer.inverse" not in full_source, "full-DPM prediction has no second inverse")
    _check(all(marker in fixed_source for marker in ("_prepare_batch", "_noisy_state", "_neutral_clean_prediction", "_physical_ego_future")), "fixed-q reuses Step-5 construction")
    _check("vector_field" not in fixed_source, "fixed-q expert future is not a Vector Field condition")
    _check(script_source.index("def _full_dpm_prediction") < script_source.index("def _load_expert_audit_arrays"), "expert audit load follows full-DPM definition")
    results["expert_future_target_only"] = True

    return results


def main() -> None:
    results = run()
    print(json.dumps({"passed": all(results.values()), "tests": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

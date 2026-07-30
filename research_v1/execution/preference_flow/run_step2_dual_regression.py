"""Run Step-2 neutral/preference dual-stream regressions on real cache scenes.

This is a verification program, not a trainer.  It loads one frozen
StylePlanner checkpoint, runs two independent DPM streams from the same noise,
and writes measured identity and branch-isolation reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from baseline.model.style_planner.diffusion_planner import Diffusion_Planner
from baseline.model.style_planner.preference_flow import (
    CLEAN_PREDICTION_EDITOR_DISABLED,
    CleanPredictionEditContext,
    DPMEvaluationRecord,
    DualStreamSampleResult,
    IdentityCleanPredictionEditor,
)
from research_v1.execution.preference_flow.run_step1_base_regression import (
    _build_model_args,
    _cache_inputs,
    _checkpoint_state,
    _clone_inputs,
    _existing_file,
    _load_frozen_styleplanner,
    _max_abs_difference,
    _model_args_path,
    _scene_entries,
    _set_sampling_seed,
    _trajectory_ade_fde,
    _write_report,
)


_IDENTITY_TOLERANCE = 1e-6


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Step-2 neutral/preference dual DPM streams with one frozen "
            "StylePlanner checkpoint.  This command does not train a model."
        )
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--scene-token-file", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-args", default=None)
    parser.add_argument("--normalization-file-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefer-ema", action="store_true")
    parser.add_argument(
        "--probe-evaluation-index",
        type=int,
        default=0,
        help="Test-only clean-prediction injection index for branch isolation.",
    )
    parser.add_argument(
        "--probe-magnitude",
        type=float,
        default=1e-3,
        help="Positive test-only clean-prediction perturbation magnitude.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


class _TinyPreferenceProbeEditor:
    """A test-only x0 perturbation used exclusively for branch isolation."""

    def __init__(self, *, evaluation_index: int, magnitude: float) -> None:
        if int(evaluation_index) < 0:
            raise ValueError("probe evaluation index must be non-negative")
        if float(magnitude) <= 0.0:
            raise ValueError("probe magnitude must be positive")
        self._evaluation_index = int(evaluation_index)
        self._magnitude = float(magnitude)
        self.reset()

    def reset(self) -> None:
        self._references: List[DPMEvaluationRecord] = []
        self._call_count = 0
        self._applied_delta = 0.0
        self._injection_applied = False

    def __call__(
        self,
        clean_prediction: torch.Tensor,
        context: CleanPredictionEditContext,
    ) -> torch.Tensor:
        if context.stream_name != "preference":
            raise RuntimeError("test preference probe was installed on a non-preference stream")
        if context.neutral_record is None:
            raise RuntimeError("preference probe did not receive its neutral reference")
        self._references.append(context.neutral_record)
        self._call_count += 1
        if context.model_evaluation_index != self._evaluation_index:
            return clean_prediction

        edited = clean_prediction.clone()
        # The flattened joint state starts with the constrained current pose.
        # Perturb one future coordinate so the test change reaches a DPM update.
        component = 4 if int(edited.shape[-1]) > 4 else 0
        edited[..., component] += self._magnitude
        self._applied_delta = float((edited - clean_prediction).abs().max().item())
        self._injection_applied = True
        return edited

    @property
    def applied_delta(self) -> float:
        return float(self._applied_delta)

    @property
    def injection_applied(self) -> bool:
        return bool(self._injection_applied)

    def neutral_reference_error(
        self,
        neutral_trace: Sequence[DPMEvaluationRecord],
    ) -> float:
        if len(self._references) != len(neutral_trace):
            raise RuntimeError(
                "preference probe received a different number of neutral references "
                f"than neutral evaluations: {len(self._references)} versus "
                f"{len(neutral_trace)}"
            )
        errors = []
        for index, (reference, neutral) in enumerate(
            zip(self._references, neutral_trace)
        ):
            if reference.model_evaluation_index != neutral.model_evaluation_index:
                raise RuntimeError(f"neutral reference index mismatch at evaluation {index}")
            if bool(reference.is_terminal_denoise) != bool(neutral.is_terminal_denoise):
                raise RuntimeError(f"neutral reference terminal flag mismatch at {index}")
            errors.extend(
                [
                    _max_abs_difference(
                        reference.current_state,
                        neutral.current_state,
                        label=f"probe neutral current state[{index}]",
                    ),
                    _max_abs_difference(
                        reference.clean_prediction,
                        neutral.clean_prediction,
                        label=f"probe neutral clean prediction[{index}]",
                    ),
                    _max_abs_difference(
                        reference.diffusion_time,
                        neutral.diffusion_time,
                        label=f"probe neutral diffusion time[{index}]",
                    ),
                    _max_abs_difference(
                        reference.log_snr,
                        neutral.log_snr,
                        label=f"probe neutral log-SNR[{index}]",
                    ),
                ]
            )
        return max(errors, default=0.0)


def _run_dual_rollout(
    model: Diffusion_Planner,
    inputs: Mapping[str, torch.Tensor],
    *,
    sampling_seed: int,
    neutral_editor: Any,
    preference_editor: Any,
) -> Dict[str, Any]:
    _set_sampling_seed(sampling_seed)
    rollout_inputs = _clone_inputs(inputs)
    rollout_inputs["dual_stream_enabled"] = True
    rollout_inputs["neutral_clean_prediction_editor"] = neutral_editor
    rollout_inputs["preference_clean_prediction_editor"] = preference_editor
    with torch.no_grad():
        _, outputs = model(rollout_inputs)
    result = outputs.get("dual_stream_result")
    neutral_prediction = outputs.get("dual_stream_neutral_prediction")
    preference_prediction = outputs.get("dual_stream_preference_prediction")
    if not isinstance(result, DualStreamSampleResult):
        raise RuntimeError("StylePlanner did not return a DualStreamSampleResult")
    if not torch.is_tensor(neutral_prediction) or not torch.is_tensor(
        preference_prediction
    ):
        raise RuntimeError("StylePlanner did not return both dual-stream trajectories")
    return {
        "result": result,
        "neutral_prediction": neutral_prediction.detach().cpu(),
        "preference_prediction": preference_prediction.detach().cpu(),
    }


def _max_trace_tensor_difference(
    left: Sequence[DPMEvaluationRecord],
    right: Sequence[DPMEvaluationRecord],
    *,
    attribute: str,
    label: str,
) -> float:
    if len(left) != len(right):
        raise RuntimeError(
            f"{label} evaluation-count mismatch: {len(left)} versus {len(right)}"
        )
    if not left:
        raise RuntimeError(f"{label} has no DPM evaluations")
    return max(
        _max_abs_difference(
            getattr(left_record, attribute),
            getattr(right_record, attribute),
            label=f"{label}[{index}]",
        )
        for index, (left_record, right_record) in enumerate(zip(left, right))
    )


def _max_later_state_change(trace: Sequence[DPMEvaluationRecord]) -> float:
    if len(trace) < 2:
        return 0.0
    initial = trace[0].current_state
    return max(
        _max_abs_difference(
            initial,
            record.current_state,
            label=f"dynamic current state[{index}]",
        )
        for index, record in enumerate(trace[1:], start=1)
    )


def _identity_scene_metrics(
    rollout: Mapping[str, Any],
    *,
    expected_evaluations: int,
) -> Dict[str, Any]:
    result: DualStreamSampleResult = rollout["result"]
    neutral_trace = result.neutral_trace
    preference_trace = result.preference_trace
    if len(neutral_trace) != expected_evaluations or len(preference_trace) != expected_evaluations:
        raise RuntimeError(
            "Unexpected DPM evaluation count: "
            f"neutral={len(neutral_trace)}, preference={len(preference_trace)}, "
            f"expected={expected_evaluations}"
        )
    for index, (neutral, preference) in enumerate(zip(neutral_trace, preference_trace)):
        if neutral.model_evaluation_index != index or preference.model_evaluation_index != index:
            raise RuntimeError("dual-stream evaluation indices are not contiguous")
        if neutral.stream_name != "neutral" or preference.stream_name != "preference":
            raise RuntimeError("dual-stream trace labels are incorrect")

    same_initial_noise_error = _max_abs_difference(
        result.neutral_initial_state,
        result.preference_initial_state,
        label="neutral versus preference initial noise",
    )
    time_error = _max_trace_tensor_difference(
        neutral_trace,
        preference_trace,
        attribute="diffusion_time",
        label="neutral versus preference diffusion time",
    )
    log_snr_error = _max_trace_tensor_difference(
        neutral_trace,
        preference_trace,
        attribute="log_snr",
        label="neutral versus preference log-SNR",
    )
    xq_error = _max_trace_tensor_difference(
        neutral_trace,
        preference_trace,
        attribute="current_state",
        label="neutral versus preference current DPM state",
    )
    x0_error = _max_trace_tensor_difference(
        neutral_trace,
        preference_trace,
        attribute="clean_prediction",
        label="neutral versus preference clean prediction",
    )
    final_error = _max_abs_difference(
        rollout["neutral_prediction"],
        rollout["preference_prediction"],
        label="neutral versus preference final trajectory",
    )
    ade, fde = _trajectory_ade_fde(
        rollout["neutral_prediction"], rollout["preference_prediction"]
    )
    first_neutral_error = _max_abs_difference(
        neutral_trace[0].current_state,
        result.neutral_initial_state,
        label="first neutral current state versus x_T",
    )
    first_preference_error = _max_abs_difference(
        preference_trace[0].current_state,
        result.preference_initial_state,
        label="first preference current state versus x_T",
    )
    neutral_later_change = _max_later_state_change(neutral_trace)
    preference_later_change = _max_later_state_change(preference_trace)
    initial_storage_isolated = (
        result.neutral_initial_state.data_ptr()
        != result.preference_initial_state.data_ptr()
    )
    trace_storage_isolated = (
        neutral_trace[0].current_state.data_ptr()
        != preference_trace[0].current_state.data_ptr()
    )
    dynamic_current_state_passed = (
        first_neutral_error <= _IDENTITY_TOLERANCE
        and first_preference_error <= _IDENTITY_TOLERANCE
        and neutral_later_change > 0.0
        and preference_later_change > 0.0
        and initial_storage_isolated
        and trace_storage_isolated
    )
    return {
        "denoiser_evaluation_count": len(neutral_trace),
        "same_initial_noise_error": same_initial_noise_error,
        "max_abs_diffusion_time_error": time_error,
        "max_abs_log_snr_error": log_snr_error,
        "max_abs_neutral_preference_xq_error": xq_error,
        "max_abs_neutral_preference_x0_error": x0_error,
        "max_abs_final_trajectory_error": final_error,
        "ade": ade,
        "fde": fde,
        "first_neutral_current_state_error": first_neutral_error,
        "first_preference_current_state_error": first_preference_error,
        "neutral_later_current_state_change": neutral_later_change,
        "preference_later_current_state_change": preference_later_change,
        "initial_storage_isolated": initial_storage_isolated,
        "trace_storage_isolated": trace_storage_isolated,
        "dynamic_current_state_passed": dynamic_current_state_passed,
    }


def _branch_scene_metrics(
    identity_rollout: Mapping[str, Any],
    probe_rollout: Mapping[str, Any],
    probe: _TinyPreferenceProbeEditor,
    *,
    injection_index: int,
) -> Dict[str, Any]:
    identity_result: DualStreamSampleResult = identity_rollout["result"]
    probe_result: DualStreamSampleResult = probe_rollout["result"]
    if injection_index >= len(probe_result.preference_trace) - 1:
        raise ValueError(
            "probe evaluation index must leave a subsequent DPM evaluation to "
            "verify propagation; got "
            f"{injection_index} for {len(probe_result.preference_trace)} evaluations"
        )

    neutral_final_error = _max_abs_difference(
        identity_rollout["neutral_prediction"],
        probe_rollout["neutral_prediction"],
        label="identity neutral versus probe neutral final trajectory",
    )
    neutral_xq_error = _max_trace_tensor_difference(
        identity_result.neutral_trace,
        probe_result.neutral_trace,
        attribute="current_state",
        label="identity neutral versus probe neutral DPM state",
    )
    neutral_x0_error = _max_trace_tensor_difference(
        identity_result.neutral_trace,
        probe_result.neutral_trace,
        attribute="clean_prediction",
        label="identity neutral versus probe neutral clean prediction",
    )
    neutral_cache_error = probe.neutral_reference_error(probe_result.neutral_trace)
    injection_clean_prediction_error = _max_abs_difference(
        probe_result.preference_trace[injection_index].clean_prediction,
        probe_result.neutral_trace[injection_index].clean_prediction,
        label="probe preference versus neutral clean prediction at injection",
    )
    propagated_state_error = max(
        _max_abs_difference(
            probe_result.preference_trace[index].current_state,
            probe_result.neutral_trace[index].current_state,
            label=f"probe preference propagation state[{index}]",
        )
        for index in range(injection_index + 1, len(probe_result.preference_trace))
    )
    preference_final_error = _max_abs_difference(
        probe_rollout["neutral_prediction"],
        probe_rollout["preference_prediction"],
        label="probe neutral versus preference final trajectory",
    )
    ade, fde = _trajectory_ade_fde(
        probe_rollout["neutral_prediction"], probe_rollout["preference_prediction"]
    )
    branch_isolation_passed = (
        probe.injection_applied
        and probe.applied_delta > 0.0
        and neutral_final_error <= _IDENTITY_TOLERANCE
        and neutral_xq_error <= _IDENTITY_TOLERANCE
        and neutral_x0_error <= _IDENTITY_TOLERANCE
        and neutral_cache_error <= _IDENTITY_TOLERANCE
        and injection_clean_prediction_error > 0.0
        and propagated_state_error > 0.0
        and preference_final_error > 0.0
    )
    return {
        "denoiser_evaluation_count": len(probe_result.preference_trace),
        "probe_evaluation_index": injection_index,
        "probe_clean_prediction_edit_max_abs_delta": probe.applied_delta,
        "neutral_final_trajectory_error": neutral_final_error,
        "neutral_xq_error": neutral_xq_error,
        "neutral_x0_error": neutral_x0_error,
        "neutral_cache_alignment_error": neutral_cache_error,
        "injection_clean_prediction_error": injection_clean_prediction_error,
        "post_injection_preference_state_error": propagated_state_error,
        "preference_final_trajectory_error": preference_final_error,
        "ade": ade,
        "fde": fde,
        "branch_isolation_passed": branch_isolation_passed,
    }


def run(args: argparse.Namespace) -> Tuple[Path, Path]:
    checkpoint_path = _existing_file(args.base_checkpoint, "--base-checkpoint")
    cache_root = Path(args.cache_root).expanduser()
    if not cache_root.is_dir():
        raise NotADirectoryError(f"--cache-root is not a directory: {cache_root}")
    scene_token_path = _existing_file(args.scene_token_file, "--scene-token-file")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device={device} was requested but torch.cuda.is_available() is false"
        )

    args_path = _model_args_path(checkpoint_path, args.model_args)
    model_args = _build_model_args(
        args_path,
        mode=CLEAN_PREDICTION_EDITOR_DISABLED,
        device=str(device),
        normalization_file_override=args.normalization_file_path,
    )
    state, checkpoint_meta = _checkpoint_state(
        checkpoint_path,
        prefer_ema=bool(args.prefer_ema),
    )
    model, model_meta = _load_frozen_styleplanner(model_args, state, checkpoint_meta)
    entries = _scene_entries(scene_token_path, cache_root)
    expected_evaluations = int(getattr(model_args, "diffusion_steps", 10)) + 1

    identity_errors = {
        "same_initial_noise_error": 0.0,
        "max_abs_diffusion_time_error": 0.0,
        "max_abs_log_snr_error": 0.0,
        "max_abs_neutral_preference_xq_error": 0.0,
        "max_abs_neutral_preference_x0_error": 0.0,
        "max_abs_final_trajectory_error": 0.0,
        "ade": 0.0,
        "fde": 0.0,
        "first_neutral_current_state_error": 0.0,
        "first_preference_current_state_error": 0.0,
    }
    branch_errors = {
        "neutral_final_trajectory_error": 0.0,
        "neutral_xq_error": 0.0,
        "neutral_x0_error": 0.0,
        "neutral_cache_alignment_error": 0.0,
        "injection_clean_prediction_error": 0.0,
        "post_injection_preference_state_error": 0.0,
        "preference_final_trajectory_error": 0.0,
        "ade": 0.0,
        "fde": 0.0,
    }
    identity_scenes: List[Dict[str, Any]] = []
    branch_scenes: List[Dict[str, Any]] = []
    all_dynamic_current_state = True
    all_branch_isolated = True

    for index, entry in enumerate(entries):
        inputs = _cache_inputs(Path(entry["cache_path"]), model_args, device)
        sampling_seed = int(args.seed) + index

        neutral_identity = IdentityCleanPredictionEditor()
        preference_identity = IdentityCleanPredictionEditor()
        identity_rollout = _run_dual_rollout(
            model,
            inputs,
            sampling_seed=sampling_seed,
            neutral_editor=neutral_identity,
            preference_editor=preference_identity,
        )
        identity_metrics = _identity_scene_metrics(
            identity_rollout,
            expected_evaluations=expected_evaluations,
        )
        identity_diagnostics = preference_identity.diagnostics()
        if int(identity_diagnostics["model_evaluation_count"]) != expected_evaluations:
            raise RuntimeError("preference identity editor saw an unexpected NFE")
        if not all(
            bool(call["has_neutral_reference"])
            for call in identity_diagnostics["calls"]
        ):
            raise RuntimeError("preference identity editor missed a neutral reference")

        scene_identity = {
            "scene_token": entry["scene_token"],
            "filename": entry["filename"],
            "sampling_seed": sampling_seed,
            **identity_metrics,
        }
        identity_scenes.append(scene_identity)
        for key in identity_errors:
            identity_errors[key] = max(
                float(identity_errors[key]), float(identity_metrics[key])
            )
        all_dynamic_current_state = (
            all_dynamic_current_state
            and bool(identity_metrics["dynamic_current_state_passed"])
        )

        probe_neutral = IdentityCleanPredictionEditor()
        probe = _TinyPreferenceProbeEditor(
            evaluation_index=int(args.probe_evaluation_index),
            magnitude=float(args.probe_magnitude),
        )
        probe_rollout = _run_dual_rollout(
            model,
            inputs,
            sampling_seed=sampling_seed,
            neutral_editor=probe_neutral,
            preference_editor=probe,
        )
        branch_metrics = _branch_scene_metrics(
            identity_rollout,
            probe_rollout,
            probe,
            injection_index=int(args.probe_evaluation_index),
        )
        scene_branch = {
            "scene_token": entry["scene_token"],
            "filename": entry["filename"],
            "sampling_seed": sampling_seed,
            **branch_metrics,
        }
        branch_scenes.append(scene_branch)
        for key in branch_errors:
            branch_errors[key] = max(
                float(branch_errors[key]), float(branch_metrics[key])
            )
        all_branch_isolated = (
            all_branch_isolated and bool(branch_metrics["branch_isolation_passed"])
        )

    identity_passed = (
        all(value <= _IDENTITY_TOLERANCE for value in identity_errors.values())
        and all_dynamic_current_state
    )
    identity_report = {
        "schema_version": "preference_flow_step2_dual_identity_v1",
        "base_checkpoint": str(checkpoint_path),
        "model_args": str(args_path),
        "cache_root": str(cache_root.resolve()),
        "scene_token_file": str(scene_token_path),
        "seed": int(args.seed),
        "device": str(device),
        "weight_source": str(model_meta["weight_source"]),
        "checkpoint_coverage": float(model_meta["coverage"]),
        "scene_count": len(identity_scenes),
        "evaluations_per_scene": expected_evaluations,
        **identity_errors,
        "dynamic_current_state_passed": all_dynamic_current_state,
        "branch_isolation_passed": all_branch_isolated,
        "identity_tolerance": _IDENTITY_TOLERANCE,
        "passed": identity_passed,
        "per_scene": identity_scenes,
    }
    branch_report = {
        "schema_version": "preference_flow_step2_branch_isolation_v1",
        "base_checkpoint": str(checkpoint_path),
        "model_args": str(args_path),
        "cache_root": str(cache_root.resolve()),
        "scene_token_file": str(scene_token_path),
        "seed": int(args.seed),
        "device": str(device),
        "weight_source": str(model_meta["weight_source"]),
        "checkpoint_coverage": float(model_meta["coverage"]),
        "scene_count": len(branch_scenes),
        "evaluations_per_scene": expected_evaluations,
        "probe_magnitude": float(args.probe_magnitude),
        "probe_evaluation_index": int(args.probe_evaluation_index),
        **branch_errors,
        "branch_isolation_passed": all_branch_isolated,
        "dynamic_current_state_passed": all_dynamic_current_state,
        "passed": all_branch_isolated,
        "per_scene": branch_scenes,
    }
    output_dir = Path(args.output_dir).expanduser()
    identity_path = output_dir / "step2_dual_identity_regression.json"
    branch_path = output_dir / "step2_branch_isolation.json"
    _write_report(identity_path, identity_report, overwrite=bool(args.overwrite))
    _write_report(branch_path, branch_report, overwrite=bool(args.overwrite))
    if not identity_passed or not all_branch_isolated:
        raise AssertionError(
            "Step-2 dual-stream verification failed.  Measured reports: "
            f"{identity_path} and {branch_path}"
        )
    return identity_path, branch_path


def main() -> None:
    args = _parser().parse_args()
    identity_path, branch_path = run(args)
    print(f"Step-2 dual identity regression passed: {identity_path}")
    print(f"Step-2 branch isolation regression passed: {branch_path}")


if __name__ == "__main__":
    main()

"""Verify Step-4 Preference Flow Adapter on frozen StylePlanner DPM streams.

This is a CUDA-friendly regression program, not a trainer.  It uses the real
Step-3 vector field and adapter inside the existing preference clean-prediction
callback, while the base StylePlanner checkpoint stays frozen and unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from baseline.model.style_planner.preference_flow import (
    EgoTrajectoryResidualDecoder,
    PreferenceFlowCleanPredictionEditor,
    PreferenceFlowConfig,
    SmoothLongitudinalTrajectoryResidualDecoder,
)
from research_v1.execution.preference_flow.run_step1_base_regression import (
    _build_model_args,
    _cache_inputs,
    _checkpoint_state,
    _existing_file,
    _load_frozen_styleplanner,
    _max_abs_difference,
    _model_args_path,
    _scene_entries,
    _trajectory_ade_fde,
    _write_report,
)
from research_v1.execution.preference_flow.run_step2_dual_regression import (
    _IDENTITY_TOLERANCE,
    _identity_scene_metrics,
    _max_trace_tensor_difference,
    _run_dual_rollout,
)
from baseline.model.style_planner.preference_flow import (
    CLEAN_PREDICTION_EDITOR_DISABLED,
)


class _ConstantPreferenceFlowProbe(nn.Module):
    """Test-only nonzero field; never part of a checkpoint or training path."""

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        diffusion_time: torch.Tensor,
        preference_coordinate: torch.Tensor,
    ) -> torch.Tensor:
        del condition, diffusion_time, preference_coordinate
        return torch.ones_like(state)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Step-4 Preference Flow Adapter with a frozen StylePlanner "
            "checkpoint.  This command never trains or saves model weights."
        )
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--scene-token-file", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-scene-count",
        type=int,
        default=5,
        help="Fail rather than silently certify a different cohort size.",
    )
    parser.add_argument("--model-args", default=None)
    parser.add_argument("--normalization-file-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefer-ema", action="store_true")
    parser.add_argument(
        "--zero-initialized-rho",
        type=float,
        default=0.75,
        help="Nonzero endpoint used to prove the zero-initialized production adapter is inert.",
    )
    parser.add_argument(
        "--probe-rho",
        type=float,
        default=0.5,
        help="Nonzero endpoint used only by the test-only constant flow probe.",
    )
    parser.add_argument(
        "--probe-final-ego-delta",
        type=float,
        default=1e-3,
        help="Target final-horizon normalized ego-x residual for the flow probe.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_endpoint(name: str, value: float, config: PreferenceFlowConfig) -> None:
    numeric = float(value)
    if not torch.isfinite(torch.tensor(numeric)).item():
        raise ValueError(f"{name} must be finite")
    if numeric < float(config.rho_min) or numeric > float(config.rho_max):
        raise ValueError(
            f"{name} must lie in [{config.rho_min}, {config.rho_max}], got {numeric}"
        )


def _make_production_editor(
    *,
    rho: float,
    config: PreferenceFlowConfig,
    device: torch.device,
    state_normalizer: Any,
) -> PreferenceFlowCleanPredictionEditor:
    decoder = SmoothLongitudinalTrajectoryResidualDecoder(
        config.latent_dim,
        ego_mean=state_normalizer.mean[0, 0],
        ego_std=state_normalizer.std[0, 0],
    )
    editor = PreferenceFlowCleanPredictionEditor(
        rho=float(rho),
        config=config,
        trajectory_decoder=decoder,
    )
    return editor.to(device).eval()


def _make_probe_editor(
    *,
    rho: float,
    final_ego_delta: float,
    config: PreferenceFlowConfig,
    device: torch.device,
) -> PreferenceFlowCleanPredictionEditor:
    if float(rho) == 0.0:
        raise ValueError("--probe-rho must be nonzero")
    if float(final_ego_delta) <= 0.0:
        raise ValueError("--probe-final-ego-delta must be positive")
    decoder = EgoTrajectoryResidualDecoder(
        config.latent_dim,
        zero_initialize_output=False,
    )
    with torch.no_grad():
        decoder.projection.weight.zero_()
        decoder.projection.bias.zero_()
        # Constant V=1 gives delta-z=rho.  The decoder's terminal ramp is one,
        # so this maps to the requested small final ego-x residual.
        decoder.projection.weight[0, 0] = float(final_ego_delta) / float(rho)
    editor = PreferenceFlowCleanPredictionEditor(
        rho=float(rho),
        config=config,
        vector_field=_ConstantPreferenceFlowProbe(),
        trajectory_decoder=decoder,
    )
    return editor.to(device).eval()


def _adapter_record_metrics(
    editor: PreferenceFlowCleanPredictionEditor,
    trace: Sequence[Any],
    *,
    expected_evaluations: int,
    expected_device: torch.device,
    expected_rho: float,
) -> Dict[str, Any]:
    records = editor.records()
    if len(records) != expected_evaluations or len(trace) != expected_evaluations:
        raise RuntimeError(
            "adapter/trace evaluation count mismatch: "
            f"adapter={len(records)}, trace={len(trace)}, expected={expected_evaluations}"
        )
    indices_aligned = all(
        int(record.model_evaluation_index) == index
        and int(trace[index].model_evaluation_index) == index
        for index, record in enumerate(records)
    )
    time_error = max(
        abs(float(record.diffusion_time) - float(trace[index].diffusion_time[0].item()))
        for index, record in enumerate(records)
    )
    rho_error = max(
        float((record.rho - float(expected_rho)).abs().max().item())
        for record in records
    )
    placement_and_shape = all(
        tuple(record.condition.shape)
        == (int(record.rho.shape[0]), int(editor.config.condition_dim))
        and tuple(record.latent_start.shape)
        == (int(record.rho.shape[0]), int(editor.config.latent_dim))
        and tuple(record.latent_end.shape)
        == tuple(record.latent_start.shape)
        and record.condition.device == trace[index].clean_prediction.device
        and record.latent_start.device == trace[index].clean_prediction.device
        and record.latent_end.device == trace[index].clean_prediction.device
        and record.ego_future_residual.device == trace[index].clean_prediction.device
        and record.condition.dtype == trace[index].clean_prediction.dtype
        and record.latent_start.dtype == trace[index].clean_prediction.dtype
        and record.latent_end.dtype == trace[index].clean_prediction.dtype
        and record.ego_future_residual.dtype == trace[index].clean_prediction.dtype
        and record.clean_prediction_device == str(trace[index].clean_prediction.device)
        and record.clean_prediction_dtype == str(trace[index].clean_prediction.dtype)
        and bool(torch.isfinite(record.condition).all().item())
        and bool(torch.isfinite(record.latent_start).all().item())
        and bool(torch.isfinite(record.latent_end).all().item())
        and bool(torch.isfinite(record.ego_future_residual).all().item())
        for index, record in enumerate(records)
    )
    vector_parameters = list(editor.vector_field.parameters())
    decoder_parameters = list(editor.trajectory_decoder.parameters())
    actual_device = trace[0].clean_prediction.device
    module_placement = all(
        parameter.device == actual_device
        and parameter.dtype == trace[0].clean_prediction.dtype
        for parameter in vector_parameters + decoder_parameters
    )
    requested_device_type_matches = actual_device.type == expected_device.type
    return {
        "adapter_evaluation_count": len(records),
        "adapter_indices_aligned": indices_aligned,
        "adapter_diffusion_time_alignment_error": float(time_error),
        "adapter_rho_error": float(rho_error),
        "adapter_device_dtype_shape_passed": bool(
            placement_and_shape and module_placement and requested_device_type_matches
        ),
        "adapter_max_latent_displacement": max(
            float((record.latent_end - record.latent_start).abs().max().item())
            for record in records
        ),
        "adapter_max_ego_future_residual": max(
            float(record.ego_future_residual.abs().max().item()) for record in records
        ),
        "adapter_all_calls_nonzero_ego_future_residual": all(
            float(record.ego_future_residual.abs().max().item()) > 0.0
            for record in records
        ),
        "adapter_max_ego_current_residual": max(
            float(record.ego_current_residual_abs_max) for record in records
        ),
        "adapter_max_non_ego_direct_residual": max(
            float(record.non_ego_residual_abs_max) for record in records
        ),
    }


def _identity_adapter_metrics(
    rollout: Mapping[str, Any],
    editor: PreferenceFlowCleanPredictionEditor,
    *,
    expected_evaluations: int,
    device: torch.device,
    rho: float,
) -> Dict[str, Any]:
    metrics = _identity_scene_metrics(
        rollout,
        expected_evaluations=expected_evaluations,
    )
    result = rollout["result"]
    adapter = _adapter_record_metrics(
        editor,
        result.preference_trace,
        expected_evaluations=expected_evaluations,
        expected_device=device,
        expected_rho=rho,
    )
    exact_identity = all(
        float(metrics[key]) <= _IDENTITY_TOLERANCE
        for key in (
            "same_initial_noise_error",
            "max_abs_diffusion_time_error",
            "max_abs_log_snr_error",
            "max_abs_neutral_preference_xq_error",
            "max_abs_neutral_preference_x0_error",
            "max_abs_final_trajectory_error",
            "ade",
            "fde",
        )
    )
    adapter_inert = (
        float(adapter["adapter_max_latent_displacement"]) == 0.0
        and float(adapter["adapter_max_ego_future_residual"]) == 0.0
        and float(adapter["adapter_max_ego_current_residual"]) == 0.0
        and float(adapter["adapter_max_non_ego_direct_residual"]) == 0.0
    )
    passed = bool(
        exact_identity
        and adapter_inert
        and metrics["dynamic_current_state_passed"]
        and adapter["adapter_indices_aligned"]
        and float(adapter["adapter_diffusion_time_alignment_error"]) == 0.0
        and float(adapter["adapter_rho_error"]) == 0.0
        and adapter["adapter_device_dtype_shape_passed"]
    )
    return {
        **metrics,
        **adapter,
        "adapter_inert": adapter_inert,
        "passed": passed,
    }


def _probe_metrics(
    reference_rollout: Mapping[str, Any],
    probe_rollout: Mapping[str, Any],
    probe_editor: PreferenceFlowCleanPredictionEditor,
    *,
    expected_evaluations: int,
    device: torch.device,
    rho: float,
) -> Dict[str, Any]:
    reference = reference_rollout["result"]
    probe = probe_rollout["result"]
    adapter = _adapter_record_metrics(
        probe_editor,
        probe.preference_trace,
        expected_evaluations=expected_evaluations,
        expected_device=device,
        expected_rho=rho,
    )
    neutral_final_error = _max_abs_difference(
        reference_rollout["neutral_prediction"],
        probe_rollout["neutral_prediction"],
        label="reference versus probe neutral trajectory",
    )
    neutral_xq_error = _max_trace_tensor_difference(
        reference.neutral_trace,
        probe.neutral_trace,
        attribute="current_state",
        label="reference versus probe neutral x_q",
    )
    neutral_x0_error = _max_trace_tensor_difference(
        reference.neutral_trace,
        probe.neutral_trace,
        attribute="clean_prediction",
        label="reference versus probe neutral x0",
    )
    first_clean_prediction_error = _max_abs_difference(
        probe.preference_trace[0].clean_prediction,
        probe.neutral_trace[0].clean_prediction,
        label="probe preference versus neutral clean x0 at first evaluation",
    )
    propagated_state_error = max(
        _max_abs_difference(
            probe.preference_trace[index].current_state,
            probe.neutral_trace[index].current_state,
            label=f"probe preference versus neutral x_q[{index}]",
        )
        for index in range(1, expected_evaluations)
    )
    final_ego_error = _max_abs_difference(
        probe_rollout["preference_prediction"][:, 0],
        probe_rollout["neutral_prediction"][:, 0],
        label="probe preference versus neutral final ego trajectory",
    )
    ade, fde = _trajectory_ade_fde(
        probe_rollout["neutral_prediction"],
        probe_rollout["preference_prediction"],
    )
    branch_isolated = (
        neutral_final_error <= _IDENTITY_TOLERANCE
        and neutral_xq_error <= _IDENTITY_TOLERANCE
        and neutral_x0_error <= _IDENTITY_TOLERANCE
    )
    direct_ego_only = (
        float(adapter["adapter_max_ego_current_residual"]) == 0.0
        and float(adapter["adapter_max_non_ego_direct_residual"]) == 0.0
        and float(adapter["adapter_max_ego_future_residual"]) > 0.0
    )
    passed = bool(
        branch_isolated
        and first_clean_prediction_error > 0.0
        and propagated_state_error > 0.0
        and final_ego_error > 0.0
        and direct_ego_only
        and adapter["adapter_all_calls_nonzero_ego_future_residual"]
        and float(adapter["adapter_max_latent_displacement"]) > 0.0
        and adapter["adapter_indices_aligned"]
        and float(adapter["adapter_diffusion_time_alignment_error"]) == 0.0
        and float(adapter["adapter_rho_error"]) == 0.0
        and adapter["adapter_device_dtype_shape_passed"]
    )
    return {
        **adapter,
        "neutral_final_trajectory_error": neutral_final_error,
        "neutral_xq_error": neutral_xq_error,
        "neutral_x0_error": neutral_x0_error,
        "first_preference_clean_prediction_error": first_clean_prediction_error,
        "post_first_edit_preference_state_error": propagated_state_error,
        "final_ego_trajectory_error": final_ego_error,
        "ade": ade,
        "fde": fde,
        "neutral_branch_isolated": branch_isolated,
        "direct_ego_only": direct_ego_only,
        "passed": passed,
    }


def _max_float_fields(
    scenes: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> Dict[str, float]:
    return {
        key: max(float(scene[key]) for scene in scenes)
        for key in keys
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

    flow_config = PreferenceFlowConfig()
    _validate_endpoint("--zero-initialized-rho", args.zero_initialized_rho, flow_config)
    _validate_endpoint("--probe-rho", args.probe_rho, flow_config)
    if float(args.probe_rho) == 0.0:
        raise ValueError("--probe-rho must be nonzero")

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
    if int(args.expected_scene_count) <= 0:
        raise ValueError("--expected-scene-count must be positive")
    if len(entries) != int(args.expected_scene_count):
        raise RuntimeError(
            "scene cohort size mismatch: "
            f"expected {args.expected_scene_count}, found {len(entries)}"
        )
    expected_evaluations = int(getattr(model_args, "diffusion_steps", 10)) + 1

    identity_scenes: List[Dict[str, Any]] = []
    probe_scenes: List[Dict[str, Any]] = []
    for index, entry in enumerate(entries):
        inputs = _cache_inputs(Path(entry["cache_path"]), model_args, device)
        sampling_seed = int(args.seed) + index

        zero_rho_editor = _make_production_editor(
            rho=0.0,
            config=flow_config,
            device=device,
            state_normalizer=model_args.state_normalizer,
        )
        zero_rho_rollout = _run_dual_rollout(
            model,
            inputs,
            sampling_seed=sampling_seed,
            neutral_editor=None,
            preference_editor=zero_rho_editor,
        )
        zero_rho_metrics = _identity_adapter_metrics(
            zero_rho_rollout,
            zero_rho_editor,
            expected_evaluations=expected_evaluations,
            device=device,
            rho=0.0,
        )

        zero_initialized_editor = _make_production_editor(
            rho=float(args.zero_initialized_rho),
            config=flow_config,
            device=device,
            state_normalizer=model_args.state_normalizer,
        )
        zero_initialized_rollout = _run_dual_rollout(
            model,
            inputs,
            sampling_seed=sampling_seed,
            neutral_editor=None,
            preference_editor=zero_initialized_editor,
        )
        zero_initialized_metrics = _identity_adapter_metrics(
            zero_initialized_rollout,
            zero_initialized_editor,
            expected_evaluations=expected_evaluations,
            device=device,
            rho=float(args.zero_initialized_rho),
        )
        identity_scenes.append(
            {
                "scene_token": entry["scene_token"],
                "filename": entry["filename"],
                "sampling_seed": sampling_seed,
                "rho_zero": zero_rho_metrics,
                "zero_initialized_nonzero_rho": zero_initialized_metrics,
            }
        )

        probe_editor = _make_probe_editor(
            rho=float(args.probe_rho),
            final_ego_delta=float(args.probe_final_ego_delta),
            config=flow_config,
            device=device,
        )
        probe_rollout = _run_dual_rollout(
            model,
            inputs,
            sampling_seed=sampling_seed,
            neutral_editor=None,
            preference_editor=probe_editor,
        )
        probe_metrics = _probe_metrics(
            zero_initialized_rollout,
            probe_rollout,
            probe_editor,
            expected_evaluations=expected_evaluations,
            device=device,
            rho=float(args.probe_rho),
        )
        probe_scenes.append(
            {
                "scene_token": entry["scene_token"],
                "filename": entry["filename"],
                "sampling_seed": sampling_seed,
                **probe_metrics,
            }
        )

    identity_zero_scenes = [scene["rho_zero"] for scene in identity_scenes]
    identity_nonzero_scenes = [
        scene["zero_initialized_nonzero_rho"] for scene in identity_scenes
    ]
    identity_error_keys = (
        "same_initial_noise_error",
        "max_abs_diffusion_time_error",
        "max_abs_log_snr_error",
        "max_abs_neutral_preference_xq_error",
        "max_abs_neutral_preference_x0_error",
        "max_abs_final_trajectory_error",
        "ade",
        "fde",
        "adapter_diffusion_time_alignment_error",
        "adapter_rho_error",
        "adapter_max_latent_displacement",
        "adapter_max_ego_future_residual",
        "adapter_max_ego_current_residual",
        "adapter_max_non_ego_direct_residual",
    )
    rho_zero_errors = _max_float_fields(identity_zero_scenes, identity_error_keys)
    zero_initialized_errors = _max_float_fields(
        identity_nonzero_scenes,
        identity_error_keys,
    )
    all_identity_passed = all(
        bool(scene["rho_zero"]["passed"])
        and bool(scene["zero_initialized_nonzero_rho"]["passed"])
        for scene in identity_scenes
    )
    all_device_dtype = all(
        bool(scene["rho_zero"]["adapter_device_dtype_shape_passed"])
        and bool(
            scene["zero_initialized_nonzero_rho"]["adapter_device_dtype_shape_passed"]
        )
        for scene in identity_scenes
    )
    all_probe_passed = all(bool(scene["passed"]) for scene in probe_scenes)

    identity_report = {
        "schema_version": "preference_flow_step4_adapter_identity_v1",
        "base_checkpoint": str(checkpoint_path),
        "model_args": str(args_path),
        "cache_root": str(cache_root.resolve()),
        "scene_token_file": str(scene_token_path),
        "seed": int(args.seed),
        "device": str(device),
        "weight_source": str(model_meta["weight_source"]),
        "checkpoint_coverage": float(model_meta["coverage"]),
        "adapter_parameters_loaded_from_checkpoint": False,
        "scene_count": len(identity_scenes),
        "evaluations_per_scene": expected_evaluations,
        "rho_zero": 0.0,
        "zero_initialized_nonzero_rho": float(args.zero_initialized_rho),
        "rho_zero_errors": rho_zero_errors,
        "zero_initialized_nonzero_rho_errors": zero_initialized_errors,
        "cuda_dtype_shape_passed": all_device_dtype,
        "rho_zero_exact_passed": all(
            bool(scene["rho_zero"]["passed"]) for scene in identity_scenes
        ),
        "zero_initialized_identity_passed": all(
            bool(scene["zero_initialized_nonzero_rho"]["passed"])
            for scene in identity_scenes
        ),
        "passed": all_identity_passed,
        "per_scene": identity_scenes,
    }
    probe_error_keys = (
        "neutral_final_trajectory_error",
        "neutral_xq_error",
        "neutral_x0_error",
        "adapter_diffusion_time_alignment_error",
        "adapter_rho_error",
        "adapter_max_ego_current_residual",
        "adapter_max_non_ego_direct_residual",
    )
    probe_report = {
        "schema_version": "preference_flow_step4_adapter_probe_v1",
        "base_checkpoint": str(checkpoint_path),
        "model_args": str(args_path),
        "cache_root": str(cache_root.resolve()),
        "scene_token_file": str(scene_token_path),
        "seed": int(args.seed),
        "device": str(device),
        "weight_source": str(model_meta["weight_source"]),
        "checkpoint_coverage": float(model_meta["coverage"]),
        "adapter_parameters_loaded_from_checkpoint": False,
        "scene_count": len(probe_scenes),
        "evaluations_per_scene": expected_evaluations,
        "probe_rho": float(args.probe_rho),
        "probe_final_ego_delta": float(args.probe_final_ego_delta),
        "max_errors": _max_float_fields(probe_scenes, probe_error_keys),
        "neutral_branch_isolated": all(
            bool(scene["neutral_branch_isolated"]) for scene in probe_scenes
        ),
        "direct_ego_only": all(
            bool(scene["direct_ego_only"]) for scene in probe_scenes
        ),
        "all_clean_predictions_edited": all(
            bool(scene["adapter_all_calls_nonzero_ego_future_residual"])
            for scene in probe_scenes
        ),
        "cuda_dtype_shape_passed": all(
            bool(scene["adapter_device_dtype_shape_passed"])
            for scene in probe_scenes
        ),
        "passed": all_probe_passed,
        "per_scene": probe_scenes,
    }
    output_dir = Path(args.output_dir).expanduser()
    identity_path = output_dir / "step4_adapter_identity_regression.json"
    probe_path = output_dir / "step4_adapter_probe_regression.json"
    _write_report(identity_path, identity_report, overwrite=bool(args.overwrite))
    _write_report(probe_path, probe_report, overwrite=bool(args.overwrite))
    if not all_identity_passed or not all_probe_passed:
        raise AssertionError(
            "Step-4 adapter verification failed.  Measured reports: "
            f"{identity_path} and {probe_path}"
        )
    return identity_path, probe_path


def main() -> None:
    args = _parser().parse_args()
    identity_path, probe_path = run(args)
    print(f"Step-4 adapter identity regression passed: {identity_path}")
    print(f"Step-4 adapter probe regression passed: {probe_path}")


if __name__ == "__main__":
    main()

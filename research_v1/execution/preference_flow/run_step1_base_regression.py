"""Run the real Step-1 disabled-versus-identity regression on the server.

This is intentionally a verification program, not a trainer and not a new
planner.  It loads one frozen StylePlanner base checkpoint and a user-curated
list of cache filenames, reuses the same sampling seed for every paired rollout,
and writes only measured results.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from baseline.model.style_planner.diffusion_planner import Diffusion_Planner
from baseline.model.style_planner.preference_flow import (
    CLEAN_PREDICTION_EDITOR_DISABLED,
    CLEAN_PREDICTION_EDITOR_IDENTITY,
    CleanPredictionTraceRecorder,
)
from baseline.utils.io import opendata
from baseline.utils.normalizer import ObservationNormalizer, StateNormalizer


_REQUIRED_CACHE_KEYS = (
    "ego_current_state",
    "neighbor_agents_past",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
    "static_objects",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether Step-1 identity clean-prediction editing exactly "
            "matches a frozen StylePlanner base checkpoint."
        )
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument(
        "--scene-token-file",
        required=True,
        help=(
            "JSON list of cache-relative .npz filenames, or a JSON object with "
            "a filenames/scenes/samples list whose entries explicitly contain "
            "filename or cache_filename.  Opaque scene tokens are rejected."
        ),
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-args",
        default=None,
        help=(
            "Exact args.json for the base checkpoint.  If omitted, only "
            "<base-checkpoint parent>/args.json is accepted; no recursive guess "
            "is made."
        ),
    )
    parser.add_argument(
        "--normalization-file-path",
        default=None,
        help="Explicitly replace normalization_file_path from args.json when needed.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prefer-ema",
        action="store_true",
        help="Use ema_state_dict when the checkpoint contains one.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacement of an existing step1_base_regression.json.",
    )
    return parser


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _existing_file(path_value: str | Path, label: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path.resolve()


def _model_args_path(checkpoint_path: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return _existing_file(explicit_path, "--model-args")
    adjacent = checkpoint_path.parent / "args.json"
    if adjacent.is_file():
        return adjacent.resolve()
    raise FileNotFoundError(
        "Cannot locate the exact model args.json.  Expected the deterministic "
        f"adjacent path {adjacent}, or pass --model-args explicitly."
    )


def _build_model_args(
    args_path: Path,
    *,
    mode: str,
    device: str,
    normalization_file_override: str | None,
) -> SimpleNamespace:
    payload = _read_json(args_path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"args.json must contain a JSON object: {args_path}")
    model_args = SimpleNamespace(**dict(payload))
    if normalization_file_override is not None:
        model_args.normalization_file_path = str(
            _existing_file(normalization_file_override, "--normalization-file-path")
        )
    if not getattr(model_args, "normalization_file_path", None):
        raise ValueError(
            "args.json has no normalization_file_path.  Pass the exact artifact "
            "with --normalization-file-path; this runner never guesses one."
        )
    normalization_path = _existing_file(
        str(model_args.normalization_file_path),
        "normalization_file_path",
    )
    model_args.normalization_file_path = str(normalization_path)

    # Baseline train.yaml calls the historical-neighbour count ``agent_num``.
    # A few older research exports use ``past_neighbor_num`` instead.  Resolve
    # this one naming alias explicitly; do not infer any numeric value.
    if not hasattr(model_args, "agent_num") and hasattr(model_args, "past_neighbor_num"):
        model_args.agent_num = int(model_args.past_neighbor_num)
    if not hasattr(model_args, "past_neighbor_num") and hasattr(model_args, "agent_num"):
        model_args.past_neighbor_num = int(model_args.agent_num)

    required_fields = (
        "agent_num",
        "predicted_neighbor_num",
        "future_len",
        "route_num",
        "lane_len",
        "hidden_dim",
        "decoder_depth",
        "num_heads",
        "diffusion_model_type",
    )
    missing_fields = [
        field for field in required_fields if not hasattr(model_args, field)
    ]
    if missing_fields:
        raise ValueError(
            "args.json is not a complete StylePlanner base configuration; "
            f"missing {missing_fields}"
        )
    guidance_fn = getattr(model_args, "guidance_fn", None)
    if guidance_fn is not None:
        raise ValueError(
            "This base regression can only reconstruct a serialized null "
            "guidance_fn.  The supplied args.json requests a non-null runtime "
            "callable, so refusing to substitute a different planner behavior."
        )

    model_args.device = str(device)
    model_args.guidance_fn = None
    model_args.clean_prediction_editor_mode = mode
    model_args.state_normalizer = StateNormalizer.from_json(model_args)
    model_args.observation_normalizer = ObservationNormalizer.from_json(model_args)
    return model_args


def _checkpoint_state(
    checkpoint_path: Path,
    *,
    prefer_ema: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Unsupported checkpoint payload type {type(payload)!r}: {checkpoint_path}"
        )
    source_name = "model"
    state: Any = payload.get("model", payload)
    if prefer_ema and payload.get("ema_state_dict") is not None:
        state = payload["ema_state_dict"]
        source_name = "ema_state_dict"
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint state is not a mapping: {checkpoint_path}")
    normalized = {
        str(key).replace("module.", "", 1): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    if not normalized:
        raise ValueError(f"Checkpoint has no tensor state_dict entries: {checkpoint_path}")
    return normalized, {
        "weight_source": source_name,
        "epoch": int(payload.get("epoch", -1)),
        "saved_loss": float(payload.get("loss", 0.0)),
    }


def _load_frozen_styleplanner(
    model_args: SimpleNamespace,
    state: Mapping[str, torch.Tensor],
    checkpoint_meta: Mapping[str, Any],
) -> Tuple[Diffusion_Planner, Dict[str, Any]]:
    model = Diffusion_Planner(model_args)
    target = model.state_dict()
    matched = {
        key: value
        for key, value in state.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    matched_numel = sum(int(target[key].numel()) for key in matched)
    total_numel = sum(int(value.numel()) for value in target.values())
    coverage = matched_numel / max(total_numel, 1)
    missing, unexpected = model.load_state_dict(dict(state), strict=False)
    if coverage < 0.999999 or missing or unexpected:
        raise RuntimeError(
            "Base checkpoint does not exactly match the supplied StylePlanner "
            "args.json.  Refusing flexible or cross-family loading: "
            f"coverage={coverage:.4%}, missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    device = torch.device(model_args.device)
    model.to(device).eval()
    metadata = dict(checkpoint_meta)
    metadata.update(
        {
            "coverage": coverage,
            "matched_parameter_tensors": len(matched),
            "target_parameter_tensors": len(target),
        }
    )
    return model, metadata


def _scene_entries(scene_token_path: Path, cache_root: Path) -> List[Dict[str, str]]:
    payload = _read_json(scene_token_path)
    raw_entries: Any = payload
    if isinstance(payload, Mapping):
        for key in ("filenames", "scenes", "samples"):
            if key in payload:
                raw_entries = payload[key]
                break
        else:
            raise ValueError(
                "scene-token file object must contain one of filenames, scenes, "
                "or samples; opaque token-only lookup is intentionally unsupported"
            )
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("scene-token file must contain a non-empty JSON list")

    resolved_root = cache_root.resolve()
    entries: List[Dict[str, str]] = []
    for index, raw_entry in enumerate(raw_entries):
        if isinstance(raw_entry, str):
            filename = raw_entry
            scene_token = raw_entry
        elif isinstance(raw_entry, Mapping):
            filename = raw_entry.get("filename", raw_entry.get("cache_filename"))
            scene_token = raw_entry.get(
                "scene_token",
                raw_entry.get("token", filename),
            )
        else:
            raise TypeError(
                f"scene entry {index} must be a string or object, got {type(raw_entry)!r}"
            )
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(
                f"scene entry {index} has no explicit filename/cache_filename"
            )
        candidate = Path(filename)
        if candidate.is_absolute():
            cache_path = candidate.resolve()
        else:
            cache_path = (resolved_root / candidate).resolve()
        try:
            cache_path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"scene entry {index} resolves outside --cache-root: {cache_path}"
            ) from error
        if cache_path.suffix.lower() != ".npz":
            raise ValueError(
                f"scene entry {index} must name a .npz cache file, got {cache_path}"
            )
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"scene entry {index} cache file is missing: {cache_path}"
            )
        entries.append(
            {
                "scene_token": str(scene_token),
                "filename": str(cache_path.relative_to(resolved_root)),
                "cache_path": str(cache_path),
            }
        )
    return entries


def _as_batched_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value).unsqueeze(0).to(device)


def _cache_inputs(
    cache_path: Path,
    model_args: SimpleNamespace,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    cache = opendata(str(cache_path))
    try:
        missing = [key for key in _REQUIRED_CACHE_KEYS if key not in cache]
        if missing:
            raise KeyError(f"cache sample {cache_path} is missing keys {missing}")
        agent_num = int(model_args.agent_num)
        inputs = {
            "ego_current_state": _as_batched_tensor(
                cache["ego_current_state"], device
            ),
            "neighbor_agents_past": _as_batched_tensor(
                cache["neighbor_agents_past"][:agent_num], device
            ),
            "lanes": _as_batched_tensor(cache["lanes"], device),
            "lanes_speed_limit": _as_batched_tensor(
                cache["lanes_speed_limit"], device
            ),
            "lanes_has_speed_limit": _as_batched_tensor(
                cache["lanes_has_speed_limit"], device
            ),
            "route_lanes": _as_batched_tensor(cache["route_lanes"], device),
            "route_lanes_speed_limit": _as_batched_tensor(
                cache["route_lanes_speed_limit"], device
            ),
            "route_lanes_has_speed_limit": _as_batched_tensor(
                cache["route_lanes_has_speed_limit"], device
            ),
            "static_objects": _as_batched_tensor(cache["static_objects"], device),
        }
    finally:
        cache.close()
    return inputs


def _clone_inputs(inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in inputs.items()}


def _set_sampling_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_rollout(
    model: Diffusion_Planner,
    inputs: Mapping[str, torch.Tensor],
    *,
    sampling_seed: int,
    observer: CleanPredictionTraceRecorder | None,
) -> Dict[str, torch.Tensor]:
    _set_sampling_seed(sampling_seed)
    rollout_inputs = _clone_inputs(inputs)
    rollout_inputs["return_dpm_intermediates"] = True
    if observer is not None:
        rollout_inputs["clean_prediction_observer"] = observer
    with torch.no_grad():
        _, outputs = model(rollout_inputs)
    prediction = outputs.get("prediction")
    intermediates = outputs.get("dpm_intermediates")
    if not torch.is_tensor(prediction) or not torch.is_tensor(intermediates):
        raise RuntimeError(
            "Step-1 regression diagnostics were not returned by StylePlanner; "
            "check that the Step-1 decoder patch is installed."
        )
    return {
        "prediction": prediction.detach().cpu(),
        "dpm_intermediates": intermediates.detach().cpu(),
    }


def _max_abs_difference(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    label: str,
) -> float:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise RuntimeError(
            f"{label} shape mismatch: {tuple(reference.shape)} versus "
            f"{tuple(candidate.shape)}"
        )
    if reference.numel() == 0:
        raise RuntimeError(f"{label} is empty")
    return float((reference - candidate).abs().max().item())


def _max_snapshot_difference(
    reference: Sequence[torch.Tensor],
    candidate: Sequence[torch.Tensor],
    *,
    label: str,
) -> float:
    if len(reference) != len(candidate):
        raise RuntimeError(
            f"{label} evaluation-count mismatch: {len(reference)} versus {len(candidate)}"
        )
    if not reference:
        raise RuntimeError(f"{label} has no clean-prediction snapshots")
    return max(
        _max_abs_difference(left, right, label=f"{label}[{index}]")
        for index, (left, right) in enumerate(zip(reference, candidate))
    )


def _trajectory_ade_fde(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> Tuple[float, float]:
    if reference.ndim != 4 or candidate.ndim != 4:
        raise RuntimeError(
            "prediction must have shape [B, P, future, state] for ADE/FDE"
        )
    # The planner's ego trajectory is P=0; ADE/FDE use its x/y coordinates.
    delta = candidate[:, 0, :, :2] - reference[:, 0, :, :2]
    distance = torch.linalg.vector_norm(delta, ord=2, dim=-1)
    return float(distance.mean().item()), float(distance[:, -1].mean().item())


def _identity_diagnostics(model: Diffusion_Planner) -> Dict[str, Any]:
    diagnostics = model.decoder.decoder.clean_prediction_editor_diagnostics()
    if diagnostics.get("editor") != CLEAN_PREDICTION_EDITOR_IDENTITY:
        raise RuntimeError(
            "Identity regression model did not construct the identity editor; "
            f"got diagnostics {diagnostics!r}"
        )
    return diagnostics


def _write_report(path: Path, report: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing regression result: {path}. "
            "Pass --overwrite only if replacement is intended."
        )
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")


def run(args: argparse.Namespace) -> Path:
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
    base_args = _build_model_args(
        args_path,
        mode=CLEAN_PREDICTION_EDITOR_DISABLED,
        device=str(device),
        normalization_file_override=args.normalization_file_path,
    )
    identity_args = _build_model_args(
        args_path,
        mode=CLEAN_PREDICTION_EDITOR_IDENTITY,
        device=str(device),
        normalization_file_override=args.normalization_file_path,
    )
    state, checkpoint_meta = _checkpoint_state(
        checkpoint_path,
        prefer_ema=bool(args.prefer_ema),
    )
    disabled_model, disabled_meta = _load_frozen_styleplanner(
        base_args,
        state,
        checkpoint_meta,
    )
    identity_model, identity_meta = _load_frozen_styleplanner(
        identity_args,
        state,
        checkpoint_meta,
    )
    if disabled_meta["coverage"] != identity_meta["coverage"]:
        raise RuntimeError("disabled and identity checkpoint coverage unexpectedly differ")

    entries = _scene_entries(scene_token_path, cache_root)
    expected_logical_nfe = int(getattr(base_args, "diffusion_steps", 10)) + 1
    aggregate = {
        "max_abs_clean_prediction_error": 0.0,
        "max_abs_intermediate_state_error": 0.0,
        "max_abs_final_trajectory_error": 0.0,
        "observer_guard_final_trajectory_error": 0.0,
        "observer_guard_intermediate_state_error": 0.0,
        "ade": 0.0,
        "fde": 0.0,
    }
    per_scene: List[Dict[str, Any]] = []
    denoiser_evaluation_count = 0
    identity_editor_call_count = 0

    for index, entry in enumerate(entries):
        inputs = _cache_inputs(Path(entry["cache_path"]), base_args, device)
        sampling_seed = int(args.seed) + index

        # This is the actual default disabled path: no x0 callback at all.
        disabled = _run_rollout(
            disabled_model,
            inputs,
            sampling_seed=sampling_seed,
            observer=None,
        )

        # A second disabled rollout installs only the read-only trace recorder.
        # It must exactly match the actual disabled rollout before its
        # solver-facing x0 values are used to compare against identity mode.
        disabled_observer = CleanPredictionTraceRecorder()
        observed_disabled = _run_rollout(
            disabled_model,
            inputs,
            sampling_seed=sampling_seed,
            observer=disabled_observer,
        )
        identity_observer = CleanPredictionTraceRecorder()
        identity = _run_rollout(
            identity_model,
            inputs,
            sampling_seed=sampling_seed,
            observer=identity_observer,
        )

        observer_guard_final = _max_abs_difference(
            disabled["prediction"],
            observed_disabled["prediction"],
            label="disabled versus observed-disabled prediction",
        )
        observer_guard_intermediate = _max_abs_difference(
            disabled["dpm_intermediates"],
            observed_disabled["dpm_intermediates"],
            label="disabled versus observed-disabled DPM intermediates",
        )
        clean_prediction_error = _max_snapshot_difference(
            disabled_observer.snapshots(),
            identity_observer.snapshots(),
            label="observed-disabled versus identity solver-facing x0",
        )
        intermediate_error = _max_abs_difference(
            disabled["dpm_intermediates"],
            identity["dpm_intermediates"],
            label="disabled versus identity DPM intermediates",
        )
        final_error = _max_abs_difference(
            disabled["prediction"],
            identity["prediction"],
            label="disabled versus identity prediction",
        )
        ade, fde = _trajectory_ade_fde(disabled["prediction"], identity["prediction"])

        disabled_trace = disabled_observer.diagnostics()
        identity_trace = identity_observer.diagnostics()
        identity_editor = _identity_diagnostics(identity_model)
        observed_nfe = int(disabled_trace["model_evaluation_count"])
        identity_calls = int(identity_editor["model_evaluation_count"])
        if observed_nfe != expected_logical_nfe:
            raise RuntimeError(
                "Unexpected logical DPM evaluation count for the fixed sampler: "
                f"expected {expected_logical_nfe}, got {observed_nfe}"
            )
        if int(identity_trace["model_evaluation_count"]) != observed_nfe:
            raise RuntimeError("identity observer and disabled observer have different NFE")
        if identity_calls != observed_nfe:
            raise RuntimeError("identity editor call count does not equal logical DPM NFE")
        if not identity_editor["calls"] or not bool(
            identity_editor["calls"][-1]["is_terminal_denoise"]
        ):
            raise RuntimeError("identity editor did not observe the denoise-to-zero terminal call")

        scene_metrics = {
            "scene_token": entry["scene_token"],
            "filename": entry["filename"],
            "sampling_seed": sampling_seed,
            "denoiser_evaluation_count": observed_nfe,
            "identity_editor_call_count": identity_calls,
            "max_abs_clean_prediction_error": clean_prediction_error,
            "max_abs_intermediate_state_error": intermediate_error,
            "max_abs_final_trajectory_error": final_error,
            "observer_guard_final_trajectory_error": observer_guard_final,
            "observer_guard_intermediate_state_error": observer_guard_intermediate,
            "ade": ade,
            "fde": fde,
        }
        per_scene.append(scene_metrics)
        denoiser_evaluation_count += observed_nfe
        identity_editor_call_count += identity_calls
        for key in aggregate:
            aggregate[key] = max(float(aggregate[key]), float(scene_metrics[key]))

    passed = all(value == 0.0 for value in aggregate.values())
    report = {
        "schema_version": "preference_flow_step1_base_regression_v1",
        "base_checkpoint": str(checkpoint_path),
        "model_args": str(args_path),
        "cache_root": str(cache_root.resolve()),
        "scene_token_file": str(scene_token_path),
        "seed": int(args.seed),
        "device": str(device),
        "weight_source": str(disabled_meta["weight_source"]),
        "checkpoint_coverage": float(disabled_meta["coverage"]),
        "scene_count": len(per_scene),
        "logical_denoiser_evaluations_per_scene": expected_logical_nfe,
        "denoiser_evaluation_count": denoiser_evaluation_count,
        "identity_editor_call_count": identity_editor_call_count,
        "max_abs_clean_prediction_error": aggregate["max_abs_clean_prediction_error"],
        "max_abs_intermediate_state_error": aggregate["max_abs_intermediate_state_error"],
        "max_abs_final_trajectory_error": aggregate["max_abs_final_trajectory_error"],
        "observer_guard_final_trajectory_error": aggregate[
            "observer_guard_final_trajectory_error"
        ],
        "observer_guard_intermediate_state_error": aggregate[
            "observer_guard_intermediate_state_error"
        ],
        "ade": aggregate["ade"],
        "fde": aggregate["fde"],
        "ade_fde_definition": "identity versus disabled ego x/y trajectory",
        "passed": passed,
        "per_scene": per_scene,
    }
    report_path = Path(args.output_dir).expanduser() / "step1_base_regression.json"
    _write_report(report_path, report, overwrite=bool(args.overwrite))
    if not passed:
        raise AssertionError(
            "Step-1 identity regression did not preserve the base result exactly. "
            f"Measured failure report: {report_path}"
        )
    return report_path


def main() -> None:
    args = _parser().parse_args()
    report_path = run(args)
    print(f"Step-1 base regression passed: {report_path}")


if __name__ == "__main__":
    main()

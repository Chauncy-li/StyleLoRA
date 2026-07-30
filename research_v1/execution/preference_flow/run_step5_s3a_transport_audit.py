"""Step 5-S3A-R: verify neutral/expert longitudinal pairing without training.

The runner compares two frozen neutral sources on the same fixed cohort:
teacher-forced fixed-q clean x0 (the existing Step-5 training construction),
and a correctly normalized full-DPM inference rollout.  It never updates a
model or passes expert futures into the full-DPM input.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from baseline.model.style_planner.preference_flow import CLEAN_PREDICTION_EDITOR_DISABLED
from baseline.utils.io import opendata
from research_v1.execution.preference_flow.longitudinal_transport_targets import (
    DEFAULT_MAX_AXIS_SPREAD,
    DEFAULT_MAX_PROJECTION_DISTANCE_M,
    DEFAULT_NEUTRAL_RHO_BAND,
    AxisCoherence,
    axis_coherence_from_style,
    build_continuous_transport_target,
    json_ready,
    pairwise_axis_statistics,
    project_expert_onto_neutral_path,
    raw_physical_feasibility_audit,
    rho_distribution,
    scalar_summary,
    trajectory_sanity_audit,
    transport_coordinate_is_identifiable,
    validate_explicit_audit_output_dir,
)
from research_v1.execution.preference_flow.run_step1_base_regression import (
    _build_model_args,
    _cache_inputs,
    _checkpoint_state,
    _existing_file,
    _load_frozen_styleplanner,
    _model_args_path,
)
from research_v1.execution.preference_flow.run_step5_tiny_preference_learning import (
    _attach_fixed_phases,
    _neutral_clean_prediction,
    _noisy_state,
    _physical_ego_future,
    _prepare_batch,
)


_SCENES = ("straight_free_drive", "straight_car_follow")
_FIXED_Q_SOURCE = "teacher_forced_fixed_q_clean_prediction"
_FULL_DPM_SOURCE = "normalized_observation_full_dpm_prediction"
_S3AR_SCHEMA_VERSION = "preference_flow_step5_s3ar_neutral_pairing_v1"
_LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
INFERENCE_CACHE_KEYS = (
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
EXPERT_AUDIT_CACHE_KEYS = ("ego_agent_future", "neighbor_agents_future")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit fixed-q and normalized full-DPM neutral/expert longitudinal pairing; no training."
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--conditioning-index", required=True)
    parser.add_argument("--cohort-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-args", default=None)
    parser.add_argument("--normalization-file-path", default=None)
    parser.add_argument("--max-axis-spread", type=float, default=DEFAULT_MAX_AXIS_SPREAD)
    parser.add_argument("--max-projection-distance-m", type=float, default=DEFAULT_MAX_PROJECTION_DISTANCE_M)
    parser.add_argument("--neutral-rho-band", type=float, default=DEFAULT_NEUTRAL_RHO_BAND)
    parser.add_argument("--dt-seconds", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefer-ema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_inputs(inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in inputs.items()}


def _frozen_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def _frozen_change(model: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]) -> float:
    changes = [
        0.0 if parameter.numel() == 0 else float((parameter.detach().cpu() - snapshot[name]).abs().max().item())
        for name, parameter in model.named_parameters()
    ]
    return max(changes, default=0.0)


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    array = value.detach().to(torch.float64).cpu().reshape(-1)
    finite = torch.isfinite(array)
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "finite": bool(finite.all().item()),
    }
    if bool(finite.any().item()):
        data = array[finite].numpy()
        result.update(scalar_summary(data))
    else:
        result.update(scalar_summary(np.empty((0,), dtype=np.float64)))
    return result


def _input_summary(inputs: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {key: _tensor_summary(value) for key, value in inputs.items()}


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(f"conditioning JSONL row {line_number} is not an object")
            yield payload


def _relative_cache_key(value: Any, cache_root: Path) -> str | None:
    text = str(value).strip().replace("\\", "/")
    if not text or (len(text) >= 3 and text[1] == ":" and text[2] == "/"):
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(cache_root)
        except ValueError:
            return None
    if any(part == ".." for part in candidate.parts):
        return None
    return candidate.as_posix().lstrip("./")


def _scene_name(record: Mapping[str, Any]) -> str:
    return str(record.get("causal_scene_bucket", record.get("scene_bucket", "unknown")))


def _axis_count_report(labels: Sequence[AxisCoherence], neutral_rho_band: float) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for count in range(4):
        grouped = [label for label in labels if len(label.active_axis_indices) == count]
        coherent = [label for label in grouped if label.one_dimensional_coherent]
        candidates = [label.rho_star for label in grouped if label.rho_star is not None]
        targets = [label.target_rho_star for label in coherent if label.target_rho_star is not None]
        report[str(count)] = {
            "active_axis_count": count,
            "sample_count": len(grouped),
            "coherent_count": len(coherent),
            "coherent_ratio": None if not grouped else len(coherent) / len(grouped),
            "rho_star_distribution_all_candidates": rho_distribution(candidates, neutral_band=neutral_rho_band),
            "rho_star_distribution_coherent_only": rho_distribution(targets, neutral_band=neutral_rho_band),
            "supports_multi_axis_one_dimensional_evidence": count >= 2,
            "interpretation": (
                "scalar_label_only_not_multi_axis_evidence"
                if count == 1
                else "multi_axis_consistency_evidence" if count >= 2 else "no_active_axis"
            ),
        }
    return report


def _label_audit(
    index_path: Path,
    cache_root: Path,
    *,
    max_axis_spread: float,
    neutral_rho_band: float,
) -> tuple[dict[str, Any], dict[str, list[tuple[Mapping[str, Any], AxisCoherence]]]]:
    total_rows = 0
    scene_counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    by_scene: dict[str, list[AxisCoherence]] = {scene: [] for scene in _SCENES}
    index_by_key: dict[str, list[tuple[Mapping[str, Any], AxisCoherence]]] = defaultdict(list)
    for record in _read_jsonl(index_path):
        total_rows += 1
        scene = _scene_name(record)
        scene_counts[scene] += 1
        if scene not in _SCENES:
            excluded["unsupported_scene_bucket"] += 1
            continue
        label = axis_coherence_from_style(record.get("style_value_condition", ()), max_axis_spread=max_axis_spread)
        by_scene[scene].append(label)
        for reason in label.exclusion_reasons:
            excluded[reason] += 1
        key = _relative_cache_key(record.get("filename", ""), cache_root)
        if key is None:
            excluded["missing_or_nonrelative_conditioning_filename"] += 1
        else:
            index_by_key[key].append((record, label))

    def scene_report(scene: str) -> dict[str, Any]:
        labels = by_scene[scene]
        coherent = [label for label in labels if label.one_dimensional_coherent]
        incoherent = [label for label in labels if not label.one_dimensional_coherent]
        reasons = Counter(reason for label in incoherent for reason in label.exclusion_reasons)
        return {
            "sample_count": len(labels),
            "coherent_count": len(coherent),
            "incoherent_count": len(incoherent),
            "coherent_ratio": None if not labels else len(coherent) / len(labels),
            "multi_axis_coherent_count": sum(len(label.active_axis_indices) >= 2 for label in coherent),
            "incoherent_reasons": dict(sorted(reasons.items())),
            "rho_star_distribution_coherent_only": rho_distribution(
                [label.target_rho_star for label in coherent if label.target_rho_star is not None],
                neutral_band=neutral_rho_band,
            ),
            "pairwise_axis_statistics": pairwise_axis_statistics(labels),
            "active_axis_count_groups": _axis_count_report(labels, neutral_rho_band),
        }

    all_labels = [label for labels in by_scene.values() for label in labels]
    all_coherent = [label for label in all_labels if label.one_dimensional_coherent]
    return {
        "schema_version": _S3AR_SCHEMA_VERSION,
        "stage": "step5_s3ar_axis_coherence",
        "conditioning_index": str(index_path),
        "cache_root": str(cache_root),
        "axis_orientation": "V6 canonical normalized axes: 0=conservative, 0.5=neutral, 1=aggressive",
        "rho_star_formula": "2 * mean(active_axes) - 1",
        "rho_star_quantized": False,
        "max_axis_spread": float(max_axis_spread),
        "total_index_rows": total_rows,
        "scene_bucket_counts": dict(sorted(scene_counts.items())),
        "eligible_free_and_car_rows": len(all_labels),
        "coherent_count": len(all_coherent),
        "incoherent_count": len(all_labels) - len(all_coherent),
        "coherent_ratio": None if not all_labels else len(all_coherent) / len(all_labels),
        "multi_axis_coherent_count": sum(len(label.active_axis_indices) >= 2 for label in all_coherent),
        "exclusion_reasons": dict(sorted(excluded.items())),
        "rho_star_distribution_coherent_only": rho_distribution(
            [label.target_rho_star for label in all_coherent if label.target_rho_star is not None],
            neutral_band=neutral_rho_band,
        ),
        "pairwise_axis_statistics": pairwise_axis_statistics(all_labels),
        "active_axis_count_groups": _axis_count_report(all_labels, neutral_rho_band),
        "scene_reports": {scene: scene_report(scene) for scene in _SCENES},
        "duplicate_conditioning_cache_keys": int(sum(len(values) > 1 for values in index_by_key.values())),
    }, index_by_key


def _cohort_entries(path: Path, cache_root: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples") if isinstance(payload, Mapping) else payload
    if not isinstance(samples, list):
        raise TypeError("--cohort-file must contain a samples list")
    entries: list[dict[str, str]] = []
    for index, row in enumerate(samples):
        if not isinstance(row, Mapping):
            raise TypeError(f"cohort row {index} is not an object")
        filename = _relative_cache_key(row.get("filename", ""), cache_root)
        scene = str(row.get("causal_scene_bucket", ""))
        if filename is None or scene not in _SCENES:
            raise ValueError(f"invalid cohort row {index}")
        cache_path = (cache_root / filename).resolve()
        try:
            cache_path.relative_to(cache_root)
        except ValueError as error:
            raise ValueError(f"cohort row {index} escapes --cache-root") from error
        if not cache_path.is_file():
            raise FileNotFoundError(f"cohort row {index} cache missing: {cache_path}")
        entries.append({"filename": filename, "scene": scene, "cache_path": str(cache_path)})
    counts = Counter(entry["scene"] for entry in entries)
    if len(entries) != 12 or any(counts[scene] != 6 for scene in _SCENES):
        raise ValueError("Step-5-S3A-R requires the fixed 12-scene cohort: 6 free-drive and 6 car-follow")
    return entries


def _lookup_label(
    index_by_key: Mapping[str, list[tuple[Mapping[str, Any], AxisCoherence]]], filename: str
) -> tuple[AxisCoherence | None, list[str]]:
    matches = index_by_key.get(filename, [])
    if not matches:
        return None, ["conditioning_index_record_missing"]
    if len(matches) != 1:
        return None, ["ambiguous_conditioning_index_record"]
    return matches[0][1], []


def _load_inference_inputs(cache_path: Path, model_args: SimpleNamespace, device: torch.device) -> dict[str, torch.Tensor]:
    """Load observed inputs only; no expert future is read on this full-DPM path."""
    return _cache_inputs(cache_path, model_args, device)


def _normalize_full_dpm_inputs_once(
    raw_inputs: Mapping[str, torch.Tensor], model_args: SimpleNamespace
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    unexpected = sorted(set(raw_inputs).difference(INFERENCE_CACHE_KEYS))
    if unexpected:
        raise ValueError(f"full-DPM observed input contains unsupported keys: {unexpected}")
    normalized = model_args.observation_normalizer(_clone_inputs(raw_inputs))
    return normalized, {
        "observation_normalization_applied": True,
        "observation_normalization_count": 1,
        "prediction_state_inverse_applied_by_decoder": True,
        "additional_state_inverse_applied_by_audit": False,
        "expert_future_in_full_dpm_input": False,
        "expert_future_used_in_input": False,
    }


def _full_dpm_prediction(
    model: torch.nn.Module, normalized_inputs: Mapping[str, torch.Tensor], *, sampling_seed: int
) -> np.ndarray:
    _seed_everything(sampling_seed)
    with torch.no_grad():
        _, outputs = model(_clone_inputs(normalized_inputs))
    prediction = outputs.get("prediction")
    if not torch.is_tensor(prediction) or prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[1] < 1:
        raise RuntimeError("frozen StylePlanner did not return a [1, P, T, 4] full-DPM prediction")
    return prediction[0, 0, :, :2].detach().cpu().to(torch.float64).numpy()


def _load_expert_audit_arrays(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read expert future only after the full-DPM forward pass has completed."""
    cache = opendata(str(cache_path))
    try:
        missing = [key for key in ("ego_current_state", *EXPERT_AUDIT_CACHE_KEYS) if key not in cache]
        if missing:
            raise KeyError(f"cache {cache_path} missing {missing}")
        current = np.asarray(cache["ego_current_state"], dtype=np.float64).reshape(-1)[:2]
        ego_future = np.asarray(cache["ego_agent_future"], dtype=np.float32)
        neighbor_future = np.asarray(cache["neighbor_agents_future"], dtype=np.float32)
    finally:
        cache.close()
    return current, ego_future[..., :2], ego_future, neighbor_future


def _fixed_q_prediction(
    model: torch.nn.Module,
    model_args: SimpleNamespace,
    raw_inputs: Mapping[str, torch.Tensor],
    ego_future: np.ndarray,
    neighbor_future: np.ndarray,
    phase: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, dict[str, Any]], dict[str, Any]]:
    """Use the exact Step-5 teacher-forced xq construction, not an approximation."""
    raw = {
        "inputs": _clone_inputs(raw_inputs),
        "ego_future": torch.as_tensor(ego_future, dtype=torch.float32, device=device).unsqueeze(0),
        "neighbors_future": torch.as_tensor(neighbor_future, dtype=torch.float32, device=device).unsqueeze(0),
    }
    normalized_inputs, all_gt, _ego_future, _neighbors_future, _neighbor_mask = _prepare_batch(raw, model_args)
    diffusion_time = torch.tensor([float(phase["diffusion_time"])], dtype=torch.float32, device=device)
    noise = torch.as_tensor(phase["noise"], dtype=torch.float32, device=device).unsqueeze(0)
    xq, _log_snr = _noisy_state(model, all_gt, diffusion_time, noise)
    clean = _neutral_clean_prediction(model, normalized_inputs, xq, diffusion_time)
    physical = _physical_ego_future(clean, model_args)[0, :, :2].detach().cpu().to(torch.float64).numpy()
    return physical, _input_summary(normalized_inputs), {
        "neutral_source": _FIXED_Q_SOURCE,
        "expert_future_used_to_construct_training_xq": True,
        "expert_future_used_as_vector_field_condition": False,
        "fixed_q_phase_source": "run_step5_tiny_preference_learning._attach_fixed_phases",
        "fixed_q_diffusion_time": float(phase["diffusion_time"]),
        "fixed_q_noise_shape": list(noise.shape),
        "observation_normalization_applied": True,
        "observation_normalization_count": 1,
        "clean_prediction_state_inverse_applied_by_decoder": False,
        "state_inverse_applied_once_by_fixed_q_audit": True,
    }


def _transport_contract(projection: Any, rho_star: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    neutral, expert = projection.neutral_progress, projection.expert_projected_progress
    for lambda_value in _LAMBDA_GRID:
        target = build_continuous_transport_target(neutral, expert, rho_star, lambda_value)
        expected = neutral + lambda_value * (expert - neutral)
        rows.append({
            "lambda": lambda_value,
            "preference_coordinate_r": target.preference_coordinate_r,
            "coordinate_error": abs(target.preference_coordinate_r - lambda_value * rho_star),
            "neutral_identity_error": float(np.max(np.abs(target.target_progress - neutral))) if lambda_value == 0.0 else None,
            "expert_endpoint_error": float(abs(target.target_progress[-1] - expert[-1])) if lambda_value == 1.0 else None,
            "interpolation_error": float(np.max(np.abs(target.target_progress - expected))),
            "finite": bool(np.isfinite(target.target_progress).all() and np.isfinite(target.target_progress_residual).all()),
        })
    return {
        "lambda_values": list(_LAMBDA_GRID),
        "target_representation": "longitudinal_progress_only",
        "has_lateral_target": False,
        "rows": rows,
        "rho_zero_identity_error": next(row["neutral_identity_error"] for row in rows if row["lambda"] == 0.0),
        "rho_star_endpoint_error": next(row["expert_endpoint_error"] for row in rows if row["lambda"] == 1.0),
        "all_finite": all(bool(row["finite"]) for row in rows),
    }


def _all_valid_contract_outputs_finite(rows: Sequence[Mapping[str, Any]]) -> bool | None:
    valid = [row for row in rows if bool(row.get("transport_contract_valid", False))]
    if not valid:
        return None
    return all(bool(row["transport_contract"].get("all_finite", False)) for row in valid)


def _base_row(entry: Mapping[str, str], label: AxisCoherence | None, label_errors: Sequence[str], metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "filename": entry["filename"],
        "scene_type": entry["scene"],
        "neutral_source": metadata.get("neutral_source"),
        "neutral_generation_metadata": dict(metadata),
        "active_axis_labels": [] if label is None else list(label.active_axes),
        "active_axis_indices": [] if label is None else list(label.active_axis_indices),
        "active_axis_count": 0 if label is None else len(label.active_axis_indices),
        "axis_spread": None if label is None else label.axis_spread,
        "rho_star": None if label is None else label.rho_star,
        "target_rho_star": None if label is None else label.target_rho_star,
        "one_dimensional_coherent": False if label is None else label.one_dimensional_coherent,
        "target_representation": "longitudinal_progress_only",
        "has_lateral_target": False,
        "invalid_reasons": list(label_errors) + ([] if label is None else list(label.exclusion_reasons)),
    }


def _source_failure(
    entry: Mapping[str, str], label: AxisCoherence | None, label_errors: Sequence[str], source: str, error: Exception | str
) -> dict[str, Any]:
    row = _base_row(entry, label, label_errors, {"neutral_source": source})
    reason = f"audit_execution_error:{type(error).__name__}:{error}" if isinstance(error, Exception) else str(error)
    row.update({
        "neutral_sanity": {"all_finite": False, "neutral_generation_invalid": True, "invalid_reasons": [reason]},
        "projection": None,
        "feasibility_raw": None,
        "transport_contract": {"skipped": True, "reason": "neutral_generation_invalid"},
        "neutral_generation_valid": False,
        "path_comparable": False,
        "transport_contract_valid": False,
    })
    row["invalid_reasons"].extend(["neutral_generation_invalid", reason])
    row["invalid_reasons"] = sorted(set(row["invalid_reasons"]))
    return row


def _audit_neutral_source(
    entry: Mapping[str, str],
    label: AxisCoherence | None,
    label_errors: Sequence[str],
    *,
    neutral_future: np.ndarray,
    current_xy: np.ndarray,
    expert_future_xy: np.ndarray,
    neighbor_future: np.ndarray,
    raw_input_summary: Mapping[str, Any],
    normalized_input_summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
    max_projection_distance_m: float,
    dt_seconds: float,
) -> dict[str, Any]:
    row = _base_row(entry, label, label_errors, metadata)
    sanity = trajectory_sanity_audit(current_xy, neutral_future, expert_future_xy, dt_seconds=dt_seconds)
    sanity["raw_input_summary"] = dict(raw_input_summary)
    sanity["normalized_input_summary"] = dict(normalized_input_summary)
    row["neutral_sanity"] = sanity
    row["neutral_generation_valid"] = not bool(sanity["neutral_generation_invalid"])
    row["path_comparable"] = False
    row["transport_contract_valid"] = False
    if not row["neutral_generation_valid"]:
        row["invalid_reasons"].append("neutral_generation_invalid")
        row["invalid_reasons"].extend(sanity["invalid_reasons"])
        row["projection"] = None
        row["feasibility_raw"] = None
        row["transport_contract"] = {"skipped": True, "reason": "neutral_generation_invalid"}
    else:
        projection = project_expert_onto_neutral_path(
            current_xy,
            neutral_future,
            expert_future_xy,
            max_projection_distance_m=max_projection_distance_m,
        )
        row["projection"] = projection.to_dict()
        row["path_comparable"] = bool(projection.valid)
        row["feasibility_raw"] = raw_physical_feasibility_audit(
            current_xy,
            neutral_future,
            projection.expert_projected_progress,
            neighbor_future,
            dt_seconds=dt_seconds,
        )
        row["invalid_reasons"].extend(projection.invalid_reasons)
        if label is not None and label.one_dimensional_coherent and projection.valid:
            rho_star = float(label.target_rho_star)
            if transport_coordinate_is_identifiable(
                projection.neutral_progress, projection.expert_projected_progress, rho_star
            ):
                contract = _transport_contract(projection, rho_star)
                row["transport_contract_valid"] = bool(contract["all_finite"])
                if not row["transport_contract_valid"]:
                    row["invalid_reasons"].append("nonfinite_transport_contract")
            else:
                contract = {"skipped": True, "reason": "zero_rho_star_has_nonidentity_expert_progress"}
                row["invalid_reasons"].append("zero_rho_star_has_nonidentity_expert_progress")
        else:
            contract = {"skipped": True, "reason": "requires_coherent_label_and_comparable_projection"}
        row["transport_contract"] = contract
    row["invalid_reasons"] = sorted(set(str(reason) for reason in row["invalid_reasons"]))
    row["valid"] = bool(row["transport_contract_valid"])
    return row


def _source_report(
    stage: str,
    source: str,
    rows: Sequence[Mapping[str, Any]],
    common: Mapping[str, Any],
) -> dict[str, Any]:
    path_rows = [row for row in rows if bool(row["path_comparable"])]
    contract_rows = [row for row in rows if bool(row["transport_contract_valid"])]
    projection_distances = [
        distance
        for row in rows
        if isinstance(row.get("projection"), Mapping)
        for distance in row["projection"].get("projection_distance_m", [])
    ]
    ade_values = [
        row["neutral_sanity"].get("direct_ade_m")
        for row in rows
        if row["neutral_sanity"].get("direct_ade_m") is not None
    ]
    fde_values = [
        row["neutral_sanity"].get("direct_fde_m")
        for row in rows
        if row["neutral_sanity"].get("direct_fde_m") is not None
    ]
    finite = _all_valid_contract_outputs_finite(rows)
    status = "has_valid_longitudinal_contract" if contract_rows else "insufficient_no_valid_contract"
    return {
        "schema_version": _S3AR_SCHEMA_VERSION,
        "stage": stage,
        "neutral_source": source,
        **dict(common),
        "scene_count": len(rows),
        "path_comparable_count": len(path_rows),
        "valid_count": len(contract_rows),
        "label_coherent_and_path_comparable_count": sum(bool(row["one_dimensional_coherent"]) and bool(row["path_comparable"]) for row in rows),
        "all_valid_contract_outputs_finite": finite,
        "method_hypothesis_status": status,
        "projection_distance_all_attempted_m": scalar_summary(projection_distances),
        "direct_ade_m": scalar_summary(ade_values),
        "direct_fde_m": scalar_summary(fde_values),
        "invalid_scenes": [
            {"filename": row["filename"], "reasons": row["invalid_reasons"]}
            for row in rows if not bool(row["transport_contract_valid"])
        ],
        "rows": list(rows),
    }


def _comparison_report(
    fixed_rows: Sequence[Mapping[str, Any]], full_rows: Sequence[Mapping[str, Any]], common: Mapping[str, Any]
) -> dict[str, Any]:
    if len(fixed_rows) != len(full_rows):
        raise RuntimeError("fixed-q and full-DPM reports lost scene alignment")
    paired = []
    for fixed, full in zip(fixed_rows, full_rows):
        if fixed["filename"] != full["filename"]:
            raise RuntimeError("fixed-q and full-DPM report filename alignment failed")
        paired.append({
            "filename": fixed["filename"],
            "scene_type": fixed["scene_type"],
            "label_coherent": fixed["one_dimensional_coherent"],
            "fixed_q_path_comparable": fixed["path_comparable"],
            "full_dpm_path_comparable": full["path_comparable"],
            "fixed_q_transport_contract_valid": fixed["transport_contract_valid"],
            "full_dpm_transport_contract_valid": full["transport_contract_valid"],
            "path_compatibility_agrees": fixed["path_comparable"] == full["path_comparable"],
        })
    fixed_path = sum(bool(row["path_comparable"]) for row in fixed_rows)
    full_path = sum(bool(row["path_comparable"]) for row in full_rows)
    fixed_contract = sum(bool(row["transport_contract_valid"]) for row in fixed_rows)
    full_contract = sum(bool(row["transport_contract_valid"]) for row in full_rows)
    if fixed_contract == 0 and full_contract == 0:
        status = "insufficient_no_valid_contract_both_sources"
    elif fixed_contract > 0 and full_contract == 0:
        status = "fixed_q_local_pairing_present_full_dpm_pairing_unresolved"
    elif fixed_contract == 0 and full_contract > 0:
        status = "full_dpm_pairing_present_fixed_q_pairing_unresolved"
    else:
        status = "valid_longitudinal_contracts_present_in_both_sources"
    fixed_projection = _source_report("fixed_q", _FIXED_Q_SOURCE, fixed_rows, {})
    full_projection = _source_report("full_dpm", _FULL_DPM_SOURCE, full_rows, {})
    return {
        "schema_version": _S3AR_SCHEMA_VERSION,
        "stage": "step5_s3ar_comparison",
        **dict(common),
        "scene_count": len(paired),
        "label_coherent_count": sum(bool(row["label_coherent"]) for row in paired),
        "fixed_q_path_comparable_count": fixed_path,
        "full_dpm_path_comparable_count": full_path,
        "fixed_q_valid_count": fixed_contract,
        "full_dpm_valid_count": full_contract,
        "label_coherent_and_fixed_q_valid_count": sum(bool(row["label_coherent"]) and bool(row["fixed_q_transport_contract_valid"]) for row in paired),
        "label_coherent_and_full_dpm_valid_count": sum(bool(row["label_coherent"]) and bool(row["full_dpm_transport_contract_valid"]) for row in paired),
        "fixed_q_projection_p95_m": fixed_projection["projection_distance_all_attempted_m"]["p95"],
        "full_dpm_projection_p95_m": full_projection["projection_distance_all_attempted_m"]["p95"],
        "fixed_q_direct_ade_m": fixed_projection["direct_ade_m"],
        "full_dpm_direct_ade_m": full_projection["direct_ade_m"],
        "fixed_q_direct_fde_m": fixed_projection["direct_fde_m"],
        "full_dpm_direct_fde_m": full_projection["direct_fde_m"],
        "fixed_q_method_hypothesis_status": fixed_projection["method_hypothesis_status"],
        "full_dpm_method_hypothesis_status": full_projection["method_hypothesis_status"],
        "method_hypothesis_status": status,
        "path_compatibility_agreement_count": sum(bool(row["path_compatibility_agrees"]) for row in paired),
        "rows": paired,
    }


def _scene_dual_audit(
    entry: Mapping[str, str],
    label: AxisCoherence | None,
    label_errors: Sequence[str],
    *,
    model: torch.nn.Module,
    model_args: SimpleNamespace,
    phase: Mapping[str, Any],
    device: torch.device,
    full_dpm_seed: int,
    max_projection_distance_m: float,
    dt_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_path = Path(entry["cache_path"])
    try:
        raw_inputs = _load_inference_inputs(cache_path, model_args, device)
        raw_summary = _input_summary(raw_inputs)
    except Exception as error:
        return (
            _source_failure(entry, label, label_errors, _FIXED_Q_SOURCE, error),
            _source_failure(entry, label, label_errors, _FULL_DPM_SOURCE, error),
        )

    full_neutral = None
    full_summary: Mapping[str, Any] = {}
    full_metadata: Mapping[str, Any] = {"neutral_source": _FULL_DPM_SOURCE}
    full_error: Exception | None = None
    try:
        normalized_full, full_metadata = _normalize_full_dpm_inputs_once(raw_inputs, model_args)
        full_summary = _input_summary(normalized_full)
        full_neutral = _full_dpm_prediction(model, normalized_full, sampling_seed=full_dpm_seed)
    except Exception as error:
        full_error = error

    try:
        current, expert_future_xy, expert_future, neighbor_future = _load_expert_audit_arrays(cache_path)
    except Exception as error:
        return (
            _source_failure(entry, label, label_errors, _FIXED_Q_SOURCE, error),
            _source_failure(entry, label, label_errors, _FULL_DPM_SOURCE, error if full_error is None else full_error),
        )

    if full_error is None and full_neutral is not None:
        full_row = _audit_neutral_source(
            entry, label, label_errors,
            neutral_future=full_neutral, current_xy=current, expert_future_xy=expert_future_xy,
            neighbor_future=neighbor_future, raw_input_summary=raw_summary,
            normalized_input_summary=full_summary, metadata=full_metadata,
            max_projection_distance_m=max_projection_distance_m, dt_seconds=dt_seconds,
        )
    else:
        full_row = _source_failure(entry, label, label_errors, _FULL_DPM_SOURCE, full_error or "full_dpm_failure")

    try:
        fixed_neutral, fixed_summary, fixed_metadata = _fixed_q_prediction(
            model, model_args, raw_inputs, expert_future, neighbor_future, phase, device
        )
        fixed_row = _audit_neutral_source(
            entry, label, label_errors,
            neutral_future=fixed_neutral, current_xy=current, expert_future_xy=expert_future_xy,
            neighbor_future=neighbor_future, raw_input_summary=raw_summary,
            normalized_input_summary=fixed_summary, metadata=fixed_metadata,
            max_projection_distance_m=max_projection_distance_m, dt_seconds=dt_seconds,
        )
    except Exception as error:
        fixed_row = _source_failure(entry, label, label_errors, _FIXED_Q_SOURCE, error)
    return fixed_row, full_row


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    checkpoint = _existing_file(args.base_checkpoint, "--base-checkpoint")
    index = _existing_file(args.conditioning_index, "--conditioning-index")
    cohort = _existing_file(args.cohort_file, "--cohort-file")
    cache_root = Path(args.cache_root).expanduser().resolve()
    if not cache_root.is_dir():
        raise NotADirectoryError(f"--cache-root is not a directory: {cache_root}")
    if not np.isfinite(args.max_axis_spread) or args.max_axis_spread < 0.0:
        raise ValueError("--max-axis-spread must be finite and non-negative")
    if not np.isfinite(args.max_projection_distance_m) or args.max_projection_distance_m != 5.0:
        raise ValueError("S3A-R keeps --max-projection-distance-m fixed at 5.0")
    if not np.isfinite(args.neutral_rho_band) or args.neutral_rho_band < 0.0:
        raise ValueError("--neutral-rho-band must be finite and non-negative")
    if not np.isfinite(args.dt_seconds) or args.dt_seconds <= 0.0:
        raise ValueError("--dt-seconds must be finite and positive")
    output = validate_explicit_audit_output_dir(args.output_dir, _repository_root())
    try:
        output.relative_to(cache_root)
        output_is_cache_child = True
    except ValueError:
        output_is_cache_child = False
    if output_is_cache_child:
        raise ValueError("--output-dir must be separate from --cache-root")
    return checkpoint, index, cohort, cache_root, output


def run(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    checkpoint_path, index_path, cohort_path, cache_root, output_dir = _validate_args(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "step5_s3ar_axis_coherence.json",
        output_dir / "step5_s3ar_neutral_sanity.json",
        output_dir / "step5_s3ar_fixed_q_projection.json",
        output_dir / "step5_s3ar_full_dpm_projection.json",
        output_dir / "step5_s3ar_comparison.json",
    )
    if any(path.exists() for path in paths) and not bool(args.overwrite):
        raise FileExistsError("S3A-R output exists; pass --overwrite to replace only S3A-R JSON files")

    axis_report, index_by_key = _label_audit(
        index_path, cache_root,
        max_axis_spread=float(args.max_axis_spread), neutral_rho_band=float(args.neutral_rho_band),
    )
    _write_json(paths[0], axis_report)
    entries = _cohort_entries(cohort_path, cache_root)
    _seed_everything(int(args.seed))
    model_args = _build_model_args(
        _model_args_path(checkpoint_path, args.model_args),
        mode=CLEAN_PREDICTION_EDITOR_DISABLED,
        device=str(args.device),
        normalization_file_override=args.normalization_file_path,
    )
    state, checkpoint_meta = _checkpoint_state(checkpoint_path, prefer_ema=bool(args.prefer_ema))
    model, checkpoint_meta = _load_frozen_styleplanner(model_args, state, checkpoint_meta)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    snapshot = _frozen_snapshot(model)
    device = torch.device(str(args.device))
    phases: list[dict[str, Any]] = [{} for _ in entries]
    _attach_fixed_phases(
        phases,
        predicted_neighbors=int(model_args.predicted_neighbor_num),
        future_len=int(model_args.future_len),
        seed=int(args.seed),
    )

    fixed_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        label, label_errors = _lookup_label(index_by_key, entry["filename"])
        fixed_row, full_row = _scene_dual_audit(
            entry, label, label_errors,
            model=model, model_args=model_args, phase=phases[index], device=device,
            full_dpm_seed=int(args.seed) + index,
            max_projection_distance_m=float(args.max_projection_distance_m), dt_seconds=float(args.dt_seconds),
        )
        fixed_rows.append(fixed_row)
        full_rows.append(full_row)

    common = {
        "base_checkpoint": str(checkpoint_path),
        "checkpoint_metadata": checkpoint_meta,
        "conditioning_index": str(index_path),
        "cohort_file": str(cohort_path),
        "cache_root": str(cache_root),
        "seed": int(args.seed),
        "device": str(args.device),
        "max_projection_distance_m": 5.0,
        "optimizer_constructed": False,
        "training_steps": 0,
        "frozen_base_parameter_max_abs_change": _frozen_change(model, snapshot),
    }
    fixed_report = _source_report("step5_s3ar_fixed_q_projection", _FIXED_Q_SOURCE, fixed_rows, common)
    full_report = _source_report("step5_s3ar_full_dpm_projection", _FULL_DPM_SOURCE, full_rows, common)
    sanity_report = {
        "schema_version": _S3AR_SCHEMA_VERSION,
        "stage": "step5_s3ar_neutral_sanity",
        **common,
        "scene_count": len(entries),
        "rows": [
            {
                "filename": fixed["filename"],
                "scene_type": fixed["scene_type"],
                "fixed_q": {
                    "neutral_generation_metadata": fixed["neutral_generation_metadata"],
                    "neutral_sanity": fixed["neutral_sanity"],
                },
                "full_dpm": {
                    "neutral_generation_metadata": full["neutral_generation_metadata"],
                    "neutral_sanity": full["neutral_sanity"],
                },
            }
            for fixed, full in zip(fixed_rows, full_rows)
        ],
    }
    comparison = _comparison_report(fixed_rows, full_rows, common)
    _write_json(paths[1], sanity_report)
    _write_json(paths[2], fixed_report)
    _write_json(paths[3], full_report)
    _write_json(paths[4], comparison)
    return paths


def main() -> None:
    paths = run(_parser().parse_args())
    print("Step-5-S3A-R audit complete (no training):")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

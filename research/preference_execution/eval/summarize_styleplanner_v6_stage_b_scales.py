"""Paired diagnostics for a multi-scale Normal-Anchor CFG sweep.

The model-aware evaluator writes one JSONL row for every
``(variant, sample, seed, rho)`` rollout.  This module matches those rows
exactly and separates two questions:

* does the first CFG scale improve on the router-only / scale-1 baseline?
* does the next CFG scale continue that improvement or introduce a regression?

The diagnostics are deliberately descriptive.  They report signed physical
axis increments, direction flips, and trajectory-proxy deltas without hiding
the result behind a single hand-tuned pass threshold.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from research.preference_execution.eval.summarize_styleplanner_v6_multiseed import (
    CANONICAL_AXIS_BY_SCENE,
    GENERATED_SWEEP_NAME,
)


SCALE_DIAGNOSTIC_NAME = "v6_stage_b_scale_diagnostics.json"
SCALE_AXIS_TABLE_NAME = "v6_stage_b_scale_axis_transitions.csv"
_TOLERANCE = 1e-12


def _read_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"No generated sweep records found in {path}")
    return records


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _float_vector(value: Any, *, length: int = 3) -> list[float] | None:
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(vector) != length or not all(math.isfinite(item) for item in vector):
        return None
    return vector


def _bool_vector(value: Any, *, length: int = 3) -> list[bool] | None:
    try:
        vector = [bool(item) for item in value]
    except TypeError:
        return None
    return vector if len(vector) == length else None


def _numeric_summary(values: Iterable[float]) -> Dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(len(finite)),
        "mean": float(statistics.fmean(finite)),
        "median": float(statistics.median(finite)),
        "std": float(statistics.pstdev(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def _fraction(flags: Sequence[bool]) -> float | None:
    return float(sum(bool(flag) for flag in flags) / len(flags)) if flags else None


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, float]:
    sample_id = str(row.get("sample_id", ""))
    if not sample_id:
        raise ValueError("Generated sweep row is missing sample_id")
    seed_index = int(row.get("seed_index", 0))
    rho = float(row.get("rho_requested", math.nan))
    if seed_index < 0 or not math.isfinite(rho):
        raise ValueError(f"Invalid seed/rho key for sample_id={sample_id!r}")
    return sample_id, seed_index, rho


def _index_rows(
    path: str | Path,
) -> tuple[Dict[tuple[str, int, float], Dict[str, Any]], list[Dict[str, Any]]]:
    records = _read_jsonl(path)
    indexed: Dict[tuple[str, int, float], Dict[str, Any]] = {}
    for record in records:
        key = _row_key(record)
        if key in indexed:
            raise ValueError(f"Duplicate generated row key {key} in {path}")
        indexed[key] = record
    return indexed, records


def _normal_axis_values(
    rows: Mapping[tuple[str, int, float], Mapping[str, Any]],
) -> Dict[tuple[str, int], tuple[list[float], list[bool]]]:
    normals: Dict[tuple[str, int], tuple[list[float], list[bool]]] = {}
    for (sample_id, seed_index, rho), row in rows.items():
        if abs(float(rho)) > 1e-8:
            continue
        values = _float_vector(row.get("generated_axis_canonical_vec"))
        valid = _bool_vector(row.get("generated_axis_valid_mask"))
        if values is None or valid is None:
            continue
        normals[(sample_id, seed_index)] = (values, valid)
    return normals


def _signed_axis_response(
    *,
    row: Mapping[str, Any],
    normal: tuple[list[float], list[bool]] | None,
    axis: int,
    rho: float,
) -> float | None:
    if normal is None or abs(float(rho)) <= 1e-8:
        return None
    values = _float_vector(row.get("generated_axis_canonical_vec"))
    valid = _bool_vector(row.get("generated_axis_valid_mask"))
    if values is None or valid is None:
        return None
    normal_values, normal_valid = normal
    if not valid[axis] or not normal_valid[axis]:
        return None
    return float(math.copysign(1.0, rho) * (values[axis] - normal_values[axis]))


def _trajectory_transition(
    previous_rows: Mapping[tuple[str, int, float], Mapping[str, Any]],
    current_rows: Mapping[tuple[str, int, float], Mapping[str, Any]],
    *,
    keys: Sequence[tuple[str, int, float]],
    scene: str,
) -> Dict[str, Any]:
    metric_specs = {
        "max_abs_accel_mps2": "increase",
        "jerk_p90_mps3": "increase",
        "collision_rate": "increase",
        "overspeed_rate": "increase",
        "min_ellipse_clearance": "decrease",
    }
    output: Dict[str, Any] = {}
    for metric, adverse_direction in metric_specs.items():
        deltas: list[float] = []
        adverse: list[bool] = []
        for key in keys:
            previous = previous_rows[key]
            current = current_rows[key]
            if str(previous.get("scene_bucket", "")) != scene:
                continue
            if abs(float(key[2])) <= 1e-8:
                continue
            previous_metrics = previous.get("trajectory_metrics", {})
            current_metrics = current.get("trajectory_metrics", {})
            if not isinstance(previous_metrics, Mapping) or not isinstance(
                current_metrics, Mapping
            ):
                continue
            previous_value = _finite_float(previous_metrics.get(metric))
            current_value = _finite_float(current_metrics.get(metric))
            if previous_value is None or current_value is None:
                continue
            delta = float(current_value - previous_value)
            deltas.append(delta)
            adverse.append(
                delta > _TOLERANCE
                if adverse_direction == "increase"
                else delta < -_TOLERANCE
            )
        output[metric] = {
            "delta": _numeric_summary(deltas),
            "adverse_direction": adverse_direction,
            "adverse_fraction": _fraction(adverse),
        }
    return output


def _axis_transition(
    previous_rows: Mapping[tuple[str, int, float], Mapping[str, Any]],
    current_rows: Mapping[tuple[str, int, float], Mapping[str, Any]],
    *,
    previous_normals: Mapping[
        tuple[str, int], tuple[list[float], list[bool]]
    ],
    current_normals: Mapping[
        tuple[str, int], tuple[list[float], list[bool]]
    ],
    keys: Sequence[tuple[str, int, float]],
    scene: str,
    axis: int,
) -> Dict[str, Any]:
    previous_responses: list[float] = []
    current_responses: list[float] = []
    deltas: list[float] = []
    amplified: list[bool] = []
    regressed: list[bool] = []
    unchanged: list[bool] = []
    previous_direction_correct: list[bool] = []
    current_direction_correct: list[bool] = []
    direction_flips: list[bool] = []
    adverse_direction_flips: list[bool] = []
    corrected_direction_flips: list[bool] = []

    for key in keys:
        sample_id, seed_index, rho = key
        previous = previous_rows[key]
        current = current_rows[key]
        if str(previous.get("scene_bucket", "")) != scene:
            continue
        previous_response = _signed_axis_response(
            row=previous,
            normal=previous_normals.get((sample_id, seed_index)),
            axis=axis,
            rho=rho,
        )
        current_response = _signed_axis_response(
            row=current,
            normal=current_normals.get((sample_id, seed_index)),
            axis=axis,
            rho=rho,
        )
        if previous_response is None or current_response is None:
            continue
        delta = float(current_response - previous_response)
        previous_responses.append(previous_response)
        current_responses.append(current_response)
        deltas.append(delta)
        amplified.append(delta > _TOLERANCE)
        regressed.append(delta < -_TOLERANCE)
        unchanged.append(abs(delta) <= _TOLERANCE)
        previous_direction_correct.append(previous_response > 0.0)
        current_direction_correct.append(current_response > 0.0)
        direction_flips.append(
            (previous_response > _TOLERANCE and current_response < -_TOLERANCE)
            or (
                previous_response < -_TOLERANCE
                and current_response > _TOLERANCE
            )
        )
        adverse_direction_flips.append(
            previous_response > _TOLERANCE
            and current_response < -_TOLERANCE
        )
        corrected_direction_flips.append(
            previous_response < -_TOLERANCE
            and current_response > _TOLERANCE
        )

    delta_summary = _numeric_summary(deltas)
    return {
        "matched_nonzero_rho_rows": int(len(deltas)),
        "previous_signed_response": _numeric_summary(previous_responses),
        "current_signed_response": _numeric_summary(current_responses),
        "signed_response_delta": delta_summary,
        "amplified_fraction": _fraction(amplified),
        "regressed_fraction": _fraction(regressed),
        "unchanged_fraction": _fraction(unchanged),
        "previous_direction_correct_fraction": _fraction(
            previous_direction_correct
        ),
        "current_direction_correct_fraction": _fraction(
            current_direction_correct
        ),
        "direction_flip_fraction": _fraction(direction_flips),
        "adverse_direction_flip_fraction": _fraction(
            adverse_direction_flips
        ),
        "corrected_direction_flip_fraction": _fraction(
            corrected_direction_flips
        ),
        "mean_response_amplified": bool(
            delta_summary["mean"] is not None
            and float(delta_summary["mean"]) > _TOLERANCE
        ),
    }


def _transition_report(
    *,
    previous_spec: Mapping[str, Any],
    current_spec: Mapping[str, Any],
    previous_rows: Mapping[tuple[str, int, float], Mapping[str, Any]],
    current_rows: Mapping[tuple[str, int, float], Mapping[str, Any]],
) -> Dict[str, Any]:
    previous_keys = set(previous_rows)
    current_keys = set(current_rows)
    common_keys = sorted(previous_keys & current_keys)
    key_contract = {
        "previous_row_count": int(len(previous_keys)),
        "current_row_count": int(len(current_keys)),
        "matched_row_count": int(len(common_keys)),
        "missing_from_current": int(len(previous_keys - current_keys)),
        "extra_in_current": int(len(current_keys - previous_keys)),
        "exact_match": bool(previous_keys == current_keys),
    }
    if not common_keys:
        raise ValueError(
            "No matched (sample, seed, rho) rows between "
            f"{previous_spec['key']} and {current_spec['key']}"
        )

    previous_normals = _normal_axis_values(previous_rows)
    current_normals = _normal_axis_values(current_rows)
    scenes: Dict[str, Any] = {}
    non_amplified_axes: list[str] = []
    negatively_incremented_axes: list[str] = []
    amplified_axes: list[str] = []
    adverse_direction_flip_axes: list[str] = []
    for scene, axis_names in CANONICAL_AXIS_BY_SCENE.items():
        if not any(
            str(previous_rows[key].get("scene_bucket", "")) == scene
            for key in common_keys
        ):
            continue
        axes = {}
        for axis, axis_name in enumerate(axis_names):
            axis_report = _axis_transition(
                previous_rows,
                current_rows,
                previous_normals=previous_normals,
                current_normals=current_normals,
                keys=common_keys,
                scene=scene,
                axis=axis,
            )
            axes[axis_name] = axis_report
            mean_delta = axis_report["signed_response_delta"]["mean"]
            if mean_delta is not None:
                axis_key = f"{scene}/{axis_name}"
                if float(mean_delta) > _TOLERANCE:
                    amplified_axes.append(axis_key)
                else:
                    non_amplified_axes.append(axis_key)
                    if float(mean_delta) < -_TOLERANCE:
                        negatively_incremented_axes.append(axis_key)
            if (
                axis_report["adverse_direction_flip_fraction"] is not None
                and float(axis_report["adverse_direction_flip_fraction"])
                > 0.0
            ):
                adverse_direction_flip_axes.append(
                    f"{scene}/{axis_name}"
                )
        scenes[scene] = {
            "axes": axes,
            "trajectory_proxy_deltas": _trajectory_transition(
                previous_rows,
                current_rows,
                keys=common_keys,
                scene=scene,
            ),
        }

    if adverse_direction_flip_axes:
        localization = "new_wrong_direction_flip_detected"
    elif negatively_incremented_axes:
        localization = "negative_mean_increment_detected"
    elif non_amplified_axes and amplified_axes:
        localization = "mixed_axis_amplification"
    elif non_amplified_axes:
        localization = "no_measurable_axis_mean_amplified"
    else:
        localization = "all_measurable_axes_mean_amplified"
    return {
        "previous_variant": str(previous_spec["key"]),
        "current_variant": str(current_spec["key"]),
        "previous_cfg_guidance_scale": float(previous_spec["cfg_scale"]),
        "current_cfg_guidance_scale": float(current_spec["cfg_scale"]),
        "row_key_contract": key_contract,
        "scenes": scenes,
        "localization": {
            "code": localization,
            "axes_with_positive_mean_increment": amplified_axes,
            "axes_without_positive_mean_increment": non_amplified_axes,
            "axes_with_negative_mean_increment": negatively_incremented_axes,
            "axes_with_new_wrong_direction_flip": adverse_direction_flip_axes,
        },
    }


def _write_axis_table(
    path: Path,
    transitions: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "previous_variant",
        "current_variant",
        "previous_cfg_guidance_scale",
        "current_cfg_guidance_scale",
        "scene",
        "axis",
        "matched_nonzero_rho_rows",
        "previous_signed_response_mean",
        "current_signed_response_mean",
        "signed_response_delta_mean",
        "signed_response_delta_min",
        "signed_response_delta_max",
        "amplified_fraction",
        "regressed_fraction",
        "direction_flip_fraction",
        "adverse_direction_flip_fraction",
        "corrected_direction_flip_fraction",
        "current_direction_correct_fraction",
        "mean_response_amplified",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for transition in transitions:
            for scene, scene_report in transition["scenes"].items():
                for axis, axis_report in scene_report["axes"].items():
                    writer.writerow(
                        {
                            "previous_variant": transition["previous_variant"],
                            "current_variant": transition["current_variant"],
                            "previous_cfg_guidance_scale": transition[
                                "previous_cfg_guidance_scale"
                            ],
                            "current_cfg_guidance_scale": transition[
                                "current_cfg_guidance_scale"
                            ],
                            "scene": scene,
                            "axis": axis,
                            "matched_nonzero_rho_rows": axis_report[
                                "matched_nonzero_rho_rows"
                            ],
                            "previous_signed_response_mean": axis_report[
                                "previous_signed_response"
                            ]["mean"],
                            "current_signed_response_mean": axis_report[
                                "current_signed_response"
                            ]["mean"],
                            "signed_response_delta_mean": axis_report[
                                "signed_response_delta"
                            ]["mean"],
                            "signed_response_delta_min": axis_report[
                                "signed_response_delta"
                            ]["min"],
                            "signed_response_delta_max": axis_report[
                                "signed_response_delta"
                            ]["max"],
                            "amplified_fraction": axis_report[
                                "amplified_fraction"
                            ],
                            "regressed_fraction": axis_report[
                                "regressed_fraction"
                            ],
                            "direction_flip_fraction": axis_report[
                                "direction_flip_fraction"
                            ],
                            "adverse_direction_flip_fraction": axis_report[
                                "adverse_direction_flip_fraction"
                            ],
                            "corrected_direction_flip_fraction": axis_report[
                                "corrected_direction_flip_fraction"
                            ],
                            "current_direction_correct_fraction": axis_report[
                                "current_direction_correct_fraction"
                            ],
                            "mean_response_amplified": axis_report[
                                "mean_response_amplified"
                            ],
                        }
                    )


def summarize_stage_b_scale_sweep(
    checkpoint_root: str | Path,
    *,
    variant_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any] | None:
    """Write router->first-scale and adjacent-scale paired diagnostics."""

    root = Path(checkpoint_root)
    router_specs = [
        spec for spec in variant_specs if str(spec["variant"]) == "router_only"
    ]
    anchor_specs = sorted(
        (
            spec
            for spec in variant_specs
            if str(spec["variant"]) == "anchor_cfg"
        ),
        key=lambda spec: float(spec["cfg_scale"]),
    )
    if len(router_specs) != 1 or not anchor_specs:
        return None

    ordered_specs = [router_specs[0], *anchor_specs]
    indexed_by_variant: Dict[
        str, Dict[tuple[str, int, float], Dict[str, Any]]
    ] = {}
    paths: Dict[str, str] = {}
    for spec in ordered_specs:
        variant_key = str(spec["key"])
        path = root / variant_key / GENERATED_SWEEP_NAME
        indexed, _ = _index_rows(path)
        indexed_by_variant[variant_key] = indexed
        paths[variant_key] = str(path)

    transitions = []
    for previous_spec, current_spec in zip(
        ordered_specs[:-1], ordered_specs[1:]
    ):
        transitions.append(
            _transition_report(
                previous_spec=previous_spec,
                current_spec=current_spec,
                previous_rows=indexed_by_variant[str(previous_spec["key"])],
                current_rows=indexed_by_variant[str(current_spec["key"])],
            )
        )

    report = {
        "artifact": "styleplanner_v6_stage_b_cfg_scale_diagnostics",
        "checkpoint_root": str(root),
        "router_only_scale_equivalent": 1.0,
        "ordered_variants": [str(spec["key"]) for spec in ordered_specs],
        "cfg_guidance_scales": [
            float(spec["cfg_scale"]) for spec in anchor_specs
        ],
        "generated_axis_paths": paths,
        "transitions": transitions,
        "interpretation": {
            "signed_response": (
                "sign(rho) * (generated raw canonical axis at rho - the same "
                "variant/sample/seed rho=0 axis). Positive means the physical "
                "response follows the commanded aggressive direction."
            ),
            "router_to_first_scale": (
                "Diagnoses whether the first requested scale improves on the "
                "already-validated Stage-A / scale-1 response."
            ),
            "adjacent_scales": (
                "Diagnoses whether the next scale continues amplification or "
                "is the first point where axes or trajectory proxies regress."
            ),
            "no_single_threshold": (
                "Localization codes are descriptive. Final acceptance must "
                "jointly inspect axis strength/direction and trajectory proxies."
            ),
        },
    }
    _write_json(root / SCALE_DIAGNOSTIC_NAME, report)
    _write_axis_table(root / SCALE_AXIS_TABLE_NAME, transitions)
    return report


__all__ = [
    "SCALE_AXIS_TABLE_NAME",
    "SCALE_DIAGNOSTIC_NAME",
    "summarize_stage_b_scale_sweep",
]

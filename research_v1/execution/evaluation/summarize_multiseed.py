"""Summarize fixed-scene, multi-seed V6 rho sweeps.

The model-aware evaluator writes one JSONL row for every
``(sample, seed, rho)`` rollout.  This module keeps the three sources of
variation separate:

* scene/sample variation;
* diffusion seed variation;
* commanded rho variation.

It intentionally performs no model inference and can therefore be rerun or
deleted without changing training/runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from research_v1.stylization.schema import CANONICAL_AXIS_BY_SCENE
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    # Keep the post-hoc summarizer dependency-light. The model evaluator has
    # NumPy/PyTorch, while a downloaded JSONL may be summarized on a plain
    # Python installation. These names mirror research_v1.stylization.schema.
    CANONICAL_AXIS_BY_SCENE = {
        "straight_free_drive": (
            "speed_utilization",
            "accel_willingness",
            "speed_response_intensity",
        ),
        "straight_car_follow": (
            "headway_tightness_from_h",
            "ttc_tightness",
            "closing_tolerance",
        ),
        "straight_lane_change": (
            "initiation_aggressiveness",
            "lateral_commitment",
            "small_gap_acceptance_from_m_gap",
        ),
    }


GENERATED_SWEEP_NAME = "v6_generated_axis_sweep.jsonl"
SUMMARY_NAME = "v6_multiseed_robustness.json"
COMPARISON_NAME = "v6_multiseed_checkpoint_comparison.json"
LONG_TABLE_NAME = "v6_multiseed_axis_metrics.csv"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-seed robustness for one StylePlanner V6 evaluation "
            "root containing epoch_*/<variant>/v6_generated_axis_sweep.jsonl."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--checkpoint-tags",
        default="",
        help="Optional comma-separated subset such as epoch_2,epoch_5.",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated subset; empty discovers every variant.",
    )
    parser.add_argument(
        "--rho-zero-ade-tolerance",
        type=float,
        default=1e-6,
        help="Maximum rho=0 ego ADE to the checkpoint empty-condition trajectory.",
    )
    return parser


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


def _parse_csv_values(raw: str) -> list[str]:
    return list(
        dict.fromkeys(value.strip() for value in str(raw).split(",") if value.strip())
    )


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _float_vector(value: Any, *, length: int = 3) -> list[float]:
    try:
        array = [float(item) for item in value]
    except (TypeError, ValueError):
        raise ValueError(f"Expected a finite vector of length {length}, got {value!r}")
    if len(array) != length or not all(math.isfinite(item) for item in array):
        raise ValueError(f"Expected a finite vector of length {length}, got {value!r}")
    return array


def _bool_vector(value: Any, *, length: int = 3) -> list[bool]:
    try:
        array = [bool(item) for item in value]
    except TypeError:
        raise ValueError(f"Expected a boolean vector of length {length}, got {value!r}")
    if len(array) != length:
        raise ValueError(f"Expected a boolean vector of length {length}, got {value!r}")
    return array


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else None


def _maximum(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(max(finite)) if finite else None


def _minimum(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(min(finite)) if finite else None


def _rankdata(values: Sequence[float]) -> list[float]:
    """Dependency-free average ranks with stable tie handling."""

    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) + 1.0
        for index in order[start:stop]:
            ranks[index] = average_rank
        start = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]
    if max(x) - min(x) <= 1e-12 or max(y) - min(y) <= 1e-12:
        return None
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    x_mean = statistics.fmean(x_rank)
    y_mean = statistics.fmean(y_rank)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_rank, y_rank)
    )
    denominator = math.sqrt(
        sum((value - x_mean) ** 2 for value in x_rank)
        * sum((value - y_mean) ** 2 for value in y_rank)
    )
    if denominator <= 1e-12:
        return None
    correlation = numerator / denominator
    return float(correlation) if math.isfinite(correlation) else None


def _numeric_summary(values: Iterable[float | None]) -> Dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(statistics.fmean(finite)),
        "std": float(statistics.pstdev(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def _sample_id_digest(sample_ids: Iterable[str]) -> str:
    normalized = "\n".join(sorted(set(str(value) for value in sample_ids)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _assign_seed_indices(records: Sequence[MutableMapping[str, Any]]) -> int:
    """Attach a comparable seed-replica index to every sample row.

    New evaluator outputs provide ``seed_index`` explicitly.  For older
    outputs, the index is reconstructed by sorting the unique rollout seeds
    within each sample.  Rollout seeds include the dataset index, so grouping
    directly by their absolute value across samples would be incorrect.
    """

    seeds_by_sample: Dict[str, set[str]] = defaultdict(set)
    explicit = all("seed_index" in record for record in records)
    if explicit:
        seen = set()
        for record in records:
            seed_index = int(record["seed_index"])
            if seed_index < 0:
                raise ValueError("seed_index must be non-negative")
            record["_seed_index"] = seed_index
            seen.add(seed_index)
        return len(seen)

    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise ValueError("Generated sweep row is missing sample_id")
        seeds_by_sample[sample_id].add(str(record.get("seed", "single")))
    seed_index_by_sample: Dict[str, Dict[str, int]] = {}
    for sample_id, seed_values in seeds_by_sample.items():
        ordered = sorted(
            seed_values,
            key=lambda value: (
                0,
                int(value),
            )
            if value.lstrip("-").isdigit()
            else (1, value),
        )
        seed_index_by_sample[sample_id] = {
            value: index for index, value in enumerate(ordered)
        }
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        record["_seed_index"] = seed_index_by_sample[sample_id][
            str(record.get("seed", "single"))
        ]
    return max(len(values) for values in seeds_by_sample.values())


def _axis_report(
    sample_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    axis: int,
) -> Dict[str, Any]:
    rho_values: list[float] = []
    responses: list[float] = []
    sample_spans: list[float] = []
    extreme_correct = 0
    extreme_total = 0
    adjacent_correct = 0
    adjacent_total = 0

    for rows in sample_groups.values():
        ordered = sorted(rows, key=lambda row: float(row["rho_requested"]))
        normal_rows = [
            row for row in ordered if abs(float(row["rho_requested"])) <= 1e-8
        ]
        if len(normal_rows) != 1:
            continue
        normal = normal_rows[0]
        try:
            normal_valid = _bool_vector(normal["generated_axis_valid_mask"])[axis]
            normal_value = _float_vector(
                normal["generated_axis_canonical_vec"]
            )[axis]
        except (KeyError, TypeError, ValueError):
            continue
        if not normal_valid:
            continue

        valid_rows: list[tuple[float, float]] = []
        for row in ordered:
            try:
                valid = _bool_vector(row["generated_axis_valid_mask"])[axis]
                raw_value = _float_vector(
                    row["generated_axis_canonical_vec"]
                )[axis]
                rho = float(row["rho_requested"])
            except (KeyError, TypeError, ValueError):
                continue
            if valid and math.isfinite(rho):
                response = float(raw_value - normal_value)
                valid_rows.append((rho, response))
                rho_values.append(rho)
                responses.append(response)

        if len(valid_rows) < 2:
            continue
        valid_rows.sort(key=lambda item: item[0])
        low_rho, low_response = valid_rows[0]
        high_rho, high_response = valid_rows[-1]
        if high_rho > low_rho + 1e-8:
            span = float(high_response - low_response)
            sample_spans.append(span)
            extreme_total += 1
            extreme_correct += int(span > 0.0)
        for left, right in zip(valid_rows[:-1], valid_rows[1:]):
            if right[0] <= left[0] + 1e-8:
                continue
            adjacent_total += 1
            adjacent_correct += int(right[1] > left[1])

    return {
        "evaluated_sample_count": int(len(sample_spans)),
        "evaluated_row_count": int(len(responses)),
        "response_span": _mean(sample_spans),
        "sample_response_span_std": (
            float(statistics.pstdev(sample_spans)) if sample_spans else None
        ),
        "sample_response_span_min": _minimum(sample_spans),
        "sample_response_span_max": _maximum(sample_spans),
        "rho_response_spearman": _spearman(rho_values, responses),
        "extreme_direction_accuracy": (
            float(extreme_correct / extreme_total) if extreme_total else None
        ),
        "extreme_direction_count": int(extreme_total),
        "adjacent_direction_accuracy": (
            float(adjacent_correct / adjacent_total) if adjacent_total else None
        ),
        "adjacent_direction_count": int(adjacent_total),
    }


def _trajectory_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    rho_zero_ade_tolerance: float,
) -> Dict[str, Any]:
    trajectory_metrics = [
        row.get("trajectory_metrics", {})
        for row in rows
        if isinstance(row.get("trajectory_metrics", {}), Mapping)
    ]

    def metric_values(key: str) -> list[float]:
        values = [
            _finite_float(metrics.get(key)) for metrics in trajectory_metrics
        ]
        return [value for value in values if value is not None]

    normal_rows = [
        row for row in rows if abs(float(row.get("rho_requested", math.inf))) <= 1e-8
    ]
    normal_ade = [
        value
        for value in (
            _finite_float(row.get("trajectory_to_checkpoint_empty_ade"))
            for row in normal_rows
        )
        if value is not None
    ]
    normal_fde = [
        value
        for value in (
            _finite_float(row.get("trajectory_to_checkpoint_empty_fde"))
            for row in normal_rows
        )
        if value is not None
    ]
    rho_zero_max_ade = _maximum(normal_ade)
    return {
        "rollout_count": int(len(rows)),
        "max_abs_accel_mps2_mean": _mean(metric_values("max_abs_accel_mps2")),
        "max_abs_accel_mps2_max": _maximum(metric_values("max_abs_accel_mps2")),
        "jerk_p90_mps3_mean": _mean(metric_values("jerk_p90_mps3")),
        "jerk_p90_mps3_max": _maximum(metric_values("jerk_p90_mps3")),
        "collision_rate_mean": _mean(metric_values("collision_rate")),
        "collision_rate_max": _maximum(metric_values("collision_rate")),
        "overspeed_rate_mean": _mean(metric_values("overspeed_rate")),
        "overspeed_rate_max": _maximum(metric_values("overspeed_rate")),
        "min_ellipse_clearance_min": _minimum(
            metric_values("min_ellipse_clearance")
        ),
        "rho_zero_pair_count": int(len(normal_ade)),
        "rho_zero_to_empty_ade_mean": _mean(normal_ade),
        "rho_zero_to_empty_ade_max": rho_zero_max_ade,
        "rho_zero_to_empty_fde_max": _maximum(normal_fde),
        "rho_zero_exact_within_tolerance": bool(
            normal_ade
            and rho_zero_max_ade is not None
            and rho_zero_max_ade <= float(rho_zero_ade_tolerance)
        ),
        "rho_zero_ade_tolerance": float(rho_zero_ade_tolerance),
    }


def summarize_generated_sweep(
    generated_axis_path: str | Path,
    *,
    rho_zero_ade_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    records = _read_jsonl(generated_axis_path)
    detected_seed_count = _assign_seed_indices(records)
    checkpoint = str(records[0].get("checkpoint", Path(generated_axis_path).stem))
    variant = str(records[0].get("variant", Path(generated_axis_path).parent.name))

    seed_reports: Dict[str, Any] = {}
    seed_indices = sorted({int(record["_seed_index"]) for record in records})
    for seed_index in seed_indices:
        seed_rows = [
            record for record in records if int(record["_seed_index"]) == seed_index
        ]
        scene_reports: Dict[str, Any] = {}
        for scene, axis_names in CANONICAL_AXIS_BY_SCENE.items():
            scene_rows = [
                row for row in seed_rows if str(row.get("scene_bucket", "")) == scene
            ]
            if not scene_rows:
                continue
            sample_groups: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in scene_rows:
                sample_groups[str(row.get("sample_id", ""))].append(row)
            scene_reports[scene] = {
                "sample_count": int(len(sample_groups)),
                "sample_id_sha256": _sample_id_digest(sample_groups),
                "axes": {
                    axis_name: _axis_report(sample_groups, axis=axis)
                    for axis, axis_name in enumerate(axis_names)
                },
                "trajectory": _trajectory_report(
                    scene_rows,
                    rho_zero_ade_tolerance=rho_zero_ade_tolerance,
                ),
            }
        seed_reports[str(seed_index)] = {
            "seed_index": int(seed_index),
            "scenes": scene_reports,
        }

    aggregate_scenes: Dict[str, Any] = {}
    for scene, axis_names in CANONICAL_AXIS_BY_SCENE.items():
        per_seed_scene = [
            report["scenes"][scene]
            for report in seed_reports.values()
            if scene in report["scenes"]
        ]
        if not per_seed_scene:
            continue
        aggregate_axes: Dict[str, Any] = {}
        for axis_name in axis_names:
            per_seed_axis = [
                report["axes"][axis_name] for report in per_seed_scene
            ]
            aggregate_axes[axis_name] = {
                "response_span": _numeric_summary(
                    report["response_span"] for report in per_seed_axis
                ),
                "rho_response_spearman": _numeric_summary(
                    report["rho_response_spearman"] for report in per_seed_axis
                ),
                "extreme_direction_accuracy": _numeric_summary(
                    report["extreme_direction_accuracy"] for report in per_seed_axis
                ),
                "adjacent_direction_accuracy": _numeric_summary(
                    report["adjacent_direction_accuracy"] for report in per_seed_axis
                ),
                "evaluated_sample_count": _numeric_summary(
                    float(report["evaluated_sample_count"])
                    for report in per_seed_axis
                ),
                "all_seed_extreme_direction_correct": bool(
                    per_seed_axis
                    and all(
                        report["extreme_direction_accuracy"] is not None
                        and report["extreme_direction_accuracy"] >= 1.0 - 1e-12
                        for report in per_seed_axis
                    )
                ),
                "all_seed_response_spans_positive": bool(
                    per_seed_axis
                    and all(
                        report["response_span"] is not None
                        and report["response_span"] > 0.0
                        for report in per_seed_axis
                    )
                ),
            }

        per_seed_trajectory = [report["trajectory"] for report in per_seed_scene]
        aggregate_scenes[scene] = {
            "seed_count": int(len(per_seed_scene)),
            "sample_count_per_seed": _numeric_summary(
                float(report["sample_count"]) for report in per_seed_scene
            ),
            "sample_id_sha256_by_seed": {
                seed_key: seed_report["scenes"][scene]["sample_id_sha256"]
                for seed_key, seed_report in seed_reports.items()
                if scene in seed_report["scenes"]
            },
            "sample_ids_identical_across_seeds": bool(
                len(
                    {
                        report["sample_id_sha256"]
                        for report in per_seed_scene
                    }
                )
                == 1
            ),
            "axes": aggregate_axes,
            "trajectory": {
                "max_abs_accel_mps2_max_across_seeds": _maximum(
                    report["max_abs_accel_mps2_max"]
                    for report in per_seed_trajectory
                    if report["max_abs_accel_mps2_max"] is not None
                ),
                "jerk_p90_mps3_max_across_seeds": _maximum(
                    report["jerk_p90_mps3_max"]
                    for report in per_seed_trajectory
                    if report["jerk_p90_mps3_max"] is not None
                ),
                "collision_rate_max_across_seeds": _maximum(
                    report["collision_rate_max"]
                    for report in per_seed_trajectory
                    if report["collision_rate_max"] is not None
                ),
                "overspeed_rate_max_across_seeds": _maximum(
                    report["overspeed_rate_max"]
                    for report in per_seed_trajectory
                    if report["overspeed_rate_max"] is not None
                ),
                "min_ellipse_clearance_min_across_seeds": _minimum(
                    report["min_ellipse_clearance_min"]
                    for report in per_seed_trajectory
                    if report["min_ellipse_clearance_min"] is not None
                ),
                "rho_zero_to_empty_ade_max_across_seeds": _maximum(
                    report["rho_zero_to_empty_ade_max"]
                    for report in per_seed_trajectory
                    if report["rho_zero_to_empty_ade_max"] is not None
                ),
                "rho_zero_exact_all_seeds": bool(
                    per_seed_trajectory
                    and all(
                        report["rho_zero_exact_within_tolerance"]
                        for report in per_seed_trajectory
                    )
                ),
                "rho_zero_ade_tolerance": float(rho_zero_ade_tolerance),
            },
        }

    seed_indices_by_sample: Dict[str, set[int]] = defaultdict(set)
    for record in records:
        seed_indices_by_sample[str(record.get("sample_id", ""))].add(
            int(record["_seed_index"])
        )
    seed_counts = [len(values) for values in seed_indices_by_sample.values()]
    return {
        "artifact": "styleplanner_v6_multiseed_rho_robustness",
        "generated_axis_path": str(generated_axis_path),
        "checkpoint": checkpoint,
        "variant": variant,
        "generated_record_count": int(len(records)),
        "detected_seed_count": int(detected_seed_count),
        "seed_indices": seed_indices,
        "seed_replication_audit": {
            "sample_count": int(len(seed_indices_by_sample)),
            "minimum_seeds_per_sample": int(min(seed_counts)),
            "maximum_seeds_per_sample": int(max(seed_counts)),
            "consistent_across_samples": bool(
                min(seed_counts) == max(seed_counts) == detected_seed_count
            ),
        },
        "seed_reports": seed_reports,
        "aggregate_across_seeds": {
            "scenes": aggregate_scenes,
        },
        "interpretation": {
            "response_span": (
                "Mean per-sample canonical response at maximum rho minus "
                "minimum rho, after same-sample rho=0 anchoring."
            ),
            "seed_aggregation": (
                "The min field is the conservative seed result; the std field "
                "measures diffusion-seed sensitivity without adding scenes."
            ),
            "scope": (
                "Multi-seed replication audits diffusion stochasticity. "
                "Increasing scene count is a separate generalization test."
            ),
        },
    }


def _discover_sweeps(
    output_root: Path,
    *,
    checkpoint_tags: Sequence[str],
    variants: Sequence[str],
) -> list[Path]:
    checkpoint_filter = set(checkpoint_tags)
    variant_filter = set(variants)
    paths = []
    for path in sorted(output_root.glob(f"*/*/{GENERATED_SWEEP_NAME}")):
        checkpoint_tag = path.parent.parent.name
        variant = path.parent.name
        if checkpoint_filter and checkpoint_tag not in checkpoint_filter:
            continue
        if variant_filter and variant not in variant_filter:
            continue
        paths.append(path)
    if not paths:
        raise FileNotFoundError(
            f"No {GENERATED_SWEEP_NAME} files matched under {output_root}"
        )
    return paths


def _write_long_table(path: Path, reports: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "checkpoint",
        "variant",
        "seed_index",
        "scene",
        "axis",
        "sample_count",
        "response_span",
        "rho_response_spearman",
        "extreme_direction_accuracy",
        "extreme_direction_count",
        "adjacent_direction_accuracy",
        "adjacent_direction_count",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            for seed_key, seed_report in report["seed_reports"].items():
                for scene, scene_report in seed_report["scenes"].items():
                    for axis, axis_report in scene_report["axes"].items():
                        writer.writerow(
                            {
                                "checkpoint": report["checkpoint"],
                                "variant": report["variant"],
                                "seed_index": seed_key,
                                "scene": scene,
                                "axis": axis,
                                "sample_count": axis_report[
                                    "evaluated_sample_count"
                                ],
                                "response_span": axis_report["response_span"],
                                "rho_response_spearman": axis_report[
                                    "rho_response_spearman"
                                ],
                                "extreme_direction_accuracy": axis_report[
                                    "extreme_direction_accuracy"
                                ],
                                "extreme_direction_count": axis_report[
                                    "extreme_direction_count"
                                ],
                                "adjacent_direction_accuracy": axis_report[
                                    "adjacent_direction_accuracy"
                                ],
                                "adjacent_direction_count": axis_report[
                                    "adjacent_direction_count"
                                ],
                            }
                        )


def summarize_evaluation_root(
    output_root: str | Path,
    *,
    checkpoint_tags: Sequence[str] = (),
    variants: Sequence[str] = (),
    rho_zero_ade_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    root = Path(output_root)
    if rho_zero_ade_tolerance < 0.0 or not math.isfinite(rho_zero_ade_tolerance):
        raise ValueError("rho_zero_ade_tolerance must be finite and non-negative")
    paths = _discover_sweeps(
        root,
        checkpoint_tags=checkpoint_tags,
        variants=variants,
    )
    reports = []
    for path in paths:
        report = summarize_generated_sweep(
            path,
            rho_zero_ade_tolerance=rho_zero_ade_tolerance,
        )
        _write_json(path.parent / SUMMARY_NAME, report)
        reports.append(report)

    sample_set_audit: Dict[str, Any] = {}
    for variant in sorted({str(report["variant"]) for report in reports}):
        variant_reports = [
            report for report in reports if str(report["variant"]) == variant
        ]
        variant_audit: Dict[str, Any] = {}
        for scene in CANONICAL_AXIS_BY_SCENE:
            digests: Dict[str, str] = {}
            for report in variant_reports:
                scene_report = report["aggregate_across_seeds"]["scenes"].get(scene)
                if scene_report is None:
                    continue
                per_seed_digests = set(
                    scene_report["sample_id_sha256_by_seed"].values()
                )
                if len(per_seed_digests) == 1:
                    digests[str(report["checkpoint"])] = next(iter(per_seed_digests))
            if digests:
                variant_audit[scene] = {
                    "sample_id_sha256_by_checkpoint": digests,
                    "identical_across_checkpoints": bool(
                        len(digests) == len(variant_reports)
                        and len(set(digests.values())) == 1
                    ),
                }
        sample_set_audit[variant] = variant_audit

    comparison = {
        "artifact": "styleplanner_v6_multiseed_checkpoint_comparison",
        "output_root": str(root),
        "checkpoint_filter": list(checkpoint_tags),
        "variant_filter": list(variants),
        "rho_zero_ade_tolerance": float(rho_zero_ade_tolerance),
        "reports": reports,
        "sample_set_audit": sample_set_audit,
        "selection_note": (
            "Use worst-seed/min statistics to reject seed-fragile checkpoints. "
            "Do not treat more seeds as a substitute for more independent scenes."
        ),
    }
    _write_json(root / COMPARISON_NAME, comparison)
    _write_long_table(root / LONG_TABLE_NAME, reports)
    return comparison


def main() -> None:
    args = _parser().parse_args()
    comparison = summarize_evaluation_root(
        args.output_root,
        checkpoint_tags=_parse_csv_values(args.checkpoint_tags),
        variants=_parse_csv_values(args.variants),
        rho_zero_ade_tolerance=float(args.rho_zero_ade_tolerance),
    )
    print(
        "[V6MultiSeed] reports="
        f"{len(comparison['reports'])} "
        f"comparison={Path(args.output_root) / COMPARISON_NAME}"
    )


if __name__ == "__main__":
    main()

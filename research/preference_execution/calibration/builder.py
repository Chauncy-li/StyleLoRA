"""Streaming builder for offline calibration exports."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from research.preference_execution.projection.schema import PROJECTION_LEVEL_ORDER
from research.preference_execution.projection.stats import VALID_STYLE_LABELS, _bucket_levels
from research.style_scene_split.index_utils import normalize_split_index_record
from research.style_scene_split.schema_v2 import style_axis_names_for_scene

from .schema import (
    CALIBRATION_SCHEMA_VERSION,
    calibration_index_path,
    calibration_summary_path,
)

VALID_RECORD_SCOPES = ("split_valid", "memory_eligible")
STYLE_SWEEP_ORDER = ("conservative", "normal", "aggressive")


def _write_json(path: str, payload: object) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    os.replace(tmp_path, path_obj)


def _iter_jsonl(path: str) -> Iterator[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_filtered_split_index(index_path: str, record_scope: str) -> Iterator[Dict[str, object]]:
    with open(index_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            record = normalize_split_index_record(json.loads(line))
            if record_scope == "split_valid" and not bool(record.get("split_valid", False)):
                continue
            if record_scope == "memory_eligible" and not bool(record.get("memory_eligible", False)):
                continue
            yield record


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _vector_mean_abs(values: np.ndarray) -> float:
    if values.size <= 0:
        return 0.0
    return float(np.mean(np.abs(values)))


def _direction_alignment_rate(reference_delta: np.ndarray, candidate_delta: np.ndarray, eps: float = 1e-6) -> float:
    active = np.abs(reference_delta) > eps
    if not np.any(active):
        return 1.0
    aligned = reference_delta[active] * candidate_delta[active] >= -eps
    return float(np.mean(aligned.astype(np.float32)))


def _bool_mask(values: Sequence[bool]) -> List[int]:
    return [int(bool(value)) for value in values]


def _select_bucket_stats_from_record(
    stats: Mapping[str, object],
    split_record: Mapping[str, object],
    *,
    selected_level: str,
    selected_bucket_key: str,
    min_bucket_size: int,
) -> Mapping[str, object] | None:
    levels = stats.get("levels", {})
    level_stats = levels.get(selected_level, {})
    bucket_stats = level_stats.get(selected_bucket_key, None)
    if bucket_stats is not None:
        return bucket_stats

    bucket_levels = _bucket_levels(split_record)
    for level_name in PROJECTION_LEVEL_ORDER:
        bucket_key = bucket_levels[level_name]
        level_stats = levels.get(level_name, {})
        bucket_stats = level_stats.get(bucket_key, None)
        if bucket_stats is None:
            continue
        if int(bucket_stats.get("count", 0)) < min_bucket_size and level_name != "scene":
            continue
        return bucket_stats

    scene_bucket = str(split_record.get("scene_bucket", "none"))
    return levels.get("scene", {}).get(scene_bucket, None)


def _resolve_style_target_vector(
    stats: Mapping[str, object],
    scene_bucket: str,
    style_label: str,
) -> np.ndarray | None:
    scene_stats = stats.get("style_prototypes", {}).get(scene_bucket, {})
    style_stats = scene_stats.get("styles", {}).get(style_label, None)
    if style_stats is not None:
        return np.asarray(style_stats.get("mean", []), dtype=np.float32)
    scene_mean = scene_stats.get("scene_mean", None)
    if scene_mean is None:
        return None
    return np.asarray(scene_mean, dtype=np.float32)


def _axis_reference_orders(stats: Mapping[str, object], scene_bucket: str) -> Dict[str, List[str]]:
    scene_stats = stats.get("style_prototypes", {}).get(scene_bucket, {})
    style_groups = scene_stats.get("styles", {})
    axis_names = list(style_axis_names_for_scene(scene_bucket))
    orders: Dict[str, List[str]] = {}
    for axis_index, axis_name in enumerate(axis_names):
        values = []
        for style_label in STYLE_SWEEP_ORDER:
            style_stats = style_groups.get(style_label, None)
            if style_stats is None:
                continue
            mean_vec = style_stats.get("mean", [])
            if len(mean_vec) != 3:
                continue
            values.append((style_label, float(mean_vec[axis_index])))
        if len(values) < 2:
            orders[axis_name] = list(STYLE_SWEEP_ORDER)
        else:
            values.sort(key=lambda item: item[1])
            orders[axis_name] = [style_label for style_label, _ in values]
    return orders


class PreferenceCalibrationDatasetBuilder:
    """Build offline calibration targets from split-index and conditioning exports."""

    def __init__(
        self,
        split_index_path: str,
        conditioning_index_path: str,
        stats_path: str,
        output_dir: str,
        *,
        record_scope: str = "split_valid",
        min_bucket_size: int = 128,
        log_interval: int = 5000,
    ) -> None:
        if record_scope not in VALID_RECORD_SCOPES:
            raise ValueError(f"record_scope must be one of {VALID_RECORD_SCOPES}, got {record_scope!r}")
        if min_bucket_size <= 0:
            raise ValueError(f"min_bucket_size must be > 0, got {min_bucket_size}")
        if log_interval <= 0:
            raise ValueError(f"log_interval must be > 0, got {log_interval}")

        self.split_index_path = split_index_path
        self.conditioning_index_path = conditioning_index_path
        self.stats_path = stats_path
        self.output_dir = output_dir
        self.record_scope = record_scope
        self.min_bucket_size = int(min_bucket_size)
        self.log_interval = int(log_interval)
        self.output_index_path = calibration_index_path(output_dir)
        self.output_summary_path = calibration_summary_path(output_dir)

        os.makedirs(self.output_dir, exist_ok=True)

    def build(self) -> Dict[str, object]:
        if not os.path.exists(self.split_index_path):
            raise FileNotFoundError(f"split_index_path not found: {self.split_index_path}")
        if not os.path.exists(self.conditioning_index_path):
            raise FileNotFoundError(f"conditioning_index_path not found: {self.conditioning_index_path}")
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(f"stats_path not found: {self.stats_path}")

        with open(self.stats_path, "r", encoding="utf-8") as file_obj:
            stats = json.load(file_obj)

        start_time = time.time()
        total_seen = 0
        total_written = 0
        scene_counter: Counter[str] = Counter()
        observed_style_counter: Counter[str] = Counter()
        target_style_counter: Counter[str] = Counter()
        direction_safe_sum_by_scene: Dict[str, float] = defaultdict(float)
        direction_effective_sum_by_scene: Dict[str, float] = defaultdict(float)
        monotonic_safe_sum_by_scene: Dict[str, float] = defaultdict(float)
        monotonic_effective_sum_by_scene: Dict[str, float] = defaultdict(float)
        observed_target_l1_sum_by_scene: Dict[str, float] = defaultdict(float)
        observed_safe_l1_sum_by_scene: Dict[str, float] = defaultdict(float)
        observed_effective_l1_sum_by_scene: Dict[str, float] = defaultdict(float)
        frontier_feature_sums_by_scene: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(6, dtype=np.float64))
        prototype_fallback_counter: Counter[str] = Counter()

        split_iter = _iter_filtered_split_index(self.split_index_path, self.record_scope)
        conditioning_iter = _iter_jsonl(self.conditioning_index_path)
        tmp_index_path = f"{self.output_index_path}.tmp"

        with open(tmp_index_path, "w", encoding="utf-8") as output_file:
            with tqdm(desc="Build calibration", unit="record") as progress:
                for split_record, conditioning_record in zip(split_iter, conditioning_iter):
                    total_seen += 1
                    calibration_record, fallback_keys = self._build_calibration_record(
                        split_record=split_record,
                        conditioning_record=conditioning_record,
                        stats=stats,
                    )
                    for fallback_key in fallback_keys:
                        prototype_fallback_counter[fallback_key] += 1
                    output_file.write(json.dumps(calibration_record, ensure_ascii=False) + "\n")
                    total_written += 1

                    scene_bucket = str(calibration_record["scene_bucket"])
                    scene_counter[scene_bucket] += 1
                    observed_style_counter[str(calibration_record["style_label"])] += 1
                    target_style_counter[str(calibration_record["target_style_label"])] += 1
                    current = calibration_record["current"]
                    sweep = calibration_record["sweep"]
                    frontier = calibration_record["frontier_features"]

                    direction_safe_sum_by_scene[scene_bucket] += float(current["direction_alignment_rate_safe"])
                    direction_effective_sum_by_scene[scene_bucket] += float(current["direction_alignment_rate_effective"])
                    monotonic_safe_sum_by_scene[scene_bucket] += float(sweep["safe_monotonic_rate"])
                    monotonic_effective_sum_by_scene[scene_bucket] += float(sweep["effective_monotonic_rate"])
                    observed_target_l1_sum_by_scene[scene_bucket] += float(current["observed_target_l1"])
                    observed_safe_l1_sum_by_scene[scene_bucket] += float(current["observed_safe_l1"])
                    observed_effective_l1_sum_by_scene[scene_bucket] += float(current["observed_effective_l1"])
                    frontier_feature_sums_by_scene[scene_bucket] += np.asarray(
                        [
                            float(frontier["quality_score"]),
                            float(frontier["style_performance_confidence"]),
                            float(frontier["clipped_axis_fraction"]),
                            float(frontier["projection_l1"]),
                            float(frontier["mean_local_axis_gate"]),
                            float(frontier["observed_scene_gate_score"]),
                        ],
                        dtype=np.float64,
                    )

                    if total_seen % self.log_interval == 0:
                        elapsed = time.time() - start_time
                        progress.set_postfix(
                            seen=total_seen,
                            written=total_written,
                            speed=f"{total_seen / max(elapsed, 1e-6):.1f}/s",
                        )
                    progress.update(1)

        try:
            next(split_iter)
            raise ValueError("split_index contains extra filtered records after conditioning export ended")
        except StopIteration:
            pass
        try:
            next(conditioning_iter)
            raise ValueError("conditioning export contains extra records after split_index filtered records ended")
        except StopIteration:
            pass

        os.replace(tmp_index_path, self.output_index_path)
        elapsed_seconds = time.time() - start_time
        summary = {
            "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
            "split_index_path": self.split_index_path,
            "conditioning_index_path": self.conditioning_index_path,
            "stats_path": self.stats_path,
            "output_dir": self.output_dir,
            "output_index_path": self.output_index_path,
            "record_scope": self.record_scope,
            "min_bucket_size": self.min_bucket_size,
            "total_seen": total_seen,
            "total_written": total_written,
            "elapsed_seconds": elapsed_seconds,
            "records_per_second": total_seen / max(elapsed_seconds, 1e-6),
            "scene_distribution": dict(scene_counter),
            "observed_style_distribution": dict(observed_style_counter),
            "target_style_distribution": dict(target_style_counter),
            "prototype_fallback_distribution": dict(prototype_fallback_counter),
            "mean_direction_alignment_rate_safe_by_scene": {
                scene_bucket: _safe_divide(direction_safe_sum_by_scene[scene_bucket], count)
                for scene_bucket, count in sorted(scene_counter.items())
            },
            "mean_direction_alignment_rate_effective_by_scene": {
                scene_bucket: _safe_divide(direction_effective_sum_by_scene[scene_bucket], count)
                for scene_bucket, count in sorted(scene_counter.items())
            },
            "mean_sweep_monotonic_rate_safe_by_scene": {
                scene_bucket: _safe_divide(monotonic_safe_sum_by_scene[scene_bucket], count)
                for scene_bucket, count in sorted(scene_counter.items())
            },
            "mean_sweep_monotonic_rate_effective_by_scene": {
                scene_bucket: _safe_divide(monotonic_effective_sum_by_scene[scene_bucket], count)
                for scene_bucket, count in sorted(scene_counter.items())
            },
            "mean_observed_target_l1_by_scene": {
                scene_bucket: _safe_divide(observed_target_l1_sum_by_scene[scene_bucket], count)
                for scene_bucket, count in sorted(scene_counter.items())
            },
            "mean_observed_safe_l1_by_scene": {
                scene_bucket: _safe_divide(observed_safe_l1_sum_by_scene[scene_bucket], count)
                for scene_bucket, count in sorted(scene_counter.items())
            },
            "mean_observed_effective_l1_by_scene": {
                scene_bucket: _safe_divide(observed_effective_l1_sum_by_scene[scene_bucket], count)
                for scene_bucket, count in sorted(scene_counter.items())
            },
            "mean_frontier_features_by_scene": {
                scene_bucket: {
                    "quality_score": _safe_divide(frontier_feature_sums_by_scene[scene_bucket][0], count),
                    "style_performance_confidence": _safe_divide(frontier_feature_sums_by_scene[scene_bucket][1], count),
                    "clipped_axis_fraction": _safe_divide(frontier_feature_sums_by_scene[scene_bucket][2], count),
                    "projection_l1": _safe_divide(frontier_feature_sums_by_scene[scene_bucket][3], count),
                    "mean_local_axis_gate": _safe_divide(frontier_feature_sums_by_scene[scene_bucket][4], count),
                    "observed_scene_gate_score": _safe_divide(frontier_feature_sums_by_scene[scene_bucket][5], count),
                }
                for scene_bucket, count in sorted(scene_counter.items())
            },
        }
        _write_json(self.output_summary_path, summary)
        return summary

    def _build_calibration_record(
        self,
        *,
        split_record: Mapping[str, object],
        conditioning_record: Mapping[str, object],
        stats: Mapping[str, object],
    ) -> tuple[Dict[str, object], List[str]]:
        sample_id = str(split_record.get("sample_id", ""))
        conditioning_sample_id = str(conditioning_record.get("sample_id", ""))
        if sample_id != conditioning_sample_id:
            raise ValueError(f"sample_id mismatch: {sample_id!r} != {conditioning_sample_id!r}")

        filename = str(split_record.get("filename", ""))
        conditioning_filename = str(conditioning_record.get("filename", ""))
        if filename != conditioning_filename:
            raise ValueError(f"filename mismatch: {filename!r} != {conditioning_filename!r}")

        scene_bucket = str(split_record.get("scene_bucket", "none"))
        conditioning_scene_bucket = str(conditioning_record.get("scene_bucket", "none"))
        if scene_bucket != conditioning_scene_bucket:
            raise ValueError(f"scene_bucket mismatch: {scene_bucket!r} != {conditioning_scene_bucket!r}")

        scene_axis_names = list(conditioning_record.get("scene_axis_names", []))
        observed_scene_vec = np.asarray(split_record.get("style_performance_vec", []), dtype=np.float32)
        target_scene_vec = np.asarray(conditioning_record.get("target_preference_scene_vec", []), dtype=np.float32)
        safe_scene_vec = np.asarray(conditioning_record.get("safe_preference_scene_vec", []), dtype=np.float32)
        effective_scene_vec = np.asarray(conditioning_record.get("effective_preference_scene_vec", []), dtype=np.float32)
        local_axis_gate = np.asarray(conditioning_record.get("local_axis_gate_values", []), dtype=np.float32)
        if not (
            len(scene_axis_names) == 3
            and observed_scene_vec.shape[0] == target_scene_vec.shape[0] == safe_scene_vec.shape[0] == effective_scene_vec.shape[0] == local_axis_gate.shape[0] == 3
        ):
            raise ValueError("Expected scene-local vectors to all have length 3")

        target_delta = target_scene_vec - observed_scene_vec
        safe_delta = safe_scene_vec - observed_scene_vec
        effective_delta = effective_scene_vec - observed_scene_vec

        direction_alignment_safe = _direction_alignment_rate(target_delta, safe_delta)
        direction_alignment_effective = _direction_alignment_rate(target_delta, effective_delta)

        axis_reference_orders = _axis_reference_orders(stats, scene_bucket)
        bucket_stats = _select_bucket_stats_from_record(
            stats,
            split_record,
            selected_level=str(conditioning_record.get("selected_bucket_level", "")),
            selected_bucket_key=str(conditioning_record.get("selected_bucket_key", "")),
            min_bucket_size=self.min_bucket_size,
        )
        if bucket_stats is None:
            raise ValueError(
                f"Unable to resolve bucket stats for sample_id={sample_id!r}, scene_bucket={scene_bucket!r}"
            )
        lower = np.asarray(bucket_stats.get("lower", []), dtype=np.float32)
        upper = np.asarray(bucket_stats.get("upper", []), dtype=np.float32)
        if lower.shape[0] != 3 or upper.shape[0] != 3:
            raise ValueError("Expected bucket lower/upper bounds to have length 3")

        sweep_target_scene_vecs: Dict[str, List[float]] = {}
        sweep_safe_scene_vecs: Dict[str, List[float]] = {}
        sweep_effective_scene_vecs: Dict[str, List[float]] = {}
        prototype_sources: Dict[str, str] = {}
        fallback_keys: List[str] = []
        for style_label in STYLE_SWEEP_ORDER:
            target_vec = _resolve_style_target_vector(stats, scene_bucket, style_label)
            if target_vec is None:
                raise ValueError(
                    f"Unable to resolve target prototype for scene_bucket={scene_bucket!r}, style_label={style_label!r}"
                )
            scene_stats = stats.get("style_prototypes", {}).get(scene_bucket, {})
            style_stats = scene_stats.get("styles", {}).get(style_label, None)
            prototype_sources[style_label] = "style_mean" if style_stats is not None else "scene_mean"
            if style_stats is None:
                fallback_keys.append(f"{scene_bucket}__{style_label}")
            safe_vec = np.clip(target_vec, lower, upper)
            effective_vec = safe_vec * local_axis_gate
            sweep_target_scene_vecs[style_label] = [float(value) for value in target_vec.tolist()]
            sweep_safe_scene_vecs[style_label] = [float(value) for value in safe_vec.tolist()]
            sweep_effective_scene_vecs[style_label] = [float(value) for value in effective_vec.tolist()]

        safe_monotonic_mask: List[bool] = []
        effective_monotonic_mask: List[bool] = []
        for axis_index, axis_name in enumerate(scene_axis_names):
            reference_order = axis_reference_orders.get(axis_name, list(STYLE_SWEEP_ORDER))
            safe_values = [float(sweep_safe_scene_vecs[label][axis_index]) for label in reference_order]
            effective_values = [float(sweep_effective_scene_vecs[label][axis_index]) for label in reference_order]
            safe_monotonic_mask.append(
                all(
                    safe_values[idx] <= safe_values[idx + 1] + 1e-6
                    for idx in range(len(safe_values) - 1)
                )
            )
            effective_monotonic_mask.append(
                all(
                    effective_values[idx] <= effective_values[idx + 1] + 1e-6
                    for idx in range(len(effective_values) - 1)
                )
            )

        mean_local_axis_gate = float(np.mean(local_axis_gate))
        calibration_record = {
            "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
            "sample_id": sample_id,
            "filename": filename,
            "scene_bucket": scene_bucket,
            "style_label": str(split_record.get("style_label", "unknown")),
            "target_style_label": str(conditioning_record.get("target_style_label", "unknown")),
            "subset_id": str(split_record.get("subset_id", "invalid")),
            "scene_axis_names": scene_axis_names,
            "quality_score": float(split_record.get("quality_score", 0.0)),
            "style_performance_confidence": float(split_record.get("style_performance_confidence", 0.0)),
            "sample_quality_valid": bool(split_record.get("sample_quality_valid", False)),
            "memory_eligible": bool(split_record.get("memory_eligible", False)),
            "current": {
                "observed_scene_vec": [float(value) for value in observed_scene_vec.tolist()],
                "target_scene_vec": [float(value) for value in target_scene_vec.tolist()],
                "safe_scene_vec": [float(value) for value in safe_scene_vec.tolist()],
                "effective_scene_vec": [float(value) for value in effective_scene_vec.tolist()],
                "target_delta_from_observed": [float(value) for value in target_delta.tolist()],
                "safe_delta_from_observed": [float(value) for value in safe_delta.tolist()],
                "effective_delta_from_observed": [float(value) for value in effective_delta.tolist()],
                "direction_alignment_mask_safe": _bool_mask(
                    [
                        target_delta[idx] * safe_delta[idx] >= -1e-6
                        or abs(float(target_delta[idx])) <= 1e-6
                        for idx in range(3)
                    ]
                ),
                "direction_alignment_mask_effective": _bool_mask(
                    [
                        target_delta[idx] * effective_delta[idx] >= -1e-6
                        or abs(float(target_delta[idx])) <= 1e-6
                        for idx in range(3)
                    ]
                ),
                "direction_alignment_rate_safe": direction_alignment_safe,
                "direction_alignment_rate_effective": direction_alignment_effective,
                "observed_target_l1": _vector_mean_abs(target_delta),
                "observed_safe_l1": _vector_mean_abs(safe_delta),
                "observed_effective_l1": _vector_mean_abs(effective_delta),
            },
            "sweep": {
                "style_labels": list(STYLE_SWEEP_ORDER),
                "prototype_source_by_style": prototype_sources,
                "reference_order_by_axis": axis_reference_orders,
                "target_scene_vecs": sweep_target_scene_vecs,
                "safe_scene_vecs": sweep_safe_scene_vecs,
                "effective_scene_vecs": sweep_effective_scene_vecs,
                "safe_monotonic_mask": _bool_mask(safe_monotonic_mask),
                "effective_monotonic_mask": _bool_mask(effective_monotonic_mask),
                "safe_monotonic_rate": float(np.mean(np.asarray(safe_monotonic_mask, dtype=np.float32))),
                "effective_monotonic_rate": float(np.mean(np.asarray(effective_monotonic_mask, dtype=np.float32))),
            },
            "frontier_features": {
                "quality_score": float(split_record.get("quality_score", 0.0)),
                "style_performance_confidence": float(split_record.get("style_performance_confidence", 0.0)),
                "clipped_axis_fraction": float(conditioning_record.get("clipped_axis_fraction", 0.0)),
                "projection_l1": float(conditioning_record.get("projection_l1", 0.0)),
                "mean_local_axis_gate": mean_local_axis_gate,
                "observed_scene_gate_score": float(conditioning_record.get("observed_scene_gate_score", 0.0)),
            },
        }
        return calibration_record, fallback_keys

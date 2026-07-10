"""Streaming builder for preference-conditioning exports."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from research.preference_execution.interaction_state.schema import AXIS_GATE_ORDER
from research.style_scene_split.schema_v2 import style_axis_names_for_scene

from .schema import (
    CONDITIONING_SCHEMA_VERSION,
    conditioning_index_path,
    conditioning_summary_path,
)


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


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _scene_axis_indices(axis_names: Sequence[str]) -> List[int]:
    return [AXIS_GATE_ORDER.index(axis_name) for axis_name in axis_names]


def _make_global_vec(axis_names: Sequence[str], local_values: Sequence[float]) -> np.ndarray:
    global_vec = np.zeros(len(AXIS_GATE_ORDER), dtype=np.float32)
    for axis_name, value in zip(axis_names, local_values):
        global_vec[AXIS_GATE_ORDER.index(axis_name)] = float(value)
    return global_vec


def _mean_dict_from_sum(
    counter: Mapping[str, int],
    sums: Mapping[str, np.ndarray],
    names_by_key: Mapping[str, Sequence[str]],
) -> Dict[str, Dict[str, float]]:
    payload: Dict[str, Dict[str, float]] = {}
    for key, count in sorted(counter.items()):
        names = list(names_by_key.get(key, ("axis_0", "axis_1", "axis_2")))
        if count <= 0:
            payload[key] = {name: 0.0 for name in names}
            continue
        values = sums[key] / float(count)
        payload[key] = {
            name: float(values[index])
            for index, name in enumerate(names)
        }
    return payload


class PreferenceConditioningDatasetBuilder:
    """Merge interaction-state gates and projected preferences into one export."""

    def __init__(
        self,
        interaction_index_path: str,
        projection_index_path: str,
        output_dir: str,
        log_interval: int = 5000,
    ) -> None:
        if log_interval <= 0:
            raise ValueError(f"log_interval must be > 0, got {log_interval}")

        self.interaction_index_path = interaction_index_path
        self.projection_index_path = projection_index_path
        self.output_dir = output_dir
        self.log_interval = int(log_interval)
        self.output_index_path = conditioning_index_path(output_dir)
        self.output_summary_path = conditioning_summary_path(output_dir)

        os.makedirs(self.output_dir, exist_ok=True)

    def build(self) -> Dict[str, object]:
        if not os.path.exists(self.interaction_index_path):
            raise FileNotFoundError(f"interaction_index_path not found: {self.interaction_index_path}")
        if not os.path.exists(self.projection_index_path):
            raise FileNotFoundError(f"projection_index_path not found: {self.projection_index_path}")

        start_time = time.time()
        total_seen = 0
        total_written = 0
        scene_counter: Counter[str] = Counter()
        target_style_counter: Counter[str] = Counter()
        dominant_scene_gate_counter: Counter[str] = Counter()
        selected_bucket_level_counter: Counter[str] = Counter()
        local_gate_sum_by_scene: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.float64))
        local_effective_ratio_sum_by_scene: Dict[str, float] = defaultdict(float)
        target_safe_l1_sum_by_scene: Dict[str, float] = defaultdict(float)
        safe_effective_l1_sum_by_scene: Dict[str, float] = defaultdict(float)

        tmp_index_path = f"{self.output_index_path}.tmp"
        interaction_iter = _iter_jsonl(self.interaction_index_path)
        projection_iter = _iter_jsonl(self.projection_index_path)

        with open(tmp_index_path, "w", encoding="utf-8") as output_file:
            with tqdm(desc="Build conditioning", unit="record") as progress:
                for interaction_record, projection_record in zip(interaction_iter, projection_iter):
                    total_seen += 1
                    conditioning_record = self._build_conditioning_record(
                        interaction_record=interaction_record,
                        projection_record=projection_record,
                    )
                    output_file.write(json.dumps(conditioning_record, ensure_ascii=False) + "\n")
                    total_written += 1

                    scene_bucket = str(conditioning_record["scene_bucket"])
                    scene_counter[scene_bucket] += 1
                    target_style_counter[str(conditioning_record["target_style_label"])] += 1
                    dominant_scene_gate_counter[str(conditioning_record["dominant_scene_gate"])] += 1
                    selected_bucket_level_counter[str(conditioning_record["selected_bucket_level"])] += 1

                    local_axis_gate = np.asarray(conditioning_record["local_axis_gate_values"], dtype=np.float64)
                    local_safe = np.asarray(conditioning_record["safe_preference_scene_vec"], dtype=np.float64)
                    local_effective = np.asarray(conditioning_record["effective_preference_scene_vec"], dtype=np.float64)
                    local_target = np.asarray(conditioning_record["target_preference_scene_vec"], dtype=np.float64)

                    local_gate_sum_by_scene[scene_bucket] += local_axis_gate
                    local_effective_ratio_sum_by_scene[scene_bucket] += _safe_divide(
                        float(np.sum(np.abs(local_effective))),
                        float(np.sum(np.abs(local_safe))),
                    )
                    target_safe_l1_sum_by_scene[scene_bucket] += float(np.mean(np.abs(local_target - local_safe)))
                    safe_effective_l1_sum_by_scene[scene_bucket] += float(np.mean(np.abs(local_safe - local_effective)))

                    if total_seen % self.log_interval == 0:
                        elapsed = time.time() - start_time
                        progress.set_postfix(
                            seen=total_seen,
                            written=total_written,
                            speed=f"{total_seen / max(elapsed, 1e-6):.1f}/s",
                        )
                    progress.update(1)

        try:
            next(interaction_iter)
            raise ValueError("interaction_state export contains extra records after projection export ended")
        except StopIteration:
            pass
        try:
            next(projection_iter)
            raise ValueError("projection export contains extra records after interaction_state export ended")
        except StopIteration:
            pass

        os.replace(tmp_index_path, self.output_index_path)
        elapsed_seconds = time.time() - start_time
        summary = {
            "conditioning_schema_version": CONDITIONING_SCHEMA_VERSION,
            "interaction_index_path": self.interaction_index_path,
            "projection_index_path": self.projection_index_path,
            "output_dir": self.output_dir,
            "output_index_path": self.output_index_path,
            "total_seen": total_seen,
            "total_written": total_written,
            "elapsed_seconds": elapsed_seconds,
            "records_per_second": total_seen / max(elapsed_seconds, 1e-6),
            "scene_distribution": dict(scene_counter),
            "target_style_distribution": dict(target_style_counter),
            "dominant_scene_gate_distribution": dict(dominant_scene_gate_counter),
            "selected_bucket_level_distribution": dict(selected_bucket_level_counter),
            "mean_local_axis_gate_by_scene": _mean_dict_from_sum(
                counter=scene_counter,
                sums=local_gate_sum_by_scene,
                names_by_key={
                    scene_bucket: style_axis_names_for_scene(scene_bucket)
                    for scene_bucket in scene_counter
                },
            ),
            "mean_effective_ratio_by_scene": {
                scene_bucket: _safe_divide(local_effective_ratio_sum_by_scene[scene_bucket], count)
                for scene_bucket, count in sorted(scene_counter.items())
            },
            "mean_target_safe_l1_by_scene": {
                scene_bucket: _safe_divide(target_safe_l1_sum_by_scene[scene_bucket], count)
                for scene_bucket, count in sorted(scene_counter.items())
            },
            "mean_safe_effective_l1_by_scene": {
                scene_bucket: _safe_divide(safe_effective_l1_sum_by_scene[scene_bucket], count)
                for scene_bucket, count in sorted(scene_counter.items())
            },
        }
        _write_json(self.output_summary_path, summary)
        return summary

    def _build_conditioning_record(
        self,
        *,
        interaction_record: Mapping[str, object],
        projection_record: Mapping[str, object],
    ) -> Dict[str, object]:
        sample_id = str(interaction_record.get("sample_id", ""))
        projection_sample_id = str(projection_record.get("sample_id", ""))
        if sample_id != projection_sample_id:
            raise ValueError(
                f"sample_id mismatch between interaction_state and projection: "
                f"{sample_id!r} != {projection_sample_id!r}"
            )

        filename = str(interaction_record.get("filename", ""))
        projection_filename = str(projection_record.get("filename", ""))
        if filename != projection_filename:
            raise ValueError(
                f"filename mismatch between interaction_state and projection: "
                f"{filename!r} != {projection_filename!r}"
            )

        scene_bucket = str(interaction_record.get("scene_bucket", "none"))
        projection_scene_bucket = str(projection_record.get("scene_bucket", "none"))
        if scene_bucket != projection_scene_bucket:
            raise ValueError(
                f"scene_bucket mismatch between interaction_state and projection: "
                f"{scene_bucket!r} != {projection_scene_bucket!r}"
            )

        scene_axis_names = list(projection_record.get("axis_names", []))
        if len(scene_axis_names) != 3:
            raise ValueError(f"Expected 3 scene axis names, got {scene_axis_names!r}")

        local_axis_indices = _scene_axis_indices(scene_axis_names)
        axis_gate_values = np.asarray(interaction_record.get("axis_gate_values", []), dtype=np.float32)
        if axis_gate_values.shape[0] != len(AXIS_GATE_ORDER):
            raise ValueError(
                f"Expected {len(AXIS_GATE_ORDER)} axis gate values, got {axis_gate_values.shape[0]}"
            )

        scene_gate_values = np.asarray(interaction_record.get("scene_gate_values", []), dtype=np.float32)
        target_scene_vec = np.asarray(projection_record.get("target_preference_vec", []), dtype=np.float32)
        safe_scene_vec = np.asarray(projection_record.get("projected_preference_vec", []), dtype=np.float32)
        lower_scene_vec = np.asarray(projection_record.get("lower_bound_vec", []), dtype=np.float32)
        upper_scene_vec = np.asarray(projection_record.get("upper_bound_vec", []), dtype=np.float32)

        if not (
            target_scene_vec.shape[0] == safe_scene_vec.shape[0] == lower_scene_vec.shape[0] == upper_scene_vec.shape[0] == 3
        ):
            raise ValueError("Expected target/safe/bound vectors to all have length 3")

        local_axis_gate_values = axis_gate_values[local_axis_indices]
        effective_scene_vec = safe_scene_vec * local_axis_gate_values

        target_global_vec = _make_global_vec(scene_axis_names, target_scene_vec.tolist())
        safe_global_vec = _make_global_vec(scene_axis_names, safe_scene_vec.tolist())
        lower_global_vec = _make_global_vec(scene_axis_names, lower_scene_vec.tolist())
        upper_global_vec = _make_global_vec(scene_axis_names, upper_scene_vec.tolist())
        effective_global_vec = safe_global_vec * axis_gate_values

        return {
            "conditioning_schema_version": CONDITIONING_SCHEMA_VERSION,
            "sample_id": sample_id,
            "filename": filename,
            "scene_bucket": scene_bucket,
            "style_label": str(interaction_record.get("style_label", "unknown")),
            "observed_style_label": str(projection_record.get("observed_style_label", "unknown")),
            "target_style_label": str(projection_record.get("target_style_label", "unknown")),
            "observed_intensity_alpha": float(projection_record.get("observed_intensity_alpha", 0.0)),
            "observed_intensity_beta": float(projection_record.get("observed_intensity_beta", 0.0)),
            "target_intensity_alpha": float(projection_record.get("target_intensity_alpha", 0.0)),
            "target_intensity_beta": float(projection_record.get("target_intensity_beta", 0.0)),
            "subset_id": str(interaction_record.get("subset_id", "invalid")),
            "split_valid": bool(interaction_record.get("split_valid", False)),
            "memory_eligible": bool(interaction_record.get("memory_eligible", False)),
            "condition_density_level": str(interaction_record.get("condition_density_level", "unknown")),
            "condition_speed_regime": str(interaction_record.get("condition_speed_regime", "unknown")),
            "condition_curvature_level": str(interaction_record.get("condition_curvature_level", "unknown")),
            "feature_names": list(interaction_record.get("feature_names", [])),
            "feature_values": [float(value) for value in interaction_record.get("feature_values", [])],
            "feature_mask": [int(value) for value in interaction_record.get("feature_mask", [])],
            "scene_gate_names": list(interaction_record.get("scene_gate_names", [])),
            "scene_gate_values": [float(value) for value in scene_gate_values.tolist()],
            "dominant_scene_gate": str(interaction_record.get("dominant_scene_gate", "")),
            "dominant_scene_gate_score": float(interaction_record.get("dominant_scene_gate_score", 0.0)),
            "observed_scene_gate_score": float(interaction_record.get("observed_scene_gate_score", 0.0)),
            "axis_gate_names": list(interaction_record.get("axis_gate_names", AXIS_GATE_ORDER)),
            "axis_gate_values": [float(value) for value in axis_gate_values.tolist()],
            "scene_axis_names": scene_axis_names,
            "scene_axis_indices": [int(index) for index in local_axis_indices],
            "local_axis_gate_values": [float(value) for value in local_axis_gate_values.tolist()],
            "target_preference_scene_vec": [float(value) for value in target_scene_vec.tolist()],
            "safe_preference_scene_vec": [float(value) for value in safe_scene_vec.tolist()],
            "effective_preference_scene_vec": [float(value) for value in effective_scene_vec.tolist()],
            "lower_bound_scene_vec": [float(value) for value in lower_scene_vec.tolist()],
            "upper_bound_scene_vec": [float(value) for value in upper_scene_vec.tolist()],
            "target_preference_global_vec": [float(value) for value in target_global_vec.tolist()],
            "safe_preference_global_vec": [float(value) for value in safe_global_vec.tolist()],
            "effective_preference_global_vec": [float(value) for value in effective_global_vec.tolist()],
            "lower_bound_global_vec": [float(value) for value in lower_global_vec.tolist()],
            "upper_bound_global_vec": [float(value) for value in upper_global_vec.tolist()],
            "projection_delta_vec": [float(value) for value in projection_record.get("projection_delta_vec", [])],
            "clipped_axis_mask": [int(value) for value in projection_record.get("clipped_axis_mask", [])],
            "clipped_axis_count": int(projection_record.get("clipped_axis_count", 0)),
            "clipped_axis_fraction": float(projection_record.get("clipped_axis_fraction", 0.0)),
            "projection_l1": float(projection_record.get("projection_l1", 0.0)),
            "selected_bucket_level": str(projection_record.get("selected_bucket_level", "")),
            "selected_bucket_key": str(projection_record.get("selected_bucket_key", "")),
            "selected_bucket_count": int(projection_record.get("selected_bucket_count", 0)),
        }

"""Streaming builder for offline interaction-state proxy datasets."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping

import numpy as np
from tqdm import tqdm

from research_v1.scene_data.index import normalize_split_index_record

from .features import build_interaction_state_features
from .gating import compute_scene_gates
from .schema import (
    AXIS_GATE_ORDER,
    FEATURE_NAME_ORDER,
    INTERACTION_STATE_SCHEMA_VERSION,
    SCENE_GATE_ORDER,
    interaction_state_index_path,
    interaction_state_summary_path,
)

VALID_RECORD_SCOPES = ("all", "split_valid", "memory_eligible")


def _write_json(path: str, payload: object) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    os.replace(tmp_path, path_obj)


def iter_normalized_split_index(index_path: str) -> Iterator[Dict[str, object]]:
    """Yield normalized split-index records one by one to avoid large memory spikes."""

    with open(index_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            yield normalize_split_index_record(json.loads(line))


def keep_record(record: Mapping[str, object], record_scope: str) -> bool:
    """Filter records by the requested export scope."""

    if record_scope == "all":
        return True
    if record_scope == "split_valid":
        return bool(record.get("split_valid", False))
    if record_scope == "memory_eligible":
        return bool(record.get("memory_eligible", False))
    raise ValueError(f"Unsupported record_scope: {record_scope}")


def build_interaction_state_record(record: Mapping[str, object]) -> Dict[str, object]:
    """Build one interaction-state JSONL record from a normalized split-index record."""

    feature_bundle = build_interaction_state_features(record)
    gate_bundle = compute_scene_gates(feature_bundle)
    observed_scene = str(record.get("scene_bucket", record.get("scene_bucket_name", "none")))
    scene_gate_dict = gate_bundle.scene_gate_dict()

    return {
        "interaction_state_schema_version": INTERACTION_STATE_SCHEMA_VERSION,
        "sample_id": str(record.get("sample_id", "")),
        "filename": str(record.get("filename", "")),
        "scene_bucket": observed_scene,
        "style_label": str(record.get("style_label", record.get("style_label_name", "unknown"))),
        "subset_id": str(record.get("subset_id", "invalid")),
        "split_valid": bool(record.get("split_valid", False)),
        "memory_eligible": bool(record.get("memory_eligible", False)),
        "condition_density_level": str(record.get("condition_density_level", "unknown")),
        "condition_speed_regime": str(record.get("condition_speed_regime", "unknown")),
        "condition_curvature_level": str(record.get("condition_curvature_level", "unknown")),
        "feature_names": list(feature_bundle.feature_names),
        "feature_values": [float(value) for value in feature_bundle.feature_values.tolist()],
        "feature_mask": [int(value) for value in feature_bundle.feature_mask.tolist()],
        "scene_gate_names": list(gate_bundle.scene_gate_names),
        "scene_gate_values": [float(value) for value in gate_bundle.scene_gate_values.tolist()],
        "dominant_scene_gate": gate_bundle.dominant_scene_gate,
        "dominant_scene_gate_score": float(gate_bundle.dominant_scene_gate_score),
        "observed_scene_gate_score": float(scene_gate_dict.get(observed_scene, 0.0)),
        "axis_gate_names": list(gate_bundle.axis_gate_names),
        "axis_gate_values": [float(value) for value in gate_bundle.axis_gate_values.tolist()],
    }


class InteractionStateDatasetBuilder:
    """Stream interaction-state proxy records from a split-index export."""

    def __init__(
        self,
        index_path: str,
        output_dir: str,
        record_scope: str = "split_valid",
        log_interval: int = 5000,
    ) -> None:
        if record_scope not in VALID_RECORD_SCOPES:
            raise ValueError(
                f"record_scope must be one of {VALID_RECORD_SCOPES}, got {record_scope!r}"
            )
        if log_interval <= 0:
            raise ValueError(f"log_interval must be > 0, got {log_interval}")

        self.index_path = index_path
        self.output_dir = output_dir
        self.record_scope = record_scope
        self.log_interval = int(log_interval)
        self.output_index_path = interaction_state_index_path(output_dir)
        self.output_summary_path = interaction_state_summary_path(output_dir)

        os.makedirs(self.output_dir, exist_ok=True)

    def build(self) -> Dict[str, object]:
        """Build the interaction-state export and return a run summary."""

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"index_path not found: {self.index_path}")

        start_time = time.time()
        total_seen = 0
        total_written = 0
        filtered_out = 0
        observed_scene_counter: Counter[str] = Counter()
        written_scene_counter: Counter[str] = Counter()
        dominant_gate_counter: Counter[str] = Counter()
        scene_gate_sum_by_scene: Dict[str, np.ndarray] = {
            scene_bucket: np.zeros(len(SCENE_GATE_ORDER), dtype=np.float64)
            for scene_bucket in SCENE_GATE_ORDER
        }
        scene_gate_count_by_scene: Counter[str] = Counter()
        feature_sum = np.zeros(len(FEATURE_NAME_ORDER), dtype=np.float64)
        feature_sq_sum = np.zeros(len(FEATURE_NAME_ORDER), dtype=np.float64)
        feature_mask_sum = np.zeros(len(FEATURE_NAME_ORDER), dtype=np.float64)

        tmp_index_path = f"{self.output_index_path}.tmp"
        with open(tmp_index_path, "w", encoding="utf-8") as output_file:
            with tqdm(desc="Build interaction_state", unit="record") as progress:
                for record in iter_normalized_split_index(self.index_path):
                    total_seen += 1
                    observed_scene = str(record.get("scene_bucket", "none"))
                    observed_scene_counter[observed_scene] += 1

                    if not keep_record(record, self.record_scope):
                        filtered_out += 1
                        if total_seen % self.log_interval == 0:
                            progress.set_postfix(
                                seen=total_seen,
                                written=total_written,
                                filtered=filtered_out,
                            )
                        progress.update(1)
                        continue

                    state_record = build_interaction_state_record(record)
                    output_file.write(json.dumps(state_record, ensure_ascii=False) + "\n")

                    total_written += 1
                    written_scene_counter[observed_scene] += 1
                    dominant_gate_counter[str(state_record["dominant_scene_gate"])] += 1

                    feature_values = np.asarray(state_record["feature_values"], dtype=np.float64)
                    feature_mask = np.asarray(state_record["feature_mask"], dtype=np.float64)
                    feature_sum += feature_values
                    feature_sq_sum += feature_values * feature_values
                    feature_mask_sum += feature_mask

                    if observed_scene in scene_gate_sum_by_scene:
                        scene_gate_sum_by_scene[observed_scene] += np.asarray(
                            state_record["scene_gate_values"],
                            dtype=np.float64,
                        )
                        scene_gate_count_by_scene[observed_scene] += 1

                    if total_seen % self.log_interval == 0:
                        elapsed = time.time() - start_time
                        progress.set_postfix(
                            seen=total_seen,
                            written=total_written,
                            filtered=filtered_out,
                            speed=f"{total_seen / max(elapsed, 1e-6):.1f}/s",
                        )
                    progress.update(1)

        os.replace(tmp_index_path, self.output_index_path)
        elapsed_seconds = time.time() - start_time
        summary = {
            "interaction_state_schema_version": INTERACTION_STATE_SCHEMA_VERSION,
            "index_path": self.index_path,
            "output_dir": self.output_dir,
            "output_index_path": self.output_index_path,
            "record_scope": self.record_scope,
            "total_seen": total_seen,
            "total_written": total_written,
            "filtered_out": filtered_out,
            "elapsed_seconds": elapsed_seconds,
            "records_per_second": total_seen / max(elapsed_seconds, 1e-6),
            "observed_scene_distribution": dict(observed_scene_counter),
            "written_scene_distribution": dict(written_scene_counter),
            "dominant_scene_gate_distribution": dict(dominant_gate_counter),
            "feature_summary": self._feature_summary(
                total_written=total_written,
                feature_sum=feature_sum,
                feature_sq_sum=feature_sq_sum,
                feature_mask_sum=feature_mask_sum,
            ),
            "mean_scene_gate_by_observed_scene": self._mean_scene_gate_by_observed_scene(
                scene_gate_sum_by_scene=scene_gate_sum_by_scene,
                scene_gate_count_by_scene=scene_gate_count_by_scene,
            ),
        }
        _write_json(self.output_summary_path, summary)
        return summary

    def _feature_summary(
        self,
        *,
        total_written: int,
        feature_sum: np.ndarray,
        feature_sq_sum: np.ndarray,
        feature_mask_sum: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        if total_written <= 0:
            return {
                feature_name: {"mean": 0.0, "std": 0.0, "mask_rate": 0.0}
                for feature_name in FEATURE_NAME_ORDER
            }

        mean = feature_sum / float(total_written)
        variance = np.maximum(feature_sq_sum / float(total_written) - mean * mean, 0.0)
        std = np.sqrt(variance)
        mask_rate = feature_mask_sum / float(total_written)
        return {
            feature_name: {
                "mean": float(mean[index]),
                "std": float(std[index]),
                "mask_rate": float(mask_rate[index]),
            }
            for index, feature_name in enumerate(FEATURE_NAME_ORDER)
        }

    def _mean_scene_gate_by_observed_scene(
        self,
        *,
        scene_gate_sum_by_scene: Dict[str, np.ndarray],
        scene_gate_count_by_scene: Mapping[str, int],
    ) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for scene_bucket in SCENE_GATE_ORDER:
            count = int(scene_gate_count_by_scene.get(scene_bucket, 0))
            if count <= 0:
                summary[scene_bucket] = {
                    gate_name: 0.0 for gate_name in SCENE_GATE_ORDER
                }
                continue
            summary[scene_bucket] = {
                gate_name: float(scene_gate_sum_by_scene[scene_bucket][index] / count)
                for index, gate_name in enumerate(SCENE_GATE_ORDER)
            }
        return summary

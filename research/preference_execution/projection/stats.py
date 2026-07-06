"""Train-split statistics for non-learned preference realizability projection."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from research.style_scene_split.index_utils import normalize_split_index_record
from research.style_scene_split.schema_v2 import style_axis_names_for_scene

from .schema import PROJECTION_LEVEL_ORDER, PROJECTION_SCHEMA_VERSION

VALID_RECORD_SCOPES = ("split_valid", "memory_eligible")
VALID_STYLE_LABELS = ("aggressive", "normal", "conservative")


def _write_json(path: str, payload: object) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    os.replace(tmp_path, path_obj)


def _iter_normalized_records(index_path: str):
    with open(index_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                yield normalize_split_index_record(json.loads(line))


def _keep_record(record: Mapping[str, object], record_scope: str) -> bool:
    if record_scope == "split_valid":
        return bool(record.get("split_valid", False))
    if record_scope == "memory_eligible":
        return bool(record.get("memory_eligible", False))
    raise ValueError(f"Unsupported record_scope: {record_scope}")


def _bucket_levels(record: Mapping[str, object]) -> Dict[str, str]:
    scene_bucket = str(record.get("scene_bucket", "none"))
    density = str(record.get("condition_density_level", "unknown"))
    speed = str(record.get("condition_speed_regime", "unknown"))
    curvature = str(record.get("condition_curvature_level", "unknown"))
    return {
        "scene_density_speed_curvature": (
            f"{scene_bucket}__density={density}__speed={speed}__curvature={curvature}"
        ),
        "scene_density_speed": f"{scene_bucket}__density={density}__speed={speed}",
        "scene_density": f"{scene_bucket}__density={density}",
        "scene": scene_bucket,
    }


def _quantile_summary(
    values: Sequence[Sequence[float]],
    *,
    quantile_low: float,
    quantile_high: float,
) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.float32)
    lower = np.quantile(arr, quantile_low, axis=0).astype(np.float32)
    upper = np.quantile(arr, quantile_high, axis=0).astype(np.float32)
    mean = arr.mean(axis=0).astype(np.float32)
    return {
        "count": int(arr.shape[0]),
        "lower": [float(value) for value in lower.tolist()],
        "upper": [float(value) for value in upper.tolist()],
        "mean": [float(value) for value in mean.tolist()],
    }


class PreferenceProjectionStatsBuilder:
    """Build hierarchical realizability envelopes from a train split."""

    def __init__(
        self,
        index_path: str,
        output_path: str,
        record_scope: str = "split_valid",
        quantile_low: float = 0.05,
        quantile_high: float = 0.95,
    ) -> None:
        if record_scope not in VALID_RECORD_SCOPES:
            raise ValueError(
                f"record_scope must be one of {VALID_RECORD_SCOPES}, got {record_scope!r}"
            )
        if not (0.0 <= quantile_low < quantile_high <= 1.0):
            raise ValueError(
                f"Expected 0 <= quantile_low < quantile_high <= 1, got "
                f"{quantile_low}, {quantile_high}"
            )

        self.index_path = index_path
        self.output_path = output_path
        self.record_scope = record_scope
        self.quantile_low = float(quantile_low)
        self.quantile_high = float(quantile_high)

    def build(self) -> Dict[str, object]:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"index_path not found: {self.index_path}")

        total_seen = 0
        total_used = 0
        scene_counter: Counter[str] = Counter()
        style_counter: Counter[str] = Counter()
        level_samples: Dict[str, MutableMapping[str, List[List[float]]]] = {
            level_name: defaultdict(list) for level_name in PROJECTION_LEVEL_ORDER
        }
        prototypes: Dict[str, MutableMapping[str, List[List[float]]]] = defaultdict(lambda: defaultdict(list))

        for record in _iter_normalized_records(self.index_path):
            total_seen += 1
            if not _keep_record(record, self.record_scope):
                continue

            scene_bucket = str(record.get("scene_bucket", "none"))
            style_label = str(record.get("style_label", "unknown"))
            style_vec = record.get("style_performance_vec", [])
            if scene_bucket == "none" or style_label not in VALID_STYLE_LABELS:
                continue
            if not isinstance(style_vec, list) or len(style_vec) != 3:
                continue

            axis_names = style_axis_names_for_scene(scene_bucket)
            total_used += 1
            scene_counter[scene_bucket] += 1
            style_counter[style_label] += 1

            style_values = [float(value) for value in style_vec]
            prototypes[scene_bucket][style_label].append(style_values)
            for level_name, bucket_key in _bucket_levels(record).items():
                level_samples[level_name][bucket_key].append(style_values)

        stats = {
            "projection_schema_version": PROJECTION_SCHEMA_VERSION,
            "index_path": self.index_path,
            "record_scope": self.record_scope,
            "quantile_low": self.quantile_low,
            "quantile_high": self.quantile_high,
            "total_seen": total_seen,
            "total_used": total_used,
            "scene_distribution": dict(scene_counter),
            "style_distribution": dict(style_counter),
            "style_prototypes": self._build_style_prototypes(prototypes),
            "levels": self._build_level_stats(level_samples),
        }
        _write_json(self.output_path, stats)
        return stats

    def _build_style_prototypes(
        self,
        prototypes: Mapping[str, Mapping[str, List[List[float]]]],
    ) -> Dict[str, object]:
        output: Dict[str, object] = {}
        for scene_bucket, scene_groups in sorted(prototypes.items()):
            axis_names = list(style_axis_names_for_scene(scene_bucket))
            scene_payload: Dict[str, object] = {
                "axis_names": axis_names,
                "styles": {},
            }
            scene_values: List[List[float]] = []
            for style_label, values in sorted(scene_groups.items()):
                if not values:
                    continue
                arr = np.asarray(values, dtype=np.float32)
                scene_values.extend(values)
                scene_payload["styles"][style_label] = {
                    "count": int(arr.shape[0]),
                    "mean": [float(value) for value in arr.mean(axis=0).tolist()],
                    "std": [float(value) for value in arr.std(axis=0).tolist()],
                }
            if scene_values:
                arr = np.asarray(scene_values, dtype=np.float32)
                scene_payload["scene_mean"] = [float(value) for value in arr.mean(axis=0).tolist()]
            output[scene_bucket] = scene_payload
        return output

    def _build_level_stats(
        self,
        level_samples: Mapping[str, Mapping[str, List[List[float]]]],
    ) -> Dict[str, object]:
        level_payload: Dict[str, object] = {}
        for level_name in PROJECTION_LEVEL_ORDER:
            groups = level_samples[level_name]
            payload = {}
            for bucket_key, values in sorted(groups.items()):
                scene_bucket = bucket_key.split("__", 1)[0]
                payload[bucket_key] = {
                    "scene_bucket": scene_bucket,
                    "axis_names": list(style_axis_names_for_scene(scene_bucket)),
                    **_quantile_summary(
                        values,
                        quantile_low=self.quantile_low,
                        quantile_high=self.quantile_high,
                    ),
                }
            level_payload[level_name] = payload
        return level_payload

"""Apply non-learned realizability projection to a target split."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from tqdm import tqdm

from research.style_scene_split.index_utils import normalize_split_index_record
from research.style_scene_split.schema_v2 import style_axis_names_for_scene

from .schema import PROJECTION_LEVEL_ORDER, PROJECTION_SCHEMA_VERSION
from .stats import VALID_RECORD_SCOPES, VALID_STYLE_LABELS, _bucket_levels

TARGET_STYLE_MODES = ("self", "fixed")


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


class PreferenceProjectionApplier:
    """Apply bucket-wise quantile projection to a target split."""

    def __init__(
        self,
        index_path: str,
        stats_path: str,
        output_index_path: str,
        output_summary_path: str,
        record_scope: str = "split_valid",
        target_style_mode: str = "self",
        fixed_style_label: str = "aggressive",
        min_bucket_size: int = 128,
        log_interval: int = 5000,
    ) -> None:
        if record_scope not in VALID_RECORD_SCOPES:
            raise ValueError(
                f"record_scope must be one of {VALID_RECORD_SCOPES}, got {record_scope!r}"
            )
        if target_style_mode not in TARGET_STYLE_MODES:
            raise ValueError(
                f"target_style_mode must be one of {TARGET_STYLE_MODES}, got {target_style_mode!r}"
            )
        if fixed_style_label not in VALID_STYLE_LABELS:
            raise ValueError(
                f"fixed_style_label must be one of {VALID_STYLE_LABELS}, got {fixed_style_label!r}"
            )
        if min_bucket_size <= 0:
            raise ValueError(f"min_bucket_size must be > 0, got {min_bucket_size}")
        if log_interval <= 0:
            raise ValueError(f"log_interval must be > 0, got {log_interval}")

        self.index_path = index_path
        self.stats_path = stats_path
        self.output_index_path = output_index_path
        self.output_summary_path = output_summary_path
        self.record_scope = record_scope
        self.target_style_mode = target_style_mode
        self.fixed_style_label = fixed_style_label
        self.min_bucket_size = int(min_bucket_size)
        self.log_interval = int(log_interval)

    def apply(self) -> Dict[str, object]:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"index_path not found: {self.index_path}")
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(f"stats_path not found: {self.stats_path}")
        Path(self.output_index_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.stats_path, "r", encoding="utf-8") as file_obj:
            stats = json.load(file_obj)

        start_time = time.time()
        total_seen = 0
        total_written = 0
        skipped = 0
        projection_l1_sum = 0.0
        clipped_record_count = 0
        clipped_axis_count = 0
        selected_level_counter: Counter[str] = Counter()
        target_style_counter: Counter[str] = Counter()
        scene_counter: Counter[str] = Counter()
        clipping_by_scene: Dict[str, list[float]] = defaultdict(list)
        clipping_by_target_style: Dict[str, list[float]] = defaultdict(list)

        tmp_index_path = f"{self.output_index_path}.tmp"
        with open(tmp_index_path, "w", encoding="utf-8") as output_file:
            with tqdm(desc="Apply preference projection", unit="record") as progress:
                for record in _iter_normalized_records(self.index_path):
                    total_seen += 1
                    if not _keep_record(record, self.record_scope):
                        skipped += 1
                        progress.update(1)
                        continue

                    projection_record = self._project_record(record, stats)
                    if projection_record is None:
                        skipped += 1
                        progress.update(1)
                        continue

                    output_file.write(json.dumps(projection_record, ensure_ascii=False) + "\n")
                    total_written += 1

                    scene_bucket = str(projection_record["scene_bucket"])
                    target_style = str(projection_record["target_style_label"])
                    clipping_fraction = float(projection_record["clipped_axis_fraction"])
                    projection_l1 = float(projection_record["projection_l1"])

                    scene_counter[scene_bucket] += 1
                    target_style_counter[target_style] += 1
                    selected_level_counter[str(projection_record["selected_bucket_level"])] += 1
                    clipping_by_scene[scene_bucket].append(clipping_fraction)
                    clipping_by_target_style[target_style].append(clipping_fraction)
                    projection_l1_sum += projection_l1
                    clipped_axis_count += int(projection_record["clipped_axis_count"])
                    if clipping_fraction > 0.0:
                        clipped_record_count += 1

                    if total_seen % self.log_interval == 0:
                        elapsed = time.time() - start_time
                        progress.set_postfix(
                            seen=total_seen,
                            written=total_written,
                            skipped=skipped,
                            speed=f"{total_seen / max(elapsed, 1e-6):.1f}/s",
                        )
                    progress.update(1)

        os.replace(tmp_index_path, self.output_index_path)
        elapsed_seconds = time.time() - start_time
        summary = {
            "projection_schema_version": PROJECTION_SCHEMA_VERSION,
            "index_path": self.index_path,
            "stats_path": self.stats_path,
            "output_index_path": self.output_index_path,
            "record_scope": self.record_scope,
            "target_style_mode": self.target_style_mode,
            "fixed_style_label": self.fixed_style_label,
            "min_bucket_size": self.min_bucket_size,
            "total_seen": total_seen,
            "total_written": total_written,
            "skipped": skipped,
            "elapsed_seconds": elapsed_seconds,
            "records_per_second": total_seen / max(elapsed_seconds, 1e-6),
            "scene_distribution": dict(scene_counter),
            "target_style_distribution": dict(target_style_counter),
            "selected_bucket_level_distribution": dict(selected_level_counter),
            "clipped_record_rate": float(clipped_record_count / max(total_written, 1)),
            "mean_clipped_axis_fraction": float(clipped_axis_count / max(total_written * 3, 1)),
            "mean_projection_l1": float(projection_l1_sum / max(total_written, 1)),
            "mean_clipped_axis_fraction_by_scene": {
                scene_bucket: float(np.mean(values)) if values else 0.0
                for scene_bucket, values in sorted(clipping_by_scene.items())
            },
            "mean_clipped_axis_fraction_by_target_style": {
                style_label: float(np.mean(values)) if values else 0.0
                for style_label, values in sorted(clipping_by_target_style.items())
            },
        }
        _write_json(self.output_summary_path, summary)
        return summary

    def _project_record(self, record: Mapping[str, object], stats: Mapping[str, object]) -> Dict[str, object] | None:
        scene_bucket = str(record.get("scene_bucket", "none"))
        if scene_bucket == "none":
            return None

        axis_names = list(style_axis_names_for_scene(scene_bucket))
        target_style_label = self._resolve_target_style_label(record)
        target_vector = self._resolve_target_vector(stats, scene_bucket, target_style_label)
        if target_vector is None:
            return None

        selected_level, selected_bucket_key, selected_bucket_stats = self._select_bucket_stats(stats, record)
        if selected_bucket_stats is None:
            return None

        lower = np.asarray(selected_bucket_stats["lower"], dtype=np.float32)
        upper = np.asarray(selected_bucket_stats["upper"], dtype=np.float32)
        target = np.asarray(target_vector, dtype=np.float32)
        projected = np.clip(target, lower, upper)
        delta = projected - target
        clipped_mask = (np.abs(delta) > 1e-6).astype(np.int64)

        return {
            "projection_schema_version": PROJECTION_SCHEMA_VERSION,
            "sample_id": str(record.get("sample_id", "")),
            "filename": str(record.get("filename", "")),
            "scene_bucket": scene_bucket,
            "style_label": str(record.get("style_label", "unknown")),
            "target_style_label": target_style_label,
            "subset_id": str(record.get("subset_id", "invalid")),
            "axis_names": axis_names,
            "target_preference_vec": [float(value) for value in target.tolist()],
            "lower_bound_vec": [float(value) for value in lower.tolist()],
            "upper_bound_vec": [float(value) for value in upper.tolist()],
            "projected_preference_vec": [float(value) for value in projected.tolist()],
            "projection_delta_vec": [float(value) for value in delta.tolist()],
            "clipped_axis_mask": [int(value) for value in clipped_mask.tolist()],
            "clipped_axis_count": int(np.sum(clipped_mask)),
            "clipped_axis_fraction": float(np.mean(clipped_mask)),
            "projection_l1": float(np.mean(np.abs(delta))),
            "selected_bucket_level": selected_level,
            "selected_bucket_key": selected_bucket_key,
            "selected_bucket_count": int(selected_bucket_stats["count"]),
            "condition_density_level": str(record.get("condition_density_level", "unknown")),
            "condition_speed_regime": str(record.get("condition_speed_regime", "unknown")),
            "condition_curvature_level": str(record.get("condition_curvature_level", "unknown")),
        }

    def _resolve_target_style_label(self, record: Mapping[str, object]) -> str:
        if self.target_style_mode == "fixed":
            return self.fixed_style_label
        observed_style = str(record.get("style_label", "unknown"))
        return observed_style if observed_style in VALID_STYLE_LABELS else self.fixed_style_label

    def _resolve_target_vector(
        self,
        stats: Mapping[str, object],
        scene_bucket: str,
        target_style_label: str,
    ) -> Sequence[float] | None:
        style_prototypes = stats.get("style_prototypes", {})
        scene_stats = style_prototypes.get(scene_bucket, {})
        style_stats = scene_stats.get("styles", {}).get(target_style_label, None)
        if style_stats is not None:
            return style_stats.get("mean", None)
        return scene_stats.get("scene_mean", None)

    def _select_bucket_stats(
        self,
        stats: Mapping[str, object],
        record: Mapping[str, object],
    ) -> tuple[str, str, Mapping[str, object] | None]:
        levels = stats.get("levels", {})
        bucket_levels = _bucket_levels(record)
        for level_name in PROJECTION_LEVEL_ORDER:
            bucket_key = bucket_levels[level_name]
            level_stats = levels.get(level_name, {})
            bucket_stats = level_stats.get(bucket_key, None)
            if bucket_stats is None:
                continue
            if int(bucket_stats.get("count", 0)) < self.min_bucket_size and level_name != "scene":
                continue
            return level_name, bucket_key, bucket_stats
        scene_bucket = str(record.get("scene_bucket", "none"))
        scene_stats = levels.get("scene", {}).get(scene_bucket, None)
        return "scene", scene_bucket, scene_stats

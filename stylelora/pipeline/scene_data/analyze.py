"""Analyze the additional fields produced by the v2 scene split.

The report covers sample quality, memory eligibility, continuous-style
distributions by subset, and style distributions conditioned on scene tags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from tqdm import tqdm

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _repo_root_str = str(_REPO_ROOT)
    if _repo_root_str not in sys.path:
        sys.path.insert(0, _repo_root_str)

from stylelora.pipeline.scene_data.paths import (
    DEFAULT_STYLE_SCENE_SPLIT_V2_DIR,
    split_index_path,
)

DEFAULT_OUTPUT_DIR = DEFAULT_STYLE_SCENE_SPLIT_V2_DIR
DEFAULT_INDEX_PATH = split_index_path(DEFAULT_OUTPUT_DIR)


def _summarize(values: Iterable[float]) -> Dict[str, float]:
    """Summarize a sequence of floating-point values."""

    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _load_index(index_path: str) -> List[Dict[str, object]]:
    """Read a split index stored in JSON Lines format."""

    records: List[Dict[str, object]] = []
    with open(index_path, "r", encoding="utf-8") as file_obj:
        for line in tqdm(file_obj, desc="Load split index v2", unit="line"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _safe_float_list(values: object) -> List[float]:
    """Safely convert a list-valued index field to floats."""

    if not isinstance(values, list):
        return []
    output: List[float] = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            output.append(value)
    return output


def main():
    parser = argparse.ArgumentParser(description="Analyze enhanced style-scene split outputs (v2)")
    parser.add_argument("--index_path", type=str, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if not os.path.exists(args.index_path):
        raise FileNotFoundError(f"index_path not found: {args.index_path}")

    records = _load_index(args.index_path)
    quality_valid_records = [record for record in records if bool(record.get("sample_quality_valid", False))]
    split_valid_records = [record for record in records if bool(record.get("split_valid", False))]
    memory_records = [record for record in records if bool(record.get("memory_eligible", False))]

    scene_counter: Counter[str] = Counter()
    style_counter: Counter[str] = Counter()
    subset_counter: Counter[str] = Counter()
    quality_reason_counter: Counter[str] = Counter()
    density_counter: Counter[str] = Counter()
    speed_regime_counter: Counter[str] = Counter()
    curvature_counter: Counter[str] = Counter()

    quality_scores: List[float] = []
    style_performance_confidences: List[float] = []
    confidence_by_subset: Dict[str, List[float]] = defaultdict(list)
    performance_by_subset: Dict[str, List[List[float]]] = defaultdict(list)
    subset_axis_names: Dict[str, List[str]] = {}
    conditional_style_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for record in tqdm(records, desc="Analyze index records v2", unit="record"):
        scene_bucket = str(record.get("scene_bucket_name", "none"))
        style_label = str(record.get("style_label_name", "unknown"))
        subset_id = str(record.get("subset_id", "invalid"))
        density_level = str(record.get("condition_density_level", "unknown"))
        speed_regime = str(record.get("condition_speed_regime", "unknown"))
        curvature_level = str(record.get("condition_curvature_level", "unknown"))

        scene_counter[scene_bucket] += 1
        style_counter[style_label] += 1
        subset_counter[subset_id] += 1
        density_counter[density_level] += 1
        speed_regime_counter[speed_regime] += 1
        curvature_counter[curvature_level] += 1

        if not bool(record.get("sample_quality_valid", False)):
            quality_reason_counter[str(record.get("quality_reason", ""))] += 1

        quality_scores.append(float(record.get("quality_score", 0.0)))
        style_performance_confidences.append(float(record.get("style_performance_confidence", 0.0)))

        if bool(record.get("memory_eligible", False)):
            confidence_by_subset[subset_id].append(float(record.get("split_confidence", 0.0)))

            style_vec = _safe_float_list(record.get("style_performance_vec", []))
            if len(style_vec) == 3:
                performance_by_subset[subset_id].append(style_vec)
                subset_axis_names[subset_id] = list(record.get("style_axis_names", []))

            condition_key = f"{scene_bucket}__density={density_level}__speed={speed_regime}__curvature={curvature_level}"
            conditional_style_counts[condition_key][style_label] += 1

    # Summarize the mean and standard deviation of each subset's style vector.
    style_performance_by_subset = {}
    for subset_id, vec_list in sorted(performance_by_subset.items()):
        arr = np.asarray(vec_list, dtype=np.float32)
        axis_names = subset_axis_names.get(subset_id, ["axis_0", "axis_1", "axis_2"])
        if arr.ndim != 2 or arr.shape[1] != 3:
            continue
        style_performance_by_subset[subset_id] = {
            "count": int(arr.shape[0]),
            "axis_names": axis_names,
            "mean": {axis_names[idx]: float(arr[:, idx].mean()) for idx in range(3)},
            "std": {axis_names[idx]: float(arr[:, idx].std()) for idx in range(3)},
        }

    summary = {
        "total_records": len(records),
        "quality_valid_records": len(quality_valid_records),
        "split_valid_records": len(split_valid_records),
        "memory_eligible_records": len(memory_records),
        "quality_valid_rate": float(len(quality_valid_records) / max(len(records), 1)),
        "split_valid_rate": float(len(split_valid_records) / max(len(records), 1)),
        "memory_eligible_rate": float(len(memory_records) / max(len(records), 1)),
        "scene_distribution": dict(scene_counter),
        "style_distribution": dict(style_counter),
        "subset_distribution": dict(subset_counter),
        "density_distribution": dict(density_counter),
        "speed_regime_distribution": dict(speed_regime_counter),
        "curvature_distribution": dict(curvature_counter),
        "quality_score_summary": _summarize(quality_scores),
        "style_performance_confidence_summary": _summarize(style_performance_confidences),
        "quality_invalid_reason_distribution": dict(quality_reason_counter),
        "confidence_by_subset": {
            subset_id: _summarize(values) for subset_id, values in sorted(confidence_by_subset.items())
        },
        "style_performance_by_subset": style_performance_by_subset,
        "conditional_style_distribution": {
            key: dict(counter) for key, counter in sorted(conditional_style_counts.items())
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()



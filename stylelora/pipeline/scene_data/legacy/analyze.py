"""Summarize high-precision straight-driving style split outputs."""

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
    _REPO_ROOT = Path(__file__).resolve().parents[4]
    _repo_root_str = str(_REPO_ROOT)
    if _repo_root_str not in sys.path:
        sys.path.insert(0, _repo_root_str)

from stylelora.pipeline.paths import DEFAULT_CACHE_ROOT as RUNTIME_DEFAULT_CACHE_ROOT

DEFAULT_CACHE_ROOT = str(RUNTIME_DEFAULT_CACHE_ROOT)
DEFAULT_OUTPUT_DIR = f"{DEFAULT_CACHE_ROOT}/style_scene_split_straight_v1"
DEFAULT_INDEX_PATH = f"{DEFAULT_OUTPUT_DIR}/split_index.jsonl"


def _summarize(values: Iterable[float]) -> Dict[str, float]:
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
    records: List[Dict[str, object]] = []
    with open(index_path, "r", encoding="utf-8") as file_obj:
        for line in tqdm(file_obj, desc="Load split index", unit="line"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Analyze style-scene split outputs")
    parser.add_argument("--index_path", type=str, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if not os.path.exists(args.index_path):
        raise FileNotFoundError(f"index_path not found: {args.index_path}")

    records = _load_index(args.index_path)
    valid_records = [record for record in records if bool(record.get("split_valid", False))]

    scene_counter: Counter[str] = Counter()
    style_counter: Counter[str] = Counter()
    subset_counter: Counter[str] = Counter()
    valid_subset_counter: Counter[str] = Counter()
    invalid_reason_counter: Counter[str] = Counter()
    confidence_by_subset: Dict[str, List[float]] = defaultdict(list)
    metrics_by_subset: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for record in tqdm(records, desc="Analyze split index", unit="record"):
        scene_counter[str(record["scene_bucket_name"])] += 1
        style_counter[str(record["style_label_name"])] += 1
        subset_counter[str(record["subset_id"])] += 1
        if not bool(record.get("split_valid", False)):
            invalid_reason_counter[str(record.get("scene_reason", ""))] += 1

    for record in tqdm(valid_records, desc="Analyze valid subsets", unit="record"):
        subset_id = str(record["subset_id"])
        valid_subset_counter[subset_id] += 1
        confidence_by_subset[subset_id].append(float(record["split_confidence"]))
        for metric_name in (
            "following_min_thw",
            "following_min_gap",
            "merge_min_gap",
            "ego_lateral_onset_step",
            "ego_speed_ratio_to_limit",
            "ego_mean_speed",
            "ego_accel_peak",
            "ego_jerk_p90",
            "ego_jerk_peak",
            "event_speed_drop_ratio",
        ):
            value = record.get(metric_name, None)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value) or value >= 1e5 or value <= -0.5:
                continue
            metrics_by_subset[subset_id][metric_name].append(value)

    summary = {
        "total_records": len(records),
        "total_valid_records": len(valid_records),
        "abstain_records": len(records) - len(valid_records),
        "abstain_rate": float((len(records) - len(valid_records)) / max(len(records), 1)),
        "scene_distribution": dict(scene_counter),
        "style_distribution": dict(style_counter),
        "subset_distribution": dict(subset_counter),
        "valid_subset_distribution": dict(valid_subset_counter),
        "invalid_reason_distribution": dict(invalid_reason_counter),
        "confidence_by_subset": {
            subset_id: _summarize(values) for subset_id, values in sorted(confidence_by_subset.items())
        },
        "metrics_by_subset": {
            subset_id: {metric_name: _summarize(values) for metric_name, values in sorted(metric_map.items())}
            for subset_id, metric_map in sorted(metrics_by_subset.items())
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


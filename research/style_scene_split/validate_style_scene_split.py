"""Validation checks for the high-precision straight-driving style split outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _repo_root_str = str(_REPO_ROOT)
    if _repo_root_str not in sys.path:
        sys.path.insert(0, _repo_root_str)

from research._runtime import DEFAULT_CACHE_ROOT as RUNTIME_DEFAULT_CACHE_ROOT

DEFAULT_CACHE_ROOT = str(RUNTIME_DEFAULT_CACHE_ROOT)
DEFAULT_OUTPUT_DIR = f"{DEFAULT_CACHE_ROOT}/style_scene_split_straight_v1"
DEFAULT_INDEX_PATH = f"{DEFAULT_OUTPUT_DIR}/split_index.jsonl"
EXPECTED_CONTEXTS = (
    "straight_free_drive",
    "straight_car_follow",
    "straight_lane_change",
)


def _load_index(index_path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(index_path, "r", encoding="utf-8") as file_obj:
        for line in tqdm(file_obj, desc="Load split index", unit="line"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _mean_metric(records: List[Dict[str, object]], key: str) -> float | None:
    values = []
    for record in records:
        value = record.get(key, None)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value) and value < 1e5 and value > -0.5:
            values.append(value)
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _ordering_triplet(values: Dict[str, float | None], ascending: bool) -> Dict[str, object]:
    ordered = ["aggressive", "normal", "conservative"]
    nums = [values.get(name, None) for name in ordered]
    if any(value is None for value in nums):
        return {"pass": False, "reason": "missing_group"}
    if ascending:
        passed = bool(nums[0] <= nums[1] <= nums[2])
        relation = "aggressive <= normal <= conservative"
    else:
        passed = bool(nums[0] >= nums[1] >= nums[2])
        relation = "aggressive >= normal >= conservative"
    return {
        "pass": passed,
        "expected_relation": relation,
        "values": {name: float(values[name]) for name in ordered},
    }


def _normal_ratio_check(style_map: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    total = sum(len(items) for items in style_map.values())
    if total == 0:
        return {"pass": False, "reason": "empty_context"}
    normal = len(style_map.get("normal", []))
    ratio = float(normal / total)
    return {
        "pass": bool(ratio >= 0.10),
        "expected_relation": "normal_ratio >= 0.10",
        "values": {
            "normal_ratio": ratio,
            "total": total,
            "normal_count": normal,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Validate straight-driving style split coverage and ordering")
    parser.add_argument("--index_path", type=str, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if not os.path.exists(args.index_path):
        raise FileNotFoundError(f"index_path not found: {args.index_path}")

    records = _load_index(args.index_path)
    valid_records = [record for record in records if bool(record.get("split_valid", False))]
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in tqdm(valid_records, desc="Validate subsets", unit="record"):
        grouped[str(record["scene_bucket_name"])][str(record["style_label_name"])].append(record)

    valid_contexts = {key for key in grouped.keys() if key != "none"}
    abstain_records = len(records) - len(valid_records)
    abstain_rate = float(abstain_records / max(len(records), 1))

    validation = {
        "total_records": len(records),
        "total_valid_records": len(valid_records),
        "abstain_records": abstain_records,
        "abstain_rate": abstain_rate,
        "coverage": {
            scene: {style: len(items) for style, items in style_map.items()}
            for scene, style_map in sorted(grouped.items())
        },
        "checks": {},
    }

    validation["checks"]["expected_context_only"] = {
        "pass": bool(valid_contexts.issubset(set(EXPECTED_CONTEXTS))),
        "expected_relation": f"contexts in {list(EXPECTED_CONTEXTS)}",
        "values": {"observed_contexts": sorted(valid_contexts)},
    }
    validation["checks"]["high_precision_abstain"] = {
        "pass": bool(0.20 <= abstain_rate <= 0.98),
        "expected_relation": "0.20 <= abstain_rate <= 0.98",
        "values": {"abstain_rate": abstain_rate},
    }

    for context_name in EXPECTED_CONTEXTS:
        if context_name in grouped:
            validation["checks"][f"{context_name}_normal_ratio"] = _normal_ratio_check(grouped[context_name])

    if "straight_car_follow" in grouped:
        thw_values = {
            style: _mean_metric(grouped["straight_car_follow"].get(style, []), "following_min_thw")
            for style in ("aggressive", "normal", "conservative")
        }
        gap_values = {
            style: _mean_metric(grouped["straight_car_follow"].get(style, []), "following_min_gap")
            for style in ("aggressive", "normal", "conservative")
        }
        validation["checks"]["straight_car_follow_thw_order"] = _ordering_triplet(thw_values, ascending=True)
        validation["checks"]["straight_car_follow_gap_order"] = _ordering_triplet(gap_values, ascending=True)

    if "straight_lane_change" in grouped:
        gap_values = {
            style: _mean_metric(grouped["straight_lane_change"].get(style, []), "merge_min_gap")
            for style in ("aggressive", "normal", "conservative")
        }
        onset_values = {
            style: _mean_metric(grouped["straight_lane_change"].get(style, []), "ego_lateral_onset_step")
            for style in ("aggressive", "normal", "conservative")
        }
        validation["checks"]["straight_lane_change_gap_order"] = _ordering_triplet(gap_values, ascending=True)
        validation["checks"]["straight_lane_change_onset_order"] = _ordering_triplet(onset_values, ascending=True)

    if "straight_free_drive" in grouped:
        ratio_values = {
            style: _mean_metric(grouped["straight_free_drive"].get(style, []), "ego_speed_ratio_to_limit")
            for style in ("aggressive", "normal", "conservative")
        }
        jerk_values = {
            style: _mean_metric(grouped["straight_free_drive"].get(style, []), "ego_jerk_p90")
            for style in ("aggressive", "normal", "conservative")
        }
        validation["checks"]["straight_free_drive_speed_ratio_order"] = _ordering_triplet(ratio_values, ascending=False)
        validation["checks"]["straight_free_drive_jerk_p90_order"] = _ordering_triplet(jerk_values, ascending=False)

    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as file_obj:
            json.dump(validation, file_obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

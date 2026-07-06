"""style_scene_split v2 的验证脚本。

和旧版 validate 相比，这里除了检查硬标签排序外，还会检查：
- quality_valid / split_valid / memory_eligible 三层关系是否自洽；
- 连续风格向量是否与硬标签有基本一致的单调关系；
- memory subset 是否仍然只来自预定义的主场景桶。
"""

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

from research.style_scene_split.defaults import (
    DEFAULT_STYLE_SCENE_SPLIT_V2_DIR,
    PRIMARY_SCENE_BUCKETS,
    split_index_path,
)

DEFAULT_OUTPUT_DIR = DEFAULT_STYLE_SCENE_SPLIT_V2_DIR
DEFAULT_INDEX_PATH = split_index_path(DEFAULT_OUTPUT_DIR)
EXPECTED_CONTEXTS = PRIMARY_SCENE_BUCKETS


def _load_index(index_path: str) -> List[Dict[str, object]]:
    """读取 v2 index。"""

    records: List[Dict[str, object]] = []
    with open(index_path, "r", encoding="utf-8") as file_obj:
        for line in tqdm(file_obj, desc="Load split index v2", unit="line"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _ordering_triplet(values: Dict[str, float | None], ascending: bool) -> Dict[str, object]:
    """验证 aggressive / normal / conservative 是否满足预期单调关系。"""

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


def _mean_axis(records: List[Dict[str, object]], axis_index: int) -> float | None:
    """计算某一组样本在指定连续风格轴上的均值。"""

    values = []
    for record in records:
        vec = record.get("style_performance_vec", [])
        if not isinstance(vec, list) or len(vec) <= axis_index:
            continue
        try:
            value = float(vec[axis_index])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def main():
    parser = argparse.ArgumentParser(description="Validate enhanced style-scene split outputs (v2)")
    parser.add_argument("--index_path", type=str, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if not os.path.exists(args.index_path):
        raise FileNotFoundError(f"index_path not found: {args.index_path}")

    records = _load_index(args.index_path)
    quality_valid_records = [record for record in records if bool(record.get("sample_quality_valid", False))]
    split_valid_records = [record for record in records if bool(record.get("split_valid", False))]
    memory_records = [record for record in records if bool(record.get("memory_eligible", False))]

    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in tqdm(memory_records, desc="Validate memory subsets v2", unit="record"):
        grouped[str(record["scene_bucket_name"])][str(record["style_label_name"])].append(record)

    valid_contexts = {key for key in grouped.keys() if key != "none"}
    validation = {
        "total_records": len(records),
        "quality_valid_records": len(quality_valid_records),
        "split_valid_records": len(split_valid_records),
        "memory_eligible_records": len(memory_records),
        "checks": {},
    }

    validation["checks"]["expected_context_only"] = {
        "pass": bool(valid_contexts.issubset(set(EXPECTED_CONTEXTS))),
        "expected_relation": f"contexts in {list(EXPECTED_CONTEXTS)}",
        "values": {"observed_contexts": sorted(valid_contexts)},
    }
    validation["checks"]["memory_not_exceed_split"] = {
        "pass": bool(len(memory_records) <= len(split_valid_records)),
        "expected_relation": "memory_eligible_records <= split_valid_records",
        "values": {
            "memory_eligible_records": len(memory_records),
            "split_valid_records": len(split_valid_records),
        },
    }
    validation["checks"]["memory_not_exceed_quality"] = {
        "pass": bool(len(memory_records) <= len(quality_valid_records)),
        "expected_relation": "memory_eligible_records <= quality_valid_records",
        "values": {
            "memory_eligible_records": len(memory_records),
            "quality_valid_records": len(quality_valid_records),
        },
    }

    # ------- 保留旧版硬标签排序逻辑，但把样本集换成 memory_eligible -------
    if "straight_car_follow" in grouped:
        headway_values = {
            style: _mean_axis(grouped["straight_car_follow"].get(style, []), 0)
            for style in ("aggressive", "normal", "conservative")
        }
        smoothness_values = {
            style: _mean_axis(grouped["straight_car_follow"].get(style, []), 2)
            for style in ("aggressive", "normal", "conservative")
        }
        validation["checks"]["straight_car_follow_headway_margin_order"] = _ordering_triplet(
            headway_values,
            ascending=True,
        )
        validation["checks"]["straight_car_follow_response_smoothness_order"] = _ordering_triplet(
            smoothness_values,
            ascending=True,
        )

    if "straight_lane_change" in grouped:
        gap_values = {
            style: _mean_axis(grouped["straight_lane_change"].get(style, []), 0)
            for style in ("aggressive", "normal", "conservative")
        }
        commit_values = {
            style: _mean_axis(grouped["straight_lane_change"].get(style, []), 1)
            for style in ("aggressive", "normal", "conservative")
        }
        smoothness_values = {
            style: _mean_axis(grouped["straight_lane_change"].get(style, []), 2)
            for style in ("aggressive", "normal", "conservative")
        }
        validation["checks"]["straight_lane_change_gap_acceptance_order"] = _ordering_triplet(
            gap_values,
            ascending=False,
        )
        validation["checks"]["straight_lane_change_lateral_commitment_order"] = _ordering_triplet(
            commit_values,
            ascending=False,
        )
        validation["checks"]["straight_lane_change_execution_smoothness_order"] = _ordering_triplet(
            smoothness_values,
            ascending=True,
        )

    if "straight_free_drive" in grouped:
        speed_pref_values = {
            style: _mean_axis(grouped["straight_free_drive"].get(style, []), 0)
            for style in ("aggressive", "normal", "conservative")
        }
        longitudinal_values = {
            style: _mean_axis(grouped["straight_free_drive"].get(style, []), 1)
            for style in ("aggressive", "normal", "conservative")
        }
        smoothness_values = {
            style: _mean_axis(grouped["straight_free_drive"].get(style, []), 2)
            for style in ("aggressive", "normal", "conservative")
        }
        validation["checks"]["straight_free_drive_speed_preference_order"] = _ordering_triplet(
            speed_pref_values,
            ascending=False,
        )
        validation["checks"]["straight_free_drive_longitudinal_intensity_order"] = _ordering_triplet(
            longitudinal_values,
            ascending=False,
        )
        validation["checks"]["straight_free_drive_smoothness_order"] = _ordering_triplet(
            smoothness_values,
            ascending=True,
        )

    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as file_obj:
            json.dump(validation, file_obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

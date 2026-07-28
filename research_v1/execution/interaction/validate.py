"""Validate exported interaction-state proxy datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _repo_root_str = str(_REPO_ROOT)
    if _repo_root_str not in sys.path:
        sys.path.insert(0, _repo_root_str)

from research_v1.paths import ensure_repo_on_path
from research_v1.execution.interaction.schema import (
    AXIS_GATE_ORDER,
    FEATURE_NAME_ORDER,
    SCENE_GATE_ORDER,
    interaction_state_index_path,
    interaction_state_output_dir,
    interaction_state_validation_path,
)
from research_v1.scene_data.paths import DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR

ensure_repo_on_path()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an interaction-state proxy export")
    parser.add_argument("--split_root", type=str, default=str(DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR))
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--index_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    return parser.parse_args()


def _load_records(index_path: str):
    with open(index_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    args = get_args()
    output_dir = args.output_dir.strip() or interaction_state_output_dir(args.split_root)
    index_path = args.index_path.strip() or interaction_state_index_path(output_dir)
    output_path = args.output_path.strip() or interaction_state_validation_path(output_dir)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index_path not found: {index_path}")

    total_records = 0
    finite_pass = True
    range_pass = True
    feature_shape_pass = True
    gate_shape_pass = True
    gate_simplex_pass = True
    dominant_gate_counter: Counter[str] = Counter()
    observed_scene_counter: Counter[str] = Counter()
    dominant_match_counter: Counter[str] = Counter()
    scene_gate_sum_by_observed_scene: Dict[str, List[float]] = defaultdict(
        lambda: [0.0 for _ in SCENE_GATE_ORDER]
    )

    for record in tqdm(_load_records(index_path), desc="Validate interaction_state", unit="record"):
        total_records += 1
        feature_values = record.get("feature_values", [])
        feature_mask = record.get("feature_mask", [])
        scene_gate_values = record.get("scene_gate_values", [])
        axis_gate_values = record.get("axis_gate_values", [])
        observed_scene = str(record.get("scene_bucket", "none"))
        dominant_scene_gate = str(record.get("dominant_scene_gate", ""))

        if len(feature_values) != len(FEATURE_NAME_ORDER) or len(feature_mask) != len(FEATURE_NAME_ORDER):
            feature_shape_pass = False
        if len(scene_gate_values) != len(SCENE_GATE_ORDER) or len(axis_gate_values) != len(AXIS_GATE_ORDER):
            gate_shape_pass = False

        for value in list(feature_values) + list(scene_gate_values) + list(axis_gate_values):
            try:
                value = float(value)
            except (TypeError, ValueError):
                finite_pass = False
                range_pass = False
                continue
            if not math.isfinite(value):
                finite_pass = False
            if value < -1e-6 or value > 1.0 + 1e-6:
                range_pass = False

        gate_sum = sum(float(value) for value in scene_gate_values)
        if abs(gate_sum - 1.0) > 1e-4:
            gate_simplex_pass = False

        observed_scene_counter[observed_scene] += 1
        dominant_gate_counter[dominant_scene_gate] += 1
        if observed_scene in SCENE_GATE_ORDER:
            for index, value in enumerate(scene_gate_values):
                scene_gate_sum_by_observed_scene[observed_scene][index] += float(value)
            if dominant_scene_gate == observed_scene:
                dominant_match_counter[observed_scene] += 1

    mean_scene_gate_by_observed_scene: Dict[str, Dict[str, float]] = {}
    dominant_match_rate_by_scene: Dict[str, float] = {}
    scene_alignment_pass = True
    for scene_bucket in SCENE_GATE_ORDER:
        count = int(observed_scene_counter.get(scene_bucket, 0))
        if count <= 0:
            mean_scene_gate_by_observed_scene[scene_bucket] = {
                gate_name: 0.0 for gate_name in SCENE_GATE_ORDER
            }
            dominant_match_rate_by_scene[scene_bucket] = 0.0
            scene_alignment_pass = False
            continue

        means = {
            gate_name: float(scene_gate_sum_by_observed_scene[scene_bucket][index] / count)
            for index, gate_name in enumerate(SCENE_GATE_ORDER)
        }
        mean_scene_gate_by_observed_scene[scene_bucket] = means
        dominant_match_rate_by_scene[scene_bucket] = float(
            dominant_match_counter.get(scene_bucket, 0) / count
        )
        aligned_gate = max(means.items(), key=lambda item: item[1])[0]
        if aligned_gate != scene_bucket:
            scene_alignment_pass = False

    validation = {
        "total_records": total_records,
        "checks": {
            "feature_shape": {"pass": feature_shape_pass, "expected_dim": len(FEATURE_NAME_ORDER)},
            "gate_shape": {
                "pass": gate_shape_pass,
                "expected_scene_gate_dim": len(SCENE_GATE_ORDER),
                "expected_axis_gate_dim": len(AXIS_GATE_ORDER),
            },
            "finite_values": {"pass": finite_pass},
            "value_range_0_1": {"pass": range_pass},
            "scene_gate_simplex": {"pass": gate_simplex_pass},
            "scene_alignment_by_mean_gate": {
                "pass": scene_alignment_pass,
                "mean_scene_gate_by_observed_scene": mean_scene_gate_by_observed_scene,
            },
        },
        "observed_scene_distribution": dict(observed_scene_counter),
        "dominant_scene_gate_distribution": dict(dominant_gate_counter),
        "dominant_match_rate_by_scene": dominant_match_rate_by_scene,
    }

    print(json.dumps(validation, ensure_ascii=False, indent=2))
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(validation, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


if __name__ == "__main__":
    main()

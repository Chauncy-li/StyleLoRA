"""Validate exported preference-conditioning datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _repo_root_str = str(_REPO_ROOT)
    if _repo_root_str not in sys.path:
        sys.path.insert(0, _repo_root_str)

from research._runtime import ensure_repo_on_path
from research.preference_execution.conditioning.schema import (
    conditioning_index_path,
    conditioning_output_dir,
    conditioning_validation_path,
)
from research.preference_execution.interaction_state.schema import AXIS_GATE_ORDER, FEATURE_NAME_ORDER, SCENE_GATE_ORDER
from research.style_scene_split.defaults import DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR

ensure_repo_on_path()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a preference-conditioning export")
    parser.add_argument("--split_root", type=str, default=str(DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR))
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--index_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    return parser.parse_args()


def _iter_records(index_path: str):
    with open(index_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    args = get_args()
    output_dir = args.output_dir.strip() or conditioning_output_dir(args.split_root)
    index_path = args.index_path.strip() or conditioning_index_path(output_dir)
    output_path = args.output_path.strip() or conditioning_validation_path(output_dir)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index_path not found: {index_path}")

    total_records = 0
    feature_shape_pass = True
    gate_shape_pass = True
    vector_shape_pass = True
    finite_pass = True
    gate_range_pass = True
    scene_gate_simplex_pass = True
    effective_global_pass = True
    local_global_consistency_pass = True
    non_scene_axes_zero_pass = True
    scene_counter: Counter[str] = Counter()
    target_style_counter: Counter[str] = Counter()

    for record in tqdm(_iter_records(index_path), desc="Validate conditioning", unit="record"):
        total_records += 1
        feature_values = record.get("feature_values", [])
        feature_mask = record.get("feature_mask", [])
        scene_gate_values = record.get("scene_gate_values", [])
        axis_gate_values = record.get("axis_gate_values", [])
        local_axis_gate_values = record.get("local_axis_gate_values", [])
        scene_axis_indices = record.get("scene_axis_indices", [])
        target_scene = record.get("target_preference_scene_vec", [])
        safe_scene = record.get("safe_preference_scene_vec", [])
        effective_scene = record.get("effective_preference_scene_vec", [])
        target_global = record.get("target_preference_global_vec", [])
        safe_global = record.get("safe_preference_global_vec", [])
        effective_global = record.get("effective_preference_global_vec", [])

        if len(feature_values) != len(FEATURE_NAME_ORDER) or len(feature_mask) != len(FEATURE_NAME_ORDER):
            feature_shape_pass = False
        if len(scene_gate_values) != len(SCENE_GATE_ORDER) or len(axis_gate_values) != len(AXIS_GATE_ORDER):
            gate_shape_pass = False
        if not (
            len(local_axis_gate_values) == len(scene_axis_indices) == 3
            and len(target_scene) == len(safe_scene) == len(effective_scene) == 3
            and len(target_global) == len(safe_global) == len(effective_global) == len(AXIS_GATE_ORDER)
        ):
            vector_shape_pass = False

        numeric_lists = [
            feature_values,
            scene_gate_values,
            axis_gate_values,
            local_axis_gate_values,
            target_scene,
            safe_scene,
            effective_scene,
            target_global,
            safe_global,
            effective_global,
        ]
        for values in numeric_lists:
            for value in values:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    finite_pass = False
                    continue
                if not math.isfinite(value):
                    finite_pass = False

        for values in (scene_gate_values, axis_gate_values, local_axis_gate_values):
            for value in values:
                value = float(value)
                if value < -1e-6 or value > 1.0 + 1e-6:
                    gate_range_pass = False

        gate_sum = sum(float(value) for value in scene_gate_values)
        if abs(gate_sum - 1.0) > 1e-4:
            scene_gate_simplex_pass = False

        axis_gate_arr = np.asarray(axis_gate_values, dtype=np.float32)
        safe_global_arr = np.asarray(safe_global, dtype=np.float32)
        effective_global_arr = np.asarray(effective_global, dtype=np.float32)
        if not np.allclose(effective_global_arr, safe_global_arr * axis_gate_arr, atol=1e-6):
            effective_global_pass = False

        local_idx_arr = np.asarray(scene_axis_indices, dtype=np.int64)
        target_scene_arr = np.asarray(target_scene, dtype=np.float32)
        safe_scene_arr = np.asarray(safe_scene, dtype=np.float32)
        effective_scene_arr = np.asarray(effective_scene, dtype=np.float32)
        local_axis_gate_arr = np.asarray(local_axis_gate_values, dtype=np.float32)
        if not (
            np.allclose(np.asarray(target_global, dtype=np.float32)[local_idx_arr], target_scene_arr, atol=1e-6)
            and np.allclose(np.asarray(safe_global, dtype=np.float32)[local_idx_arr], safe_scene_arr, atol=1e-6)
            and np.allclose(np.asarray(effective_global, dtype=np.float32)[local_idx_arr], effective_scene_arr, atol=1e-6)
            and np.allclose(axis_gate_arr[local_idx_arr], local_axis_gate_arr, atol=1e-6)
        ):
            local_global_consistency_pass = False

        mask = np.ones(len(AXIS_GATE_ORDER), dtype=bool)
        for index in local_idx_arr.tolist():
            mask[index] = False
        if not (
            np.allclose(np.asarray(target_global, dtype=np.float32)[mask], 0.0, atol=1e-6)
            and np.allclose(np.asarray(safe_global, dtype=np.float32)[mask], 0.0, atol=1e-6)
            and np.allclose(np.asarray(effective_global, dtype=np.float32)[mask], 0.0, atol=1e-6)
        ):
            non_scene_axes_zero_pass = False

        scene_counter[str(record.get("scene_bucket", "none"))] += 1
        target_style_counter[str(record.get("target_style_label", "unknown"))] += 1

    validation = {
        "total_records": total_records,
        "checks": {
            "feature_shape": {"pass": feature_shape_pass, "expected_dim": len(FEATURE_NAME_ORDER)},
            "gate_shape": {
                "pass": gate_shape_pass,
                "expected_scene_gate_dim": len(SCENE_GATE_ORDER),
                "expected_axis_gate_dim": len(AXIS_GATE_ORDER),
            },
            "vector_shape": {"pass": vector_shape_pass, "expected_scene_axis_dim": 3, "expected_global_axis_dim": len(AXIS_GATE_ORDER)},
            "finite_values": {"pass": finite_pass},
            "gate_range_0_1": {"pass": gate_range_pass},
            "scene_gate_simplex": {"pass": scene_gate_simplex_pass},
            "effective_global_consistency": {"pass": effective_global_pass},
            "local_global_consistency": {"pass": local_global_consistency_pass},
            "non_scene_axes_zero": {"pass": non_scene_axes_zero_pass},
        },
        "scene_distribution": dict(scene_counter),
        "target_style_distribution": dict(target_style_counter),
    }

    print(json.dumps(validation, ensure_ascii=False, indent=2))
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(validation, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


if __name__ == "__main__":
    main()

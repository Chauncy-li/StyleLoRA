"""Validate exported preference projection datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

from tqdm import tqdm

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _repo_root_str = str(_REPO_ROOT)
    if _repo_root_str not in sys.path:
        sys.path.insert(0, _repo_root_str)

from research._runtime import ensure_repo_on_path
from research.preference_execution.projection.schema import (
    PROJECTION_LEVEL_ORDER,
    projection_index_path,
    projection_output_dir,
    projection_validation_path,
)
from research.style_scene_split.defaults import DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR

ensure_repo_on_path()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a preference projection export")
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
    output_dir = args.output_dir.strip() or projection_output_dir(args.split_root)
    index_path = args.index_path.strip() or projection_index_path(output_dir)
    output_path = args.output_path.strip() or projection_validation_path(output_dir)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index_path not found: {index_path}")

    total_records = 0
    finite_pass = True
    bounds_pass = True
    shape_pass = True
    level_pass = True
    clipping_fraction_pass = True
    selected_level_counter: Counter[str] = Counter()
    scene_counter: Counter[str] = Counter()

    for record in tqdm(_iter_records(index_path), desc="Validate projection", unit="record"):
        total_records += 1
        target_vec = record.get("target_preference_vec", [])
        lower_vec = record.get("lower_bound_vec", [])
        upper_vec = record.get("upper_bound_vec", [])
        projected_vec = record.get("projected_preference_vec", [])
        delta_vec = record.get("projection_delta_vec", [])
        clipped_mask = record.get("clipped_axis_mask", [])

        if not (
            len(target_vec) == len(lower_vec) == len(upper_vec) == len(projected_vec) == len(delta_vec) == len(clipped_mask) == 3
        ):
            shape_pass = False

        for values in (target_vec, lower_vec, upper_vec, projected_vec, delta_vec):
            for value in values:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    finite_pass = False
                    continue
                if not math.isfinite(value):
                    finite_pass = False

        for lower, upper, projected in zip(lower_vec, upper_vec, projected_vec):
            lower = float(lower)
            upper = float(upper)
            projected = float(projected)
            if projected < lower - 1e-6 or projected > upper + 1e-6:
                bounds_pass = False

        clipping_fraction = float(record.get("clipped_axis_fraction", 0.0))
        if clipping_fraction < -1e-6 or clipping_fraction > 1.0 + 1e-6:
            clipping_fraction_pass = False

        selected_level = str(record.get("selected_bucket_level", ""))
        if selected_level not in PROJECTION_LEVEL_ORDER:
            level_pass = False

        selected_level_counter[selected_level] += 1
        scene_counter[str(record.get("scene_bucket", "none"))] += 1

    validation = {
        "total_records": total_records,
        "checks": {
            "shape": {"pass": shape_pass, "expected_axis_dim": 3},
            "finite_values": {"pass": finite_pass},
            "projected_inside_bounds": {"pass": bounds_pass},
            "selected_bucket_level_valid": {"pass": level_pass},
            "clipping_fraction_range": {"pass": clipping_fraction_pass},
        },
        "selected_bucket_level_distribution": dict(selected_level_counter),
        "scene_distribution": dict(scene_counter),
    }

    print(json.dumps(validation, ensure_ascii=False, indent=2))
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(validation, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


if __name__ == "__main__":
    main()

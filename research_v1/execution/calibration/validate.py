"""Validate exported calibration datasets."""

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

from research_v1.paths import ensure_repo_on_path
from research_v1.execution.calibration.schema import (
    calibration_index_path,
    calibration_output_dir,
    calibration_validation_path,
)
from research_v1.execution.calibration.builder import STYLE_SWEEP_ORDER
from research_v1.scene_data.paths import DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR

ensure_repo_on_path()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a preference-calibration export")
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


def _all_finite(values):
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
    return True


def main() -> None:
    args = get_args()
    output_dir = args.output_dir.strip() or calibration_output_dir(args.split_root)
    index_path = args.index_path.strip() or calibration_index_path(output_dir)
    output_path = args.output_path.strip() or calibration_validation_path(output_dir)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index_path not found: {index_path}")

    total_records = 0
    current_shape_pass = True
    sweep_shape_pass = True
    finite_pass = True
    rate_range_pass = True
    monotonic_mask_shape_pass = True
    style_label_set_pass = True
    scene_counter: Counter[str] = Counter()
    target_style_counter: Counter[str] = Counter()

    for record in tqdm(_iter_records(index_path), desc="Validate calibration", unit="record"):
        total_records += 1
        scene_counter[str(record.get("scene_bucket", "none"))] += 1
        target_style_counter[str(record.get("target_style_label", "unknown"))] += 1

        scene_axis_names = record.get("scene_axis_names", [])
        current = record.get("current", {})
        sweep = record.get("sweep", {})
        frontier = record.get("frontier_features", {})

        if not (
            len(scene_axis_names) == 3
            and len(current.get("observed_scene_vec", [])) == 3
            and len(current.get("target_scene_vec", [])) == 3
            and len(current.get("safe_scene_vec", [])) == 3
            and len(current.get("effective_scene_vec", [])) == 3
            and len(current.get("target_delta_from_observed", [])) == 3
            and len(current.get("safe_delta_from_observed", [])) == 3
            and len(current.get("effective_delta_from_observed", [])) == 3
        ):
            current_shape_pass = False

        if list(sweep.get("style_labels", [])) != list(STYLE_SWEEP_ORDER):
            style_label_set_pass = False

        for style_label in STYLE_SWEEP_ORDER:
            if not (
                len(sweep.get("target_scene_vecs", {}).get(style_label, [])) == 3
                and len(sweep.get("safe_scene_vecs", {}).get(style_label, [])) == 3
                and len(sweep.get("effective_scene_vecs", {}).get(style_label, [])) == 3
            ):
                sweep_shape_pass = False

        if not (
            len(sweep.get("safe_monotonic_mask", [])) == 3
            and len(sweep.get("effective_monotonic_mask", [])) == 3
        ):
            monotonic_mask_shape_pass = False

        numeric_fields = []
        numeric_fields.extend(current.get("observed_scene_vec", []))
        numeric_fields.extend(current.get("target_scene_vec", []))
        numeric_fields.extend(current.get("safe_scene_vec", []))
        numeric_fields.extend(current.get("effective_scene_vec", []))
        numeric_fields.extend(current.get("target_delta_from_observed", []))
        numeric_fields.extend(current.get("safe_delta_from_observed", []))
        numeric_fields.extend(current.get("effective_delta_from_observed", []))
        numeric_fields.extend(
            [
                current.get("direction_alignment_rate_safe", 0.0),
                current.get("direction_alignment_rate_effective", 0.0),
                current.get("observed_target_l1", 0.0),
                current.get("observed_safe_l1", 0.0),
                current.get("observed_effective_l1", 0.0),
                sweep.get("safe_monotonic_rate", 0.0),
                sweep.get("effective_monotonic_rate", 0.0),
                frontier.get("quality_score", 0.0),
                frontier.get("style_performance_confidence", 0.0),
                frontier.get("clipped_axis_fraction", 0.0),
                frontier.get("projection_l1", 0.0),
                frontier.get("mean_local_axis_gate", 0.0),
                frontier.get("observed_scene_gate_score", 0.0),
            ]
        )
        for style_label in STYLE_SWEEP_ORDER:
            numeric_fields.extend(sweep.get("target_scene_vecs", {}).get(style_label, []))
            numeric_fields.extend(sweep.get("safe_scene_vecs", {}).get(style_label, []))
            numeric_fields.extend(sweep.get("effective_scene_vecs", {}).get(style_label, []))
        if not _all_finite(numeric_fields):
            finite_pass = False

        rates = [
            current.get("direction_alignment_rate_safe", 0.0),
            current.get("direction_alignment_rate_effective", 0.0),
            sweep.get("safe_monotonic_rate", 0.0),
            sweep.get("effective_monotonic_rate", 0.0),
            frontier.get("style_performance_confidence", 0.0),
            frontier.get("clipped_axis_fraction", 0.0),
            frontier.get("mean_local_axis_gate", 0.0),
            frontier.get("observed_scene_gate_score", 0.0),
        ]
        for value in rates:
            value = float(value)
            if value < -1e-6 or value > 1.0 + 1e-6:
                rate_range_pass = False

    validation = {
        "total_records": total_records,
        "checks": {
            "current_shape": {"pass": current_shape_pass, "expected_scene_axis_dim": 3},
            "sweep_shape": {"pass": sweep_shape_pass, "expected_style_count": len(STYLE_SWEEP_ORDER)},
            "monotonic_mask_shape": {"pass": monotonic_mask_shape_pass, "expected_axis_dim": 3},
            "style_label_set": {"pass": style_label_set_pass, "expected_style_labels": list(STYLE_SWEEP_ORDER)},
            "finite_values": {"pass": finite_pass},
            "rate_range_0_1": {"pass": rate_range_pass},
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

"""Build offline calibration exports from conditioning outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _repo_root_str = str(_REPO_ROOT)
    if _repo_root_str not in sys.path:
        sys.path.insert(0, _repo_root_str)

from research._runtime import ensure_repo_on_path
from research.preference_execution.calibration.builder import PreferenceCalibrationDatasetBuilder
from research.preference_execution.calibration.schema import calibration_output_dir
from research.preference_execution.conditioning.schema import conditioning_index_path, conditioning_output_dir
from research.preference_execution.projection.schema import projection_output_dir, projection_stats_path
from research.style_scene_split.defaults import (
    DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR,
    DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR,
    split_index_path,
)

ensure_repo_on_path()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline preference-calibration export")
    parser.add_argument("--split_root", type=str, default=str(DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR))
    parser.add_argument("--train_split_root", type=str, default=str(DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR))
    parser.add_argument("--split_index_path", type=str, default="")
    parser.add_argument("--conditioning_index_path", type=str, default="")
    parser.add_argument("--stats_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--record_scope", type=str, default="split_valid", choices=["split_valid", "memory_eligible"])
    parser.add_argument("--min_bucket_size", type=int, default=128)
    parser.add_argument("--log_interval", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    output_dir = args.output_dir.strip() or calibration_output_dir(args.split_root)
    split_index = args.split_index_path.strip() or split_index_path(args.split_root)
    conditioning_index = args.conditioning_index_path.strip() or conditioning_index_path(
        conditioning_output_dir(args.split_root)
    )
    stats_path = args.stats_path.strip() or projection_stats_path(projection_output_dir(args.train_split_root))

    builder = PreferenceCalibrationDatasetBuilder(
        split_index_path=split_index,
        conditioning_index_path=conditioning_index,
        stats_path=stats_path,
        output_dir=output_dir,
        record_scope=args.record_scope,
        min_bucket_size=args.min_bucket_size,
        log_interval=args.log_interval,
    )
    summary = builder.build()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

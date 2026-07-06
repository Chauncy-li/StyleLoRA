"""Build preference-conditioning exports from interaction-state and projection outputs."""

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
from research.preference_execution.conditioning.builder import PreferenceConditioningDatasetBuilder
from research.preference_execution.conditioning.schema import conditioning_output_dir
from research.preference_execution.interaction_state.schema import (
    interaction_state_index_path,
    interaction_state_output_dir,
)
from research.preference_execution.projection.schema import projection_index_path, projection_output_dir
from research.style_scene_split.defaults import DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR

ensure_repo_on_path()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a preference-conditioning export")
    parser.add_argument("--split_root", type=str, default=str(DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR))
    parser.add_argument("--interaction_index_path", type=str, default="")
    parser.add_argument("--projection_index_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    interaction_dir = interaction_state_output_dir(args.split_root)
    projection_dir = projection_output_dir(args.split_root)
    output_dir = args.output_dir.strip() or conditioning_output_dir(args.split_root)
    interaction_index_path = args.interaction_index_path.strip() or interaction_state_index_path(interaction_dir)
    projection_index_path_value = args.projection_index_path.strip() or projection_index_path(projection_dir)

    builder = PreferenceConditioningDatasetBuilder(
        interaction_index_path=interaction_index_path,
        projection_index_path=projection_index_path_value,
        output_dir=output_dir,
        log_interval=args.log_interval,
    )
    summary = builder.build()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

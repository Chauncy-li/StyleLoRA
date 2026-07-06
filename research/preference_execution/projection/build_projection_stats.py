"""Build non-learned preference projection statistics from the train split."""

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
from research.preference_execution.projection.schema import projection_output_dir, projection_stats_path
from research.preference_execution.projection.stats import PreferenceProjectionStatsBuilder
from research.style_scene_split.defaults import DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR, split_index_path

ensure_repo_on_path()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build non-learned preference projection stats")
    parser.add_argument("--split_root", type=str, default=str(DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR))
    parser.add_argument("--index_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--record_scope", type=str, default="split_valid", choices=["split_valid", "memory_eligible"])
    parser.add_argument("--quantile_low", type=float, default=0.05)
    parser.add_argument("--quantile_high", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    split_root = args.split_root
    output_dir = projection_output_dir(split_root)
    index_path = args.index_path.strip() or split_index_path(split_root)
    output_path = args.output_path.strip() or projection_stats_path(output_dir)

    builder = PreferenceProjectionStatsBuilder(
        index_path=index_path,
        output_path=output_path,
        record_scope=args.record_scope,
        quantile_low=args.quantile_low,
        quantile_high=args.quantile_high,
    )
    summary = builder.build()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Build an interaction-state proxy dataset from a style_scene_split_v2 export."""

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
from research.preference_execution.interaction_state.builder import InteractionStateDatasetBuilder
from research.preference_execution.interaction_state.schema import interaction_state_output_dir
from research.style_scene_split.defaults import DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR, split_index_path

ensure_repo_on_path()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build interaction-state proxy records from style_scene_split_v2")
    parser.add_argument("--split_root", type=str, default=str(DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR))
    parser.add_argument("--index_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument(
        "--record_scope",
        type=str,
        default="split_valid",
        choices=["all", "split_valid", "memory_eligible"],
    )
    parser.add_argument("--log_interval", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    split_root = Path(args.split_root)
    index_path = args.index_path.strip() or split_index_path(str(split_root))
    output_dir = args.output_dir.strip() or interaction_state_output_dir(str(split_root))

    builder = InteractionStateDatasetBuilder(
        index_path=index_path,
        output_dir=output_dir,
        record_scope=args.record_scope,
        log_interval=args.log_interval,
    )
    summary = builder.build()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

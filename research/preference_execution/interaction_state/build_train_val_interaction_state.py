"""Build interaction-state proxy datasets for both train and val straight-scene splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _repo_root_str = str(_REPO_ROOT)
    if _repo_root_str not in sys.path:
        sys.path.insert(0, _repo_root_str)

from research._runtime import ensure_repo_on_path
from research.preference_execution.interaction_state.builder import InteractionStateDatasetBuilder
from research.preference_execution.interaction_state.schema import interaction_state_output_dir
from research.style_scene_split.defaults import (
    DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR,
    DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR,
    split_index_path,
)

ensure_repo_on_path()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val interaction-state proxy datasets")
    parser.add_argument("--split_name", type=str, default="all", choices=["train", "val", "all"])
    parser.add_argument("--train_split_root", type=str, default=str(DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR))
    parser.add_argument("--val_split_root", type=str, default=str(DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR))
    parser.add_argument(
        "--record_scope",
        type=str,
        default="split_valid",
        choices=["all", "split_valid", "memory_eligible"],
    )
    parser.add_argument("--log_interval", type=int, default=5000)
    return parser.parse_args()


def _run_one(split_name: str, split_root: str, record_scope: str, log_interval: int) -> Dict[str, object]:
    builder = InteractionStateDatasetBuilder(
        index_path=split_index_path(split_root),
        output_dir=interaction_state_output_dir(split_root),
        record_scope=record_scope,
        log_interval=log_interval,
    )
    print(
        f"[InteractionState] split={split_name} split_root={split_root} "
        f"record_scope={record_scope}"
    )
    return builder.build()


def main() -> None:
    args = get_args()
    summaries: Dict[str, object] = {"splits": {}}

    if args.split_name in {"train", "all"}:
        summaries["splits"]["train"] = _run_one(
            split_name="train",
            split_root=args.train_split_root,
            record_scope=args.record_scope,
            log_interval=args.log_interval,
        )
    if args.split_name in {"val", "all"}:
        summaries["splits"]["val"] = _run_one(
            split_name="val",
            split_root=args.val_split_root,
            record_scope=args.record_scope,
            log_interval=args.log_interval,
        )

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Shared defaults for the straight-scene split research pipeline."""

from __future__ import annotations

from pathlib import Path

from stylelora.pipeline.paths import (
    DEFAULT_CACHE_ROOT as RUNTIME_DEFAULT_CACHE_ROOT,
    DEFAULT_NUPLAN_DATA_PATH,
    DEFAULT_NUPLAN_LOG_NAMES_PATH,
    DEFAULT_NUPLAN_MAP_PATH,
    DEFAULT_PLANNER_CACHE_DIR as RUNTIME_DEFAULT_PLANNER_CACHE_DIR,
    DEFAULT_PLANNER_CACHE_LIST_PATH,
)

PRIMARY_SCENE_BUCKETS = (
    "straight_free_drive",
    "straight_car_follow",
    "straight_lane_change",
)

DEFAULT_CACHE_ROOT = str(RUNTIME_DEFAULT_CACHE_ROOT)
DEFAULT_PLANNER_CACHE_DIR = str(RUNTIME_DEFAULT_PLANNER_CACHE_DIR)
DEFAULT_DATA_LIST_PATH = str(DEFAULT_PLANNER_CACHE_LIST_PATH)
DEFAULT_RAW_DATA_PATH = DEFAULT_NUPLAN_DATA_PATH
DEFAULT_MAP_PATH = DEFAULT_NUPLAN_MAP_PATH
DEFAULT_LOG_NAMES_PATH = DEFAULT_NUPLAN_LOG_NAMES_PATH

DEFAULT_STYLE_SCENE_SPLIT_V1_DIR = str(Path(DEFAULT_CACHE_ROOT) / "style_scene_split_straight_v1")
DEFAULT_STYLE_SCENE_SPLIT_V2_DIR = str(Path(DEFAULT_CACHE_ROOT) / "style_scene_split_straight_v2")
DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR = str(Path(DEFAULT_CACHE_ROOT) / "style_scene_split_straight_train_v2")
DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR = str(Path(DEFAULT_CACHE_ROOT) / "style_scene_split_straight_val_v2")


def split_index_path(output_dir: str) -> str:
    """Return the canonical split index path for a style-scene output folder."""

    return str(Path(output_dir) / "split_index.jsonl")



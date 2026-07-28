"""Schema and file conventions for preference projection datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

PROJECTION_SCHEMA_VERSION = 1

PROJECTION_LEVEL_ORDER: Tuple[str, ...] = (
    "scene_density_speed_curvature",
    "scene_density_speed",
    "scene_density",
    "scene",
)

DEFAULT_OUTPUT_SUBDIR = "projection"
DEFAULT_STATS_FILENAME = "projection_stats.json"
DEFAULT_INDEX_FILENAME = "projection_index.jsonl"
DEFAULT_SUMMARY_FILENAME = "projection_summary.json"
DEFAULT_VALIDATION_FILENAME = "validate_projection.json"


def projection_output_dir(split_root: str) -> str:
    return str(Path(split_root) / DEFAULT_OUTPUT_SUBDIR)


def projection_stats_path(output_dir: str) -> str:
    return str(Path(output_dir) / DEFAULT_STATS_FILENAME)


def projection_index_path(output_dir: str) -> str:
    return str(Path(output_dir) / DEFAULT_INDEX_FILENAME)


def projection_summary_path(output_dir: str) -> str:
    return str(Path(output_dir) / DEFAULT_SUMMARY_FILENAME)


def projection_validation_path(output_dir: str) -> str:
    return str(Path(output_dir) / DEFAULT_VALIDATION_FILENAME)

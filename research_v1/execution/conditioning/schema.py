"""Schema and file conventions for preference-conditioning exports."""

from __future__ import annotations

from pathlib import Path

CONDITIONING_SCHEMA_VERSION = 1

DEFAULT_OUTPUT_SUBDIR = "conditioning"
DEFAULT_INDEX_FILENAME = "conditioning_index.jsonl"
DEFAULT_SUMMARY_FILENAME = "conditioning_summary.json"
DEFAULT_VALIDATION_FILENAME = "validate_conditioning.json"


def conditioning_output_dir(split_root: str) -> str:
    """Return the canonical conditioning output directory for a split root."""

    return str(Path(split_root) / DEFAULT_OUTPUT_SUBDIR)


def conditioning_index_path(output_dir: str) -> str:
    """Return the canonical conditioning JSONL path."""

    return str(Path(output_dir) / DEFAULT_INDEX_FILENAME)


def conditioning_summary_path(output_dir: str) -> str:
    """Return the canonical conditioning summary JSON path."""

    return str(Path(output_dir) / DEFAULT_SUMMARY_FILENAME)


def conditioning_validation_path(output_dir: str) -> str:
    """Return the canonical conditioning validation JSON path."""

    return str(Path(output_dir) / DEFAULT_VALIDATION_FILENAME)

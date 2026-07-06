"""Schema and file conventions for preference-calibration exports."""

from __future__ import annotations

from pathlib import Path

CALIBRATION_SCHEMA_VERSION = 1

DEFAULT_OUTPUT_SUBDIR = "calibration"
DEFAULT_INDEX_FILENAME = "calibration_index.jsonl"
DEFAULT_SUMMARY_FILENAME = "calibration_summary.json"
DEFAULT_VALIDATION_FILENAME = "validate_calibration.json"


def calibration_output_dir(split_root: str) -> str:
    """Return the canonical calibration output directory for a split root."""

    return str(Path(split_root) / DEFAULT_OUTPUT_SUBDIR)


def calibration_index_path(output_dir: str) -> str:
    """Return the canonical calibration JSONL path."""

    return str(Path(output_dir) / DEFAULT_INDEX_FILENAME)


def calibration_summary_path(output_dir: str) -> str:
    """Return the canonical calibration summary JSON path."""

    return str(Path(output_dir) / DEFAULT_SUMMARY_FILENAME)


def calibration_validation_path(output_dir: str) -> str:
    """Return the canonical calibration validation JSON path."""

    return str(Path(output_dir) / DEFAULT_VALIDATION_FILENAME)

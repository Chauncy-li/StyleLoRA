"""Small, planner-agnostic helpers for exporting optional runtime debug traces."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping


def _json_default(value: object) -> object:
    """Convert common tensor/array-like debug values without research modules."""
    if hasattr(value, "detach"):
        value = value.detach()  # type: ignore[union-attr]
    if hasattr(value, "cpu"):
        value = value.cpu()  # type: ignore[union-attr]
    if hasattr(value, "tolist"):
        return value.tolist()  # type: ignore[union-attr]
    return str(value)


def build_runtime_trace_row(
    *,
    step_index: int,
    iteration_index: int,
    time_us: int,
    debug: Mapping[str, object],
) -> dict[str, object]:
    """Build a generic trace row while preserving the complete debug payload."""
    return {
        "step_index": int(step_index),
        "iteration_index": int(iteration_index),
        "time_us": int(time_us),
        "debug": dict(debug),
    }


def append_runtime_trace_jsonl(path: str | Path, row: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(dict(row), ensure_ascii=False, default=_json_default) + "\n")


def append_runtime_trace_csv(path: str | Path, row: Mapping[str, object]) -> None:
    """Write stable scalar columns plus the complete debug object as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step_index", "iteration_index", "time_us", "debug_json"]
    flat_row = {
        "step_index": int(row.get("step_index", 0)),
        "iteration_index": int(row.get("iteration_index", 0)),
        "time_us": int(row.get("time_us", 0)),
        "debug_json": json.dumps(row.get("debug", {}), ensure_ascii=False, default=_json_default),
    }
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(flat_row)

"""Build a retrieval-ready memory index directly from style_scene_split outputs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List

from tqdm import tqdm

from research_v1.paths import DEFAULT_CACHE_ROOT as RUNTIME_DEFAULT_CACHE_ROOT
from research_v1.scene_data.index import (
    SPLIT_BOOLEAN_FIELDS,
    SPLIT_INTEGER_FIELDS,
    SPLIT_OPTIONAL_FLOAT_FIELDS,
    SPLIT_REQUIRED_FLOAT_FIELDS,
    load_split_index,
    write_jsonl_records,
)


DEFAULT_CACHE_ROOT = str(RUNTIME_DEFAULT_CACHE_ROOT)
DEFAULT_OUTPUT_DIR = f"{DEFAULT_CACHE_ROOT}/style_scene_split_straight_v1"
DEFAULT_SPLIT_INDEX_PATH = f"{DEFAULT_OUTPUT_DIR}/split_index.jsonl"
DEFAULT_MEMORY_INDEX_PATH = f"{DEFAULT_OUTPUT_DIR}/memory_index.jsonl"
MEMORY_INDEX_SCHEMA_VERSION = 1


def _to_memory_record(record: Dict[str, object]) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "memory_index_schema_version": MEMORY_INDEX_SCHEMA_VERSION,
        "sample_id": record["sample_id"],
        "filename": record["filename"],
        "source_mode": record["source_mode"],
        "style_cache_path": record.get("style_cache_path"),
        "cache_path": record.get("cache_path"),
        "planner_cache_path": record.get("planner_cache_path"),
        "sidecar_path": record.get("sidecar_path"),
        "scene_bucket": record["scene_bucket"],
        "style_label": record["style_label"],
        "subset_id": record["subset_id"],
        "topology_bucket": record["topology_bucket"],
        "primary_bucket": record["primary_bucket"],
        "secondary_bucket": record["secondary_bucket"],
        "scene_confidence": record["scene_confidence"],
        "style_confidence": record["style_confidence"],
        "split_confidence": record["split_confidence"],
        "token": record["token"],
        "map_name": record["map_name"],
        "log_name": record["log_name"],
        "scenario_name": record["scenario_name"],
        "scenario_type": record["scenario_type"],
        "scene_reason": record["scene_reason"],
        "style_reason": record["style_reason"],
        "scene_score_vec": record["scene_score_vec"],
        "style_score_vec": record["style_score_vec"],
    }

    for key in SPLIT_REQUIRED_FLOAT_FIELDS:
        payload[key] = record[key]
    for key in SPLIT_OPTIONAL_FLOAT_FIELDS:
        payload[key] = record[key]
    for key in SPLIT_BOOLEAN_FIELDS:
        payload[key] = record[key]
    for key in SPLIT_INTEGER_FIELDS:
        payload[key] = record[key]

    return payload


def build_memory_index(
    split_index_path: str,
    output_path: str,
    require_style_cache: bool = True,
) -> Dict[str, object]:
    if not os.path.exists(split_index_path):
        raise FileNotFoundError(f"split_index_path not found: {split_index_path}")

    records = load_split_index(split_index_path, normalize=True)
    scene_counter: Counter[str] = Counter()
    style_counter: Counter[str] = Counter()
    subset_counter: Counter[str] = Counter()
    memory_records: List[Dict[str, object]] = []
    skipped_missing_cache = 0

    for record in tqdm(records, desc="Build memory index", unit="record"):
        if not bool(record["split_valid"]):
            continue
        if require_style_cache and not record.get("style_cache_path"):
            skipped_missing_cache += 1
            continue
        memory_record = _to_memory_record(record)
        memory_records.append(memory_record)
        scene_counter[str(memory_record["scene_bucket"])] += 1
        style_counter[str(memory_record["style_label"])] += 1
        subset_counter[str(memory_record["subset_id"])] += 1

    write_jsonl_records(output_path, memory_records)

    summary = {
        "memory_index_schema_version": MEMORY_INDEX_SCHEMA_VERSION,
        "split_index_path": split_index_path,
        "output_path": output_path,
        "total_split_records": len(records),
        "memory_records": len(memory_records),
        "skipped_missing_style_cache": skipped_missing_cache,
        "require_style_cache": bool(require_style_cache),
        "scene_distribution": dict(scene_counter),
        "style_distribution": dict(style_counter),
        "subset_distribution": dict(subset_counter),
    }

    summary_path = f"{output_path}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    return summary


def get_args():
    parser = argparse.ArgumentParser(description="Build retrieval-ready memory_index.jsonl from style_scene_split")
    parser.add_argument("--split_index_path", type=str, default=DEFAULT_SPLIT_INDEX_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_MEMORY_INDEX_PATH)
    parser.add_argument("--require_style_cache", type=int, default=1)
    return parser.parse_args()


def main():
    args = get_args()
    summary = build_memory_index(
        split_index_path=args.split_index_path,
        output_path=args.output_path,
        require_style_cache=bool(args.require_style_cache),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

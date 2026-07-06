"""Extract retrieval embeddings directly from style_scene_split memory records."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from research._runtime import DEFAULT_CACHE_ROOT as RUNTIME_DEFAULT_CACHE_ROOT
from research.style_scene_split.embedding_utils import (
    build_explicit_sidecar_embedding,
    build_pooled_scene_embedding,
    robust_standardize,
)
from research.style_scene_split.index_utils import write_jsonl_records


DEFAULT_CACHE_ROOT = str(RUNTIME_DEFAULT_CACHE_ROOT)
DEFAULT_OUTPUT_DIR = f"{DEFAULT_CACHE_ROOT}/style_scene_split_straight_v1"
DEFAULT_MEMORY_INDEX_PATH = f"{DEFAULT_OUTPUT_DIR}/memory_index.jsonl"
SCENE_EMBEDDING_SCHEMA_VERSION = 1


def _open_npz(path: str):
    return np.load(path, allow_pickle=False)


def _load_memory_index(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract_record_embedding(memory_record: Dict[str, object], embedding_mode: str) -> Tuple[np.ndarray, List[str]]:
    if embedding_mode == "explicit_sidecar":
        return build_explicit_sidecar_embedding(memory_record)

    if embedding_mode == "pooled_scene_stats":
        style_cache_path = memory_record.get("style_cache_path", "")
        if not style_cache_path or not os.path.exists(style_cache_path):
            raise FileNotFoundError(f"style_cache_path not found for sample_id={memory_record['sample_id']}: {style_cache_path}")
        cache_data = _open_npz(style_cache_path)
        try:
            return build_pooled_scene_embedding(cache_data)
        finally:
            cache_data.close()

    raise ValueError(f"Unsupported embedding_mode: {embedding_mode}")


def extract_scene_embeddings(
    memory_index_path: str,
    output_path: str,
    embedding_mode: str,
) -> Dict[str, object]:
    if not os.path.exists(memory_index_path):
        raise FileNotFoundError(f"memory_index_path not found: {memory_index_path}")

    memory_records = _load_memory_index(memory_index_path)
    raw_vectors: List[np.ndarray] = []
    kept_records: List[Dict[str, object]] = []
    feature_names: List[str] = []
    skipped: List[Dict[str, str]] = []
    scene_counter: Counter[str] = Counter()

    for memory_record in tqdm(memory_records, desc="Extract scene embeddings", unit="record"):
        try:
            vector, vector_feature_names = _extract_record_embedding(memory_record, embedding_mode)
            if not feature_names:
                feature_names = list(vector_feature_names)
            elif feature_names != list(vector_feature_names):
                raise ValueError(
                    f"Inconsistent feature schema for sample_id={memory_record['sample_id']}: "
                    f"{len(vector_feature_names)} != {len(feature_names)}"
                )
            raw_vectors.append(vector.astype(np.float32))
            kept_records.append(memory_record)
            scene_counter[str(memory_record["scene_bucket"])] += 1
        except Exception as exc:
            if len(skipped) < 100:
                skipped.append({"sample_id": str(memory_record.get("sample_id", "")), "reason": str(exc)})

    if not raw_vectors:
        raise RuntimeError(f"No embeddings extracted from memory_index_path={memory_index_path}")

    raw_matrix = np.stack(raw_vectors, axis=0)
    embedding_matrix, stats = robust_standardize(raw_matrix)

    payload_records: List[Dict[str, object]] = []
    for memory_record, embedding in zip(kept_records, embedding_matrix):
        payload_records.append(
            {
                "scene_embedding_schema_version": SCENE_EMBEDDING_SCHEMA_VERSION,
                "embedding_mode": embedding_mode,
                "sample_id": memory_record["sample_id"],
                "scene_bucket": memory_record["scene_bucket"],
                "style_label": memory_record["style_label"],
                "subset_id": memory_record["subset_id"],
                "feature_dim": int(embedding.shape[0]),
                "embedding": [float(item) for item in embedding.tolist()],
            }
        )

    write_jsonl_records(output_path, payload_records)

    meta = {
        "scene_embedding_schema_version": SCENE_EMBEDDING_SCHEMA_VERSION,
        "memory_index_path": memory_index_path,
        "output_path": output_path,
        "embedding_mode": embedding_mode,
        "records_in_memory_index": len(memory_records),
        "records_with_embedding": len(payload_records),
        "feature_dim": int(embedding_matrix.shape[1]),
        "feature_names": feature_names,
        "normalization_stats": stats,
        "scene_distribution": dict(scene_counter),
        "skipped_preview": skipped,
    }
    meta_path = f"{output_path}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as file_obj:
        json.dump(meta, file_obj, ensure_ascii=False, indent=2)
    return meta


def get_args():
    parser = argparse.ArgumentParser(description="Extract scene embeddings from style_scene_split memory index")
    parser.add_argument("--memory_index_path", type=str, default=DEFAULT_MEMORY_INDEX_PATH)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument(
        "--embedding_mode",
        type=str,
        default="explicit_sidecar",
        choices=["explicit_sidecar", "pooled_scene_stats"],
    )
    return parser.parse_args()


def main():
    args = get_args()
    output_path = args.output_path
    if not output_path:
        output_path = os.path.join(
            os.path.dirname(args.memory_index_path),
            f"scene_embeddings__{args.embedding_mode}.jsonl",
        )

    meta = extract_scene_embeddings(
        memory_index_path=args.memory_index_path,
        output_path=output_path,
        embedding_mode=args.embedding_mode,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

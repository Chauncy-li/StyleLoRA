"""Build offline top-k retrieval cache from style_scene_split memory records."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from research_v1.paths import DEFAULT_CACHE_ROOT as RUNTIME_DEFAULT_CACHE_ROOT
from research_v1.scene_data.index import bucket_metric_keys, write_jsonl_records


DEFAULT_CACHE_ROOT = str(RUNTIME_DEFAULT_CACHE_ROOT)
DEFAULT_OUTPUT_DIR = f"{DEFAULT_CACHE_ROOT}/style_scene_split_straight_v1"
DEFAULT_MEMORY_INDEX_PATH = f"{DEFAULT_OUTPUT_DIR}/memory_index.jsonl"
DEFAULT_SCENE_EMBEDDING_PATH = f"{DEFAULT_OUTPUT_DIR}/scene_embeddings__explicit_sidecar.jsonl"
TOPK_CACHE_SCHEMA_VERSION = 1


def _load_jsonl(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_memory_map(path: str) -> Dict[str, Dict[str, object]]:
    records = _load_jsonl(path)
    return {str(record["sample_id"]): dict(record) for record in records}


def _load_embedding_map(path: str) -> Dict[str, np.ndarray]:
    records = _load_jsonl(path)
    embeddings: Dict[str, np.ndarray] = {}
    for record in records:
        embeddings[str(record["sample_id"])] = np.asarray(record["embedding"], dtype=np.float32)
    return embeddings


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def _robust_scales(records: Iterable[Mapping[str, object]], keys: Sequence[str]) -> Dict[str, float]:
    values_by_key: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        for key in keys:
            value = record.get(key, None)
            if value is None:
                continue
            value = float(value)
            if np.isfinite(value):
                values_by_key[key].append(value)

    scales: Dict[str, float] = {}
    for key in keys:
        values = np.asarray(values_by_key.get(key, []), dtype=np.float32)
        if values.size == 0:
            scales[key] = 1.0
            continue
        q25, q75 = np.percentile(values, [25.0, 75.0])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = float(np.std(values))
        scales[key] = 1.0 if not np.isfinite(scale) or scale <= 1e-6 else scale
    return scales


def _metric_similarity(
    query_record: Mapping[str, object],
    gallery_record: Mapping[str, object],
    scales: Mapping[str, float],
    keys: Sequence[str],
) -> float:
    similarities: List[float] = []
    for key in keys:
        query_value = query_record.get(key, None)
        gallery_value = gallery_record.get(key, None)
        if query_value is None or gallery_value is None:
            continue
        query_value = float(query_value)
        gallery_value = float(gallery_value)
        if not np.isfinite(query_value) or not np.isfinite(gallery_value):
            continue
        scale = max(float(scales.get(key, 1.0)), 1e-6)
        similarities.append(1.0 / (1.0 + abs(query_value - gallery_value) / scale))
    if not similarities:
        return 0.0
    return float(np.mean(np.asarray(similarities, dtype=np.float32)))


def build_topk_cache(
    query_memory_index_path: str,
    gallery_memory_index_path: str,
    query_embedding_path: str,
    gallery_embedding_path: str,
    output_path: str,
    top_k: int = 10,
    style_match_policy: str = "prefer",
    weight_scene: float = 0.60,
    weight_metric: float = 0.30,
    weight_confidence: float = 0.10,
    style_match_bonus: float = 0.05,
) -> Dict[str, object]:
    for path in (
        query_memory_index_path,
        gallery_memory_index_path,
        query_embedding_path,
        gallery_embedding_path,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"path not found: {path}")

    query_memory = _load_memory_map(query_memory_index_path)
    gallery_memory = _load_memory_map(gallery_memory_index_path)
    query_embeddings = _load_embedding_map(query_embedding_path)
    gallery_embeddings = _load_embedding_map(gallery_embedding_path)

    gallery_by_bucket: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for sample_id, record in gallery_memory.items():
        if sample_id not in gallery_embeddings:
            continue
        gallery_by_bucket[str(record["scene_bucket"])].append(record)

    scale_by_bucket = {
        bucket: _robust_scales(records, bucket_metric_keys(bucket))
        for bucket, records in gallery_by_bucket.items()
    }

    topk_records: List[Dict[str, object]] = []
    scene_counter: Counter[str] = Counter()
    missing_embedding = 0
    empty_gallery = 0
    style_match_counter: Counter[str] = Counter()

    query_items = sorted(query_memory.items(), key=lambda item: item[0])
    for query_id, query_record in tqdm(query_items, desc="Build top-k cache", unit="query"):
        query_embedding = query_embeddings.get(query_id, None)
        if query_embedding is None:
            missing_embedding += 1
            continue

        scene_bucket = str(query_record["scene_bucket"])
        gallery_candidates = gallery_by_bucket.get(scene_bucket, [])
        metric_keys = bucket_metric_keys(scene_bucket)
        metric_scales = scale_by_bucket.get(scene_bucket, {})

        ranked_neighbors: List[Dict[str, object]] = []
        for gallery_record in gallery_candidates:
            gallery_id = str(gallery_record["sample_id"])
            if gallery_id == query_id:
                continue

            if style_match_policy == "require" and str(gallery_record["style_label"]) != str(query_record["style_label"]):
                continue

            gallery_embedding = gallery_embeddings.get(gallery_id, None)
            if gallery_embedding is None:
                continue

            scene_similarity = _cosine_similarity(query_embedding, gallery_embedding)
            metric_similarity = _metric_similarity(query_record, gallery_record, metric_scales, metric_keys)
            confidence_term = float(gallery_record.get("split_confidence", 0.0))
            style_match = str(gallery_record["style_label"]) == str(query_record["style_label"])

            final_score = (
                weight_scene * scene_similarity
                + weight_metric * metric_similarity
                + weight_confidence * confidence_term
            )
            if style_match_policy == "prefer" and style_match:
                final_score += float(style_match_bonus)

            ranked_neighbors.append(
                {
                    "sample_id": gallery_id,
                    "subset_id": gallery_record["subset_id"],
                    "scene_bucket": gallery_record["scene_bucket"],
                    "style_label": gallery_record["style_label"],
                    "style_cache_path": gallery_record.get("style_cache_path"),
                    "sidecar_path": gallery_record.get("sidecar_path"),
                    "score": float(final_score),
                    "scene_similarity": float(scene_similarity),
                    "metric_similarity": float(metric_similarity),
                    "split_confidence": float(confidence_term),
                    "style_match": bool(style_match),
                }
            )

        ranked_neighbors.sort(key=lambda item: item["score"], reverse=True)
        ranked_neighbors = ranked_neighbors[: max(int(top_k), 0)]
        if not ranked_neighbors:
            empty_gallery += 1

        style_match_hits = int(sum(1 for item in ranked_neighbors if bool(item["style_match"])))
        style_match_counter[scene_bucket] += style_match_hits
        scene_counter[scene_bucket] += 1
        topk_records.append(
            {
                "topk_cache_schema_version": TOPK_CACHE_SCHEMA_VERSION,
                "sample_id": query_id,
                "scene_bucket": query_record["scene_bucket"],
                "style_label": query_record["style_label"],
                "subset_id": query_record["subset_id"],
                "style_cache_path": query_record.get("style_cache_path"),
                "sidecar_path": query_record.get("sidecar_path"),
                "top_k": int(top_k),
                "style_match_policy": style_match_policy,
                "neighbors": ranked_neighbors,
            }
        )

    write_jsonl_records(output_path, topk_records)

    summary = {
        "topk_cache_schema_version": TOPK_CACHE_SCHEMA_VERSION,
        "query_memory_index_path": query_memory_index_path,
        "gallery_memory_index_path": gallery_memory_index_path,
        "query_embedding_path": query_embedding_path,
        "gallery_embedding_path": gallery_embedding_path,
        "output_path": output_path,
        "top_k": int(top_k),
        "style_match_policy": style_match_policy,
        "query_records": len(query_memory),
        "topk_records": len(topk_records),
        "missing_query_embedding": missing_embedding,
        "empty_gallery_queries": empty_gallery,
        "scene_distribution": dict(scene_counter),
        "topk_style_match_hits": dict(style_match_counter),
        "weight_scene": float(weight_scene),
        "weight_metric": float(weight_metric),
        "weight_confidence": float(weight_confidence),
        "style_match_bonus": float(style_match_bonus),
    }
    summary_path = f"{output_path}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    return summary


def get_args():
    parser = argparse.ArgumentParser(description="Build offline top-k retrieval cache from style_scene_split outputs")
    parser.add_argument("--query_memory_index_path", type=str, default=DEFAULT_MEMORY_INDEX_PATH)
    parser.add_argument("--gallery_memory_index_path", type=str, default="")
    parser.add_argument("--query_embedding_path", type=str, default=DEFAULT_SCENE_EMBEDDING_PATH)
    parser.add_argument("--gallery_embedding_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--style_match_policy", type=str, default="prefer", choices=["ignore", "prefer", "require"])
    parser.add_argument("--weight_scene", type=float, default=0.60)
    parser.add_argument("--weight_metric", type=float, default=0.30)
    parser.add_argument("--weight_confidence", type=float, default=0.10)
    parser.add_argument("--style_match_bonus", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = get_args()
    gallery_memory_index_path = args.gallery_memory_index_path or args.query_memory_index_path
    gallery_embedding_path = args.gallery_embedding_path or args.query_embedding_path
    output_path = args.output_path
    if not output_path:
        output_path = os.path.join(
            os.path.dirname(args.query_memory_index_path),
            f"topk_cache__{args.style_match_policy}__k{args.top_k}.jsonl",
        )

    summary = build_topk_cache(
        query_memory_index_path=args.query_memory_index_path,
        gallery_memory_index_path=gallery_memory_index_path,
        query_embedding_path=args.query_embedding_path,
        gallery_embedding_path=gallery_embedding_path,
        output_path=output_path,
        top_k=args.top_k,
        style_match_policy=args.style_match_policy,
        weight_scene=args.weight_scene,
        weight_metric=args.weight_metric,
        weight_confidence=args.weight_confidence,
        style_match_bonus=args.style_match_bonus,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

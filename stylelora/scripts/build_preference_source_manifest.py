"""把场景划分索引转换成无风格标签的偏好 source manifest。

这是 scene split 与弱排序偏好构建之间的接口层。它只保留场景、缓存位置和
可追溯元数据，不读取或筛选 aggressive/normal/conservative 标签。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from stylelora.pipeline.scene_data.index import normalize_split_index_record

SUPPORTED_SCENES = ("straight_free_drive", "straight_car_follow")


def _iter_jsonl(path: Path) -> Iterable[dict]:
    """逐行读取 split_index.jsonl，避免一次性占用大量内存。"""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc


def _portable_cache_path(raw_path: str, cache_root: Path | None) -> str:
    """尽可能把缓存绝对路径改成相对 cache root 的可迁移路径。"""
    path = Path(raw_path)
    if cache_root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(cache_root.resolve()))
    except ValueError:
        return str(path)


def _convert_record(record: Mapping[str, object], split: str, cache_root: Path | None) -> tuple[dict | None, str | None]:
    """规范化单条 scene split 记录；返回 (输出行, 跳过原因)。"""
    row = normalize_split_index_record(record)
    if not bool(row.get("split_valid", False)):
        return None, "split_invalid"
    scene = str(row.get("scene_bucket", ""))
    if scene not in SUPPORTED_SCENES:
        return None, "unsupported_scene"
    cache_path = row.get("cache_path") or row.get("style_cache_path") or row.get("planner_cache_path")
    if not cache_path:
        return None, "missing_cache_path"
    token = str(row.get("token") or row.get("sample_id") or "")
    output = {
        "cache_path": _portable_cache_path(str(cache_path), cache_root),
        "scene_type": scene,
        "log_name": str(row.get("log_name", "")),
        "token": token,
        "source_index": str(row.get("sample_id") or token),
        "split": split,
    }
    return output, None


def _write_partition(index_path: Path, output_path: Path, split: str, cache_root: Path | None) -> dict:
    """转换一个数据划分并原子写出 JSONL。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    counts: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in _iter_jsonl(index_path):
            converted, reason = _convert_record(record, split, cache_root)
            if converted is None:
                counts[f"skipped:{reason}"] += 1
                continue
            handle.write(json.dumps(converted, ensure_ascii=False, sort_keys=True) + "\n")
            counts[f"scene:{converted['scene_type']}"] += 1
            counts["written"] += 1
    os.replace(temporary, output_path)
    return {"input": str(index_path.resolve()), "output": str(output_path.resolve()), **dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="scene split → 无分类标签的偏好 source manifest")
    parser.add_argument("--train-index", required=True, help="训练集 split_index.jsonl")
    parser.add_argument("--val-index", required=True, help="验证集 split_index.jsonl")
    parser.add_argument("--test-index", default=None, help="可选测试集 split_index.jsonl")
    parser.add_argument("--output-dir", required=True, help="输出 train/val/test.jsonl 的目录")
    parser.add_argument("--cache-root", default=None, help="可选；将其下缓存路径写成相对路径")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_root = Path(args.cache_root) if args.cache_root else None
    inputs = [("train", args.train_index), ("val", args.val_index), ("test", args.test_index)]
    report = {}
    for split, source in inputs:
        if source:
            report[split] = _write_partition(Path(source), output_dir / f"{split}.jsonl", split, cache_root)
    report_path = output_dir / "source_manifest_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


"""Build weak-ordering preference manifests from source caches.

从源缓存构建弱排序偏好 manifest（只读缓存数据，不要求 style 分类标签）：

流程（两遍）：
1. 第一遍：用 CacheOnlyDataset 批量加载缓存（不解析 style，混合/模糊/未分类轨迹
   均可进入），对每条专家轨迹计算原始三轴物理指标（metrics.scene_physics_axes），
   逐样本写入中间 JSONL（含原始值），同时按 (scene, axis) 汇总经验 CDF；
2. 第二遍：流式读取中间 JSONL，用已 finalize 的 CDF 把原始值转为场景内百分位
   （叠加上物理方向先验），聚合 preference_rank（加权中位数）与 rank_confidence
   （置信度），写出弱偏好 manifest（JSONL）+ 审计报告。

不使用 aggr/norm/cons 分类标签；仅使用物理方向先验 + 场景内百分位。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from torch.utils.data import DataLoader

from research_lora.data.dataset import style_collate
from research_lora.evaluation.reports import write_json

from research_lora_2.data.axes import AXES_BY_SCENE, direction_sign
from research_lora_2.data.loader import CacheOnlyDataset
from research_lora_2.data.metrics import scene_physics_axes
from research_lora_2.data.preference import SceneCDFTable, aggregate_weak_rank
from research_lora_2.data.schema import ARTIFACT_VERSION, PreferenceSample
from research_lora_2.paths import DEFAULT_PREFERENCE_MANIFEST, DEFAULT_TRAIN_MANIFEST, ensure_repo_on_path


def _iter_jsonl(path: Path):
    """逐行读取 JSONL 文件。"""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl_append(path: Path, rows, *, first: bool = False) -> None:
    """追加写出 JSONL；first=True 时以写入模式清空重建，否则追加。"""
    mode = "w" if first else "a"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _rows_hash(rows) -> str:
    """对 manifest 原始行计算稳定 SHA-256（不依赖 StyleSample，支持无标签行）。"""
    canonical = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="Build weak-ordering preference manifest from source caches.")
    parser.add_argument("--manifest", default=str(DEFAULT_TRAIN_MANIFEST), help="Source manifest JSONL (train rows).")
    parser.add_argument("--cache-root", required=True, help="Cache root for relative cache_path entries.")
    parser.add_argument("--output", default=str(DEFAULT_PREFERENCE_MANIFEST), help="Output weak-preference manifest JSONL.")
    parser.add_argument("--intermediate", default=None,
                        help="中间原始指标 JSONL 路径；省略时与 --output 同目录生成 *_raw.jsonl。")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--drop-invalid-axes", type=int, default=0, choices=(0, 1),
                        help="1=丢弃三个轴全部无效的样本；0=保留（preference_rank 用 0.5 兜底）。")
    parser.add_argument("--save-cdf", default=None,
                        help="第一遍拟合完成后把训练集 CDF 保存到该 JSON 路径（train 用）。")
    parser.add_argument("--load-cdf", default=None,
                        help="加载预训练 CDF（JSON），不重新拟合，原样用于百分位计算（val/test 用）。")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.workers < 0:
        parser.error("--batch-size 必须为正，--workers 必须非负")

    manifests_root = Path(args.output).parent
    manifests_root.mkdir(parents=True, exist_ok=True)
    intermediate_path = Path(args.intermediate) if args.intermediate else manifests_root / f"{Path(args.output).stem}_raw.jsonl"
    # 中间文件：清除历史残留，确保从空文件开始追加
    intermediate_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path.unlink(missing_ok=True)
    source_manifest = Path(args.manifest).resolve()

    # ---------- 第一遍：计算物理三轴 + 汇总经验 CDF + 写中间文件 ----------
    dataset = CacheOnlyDataset(args.manifest, root=args.cache_root)
    records_hash = _rows_hash(dataset.rows)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=style_collate)

    # 是否在本轮重新拟合 CDF：train 拟合（可 save），val/test 加载训练集 CDF（不拟合）
    fit_cdf = args.load_cdf is None
    if args.load_cdf:
        cdf_table = SceneCDFTable.load(args.load_cdf)
    else:
        cdf_table = SceneCDFTable()
    total, valid_count, invalid_count = 0, 0, 0
    scene_counts = Counter()
    scenes = set(AXES_BY_SCENE)
    raw_rows = []
    skipped_malformed = Counter()
    # 全局样本序号（跨 batch 递增，不使用 batch 内索引）
    global_index = 0

    for batch in loader:
        tensors = batch["tensors"]
        metadata_list = batch["metadata"]
        for batch_pos, metadata in enumerate(metadata_list):
            scene = metadata.get("scene_type", "")
            total += 1
            if scene not in scenes:
                skipped_malformed["unsupported_scene"] += 1
                global_index += 1
                continue
            scene_counts[scene] += 1
            try:
                values, valid = scene_physics_axes(tensors, batch_pos, scene)
            except (KeyError, ValueError, TypeError, RuntimeError) as error:
                skipped_malformed[f"metric_error:{type(error).__name__}"] += 1
                global_index += 1
                continue
            row = {
                "index": global_index,
                "cache_path": metadata.get("cache_path", metadata.get("filename", metadata.get("path", ""))),
                "scene_type": scene,
                "log_name": metadata.get("log_name", ""),
                "token": metadata.get("token", ""),
                "source_index": metadata.get("source_index", ""),
                "axis_names": list(AXES_BY_SCENE[scene]),
                "axis_raw": [float(v) for v in values],
                "axis_valid": [bool(v) for v in valid],
            }
            global_index += 1
            raw_rows.append(row)
            if any(valid):
                valid_count += 1
                for axis, value, is_ok in zip(AXES_BY_SCENE[scene], values, valid):
                    if is_ok and fit_cdf:
                        cdf_table.append(scene, axis, value)
            else:
                invalid_count += 1
            # 每攒够一批立即追加写（首批发 w 清空，后续追加避免覆盖）
            if len(raw_rows) >= args.batch_size * 8:
                _write_jsonl_append(intermediate_path, raw_rows, first=(not intermediate_path.exists() or intermediate_path.stat().st_size == 0))
                raw_rows = []

    if raw_rows:
        _write_jsonl_append(intermediate_path, raw_rows, first=False)
        raw_rows = []

    if fit_cdf:
        cdf_table.finalize_all()
        if args.save_cdf:
            cdf_table.save(args.save_cdf)

    skipped_rows = dict(skipped_malformed)
    total_processed = total - sum(skipped_rows.values()) if skipped_rows else total
    if total_processed == 0:
        raise ValueError("No valid rows were produced; check manifest / cache / scene coverage")
    if valid_count == 0:
        raise ValueError("No valid axis values at all; cannot build empirical CDFs")

    # ---------- 第二遍：流式读中间文件 → 百分位 → 聚合 → 写偏好 manifest ----------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    per_scene_rank_hist = defaultdict(list)
    per_scene_conf_hist = defaultdict(list)
    axis_coverage = Counter()
    dropped_all_invalid = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in _iter_jsonl(intermediate_path):
            scene = row["scene_type"]
            axis_names = row["axis_names"]
            axis_raw = row["axis_raw"]
            axis_valid = row["axis_valid"]
            axis_percentiles = []
            for axis, value, is_ok in zip(axis_names, axis_raw, axis_valid):
                if is_ok:
                    q = cdf_table.percentile(scene, axis, value, sign=direction_sign(scene, axis))
                    axis_percentiles.append(q)
                    axis_coverage[(scene, axis)] += 1
                else:
                    axis_percentiles.append(float("nan"))
            if not any(axis_valid):
                dropped_all_invalid += 1
                if args.drop_invalid_axes:
                    continue
                rank, confidence = 0.5, 0.0
            else:
                rank, confidence = aggregate_weak_rank(axis_percentiles, axis_valid)
            sample = PreferenceSample(
                scene_type=scene,
                axis_names=tuple(axis_names),
                axis_raw=tuple(axis_raw),
                axis_valid=tuple(axis_valid),
                axis_percentiles=tuple(axis_percentiles),
                preference_rank=rank,
                rank_confidence=confidence,
                cache_path=row["cache_path"],
                log_name=row["log_name"],
                token=row["token"],
                source_index=row["source_index"],
            )
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
            per_scene_rank_hist[scene].append(rank)
            per_scene_conf_hist[scene].append(confidence)

    if written == 0:
        raise ValueError("Weak-preference manifest produced zero rows; refusing to write empty artifact")

    # ---------- 审计 ----------
    audit = {
        "artifact_version": ARTIFACT_VERSION,
        "metric_space": "expert_physical_three_axes",
        "source": {
            "manifest": str(source_manifest),
            "manifest_hash": records_hash,
            "cache_root": str(Path(args.cache_root).resolve()),
        },
        "counts": {
            "rows_loaded": total,
            "rows_processed": total_processed,
            "rows_written": written,
            "rows_with_valid_axes": valid_count,
            "rows_no_valid_axis": invalid_count,
            "rows_dropped_all_invalid": dropped_all_invalid,
            "scene_counts": dict(scene_counts),
        },
        "source_manifest_rows": len(dataset.rows),
        "cdf_source": {
            "fit_in_this_run": bool(fit_cdf),
            "save_path": args.save_cdf or "",
            "load_path": args.load_cdf or "",
        },
        "skipped_rows": skipped_rows,
        "axis_coverage": {"|".join(key): value for key, value in sorted(axis_coverage.items())},
        "preference_rank_distribution": {
            scene: {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": sum(values) / len(values) if values else None,
            }
            for scene, values in sorted(per_scene_rank_hist.items())
        },
        "rank_confidence_distribution": {
            scene: {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": sum(values) / len(values) if values else None,
            }
            for scene, values in sorted(per_scene_conf_hist.items())
        },
    }
    audit_path = output_path.parent / "audit_preference.json"
    write_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Wrote weak-preference manifest to {output_path}")


if __name__ == "__main__":
    main()
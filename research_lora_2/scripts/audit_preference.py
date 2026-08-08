"""Audit the weak-ordering preference manifest (statistics + optional external contrast).

弱排序偏好数据质量审计：
- 逐 (scene, axis) 的 valid 覆盖率与百分位分布；
- preference_rank（q）的全局/场景内分布；
- rank_confidence（c）分布与低置信样本占比；
- 无有效轴样本计数；
- 可选外部对照：若提供 research_lora 的 StyleSample manifest（含 aggr/norm/cons
  风格标签），计算 q 与该分类的 Spearman 相关——仅作外部对照实验，不进训练。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research_lora.data.manifest import load_style_index

from research_lora_2.data.schema import ARTIFACT_VERSION, PreferenceSample
from research_lora_2.paths import DEFAULT_PREFERENCE_MANIFEST, ensure_repo_on_path

# 外部对照的风格序号（仅用于 Spearman 相关系数，不影响偏好训练）
_STYLE_ORDINAL = {"conservative": -1, "normal": 0, "aggressive": 1}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(len(values)),
        "min": float(array.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """平均秩：并列值组内所有元素共享组内均值秩。

    用于 Spearman 相关系数的并列处理；1-based 与 0-based 秩对相关系数等价，
    因此这里直接用 0-based 平均秩。
    """
    n = values.shape[0]
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg = (start + end - 1) / 2.0
        ranks[start:end] = avg
        start = end
    result = np.empty_like(ranks)
    result[order] = ranks
    return result


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 8:
        return None
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ra = _average_ranks(aa)
    rb = _average_ranks(bb)
    dev_a, dev_b = ra - ra.mean(), rb - rb.mean()
    denom = np.linalg.norm(dev_a) * np.linalg.norm(dev_b)
    if denom == 0:
        return None
    return float((dev_a * dev_b).sum() / denom)


def main() -> None:
    ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="Audit weak-ordering preference manifest quality.")
    parser.add_argument("--manifest", default=str(DEFAULT_PREFERENCE_MANIFEST), help="Weak-preference manifest JSONL.")
    parser.add_argument("--style-manifest", default=None,
                        help="可选 research_lora StyleSample manifest 用于外部分类对照（不进训练）。")
    parser.add_argument("--output", default=None, help="审计报告输出路径；默认与 manifest 同目录 audit_preference.json。")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        parser.error(f"Manifest does not exist: {manifest_path}")

    samples: list[PreferenceSample] = []
    for row in _iter_jsonl(manifest_path):
        samples.append(PreferenceSample.from_mapping(row))

    # ---------- 基础统计 ----------
    scene_counts = Counter(sample.scene_type for sample in samples)
    axis_coverage: Counter[tuple[str, str]] = Counter()
    axis_percentiles: dict[tuple[str, str], list[float]] = defaultdict(list)
    axis_raw: dict[tuple[str, str], list[float]] = defaultdict(list)
    all_ranks: list[float] = []
    all_confidences: list[float] = []
    per_scene_ranks: dict[str, list[float]] = defaultdict(list)
    per_scene_confidences: dict[str, list[float]] = defaultdict(list)
    no_valid_axis = 0

    for sample in samples:
        for axis, q, v in zip(sample.axis_names, sample.axis_percentiles, sample.axis_valid):
            if v and q is not None and np.isfinite(q):
                axis_coverage[(sample.scene_type, axis)] += 1
                axis_percentiles[(sample.scene_type, axis)].append(float(q))
                axis_raw[(sample.scene_type, axis)].append(float(sample.axis_raw[sample.axis_names.index(axis)]))
        if sample.valid_axes == 0:
            no_valid_axis += 1
        all_ranks.append(sample.preference_rank)
        all_confidences.append(sample.rank_confidence)
        per_scene_ranks[sample.scene_type].append(sample.preference_rank)
        per_scene_confidences[sample.scene_type].append(sample.rank_confidence)

    low_confidence = sum(1 for c in all_confidences if c < 0.5)

    # ---------- 外部对照（仅当提供 style manifest 时）----------
    contrast = None
    if args.style_manifest:
        style_samples = load_style_index(args.style_manifest)
        key_to_style = {
            f"{s.log_name}:{s.token}" if s.log_name or s.token else s.cache_path: s.style
            for s in style_samples
        }
        paired_a, paired_b = [], []
        for sample in samples:
            style = key_to_style.get(sample.key)
            if style is not None and style in _STYLE_ORDINAL:
                paired_a.append(sample.preference_rank)
                paired_b.append(float(_STYLE_ORDINAL[style]))
        contrast = {
            "paired_samples": len(paired_a),
            "spearman_rank_vs_external_style": _spearman(paired_a, paired_b),
        }

    report = {
        "artifact_version": ARTIFACT_VERSION,
        "source": {"manifest": str(manifest_path.resolve())},
        "counts": {
            "samples": len(samples),
            "scene_counts": dict(scene_counts),
            "samples_no_valid_axis": no_valid_axis,
            "low_confidence_samples": low_confidence,
            "low_confidence_ratio": round(low_confidence / max(len(samples), 1), 4),
        },
        "axis_coverage": {"|".join(key): value for key, value in sorted(axis_coverage.items())},
        "axis_percentile_distribution": {
            "|".join(key): _percentiles(values) for key, values in sorted(axis_percentiles.items())
        },
        "axis_raw_distribution": {
            "|".join(key): _percentiles(values) for key, values in sorted(axis_raw.items())
        },
        "preference_rank_distribution": {
            "all": _percentiles(all_ranks),
            **{scene: _percentiles(values) for scene, values in sorted(per_scene_ranks.items())},
        },
        "rank_confidence_distribution": {
            "all": _percentiles(all_confidences),
            **{scene: _percentiles(values) for scene, values in sorted(per_scene_confidences.items())},
        },
        "external_style_contrast": contrast,
    }

    output_path = Path(args.output) if args.output else manifest_path.parent / "audit_preference.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote audit report to {output_path}")


if __name__ == "__main__":
    main()
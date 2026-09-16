"""按场景、置信度和有效轴数量分析 CSPQ 偏好空间对齐。

输入来自 export_preference_predictions.py。脚本输出分组指标、rank/scene
confidence 过滤曲线以及可直接用于论文绘图的 CSV，不重新加载模型。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from stylelora.scripts.evaluate_preference_encoder import _spearman


SCENES = ("straight_free_drive", "straight_car_follow")
AXIS_COUNT = 3


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _spearman_mae(prediction: Sequence[float], target: Sequence[float]) -> dict:
    if len(prediction) < 4:
        return {"count": len(prediction), "spearman": None, "mae": None}
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    return {
        "count": int(pred.shape[0]),
        "spearman": _json_number(_spearman(pred, truth)),
        "mae": float(np.mean(np.abs(pred - truth))),
    }


def _pairwise_accuracy(rows: Sequence[Mapping[str, object]], *, max_pairs: int,
                       min_rank_gap: float, rng: np.random.Generator) -> dict:
    """随机抽取有明确 rank 间隔的样本对，计算 s 排序正确率。"""
    n = len(rows)
    if n < 2 or max_pairs <= 0:
        return {"accuracy": None, "pairs": 0, "mean_rank_gap": None}
    ranks = np.asarray([float(row["rank"]) for row in rows], dtype=np.float64)
    scores = np.asarray([float(row["s"]) for row in rows], dtype=np.float64)
    correct: list[np.ndarray] = []
    gaps: list[np.ndarray] = []
    collected = 0
    attempts = 0
    while collected < max_pairs and attempts < 12:
        draw = min(max_pairs * 2, max(2048, (max_pairs - collected) * 3))
        left = rng.integers(0, n, size=draw)
        right = rng.integers(0, n, size=draw)
        rank_delta = ranks[left] - ranks[right]
        score_delta = scores[left] - scores[right]
        mask = (left != right) & (np.abs(rank_delta) >= min_rank_gap)
        if mask.any():
            usable = min(int(mask.sum()), max_pairs - collected)
            selected = np.flatnonzero(mask)[:usable]
            # score 完全相同按 0.5 计分，避免任意打破平局。
            product = rank_delta[selected] * score_delta[selected]
            correct.append(np.where(product > 0, 1.0, np.where(product == 0, 0.5, 0.0)))
            gaps.append(np.abs(rank_delta[selected]))
            collected += usable
        attempts += 1
    if not correct:
        return {"accuracy": None, "pairs": 0, "mean_rank_gap": None}
    values = np.concatenate(correct)
    rank_gaps = np.concatenate(gaps)
    return {
        "accuracy": float(values.mean()),
        "pairs": int(values.shape[0]),
        "mean_rank_gap": float(rank_gaps.mean()),
    }


def _group_metrics(rows: Sequence[Mapping[str, object]], *, max_pairs: int,
                   min_rank_gap: float, seed: int) -> dict:
    """计算一个分组的标量、三轴和 pairwise 对齐指标。"""
    s_values = [float(row["s"]) for row in rows]
    ranks = [float(row["rank"]) for row in rows]
    axis_metrics = {}
    for axis in range(AXIS_COUNT):
        prediction, target = [], []
        for row in rows:
            valid = list(row.get("axis_valid", []))
            q = list(row.get("q", []))
            q_hat = list(row.get("q_hat", []))
            if axis >= len(valid) or not bool(valid[axis]) or axis >= len(q) or axis >= len(q_hat):
                continue
            truth, pred = _number(q[axis]), _number(q_hat[axis])
            if truth is not None and pred is not None:
                target.append(truth)
                prediction.append(pred)
        axis_metrics[f"axis_{axis}"] = _spearman_mae(prediction, target)

    rank_conf = [float(row["rank_confidence"]) for row in rows]
    scene_conf = [_number(row.get("scene_confidence")) for row in rows]
    scene_conf = [value for value in scene_conf if value is not None]
    return {
        "count": len(rows),
        "s_vs_rank": _spearman_mae(s_values, ranks),
        "pairwise": _pairwise_accuracy(
            rows, max_pairs=max_pairs, min_rank_gap=min_rank_gap,
            rng=np.random.default_rng(seed),
        ),
        "q_hat_vs_q": axis_metrics,
        "mean_rank_confidence": float(np.mean(rank_conf)) if rank_conf else None,
        "mean_scene_confidence": float(np.mean(scene_conf)) if scene_conf else None,
        "scene_confidence_coverage": len(scene_conf) / len(rows) if rows else 0.0,
    }


def _bin_name(value: float | None, edges: Sequence[float]) -> str:
    if value is None:
        return "missing"
    for lower, upper in zip(edges[:-1], edges[1:]):
        if lower <= value < upper:
            return f"[{lower:.1f},{upper:.1f})"
    return f"[{edges[-1]:.1f},1.0]"


def _partition(rows: Sequence[dict], key_fn: Callable[[dict], str]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return groups


def _flatten_metrics(family: str, value: str, metrics: Mapping[str, object]) -> dict:
    row = {
        "group_family": family,
        "group_value": value,
        "count": metrics["count"],
        "s_spearman": metrics["s_vs_rank"]["spearman"],
        "s_mae": metrics["s_vs_rank"]["mae"],
        "pairwise_accuracy": metrics["pairwise"]["accuracy"],
        "pairwise_pairs": metrics["pairwise"]["pairs"],
        "mean_rank_confidence": metrics["mean_rank_confidence"],
        "mean_scene_confidence": metrics["mean_scene_confidence"],
        "scene_confidence_coverage": metrics["scene_confidence_coverage"],
    }
    for axis in range(AXIS_COUNT):
        item = metrics["q_hat_vs_q"][f"axis_{axis}"]
        row[f"axis_{axis}_count"] = item["count"]
        row[f"axis_{axis}_spearman"] = item["spearman"]
        row[f"axis_{axis}_mae"] = item["mae"]
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_thresholds(raw: str) -> list[float]:
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(value < 0 or value > 1 for value in values):
        raise ValueError("置信度阈值必须是 [0,1] 内的逗号分隔数值")
    return values


def _curve(rows: Sequence[dict], thresholds: Sequence[float], field: str, *,
           max_pairs: int, min_rank_gap: float, seed: int) -> list[dict]:
    """按阈值过滤，并同时报告全局及两个场景。"""
    output = []
    for threshold_index, threshold in enumerate(thresholds):
        candidates = [
            row for row in rows
            if _number(row.get(field)) is not None and float(row[field]) >= threshold
        ]
        for scene in ("all", *SCENES):
            subset = candidates if scene == "all" else [
                row for row in candidates if row["scene_type"] == scene
            ]
            metrics = _group_metrics(
                subset, max_pairs=max_pairs, min_rank_gap=min_rank_gap,
                seed=seed + threshold_index * 17 + (0 if scene == "all" else SCENES.index(scene) + 1),
            )
            flat = _flatten_metrics(field, f"{threshold:.2f}|{scene}", metrics)
            flat["threshold"] = threshold
            flat["scene"] = scene
            flat["retained_ratio"] = len(subset) / (
                len(rows) if scene == "all"
                else max(1, sum(row["scene_type"] == scene for row in rows))
            )
            output.append(flat)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="分组分析 CSPQ 偏好空间对齐")
    parser.add_argument("--predictions", required=True,
                        help="export_preference_predictions.py 输出 JSONL。")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank-thresholds", default="0,0.2,0.4,0.6,0.8")
    parser.add_argument("--scene-thresholds", default="0,0.2,0.4,0.6,0.8")
    parser.add_argument("--max-pairs", type=int, default=100000)
    parser.add_argument("--min-rank-gap", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.max_pairs <= 0 or not 0 <= args.min_rank_gap <= 1:
        parser.error("--max-pairs 必须为正；--min-rank-gap 必须在 [0,1]")
    rows = list(_iter_jsonl(Path(args.predictions)))
    if not rows:
        raise ValueError("预测文件为空")

    rank_thresholds = _parse_thresholds(args.rank_thresholds)
    scene_thresholds = _parse_thresholds(args.scene_thresholds)
    confidence_edges = (0.0, 0.2, 0.4, 0.6, 0.8)

    partitions = {
        "all": {"all": rows},
        "scene": _partition(rows, lambda row: str(row["scene_type"])),
        "rank_confidence_bin": _partition(
            rows, lambda row: _bin_name(_number(row.get("rank_confidence")), confidence_edges)),
        "scene_confidence_bin": _partition(
            rows, lambda row: _bin_name(_number(row.get("scene_confidence")), confidence_edges)),
        "valid_axis_count": _partition(rows, lambda row: str(int(row["valid_axis_count"]))),
        "scene_label_match": _partition(
            rows, lambda row: ("missing" if row.get("scene_label_match") is None
                               else str(bool(row["scene_label_match"])).lower())),
    }

    grouped_report: dict[str, dict] = {}
    grouped_csv = []
    group_seed = args.seed
    for family, groups in partitions.items():
        grouped_report[family] = {}
        for value, subset in sorted(groups.items()):
            metrics = _group_metrics(
                subset, max_pairs=args.max_pairs, min_rank_gap=args.min_rank_gap,
                seed=group_seed,
            )
            grouped_report[family][value] = metrics
            grouped_csv.append(_flatten_metrics(family, value, metrics))
            group_seed += 1

    rank_curve = _curve(
        rows, rank_thresholds, "rank_confidence",
        max_pairs=args.max_pairs, min_rank_gap=args.min_rank_gap, seed=args.seed + 1000,
    )
    scene_curve = _curve(
        rows, scene_thresholds, "scene_confidence",
        max_pairs=args.max_pairs, min_rank_gap=args.min_rank_gap, seed=args.seed + 2000,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(Path(args.predictions).resolve()),
        "samples": len(rows),
        "settings": {
            "rank_thresholds": rank_thresholds,
            "scene_thresholds": scene_thresholds,
            "max_pairs": args.max_pairs,
            "min_rank_gap": args.min_rank_gap,
            "seed": args.seed,
        },
        "groups": grouped_report,
        "rank_confidence_curve": rank_curve,
        "scene_confidence_curve": scene_curve,
    }
    report_path = output_dir / "grouped_alignment_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "grouped_alignment.csv", grouped_csv)
    _write_csv(output_dir / "rank_confidence_curve.csv", rank_curve)
    _write_csv(output_dir / "scene_confidence_curve.csv", scene_curve)

    print(json.dumps({
        "samples": len(rows),
        "overall": grouped_report["all"]["all"],
        "outputs": {
            "report": str(report_path),
            "groups_csv": str(output_dir / "grouped_alignment.csv"),
            "rank_curve_csv": str(output_dir / "rank_confidence_curve.csv"),
            "scene_curve_csv": str(output_dir / "scene_confidence_curve.csv"),
        },
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

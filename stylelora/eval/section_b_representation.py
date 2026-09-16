"""生成实验小节 B 的风格表征表格与潜在空间图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stylelora.eval.common import (
    COLORS, add_panel_labels, iter_jsonl, load_json, mpl, parse_named_paths,
    save_figure, section_dir, style_axes, write_csv,
)


def _report_row(name: str, report: dict) -> dict:
    val = report.get("val", report)
    scalar = val.get("s_vs_rank", {}).get("all", val.get("s_vs_rank", {}))
    axes = val.get("q_hat_vs_q", {})
    nearest = val.get("nearest_rank_error", {})
    dims = val.get("latent_dims", {})
    row = {
        "method": name,
        "s_spearman": scalar.get("spearman"),
        "s_mae": scalar.get("mae"),
        "effective_dim_95": dims.get("effective_dim_95"),
        "nearest_rank_error": nearest.get("same_scene", nearest.get("nearest", None)),
        "random_rank_error": nearest.get("random_same_scene", nearest.get("random", None)),
    }
    for index in range(3):
        metric = axes.get(f"axis_{index}", {})
        row[f"axis_{index}_spearman"] = metric.get("spearman")
        row[f"axis_{index}_mae"] = metric.get("mae")
    return row


def _load_predictions(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks, scores, latents = [], [], []
    for row_index, row in enumerate(iter_jsonl(path), start=1):
        if row.get("z") is None or row.get("s") is None or row.get("rank") is None:
            continue
        ranks.append(float(row["rank"]))
        scores.append(float(row["s"]))
        latents.append(np.asarray(row["z"], dtype=np.float64))
        if row_index == 1 or row_index % 2000 == 0:
            print(f"[progress] 读取表征预测: {row_index} 行", flush=True)
    if not latents:
        raise ValueError("预测文件没有同时包含 rank、s 和 z 的有效样本")
    rank = np.asarray(ranks); score = np.asarray(scores); latent = np.stack(latents)
    if rank.size > max_points:
        pick = np.random.default_rng(seed).choice(rank.size, max_points, replace=False)
        rank, score, latent = rank[pick], score[pick], latent[pick]
    return rank, score, latent


def _binned_calibration(rank: np.ndarray, score: np.ndarray, bins: int = 10):
    edges = np.quantile(rank, np.linspace(0, 1, bins + 1))
    x, y, yerr = [], [], []
    for index in range(bins):
        mask = (rank >= edges[index]) & (rank <= edges[index + 1] if index == bins - 1 else rank < edges[index + 1])
        if not np.any(mask):
            continue
        x.append(float(rank[mask].mean())); y.append(float(score[mask].mean()))
        yerr.append(float(score[mask].std(ddof=1) / np.sqrt(mask.sum())) if mask.sum() > 1 else 0.0)
    return x, y, yerr


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 CAST 论文表2和图1：结构化风格表征。")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--encoder-report", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ablation-report", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--max-points", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    output = section_dir(args.output_root, "B")
    print("[stage 1/4] 读取偏好编码器报告。", flush=True)
    rows = [_report_row("Full CAST", load_json(args.encoder_report))]
    for name, path in parse_named_paths(args.ablation_report).items():
        rows.append(_report_row(name, load_json(path)))
    fields = ["method", "s_spearman", "s_mae", "axis_0_spearman", "axis_0_mae",
              "axis_1_spearman", "axis_1_mae", "axis_2_spearman", "axis_2_mae",
              "effective_dim_95", "nearest_rank_error", "random_rank_error"]
    write_csv(output / "table_2_style_representation.csv", rows, fields)
    print("[stage 2/4] 表2写入完成。", flush=True)

    print("[stage 3/4] 读取逐样本预测并计算 PCA 与校准曲线。", flush=True)
    rank, score, latent = _load_predictions(Path(args.predictions), args.max_points, args.seed)
    centered = latent - latent.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[:2].T
    x, y, yerr = _binned_calibration(rank, score)
    plt = mpl(); fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
    points = axes[0].scatter(
        projection[:, 0], projection[:, 1], c=rank, s=7, alpha=0.55,
        cmap="viridis", edgecolors="none", rasterized=True,
    )
    axes[0].set(title="Structured latent style space", xlabel="PC1", ylabel="PC2")
    colorbar = fig.colorbar(points, ax=axes[0], label="Reference style rank", pad=0.02)
    colorbar.outline.set_linewidth(0.6)
    axes[1].errorbar(
        x, y, yerr=np.asarray(yerr) * 1.96, marker="o", capsize=2.5,
        color=COLORS["blue"], markerfacecolor="white", markeredgewidth=1.1,
        label="Binned mean ± 95% CI",
    )
    axes[1].set(title="Style-score calibration", xlabel="Reference style rank", ylabel="Predicted style score $s$")
    axes[1].legend(loc="best")
    style_axes(axes)
    add_panel_labels(axes)
    fig.tight_layout(pad=0.7, w_pad=1.0)
    save_figure(fig, output / "figure_1_style_representation")
    plt.close(fig)
    print("[stage 4/4] 图1的 PNG/PDF 写入完成。", flush=True)
    print(f"表2、图1 -> {output}")


if __name__ == "__main__":
    main()

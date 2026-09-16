"""生成实验小节 D 的自适应强度校准图和门控对比表。"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from stylelora.eval.common import (
    COLORS, add_panel_labels, load_json, mpl, parse_named_paths, save_figure,
    section_dir, style_axes, write_csv,
)
from stylelora.model.scene_gate import load_scene_gate_checkpoint
from stylelora.scripts.train_scene_gate import SceneGateDataset


@torch.no_grad()
def _gate_predictions(checkpoint: str, targets: str, feature_npy: str, feature_index: str,
                      device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = SceneGateDataset(targets, feature_npy, feature_index)
    gate, _ = load_scene_gate_checkpoint(checkpoint, device)
    predictions, references = [], []
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    progress_interval = max(1, total_batches // 20)
    for batch_index, start in enumerate(range(0, len(dataset), batch_size), start=1):
        batch = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        h_c = torch.stack([row["h_c"] for row in batch]).to(device)
        predictions.append(gate(h_c).cpu().numpy())
        references.append(torch.stack([row["target"] for row in batch]).numpy())
        if batch_index == 1 or batch_index == total_batches or batch_index % progress_interval == 0:
            print(
                f"[progress] 门控验证集前向: {batch_index}/{total_batches} "
                f"({batch_index / max(total_batches, 1):.1%})",
                flush=True,
            )
    return np.concatenate(predictions), np.concatenate(references)


def _mean_record(report: dict, key: str) -> float | None:
    values = [float(row[key]) for row in report.get("records", [])
              if isinstance(row, dict) and row.get(key) is not None and np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else None


def _comparison_row(name: str, report: dict) -> dict:
    per_rho = list(report.get("per_rho", {}).values())
    def means(metric: str):
        values = [entry.get(metric, {}).get("mean") for entry in per_rho if isinstance(entry, dict)]
        return [float(value) for value in values if value is not None and np.isfinite(float(value))]
    direction = report.get("direction_check", {})
    continuous = report.get("continuous_rho", {})
    coverage = continuous.get("effective_control_coverage", {})
    performance_cost = continuous.get("performance_cost_vs_rho_zero", {})
    return {
        "method": name,
        "style_span_mean": continuous.get("style_span", {}).get("mean"),
        "mean_samplewise_slope": continuous.get("average_slope", {}).get("mean"),
        "positive_slope_rate": continuous.get("positive_slope_rate"),
        "nonzero_response_rate": continuous.get("nonzero_response_rate"),
        "effective_control_coverage": coverage.get("overall"),
        "command_retention_ratio_mean": coverage.get("command_retention_ratio", {}).get("mean"),
        "ade_cost_vs_zero_mean": performance_cost.get("ade_delta", {}).get("mean"),
        "fde_cost_vs_zero_mean": performance_cost.get("fde_delta", {}).get("mean"),
        "jerk_cost_vs_zero_mean": performance_cost.get("abs_jerk_p90_delta", {}).get("mean"),
        "sign_correct_vs_zero_rate": direction.get("sign_correct_vs_zero_rate"),
        "high_low_direction_rate": direction.get("plus_minus_rate"),
        "mean_ade_over_rho": float(np.mean(means("ade"))) if means("ade") else None,
        "mean_fde_over_rho": float(np.mean(means("fde"))) if means("fde") else None,
        "mean_lateral_shift_vs_baseline": _mean_record(report, "lateral_shift_vs_baseline"),
        "mean_abs_jerk_p90": _mean_record(report, "planned_abs_jerk_p90"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 CAST 论文表4和图3：自适应强度调制。")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gate-checkpoint", required=True)
    parser.add_argument("--gate-report", required=True)
    parser.add_argument("--val-targets", required=True)
    parser.add_argument("--val-feature-npy", required=True)
    parser.add_argument("--val-feature-index", required=True)
    parser.add_argument("--adaptive-open-report", required=True)
    parser.add_argument("--comparison-report", action="append", default=[], metavar="NAME=PATH",
                        help="至少传入 No gate；有固定上限结果时再传 Global fixed cap。")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = section_dir(args.output_root, "D")
    print("[stage 1/4] 加载门控、验证目标和场景表征。", flush=True)
    prediction, target = _gate_predictions(
        args.gate_checkpoint, args.val_targets, args.val_feature_npy,
        args.val_feature_index, args.device, args.batch_size,
    )
    gate_report = load_json(args.gate_report).get("best_validation", {})
    adaptive = load_json(args.adaptive_open_report)
    print("[stage 2/4] 门控前向与开环报告读取完成。", flush=True)
    rows = []
    for name, path in parse_named_paths(args.comparison_report).items():
        rows.append(_comparison_row(name, load_json(path)))
    adaptive_row = _comparison_row("Adaptive gate", adaptive)
    adaptive_row.update({
        "gate_mae_low": gate_report.get("mae_low"),
        "gate_mae_high": gate_report.get("mae_high"),
        "gate_overestimate_low_rate": gate_report.get("overestimate_low_rate"),
        "gate_overestimate_high_rate": gate_report.get("overestimate_high_rate"),
    })
    rows.append(adaptive_row)
    fields = [
        "method", "style_span_mean", "mean_samplewise_slope", "positive_slope_rate",
        "nonzero_response_rate", "effective_control_coverage", "command_retention_ratio_mean",
        "ade_cost_vs_zero_mean", "fde_cost_vs_zero_mean", "jerk_cost_vs_zero_mean",
        "sign_correct_vs_zero_rate", "high_low_direction_rate",
        "mean_ade_over_rho", "mean_fde_over_rho",
        "mean_lateral_shift_vs_baseline", "mean_abs_jerk_p90",
        "gate_mae_low", "gate_mae_high", "gate_overestimate_low_rate",
        "gate_overestimate_high_rate",
    ]
    write_csv(output / "table_4_adaptive_gate_comparison.csv", rows, fields)
    print("[stage 3/4] 表4写入完成。", flush=True)

    plt = mpl(); fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
    colors = (COLORS["blue"], COLORS["orange"])
    for index, label in enumerate(("Low-direction cap", "High-direction cap")):
        sample = np.linspace(0, prediction.shape[0] - 1, min(5000, prediction.shape[0]), dtype=int)
        axes[0].scatter(
            target[sample, index], prediction[sample, index], s=7, alpha=0.28,
            color=colors[index], edgecolors="none", label=label, rasterized=True,
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], linewidth=1,
                 label="Ideal calibration")
    axes[0].set(title="Adaptive-cap calibration", xlabel="Target cap", ylabel="Predicted cap", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(frameon=False)

    records = [row for row in adaptive.get("records", []) if isinstance(row, dict)]
    requested = sorted({float(row["rho"]) for row in records})
    effective = [[float(row.get("effective_rho", row["rho"])) for row in records
                  if abs(float(row["rho"]) - rho) < 1e-9] for rho in requested]
    boxes = axes[1].boxplot(
        effective, positions=requested, widths=0.12, showfliers=False,
        manage_ticks=False, patch_artist=True,
        medianprops={"color": COLORS["orange"], "linewidth": 1.4},
        whiskerprops={"color": COLORS["blue"], "linewidth": 1.0},
        capprops={"color": COLORS["blue"], "linewidth": 1.0},
    )
    for box in boxes["boxes"]:
        box.set(facecolor=COLORS["sky"], alpha=0.35, edgecolor=COLORS["blue"], linewidth=0.9)
    axes[1].plot(requested, requested, linestyle="--", color=COLORS["gray"], linewidth=1,
                 label="Requested = effective")
    axes[1].set(title="Requested and effective style intensity", xlabel="Requested $\\rho$", ylabel="Effective $\\rho$")
    axes[1].legend(frameon=False)
    style_axes(axes)
    add_panel_labels(axes)
    fig.tight_layout(pad=0.7, w_pad=1.0)
    save_figure(fig, output / "figure_3_adaptive_intensity")
    plt.close(fig)
    print("[stage 4/4] 图3的 PNG/PDF 写入完成。", flush=True)
    print(f"表4、图3 -> {output}")


if __name__ == "__main__":
    main()

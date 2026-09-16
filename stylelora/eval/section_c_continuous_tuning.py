"""生成实验小节 C 的连续风格调节曲线和主结果表。"""

from __future__ import annotations

import argparse

import numpy as np

from stylelora.eval.common import (
    COLORS, add_panel_labels, aggregate_record_metric, load_json, mpl,
    parse_named_paths, rho_rows, save_figure, section_dir, style_axes, write_csv,
)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _spearman(x, y) -> float | None:
    if len(x) < 2:
        return None
    a, b = _rank(np.asarray(x, dtype=np.float64)), _rank(np.asarray(y, dtype=np.float64))
    return float(np.corrcoef(a, b)[0, 1])


def _summary_row(name: str, report: dict) -> dict:
    rho_entries = rho_rows(report)
    rhos = [rho for rho, _ in rho_entries]
    style = [float(entry.get("s", {}).get("mean")) for _, entry in rho_entries]
    ade = [float(entry.get("ade", {}).get("mean")) for _, entry in rho_entries]
    fde = [float(entry.get("fde", {}).get("mean")) for _, entry in rho_entries]
    direction = report.get("direction_check", {})
    continuous = report.get("continuous_rho", {})
    sample_monotonic = continuous.get("samplewise_s_monotonic_rate")
    if sample_monotonic is None:
        sample_monotonic = float(bool(report.get("monotonicity", {}).get("monotonic_increasing", False)))
    coverage = continuous.get("effective_control_coverage", {})
    performance_cost = continuous.get("performance_cost_vs_rho_zero", {})
    return {
        "method": name,
        "rho_style_spearman": _spearman(rhos, style),
        "sample_monotonic_rate": sample_monotonic,
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
        "rho_zero_identity_max_error": report.get("identity_max_mismatch"),
        "mean_ade_over_rho": float(np.mean(ade)) if ade else None,
        "mean_fde_over_rho": float(np.mean(fde)) if fde else None,
    }


def _series(report: dict, metric: str):
    aggregated = aggregate_record_metric(report, metric)
    if not aggregated:
        raise ValueError(
            f"开环报告缺少逐样本字段 {metric}；请使用更新后的 "
            "stylelora.scripts.evaluate_preference_lora 重新生成报告"
        )
    rho = np.asarray(sorted(aggregated), dtype=np.float64)
    mean = np.asarray([aggregated[value][0] for value in rho])
    low = np.asarray([aggregated[value][1] for value in rho])
    high = np.asarray([aggregated[value][2] for value in rho])
    return rho, mean, low, high


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 CAST 论文表3和图2：连续风格调节。")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cast-report", required=True)
    parser.add_argument("--comparison-report", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()

    output = section_dir(args.output_root, "C")
    print(f"[stage 1/3] 读取密集 rho 开环报告：{args.cast_report}", flush=True)
    cast = load_json(args.cast_report)
    rows = [_summary_row("Full CAST", cast)]
    for name, path in parse_named_paths(args.comparison_report).items():
        rows.append(_summary_row(name, load_json(path)))
    write_csv(output / "table_3_continuous_tuning.csv", rows)
    print("[stage 2/3] 表3写入完成，开始绘制三个连续响应面板。", flush=True)

    metrics = [
        ("s", "Style score $s$", "Continuous style response"),
        ("planned_mean_speed", "Mean planned speed (m/s)", "Longitudinal speed response"),
        ("planned_abs_jerk_p90", "Absolute jerk P90 (m/s³)", "Dynamic response"),
    ]
    plt = mpl(); fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    for axis, (metric, ylabel, title) in zip(axes, metrics):
        rho, mean, low, high = _series(cast, metric)
        axis.plot(
            rho, mean, marker="o", color=COLORS["blue"],
            markerfacecolor="white", markeredgewidth=1.0, zorder=3,
        )
        axis.fill_between(rho, low, high, color=COLORS["sky"], alpha=0.22, label="95% CI")
        axis.axvline(0.0, color=COLORS["gray"], linewidth=0.9, linestyle=":")
        axis.set(title=title, xlabel="Requested style intensity $\\rho$", ylabel=ylabel)
    axes[0].legend(frameon=False)
    style_axes(axes)
    add_panel_labels(axes)
    fig.tight_layout(pad=0.6, w_pad=0.8)
    save_figure(fig, output / "figure_2_continuous_style_tuning")
    plt.close(fig)
    print("[stage 3/3] 图2的 PNG/PDF 写入完成。", flush=True)
    print(f"表3、图2 -> {output}")


if __name__ == "__main__":
    main()

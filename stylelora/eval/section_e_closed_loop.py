"""生成实验小节 E 的 NuPlan 正式闭环表格与配对结果图。"""

from __future__ import annotations

import argparse

import numpy as np

from stylelora.eval.common import (
    COLORS, add_panel_labels, bootstrap_mean_ci, load_json, mpl,
    parse_named_paths, save_figure, section_dir, style_axes, write_csv,
)


SAFETY = (
    "no_ego_at_fault_collisions", "time_to_collision_within_bound",
    "drivable_area_compliance", "driving_direction_compliance",
    "speed_limit_compliance", "ego_is_comfortable",
)


def _style_entry(report: dict, rho: float) -> dict:
    proxy = report.get("style_proxy", {}).get("per_rho", {})
    entry = proxy.get(f"rho_{rho:+.2f}", {})
    return entry.get("metrics_on_common_steps", {}) if isinstance(entry, dict) else {}


def _paired_deltas(report: dict, rho: float) -> list[float]:
    key = f"rho_{rho:+.2f}"
    entry = report.get("paired_official_differences_vs_rho_zero", {}).get(key, {})
    return [float(row["challenge_score_delta"]) for row in entry.get("per_scenario", [])
            if row.get("challenge_score_delta") is not None]


def _rows(name: str, report: dict, seed: int) -> list[dict]:
    output = []
    for record in sorted(report.get("records", []), key=lambda row: float(row["rho"])):
        rho = float(record["rho"])
        official = record.get("official_aggregator", {})
        style = _style_entry(report, rho)
        deltas = _paired_deltas(report, rho)
        mean_delta, ci_low, ci_high, paired_count = bootstrap_mean_ci(deltas, seed=seed)
        row = {
            "method": name, "rho": rho,
            "scenario_count": official.get("scenario_count", record.get("scenario_count")),
            "challenge_score": official.get("challenge_score"),
            "challenge_delta_vs_rho0": 0.0 if abs(rho) < 1e-9 else mean_delta,
            "delta_ci95_low": 0.0 if abs(rho) < 1e-9 else ci_low,
            "delta_ci95_high": 0.0 if abs(rho) < 1e-9 else ci_high,
            "paired_scenarios": official.get("scenario_count") if abs(rho) < 1e-9 else paired_count,
            "current_ego_speed": style.get("current_ego_speed", {}).get("mean"),
            "planned_abs_jerk_p90": style.get("planned_abs_jerk_p90", {}).get("mean"),
            "effective_rho": style.get("effective_rho", {}).get("mean"),
        }
        for metric in SAFETY:
            row[metric] = official.get("safety_metric_means", {}).get(metric)
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 CAST 论文表5和图4：正式闭环结果。")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cast-report", required=True)
    parser.add_argument("--comparison-report", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    output = section_dir(args.output_root, "E")
    print(f"[stage 1/3] 读取正式闭环报告：{args.cast_report}", flush=True)
    cast = load_json(args.cast_report)
    rows = []
    for name, path in parse_named_paths(args.comparison_report).items():
        print(f"[progress] 汇总闭环对比方法：{name}", flush=True)
        rows.extend(_rows(name, load_json(path), args.seed))
    rows.extend(_rows("Full CAST", cast, args.seed))
    fields = ["method", "rho", "scenario_count", "challenge_score", "challenge_delta_vs_rho0",
              "delta_ci95_low", "delta_ci95_high", "paired_scenarios", *SAFETY,
              "current_ego_speed", "planned_abs_jerk_p90", "effective_rho"]
    write_csv(output / "table_5_closed_loop_performance.csv", rows, fields)
    print("[stage 2/3] 表5写入完成。", flush=True)

    cast_rows = _rows("Full CAST", cast, args.seed)
    rho = np.asarray([row["rho"] for row in cast_rows])
    speed = np.asarray([row["current_ego_speed"] for row in cast_rows], dtype=float)
    jerk = np.asarray([row["planned_abs_jerk_p90"] for row in cast_rows], dtype=float)
    plt = mpl(); fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    axes[0].plot(
        rho, speed, marker="o", color=COLORS["blue"], markerfacecolor="white",
        markeredgewidth=1.0,
    )
    axes[0].axvline(0.0, color=COLORS["gray"], linewidth=0.9, linestyle=":")
    axes[0].set(title="Closed-loop speed", xlabel="Requested $\\rho$", ylabel="Ego speed (m/s)")
    axes[1].plot(
        rho, jerk, marker="s", color=COLORS["orange"], markerfacecolor="white",
        markeredgewidth=1.0,
    )
    axes[1].axvline(0.0, color=COLORS["gray"], linewidth=0.9, linestyle=":")
    axes[1].set(title="Closed-loop dynamics", xlabel="Requested $\\rho$",
                ylabel="Absolute jerk P90 (m/s³)")

    nonzero = [value for value in rho if abs(value) > 1e-9]
    distributions = [_paired_deltas(cast, float(value)) for value in nonzero]
    boxes = axes[2].boxplot(
        distributions, labels=[f"{value:+.1f}" for value in nonzero],
        showfliers=False, patch_artist=True,
        medianprops={"color": COLORS["orange"], "linewidth": 1.4},
    )
    for box in boxes["boxes"]:
        box.set(facecolor=COLORS["sky"], alpha=0.35, edgecolor=COLORS["blue"], linewidth=0.9)
    axes[2].axhline(0.0, color=COLORS["gray"], linewidth=0.9, linestyle="--")
    axes[2].set(title="Paired challenge score", xlabel="Requested $\\rho$",
                ylabel="Difference from $\\rho=0$")
    style_axes(axes)
    add_panel_labels(axes)
    fig.tight_layout(pad=0.6, w_pad=0.8)
    save_figure(fig, output / "figure_4_closed_loop_performance")
    plt.close(fig)
    print("[stage 3/3] 图4的 PNG/PDF 写入完成。", flush=True)
    print(f"表5、图4 -> {output}")


if __name__ == "__main__":
    main()

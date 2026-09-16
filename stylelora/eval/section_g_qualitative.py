"""生成实验小节 G 的代表性闭环轨迹与速度剖面对比图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stylelora.eval.common import (
    COLORS, add_panel_labels, mpl, parse_named_paths, save_figure, section_dir, style_axes,
)


def _scalar(payload, name: str, default: float = float("nan")) -> float:
    if name not in payload:
        return default
    values = np.asarray(payload[name]).reshape(-1)
    return float(values[0]) if values.size else default


def _index_steps(
    root: Path,
    allowed_tokens: set[str] | None = None,
) -> dict[tuple[str, int, int], Path]:
    """采用与正式闭环报告相同的目录、时间和迭代编号进行配对。"""
    indexed: dict[tuple[str, int, int], Path] = {}
    paths = sorted(root.rglob("*.npz"))
    print(f"[progress] 扫描 {root}: 找到 {len(paths)} 个 npz。", flush=True)
    progress_interval = max(1, len(paths) // 20)
    for path_index, path in enumerate(paths, start=1):
        try:
            with np.load(path, allow_pickle=False) as payload:
                if "generated_ego_future" not in payload or "time_us" not in payload:
                    continue
                scenario_token = (
                    str(np.asarray(payload["scenario_token"]).reshape(-1)[0])
                    if "scenario_token" in payload else ""
                )
                if allowed_tokens is not None and scenario_token not in allowed_tokens:
                    continue
                time_us = int(np.asarray(payload["time_us"]).reshape(-1)[0])
                iteration = int(np.asarray(payload.get("iteration_index", 0)).reshape(-1)[0])
        except (OSError, ValueError, KeyError):
            continue
        relative_parent = path.relative_to(root).parent.as_posix()
        key = (relative_parent, time_us, iteration)
        if key in indexed:
            raise ValueError(
                f"{root} 中存在重复闭环时刻：{relative_parent}, "
                f"time_us={time_us}, iteration={iteration}"
            )
        indexed[key] = path
        if path_index == 1 or path_index == len(paths) or path_index % progress_interval == 0:
            print(
                f"[progress] 索引闭环 npz: {path_index}/{len(paths)} "
                f"({path_index / max(len(paths), 1):.1%})",
                flush=True,
            )
    return indexed


def _load_step(path: Path, dt: float) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        future = np.asarray(payload["generated_ego_future"], dtype=np.float64)
        current = np.asarray(payload.get("ego_current_state", np.zeros(2)), dtype=np.float64).reshape(-1)
        current_xy = current[:2] if current.size >= 2 else np.zeros(2, dtype=np.float64)
        trajectory = np.concatenate((current_xy[None], future[:, :2]), axis=0)
        speed = np.linalg.norm(np.diff(trajectory, axis=0), axis=-1) / dt
        neighbors = np.asarray(payload.get("neighbor_current_state", np.zeros((0, 2))), dtype=np.float64)
        if neighbors.ndim != 2 or neighbors.shape[-1] < 2:
            neighbors = np.zeros((0, 2), dtype=np.float64)
        return {
            "trajectory": trajectory,
            "speed": speed,
            "neighbors": neighbors[:, :2],
            "requested_rho": _scalar(payload, "requested_rho"),
            "effective_rho": _scalar(payload, "effective_rho"),
        }


def _select_keys(
    indices: dict[float, dict[tuple[str, int, int], Path]], count: int
) -> list[tuple[str, int, int]]:
    common = sorted(set.intersection(*(set(index) for index in indices.values())))
    if not common:
        raise ValueError("不同 rho 的 raw 根目录之间没有可配对的闭环时刻")

    # 固定、可复现地选择：风格差异最大、门控最弱和门控最强的时刻。
    scored = []
    low_rho, high_rho = min(indices), max(indices)
    progress_interval = max(1, len(common) // 20)
    for key_index, key in enumerate(common, start=1):
        low = _load_step(indices[low_rho][key], 0.1)
        high = _load_step(indices[high_rho][key], 0.1)
        endpoint_gap = float(np.linalg.norm(low["trajectory"][-1] - high["trajectory"][-1]))
        effective = abs(high["effective_rho"])
        scored.append((key, endpoint_gap, effective))
        if key_index == 1 or key_index == len(common) or key_index % progress_interval == 0:
            print(
                f"[progress] 评估定性候选: {key_index}/{len(common)} "
                f"({key_index / len(common):.1%})",
                flush=True,
            )

    candidates = [
        max(scored, key=lambda item: item[1]),
        min(scored, key=lambda item: (item[2], item[0])),
        max(scored, key=lambda item: (item[2], item[0])),
    ]
    chosen = []
    for item in candidates + sorted(scored, key=lambda item: item[1], reverse=True):
        if item[0] not in chosen:
            chosen.append(item[0])
        if len(chosen) == min(count, len(common)):
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 CAST 论文图5：代表性闭环轨迹与速度剖面。")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--raw-root", action="append", required=True, metavar="RHO=PATH",
                        help="重复传入三个闭环 raw 根目录，例如 --raw-root=-1=...。")
    parser.add_argument("--time-us", action="append", type=int, default=[],
                        help="可选：指定需要绘制的仿真时间戳；不传时自动选取三个代表时刻。")
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--allowed-tokens-file",
        default=None,
        help="可选 JSON token 列表；提供后只从正式共同可行场景中选择定性样本。",
    )
    args = parser.parse_args()

    named = parse_named_paths(args.raw_root)
    roots = {float(name): path for name, path in named.items()}
    required = {-1.0, 0.0, 1.0}
    if set(roots) != required:
        raise ValueError("图5只接受且必须同时提供 rho=-1、0、+1 三个 raw 根目录")
    allowed_tokens = None
    if args.allowed_tokens_file:
        payload = json.loads(Path(args.allowed_tokens_file).read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("--allowed-tokens-file 必须是非空 JSON token 列表")
        allowed_tokens = {str(token) for token in payload}
    print("[stage 1/3] 索引 rho=-1、0、+1 的闭环 raw 数据。", flush=True)
    indices = {
        rho: _index_steps(path, allowed_tokens=allowed_tokens)
        for rho, path in roots.items()
    }
    if any(not index for index in indices.values()):
        raise ValueError("至少一个 raw 根目录中没有找到有效 npz")

    if args.time_us:
        common = set.intersection(*(set(index) for index in indices.values()))
        keys = [key for time_us in args.time_us for key in sorted(common) if key[1] == time_us]
        if not keys:
            raise ValueError("指定的 time_us 在三个 rho 中没有共同样本")
        keys = keys[:args.max_cases]
    else:
        keys = _select_keys(indices, args.max_cases)
    print(f"[stage 2/3] 已选定 {len(keys)} 个公共闭环时刻，开始绘图。", flush=True)

    plt = mpl()
    fig, axes = plt.subplots(len(keys), 2, figsize=(7.2, 2.25 * len(keys)), squeeze=False)
    colors = {-1.0: COLORS["blue"], 0.0: COLORS["gray"], 1.0: COLORS["orange"]}
    linestyles = {-1.0: "--", 0.0: "-", 1.0: "-."}
    legend_handles, legend_labels = [], []
    for row_index, key in enumerate(keys):
        trajectory_axis, speed_axis = axes[row_index]
        for rho in (-1.0, 0.0, 1.0):
            sample = _load_step(indices[rho][key], args.dt)
            effective = sample["effective_rho"]
            label = rf"requested $\rho$={rho:+.0f}, effective={effective:+.2f}"
            trajectory = sample["trajectory"]
            trajectory_line, = trajectory_axis.plot(
                trajectory[:, 0], trajectory[:, 1], color=colors[rho],
                linestyle=linestyles[rho], linewidth=1.8, label=label,
            )
            trajectory_axis.scatter(
                trajectory[-1, 0], trajectory[-1, 1], s=17, color=colors[rho],
                marker="o", edgecolor="white", linewidth=0.45, zorder=4,
            )
            if row_index == 0:
                legend_handles.append(trajectory_line)
                legend_labels.append(label)
            time = np.arange(1, len(sample["speed"]) + 1) * args.dt
            speed_axis.plot(
                time, sample["speed"], color=colors[rho], linestyle=linestyles[rho],
                linewidth=1.8, label=label,
            )
            if rho == 0.0 and sample["neighbors"].size:
                trajectory_axis.scatter(sample["neighbors"][:, 0], sample["neighbors"][:, 1],
                                        color=COLORS["light_gray"], edgecolor=COLORS["gray"],
                                        linewidth=0.35, s=12, alpha=0.75)
        trajectory_axis.set(
            title=f"Case {row_index + 1}: trajectory (time_us={key[1]})",
            xlabel="Longitudinal position (m)", ylabel="Lateral position (m)",
        )
        trajectory_axis.axis("equal")
        speed_axis.set(
            title=f"Case {row_index + 1}: planned speed profile",
            xlabel="Planning horizon (s)", ylabel="Speed (m/s)",
        )
    style_axes(axes)
    add_panel_labels(axes)
    fig.legend(
        legend_handles, legend_labels, loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 1.005), frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), pad=0.65, h_pad=1.0, w_pad=0.9)
    output = section_dir(args.output_root, "G")
    save_figure(fig, output / "figure_5_qualitative_cases")
    plt.close(fig)
    print("[stage 3/3] 图5的 PNG/PDF 写入完成。", flush=True)
    print(f"图5 -> {output}")


if __name__ == "__main__":
    main()

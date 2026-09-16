"""论文图表脚本共享的轻量读写、统计和绘图工具。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SECTION_DIRS = {
    "A": "A_experimental_setup",
    "B": "B_style_representation",
    "C": "C_continuous_tuning",
    "D": "D_adaptive_tuning",
    "E": "E_closed_loop",
    "F": "F_ablation_efficiency",
    "G": "G_qualitative",
}

# Okabe–Ito 色盲友好配色；各图固定语义，避免同一方法在不同图中换色。
COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#E69F00",
    "gray": "#6B6B6B",
    "light_gray": "#D9D9D9",
}


def section_dir(output_root: str | Path, section: str) -> Path:
    """创建并返回固定的小节输出目录。"""
    if section not in SECTION_DIRS:
        raise ValueError(f"未知论文小节：{section}")
    path = Path(output_root) / SECTION_DIRS[section]
    path.mkdir(parents=True, exist_ok=True)
    return path


def progress(label: str, current: int, total: int, *, updates: int = 10) -> None:
    """按固定比例输出轻量进度，不改变循环或数据处理逻辑。"""
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    interval = max(1, math.ceil(total / max(int(updates), 1)))
    if current in {1, total} or current % interval == 0:
        print(f"[progress] {label}: {current}/{total} ({current / total:.1%})", flush=True)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if isinstance(row, dict):
                yield row


def count_jsonl(path: str | Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    """以 UTF-8 BOM 写出 CSV，便于 Excel 和论文脚本直接读取。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_named_paths(values: Sequence[str] | None) -> dict[str, Path]:
    """解析重复的 ``名称=路径`` 参数。"""
    output: dict[str, Path] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"参数必须采用 名称=路径：{raw}")
        name, path = raw.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path or name in output:
            raise ValueError(f"名称为空、路径为空或名称重复：{raw}")
        output[name] = Path(path)
    return output


def mean_ci(values: Sequence[float], *, confidence_z: float = 1.96) -> tuple[float, float, float, int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(array.mean())
    if array.size == 1:
        return mean, mean, mean, 1
    half = confidence_z * float(array.std(ddof=1)) / math.sqrt(array.size)
    return mean, mean - half, mean + half, int(array.size)


def bootstrap_mean_ci(values: Sequence[float], *, seed: int = 17, samples: int = 5000) -> tuple[float, float, float, int]:
    """对逐场景配对差异计算确定性的均值 bootstrap 95% CI。"""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan"), float("nan"), 0
    if array.size == 1:
        value = float(array[0])
        return value, value, value, 1
    rng = np.random.default_rng(seed)
    chunk = 512
    means = []
    remaining = int(samples)
    while remaining > 0:
        take = min(chunk, remaining)
        indices = rng.integers(0, array.size, size=(take, array.size))
        means.append(array[indices].mean(axis=1))
        remaining -= take
    boot = np.concatenate(means)
    return float(array.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), int(array.size)


def mpl():
    """延迟加载无界面绘图后端，服务器无需显示器。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.7,
        "lines.markersize": 4.5,
        "axes.grid": False,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def style_axes(axes) -> None:
    """统一坐标轴留白、浅色横向网格和边框，保证缩印后仍清晰。"""
    array = np.asarray(axes, dtype=object).reshape(-1)
    for axis in array:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", linestyle="--", dashes=(2.5, 2.5))
        axis.set_axisbelow(True)


def add_panel_labels(axes) -> None:
    """为多面板图加入期刊常用的 (a)、(b)… 标签。"""
    array = np.asarray(axes, dtype=object).reshape(-1)
    for index, axis in enumerate(array):
        axis.text(
            -0.12, 1.04, f"({chr(ord('a') + index)})",
            transform=axis.transAxes, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left",
        )


def save_figure(fig, base_path: str | Path) -> None:
    """每张论文图同时保存 PNG 和矢量 PDF。"""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")


def rho_rows(report: Mapping[str, object]) -> list[tuple[float, Mapping[str, object]]]:
    output = []
    for key, value in dict(report.get("per_rho", {})).items():
        if not isinstance(value, Mapping):
            continue
        try:
            rho = float(str(key).replace("rho_", ""))
        except ValueError:
            continue
        output.append((rho, value))
    return sorted(output)


def aggregate_record_metric(report: Mapping[str, object], metric: str) -> dict[float, tuple[float, float, float, int]]:
    groups: dict[float, list[float]] = {}
    for row in report.get("records", []):
        if not isinstance(row, Mapping):
            continue
        rho, value = finite(row.get("rho")), finite(row.get(metric))
        if rho is not None and value is not None:
            groups.setdefault(rho, []).append(value)
    return {rho: mean_ci(values) for rho, values in sorted(groups.items())}

"""Plot runtime preference traces for closed-loop case studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for _path in (REPO_ROOT,):
    _path_str = str(_path)
    if _path.exists() and _path_str not in sys.path:
        sys.path.insert(0, _path_str)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    matplotlib = None
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc

from research_v1.execution.interaction.schema import SCENE_GATE_ORDER
from research_v1.execution.runtime.trace_export import build_runtime_trace_row
from research_v1.scene_data.schema import style_axis_names_for_scene

SCENE_COLORS = {
    "straight_free_drive": "#f3c969",
    "straight_car_follow": "#7cb7ff",
    "straight_lane_change": "#89d6a3",
}
AXIS_COLORS = ("#3366cc", "#dc3912", "#109618")


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, object]]:
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_runtime_rows(trace_path: Path) -> List[Mapping[str, object]]:
    rows = list(_iter_jsonl(trace_path))
    if rows:
        return rows
    return []


def _load_rows_from_step_trace(step_trace_path: Path) -> List[Mapping[str, object]]:
    rows: List[Mapping[str, object]] = []
    for payload in _iter_jsonl(step_trace_path):
        runtime_debug = payload.get("runtime_preference")
        if not isinstance(runtime_debug, Mapping):
            continue
        rows.append(
            build_runtime_trace_row(
                step_index=int(payload.get("step_index", 0)),
                iteration_index=int(payload.get("iteration_index", 0)),
                time_us=int(payload.get("time_us", 0)),
                debug=runtime_debug,
            )
        )
    return rows


def _contiguous_scene_segments(scene_buckets: List[str]) -> List[tuple[int, int, str]]:
    if not scene_buckets:
        return []
    segments: List[tuple[int, int, str]] = []
    start = 0
    current = scene_buckets[0]
    for index, scene_bucket in enumerate(scene_buckets[1:], start=1):
        if scene_bucket != current:
            segments.append((start, index - 1, current))
            start = index
            current = scene_bucket
    segments.append((start, len(scene_buckets) - 1, current))
    return segments


def _shade_scene_segments(ax: plt.Axes, steps: List[int], scene_buckets: List[str]) -> None:
    if not steps:
        return
    for start, end, scene_bucket in _contiguous_scene_segments(scene_buckets):
        color = SCENE_COLORS.get(scene_bucket)
        if color is None:
            continue
        left = steps[start]
        right = steps[end]
        if end + 1 < len(steps):
            right = steps[end + 1]
        ax.axvspan(left, right, color=color, alpha=0.08, linewidth=0.0)


def _plot_scene_group(
    ax: plt.Axes,
    *,
    steps: List[int],
    rows: List[Mapping[str, object]],
    scene_bucket: str,
    scene_buckets: List[str],
) -> None:
    _shade_scene_segments(ax, steps, scene_buckets)
    axis_names = list(style_axis_names_for_scene(scene_bucket))
    for axis_index, axis_name in enumerate(axis_names):
        color = AXIS_COLORS[axis_index % len(AXIS_COLORS)]
        target_values = []
        safe_values = []
        effective_values = []
        for row in rows:
            global_axes = row.get("global_axes", {})
            if not isinstance(global_axes, Mapping):
                global_axes = {}
            axis_payload = global_axes.get(axis_name, {})
            if not isinstance(axis_payload, Mapping):
                axis_payload = {}
            target_values.append(float(axis_payload.get("p_target", 0.0)))
            safe_values.append(float(axis_payload.get("p_safe", 0.0)))
            effective_values.append(float(axis_payload.get("p_eff", 0.0)))
        ax.plot(steps, target_values, linestyle="--", color=color, alpha=0.75, label=f"{axis_name} target")
        ax.plot(steps, safe_values, linestyle=":", color=color, alpha=0.9, label=f"{axis_name} safe")
        ax.plot(steps, effective_values, linestyle="-", color=color, linewidth=1.8, label=f"{axis_name} eff")
    ax.set_ylabel("pref")
    ax.set_title(scene_bucket)
    ax.grid(True, linestyle=":", alpha=0.25)


def _plot_gate_panel(
    ax: plt.Axes,
    *,
    steps: List[int],
    rows: List[Mapping[str, object]],
    scene_buckets: List[str],
) -> None:
    _shade_scene_segments(ax, steps, scene_buckets)
    for scene_bucket in SCENE_GATE_ORDER:
        values = []
        for row in rows:
            scene_gates = row.get("scene_gates", {})
            if not isinstance(scene_gates, Mapping):
                scene_gates = {}
            values.append(float(scene_gates.get(scene_bucket, 0.0)))
        ax.plot(steps, values, linewidth=1.8, label=f"gate:{scene_bucket}")

    intensity_values = []
    for row in rows:
        command = row.get("command", {})
        if not isinstance(command, Mapping):
            command = {}
        intensity_values.append(float(command.get("style_intensity", 0.0)))
    ax.plot(steps, intensity_values, color="black", linestyle="--", linewidth=1.5, label="style_intensity")
    ax.set_ylabel("gate")
    ax.set_xlabel("step")
    ax.grid(True, linestyle=":", alpha=0.25)


def _plot_trace(rows: List[Mapping[str, object]], output_path: Path, *, title: str) -> None:
    steps = [int(row.get("step_index", index)) for index, row in enumerate(rows)]
    scene_buckets = [
        str(dict(row.get("scene", {})).get("scene_bucket", "none"))
        if isinstance(row.get("scene"), Mapping)
        else "none"
        for row in rows
    ]

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    fig.suptitle(title)
    for ax, scene_bucket in zip(axes[:3], SCENE_GATE_ORDER):
        _plot_scene_group(ax, steps=steps, rows=rows, scene_bucket=scene_bucket, scene_buckets=scene_buckets)
    _plot_gate_panel(axes[3], steps=steps, rows=rows, scene_buckets=scene_buckets)

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        cur_handles, cur_labels = ax.get_legend_handles_labels()
        handles.extend(cur_handles)
        labels.extend(cur_labels)
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _resolve_scenario_dirs(args: argparse.Namespace) -> List[Path]:
    if args.scenario_dir:
        return [Path(args.scenario_dir)]
    if args.raw_step_root:
        return sorted(path for path in Path(args.raw_step_root).glob("scenario_*") if path.is_dir())
    raise ValueError("Either --scenario_dir or --raw_step_root must be provided.")


def _load_rows_for_scenario(scenario_dir: Path) -> List[Mapping[str, object]]:
    trace_path = scenario_dir / "runtime_preference_trace.jsonl"
    if trace_path.exists():
        rows = _load_runtime_rows(trace_path)
        if rows:
            return rows
    step_trace_path = scenario_dir / "step_trace.jsonl"
    if step_trace_path.exists():
        return _load_rows_from_step_trace(step_trace_path)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot runtime preference case-study traces.")
    parser.add_argument("--scenario_dir", default=None, help="Path to one raw scenario export directory.")
    parser.add_argument("--raw_step_root", default=None, help="Root containing scenario_* runtime export folders.")
    parser.add_argument("--output_name", default="runtime_preference_case.png")
    args = parser.parse_args()
    if plt is None:
        raise SystemExit(
            "matplotlib is required for runtime trace plotting. "
            f"Import failed with: {_MATPLOTLIB_IMPORT_ERROR}"
        )

    scenario_dirs = _resolve_scenario_dirs(args)
    plotted = 0
    for scenario_dir in scenario_dirs:
        rows = _load_rows_for_scenario(scenario_dir)
        if not rows:
            continue
        first_command = rows[0].get("command", {})
        if not isinstance(first_command, Mapping):
            first_command = {}
        title = (
            f"{scenario_dir.name} | "
            f"style={first_command.get('style_label', 'normal')} | "
            f"intensity={float(first_command.get('style_intensity', 0.0)):.2f}"
        )
        _plot_trace(rows, scenario_dir / args.output_name, title=title)
        plotted += 1
        print(f"[RuntimeTracePlot] saved {scenario_dir / args.output_name}")
    print(f"[RuntimeTracePlot] plotted={plotted}")


if __name__ == "__main__":
    main()

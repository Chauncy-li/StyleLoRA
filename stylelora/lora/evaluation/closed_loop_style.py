"""Aggregate rho-conditioned style proxies from NuPlan closed-loop raw exports.

The official NuPlan metrics answer whether a planner is safe and completes the
task.  They do not answer whether ``rho`` actually changes driving style.  This
module reads the step-level ``npz`` files exported by
``baseline.simulation.planner.DiffusionPlanner`` and measures the planned ego
trajectory under the recurrent closed-loop state seen at every simulation step.

Only samples present at every rho are used for rho-to-rho trend comparisons.
This prevents an early termination at one rho from silently changing the
evaluation population.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np


METRIC_DIRECTIONS = {
    "current_ego_speed": "increasing",
    "current_min_front_gap": "decreasing",
    "current_time_headway": "decreasing",
    "planned_mean_speed": "increasing",
    "planned_p90_speed": "increasing",
    "planned_accel_p90": "increasing",
    "planned_brake_p90": "increasing",
    "planned_abs_jerk_p90": "increasing",
    "planned_min_front_gap": "decreasing",
    "planned_min_time_headway": "decreasing",
}


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else None


def _step_key(path: Path, raw_root: Path, iteration_index: int, time_us: int) -> str:
    # NuPlan may construct one planner per scenario.  In that case every planner
    # starts its local directory counter at ``scenario_000001``.  Simulation
    # timestamps remain scenario-specific and stable across rho, so they are the
    # correct paired key (the relative parent alone is not).
    relative_parent = path.relative_to(raw_root).parent.as_posix()
    return f"{relative_parent}:time_{time_us}:iteration_{iteration_index:06d}"


def _step_metrics(
    path: Path,
    raw_root: Path,
    dt: float,
    allowed_tokens: set[str] | None = None,
) -> tuple[str, dict[str, float | bool]] | None:
    """Extract generic physical style proxies from one exported planner step."""
    with np.load(path, allow_pickle=False) as payload:
        scenario_token = (
            str(np.asarray(payload["scenario_token"]).reshape(-1)[0])
            if "scenario_token" in payload else ""
        )
        if allowed_tokens is not None and scenario_token not in allowed_tokens:
            return None
        if "generated_ego_future" not in payload:
            return None
        ego_future = np.asarray(payload["generated_ego_future"], dtype=np.float64)
        if ego_future.ndim != 2 or ego_future.shape[0] < 2 or ego_future.shape[1] < 2:
            return None
        current_payload = payload["ego_current_state"] if "ego_current_state" in payload else np.zeros(2)
        current = np.asarray(current_payload, dtype=np.float64).reshape(-1)
        current_xy = current[:2] if current.size >= 2 else np.zeros(2, dtype=np.float64)
        full_xy = np.concatenate((current_xy[None], ego_future[:, :2]), axis=0)
        speed = np.linalg.norm(np.diff(full_xy, axis=0), axis=-1) / dt
        acceleration = np.diff(speed) / dt
        jerk = np.diff(acceleration) / dt

        positive_accel = acceleration[acceleration > 0]
        braking = -acceleration[acceleration < 0]
        metrics: dict[str, float | bool] = {
            "current_ego_speed": float(np.linalg.norm(current[4:6])) if current.size >= 6 else float("nan"),
            "current_longitudinal_accel": float(current[6]) if current.size >= 7 else float("nan"),
            "planned_mean_speed": float(np.mean(speed)),
            "planned_p90_speed": float(np.quantile(speed, 0.9)),
            "planned_accel_p90": float(np.quantile(positive_accel, 0.9)) if positive_accel.size else 0.0,
            "planned_brake_p90": float(np.quantile(braking, 0.9)) if braking.size else 0.0,
            "planned_abs_jerk_p90": float(np.quantile(np.abs(jerk), 0.9)) if jerk.size else 0.0,
            "planned_lateral_displacement": float(np.max(np.abs(ego_future[:, 1] - current_xy[1]))),
        }
        # 门控关闭的旧导出没有这些字段；用 NaN 保持向后兼容。
        for name in ("requested_rho", "effective_rho", "gate_cap_low", "gate_cap_high"):
            metrics[name] = (
                float(np.asarray(payload[name]).reshape(-1)[0]) if name in payload else float("nan")
            )
        for name in (
            "gate_proposed_rho",
            "initial_requested_rho",
            "accepted_rho",
            "candidate_attempts",
            "candidate_accepted",
            "baseline_fallback",
            "baseline_hard_valid",
        ):
            metrics[name] = (
                float(np.asarray(payload[name]).reshape(-1)[0])
                if name in payload else float("nan")
            )
        for name in (
            "trajectory_repair_applied",
            "trajectory_repair_longitudinal_scale",
            "trajectory_repair_lateral_scale",
            "trajectory_repair_baseline_fallback",
            "trajectory_repair_candidate_attempts",
            "trajectory_repair_baseline_hard_valid",
            "trajectory_repair_collision_triggered",
            "trajectory_repair_drivable_triggered",
            "trajectory_repair_selected_alpha",
            "trajectory_repair_min_clearance_m",
            "trajectory_repair_style_offroad_fraction",
            "trajectory_repair_selected_offroad_fraction",
            "trajectory_repair_drivable_check_time_ms",
            "trajectory_repair_model_inference_time_ms",
            "trajectory_repair_time_ms",
            "trajectory_repair_total_time_ms",
        ):
            metrics[name] = (
                float(np.asarray(payload[name]).reshape(-1)[0])
                if name in payload else float("nan")
            )

        current_gaps: list[float] = []
        if "neighbor_current_state" in payload:
            neighbor_current = np.asarray(payload["neighbor_current_state"], dtype=np.float64)
        elif "neighbor_agents_past" in payload:
            neighbor_past = np.asarray(payload["neighbor_agents_past"], dtype=np.float64)
            neighbor_current = (
                neighbor_past[:, -1]
                if neighbor_past.ndim == 3 and neighbor_past.shape[1] else np.zeros((0, 2))
            )
        else:
            neighbor_current = np.zeros((0, 2), dtype=np.float64)
        if neighbor_current.ndim == 2 and neighbor_current.shape[-1] >= 2:
            neighbor_current = neighbor_current[:, :2]
            relative_current = neighbor_current - current_xy
            valid_current = (
                (relative_current[:, 0] > 0)
                & (np.abs(relative_current[:, 1]) < 4.0)
                & (np.abs(neighbor_current).sum(axis=-1) > 1e-4)
            )
            if np.any(valid_current):
                current_gaps.append(float(np.min(relative_current[valid_current, 0])))
        current_gap = min(current_gaps) if current_gaps else float("nan")
        metrics["has_current_lead"] = bool(current_gaps)
        metrics["current_min_front_gap"] = current_gap
        current_speed = _finite(metrics["current_ego_speed"])
        metrics["current_time_headway"] = (
            current_gap / max(float(current_speed), 0.1)
            if current_gaps and current_speed is not None else float("nan")
        )

        gaps: list[float] = []
        headways: list[float] = []
        if "generated_neighbor_future" in payload:
            neighbors = np.asarray(payload["generated_neighbor_future"], dtype=np.float64)
            if neighbors.ndim == 3 and neighbors.shape[-1] >= 2:
                horizon = min(ego_future.shape[0], neighbors.shape[1])
                for step in range(horizon):
                    neighbor_xy = neighbors[:, step, :2]
                    relative = neighbor_xy - ego_future[step, :2]
                    valid = (
                        (relative[:, 0] > 0)
                        & (np.abs(relative[:, 1]) < 4.0)
                        & (np.abs(neighbor_xy).sum(axis=-1) > 1e-4)
                    )
                    if np.any(valid):
                        gap = float(np.min(relative[valid, 0]))
                        gaps.append(gap)
                        headways.append(gap / max(float(speed[min(step, speed.size - 1)]), 0.1))
        metrics["has_planned_lead"] = bool(gaps)
        metrics["planned_min_front_gap"] = float(np.quantile(gaps, 0.1)) if gaps else float("nan")
        metrics["planned_min_time_headway"] = float(np.quantile(headways, 0.1)) if headways else float("nan")
        iteration_payload = payload["iteration_index"] if "iteration_index" in payload else np.asarray(-1)
        time_payload = payload["time_us"] if "time_us" in payload else np.asarray(-1)
        iteration = int(np.asarray(iteration_payload).reshape(-1)[0])
        time_us = int(np.asarray(time_payload).reshape(-1)[0])
    key = _step_key(path, raw_root, iteration, time_us)
    if scenario_token:
        key = f"token_{scenario_token}:{key}"
    return key, metrics


def _load_run(
    raw_root: Path,
    dt: float,
    allowed_tokens: set[str] | None = None,
) -> dict[str, dict[str, float | bool]]:
    rows: dict[str, dict[str, float | bool]] = {}
    for path in sorted(raw_root.rglob("step_*.npz")):
        extracted = _step_metrics(path, raw_root, dt, allowed_tokens)
        if extracted is not None:
            key, metrics = extracted
            if key in rows:
                raise ValueError(f"闭环 raw export 出现重复 step key: {key}")
            rows[key] = metrics
    return rows


def _distribution(rows: list[Mapping[str, object]], metric: str) -> dict[str, float | int | None]:
    values = [_finite(row.get(metric)) for row in rows]
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not finite.size:
        return {"count": 0, "mean": None, "std": None, "p10": None, "median": None, "p90": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p10": _quantile(finite, 0.1),
        "median": _quantile(finite, 0.5),
        "p90": _quantile(finite, 0.9),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    xr, yr = _rankdata(np.asarray(x, dtype=np.float64)), _rankdata(np.asarray(y, dtype=np.float64))
    xr, yr = xr - xr.mean(), yr - yr.mean()
    denominator = float(np.linalg.norm(xr) * np.linalg.norm(yr))
    return float(np.dot(xr, yr) / denominator) if denominator > 0 else None


def analyze_closed_loop_style_runs(
    raw_roots: Mapping[float, str | Path],
    *,
    dt: float = 0.1,
    allowed_tokens: set[str] | None = None,
) -> dict[str, object]:
    """Compare closed-loop planned style proxies over a fixed rho grid."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    runs = {
        float(rho): _load_run(Path(path), dt, allowed_tokens)
        for rho, path in raw_roots.items()
    }
    if not runs:
        return {"available": False, "failure": "没有 rho run"}
    empty = [rho for rho, rows in runs.items() if not rows]
    if empty:
        return {"available": False, "failure": f"以下 rho 没有 raw step npz: {empty}"}

    common_keys = set.intersection(*(set(rows) for rows in runs.values()))
    common_planned_lead_keys = {
        key for key in common_keys
        if all(bool(rows[key].get("has_planned_lead")) for rows in runs.values())
    }
    common_current_lead_keys = {
        key for key in common_keys
        if all(bool(rows[key].get("has_current_lead")) for rows in runs.values())
    }
    metric_names = tuple(METRIC_DIRECTIONS) + (
        "current_longitudinal_accel", "planned_lateral_displacement",
        "requested_rho", "effective_rho", "gate_cap_low", "gate_cap_high",
        "gate_proposed_rho", "initial_requested_rho", "accepted_rho", "candidate_attempts",
        "candidate_accepted", "baseline_fallback", "baseline_hard_valid",
        "trajectory_repair_applied", "trajectory_repair_longitudinal_scale",
        "trajectory_repair_lateral_scale", "trajectory_repair_baseline_fallback",
        "trajectory_repair_candidate_attempts", "trajectory_repair_baseline_hard_valid",
        "trajectory_repair_collision_triggered", "trajectory_repair_drivable_triggered",
        "trajectory_repair_selected_alpha", "trajectory_repair_min_clearance_m",
        "trajectory_repair_style_offroad_fraction",
        "trajectory_repair_selected_offroad_fraction",
        "trajectory_repair_drivable_check_time_ms",
        "trajectory_repair_model_inference_time_ms",
        "trajectory_repair_time_ms", "trajectory_repair_total_time_ms",
    )
    per_rho: dict[str, object] = {}
    rho_means: dict[str, list[tuple[float, float]]] = {name: [] for name in metric_names}
    for rho in sorted(runs):
        all_rows = list(runs[rho].values())
        common_rows = [runs[rho][key] for key in sorted(common_keys)]
        planned_lead_rows = [runs[rho][key] for key in sorted(common_planned_lead_keys)]
        current_lead_rows = [runs[rho][key] for key in sorted(common_current_lead_keys)]
        summaries = {}
        for metric in metric_names:
            if metric in {"planned_min_front_gap", "planned_min_time_headway"}:
                population = planned_lead_rows
            elif metric in {"current_min_front_gap", "current_time_headway"}:
                population = current_lead_rows
            else:
                population = common_rows
            summaries[metric] = _distribution(population, metric)
            mean = summaries[metric]["mean"]
            if mean is not None:
                rho_means[metric].append((rho, float(mean)))
        per_rho[f"rho_{rho:+.2f}"] = {
            "exported_steps": len(all_rows),
            "common_steps": len(common_rows),
            "common_current_lead_steps": len(current_lead_rows),
            "common_planned_lead_steps": len(planned_lead_rows),
            "current_lead_rate_all_steps": float(np.mean([
                bool(row.get("has_current_lead")) for row in all_rows
            ])),
            "planned_lead_rate_all_steps": float(np.mean([
                bool(row.get("has_planned_lead")) for row in all_rows
            ])),
            "trajectory_repair_summary": {
                "trigger_rate": summaries[
                    "trajectory_repair_applied"
                ]["mean"],
                "collision_trigger_rate": summaries[
                    "trajectory_repair_collision_triggered"
                ]["mean"],
                "drivable_trigger_rate": summaries[
                    "trajectory_repair_drivable_triggered"
                ]["mean"],
                "baseline_fallback_rate": summaries[
                    "trajectory_repair_baseline_fallback"
                ]["mean"],
                "selected_alpha_mean": summaries[
                    "trajectory_repair_selected_alpha"
                ]["mean"],
                "selected_alpha_distribution": summaries[
                    "trajectory_repair_selected_alpha"
                ],
                "style_offroad_fraction": summaries[
                    "trajectory_repair_style_offroad_fraction"
                ],
                "selected_offroad_fraction": summaries[
                    "trajectory_repair_selected_offroad_fraction"
                ],
                "drivable_check_time_ms": summaries[
                    "trajectory_repair_drivable_check_time_ms"
                ],
                "model_inference_time_ms": summaries[
                    "trajectory_repair_model_inference_time_ms"
                ],
                "collision_repair_time_ms": summaries[
                    "trajectory_repair_time_ms"
                ],
                "total_time_ms": summaries[
                    "trajectory_repair_total_time_ms"
                ],
            },
            "metrics_on_common_steps": summaries,
        }

    trends: dict[str, object] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        points = rho_means[metric]
        rho_values = [point[0] for point in points]
        means = [point[1] for point in points]
        correlation = _spearman(rho_values, means)
        correct = None if correlation is None else (
            correlation >= 0 if direction == "increasing" else correlation <= 0
        )
        trends[metric] = {
            "expected": direction,
            "rho_spearman": correlation,
            "direction_correct": correct,
            "rho_values": rho_values,
            "mean_sequence": means,
            "endpoint_delta": means[-1] - means[0] if len(means) >= 2 else None,
        }

    core = ("current_ego_speed", "current_min_front_gap", "current_time_headway")
    core_available = [trends[name]["direction_correct"] for name in core if trends[name]["direction_correct"] is not None]
    return {
        "available": True,
        "dt": dt,
        "common_steps": len(common_keys),
        "common_current_lead_steps": len(common_current_lead_keys),
        "common_planned_lead_steps": len(common_planned_lead_keys),
        "per_rho": per_rho,
        "trends": trends,
        "core_direction_correct_count": int(sum(bool(value) for value in core_available)),
        "core_direction_available_count": len(core_available),
        "note": "这些是闭环状态下每一步重新规划的轨迹风格代理；安全和任务完成度以 NuPlan 官方指标为准。",
    }

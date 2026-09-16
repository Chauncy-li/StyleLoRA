"""Run fixed-scenario NuPlan closed-loop rho jobs and summarize safety/style."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from stylelora.closed_loop_constants import ROUTE_FAILURE_TOKENS
from stylelora.lora.evaluation.closed_loop_style import analyze_closed_loop_style_runs
from stylelora.lora.evaluation.reports import write_json


# 旧版正式实验默认使用同一组 45 个场景；论文扩展实验可通过
# --expected-scenario-count 可显式改为论文候选场景数，同时保留旧命令兼容性。
FORMAL_SCENARIO_COUNT = 45
FORMAL_RHOS = (-1.0, -0.5, 0.0, 0.5, 1.0)
# 仅报告 NuPlan challenge 原生输出的安全与合规指标。
OFFICIAL_SAFETY_METRICS = (
    "no_ego_at_fault_collisions",
    "time_to_collision_within_bound",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "speed_limit_compliance",
    "ego_is_comfortable",
)


def _json_number(value: object) -> float | int | None:
    """把 pandas/numpy 标量转换为可安全写入 JSON 的有限数值。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _prepare_formal_tokens(
    tokens: list[str], expected_count: int = FORMAL_SCENARIO_COUNT
) -> tuple[list[str], list[str]]:
    """删除已确认的地图路由失败 token，并核对正式评测场景数量。"""
    if expected_count <= 0:
        raise ValueError("expected_count 必须为正整数")
    removed = [token for token in tokens if token in ROUTE_FAILURE_TOKENS]
    filtered = [token for token in tokens if token not in ROUTE_FAILURE_TOKENS]
    if len(filtered) != expected_count:
        raise ValueError(
            "正式闭环评测在删除固定路由失败 token 后必须恰好包含 "
            f"{expected_count} 个场景，当前为 {len(filtered)} 个"
        )
    return filtered, removed


def _challenge_result_dir(result_dir: Path, challenge: str) -> Path:
    """让指标路径包含 challenge 名称，满足 NuPlan 官方 aggregator 的筛选规则。"""
    return result_dir / challenge


def _metric_summary(result_dir: Path, minimums: dict[str, float]) -> dict:
    from baseline.simulation.simulation_metrics import find_metric_column, load_metrics_dataframe

    frame = load_metrics_dataframe(str(result_dir))
    if frame is None or frame.empty:
        return {"closed_loop_pass": False, "failure": "NuPlan 未产生可读取的逐场景指标"}
    numeric = {
        str(column): float(frame[column].mean())
        for column in frame.select_dtypes(include="number").columns
    }
    resolved = {name: find_metric_column(tuple(numeric), (name,)) for name in minimums}
    missing = [name for name, column in resolved.items() if column is None]
    required_values = {
        name: numeric[column] for name, column in resolved.items() if column is not None
    }
    failed = {
        name: required_values[name]
        for name, threshold in minimums.items()
        if name in required_values and required_values[name] < threshold
    }
    return {
        "closed_loop_pass": not missing and not failed,
        "scenario_count": int(len(frame)),
        "metric_means": numeric,
        "required_metric_values": required_values,
        "missing_required_metrics": missing,
        "below_threshold": failed,
    }


def _runner_summary(result_dir: Path, expected_tokens: list[str] | None = None) -> dict:
    """读取 NuPlan 的逐场景运行状态，并核对正式评测的固定 token 集合。"""
    import pandas as pd

    path = result_dir / "runner_report.parquet"
    if not path.exists():
        return {"all_succeeded": False, "failure": "缺少 runner_report.parquet"}
    frame = pd.read_parquet(path)
    if frame.empty or "succeeded" not in frame.columns:
        return {"all_succeeded": False, "failure": "runner_report.parquet 为空或缺少 succeeded 列"}
    succeeded = frame["succeeded"].fillna(False).astype(bool)
    failed_rows = frame.loc[~succeeded]
    errors = []
    if "error_message" in failed_rows.columns:
        errors = [str(value) for value in failed_rows["error_message"].dropna().head(3)]
    failed_count = int((~succeeded).sum())
    actual_tokens = [str(value) for value in frame["scenario_name"]] if "scenario_name" in frame else []
    successful_tokens = (
        [str(value) for value in frame.loc[succeeded, "scenario_name"]]
        if "scenario_name" in frame else []
    )
    failed_tokens = (
        [str(value) for value in failed_rows["scenario_name"]]
        if "scenario_name" in failed_rows else []
    )
    failed_scenarios = []
    for _, row in failed_rows.iterrows():
        failed_scenarios.append({
            "token": str(row.get("scenario_name", "")),
            "error_message": str(row.get("error_message", "")),
        })
    duplicate_tokens = sorted({token for token in actual_tokens if actual_tokens.count(token) > 1})
    expected = set(expected_tokens or [])
    actual = set(actual_tokens)
    missing_tokens = sorted(expected - actual)
    unexpected_tokens = sorted(actual - expected) if expected_tokens is not None else []
    token_set_matches = expected_tokens is None or (
        not duplicate_tokens and not missing_tokens and not unexpected_tokens
    )
    return {
        "all_succeeded": failed_count == 0 and token_set_matches,
        "runner_count": int(len(frame)),
        "successful_count": int(succeeded.sum()),
        "failed_count": failed_count,
        "scenario_tokens": actual_tokens,
        "successful_tokens": successful_tokens,
        "failed_tokens": failed_tokens,
        "failed_scenarios": failed_scenarios,
        "token_set_matches": token_set_matches,
        "missing_tokens": missing_tokens,
        "unexpected_tokens": unexpected_tokens,
        "duplicate_tokens": duplicate_tokens,
        "error_samples": errors,
    }


def _official_aggregator_summary(
    result_dir: Path,
    expected_tokens: list[str],
    allow_partial: bool = False,
) -> dict:
    """读取 NuPlan 官方 challenge aggregator，并保留逐场景安全指标用于配对比较。"""
    from baseline.simulation.simulation_metrics import find_metric_column, read_parquet_frame

    files = sorted(result_dir.glob("aggregator_metric/*.parquet"))
    if not files:
        return {"available": False, "failure": "缺少 NuPlan 官方 aggregator parquet"}
    frame = read_parquet_frame(str(files[-1]))
    if frame is None or frame.empty or "scenario" not in frame.columns or "score" not in frame.columns:
        return {"available": False, "failure": "NuPlan 官方 aggregator parquet 为空或字段不完整"}

    # 官方文件最后包含场景类型汇总和 final_score；只有 num_scenarios 为空的行才是逐场景结果。
    if "num_scenarios" not in frame.columns:
        return {"available": False, "failure": "NuPlan 官方 aggregator 缺少 num_scenarios 列"}
    scenario_frame = frame.loc[frame["num_scenarios"].isna()].copy()
    scenario_frame["scenario"] = scenario_frame["scenario"].astype(str)
    actual_tokens = scenario_frame["scenario"].tolist()
    missing_tokens = sorted(set(expected_tokens) - set(actual_tokens))
    unexpected_tokens = sorted(set(actual_tokens) - set(expected_tokens))
    duplicate_tokens = sorted({token for token in actual_tokens if actual_tokens.count(token) > 1})
    invalid_set = bool(unexpected_tokens or duplicate_tokens)
    incomplete_set = bool(missing_tokens or len(actual_tokens) != len(expected_tokens))
    if invalid_set or (incomplete_set and not allow_partial):
        return {
            "available": False,
            "failure": (
                "NuPlan 官方 aggregator 的逐场景集合与固定 "
                f"{len(expected_tokens)} 场景不一致"
            ),
            "missing_tokens": missing_tokens,
            "unexpected_tokens": unexpected_tokens,
            "duplicate_tokens": duplicate_tokens,
        }

    final_rows = frame.loc[frame["scenario"].astype(str).str.lower().eq("final_score")]
    if final_rows.empty and not allow_partial:
        return {"available": False, "failure": "NuPlan 官方 aggregator 缺少 final_score 行"}
    final_row = final_rows.iloc[-1] if not final_rows.empty else None
    columns = tuple(str(column) for column in frame.columns)
    safety_columns = {
        name: find_metric_column(columns, (name,)) for name in OFFICIAL_SAFETY_METRICS
    }
    missing_metrics = [name for name, column in safety_columns.items() if column is None]
    if missing_metrics:
        return {
            "available": False,
            "failure": "NuPlan 官方 aggregator 缺少安全指标",
            "missing_safety_metrics": missing_metrics,
        }

    required_columns = ["score", *safety_columns.values()]
    if scenario_frame.empty:
        return {"available": False, "failure": "NuPlan 官方 aggregator 没有成功场景行"}
    valid_rows = scenario_frame.apply(
        lambda row: all(_json_number(row[column]) is not None for column in required_columns),
        axis=1,
    )
    invalid_metric_tokens = scenario_frame.loc[~valid_rows, "scenario"].astype(str).tolist()
    if invalid_metric_tokens and not allow_partial:
        return {"available": False, "failure": "NuPlan 官方 aggregator 的逐场景指标包含空值或非有限数值"}
    if allow_partial:
        scenario_frame = scenario_frame.loc[valid_rows].copy()
        actual_tokens = scenario_frame["scenario"].astype(str).tolist()
        missing_tokens = sorted(set(expected_tokens) - set(actual_tokens))
        if scenario_frame.empty:
            return {"available": False, "failure": "NuPlan 官方 aggregator 没有指标完整的成功场景"}
    if not allow_partial and final_row is not None and any(
        _json_number(final_row[column]) is None for column in required_columns
    ):
        return {"available": False, "failure": "NuPlan 官方 aggregator 的 final_score 指标包含空值或非有限数值"}

    per_scenario = []
    for _, row in scenario_frame.sort_values("scenario").iterrows():
        per_scenario.append({
            "token": str(row["scenario"]),
            "log_name": str(row.get("log_name", "")),
            "challenge_score": _json_number(row["score"]),
            "safety_metrics": {
                name: _json_number(row[column]) for name, column in safety_columns.items()
            },
        })
    challenge_score = (
        float(sum(float(row["challenge_score"]) for row in per_scenario) / len(per_scenario))
        if allow_partial else _json_number(final_row["score"])
    )
    safety_metric_means = (
        {
            name: float(sum(
                float(row["safety_metrics"][name]) for row in per_scenario
            ) / len(per_scenario))
            for name in OFFICIAL_SAFETY_METRICS
        }
        if allow_partial else {
            name: _json_number(final_row[column]) for name, column in safety_columns.items()
        }
    )
    return {
        "available": True,
        "file": str(files[-1]),
        "scenario_count": len(per_scenario),
        "candidate_scenario_count": len(expected_tokens),
        "partial": bool(missing_tokens),
        "missing_tokens": missing_tokens,
        "unexpected_tokens": unexpected_tokens,
        "duplicate_tokens": duplicate_tokens,
        "invalid_metric_tokens": invalid_metric_tokens,
        "challenge_score": challenge_score,
        "safety_metric_means": safety_metric_means,
        "per_scenario": per_scenario,
    }


def _resolve_normalization_file(args_file: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    payload = json.loads(args_file.read_text(encoding="utf-8"))
    value = payload.get("normalization_file_path", payload.get("normalization_file"))
    if not value:
        raise ValueError("args.json 中没有 normalization_file_path，请显式传 --normalization-file")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = args_file.parent / path
    return path.resolve()


def _load_tokens(path: str | None) -> list[str]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("scenario_tokens", payload.get("tokens"))
    if not isinstance(payload, list) or not payload or not all(isinstance(item, str) and item for item in payload):
        raise ValueError("--scenario-tokens-file 必须是非空 JSON 字符串列表，或包含 tokens/scenario_tokens 的对象")
    if len(payload) != len(set(payload)):
        raise ValueError("--scenario-tokens-file 包含重复 token")
    return payload


def _rho_dir(root: Path, rho: float) -> Path:
    name = f"rho_{rho:+.2f}".replace("+", "plus").replace("-", "minus")
    return root / name


def _metric_deltas(records: list[dict]) -> dict[str, dict[str, float]]:
    zero = next((row for row in records if abs(float(row["rho"])) < 1e-12), None)
    if zero is None or not zero.get("metric_means"):
        return {}
    baseline = zero["metric_means"]
    output = {}
    for row in records:
        means = row.get("metric_means", {})
        output[f"rho_{float(row['rho']):+.2f}"] = {
            name: float(value) - float(baseline[name])
            for name, value in means.items()
            if name in baseline
        }
    return output


def _rho_zero_baseline(records: list[dict]) -> dict:
    """显式提取 rho=0 的官方基线结果，避免只在差值中隐式出现。"""
    zero = next((row for row in records if abs(float(row["rho"])) < 1e-12), None)
    if zero is None:
        raise ValueError("正式闭环结果缺少 rho=0 基线")
    official = zero.get("official_aggregator", {})
    if not official.get("available", False):
        raise ValueError("rho=0 缺少可用的 NuPlan 官方 aggregator 结果")
    return {
        "rho": 0.0,
        "scenario_count": official["scenario_count"],
        "challenge_score": official["challenge_score"],
        "safety_metric_means": official["safety_metric_means"],
        "per_scenario": official["per_scenario"],
    }


def _paired_official_differences(records: list[dict]) -> dict[str, object]:
    """以 rho=0 为同场景基线，计算 challenge 分数和官方安全指标的逐场景配对差异。"""
    baseline = _rho_zero_baseline(records)
    baseline_rows = {str(row["token"]): row for row in baseline["per_scenario"]}
    paired: dict[str, object] = {}
    for record in records:
        rho = float(record["rho"])
        if abs(rho) < 1e-12:
            continue
        official = record.get("official_aggregator", {})
        if not official.get("available", False):
            raise ValueError(f"rho={rho:+.2f} 缺少可用的 NuPlan 官方 aggregator 结果")
        current_rows = {str(row["token"]): row for row in official["per_scenario"]}
        if set(current_rows) != set(baseline_rows):
            raise ValueError(f"rho={rho:+.2f} 与 rho=0 的逐场景集合不一致")

        scenario_differences = []
        for token in sorted(baseline_rows):
            base = baseline_rows[token]
            current = current_rows[token]
            scenario_differences.append({
                "token": token,
                "log_name": current["log_name"],
                "challenge_score_delta": (
                    float(current["challenge_score"]) - float(base["challenge_score"])
                ),
                "safety_metric_deltas": {
                    name: float(current["safety_metrics"][name]) - float(base["safety_metrics"][name])
                    for name in OFFICIAL_SAFETY_METRICS
                },
            })
        paired[f"rho_{rho:+.2f}"] = {
            "paired_scenario_count": len(scenario_differences),
            "mean_challenge_score_delta": float(sum(
                row["challenge_score_delta"] for row in scenario_differences
            ) / len(scenario_differences)),
            "mean_safety_metric_deltas": {
                name: float(sum(
                    row["safety_metric_deltas"][name] for row in scenario_differences
                ) / len(scenario_differences))
                for name in OFFICIAL_SAFETY_METRICS
            },
            "per_scenario": scenario_differences,
        }
    return paired


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="在固定 NuPlan 场景上运行多 rho 闭环，并汇总官方安全指标与闭环风格代理"
    )
    parser.add_argument("--args-file", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--high-adapter", "--aggressive-adapter", dest="high_adapter", required=True)
    parser.add_argument("--low-adapter", "--conservative-adapter", dest="low_adapter", required=True)
    parser.add_argument("--normalization-file", default=None, help="默认从 args.json 的 normalization_file_path 读取")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--enable-scene-gate", action="store_true",
                        help="显式启用场景强度上限门控；默认保持现有 LoRA 闭环行为")
    parser.add_argument("--scene-gate-checkpoint", default=None)
    parser.add_argument(
        "--conditional-router-checkpoint",
        default=None,
        help="可选动态条件 LoRA 路由；省略时保持原闭环行为。",
    )
    parser.add_argument(
        "--enable-bounded-style",
        action="store_true",
        help="启用有限强度候选验证与 baseline 回退。",
    )
    parser.add_argument(
        "--bounded-candidate-ratios",
        default="1,0.75,0.5,0.25",
        help="相对请求强度的有限候选比例，必须包含 1，按从大到小验证。",
    )
    parser.add_argument("--bounded-max-mean-accel-degradation", type=float, default=0.5)
    parser.add_argument("--bounded-max-mean-jerk-degradation", type=float, default=2.0)
    parser.add_argument("--bounded-max-progress-loss-m", type=float, default=2.0)
    parser.add_argument(
        "--bounded-max-lateral-deviation-m",
        "--bounded-max-path-deviation-m",
        dest="bounded_max_lateral_deviation_m",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--enable-trajectory-repair",
        action="store_true",
        help="启用不改变 rho 的基线锚定轨迹修复；默认关闭。",
    )
    parser.add_argument("--data-root", required=True, help="NuPlan db 文件或目录，例如 splits/train_boston")
    parser.add_argument(
        "--trajectory-repair-mode",
        choices=("legacy", "collision_parallel"),
        default="legacy",
        help="Optional repair implementation; legacy preserves all previous behavior.",
    )
    parser.add_argument(
        "--collision-repair-scales",
        default="1,0.875,0.75,0.625,0.5,0.375,0.25,0.125,0",
        help="Descending baseline-to-style interpolation scales for collision_parallel mode.",
    )
    parser.add_argument("--collision-repair-horizon-s", type=float, default=2.0)
    parser.add_argument("--collision-repair-min-clearance-m", type=float, default=0.5)
    parser.add_argument(
        "--repair-check-drivable-area",
        action="store_true",
        help="Also require the short-horizon ego footprint to remain in DRIVABLE_AREA.",
    )
    parser.add_argument("--repair-drivable-horizon-s", type=float, default=2.0)
    parser.add_argument("--maps-root", required=True)
    parser.add_argument("--scenario-filter", default="boston")
    parser.add_argument(
        "--scenario-tokens-file",
        default=None,
        help="正式评测的 JSON token 列表；程序会删除已知路由失败 token 并核对固定场景数量",
    )
    parser.add_argument(
        "--expected-scenario-count",
        type=int,
        default=FORMAL_SCENARIO_COUNT,
        help="删除已知路由失败 token 后应保留的正式场景数；默认45，论文一次性采集可使用300",
    )
    parser.add_argument(
        "--scenario-validation-only",
        action="store_true",
        help="仅以 rho=0 检查候选场景可执行性，允许单场景失败并输出有效/失败 token",
    )
    parser.add_argument(
        "--minimum-valid-scenarios",
        type=int,
        default=200,
        help="场景预筛查至少需要保留的有效 token 数，默认200",
    )
    parser.add_argument(
        "--allow-partial-scenarios",
        action="store_true",
        help="允许个别场景失败并继续完成全部 rho；仅用于一次性候选场景采集，之后必须取共同成功 token 交集",
    )
    parser.add_argument("--limit-total-scenarios", type=int, default=None)
    parser.add_argument("--rhos", default="-1,-0.5,0,0.5,1")
    parser.add_argument(
        "--allow-rho-shard",
        action="store_true",
        help=(
            "允许把自定义 rho 网格拆到多个 GPU 独立采集；默认关闭，关闭时继续执行原正式五点网格检查。"
        ),
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--challenge", default="closed_loop_nonreactive_agents")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--worker", default="sequential")
    parser.add_argument("--config-root", default=str(repository / "baseline" / "config"))
    parser.add_argument("--render", action="store_true", help="默认不渲染；开启会显著增加时间和磁盘占用")
    parser.add_argument("--skip-existing", action="store_true", help="已有 metrics parquet 时跳过该 rho 仿真")
    parser.add_argument("--nuplan-overrides-json", default="[]", help="附加 Hydra override JSON 列表")
    parser.add_argument(
        "--required-minimums-json",
        default='{"no_ego_at_fault_collisions":1.0,"drivable_area_compliance":1.0,"time_to_collision_within_bound":1.0,"ego_is_comfortable":1.0}',
    )
    args = parser.parse_args()

    args_file = Path(args.args_file).expanduser().resolve()
    checkpoint = Path(args.baseline_checkpoint).expanduser().resolve()
    high_adapter = Path(args.high_adapter).expanduser().resolve()
    low_adapter = Path(args.low_adapter).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    maps_root = Path(args.maps_root).expanduser().resolve()
    config_root = Path(args.config_root).expanduser().resolve()
    normalization = _resolve_normalization_file(args_file, args.normalization_file)
    if args.enable_scene_gate and not args.scene_gate_checkpoint:
        parser.error("--enable-scene-gate 必须同时提供 --scene-gate-checkpoint")
    if args.enable_bounded_style and not args.conditional_router_checkpoint:
        parser.error("--enable-bounded-style 必须提供新版条件路由 checkpoint")
    if args.enable_bounded_style and args.enable_scene_gate:
        parser.error("新版有边界执行不再使用场景门控，请移除 --enable-scene-gate")
    if args.enable_bounded_style and args.enable_trajectory_repair:
        parser.error("--enable-bounded-style 与 --enable-trajectory-repair 不能同时启用")
    candidate_ratios = sorted(
        {float(item) for item in args.bounded_candidate_ratios.split(",") if item.strip()},
        reverse=True,
    )
    if not candidate_ratios or candidate_ratios[0] != 1.0 or any(
        value <= 0.0 or value > 1.0 for value in candidate_ratios
    ):
        parser.error("--bounded-candidate-ratios 必须包含 1，且全部位于 (0,1]")
    collision_repair_scales = tuple(
        float(item) for item in args.collision_repair_scales.split(",") if item.strip()
    )
    if (
        not collision_repair_scales
        or collision_repair_scales[0] != 1.0
        or collision_repair_scales[-1] != 0.0
        or any(value < 0.0 or value > 1.0 for value in collision_repair_scales)
        or any(
            right >= left
            for left, right in zip(collision_repair_scales, collision_repair_scales[1:])
        )
    ):
        parser.error("--collision-repair-scales must strictly decrease from 1 to 0")
    if args.collision_repair_horizon_s <= 0.0:
        parser.error("--collision-repair-horizon-s must be positive")
    if args.collision_repair_min_clearance_m < 0.0:
        parser.error("--collision-repair-min-clearance-m must be non-negative")
    if args.repair_drivable_horizon_s <= 0.0:
        parser.error("--repair-drivable-horizon-s must be positive")
    if args.scenario_validation_only and args.enable_scene_gate:
        parser.error("场景预筛查固定使用 rho=0 baseline，不应启用场景门控")
    scene_gate_checkpoint = (
        Path(args.scene_gate_checkpoint).expanduser().resolve()
        if args.scene_gate_checkpoint else None
    )
    conditional_router_checkpoint = (
        Path(args.conditional_router_checkpoint).expanduser().resolve()
        if args.conditional_router_checkpoint else None
    )
    required = (args_file, checkpoint, high_adapter, low_adapter, data_root, maps_root,
                config_root, normalization) + ((scene_gate_checkpoint,) if args.enable_scene_gate else ())
    if conditional_router_checkpoint is not None:
        required = required + (conditional_router_checkpoint,)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("闭环输入不存在:\n  - " + "\n  - ".join(missing))

    extra = json.loads(args.nuplan_overrides_json)
    minimums = {key: float(value) for key, value in json.loads(args.required_minimums_json).items()}
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise ValueError("--nuplan-overrides-json 必须是 Hydra override 字符串 JSON 列表")
    source_tokens = _load_tokens(args.scenario_tokens_file)
    # 传入 token 文件代表正式评测；无 token 文件时保留原有的少量场景冒烟测试入口。
    if args.expected_scenario_count <= 0:
        parser.error("--expected-scenario-count 必须为正整数")
    if args.minimum_valid_scenarios <= 0:
        parser.error("--minimum-valid-scenarios 必须为正整数")
    tokens, removed_route_failure_tokens = (
        _prepare_formal_tokens(source_tokens, args.expected_scenario_count)
        if source_tokens else ([], [])
    )
    rhos = [float(item.strip()) for item in args.rhos.split(",") if item.strip()]
    if not rhos or len(rhos) != len(set(rhos)):
        raise ValueError("--rhos 必须包含不重复数值")
    if any(not math.isfinite(rho) or rho < -1.0 or rho > 1.0 for rho in rhos):
        raise ValueError("--rhos 必须全部位于 [-1,1] 且为有限数值")
    if not args.allow_rho_shard and 0.0 not in rhos:
        raise ValueError("闭环 rho 网格必须包含 0，以提供基线保持参照")
    if args.scenario_validation_only:
        if args.allow_rho_shard:
            parser.error("场景预筛查不需要 --allow-rho-shard")
        if not tokens:
            parser.error("--scenario-validation-only 必须提供 --scenario-tokens-file")
        if rhos != [0.0]:
            parser.error("场景预筛查必须使用 --rhos=0")
    elif tokens and not args.allow_rho_shard and tuple(rhos) != FORMAL_RHOS:
        raise ValueError(f"正式闭环评测必须按固定 rho 网格运行：{FORMAL_RHOS}")
    if args.limit_total_scenarios is not None and args.limit_total_scenarios <= 0:
        raise ValueError("--limit-total-scenarios 必须为正数")

    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    devkit = repository / "nuplan-devkit"
    search = (
        "[pkg://nuplan.planning.script.config.common,"
        "pkg://nuplan.planning.script.experiments,"
        f"file://{config_root},file://{repository / 'stylelora' / 'config'}]"
    )
    child_env = os.environ.copy()
    python_paths = [str(devkit), str(repository)]
    if child_env.get("PYTHONPATH"):
        python_paths.append(child_env["PYTHONPATH"])
    child_env.update({
        "PYTHONPATH": os.pathsep.join(python_paths),
        "NUPLAN_DEVKIT_ROOT": str(devkit),
        "NUPLAN_DATA_ROOT": str(data_root),
        "NUPLAN_MAPS_ROOT": str(maps_root),
        "NUPLAN_EXP_ROOT": str(root),
        "HYDRA_FULL_ERROR": "1",
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
        "SWANLAB_MODE": "disabled",
    })

    report: dict[str, object] = {
        "settings": {
            "rhos": rhos,
            "rho_shard_mode": bool(args.allow_rho_shard),
            "scenario_filter": args.scenario_filter,
            "scenario_tokens_file": args.scenario_tokens_file,
            "expected_scenario_count": args.expected_scenario_count,
            "scenario_token_count": len(tokens),
            "scenario_tokens": tokens,
            "removed_route_failure_tokens": removed_route_failure_tokens,
            "fixed_route_failure_tokens": sorted(ROUTE_FAILURE_TOKENS),
            "challenge": args.challenge,
            "data_root": str(data_root),
            "maps_root": str(maps_root),
            "scene_gate_enabled": bool(args.enable_scene_gate),
            "scene_gate_checkpoint": str(scene_gate_checkpoint) if scene_gate_checkpoint else None,
            "conditional_router_enabled": bool(conditional_router_checkpoint),
            "conditional_router_checkpoint": (
                str(conditional_router_checkpoint) if conditional_router_checkpoint else None
            ),
            "bounded_style_enabled": bool(args.enable_bounded_style),
            "trajectory_repair_enabled": bool(args.enable_trajectory_repair),
            "trajectory_repair_mode": args.trajectory_repair_mode,
            "collision_repair_scales": list(collision_repair_scales),
            "collision_repair_horizon_s": args.collision_repair_horizon_s,
            "collision_repair_min_clearance_m": args.collision_repair_min_clearance_m,
            "repair_check_drivable_area": bool(args.repair_check_drivable_area),
            "repair_drivable_horizon_s": args.repair_drivable_horizon_s,
            "bounded_candidate_ratios": candidate_ratios,
            "bounded_max_mean_accel_degradation": args.bounded_max_mean_accel_degradation,
            "bounded_max_mean_jerk_degradation": args.bounded_max_mean_jerk_degradation,
            "bounded_max_progress_loss_m": args.bounded_max_progress_loss_m,
            "bounded_max_lateral_deviation_m": args.bounded_max_lateral_deviation_m,
            "scenario_validation_only": bool(args.scenario_validation_only),
            "minimum_valid_scenarios": args.minimum_valid_scenarios,
            "allow_partial_scenarios": bool(args.allow_partial_scenarios),
        },
        "records": [],
    }
    report_path = root / "closed_loop_lora_report.json"
    raw_roots: dict[float, Path] = {}
    try:
        for rho in rhos:
            result_dir = _rho_dir(root, rho)
            challenge_dir = _challenge_result_dir(result_dir, args.challenge)
            raw_dir = result_dir / "raw_step_data"
            video_dir = result_dir / "simulation_video"
            raw_roots[rho] = raw_dir
            existing_metrics = list(challenge_dir.glob("metrics/*.parquet"))
            existing_aggregator = list(challenge_dir.glob("aggregator_metric/*.parquet"))
            existing_runner_report = challenge_dir / "runner_report.parquet"
            if not (
                args.skip_existing
                and existing_metrics
                and existing_aggregator
                and existing_runner_report.exists()
            ):
                command = [
                    sys.executable,
                    "-m",
                    "nuplan.planning.script.run_simulation",
                    f"+simulation={args.challenge}",
                    "planner=lora_diffusion_planner",
                    f"planner.lora_diffusion_planner.config.args_file={args_file}",
                    f"planner.lora_diffusion_planner.ckpt_path={checkpoint}",
                    f"planner.lora_diffusion_planner.config.lora_aggressive_adapter={high_adapter}",
                    f"planner.lora_diffusion_planner.config.lora_conservative_adapter={low_adapter}",
                    f"planner.lora_diffusion_planner.config.lora_normalization_file={normalization}",
                    f"planner.lora_diffusion_planner.config.lora_rank={args.rank}",
                    f"planner.lora_diffusion_planner.config.lora_rho={rho}",
                    "planner.lora_diffusion_planner.config.conditional_router_enabled="
                    f"{'true' if conditional_router_checkpoint else 'false'}",
                    f"planner.lora_diffusion_planner.config.scene_gate_enabled={'true' if args.enable_scene_gate else 'false'}",
                    "planner.lora_diffusion_planner.config.bounded_style_enabled="
                    f"{'true' if args.enable_bounded_style else 'false'}",
                    "planner.lora_diffusion_planner.config.trajectory_repair_enabled="
                    f"{'true' if args.enable_trajectory_repair else 'false'}",
                    "planner.lora_diffusion_planner.config.trajectory_repair_mode="
                    f"{args.trajectory_repair_mode}",
                    "planner.lora_diffusion_planner.config.collision_repair_scales="
                    f"{json.dumps(collision_repair_scales, separators=(',', ':'))}",
                    "planner.lora_diffusion_planner.config.collision_repair_horizon_s="
                    f"{args.collision_repair_horizon_s}",
                    "planner.lora_diffusion_planner.config.collision_repair_min_clearance_m="
                    f"{args.collision_repair_min_clearance_m}",
                    "planner.lora_diffusion_planner.config.repair_check_drivable_area="
                    f"{'true' if args.repair_check_drivable_area else 'false'}",
                    "planner.lora_diffusion_planner.config.repair_drivable_horizon_s="
                    f"{args.repair_drivable_horizon_s}",
                    "planner.lora_diffusion_planner.config.bounded_candidate_ratios="
                    f"{json.dumps(candidate_ratios, separators=(',', ':'))}",
                    "planner.lora_diffusion_planner.config.bounded_max_mean_accel_degradation="
                    f"{args.bounded_max_mean_accel_degradation}",
                    "planner.lora_diffusion_planner.config.bounded_max_mean_jerk_degradation="
                    f"{args.bounded_max_mean_jerk_degradation}",
                    "planner.lora_diffusion_planner.config.bounded_max_progress_loss_m="
                    f"{args.bounded_max_progress_loss_m}",
                    "planner.lora_diffusion_planner.config.bounded_max_lateral_deviation_m="
                    f"{args.bounded_max_lateral_deviation_m}",
                    (
                        f"planner.lora_diffusion_planner.config.render_save_dir={video_dir}"
                        if args.render else "planner.lora_diffusion_planner.config.render_save_dir=null"
                    ),
                    f"planner.lora_diffusion_planner.config.raw_data_save_dir={raw_dir}",
                    f"planner.lora_diffusion_planner.device={args.device}",
                    "scenario_builder=nuplan",
                    f"scenario_filter={args.scenario_filter}",
                    f"scenario_builder.db_files={data_root}",
                    f"experiment_uid=stylelora/{args.challenge}/{result_dir.name}",
                    f"output_dir={challenge_dir}",
                    f"hydra.run.dir={challenge_dir}",
                    f"hydra.searchpath={search}",
                    f"worker={args.worker}",
                    "verbose=true",
                    "enable_simulation_progress_bar=true",
                    "number_of_gpus_allocated_per_simulation=1.0",
                ]
                if args.alpha is not None:
                    command.append(f"planner.lora_diffusion_planner.config.lora_alpha={args.alpha}")
                if args.enable_scene_gate:
                    command.append(
                        f"planner.lora_diffusion_planner.config.scene_gate_checkpoint={scene_gate_checkpoint}"
                    )
                if conditional_router_checkpoint is not None:
                    command.append(
                        "planner.lora_diffusion_planner.config.conditional_router_checkpoint="
                        f"{conditional_router_checkpoint}"
                    )
                if tokens:
                    compact = json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))
                    command.extend([
                        f"scenario_filter.scenario_tokens={compact}",
                        f"scenario_filter.limit_total_scenarios={len(tokens)}",
                    ])
                elif args.limit_total_scenarios is not None:
                    command.append(f"scenario_filter.limit_total_scenarios={args.limit_total_scenarios}")
                command.extend(extra)
                print("[closed-loop]", " ".join(command), flush=True)
                subprocess.run(command, check=True, env=child_env)
            runner_summary = _runner_summary(challenge_dir, tokens if tokens else None)
            metric_summary = _metric_summary(challenge_dir, minimums)
            official_summary = (
                {"available": False, "skipped": "场景预筛查不生成正式 challenge 汇总"}
                if args.scenario_validation_only
                else (
                    _official_aggregator_summary(
                        challenge_dir,
                        tokens,
                        allow_partial=args.allow_partial_scenarios,
                    )
                    if tokens else {"available": False, "failure": "冒烟测试不生成正式场景配对报告"}
                )
            )
            record = {
                "rho": rho,
                "result_dir": str(challenge_dir),
                "runner": runner_summary,
                "official_aggregator": official_summary,
                **metric_summary,
            }
            report["records"].append(record)
            write_json(report_path, report)
            if args.scenario_validation_only:
                if not runner_summary.get("token_set_matches", False):
                    raise RuntimeError(f"候选场景集合与 runner_report 不一致：{runner_summary}")
                if int(runner_summary.get("runner_count", 0)) != len(tokens):
                    raise RuntimeError(
                        f"候选场景应运行 {len(tokens)} 个，runner_report 实际为 "
                        f"{runner_summary.get('runner_count', 0)} 个"
                    )
                continue
            if args.allow_partial_scenarios:
                if runner_summary.get("unexpected_tokens") or runner_summary.get("duplicate_tokens"):
                    raise RuntimeError(
                        f"rho={rho:+.2f} runner_report 出现候选集外或重复 token：{runner_summary}"
                    )
                if int(runner_summary.get("successful_count", 0)) <= 0:
                    raise RuntimeError(f"rho={rho:+.2f} 没有任何成功场景")
            elif not runner_summary.get("all_succeeded", False):
                raise RuntimeError(
                    f"rho={rho:+.2f} 存在失败的闭环场景：{runner_summary}"
                )
            # NuPlan may return exit code 0 even when every individual runner
            # failed.  Missing metrics must therefore be treated as a failed job.
            if metric_summary.get("failure"):
                raise RuntimeError(
                    f"rho={rho:+.2f} 闭环仿真没有生成有效指标：{metric_summary['failure']}"
                )
            if tokens and not official_summary.get("available", False):
                raise RuntimeError(
                    f"rho={rho:+.2f} 没有生成正式 challenge 综合分：{official_summary}"
                )
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(report_path, report)
        raise

    records = report["records"]
    if args.scenario_validation_only:
        runner = records[0]["runner"]
        valid_tokens = list(runner.get("successful_tokens", []))
        failed_tokens = list(runner.get("failed_tokens", []))
        valid_path = root / "valid_scenario_tokens.json"
        failed_path = root / "failed_scenario_tokens.json"
        valid_path.write_text(
            json.dumps(valid_tokens, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        failed_path.write_text(
            json.dumps(failed_tokens, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report["scenario_validation"] = {
            "candidate_count": len(tokens),
            "valid_count": len(valid_tokens),
            "failed_count": len(failed_tokens),
            "minimum_valid_scenarios": args.minimum_valid_scenarios,
            "valid_tokens_file": str(valid_path),
            "failed_tokens_file": str(failed_path),
            "failed_scenarios": runner.get("failed_scenarios", []),
        }
        if len(valid_tokens) < args.minimum_valid_scenarios:
            report["failure"] = (
                f"有效场景仅有 {len(valid_tokens)} 个，少于要求的 "
                f"{args.minimum_valid_scenarios} 个"
            )
            write_json(report_path, report)
            raise RuntimeError(report["failure"])
        write_json(report_path, report)
        print(json.dumps(report["scenario_validation"], ensure_ascii=False, indent=2))
        return

    report["closed_loop_pass"] = bool(records) and all(row.get("closed_loop_pass", False) for row in records)
    report["metric_deltas_vs_rho_zero"] = _metric_deltas(records)
    if tokens:
        if args.allow_partial_scenarios:
            successful_sets = [
                {row["token"] for row in record["official_aggregator"].get("per_scenario", [])}
                for record in records
            ]
            common_tokens = [
                token for token in tokens
                if all(token in successful for successful in successful_sets)
            ]
            report["partial_collection"] = {
                "candidate_count": len(tokens),
                "common_successful_count_within_configuration": len(common_tokens),
                "common_successful_tokens_within_configuration": common_tokens,
                "note": "该报告是一次性候选场景采集结果；正式统计须再取所有消融配置的共同成功 token 交集。",
            }
        else:
            if any(abs(float(record["rho"])) < 1e-12 for record in records):
                report["rho_zero_baseline"] = _rho_zero_baseline(records)
                report["paired_official_differences_vs_rho_zero"] = _paired_official_differences(records)
            elif args.allow_rho_shard:
                report["rho_shard"] = {
                    "contains_rho_zero": False,
                    "note": "该分片不含 rho=0；跨分片配对差异应在全部分片完成后统一计算。",
                }
    report["style_proxy"] = analyze_closed_loop_style_runs(raw_roots)
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

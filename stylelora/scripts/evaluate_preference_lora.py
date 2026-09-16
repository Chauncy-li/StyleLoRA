"""Evaluate continuous-preference LoRA via fixed-seed rho scanning (full metrics).

4b 完整指标：
- 双独立 baseline：planner(注入LoRA) 与 baseline_plain(未注入) 各自 load_plain_baseline；
- identity：adapter 关闭 vs 独立 baseline，同 batch+seed；
- rho 扫描：固定 batch，内层遍历 rho（同 seed），prediction 物理轨迹过冻结 CSPQ；
- 分场景 s/z 统计（straight_free_drive / straight_car_follow 分开）；
- z 目标分布 MMD：用 latent bank 对应 rank 区间（high/low）做参考分布；
- rho 单调性检查（s 随 rho 是否单调增）与正确/相反方向对比；
- 三轴物理指标（复用 scene_style_vector）、ADE/FDE、邻车变化、rollout 耗时。
"""

from __future__ import annotations

import argparse
import copy
import json
from itertools import islice
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stylelora.lora.evaluation.rollout import rollout_with_rho
from stylelora.lora.evaluation.style_metrics import (
    ade_fde,
    mmd_rbf,
    open_loop_behavior_metrics,
    scene_style_vector,
)
from stylelora.lora.model.checkpoint import load_adapter_checkpoint
from stylelora.lora.model.style_lora_planner import StyleLoRAPlanner
from stylelora.lora.runtime import load_plain_baseline, prepare_diffusion_batch
from stylelora.lora.training.losses import build_noisy_inputs

from stylelora.data.preference_lora_dataset import (
    PreferenceLoRADataset, SceneBalancedLoRASampler, preference_lora_collate,
)
from stylelora.paths import (
    DEFAULT_ENCODER_CHECKPOINT, DEFAULT_FEATURE_INDEX, DEFAULT_FEATURE_NPY,
    DEFAULT_PREFERENCE_MANIFEST, ensure_repo_on_path,
)
from stylelora.model.scene_gate import load_scene_gate_checkpoint
from stylelora.model.conditional_lora_router import load_conditional_router_checkpoint
from stylelora.training.preference_lora import load_frozen_cspq

SCENE_NAMES = ("straight_free_drive", "straight_car_follow")
HIGH_RANK = (0.8, 1.0)
LOW_RANK = (0.0, 0.2)

PHYSICAL_VALUE_METRICS = (
    "planned_mean_speed_mps",
    "planned_progress_m",
    "planned_accel_p90_mps2",
    "planned_decel_p90_mps2",
    "planned_abs_jerk_p90_mps3",
    "route_aligned_progress_m",
    "min_lead_gap_m",
    "min_time_headway_s",
    "min_ttc_s",
    "drivable_area_proxy_fraction",
    "offroad_proxy_fraction",
    "min_collision_clearance_m",
)
PHYSICAL_RATE_METRICS = (
    "route_alignment_valid",
    "drivable_area_proxy_valid",
    "lead_metric_valid",
    "ttc_closing_event",
    "collision_proxy",
)
PHYSICAL_EXPECTED_DIRECTIONS = {
    "planned_mean_speed_mps": 1,
    "planned_progress_m": 1,
    "planned_accel_p90_mps2": 1,
    "planned_decel_p90_mps2": 1,
    "planned_abs_jerk_p90_mps3": 1,
    "route_aligned_progress_m": 1,
    "min_lead_gap_m": -1,
    "min_time_headway_s": -1,
    "min_ttc_s": -1,
}


def _featurize_ego(pred_phys: torch.Tensor) -> torch.Tensor:
    """prediction 已是物理轨迹（[B,T,4]: x,y,cosθ,sinθ），直接构 token（不回 normalizer.inverse）。

    解码器/缓存管线的物理状态约定为 [x, y, cosθ, sinθ]：第 2/3 维已经是
    归一化方向向量，不是 heading 弧度，因此只做 L2 归一化后拼接，
    绝不能再当成 heading 去算 cos/sin（会产生错误的二次编码）。
    """
    pos = pred_phys[:, :, :2]
    delta = torch.zeros_like(pos)
    if pos.shape[1] > 1:
        delta[:, 1:] = pos[:, 1:] - pos[:, :-1]
    heading_vec = torch.nn.functional.normalize(pred_phys[:, :, 2:4], dim=-1, eps=1e-6)
    return torch.cat((pos, delta, heading_vec), dim=-1)


def _ego_physical(pred: torch.Tensor) -> torch.Tensor:
    """取出 ego token 的物理轨迹 [B,T,4]；兼容 [B,P,T,4]（token=0 为 ego）与 [B,T,4]。"""
    return pred[:, 0] if pred.ndim == 4 else pred


def _prep(batch, device, obs_normalizer):
    tensors = {k: v.to(device) for k, v in batch["tensors"].items()}
    # 用真实 scene_id 推断场景类型（不要假设 batch 内第 0 个样本一定是 free_drive；
    # SceneBalancedLoRASampler 只保证每场景各半，不保证顺序）。
    meta = [{"scene_type": SCENE_NAMES[int(sid)]} for sid in batch["scene_id"].cpu().tolist()]
    return prepare_diffusion_batch({"tensors": tensors, "metadata": meta}, device, obs_normalizer,
                                   return_style_context=False)  # 只返回 2 值


def _style_vector_for_item(tensors: dict, prediction: torch.Tensor, index: int, scene: str):
    """单个样本的三轴物理风格向量（预测轨迹 -> scene_style_vector）。"""
    return scene_style_vector(
        scene=scene,
        ego_future=_ego_physical(prediction[index:index + 1])[0],
        ego_current=tensors["ego_current_state"][index],
        neighbors_past=tensors["neighbor_agents_past"][index],
        neighbors_future=tensors["neighbors_future_gt"][index],
        route_limits=tensors["route_lanes_speed_limit"][index],
        route_has_limits=tensors["route_lanes_has_speed_limit"][index],
        lane_limits=tensors["lanes_speed_limit"][index],
        lane_has_limits=tensors["lanes_has_speed_limit"][index],
    )


def _tensor_item(tensors: dict, key: str, index: int) -> torch.Tensor | None:
    value = tensors.get(key)
    return None if value is None else value[index]


def _behavior_metrics_for_item(tensors: dict, prediction: torch.Tensor, index: int) -> dict:
    """Compute open-loop physical values from one generated ego trajectory."""
    return open_loop_behavior_metrics(
        ego_future=_ego_physical(prediction[index:index + 1])[0],
        ego_current=tensors["ego_current_state"][index],
        neighbors_past=tensors["neighbor_agents_past"][index],
        neighbors_future=tensors["neighbors_future_gt"][index],
        neighbor_future_valid_mask=_tensor_item(
            tensors, "neighbor_agents_future_mask", index
        ),
        route_lanes=_tensor_item(tensors, "route_lanes", index),
        route_lanes_mask=_tensor_item(tensors, "route_lanes_mask", index),
        lanes=_tensor_item(tensors, "lanes", index),
        lanes_mask=_tensor_item(tensors, "lanes_mask", index),
        static_objects=_tensor_item(tensors, "static_objects", index),
    )


def _latent_roi(ds: PreferenceLoRADataset) -> torch.Tensor:
    """收集 latent bank 中与 dataset 样本行对应的 z（作为目标分布参考）。

    复用 dataset 已加载的 latent bank，避免重复读取 npy。
    """
    if ds.latent_rows and hasattr(ds, "_latent"):
        return torch.as_tensor(np.asarray(ds._latent)[ds.latent_rows], dtype=torch.float32)
    return torch.stack([ds[i]["z_target"] for i in range(len(ds))])


def _mean_std(values: list) -> dict:
    return {"mean": float(np.mean(values)) if values else float("nan"),
            "std": float(np.std(values)) if values else float("nan")}


def _distribution(values: list[float]) -> dict[str, float | int]:
    """汇总连续 rho 验证指标，保留尾部误差以避免均值掩盖异常样本。"""
    if not values:
        return {"count": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "p05": float("nan"), "median": float("nan"),
                "p95": float("nan"), "max": float("nan")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _finite_values(rows: list[dict], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return values


def physical_behavior_summary(rows: list[dict]) -> dict:
    """Aggregate stored physical values without using ADE/FDE."""
    return {
        "metrics": {
            metric: _distribution(_finite_values(rows, metric))
            for metric in PHYSICAL_VALUE_METRICS
        },
        "rates": {
            metric: {
                "count": len(rows),
                "rate": float(np.mean([bool(row.get(metric, False)) for row in rows]))
                if rows else float("nan"),
            }
            for metric in PHYSICAL_RATE_METRICS
        },
    }


def physical_response_metrics(records: list[dict], rho_list: list[float]) -> dict:
    """Measure sample-paired physical response directions over the rho grid."""
    ordered_rhos = sorted(float(rho) for rho in rho_list)
    zero_rho = next((rho for rho in ordered_rhos if abs(rho) < 1e-9), None)
    by_sample: dict[tuple[int, int, str], dict[float, dict]] = defaultdict(dict)
    for row in records:
        sample_key = (int(row["batch"]), int(row["sample"]), str(row.get("key", "")))
        by_sample[sample_key][float(row["rho"])] = row

    result: dict[str, object] = {
        "definition": (
            "Paired open-loop physical response; no ADE/FDE is used. Increasing rho is "
            "expected to increase progress/speed/acceleration/braking/jerk and decrease "
            "gap/THW/TTC. Safety proxy rates are descriptive, not style targets."
        ),
        "rho_grid": ordered_rhos,
        "metrics": {},
    }
    for metric, direction in PHYSICAL_EXPECTED_DIRECTIONS.items():
        sequences: list[list[float]] = []
        for values_by_rho in by_sample.values():
            values = [values_by_rho.get(rho, {}).get(metric) for rho in ordered_rhos]
            if all(value is not None and np.isfinite(float(value)) for value in values):
                sequences.append([float(value) for value in values])

        monotonic = 0
        directed_endpoint_changes: list[float] = []
        correct_vs_zero = 0
        comparisons_vs_zero = 0
        for sequence in sequences:
            directed = [direction * value for value in sequence]
            monotonic += int(all(
                directed[index] >= directed[index - 1] - 1e-6
                for index in range(1, len(directed))
            ))
            directed_endpoint_changes.append(directed[-1] - directed[0])
            if zero_rho is not None:
                zero_index = ordered_rhos.index(zero_rho)
                for index, rho in enumerate(ordered_rhos):
                    if abs(rho) < 1e-9:
                        continue
                    comparisons_vs_zero += 1
                    delta = direction * (sequence[index] - sequence[zero_index])
                    correct_vs_zero += int(delta * rho > 0.0)

        metric_result = {
            "expected_direction_as_rho_increases": "increasing" if direction > 0 else "decreasing",
            "complete_sample_count": len(sequences),
            "sample_monotonic_rate": monotonic / len(sequences) if sequences else None,
            "direction_correct_vs_zero_rate": (
                correct_vs_zero / comparisons_vs_zero if comparisons_vs_zero else None
            ),
            "comparisons_vs_zero": comparisons_vs_zero,
            "directed_endpoint_change": _distribution(directed_endpoint_changes),
            "mean_sequence": [
                float(np.mean([sequence[index] for sequence in sequences]))
                if sequences else None
                for index in range(len(ordered_rhos))
            ],
        }
        result["metrics"][metric] = metric_result
    return result


def continuous_rho_metrics(
    records: list[dict],
    rho_list: list[float],
    *,
    style_response_epsilon: float = 0.01,
    max_ade_cost: float = 0.5,
    max_fde_cost: float = 1.0,
    max_jerk_cost: float = 2.0,
) -> dict:
    """计算逐样本的偏好 latent 插值误差和标量偏好单调率。

    负、正方向分别以 rho=0 和扫描端点作为线性插值端点。该统计与训练时
    ICT 的 latent MSE 定义一致，但不会改变模型、推理结果或原有评测指标。
    """
    ordered_rhos = sorted(float(rho) for rho in rho_list)
    if style_response_epsilon < 0:
        raise ValueError("style_response_epsilon must be non-negative")
    if any(value < 0 for value in (max_ade_cost, max_fde_cost, max_jerk_cost)):
        raise ValueError("performance-cost budgets must be non-negative")
    zero_rho = next((rho for rho in ordered_rhos if abs(rho) < 1e-9), None)
    if zero_rho is None:
        raise ValueError("连续 rho 验证要求扫描网格包含 rho=0")

    by_sample: dict[tuple[int, int], dict[float, dict]] = defaultdict(dict)
    for row in records:
        by_sample[(int(row["batch"]), int(row["sample"]))][float(row["rho"])] = row

    complete_samples = [values for values in by_sample.values()
                        if all(rho in values for rho in ordered_rhos)]
    monotonic_count = 0
    style_spans: list[float] = []
    average_slopes: list[float] = []
    regression_slopes: list[float] = []
    positive_slope_count = 0
    response_sample_count = 0
    rho_array = np.asarray(ordered_rhos, dtype=np.float64)
    rho_centered = rho_array - float(rho_array.mean())
    slope_denominator = float(np.square(rho_centered).sum())
    for values in complete_samples:
        sequence = [float(values[rho]["s"]) for rho in ordered_rhos]
        monotonic_count += int(all(sequence[i] >= sequence[i - 1] - 1e-3
                                   for i in range(1, len(sequence))))
        style_span = sequence[-1] - sequence[0]
        style_spans.append(style_span)
        rho_span = ordered_rhos[-1] - ordered_rhos[0]
        average_slopes.append(style_span / rho_span if rho_span > 0 else float("nan"))
        sequence_array = np.asarray(sequence, dtype=np.float64)
        slope = (
            float(np.sum(rho_centered * (sequence_array - sequence_array.mean())) / slope_denominator)
            if slope_denominator > 0 else float("nan")
        )
        regression_slopes.append(slope)
        positive_slope_count += int(slope > 1e-3)
        zero_style = float(values[zero_rho]["s"])
        endpoint_response = max(
            abs(float(values[ordered_rhos[0]]["s"]) - zero_style),
            abs(float(values[ordered_rhos[-1]]["s"]) - zero_style),
        )
        response_sample_count += int(endpoint_response >= style_response_epsilon)

    def _finite_value(row: dict, key: str) -> float | None:
        value = row.get(key)
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    control_by_rho: dict[str, dict[str, object]] = {}
    all_response_flags: list[float] = []
    all_feasible_control_flags: list[float] = []
    all_retention_ratios: list[float] = []
    performance_cost_values: dict[str, list[float]] = {
        "ade_delta": [], "fde_delta": [], "abs_jerk_p90_delta": [],
    }
    for rho in ordered_rhos:
        if abs(rho - zero_rho) < 1e-9:
            continue
        response_flags: list[float] = []
        feasible_control_flags: list[float] = []
        retention_ratios: list[float] = []
        rho_costs = {key: [] for key in performance_cost_values}
        for values in complete_samples:
            row, base = values[rho], values[zero_rho]
            responded = abs(float(row["s"]) - float(base["s"])) >= style_response_epsilon
            response_flags.append(float(responded))

            effective_rho = _finite_value(row, "effective_rho")
            if effective_rho is not None and abs(rho) > 1e-9:
                retention_ratios.append(min(abs(effective_rho / rho), 1.0))

            costs: dict[str, float] = {}
            for output_name, record_name in (
                ("ade_delta", "ade"),
                ("fde_delta", "fde"),
                ("abs_jerk_p90_delta", "planned_abs_jerk_p90"),
            ):
                current = _finite_value(row, record_name)
                reference = _finite_value(base, record_name)
                if current is not None and reference is not None:
                    costs[output_name] = current - reference
                    rho_costs[output_name].append(current - reference)
                    performance_cost_values[output_name].append(current - reference)

            required_costs = ("ade_delta", "fde_delta", "abs_jerk_p90_delta")
            if all(key in costs for key in required_costs):
                within_budget = (
                    costs["ade_delta"] <= max_ade_cost
                    and costs["fde_delta"] <= max_fde_cost
                    and costs["abs_jerk_p90_delta"] <= max_jerk_cost
                )
                feasible_control_flags.append(float(responded and within_budget))

        all_response_flags.extend(response_flags)
        all_feasible_control_flags.extend(feasible_control_flags)
        all_retention_ratios.extend(retention_ratios)
        control_by_rho[f"rho_{rho:.2f}"] = {
            "response_rate": float(np.mean(response_flags)) if response_flags else float("nan"),
            "effective_control_coverage": (
                float(np.mean(feasible_control_flags)) if feasible_control_flags else None
            ),
            "effective_control_coverage_role": "appendix_imitation_diagnostic",
            "command_retention_ratio": _distribution(retention_ratios),
            "performance_cost": {
                key: _distribution(values) for key, values in rho_costs.items()
            },
        }

    result: dict[str, object] = {
        "complete_sample_count": len(complete_samples),
        "samplewise_s_monotonic_rate": (
            monotonic_count / len(complete_samples) if complete_samples else float("nan")
        ),
        "style_span": _distribution(style_spans),
        "average_slope": _distribution(average_slopes),
        "regression_slope": _distribution(regression_slopes),
        "positive_slope_rate": (
            positive_slope_count / len(complete_samples) if complete_samples else float("nan")
        ),
        "nonzero_response_rate": (
            response_sample_count / len(complete_samples) if complete_samples else float("nan")
        ),
        "effective_control_coverage": {
            "paper_role": "appendix_imitation_diagnostic_not_main_physical_metric",
            "definition": (
                "abs(style_delta)>=epsilon and ADE/FDE/jerk costs remain within configured budgets"
            ),
            "style_response_epsilon": style_response_epsilon,
            "max_ade_cost": max_ade_cost,
            "max_fde_cost": max_fde_cost,
            "max_jerk_cost": max_jerk_cost,
            "overall_response_rate": (
                float(np.mean(all_response_flags)) if all_response_flags else float("nan")
            ),
            "overall": (
                float(np.mean(all_feasible_control_flags))
                if all_feasible_control_flags else None
            ),
            "command_retention_ratio": _distribution(all_retention_ratios),
            "by_rho": control_by_rho,
        },
        "performance_cost_vs_rho_zero": {
            key: _distribution(values) for key, values in performance_cost_values.items()
        },
    }
    direction_endpoints = {
        "low": min(ordered_rhos),
        "high": max(ordered_rhos),
    }
    for direction, endpoint_rho in direction_endpoints.items():
        if (direction == "low" and endpoint_rho >= 0) or (direction == "high" and endpoint_rho <= 0):
            result[direction] = {"available": False, "endpoint_rho": endpoint_rho}
            continue

        intermediate_rhos = [
            rho for rho in ordered_rhos
            if min(zero_rho, endpoint_rho) < rho < max(zero_rho, endpoint_rho)
        ]
        latent_mse: list[float] = []
        latent_relative_rmse: list[float] = []
        scalar_abs_error: list[float] = []
        by_rho_errors: dict[float, dict[str, list[float]]] = {
            rho: {"latent_mse": [], "latent_relative_rmse": [], "scalar_abs_error": []}
            for rho in intermediate_rhos
        }
        used_samples = 0
        for values in complete_samples:
            base = values[zero_rho]
            endpoint = values[endpoint_rho]
            z_base = np.asarray(base["z"], dtype=np.float64)
            z_endpoint = np.asarray(endpoint["z"], dtype=np.float64)
            endpoint_rmse = float(np.sqrt(np.mean(np.square(z_endpoint - z_base))))
            used_samples += 1
            for rho in intermediate_rhos:
                ratio = abs(rho / endpoint_rho)
                z_target = (1.0 - ratio) * z_base + ratio * z_endpoint
                z_mid = np.asarray(values[rho]["z"], dtype=np.float64)
                mse = float(np.mean(np.square(z_mid - z_target)))
                relative_rmse = (float(np.sqrt(mse) / endpoint_rmse)
                                 if endpoint_rmse > 1e-8 else None)
                s_target = ((1.0 - ratio) * float(base["s"])
                            + ratio * float(endpoint["s"]))
                s_error = abs(float(values[rho]["s"]) - s_target)

                latent_mse.append(mse)
                if relative_rmse is not None:
                    latent_relative_rmse.append(relative_rmse)
                scalar_abs_error.append(s_error)
                by_rho_errors[rho]["latent_mse"].append(mse)
                if relative_rmse is not None:
                    by_rho_errors[rho]["latent_relative_rmse"].append(relative_rmse)
                by_rho_errors[rho]["scalar_abs_error"].append(s_error)

        result[direction] = {
            "available": bool(intermediate_rhos and used_samples),
            "endpoint_rho": endpoint_rho,
            "sample_count": used_samples,
            "intermediate_rhos": intermediate_rhos,
            "latent_interpolation_mse": _distribution(latent_mse),
            "latent_relative_interpolation_rmse": _distribution(latent_relative_rmse),
            "scalar_s_interpolation_abs_error": _distribution(scalar_abs_error),
            "by_rho": {
                f"rho_{rho:.2f}": {name: _distribution(values)
                                    for name, values in metrics.items()}
                for rho, metrics in by_rho_errors.items()
            },
        }
    return result


def main() -> None:
    ensure_repo_on_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-file", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--adapter-high", required=True)
    parser.add_argument("--adapter-low", required=True)
    parser.add_argument("--cspq-checkpoint", default=str(DEFAULT_ENCODER_CHECKPOINT))
    parser.add_argument("--manifest", default=str(DEFAULT_PREFERENCE_MANIFEST))
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--feature-npy", default=str(DEFAULT_FEATURE_NPY))
    parser.add_argument("--feature-index", default=str(DEFAULT_FEATURE_INDEX))
    parser.add_argument("--latent-bank", required=True)
    parser.add_argument("--latent-bank-index", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--enable-scene-gate", action="store_true",
                        help="显式启用场景强度上限门控；默认保持现有 LoRA 行为。")
    parser.add_argument("--scene-gate-checkpoint", default=None)
    parser.add_argument(
        "--conditional-router-checkpoint",
        default=None,
        help="可选动态条件 LoRA 路由；省略时完整保持原评测逻辑。",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-eval-batches", type=int, default=10)
    parser.add_argument(
        "--sampling-mode", choices=("balanced", "full"), default="balanced",
        help="balanced 保持原有场景平衡采样；full 顺序遍历完整验证 manifest。",
    )
    parser.add_argument("--rho-min", type=float, default=-1.0)
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument("--rho-steps", type=int, default=9)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None,
                        help="LoRA alpha；与训练 --alpha 保持一致，否则 rho 强度缩放不一致。")
    parser.add_argument("--identity-tolerance", type=float, default=1e-4,
                        help="identity_max_mismatch 超过该容差时直接报错（默认 1e-4）。")
    parser.add_argument(
        "--style-response-epsilon", type=float, default=0.01,
        help="有效风格响应要求相对 rho=0 的标量风格变化至少达到该值。",
    )
    parser.add_argument("--max-ade-cost", type=float, default=0.5)
    parser.add_argument("--max-fde-cost", type=float, default=1.0)
    parser.add_argument("--max-jerk-cost", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.enable_scene_gate and not args.scene_gate_checkpoint:
        parser.error("--enable-scene-gate 必须同时提供 --scene-gate-checkpoint")
    if args.style_response_epsilon < 0:
        parser.error("--style-response-epsilon 不能为负")
    if any(value < 0 for value in (args.max_ade_cost, args.max_fde_cost, args.max_jerk_cost)):
        parser.error("性能代价预算不能为负")

    device = torch.device(args.device)
    cspq = load_frozen_cspq(args.cspq_checkpoint, args.device)
    rho_list = [float(round(x, 3)) for x in np.linspace(args.rho_min, args.rho_max, args.rho_steps)]

    ds = PreferenceLoRADataset(args.manifest, args.cache_root, args.latent_bank, args.latent_bank_index,
                               args.feature_npy, args.feature_index, direction="high", rank_low=0.0, rank_high=1.0)
    if len(ds) == 0:
        raise ValueError("评测 manifest 无样本")
    if args.sampling_mode == "balanced":
        sampler = SceneBalancedLoRASampler(
            ds, args.batch_size, generator=torch.Generator().manual_seed(args.seed)
        )
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=preference_lora_collate)
    else:
        # 全量模式只改变评测样本的遍历方式，不修改数据、模型或任何指标定义。
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, collate_fn=preference_lora_collate,
        )
    print(
        f"[stage 1/6] 读取固定评测批次：sampling_mode={args.sampling_mode}，"
        f"最多 {args.n_eval_batches}，"
        f"采样器可提供 {len(loader)} 批。",
        flush=True,
    )
    fixed_batch_count = min(args.n_eval_batches, len(loader))
    if args.sampling_mode == "balanced":
        fixed_batches = list(islice(loader, fixed_batch_count))

        def iterate_fixed_batches():
            return iter(fixed_batches)

        evaluated_samples = sum(len(batch["key"]) for batch in fixed_batches)
    else:
        # 全量输入包含较大的地图和邻车张量；每次按固定顺序流式重读，避免一次性驻留内存。
        def iterate_fixed_batches():
            return islice(iter(loader), fixed_batch_count)

        evaluated_samples = min(len(ds), fixed_batch_count * args.batch_size)
    print(
        f"[stage 1/6] 已固定 {fixed_batch_count} 个评测批次，"
        f"共 {evaluated_samples} 个样本。",
        flush=True,
    )

    # 参考 latent 分布：high/low rank 区间各自作为目标 z 分布
    ref_high = PreferenceLoRADataset(
        args.manifest, args.cache_root, args.latent_bank, args.latent_bank_index,
        args.feature_npy, args.feature_index, direction="high", rank_low=HIGH_RANK[0], rank_high=HIGH_RANK[1])
    ref_low = PreferenceLoRADataset(
        args.manifest, args.cache_root, args.latent_bank, args.latent_bank_index,
        args.feature_npy, args.feature_index, direction="low", rank_low=LOW_RANK[0], rank_high=LOW_RANK[1])
    ref_z = {"high": _latent_roi(ref_high), "low": _latent_roi(ref_low)}
    for name, ref in ref_z.items():
        if ref.numel() == 0:
            raise ValueError(f"参考 latent 分布 {name} 为空，请检查 manifest 的 rank 覆盖")

    # 双独立 baseline
    baseline_plain, config = load_plain_baseline(args.args_file, args.baseline_checkpoint, args.device)
    baseline_plain = baseline_plain.to(device).eval()
    # Keep an independent object while guaranteeing an exactly identical base
    # state, including any state absent from a non-strict baseline checkpoint.
    model_lora = copy.deepcopy(baseline_plain)
    planner = StyleLoRAPlanner(model_lora, rank=args.rank, alpha=args.alpha, dropout=0.0).to(device)
    for ckpt in (args.adapter_high, args.adapter_low):
        load_adapter_checkpoint(ckpt, planner, baseline_checkpoint=args.baseline_checkpoint,
                                normalization_file=config.normalization_file_path, strict_hash=True)
    gate_metadata = None
    router_metadata = None
    if args.conditional_router_checkpoint:
        router, prototypes, router_metadata = load_conditional_router_checkpoint(
            args.conditional_router_checkpoint, device
        )
        planner.attach_conditional_router(router, prototypes, enabled=True, trainable=False)
    if args.enable_scene_gate:
        gate, gate_metadata = load_scene_gate_checkpoint(args.scene_gate_checkpoint, device)
        planner.attach_scene_gate(gate, enabled=True)
    planner.eval()
    print(
        f"[stage 2/6] baseline、High/Low 适配器和偏好编码器加载完成；"
        f"conditional_router={'on' if args.conditional_router_checkpoint else 'off'}，"
        f"scene_gate={'on' if args.enable_scene_gate else 'off'}。",
        flush=True,
    )

    report = {
        "adapter_high": args.adapter_high,
        "adapter_low": args.adapter_low,
        "sampling_mode": args.sampling_mode,
        "dataset_samples": len(ds),
        "evaluated_batches": fixed_batch_count,
        "evaluated_samples": evaluated_samples,
        "rho_grid": rho_list,
        "conditional_router": {
            "enabled": bool(args.conditional_router_checkpoint),
            "checkpoint": args.conditional_router_checkpoint,
            "format": router_metadata.get("format") if router_metadata else None,
        },
        "scene_gate": {
            "enabled": bool(args.enable_scene_gate),
            "checkpoint": args.scene_gate_checkpoint,
            "format": gate_metadata.get("format") if gate_metadata else None,
        },
        "open_loop_physical_metrics": {
            "enabled": True,
            "dt_s": 0.1,
            "uses_ade_fde": False,
            "route_progress": "ego displacement projected onto cached route-lane tangents",
            "drivable_area": "cached lane-center/boundary corridor proxy",
            "collision": (
                "three-disc ego footprint versus ground-truth neighbor futures and cached "
                "static objects; open-loop proxy, not an official NuPlan closed-loop score"
            ),
        },
    }

    # ---------- rho=0 identity + 邻居基准预测（同 batch+seed） ----------
    # 说明：rollout_with_rho 在 fork_rng 内同时设置 torch.manual_seed 与
    # torch.cuda.manual_seed_all；identity 阶段必须用同样的 seed 设定，
    # 才能让邻居基准预测与 rho 扫描共享完全相同的采样噪声。
    fork_devices = [
        device.index if device.index is not None else torch.cuda.current_device()
    ] if device.type == "cuda" else []
    max_diff = 0.0
    identity_interval = max(1, fixed_batch_count // 10)
    for batch_index, batch in enumerate(iterate_fixed_batches(), start=1):
        prepped, futures = _prep(batch, device, config.observation_normalizer)
        ego_future, neighbors_future, _ = futures
        future = torch.cat((ego_future[:, None], neighbors_future), dim=1)
        fixed_time = torch.full((future.shape[0],), 0.5, device=device, dtype=future.dtype)
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(args.seed)
            if fork_devices:
                torch.cuda.manual_seed_all(args.seed)
            fixed_noise = torch.randn_like(future)
        fixed_inputs, _, _ = build_noisy_inputs(
            prepped, futures, planner.sde.marginal_prob, config.state_normalizer,
            time=fixed_time, noise=fixed_noise,
        )
        planner.disable_adapter()
        with torch.no_grad():
            # 两个前向分别重置同一 seed：保证二者共享完全相同的初始采样噪声，
            # 否则第二次前向的 RNG 已前进，identity_max_mismatch 会混入采样随机性。
            _, out_wrap = planner(fixed_inputs)
            _, out_plain = baseline_plain(fixed_inputs)
        for key in set(out_wrap) & set(out_plain):
            a, b = out_wrap[key], out_plain[key]
            if torch.is_tensor(a) and tuple(a.shape) == tuple(b.shape):
                max_diff = max(max_diff, float((a - b).abs().max()))
        if batch_index == 1 or batch_index == fixed_batch_count or batch_index % identity_interval == 0:
            print(
                f"[progress] rho=0 恒等检查: {batch_index}/{fixed_batch_count} "
                f"({batch_index / fixed_batch_count:.1%})",
                flush=True,
            )
    # identity 完成后必须重新启用适配器，否则后续 rho 扫描的 set_strength
    # 仍按 _enabled=False 路由，所有 rho 实际都是关闭适配器的 baseline。
    planner.enable_adapter()
    report["identity_max_mismatch"] = float(max_diff)
    if max_diff > args.identity_tolerance:
        raise RuntimeError(
            f"identity_max_mismatch={max_diff:.3e} 超过容差 {args.identity_tolerance:.3e}；"
            "LoRA 注入非恒等，评测不可信，请检查注入层/适配器加载。")
    print(f"[stage 3/6] 恒等检查完成：max_mismatch={max_diff:.3e}。", flush=True)

    # ---------- rho 扫描（固定 batch，内层 rho 同 seed） ----------
    # Full rho=0 rollouts are trajectory references, not the injection identity
    # test. They use the same wrapped model and seed as the subsequent rho scan.
    baseline_preds = []
    baseline_interval = max(1, fixed_batch_count // 10)
    for batch_index, batch in enumerate(iterate_fixed_batches(), start=1):
        prepped, _ = _prep(batch, device, config.observation_normalizer)
        with torch.no_grad():
            base_out, _ = rollout_with_rho(planner, prepped, 0.0, seed=args.seed)
        base_pred = base_out.get("prediction", base_out.get("x_start"))
        if base_pred is None or base_pred.ndim not in (3, 4):
            raise RuntimeError(
                f"Decoded prediction shape unexpected: "
                f"{None if base_pred is None else tuple(base_pred.shape)}"
            )
        baseline_preds.append(
            base_pred.detach().cpu() if args.sampling_mode == "full" else base_pred.detach()
        )
        if batch_index == 1 or batch_index == fixed_batch_count or batch_index % baseline_interval == 0:
            print(
                f"[progress] rho=0 轨迹基准: {batch_index}/{fixed_batch_count} "
                f"({batch_index / fixed_batch_count:.1%})",
                flush=True,
            )
    print("[stage 4/6] rho=0 轨迹基准生成完成。", flush=True)

    records = []
    scan_total = len(rho_list) * fixed_batch_count
    scan_completed = 0
    scan_interval = max(1, scan_total // 50)
    for rho_index, rho in enumerate(rho_list, start=1):
        print(f"[rho {rho_index}/{len(rho_list)}] 开始评测 rho={rho:+.3f}。", flush=True)
        planner.set_strength(rho)
        for bi, batch in enumerate(iterate_fixed_batches()):
            prepped, futures = _prep(batch, device, config.observation_normalizer)
            with torch.no_grad():
                out, seconds = rollout_with_rho(planner, prepped, rho, seed=args.seed)
            gate_debug = planner.last_scene_gate
            router_debug = planner.last_conditional_router
            pred = out.get("prediction", out.get("x_start"))
            if pred is None or pred.ndim not in (3, 4):
                raise RuntimeError(f"rho={rho} prediction missing or shape {None if pred is None else tuple(pred.shape)}")
            pred = pred.detach()
            pref = cspq(_featurize_ego(_ego_physical(pred)), batch["h_c"].to(device))
            s_out = pref["s"].squeeze(-1)      # [B]
            z_out = pref["z"]                  # [B, z_dim]
            scenes = [SCENE_NAMES[int(sid)] for sid in batch["scene_id"].cpu().tolist()]
            base_pred = baseline_preds[bi].to(pred.device)
            # 邻车变化：自适应 vs baseline(rho=0) 邻居 token 预测差异
            neighbor_change = None
            if pred.ndim == 4 and base_pred.ndim == 4:
                neighbor_change = (pred[:, 1:] - base_pred[:, 1:]).abs().mean().item()
            elif pred.ndim == 4:
                neighbor_change = pred[:, 1:].abs().mean().item()
            # 三轴指标：batch["tensors"] 来自 DataLoader 是 CPU 张量，必须用 CPU 预测，
            # 否则 scene_style_vector 内部 torch.cat 会 device mismatch；
            # CSPQ/ADE 仍使用 GPU pred。
            pred_cpu = pred.detach().cpu()
            n_samples = pred.shape[0]  # [B,T,D] 与 [B,P,T,D] 都按 batch 遍历
            for i in range(n_samples):
                phys_ego = _ego_physical(pred[i:i + 1])[0]
                base_phys_ego = _ego_physical(base_pred[i:i + 1])[0]
                gt_ego = futures[0][i:i + 1]
                ade_fde_dict = ade_fde(phys_ego.unsqueeze(0), gt_ego)
                scene = scenes[i]
                vector, valid = _style_vector_for_item(batch["tensors"], pred_cpu, i, scene)
                behavior = _behavior_metrics_for_item(batch["tensors"], pred_cpu, i)
                effective_rho = (
                    float(gate_debug["effective_rho"][i]) if gate_debug is not None else float(rho)
                )
                router_mean = (
                    float(router_debug["coefficient_mean"][i])
                    if router_debug is not None else None
                )
                router_std = (
                    float(router_debug["coefficient_std"][i])
                    if router_debug is not None else None
                )
                records.append({
                    "key": str(batch["key"][i]),
                    "rho": float(rho), "batch": bi, "sample": i, "scene": scene,
                    "effective_rho": effective_rho,
                    "conditional_coefficient_mean": router_mean,
                    "conditional_coefficient_std": router_std,
                    "gate_cap_low": (
                        float(gate_debug["cap_low"][i]) if gate_debug is not None else None
                    ),
                    "gate_cap_high": (
                        float(gate_debug["cap_high"][i]) if gate_debug is not None else None
                    ),
                    "s": float(s_out[i]), "z": [float(x) for x in z_out[i].cpu().tolist()],
                    "style_vector": [float(x) for x in vector.cpu().tolist()],
                    "style_valid": bool(valid.all()),
                    "ade": ade_fde_dict["ade"], "fde": ade_fde_dict["fde"],
                    # Keep old aliases so existing result readers continue to work.
                    "planned_mean_speed": behavior["planned_mean_speed_mps"],
                    "planned_abs_jerk_p90": behavior["planned_abs_jerk_p90_mps3"],
                    "lateral_shift_vs_baseline": float(
                        (phys_ego[:, 1] - base_phys_ego[:, 1]).abs().mean()
                    ),
                    "neighbor_change": neighbor_change,
                    "seconds_per_batch": seconds,
                    **behavior,
                })
            scan_completed += 1
            if scan_completed == 1 or scan_completed == scan_total or scan_completed % scan_interval == 0:
                print(
                    f"[progress] rho 扫描: {scan_completed}/{scan_total} "
                    f"({scan_completed / max(scan_total, 1):.1%})",
                    flush=True,
                )
        print(f"[rho {rho_index}/{len(rho_list)}] rho={rho:+.3f} 完成。", flush=True)

    # ---------- 跨 rho 共同有效样本集 ----------
    # car-follow 的前车有效性会随生成 ego 轨迹与 rho 一起变化；若每个 rho 各自
    # 取 valid 样本，不同 rho 的三轴均值会来自不同样本集，比较不公平。
    # 因此只保留"所有 rho 下都 style_valid"的样本做三轴均值比较
    # （沿用 evaluate_open_loop._mark_common_style_validity 的思想）。
    valid_by_rho: dict[str, dict[tuple[int, int], set[float]]] = defaultdict(lambda: defaultdict(set))
    for r in records:
        if r["style_valid"]:
            valid_by_rho[r["scene"]][(r["batch"], r["sample"])].add(r["rho"])
    expected_rhos = set(rho_list)
    common_valid: dict[str, set[tuple[int, int]]] = {
        scene: {key for key, valid_set in valid_by_rho[scene].items() if valid_set == expected_rhos}
        for scene in SCENE_NAMES
    }

    # ---------- 聚合 per-rho ----------
    per_rho = {}
    for rho in rho_list:
        rows = [r for r in records if abs(r["rho"] - rho) < 1e-9]
        s_all = [r["s"] for r in rows]
        z_all = torch.tensor([r["z"] for r in rows], dtype=torch.float32) if rows else torch.zeros(0)
        entry = {
            "count": len(rows),
            "s": _mean_std(s_all),
            "ade": _mean_std([r["ade"] for r in rows]),
            "fde": _mean_std([r["fde"] for r in rows]),
            "effective_rho": _mean_std([r["effective_rho"] for r in rows]),
            "conditional_coefficient_mean": _mean_std([
                r["conditional_coefficient_mean"] for r in rows
                if r["conditional_coefficient_mean"] is not None
            ]),
            "conditional_coefficient_std": _mean_std([
                r["conditional_coefficient_std"] for r in rows
                if r["conditional_coefficient_std"] is not None
            ]),
            "neighbor_change_mean": float(np.mean([r["neighbor_change"] for r in rows])) if rows and rows[0]["neighbor_change"] is not None else None,
            "seconds_per_batch": _mean_std([r["seconds_per_batch"] for r in rows]),
            "physical_behavior": physical_behavior_summary(rows),
            "by_scene": {},
        }
        for scene in SCENE_NAMES:
            scene_rows = [r for r in rows if r["scene"] == scene]
            zs = torch.tensor([r["z"] for r in scene_rows], dtype=torch.float32) if scene_rows else torch.zeros(0)
            common_keys = common_valid[scene]
            # 三轴均值只统计跨 rho 共同有效样本（free 恒有效；car-follow 受前车有效性影响）
            comp_rows = [r for r in scene_rows if (r["batch"], r["sample"]) in common_keys]
            vecs = [r["style_vector"] for r in comp_rows]
            entry["by_scene"][scene] = {
                "count": len(scene_rows),
                "s": _mean_std([r["s"] for r in scene_rows]),
                "z_mean": zs.mean(dim=0).tolist() if zs.numel() else [],
                "z_std": zs.std(dim=0).tolist() if zs.numel() else [],
                "style": {
                    "valid_rate": float(np.mean([r["style_valid"] for r in scene_rows])) if scene_rows else float("nan"),
                    "comparison_valid_rate": float(np.mean([(r["batch"], r["sample"]) in common_keys for r in scene_rows])) if scene_rows else float("nan"),
                    "comparison_valid_count": len(comp_rows),
                    "mean_axis_vector": np.mean(vecs, axis=0).tolist() if vecs else [],
                },
                "physical_behavior": physical_behavior_summary(scene_rows),
            }
        if z_all.numel():
            entry["mmd_z_high_ref"] = float(mmd_rbf(z_all, ref_z["high"]))
            entry["mmd_z_low_ref"] = float(mmd_rbf(z_all, ref_z["low"]))
        per_rho[f"rho_{rho:.2f}"] = entry
    report["per_rho"] = per_rho
    report["common_valid_by_scene"] = {scene: sorted(keys) for scene, keys in common_valid.items()}

    # ---------- rho 单调性（s 随 rho 单调不减） ----------
    ascending = sorted(rho_list)
    s_seq = [per_rho[f"rho_{r:.2f}"]["s"]["mean"] for r in ascending]
    eps = 1e-3
    violations = sum(1 for i in range(1, len(s_seq)) if s_seq[i] < s_seq[i - 1] - eps)
    report["monotonicity"] = {
        "rho_ascending": ascending, "s_sequence": s_seq,
        "monotonic_increasing": violations == 0, "decreasing_violations": violations,
    }

    # ---------- 正确方向 vs 相反方向 ----------
    by_sample: dict[tuple[int, int], dict[float, float]] = defaultdict(dict)
    for r in records:
        by_sample[(r["batch"], r["sample"])][r["rho"]] = r["s"]
    correct_zero, zero_total = 0, 0
    plus_minus_correct, plus_minus_total = 0, 0
    for key, s_by in by_sample.items():
        s0 = s_by.get(0.0)
        for rho in sorted(s_by):
            if abs(rho) < 1e-9 or s0 is None:
                continue
            zero_total += 1
            if np.sign(s_by[rho] - s0) == np.sign(rho):
                correct_zero += 1
            if rho > 0 and -rho in s_by:
                plus_minus_total += 1
                plus_minus_correct += int(s_by[rho] > s_by[-rho])
    report["direction_check"] = {
        "sign_correct_vs_zero_rate": (correct_zero / zero_total) if zero_total else float("nan"),
        "samples_compared_vs_zero": zero_total,
        "plus_minus_rate": (plus_minus_correct / plus_minus_total) if plus_minus_total else float("nan"),
        "samples_compared_plus_minus": plus_minus_total,
    }

    # 密集 rho 网格下逐样本衡量偏好流形是否接近端点间的线性、单调路径。
    report["continuous_rho"] = continuous_rho_metrics(
        records,
        rho_list,
        style_response_epsilon=args.style_response_epsilon,
        max_ade_cost=args.max_ade_cost,
        max_fde_cost=args.max_fde_cost,
        max_jerk_cost=args.max_jerk_cost,
    )
    report["physical_response"] = physical_response_metrics(records, rho_list)
    print("[stage 5/6] 指标聚合完成，正在写入 JSON 报告。", flush=True)

    report["records"] = records
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_report).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    summary = {k: v for k, v in report.items() if k != "records"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"[stage 6/6] 开环报告完成：{args.output_report}", flush=True)


if __name__ == "__main__":
    main()

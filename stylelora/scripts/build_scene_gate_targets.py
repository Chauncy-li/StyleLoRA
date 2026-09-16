"""用冻结 baseline/LoRA 的配对 rho 扫描生成场景门控监督。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stylelora.data.preference_lora_dataset import PreferenceLoRADataset, preference_lora_collate
from stylelora.lora.evaluation.rollout import rollout_with_rho
from stylelora.lora.model.checkpoint import load_adapter_checkpoint
from stylelora.lora.model.style_lora_planner import StyleLoRAPlanner
from stylelora.lora.runtime import load_plain_baseline, prepare_diffusion_batch
from stylelora.lora.safety.trajectory_acceptance import cached_hard_feasibility_mask
from stylelora.paths import ensure_repo_on_path
from stylelora.model.conditional_lora_router import load_conditional_router_checkpoint
from stylelora.training.preference_lora import load_frozen_cspq


SCENE_NAMES = ("straight_free_drive", "straight_car_follow")


def _ego_prediction(output: dict) -> torch.Tensor:
    prediction = output.get("prediction", output.get("x_start"))
    if prediction is None or prediction.ndim not in (3, 4):
        raise RuntimeError(
            f"轨迹输出缺失或形状异常: {None if prediction is None else tuple(prediction.shape)}"
        )
    return prediction[:, 0] if prediction.ndim == 4 else prediction


def _trajectory_tokens(ego: torch.Tensor) -> torch.Tensor:
    pos = ego[..., :2]
    delta = torch.zeros_like(pos)
    if pos.shape[1] > 1:
        delta[:, 1:] = pos[:, 1:] - pos[:, :-1]
    heading = F.normalize(ego[..., 2:4], dim=-1, eps=1e-6)
    return torch.cat((pos, delta, heading), dim=-1)


def _ade_fde_per_sample(pred: torch.Tensor, expert: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    distance = torch.linalg.vector_norm(pred[..., :2] - expert[..., :2], dim=-1)
    return distance.mean(dim=1), distance[:, -1]


def _finite_diff(value: torch.Tensor, order: int) -> torch.Tensor:
    for _ in range(order):
        value = value[:, 1:] - value[:, :-1]
    return value


def _dynamics_per_sample(
    pred: torch.Tensor,
    expert: torch.Tensor,
    current_xy: torch.Tensor,
    *,
    dt: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回逐样本加速度与 jerk 的归一化专家误差。"""
    pred_xy = torch.cat((current_xy, pred[..., :2]), dim=1)
    expert_xy = torch.cat((current_xy, expert[..., :2]), dim=1)
    pred_acc = _finite_diff(pred_xy, 2) / (dt ** 2)
    expert_acc = _finite_diff(expert_xy, 2) / (dt ** 2)
    pred_jerk = _finite_diff(pred_xy, 3) / (dt ** 3)
    expert_jerk = _finite_diff(expert_xy, 3) / (dt ** 3)
    acceleration = F.huber_loss(pred_acc / 3.0, expert_acc / 3.0, reduction="none").mean((1, 2))
    jerk = F.huber_loss(pred_jerk / 20.0, expert_jerk / 20.0, reduction="none").mean((1, 2))
    return acceleration, jerk


def _lateral_excess_per_sample(
    adaptive: torch.Tensor,
    baseline: torch.Tensor,
    expert: torch.Tensor,
    current_xy: torch.Tensor,
    *,
    tolerance: float,
    topk_ratio: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """复用 LoRA 训练定义，返回每个样本最差时间点的横向超额误差。"""
    expert_path = torch.cat((current_xy, expert[..., :2]), dim=1)
    expert_step = expert_path[:, 1:] - expert_path[:, :-1]
    step_norm = torch.linalg.vector_norm(expert_step, dim=-1, keepdim=True)
    step_tangent = expert_step / step_norm.clamp_min(eps)
    heading = baseline[..., 2:4]
    heading_norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    heading_tangent = heading / heading_norm.clamp_min(eps)
    fallback = torch.zeros_like(expert_step)
    fallback[..., 0] = 1.0
    tangent = torch.where(
        step_norm > eps,
        step_tangent,
        torch.where(heading_norm > eps, heading_tangent, fallback),
    )
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
    adaptive_error = ((adaptive[..., :2] - expert[..., :2]) * normal).sum(dim=-1).abs()
    baseline_error = ((baseline[..., :2] - expert[..., :2]) * normal).sum(dim=-1).abs()
    excess = F.relu(adaptive_error - baseline_error - tolerance).square()
    topk = max(1, math.ceil(excess.shape[1] * topk_ratio))
    return torch.topk(excess, topk, dim=1).values.mean(dim=1)


def _minimum_neighbor_distance(
    ego: torch.Tensor,
    neighbors: torch.Tensor,
    neighbor_invalid: torch.Tensor,
) -> torch.Tensor:
    horizon = min(ego.shape[1], neighbors.shape[2])
    distance = torch.linalg.vector_norm(
        ego[:, None, :horizon, :2] - neighbors[:, :, :horizon, :2], dim=-1
    )
    invalid = neighbor_invalid[:, :, :horizon].bool()
    distance = distance.masked_fill(invalid, float("inf"))
    return distance.flatten(1).min(dim=1).values


def _baseline_relative_soft_metrics(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    current_xy: torch.Tensor,
    *,
    dt: float = 0.1,
) -> dict[str, torch.Tensor]:
    """计算只用于门控建议的 baseline 相对软质量指标。"""
    candidate_xy = torch.cat((current_xy, candidate[..., :2]), dim=1)
    baseline_xy = torch.cat((current_xy, baseline[..., :2]), dim=1)

    def _physical_metrics(xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        speed = torch.linalg.vector_norm(torch.diff(xy, dim=1), dim=-1) / dt
        acceleration = torch.diff(speed, dim=1) / dt
        jerk = torch.diff(acceleration, dim=1) / dt
        mean_acceleration = (
            acceleration.abs().mean(dim=1)
            if acceleration.shape[1]
            else acceleration.new_zeros(acceleration.shape[0])
        )
        mean_jerk = (
            jerk.abs().mean(dim=1)
            if jerk.shape[1]
            else jerk.new_zeros(jerk.shape[0])
        )
        progress = torch.linalg.vector_norm(torch.diff(xy, dim=1), dim=-1).sum(dim=1)
        return mean_acceleration, mean_jerk, progress

    candidate_acceleration, candidate_jerk, candidate_progress = _physical_metrics(candidate_xy)
    baseline_acceleration, baseline_jerk, baseline_progress = _physical_metrics(baseline_xy)
    return {
        "mean_accel_degradation": F.relu(candidate_acceleration - baseline_acceleration),
        "mean_jerk_degradation": F.relu(candidate_jerk - baseline_jerk),
        "progress_loss": F.relu(baseline_progress - candidate_progress),
        "mean_path_deviation": torch.linalg.vector_norm(
            candidate[..., :2] - baseline[..., :2], dim=-1
        ).mean(dim=1),
    }


def _normalized_positive_excess(
    value: torch.Tensor,
    baseline: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    return F.relu(value - baseline) / max(float(tolerance), 1e-8)


def _candidate_acceptance(
    *,
    sign: float,
    s_value: torch.Tensor,
    s_zero: torch.Tensor,
    ade: torch.Tensor,
    fde: torch.Tensor,
    acceleration: torch.Tensor,
    jerk: torch.Tensor,
    lateral: torch.Tensor,
    min_distance: torch.Tensor,
    hard_physical: torch.Tensor,
    relative_metrics: dict[str, torch.Tensor],
    baseline_metrics: dict[str, torch.Tensor],
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回最终通过、硬条件通过和软质量代价。"""
    direction_ok = (s_value - s_zero) * sign >= -args.style_direction_epsilon
    required_distance = torch.maximum(
        baseline_metrics["min_distance"] - args.neighbor_distance_tolerance,
        torch.full_like(min_distance, args.min_neighbor_distance),
    )
    hard_pass = (
        direction_ok
        & hard_physical
        & (lateral <= args.lateral_loss_max)
        & (min_distance >= required_distance)
    )

    # Low 的合理减速会自然增大模仿误差和进度损失，因此只扩大对应软预算。
    imitation_multiplier = args.low_imitation_tolerance_multiplier if sign < 0 else 1.0
    progress_multiplier = args.low_progress_tolerance_multiplier if sign < 0 else 1.0
    jerk_multiplier = args.high_jerk_tolerance_multiplier if sign > 0 else 1.0

    imitation_cost = 0.5 * (
        _normalized_positive_excess(
            ade, baseline_metrics["ade"], args.ade_degradation * imitation_multiplier
        )
        + _normalized_positive_excess(
            fde, baseline_metrics["fde"], args.fde_degradation * imitation_multiplier
        )
    )
    comfort_cost = 0.25 * (
        _normalized_positive_excess(
            acceleration,
            baseline_metrics["acceleration"],
            args.acceleration_degradation,
        )
        + _normalized_positive_excess(
            jerk,
            baseline_metrics["jerk"],
            args.jerk_degradation * jerk_multiplier,
        )
        + relative_metrics["mean_accel_degradation"]
        / max(float(args.max_mean_accel_degradation), 1e-8)
        + relative_metrics["mean_jerk_degradation"]
        / max(float(args.max_mean_jerk_degradation) * jerk_multiplier, 1e-8)
    )
    geometry_cost = 0.5 * (
        lateral / max(float(args.lateral_soft_loss_scale), 1e-8)
        + relative_metrics["mean_path_deviation"]
        / max(float(args.max_mean_path_deviation_m), 1e-8)
    )
    progress_cost = relative_metrics["progress_loss"] / max(
        float(args.max_progress_loss_m) * progress_multiplier, 1e-8
    )
    soft_cost = 0.25 * (imitation_cost + comfort_cost + geometry_cost + progress_cost)
    soft_cost = torch.nan_to_num(
        soft_cost,
        nan=float(args.soft_budget) + 1.0,
        posinf=float(args.soft_budget) + 1.0,
        neginf=0.0,
    )
    passed = hard_pass & (soft_cost <= args.soft_budget)
    return passed.detach(), hard_pass.detach(), soft_cost.detach()


def main() -> None:
    ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="扫描固定 LoRA，生成连续场景门控强度上限标签")
    parser.add_argument("--args-file", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--adapter-high", required=True)
    parser.add_argument("--adapter-low", required=True)
    parser.add_argument(
        "--conditional-router-checkpoint",
        default=None,
        help="可选；为新动态 LoRA 重新生成门控标签，省略时保持旧逻辑。",
    )
    parser.add_argument("--cspq-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--latent-bank", required=True)
    parser.add_argument("--latent-bank-index", required=True)
    parser.add_argument("--feature-npy", required=True)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rho-magnitudes", default="0.05,0.1,0.2,0.4,0.7,1.0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--style-direction-epsilon", type=float, default=1e-3)
    parser.add_argument("--ade-degradation", type=float, default=0.5)
    parser.add_argument("--fde-degradation", type=float, default=1.0)
    parser.add_argument("--acceleration-degradation", type=float, default=0.1)
    parser.add_argument("--jerk-degradation", type=float, default=0.1)
    parser.add_argument("--lateral-tolerance", type=float, default=0.3)
    parser.add_argument("--lateral-topk-ratio", type=float, default=0.2)
    parser.add_argument(
        "--lateral-loss-max", type=float, default=0.25,
        help="严重横向超额误差的硬否决阈值",
    )
    parser.add_argument(
        "--lateral-soft-loss-scale", type=float, default=0.01,
        help="轻微横向超额误差进入软质量预算时的归一化尺度",
    )
    parser.add_argument("--min-neighbor-distance", type=float, default=1.5)
    parser.add_argument("--neighbor-distance-tolerance", type=float, default=0.5)
    parser.add_argument("--max-mean-accel-degradation", type=float, default=0.5)
    parser.add_argument("--max-mean-jerk-degradation", type=float, default=2.0)
    parser.add_argument("--max-progress-loss-m", type=float, default=2.0)
    parser.add_argument("--max-mean-path-deviation-m", type=float, default=1.5)
    parser.add_argument("--soft-budget", type=float, default=1.0)
    parser.add_argument("--low-imitation-tolerance-multiplier", type=float, default=2.0)
    parser.add_argument("--low-progress-tolerance-multiplier", type=float, default=2.0)
    parser.add_argument("--high-jerk-tolerance-multiplier", type=float, default=0.75)
    args = parser.parse_args()

    magnitudes = sorted({float(item) for item in args.rho_magnitudes.split(",") if item.strip()})
    if not magnitudes or magnitudes[0] <= 0 or magnitudes[-1] > 1:
        parser.error("--rho-magnitudes 必须是 (0,1] 内的数值")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("--batch-size 必须为正，--workers 必须非负")
    if not 0 < args.lateral_topk_ratio <= 1:
        parser.error("--lateral-topk-ratio 必须位于 (0,1]")

    positive_values = (
        args.lateral_loss_max,
        args.lateral_soft_loss_scale,
        args.soft_budget,
        args.low_imitation_tolerance_multiplier,
        args.low_progress_tolerance_multiplier,
        args.high_jerk_tolerance_multiplier,
    )
    if any(value <= 0 for value in positive_values):
        parser.error("门控硬/软预算及方向倍率必须为正数")

    device = torch.device(args.device)
    print("[阶段 1/4] 正在读取并对齐 manifest、latent bank 与场景特征...", flush=True)
    dataset = PreferenceLoRADataset(
        args.manifest, args.cache_root, args.latent_bank, args.latent_bank_index,
        args.feature_npy, args.feature_index, direction="high", rank_low=0.0, rank_high=1.0,
    )
    if len(dataset) == 0:
        raise ValueError("门控标签数据集为空")
    print(
        f"[阶段 1/4] 数据对齐完成：samples={len(dataset)}, missing={dataset.missing}",
        flush=True,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", collate_fn=preference_lora_collate,
    )

    print("[阶段 2/4] 正在加载冻结 baseline 到 GPU...", flush=True)
    baseline, config = load_plain_baseline(args.args_file, args.baseline_checkpoint, args.device)
    print("[阶段 2/4] baseline 加载完成；正在注入并加载 High/Low LoRA...", flush=True)
    planner = StyleLoRAPlanner(baseline, rank=args.rank, alpha=args.alpha, dropout=0.0).to(device)
    for checkpoint in (args.adapter_high, args.adapter_low):
        load_adapter_checkpoint(
            checkpoint, planner, baseline_checkpoint=args.baseline_checkpoint,
            normalization_file=config.normalization_file_path, strict_hash=True,
        )
    if args.conditional_router_checkpoint:
        router, prototypes, _ = load_conditional_router_checkpoint(
            args.conditional_router_checkpoint, device
        )
        planner.attach_conditional_router(router, prototypes, enabled=True, trainable=False)
    planner.eval()
    print(
        "[阶段 2/4] High/Low LoRA 加载完成；"
        f"conditional_router={'on' if args.conditional_router_checkpoint else 'off'}。",
        flush=True,
    )
    print("[阶段 3/4] 正在加载冻结 CSPQ 偏好编码器...", flush=True)
    cspq = load_frozen_cspq(args.cspq_checkpoint, args.device)
    print("[阶段 3/4] CSPQ 加载完成。", flush=True)

    rows: list[dict] = []
    acceptance_audit = {
        direction: {
            magnitude: {"count": 0, "hard_pass": 0, "final_pass": 0, "soft_cost_sum": 0.0}
            for magnitude in magnitudes
        }
        for direction in ("low", "high")
    }
    total_batches = len(loader)
    print(
        f"[阶段 4/4] 开始 rho 扫描：batches={total_batches}, "
        f"每个 batch 共 {1 + 2 * len(magnitudes)} 次 rollout。",
        flush=True,
    )
    loader_iterator = iter(loader)
    for batch_index in range(total_batches):
        print(f"[batch {batch_index + 1}/{total_batches}] 正在从 cache 读取数据...", flush=True)
        batch = next(loader_iterator)
        print(f"[batch {batch_index + 1}/{total_batches}] cache 读取完成，运行 rho=0...", flush=True)
        tensors = {key: value.to(device) for key, value in batch["tensors"].items()}
        metadata = [{"scene_type": SCENE_NAMES[int(scene_id)]}
                    for scene_id in batch["scene_id"].tolist()]
        inputs, futures = prepare_diffusion_batch(
            {"tensors": tensors, "metadata": metadata}, device,
            config.observation_normalizer, return_style_context=False,
        )
        expert, neighbors, neighbor_invalid = futures
        current_xy = tensors["ego_current_state"][:, None, :2]
        seed = args.seed + batch_index
        with torch.no_grad():
            zero_output, _ = rollout_with_rho(planner, inputs, 0.0, seed=seed)
            zero_ego = _ego_prediction(zero_output)
            s_zero = cspq(_trajectory_tokens(zero_ego), batch["h_c"].to(device))["s"].squeeze(-1)
            zero_ade, zero_fde = _ade_fde_per_sample(zero_ego, expert)
            zero_acc, zero_jerk = _dynamics_per_sample(zero_ego, expert, current_xy)
            zero_distance = _minimum_neighbor_distance(zero_ego, neighbors, neighbor_invalid)
        baseline_metrics = {
            "ade": zero_ade,
            "fde": zero_fde,
            "acceleration": zero_acc,
            "jerk": zero_jerk,
            "min_distance": zero_distance,
        }
        print(f"[batch {batch_index + 1}/{total_batches}] rho=0 完成。", flush=True)

        batch_size = zero_ego.shape[0]
        caps = {"low": torch.zeros(batch_size, device=device),
                "high": torch.zeros(batch_size, device=device)}
        prefix_open = {"low": torch.ones(batch_size, dtype=torch.bool, device=device),
                       "high": torch.ones(batch_size, dtype=torch.bool, device=device)}
        seen_failure = {"low": torch.zeros(batch_size, dtype=torch.bool, device=device),
                        "high": torch.zeros(batch_size, dtype=torch.bool, device=device)}
        nonmonotonic = {"low": torch.zeros(batch_size, device=device),
                        "high": torch.zeros(batch_size, device=device)}
        accepted_soft_cost = {"low": torch.zeros(batch_size, device=device),
                              "high": torch.zeros(batch_size, device=device)}
        pass_patterns = {"low": [[] for _ in range(batch_size)],
                         "high": [[] for _ in range(batch_size)]}

        for direction, sign in (("low", -1.0), ("high", 1.0)):
            for magnitude in magnitudes:
                rho = sign * magnitude
                print(
                    f"[batch {batch_index + 1}/{total_batches}] "
                    f"运行 rho={rho:+.2f}...",
                    flush=True,
                )
                with torch.no_grad():
                    output, _ = rollout_with_rho(planner, inputs, rho, seed=seed)
                    ego = _ego_prediction(output)
                    s_value = cspq(_trajectory_tokens(ego), batch["h_c"].to(device))["s"].squeeze(-1)
                    ade, fde = _ade_fde_per_sample(ego, expert)
                    acceleration, jerk = _dynamics_per_sample(ego, expert, current_xy)
                    lateral = _lateral_excess_per_sample(
                        ego, zero_ego, expert, current_xy,
                        tolerance=args.lateral_tolerance, topk_ratio=args.lateral_topk_ratio,
                    )
                    min_distance = _minimum_neighbor_distance(ego, neighbors, neighbor_invalid)
                    relative_metrics = _baseline_relative_soft_metrics(
                        ego, zero_ego, current_xy
                    )
                    hard_physical = cached_hard_feasibility_mask(ego, current_xy)
                    passed, hard_pass, soft_cost = _candidate_acceptance(
                        sign=sign, s_value=s_value, s_zero=s_zero,
                        ade=ade, fde=fde, acceleration=acceleration, jerk=jerk,
                        lateral=lateral, min_distance=min_distance,
                        hard_physical=hard_physical,
                        relative_metrics=relative_metrics,
                        baseline_metrics=baseline_metrics, args=args,
                    )
                    # 硬安全条件负责否决，轻微的轨迹质量退化只进入软预算。
                audit = acceptance_audit[direction][magnitude]
                audit["count"] += int(passed.numel())
                audit["hard_pass"] += int(hard_pass.sum().item())
                audit["final_pass"] += int(passed.sum().item())
                audit["soft_cost_sum"] += float(soft_cost.sum().item())
                for index, value in enumerate(passed.detach().cpu().tolist()):
                    pass_patterns[direction][index].append(bool(value))
                nonmonotonic[direction] += (seen_failure[direction] & passed).float()
                prefix_open[direction] &= passed
                caps[direction] = torch.where(
                    prefix_open[direction], torch.full_like(caps[direction], magnitude), caps[direction]
                )
                accepted_soft_cost[direction] = torch.where(
                    prefix_open[direction], soft_cost, accepted_soft_cost[direction]
                )
                seen_failure[direction] |= ~passed

        for index, key in enumerate(batch["key"]):
            inconsistency = (
                float(nonmonotonic["low"][index]) + float(nonmonotonic["high"][index])
            ) / (2.0 * len(magnitudes))
            mean_soft_cost = 0.5 * (
                float(accepted_soft_cost["low"][index])
                + float(accepted_soft_cost["high"][index])
            )
            soft_quality = 1.0 / (1.0 + mean_soft_cost)
            confidence = (
                float(batch["confidence"][index])
                * max(0.0, 1.0 - inconsistency)
                * soft_quality
            )
            baseline_row = {}
            for name, value in baseline_metrics.items():
                number = float(value[index])
                baseline_row[name] = number if math.isfinite(number) else None
            rows.append({
                "key": str(key),
                "scene_type": metadata[index]["scene_type"],
                "c_low_target": float(caps["low"][index]),
                "c_high_target": float(caps["high"][index]),
                "target_confidence": confidence,
                "rho_magnitudes": magnitudes,
                "low_pass": pass_patterns["low"][index],
                "high_pass": pass_patterns["high"][index],
                "low_accepted_soft_cost": float(accepted_soft_cost["low"][index]),
                "high_accepted_soft_cost": float(accepted_soft_cost["high"][index]),
                "baseline": baseline_row,
            })
        print(
            f"[batch {batch_index + 1}/{total_batches}] 完成，累计标签 {len(rows)} 条。",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    low = np.asarray([row["c_low_target"] for row in rows], dtype=np.float64)
    high = np.asarray([row["c_high_target"] for row in rows], dtype=np.float64)
    summary = {
        "count": len(rows),
        "rho_magnitudes": magnitudes,
        "low": {"mean": float(low.mean()), "std": float(low.std()), "zero_rate": float((low == 0).mean())},
        "high": {"mean": float(high.mean()), "std": float(high.std()), "zero_rate": float((high == 0).mean())},
        "has_learnable_variation": bool(low.std() >= 0.02 or high.std() >= 0.02),
        "acceptance_by_magnitude": {
            direction: {
                str(magnitude): {
                    "hard_pass_rate": values["hard_pass"] / max(values["count"], 1),
                    "final_pass_rate": values["final_pass"] / max(values["count"], 1),
                    "mean_soft_cost": values["soft_cost_sum"] / max(values["count"], 1),
                }
                for magnitude, values in by_magnitude.items()
            }
            for direction, by_magnitude in acceptance_audit.items()
        },
        "settings": vars(args),
    }
    report_path = output.with_suffix(".report.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"门控标签 -> {output}")


if __name__ == "__main__":
    main()

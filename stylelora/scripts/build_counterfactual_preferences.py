"""从同一场景的多强度策略响应构造反事实轨迹偏好对。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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
from stylelora.lora.safety.trajectory_acceptance import cached_relative_feasibility_mask
from stylelora.model.conditional_lora_router import load_conditional_router_checkpoint
from stylelora.training.preference_lora import load_frozen_cspq


SCENE_NAMES = ("straight_free_drive", "straight_car_follow")


def _ego_prediction(output: dict) -> torch.Tensor:
    prediction = output.get("prediction", output.get("x_start"))
    if prediction is None or prediction.ndim not in (3, 4):
        raise RuntimeError(
            f"轨迹输出缺失或形状异常：{None if prediction is None else tuple(prediction.shape)}"
        )
    return prediction[:, 0] if prediction.ndim == 4 else prediction


def _trajectory_tokens(ego: torch.Tensor) -> torch.Tensor:
    position = ego[..., :2]
    delta = torch.zeros_like(position)
    if position.shape[1] > 1:
        delta[:, 1:] = position[:, 1:] - position[:, :-1]
    heading = F.normalize(ego[..., 2:4], dim=-1, eps=1e-6)
    return torch.cat((position, delta, heading), dim=-1)


def _minimum_neighbor_distance(
    ego: torch.Tensor,
    neighbors: torch.Tensor,
    neighbor_invalid: torch.Tensor,
) -> torch.Tensor:
    horizon = min(ego.shape[1], neighbors.shape[2])
    if neighbors.shape[1] == 0 or horizon == 0:
        return ego.new_full((ego.shape[0],), float("inf"))
    distance = torch.linalg.vector_norm(
        ego[:, None, :horizon, :2] - neighbors[:, :, :horizon, :2], dim=-1
    )
    distance = distance.masked_fill(neighbor_invalid[:, :, :horizon].bool(), float("inf"))
    return distance.flatten(1).min(dim=1).values


def _target_magnitude(key: str, direction: str, magnitudes: list[float], seed: int) -> float:
    """按样本稳定散列选择训练强度，使整个数据集覆盖连续控制区间。"""
    payload = f"{seed}:{direction}:{key}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % len(magnitudes)
    return magnitudes[index]


def _continuous_prefix_masks(
    acceptable_by_magnitude: dict[float, torch.Tensor],
    magnitudes: list[float],
) -> dict[float, torch.Tensor]:
    """强度上限只保留从零开始连续通过检查的可行前缀。"""
    if not magnitudes:
        raise ValueError("magnitudes 不能为空")
    ordered = sorted(magnitudes)
    first = acceptable_by_magnitude[ordered[0]].bool()
    prefix = torch.ones_like(first, dtype=torch.bool)
    result = {}
    for magnitude in ordered:
        current = acceptable_by_magnitude[magnitude].bool()
        if current.shape != first.shape:
            raise ValueError("不同强度的可行性掩码形状不一致")
        prefix = prefix & current
        result[magnitude] = prefix.clone()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build same-scene counterfactual trajectory preferences.")
    parser.add_argument("--args-file", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--adapter-high", required=True)
    parser.add_argument("--adapter-low", required=True)
    parser.add_argument("--conditional-router-checkpoint", required=True)
    parser.add_argument("--cspq-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--latent-bank", required=True)
    parser.add_argument("--latent-bank-index", required=True)
    parser.add_argument("--feature-npy", required=True)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--rho-magnitudes", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-style-delta", type=float, default=1e-3)
    parser.add_argument("--style-confidence-scale", type=float, default=0.02)
    parser.add_argument("--min-neighbor-distance", type=float, default=1.5)
    parser.add_argument("--neighbor-distance-tolerance", type=float, default=0.5)
    parser.add_argument("--max-mean-accel-degradation", type=float, default=0.5)
    parser.add_argument("--max-mean-jerk-degradation", type=float, default=2.0)
    parser.add_argument("--max-progress-loss-m", type=float, default=2.0)
    parser.add_argument(
        "--max-mean-lateral-deviation-m",
        "--max-mean-path-deviation-m",
        dest="max_mean_lateral_deviation_m",
        type=float,
        default=1.5,
    )
    args = parser.parse_args()

    magnitudes = sorted({float(value) for value in args.rho_magnitudes.split(",") if value.strip()})
    if not magnitudes or magnitudes[0] <= 0 or magnitudes[-1] > 1:
        parser.error("--rho-magnitudes 必须全部位于 (0,1]")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch-size 必须为正，workers 不能为负")
    if args.min_style_delta < 0 or args.style_confidence_scale <= 0:
        parser.error("风格间隔不能为负，置信度尺度必须为正")

    device = torch.device(args.device)
    print("[阶段 1/4] 对齐原始偏好、场景特征和缓存……", flush=True)
    dataset = PreferenceLoRADataset(
        args.manifest,
        args.cache_root,
        args.latent_bank,
        args.latent_bank_index,
        args.feature_npy,
        args.feature_index,
        direction="counterfactual",
        rank_low=0.0,
        rank_high=1.0,
    )
    if not len(dataset):
        raise ValueError("反事实偏好数据源为空")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=preference_lora_collate,
    )
    print(f"[数据] samples={len(dataset)}, missing={dataset.missing}", flush=True)

    print("[阶段 2/4] 加载冻结策略、条件路由和风格编码器……", flush=True)
    baseline, config = load_plain_baseline(args.args_file, args.baseline_checkpoint, args.device)
    planner = StyleLoRAPlanner(baseline, rank=args.rank, alpha=args.alpha, dropout=0.0).to(device)
    for checkpoint in (args.adapter_high, args.adapter_low):
        load_adapter_checkpoint(
            checkpoint,
            planner,
            baseline_checkpoint=args.baseline_checkpoint,
            normalization_file=config.normalization_file_path,
            strict_hash=True,
        )
    router, prototypes, _ = load_conditional_router_checkpoint(
        args.conditional_router_checkpoint, device
    )
    if router.use_diffusion_time:
        raise ValueError("反事实偏好对只接受不含扩散时间输入的 v2 条件路由")
    planner.attach_conditional_router(router, prototypes, enabled=True, trainable=False)
    planner.eval()
    cspq = load_frozen_cspq(args.cspq_checkpoint, args.device)

    output_bank = Path(args.output_bank)
    output_index = Path(args.output_index)
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    temporary_bank = output_bank.with_suffix(output_bank.suffix + ".tmp")
    temporary_index = output_index.with_suffix(output_index.suffix + ".tmp")

    print(
        f"[阶段 3/4] 扫描同场景候选：batches={len(loader)}, "
        f"每批 {1 + 2 * len(magnitudes)} 次 rollout……",
        flush=True,
    )
    counts: Counter[str] = Counter()
    confidences: list[float] = []
    pair_row = 0
    trajectory_shape: tuple[int, int] | None = None
    with temporary_bank.open("wb") as bank_handle, temporary_index.open(
        "w", encoding="utf-8", newline="\n"
    ) as index_handle:
        for batch_index, batch in enumerate(loader):
            tensors = {key: value.to(device) for key, value in batch["tensors"].items()}
            metadata = [
                {"scene_type": SCENE_NAMES[int(scene_id)]}
                for scene_id in batch["scene_id"].tolist()
            ]
            inputs, futures = prepare_diffusion_batch(
                {"tensors": tensors, "metadata": metadata},
                device,
                config.observation_normalizer,
                return_style_context=False,
            )
            _, neighbors, neighbor_invalid = futures
            current_xy = tensors["ego_current_state"][:, None, :2]
            seed = args.seed + batch_index
            with torch.no_grad():
                zero_output, _ = rollout_with_rho(planner, inputs, 0.0, seed=seed)
                zero_ego = _ego_prediction(zero_output)
                s_zero = cspq(
                    _trajectory_tokens(zero_ego), batch["h_c"].to(device)
                )["s"].squeeze(-1)
                baseline_distance = _minimum_neighbor_distance(
                    zero_ego, neighbors, neighbor_invalid
                )

            candidates: dict[str, dict[float, dict[str, torch.Tensor]]] = {
                "low": {},
                "high": {},
            }
            for direction, sign in (("low", -1.0), ("high", 1.0)):
                for magnitude in magnitudes:
                    rho = sign * magnitude
                    with torch.no_grad():
                        output, _ = rollout_with_rho(planner, inputs, rho, seed=seed)
                        ego = _ego_prediction(output)
                        s_value = cspq(
                            _trajectory_tokens(ego), batch["h_c"].to(device)
                        )["s"].squeeze(-1)
                        feasible = cached_relative_feasibility_mask(
                            ego,
                            zero_ego,
                            current_xy,
                            max_mean_accel_degradation=args.max_mean_accel_degradation,
                            max_mean_jerk_degradation=args.max_mean_jerk_degradation,
                            max_progress_loss_m=args.max_progress_loss_m,
                            max_mean_lateral_deviation_m=args.max_mean_lateral_deviation_m,
                        )
                        distance = _minimum_neighbor_distance(ego, neighbors, neighbor_invalid)
                        required_distance = torch.maximum(
                            baseline_distance - args.neighbor_distance_tolerance,
                            torch.full_like(distance, args.min_neighbor_distance),
                        )
                        gain = sign * (s_value - s_zero)
                        acceptable = (
                            feasible
                            & (distance >= required_distance)
                            & (gain >= args.min_style_delta)
                        )
                    candidates[direction][magnitude] = {
                        "ego": ego.detach().cpu(),
                        "s": s_value.detach().cpu(),
                        "gain": gain.detach().cpu(),
                        "acceptable": acceptable.detach().cpu(),
                    }

                prefix_masks = _continuous_prefix_masks(
                    {
                        magnitude: candidates[direction][magnitude]["acceptable"]
                        for magnitude in magnitudes
                    },
                    magnitudes,
                )
                for magnitude in magnitudes:
                    acceptable = prefix_masks[magnitude]
                    candidates[direction][magnitude]["acceptable"] = acceptable
                    counts[f"{direction}_rho_{magnitude:.2f}_total"] += int(
                        acceptable.numel()
                    )
                    counts[f"{direction}_rho_{magnitude:.2f}_acceptable"] += int(
                        acceptable.sum().item()
                    )

            zero_cpu = zero_ego.detach().cpu()
            s_zero_cpu = s_zero.detach().cpu()
            for sample_index, key in enumerate(batch["key"]):
                for direction, sign in (("low", -1.0), ("high", 1.0)):
                    target_magnitude = _target_magnitude(
                        str(key), direction, magnitudes, args.seed
                    )
                    feasible_magnitudes = [
                        magnitude
                        for magnitude in magnitudes
                        if magnitude <= target_magnitude
                        and bool(candidates[direction][magnitude]["acceptable"][sample_index])
                    ]
                    counts[f"{direction}_requested_total"] += 1
                    if not feasible_magnitudes:
                        counts[f"{direction}_requested_without_feasible_style"] += 1
                        continue

                    # 只保留当前请求范围内可实现的最大风格强度；不可行请求不参与策略梯度。
                    preferred_magnitude = max(feasible_magnitudes)
                    preferred = candidates[direction][preferred_magnitude]
                    preferred_ego = preferred["ego"][sample_index]
                    preferred_rho = sign * preferred_magnitude
                    preferred_s = float(preferred["s"][sample_index])
                    rejected_ego = zero_cpu[sample_index]
                    rejected_s = float(s_zero_cpu[sample_index])
                    counts[f"{direction}_feasible_style"] += 1

                    pair = torch.stack((preferred_ego, rejected_ego)).float().numpy()
                    if not np.isfinite(pair).all():
                        raise FloatingPointError(
                            f"反事实轨迹出现 NaN/Inf：key={key}, direction={direction}"
                        )
                    if trajectory_shape is None:
                        trajectory_shape = tuple(int(value) for value in pair.shape[1:])
                    if tuple(pair.shape[1:]) != trajectory_shape or pair.shape[0] != 2:
                        raise RuntimeError("生成的反事实轨迹形状不一致")
                    bank_handle.write(np.ascontiguousarray(pair, dtype=np.float32).tobytes())

                    style_gap = abs(preferred_s - rejected_s)
                    confidence = float(batch["confidence"][sample_index])
                    confidence *= min(1.0, style_gap / args.style_confidence_scale)
                    confidence = max(1e-4, min(1.0, confidence))
                    row = {
                        "key": str(key),
                        "pair_row": pair_row,
                        "trajectory_shape": list(trajectory_shape),
                        "scene_type": metadata[sample_index]["scene_type"],
                        "direction": direction,
                        "requested_rho": preferred_rho,
                        "preferred_s": preferred_s,
                        "rejected_s": rejected_s,
                        "pair_confidence": confidence,
                    }
                    index_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    confidences.append(confidence)
                    counts["style"] += 1
                    counts[direction] += 1
                    pair_row += 1

            if batch_index == 0 or (batch_index + 1) % 25 == 0 or batch_index + 1 == len(loader):
                print(
                    f"[进度] {batch_index + 1}/{len(loader)} "
                    f"({(batch_index + 1) / len(loader):.1%}), pairs={pair_row}",
                    flush=True,
                )

    if not pair_row or trajectory_shape is None:
        raise RuntimeError("没有生成反事实偏好对")
    temporary_bank.replace(output_bank)
    temporary_index.replace(output_index)
    expected_bytes = pair_row * 2 * trajectory_shape[0] * trajectory_shape[1] * 4
    if output_bank.stat().st_size != expected_bytes:
        raise RuntimeError("反事实轨迹库写入不完整")

    summary = {
        "method": "same_scene_feasible_style_pairs_v2",
        "source_samples": len(dataset),
        "pair_count": pair_row,
        "trajectory_shape": [pair_row, 2, *trajectory_shape],
        "dtype": "float32",
        "counts": dict(sorted(counts.items())),
        "pair_confidence": {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
        },
        "settings": vars(args),
    }
    report_path = output_index.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[阶段 4/4] 反事实偏好对构造完成", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[done] bank={output_bank}", flush=True)
    print(f"[done] index={output_index}", flush=True)
    print(f"[done] report={report_path}", flush=True)


if __name__ == "__main__":
    main()

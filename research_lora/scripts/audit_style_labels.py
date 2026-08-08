"""Audit whether manifest style labels separate expert trajectories in proxy metric space.

数据标签审计（只读）：不加载 LoRA、不执行模型推理。脚本直接从缓存读取专家真值
``ego_future_gt``，复用开放环评测的 ``scene_style_vector``，检查同一场景中
aggressive / conservative 标签对应的专家轨迹是否具有可分辨的三轴分布。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from research_lora.data.dataset import StyleManifestDataset, style_collate
from research_lora.evaluation.reports import write_json
from research_lora.evaluation.style_metrics import AXES_BY_SCENE, mmd_rbf, per_axis_wasserstein, scene_style_vector


SCENES = ("straight_free_drive", "straight_car_follow")
STYLES = ("aggressive", "conservative", "normal")


def _vector_for_expert(tensors: Mapping[str, torch.Tensor], index: int, scene: str) -> tuple[torch.Tensor, torch.Tensor]:
    """用单条专家真值轨迹计算与开放环评测完全相同的代理风格向量。"""
    return scene_style_vector(
        scene=scene,
        ego_future=tensors["ego_future_gt"][index],
        ego_current=tensors["ego_current_state"][index],
        neighbors_past=tensors["neighbor_agents_past"][index],
        neighbors_future=tensors["neighbors_future_gt"][index],
        route_limits=tensors["route_lanes_speed_limit"][index],
        route_has_limits=tensors["route_lanes_has_speed_limit"][index],
        lane_limits=tensors["lanes_speed_limit"][index],
        lane_has_limits=tensors["lanes_has_speed_limit"][index],
    )


def _axis_summary(vectors: torch.Tensor, axis_names: Sequence[str]) -> dict[str, dict[str, float]]:
    """返回每个代理轴的样本数、均值、标准差和 P5/P50/P95 分位数。"""
    if vectors.ndim != 2 or vectors.shape[1] != len(axis_names):
        raise ValueError("vectors must have shape [samples, axes]")
    quantiles = torch.quantile(vectors, torch.tensor((0.05, 0.50, 0.95)), dim=0)
    means = vectors.mean(dim=0)
    # unbiased=False 使只有一个有效样本的小组也能得到定义明确的标准差 0。
    stds = vectors.std(dim=0, unbiased=False)
    return {
        name: {
            "count": int(vectors.shape[0]),
            "mean": float(means[index]),
            "std": float(stds[index]),
            "p05": float(quantiles[0, index]),
            "p50": float(quantiles[1, index]),
            "p95": float(quantiles[2, index]),
        }
        for index, name in enumerate(axis_names)
    }


def _interval_overlap(a: torch.Tensor, b: torch.Tensor, axis_names: Sequence[str]) -> dict[str, dict[str, float | list[float]]]:
    """比较两组各轴 P5–P95 区间，并返回交集相对并集的覆盖比例。"""
    a_low, a_high = torch.quantile(a, 0.05, dim=0), torch.quantile(a, 0.95, dim=0)
    b_low, b_high = torch.quantile(b, 0.05, dim=0), torch.quantile(b, 0.95, dim=0)
    result = {}
    for index, name in enumerate(axis_names):
        lower, upper = torch.maximum(a_low[index], b_low[index]), torch.minimum(a_high[index], b_high[index])
        intersection = (upper - lower).clamp_min(0)
        union = torch.maximum(a_high[index], b_high[index]) - torch.minimum(a_low[index], b_low[index])
        # 区间完全退化时，只有两端相同才视为完全重叠。
        ratio = torch.where(union > 0, intersection / union.clamp_min(1e-12),
                            (intersection == 0).to(union.dtype))
        result[name] = {
            "aggressive_p05_p95": [float(a_low[index]), float(a_high[index])],
            "conservative_p05_p95": [float(b_low[index]), float(b_high[index])],
            "intersection_width": float(intersection),
            "union_width": float(union),
            "overlap_ratio": float(ratio),
        }
    return result


def _mmd_sample(vectors: torch.Tensor, *, maximum: int, generator: torch.Generator) -> torch.Tensor:
    """为 MMD 取确定性随机子样本，避免完整二次距离矩阵占用过多内存。"""
    if maximum <= 0 or vectors.shape[0] <= maximum:
        return vectors
    indices = torch.randperm(vectors.shape[0], generator=generator)[:maximum]
    return vectors[indices]


def _pair_comparison(aggressive: torch.Tensor | None, conservative: torch.Tensor | None, *,
                     axis_names: Sequence[str], mmd_max_samples: int, seed: int) -> dict[str, object]:
    """比较同一场景 aggressive 与 conservative 两个专家分布。"""
    if aggressive is None or conservative is None:
        return {"status": "missing_valid_samples", "mmd_rbf": None, "wasserstein_by_axis": None,
                "p05_p95_interval_overlap_by_axis": None}
    generator = torch.Generator().manual_seed(seed)
    aggressive_mmd = _mmd_sample(aggressive, maximum=mmd_max_samples, generator=generator)
    conservative_mmd = _mmd_sample(conservative, maximum=mmd_max_samples, generator=generator)
    return {
        "status": "ok",
        "aggressive_valid_samples": int(aggressive.shape[0]),
        "conservative_valid_samples": int(conservative.shape[0]),
        "mmd_rbf": mmd_rbf(aggressive_mmd, conservative_mmd),
        "mmd_samples_per_style": {"aggressive": int(aggressive_mmd.shape[0]),
                                  "conservative": int(conservative_mmd.shape[0])},
        "wasserstein_by_axis": per_axis_wasserstein(aggressive, conservative, axis_names),
        "p05_p95_interval_overlap_by_axis": _interval_overlap(aggressive, conservative, axis_names),
    }


def main() -> None:
    """命令行入口：输出专家轨迹代理风格空间中的标签可分性审计 JSON。"""
    parser = argparse.ArgumentParser(description="Audit style-label separability from expert cache trajectories.")
    parser.add_argument("--manifest", required=True, help="待审计的单个 train 或 val manifest JSONL。")
    parser.add_argument("--cache-root", required=True, help="缓存根目录；相对 cache_path 据此解析。")
    parser.add_argument("--output", required=True, help="审计 JSON 输出路径。")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mmd-max-samples", type=int, default=2048,
                        help="每个风格参与 MMD 的最大样本数；轴统计和 Wasserstein 始终使用全部有效样本。")
    parser.add_argument("--seed", type=int, default=17, help="MMD 子样本的固定随机种子。")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0 or args.mmd_max_samples <= 0:
        parser.error("--batch-size 和 --mmd-max-samples 必须为正，--workers 必须非负")

    dataset = StyleManifestDataset(args.manifest, root=args.cache_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=style_collate)
    total, invalid = defaultdict(int), defaultdict(int)
    groups: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)

    # 逐条使用专家真值计算三轴；没有有效前车的 car-follow 记录为无效向量并单独计数。
    for batch in loader:
        tensors = batch["tensors"]
        for index, metadata in enumerate(batch["metadata"]):
            scene, style = metadata["scene_type"], metadata["style"]
            key = (scene, style)
            total[key] += 1
            vector, valid = _vector_for_expert(tensors, index, scene)
            if bool(valid.all()):
                groups[key].append(vector.cpu())
            else:
                invalid[key] += 1

    report_scenes: dict[str, object] = {}
    for scene in SCENES:
        axis_names = AXES_BY_SCENE[scene]
        style_reports: dict[str, object] = {}
        stacked: dict[str, torch.Tensor | None] = {}
        for style in STYLES:
            key = (scene, style)
            vectors = torch.stack(groups[key]) if groups[key] else None
            stacked[style] = vectors
            style_reports[style] = {
                "total_samples": total[key],
                "valid_samples": 0 if vectors is None else int(vectors.shape[0]),
                "invalid_samples": invalid[key],
                "axis_summary": {} if vectors is None else _axis_summary(vectors, axis_names),
            }
        # 同场景内仅比较 aggressive 与 conservative；normal 只保留为标签分布参照。
        report_scenes[scene] = {
            "axis_names": list(axis_names),
            "styles": style_reports,
            "aggressive_vs_conservative": _pair_comparison(
                stacked["aggressive"], stacked["conservative"], axis_names=axis_names,
                mmd_max_samples=args.mmd_max_samples, seed=args.seed,
            ),
        }

    report = {
        "metric_space": "expert_trajectory_proxy_axes",
        "purpose": "Audit label separability before LoRA training; this is not a model evaluation.",
        "manifest": str(Path(args.manifest).resolve()),
        "cache_root": str(Path(args.cache_root).resolve()),
        "mmd_max_samples": args.mmd_max_samples,
        "mmd_seed": args.seed,
        "scenes": report_scenes,
    }
    write_json(args.output, report)
    print(f"Wrote expert style-label audit to {args.output}")


if __name__ == "__main__":
    main()

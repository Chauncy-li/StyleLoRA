"""Build frozen scene/style three-axis prototypes from expert trajectories in train.jsonl.

训练集原型构建（只读）：只从 train manifest 的专家轨迹计算三轴统计量和
``scene × style`` 原型。验证集不会被读取，也不会参与标准化或原型均值。

输出产物（冻结原型 JSON）被训练/评测下游消费，用于：
1. 按场景对专家三轴风格向量做均值/标准差标准化；
2. 生成 aggressive / conservative 的"风格原型"（原始均值 + 标准化均值），
   作为风格条件或参照中心。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from research_lora.data.dataset import StyleManifestDataset, style_collate
from research_lora.data.manifest import manifest_hash
from research_lora.evaluation.reports import write_json
from research_lora.evaluation.style_metrics import AXES_BY_SCENE, scene_style_vector


# 覆盖 style_metrics.AXES_BY_SCENE 的两个场景：直线自由驾驶 / 直线跟车
SCENES = tuple(AXES_BY_SCENE)
# 需要构建原型的风格（normal 只做统计计数，不生成原型 -> aggressive/conservative 才生成原型）
STYLES = ("aggressive", "conservative", "normal")


def _expert_vector(tensors: dict[str, torch.Tensor], index: int, scene: str) -> tuple[torch.Tensor, torch.Tensor]:
    """以与开放环评测一致的三轴定义计算一条专家轨迹的风格向量。

    Args:
        tensors: 一个批次的数据字典（含自车/邻居/限速等张量）。
        index: 该样本在批次中的索引。
        scene: 场景类型（straight_free_drive / straight_car_follow）。

    Returns:
        (axes[3], valid[3])：三维风格向量与各轴有效性掩码。
    """
    neighbors_future = tensors["neighbors_future_gt"][index]
    if "neighbor_agents_future_mask" in tensors:
        # 缓存掩码 True 表示有效帧；失效帧置零后再按既有评测指标寻找前车。
        neighbors_future = neighbors_future.clone()
        neighbors_future[~tensors["neighbor_agents_future_mask"][index].bool()] = 0
    # 与评估脚本共用 scene_style_vector，保证原型(训练侧)与评测(采样侧)三轴定义完全一致
    return scene_style_vector(
        scene=scene,
        ego_future=tensors["ego_future_gt"][index],
        ego_current=tensors["ego_current_state"][index],
        neighbors_past=tensors["neighbor_agents_past"][index],
        neighbors_future=neighbors_future,
        route_limits=tensors["route_lanes_speed_limit"][index],
        route_has_limits=tensors["route_lanes_has_speed_limit"][index],
        lane_limits=tensors["lanes_speed_limit"][index],
        lane_has_limits=tensors["lanes_has_speed_limit"][index],
    )


def _summary(vectors: torch.Tensor) -> dict[str, object]:
    """返回训练集标准化所需的均值、总体标准差和有效样本数。

    Args:
        vectors: 同一场景下堆叠的专家风格向量 [N, 3]。

    Returns:
        {"mean": [3], "std": [3], "valid_samples": N}。
    """
    mean = vectors.mean(dim=0)
    # std 至少为 1e-6，防止某一轴退化时出现除零；真实非退化轴不受影响。
    std = vectors.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {"mean": mean.tolist(), "std": std.tolist(), "valid_samples": int(vectors.shape[0])}


def main() -> None:
    """命令行入口：从 train manifest 构建场景/风格三轴冻结原型。

    流程：
    1. 校验清单只含 train 行（非 train 拆分一律拒绝）；
    2. 遍历数据，对每个专家样本计算三轴风格向量；
       无有效前车的跟车样本（三轴定义不成立）计入 invalid 并跳过；
    3. 按场景聚合全部有效向量，计算标准化统计量（mean/std）；
    4. 对 aggressive/conservative 分别计算原始均值原型与标准化均值原型；
    5. 写出冻结原型 JSON（含 artifact_version、来源 hash、场景报告等）。

    参数：
        --manifest: 仅允许传入 train.jsonl。
        --cache-root: 相对 cache_path 的缓存根目录。
        --output: 输出冻结原型 JSON。
        --batch-size: 批大小（默认 16）。
        --workers: DataLoader 工作进程数（默认 4）。
    """
    parser = argparse.ArgumentParser(description="Build train-only scene/style three-axis prototypes.")
    parser.add_argument("--manifest", required=True, help="仅允许传入 train.jsonl。")
    parser.add_argument("--cache-root", required=True, help="相对 cache_path 的缓存根目录。")
    parser.add_argument("--output", required=True, help="输出冻结原型 JSON。")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    # 基础参数合法性校验
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("--batch-size 必须为正，--workers 必须非负")

    # 1) 构建训练数据集；严格拒绝任何非 train 拆分（只读构建的保证）
    dataset = StyleManifestDataset(args.manifest, root=args.cache_root)
    non_train_splits = sorted({sample.split for sample in dataset.samples if sample.split != "train"})
    if non_train_splits:
        raise ValueError(f"Prototype construction only accepts train rows, found splits={non_train_splits}")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=style_collate)

    # 2) 统计累加器：total=各(场景,风格)总样本数；invalid=无效样本数；
    #    groups=按(场景,风格)收集有效向量；by_scene=按场景收集有效向量（用于统计量）
    total, invalid = defaultdict(int), defaultdict(int)
    groups: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    by_scene: dict[str, list[torch.Tensor]] = defaultdict(list)
    for batch in loader:
        tensors = batch["tensors"]
        for index, metadata in enumerate(batch["metadata"]):
            scene, style = metadata["scene_type"], metadata["style"]
            total[(scene, style)] += 1
            vector, valid = _expert_vector(tensors, index, scene)
            # 无有效前车的跟车样本不具备三轴定义，不能进入统计量或原型。
            if not bool(valid.all()):
                invalid[(scene, style)] += 1
                continue
            # 统一为 float32 后再聚合
            vector = vector.to(dtype=torch.float32)
            groups[(scene, style)].append(vector)
            by_scene[scene].append(vector)

    # 3) 逐场景：计算标准化统计量 + 各风格计数 + aggressive/conservative 原型
    scene_report: dict[str, object] = {}
    for scene in SCENES:
        # 每个场景必须至少有 1 个有效三轴向量，否则无统计量可用
        if not by_scene[scene]:
            raise ValueError(f"No valid expert three-axis vectors for {scene}")
        scene_vectors = torch.stack(by_scene[scene])
        statistics = _summary(scene_vectors)
        mean = torch.tensor(statistics["mean"], dtype=torch.float32)
        std = torch.tensor(statistics["std"], dtype=torch.float32)
        prototypes, style_counts = {}, {}
        for style in STYLES:
            vectors = groups[(scene, style)]
            # 记录该风格的总样本/有效样本/无效样本数（normal 仅计数，不生成原型）
            style_counts[style] = {"total_samples": total[(scene, style)], "valid_samples": len(vectors),
                                   "invalid_samples": invalid[(scene, style)]}
            if style not in ("aggressive", "conservative"):
                continue
            # 目标风格必须有有效样本，否则无法生成原型
            if not vectors:
                raise ValueError(f"No valid train prototypes for {scene} / {style}")
            raw_mean = torch.stack(vectors).mean(dim=0)
            # 原型同时给出原始均值与"用场景统计量标准化后"的均值（供对齐风格条件）
            prototypes[style] = {
                "raw_mean": raw_mean.tolist(),
                "standardized_mean": ((raw_mean - mean) / std).tolist(),
                "valid_samples": len(vectors),
            }
        scene_report[scene] = {
            "axis_names": list(AXES_BY_SCENE[scene]),
            "statistics": statistics,
            "styles": style_counts,
            "prototypes": prototypes,
        }

    # 4) 装配最终报告并写出 JSON
    report = {
        "artifact_version": 1,
        "metric_space": "expert_trajectory_proxy_axes",
        "source": {
            "training_split_only": True,
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_hash": manifest_hash(dataset.samples),
            "cache_root": str(Path(args.cache_root).resolve()),
        },
        "scenes": scene_report,
    }
    write_json(args.output, report)
    print(f"Wrote train-only style prototypes to {args.output}")


if __name__ == "__main__":
    main()
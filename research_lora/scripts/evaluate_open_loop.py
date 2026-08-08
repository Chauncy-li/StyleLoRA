"""Paired-seed DPM-Solver rho sweep in generated-trajectory proxy metric space.

成对种子（paired-seed）的开环评估：在"生成轨迹的代理风格度量空间"里做 rho 扫描。
核心思路：
1. 用"固定随机种子"的方式对不同风格强度 rho 执行 rollout（默认 -1.0 保守 … +1.0 激进，
   也可用 --rhos 自定义网格），保证各 rho 生成的轨迹差异只来自 LoRA 增量，而不是采样随机性；
2. 把每个 rho 生成的轨迹映射为场景特定的三维风格向量（自由驾驶/跟车），
   与参考清单（reference manifest）中的专家风格分布对比（Wasserstein / MMD / 聚类命中率）；
3. 跟车场景的前车有效性会随 rho 生成的 ego 轨迹变化，为避免大强度下悄悄丢弃困难样本，
   每个场景只使用"在所有 rho 下都有效"的样本交集（comparison_valid）做风格分布比较，
   同时保留原始 style_valid 与审计信息用于复查；
4. 每个样本同时记录 ADE/FDE 与邻居轨迹变化量，评估风格化的轨迹质量代价。
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from typing import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Sampler

from research_lora.data.dataset import StyleManifestDataset, style_collate
from research_lora.data.schema import StyleSample
from research_lora.evaluation.reports import write_json
from research_lora.evaluation.rollout import rho_grid, rollout_with_rho
from research_lora.evaluation.style_metrics import ade_fde, aggregate_style_evaluation, scene_style_vector
from research_lora.model.checkpoint import load_adapter_checkpoint
from research_lora.runtime import load_baseline, prepare_diffusion_batch


class _SceneBalancedEvaluationSampler(Sampler[int]):
    """为开放环评测固定抽取 1:1 的 free-drive 与 car-follow 场景。

    该采样器只影响待评测的场景上下文，不按样本原有风格筛选，也不改变完整
    reference manifest 用于构造目标专家分布的方式。
    即评测样本只保证两类场景数量 1:1 平衡（自由驾驶 50% + 跟车 50%）。
    """

    # 评测所需的两个场景
    _SCENES = ("straight_free_drive", "straight_car_follow")

    def __init__(self, samples: Sequence[StyleSample], *, num_samples: int, seed: int) -> None:
        """初始化：校验样本数并预分组。

        Args:
            samples: 评测数据集的样本列表。
            num_samples: 需要抽样的样本总数（必须为正的偶数，才能 1:1 等分两类场景）。
            seed: 采样随机种子（保证每次评测抽取相同的场景分布）。

        Raises:
            ValueError: num_samples 非正或非偶数；或两类场景中有一类没有样本。
        """
        # 开放环 1:1 平衡评测要求总样本数为"正偶数"
        if num_samples <= 0 or num_samples % 2:
            raise ValueError("Open-loop scene-balanced evaluation requires a positive, even max_batches * batch_size")
        self.seed, self.num_samples = int(seed), int(num_samples)
        # 按场景类型分组：每类场景一个索引列表
        self.groups = {scene: [index for index, sample in enumerate(samples) if sample.scene_type == scene]
                       for scene in self._SCENES}
        # 两类场景都必须有样本，否则无法做 1:1 平衡
        if not all(self.groups.values()):
            missing = [scene for scene, indices in self.groups.items() if not indices]
            raise ValueError(f"Open-loop evaluation requires both scenes; missing={missing}")

    def __len__(self) -> int:
        """返回抽样总数。"""
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        """按固定种子从两类场景各抽一半样本，打乱后返回索引迭代器。"""
        # 固定种子保证评测可复现
        rng = random.Random(self.seed)
        per_scene = self.num_samples // 2
        # 每类场景各抽 per_scene 个（带放回）
        selected = [rng.choice(self.groups[scene]) for scene in self._SCENES for _ in range(per_scene)]
        # 打乱顺序后返回
        rng.shuffle(selected)
        return iter(selected)

    def report(self) -> dict[str, int]:
        """返回本次评测抽样配置摘要（种子、总数、各场景数量）。"""
        per_scene = self.num_samples // 2
        return {"seed": self.seed, "samples": self.num_samples,
                "straight_free_drive": per_scene, "straight_car_follow": per_scene}


def _parse_rhos(raw: str | None) -> tuple[float, ...]:
    """解析命令行 rho 网格；未指定时保留原有标准九点扫描。

    Args:
        raw: --rhos 参数值（逗号分隔的浮点数字符串），None 表示使用默认网格。

    Returns:
        校验通过后的 rho 元组（顺序与输入一致）。

    Raises:
        ValueError: 解析失败、包含非有限值/重复值、为空，或未包含 rho=0（基线对照）。
    """
    # 未指定 --rhos：回退到 rollout.rho_grid() 的标准九点网格（-1.0 … +1.0，步长 0.25）
    if raw is None:
        return rho_grid()
    # 按逗号拆分并转换为浮点数
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--rhos 必须是逗号分隔的浮点数，例如 --rhos=-2,-1,0,1,2") from exc
    # 必须非空且所有值都是有限浮点数
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("--rhos 必须包含至少一个有限浮点数")
    # 不允许出现重复的强度点
    if len(set(values)) != len(values):
        raise ValueError("--rhos 不允许包含重复强度")
    # 必须包含 rho=0，作为 baseline（正常/未风格化）的对照
    if 0.0 not in values:
        raise ValueError("--rhos 必须包含 rho=0 作为 baseline 对照")
    return values


def _mark_common_style_validity(records: list[dict], rhos: Sequence[float]) -> dict[str, dict]:
    """标记每个场景在全部 rho 下共同有效的固定风格比较样本集。

    背景：car-follow（跟车）场景的前车有效性会随生成 ego 轨迹与 rho 一起改变。
    若每个 rho 各自丢弃无效样本，那么强度较大时可能"悄悄隐去"难以处理的样本，
    导致风格比较在不同 rho 之间不公平。因此这里对每个场景：
    1. 只保留"在每个 rho 下都有记录、且所有 rho 下 style_valid 均为 True"的样本，
       即跨 rho 有效样本的**交集**；
    2. 把结果写入每条记录的 ``comparison_valid`` 字段（True = 进入风格分布比较）；
    3. 原始 ``style_valid`` 字段保留不动，仅用于审计。

    Args:
        records: 评测记录列表（每条含 scene / sample_id / rho / style_valid）。
        rhos: 本次使用的全部 rho 值（用来判断样本是否在"每个 rho"下都有记录）。

    Returns:
        每个场景的审计字典：
        {scene: {"evaluated_samples": 该场景出现过的样本数,
                 "common_valid_samples": 跨 rho 共同有效的样本数,
                 "by_rho": [{"rho": …, "evaluated": 该 rho 的记录数,
                             "raw_style_valid": 该 rho 下原始 style_valid=True 的记录数}, …]}}
    """
    # 期望出现的 rho 集合，用于判断"样本是否在所有 rho 下都有记录"
    expected_rhos = set(float(rho) for rho in rhos)
    # 三级索引：scene -> sample_id -> rho -> style_valid
    by_scene_sample: dict[str, dict[str, dict[float, bool]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        by_scene_sample[str(record["scene"])][str(record["sample_id"])][float(record["rho"])] = bool(record["style_valid"])

    common_by_scene = {}
    audit = {}
    for scene, samples in by_scene_sample.items():
        # 共同有效样本 = 在所有期望 rho 下都有记录、且全部有效
        common = {sample_id for sample_id, status_by_rho in samples.items()
                  if set(status_by_rho) == expected_rhos and all(status_by_rho.values())}
        common_by_scene[scene] = common
        # 审计信息：每个 rho 的评估记录数与原始有效数，便于发现有效性的漂移
        by_rho = []
        for rho in rhos:
            rows = [record for record in records if record["scene"] == scene and float(record["rho"]) == float(rho)]
            by_rho.append({"rho": float(rho), "evaluated": len(rows),
                           "raw_style_valid": sum(bool(record["style_valid"]) for record in rows)})
        audit[scene] = {"evaluated_samples": len(samples), "common_valid_samples": len(common), "by_rho": by_rho}

    # 把交集结果写回每条记录的 comparison_valid 字段
    for record in records:
        record["comparison_valid"] = str(record["sample_id"]) in common_by_scene[str(record["scene"])]
    return audit


def _vector_for_item(tensors, index: int, scene: str, *, use_expert: bool = False):
    """为单个样本计算"场景风格向量"。

    Args:
        tensors: 一个批次的数据字典（含自车/邻居状态、限速等）。
        index: 该样本在批次中的索引。
        scene: 场景类型（straight_free_drive / straight_car_follow）。
        use_expert: True 时用专家真值（ego_future_gt）计算，用于构建参考分布；
                    False 时用模型预测（prediction 的第 0 个 token 即自车）计算。

    Returns:
        scene_style_vector 的返回值：(axes[3], valid[3])——三维风格向量及其有效性掩码。
    """
    # 选择自车轨迹来源：专家真值 或 模型预测（token 0 为自车）
    ego = tensors["ego_future_gt"][index] if use_expert else tensors["prediction"][index, 0]
    # 把该样本的场景上下文传入场景风格度量函数
    return scene_style_vector(scene=scene, ego_future=ego, ego_current=tensors["ego_current_state"][index],
                              neighbors_past=tensors["neighbor_agents_past"][index],
                              neighbors_future=tensors["neighbors_future_gt"][index],
                              route_limits=tensors["route_lanes_speed_limit"][index],
                              route_has_limits=tensors["route_lanes_has_speed_limit"][index],
                              lane_limits=tensors["lanes_speed_limit"][index],
                              lane_has_limits=tensors["lanes_has_speed_limit"][index])


def _reference_distributions(dataset, batch_size: int, workers: int):
    """从参考数据集中构建"目标专家风格分布"。

    遍历参考数据集，对每个专家样本用专家真值轨迹计算风格向量，
    按 (场景, 风格) 分组堆叠成张量，作为后续对比的专家参考分布。

    Args:
        dataset: 参考数据集（reference manifest）。
        batch_size: 批大小。
        workers: DataLoader 工作进程数。

    Returns:
        {(scene, style): 专家风格向量张量 [样本数, 3]} 的字典；无有效向量的组被丢弃。
    """
    groups = {}
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, collate_fn=style_collate)
    for batch in loader:
        tensors = batch["tensors"]
        # 逐样本处理元信息与张量
        for index, metadata in enumerate(batch["metadata"]):
            # 用专家真值计算风格向量（use_expert=True）
            vector, valid = _vector_for_item(tensors, index, metadata["scene_type"], use_expert=True)
            # 只有三个轴全部有效时才纳入参考分布
            if valid.all():
                groups.setdefault((metadata["scene_type"], metadata["style"]), []).append(vector.cpu())
    # 堆叠为 [样本数, 3] 的张量并返回
    return {key: torch.stack(values) for key, values in groups.items() if values}


def main() -> None:
    """命令行入口：成对种子 rho 扫描的开环风格评估。

    参数说明：
        --args-file: 模型配置文件路径。
        --baseline-checkpoint: 基线权重 checkpoint 路径。
        --aggressive-adapter: 激进风格 LoRA 适配器权重路径。
        --conservative-adapter: 保守风格 LoRA 适配器权重路径。
        --manifest: 评测样本清单。
        --reference-manifest: 参考（专家）样本清单；缺省时复用 --manifest。
        --cache-root: 数据缓存根目录。
        --output: 结果 JSON 输出路径。
        --rank: LoRA 秩（默认 4）。
        --batch-size: 批大小（默认 8）。
        --workers: DataLoader 工作进程数（默认 4）。
        --max-batches: 最多评测批次数（默认 20）。
        --seed: 成对 rollout 与场景抽样的随机种子（默认 17）。
        --rhos: 可选逗号分隔 rho 网格；必须包含 0，缺省时用标准九点网格。
        --device: 计算设备（默认 cuda）。
    """
    parser = argparse.ArgumentParser(description="Paired-seed DPM-Solver rho sweep in generated-trajectory proxy metric space.")
    parser.add_argument("--args-file", required=True); parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--aggressive-adapter", required=True); parser.add_argument("--conservative-adapter", required=True)
    parser.add_argument("--manifest", required=True); parser.add_argument("--reference-manifest", default=None)
    parser.add_argument("--cache-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=4); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=20); parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rhos", default=None, help="可选逗号分隔 rho 网格，必须包含 0。")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    # 解析（或默认）要扫描的 rho 网格
    rhos = _parse_rhos(args.rhos)

    # 1) 加载带 LoRA 注入的风格规划器，并加载激进/保守两个适配器权重；进入评估模式
    planner, config = load_baseline(args.args_file, args.baseline_checkpoint, args.device, rank=args.rank)
    load_adapter_checkpoint(args.aggressive_adapter, planner, baseline_checkpoint=args.baseline_checkpoint, normalization_file=config.normalization_file_path)
    load_adapter_checkpoint(args.conservative_adapter, planner, baseline_checkpoint=args.baseline_checkpoint, normalization_file=config.normalization_file_path)
    planner.eval()

    # 2) 构建评测数据集与参考数据集（缺省参考清单时用评测清单本身）
    eval_dataset = StyleManifestDataset(args.manifest, root=args.cache_root, predicted_neighbor_num=config.predicted_neighbor_num)
    ref_dataset = StyleManifestDataset(args.reference_manifest or args.manifest, root=args.cache_root, predicted_neighbor_num=config.predicted_neighbor_num)

    # 3) 从参考数据集构建专家风格分布，并校验四类组合（2 场景 × 2 风格）都齐备
    references = _reference_distributions(ref_dataset, args.batch_size, args.workers)
    required_refs = {(scene, style) for scene in ("straight_free_drive", "straight_car_follow") for style in ("aggressive", "conservative")}
    missing = required_refs - set(references)
    if missing:
        raise ValueError(f"Reference manifest lacks valid expert trajectories for {sorted(missing)}")

    # 4) 用 1:1 场景平衡采样器构建评测 DataLoader（总样本 = max_batches * batch_size）
    records = []
    evaluation_sampler = _SceneBalancedEvaluationSampler(
        eval_dataset.samples, num_samples=args.max_batches * args.batch_size, seed=args.seed
    )
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, sampler=evaluation_sampler,
                        num_workers=args.workers, collate_fn=style_collate)

    # 5) 逐批次做"成对种子"的 rho 扫描 rollout
    for batch_id, batch in enumerate(loader):
        # 5.1) 数据准备 + 先用 rho=0（正常/基线）做一次 rollout，作为邻居变化量的参照
        inputs, futures = prepare_diffusion_batch(batch, torch.device(args.device), config.observation_normalizer)
        baseline_output, _ = rollout_with_rho(planner, inputs, 0.0, seed=args.seed + batch_id)
        baseline_prediction = baseline_output["prediction"]

        # 5.2) 遍历指定 rho 网格；每个 rho 用"同一批次 + 同一种子" rollout
        for rho in rhos:
            output, seconds = rollout_with_rho(planner, inputs, rho, seed=args.seed + batch_id)
            prediction = output["prediction"]
            # 把预测轨迹并入原始张量，供逐样本计算风格向量
            payload = {**batch["tensors"], "prediction": prediction.detach().cpu()}
            for index, metadata in enumerate(batch["metadata"]):
                # 5.2.1) 计算该样本的风格向量（用模型预测轨迹）
                vector, valid = _vector_for_item(payload, index, metadata["scene_type"])
                # 5.2.2) 目标风格按 rho 符号推导：正=激进，负=保守，0=正常
                target_style = "aggressive" if rho > 0 else "conservative" if rho < 0 else "normal"
                # 5.2.3) 记录一条评测记录：含 rho/场景/sample_id/风格向量/ADE/FDE/邻居轨迹变化量/耗时
                records.append({"rho": rho, "batch": batch_id, "sample_id": f"{batch_id}:{index}",
                                "token": metadata["token"], "scene": metadata["scene_type"],
                                "target_style": target_style, "seconds": seconds / len(batch["metadata"]),
                                "style_vector": vector.cpu().tolist(), "style_valid": bool(valid.all()),
                                **ade_fde(prediction[index, 0], futures[0][index]),
                                "neighbor_prediction_change": float((prediction[index, 1:] - baseline_prediction[index, 1:]).abs().mean())})
        # 5.3) 达到最大批次则停止
        if batch_id + 1 >= args.max_batches: break

    # 6) 固定各场景跨 rho 的共同有效样本集，再做风格分布聚合。
    #    ADE/FDE 与邻居变化仍保留全部 records；只有风格分布比较使用 comparison_valid。
    validity_audit = _mark_common_style_validity(records, rhos)
    # 把 comparison_valid 提升为 style_valid，供 aggregate_style_evaluation 只统计共同有效样本
    comparison_records = [{**record, "style_valid": bool(record["comparison_valid"])} for record in records]
    summaries = aggregate_style_evaluation(comparison_records, references)

    # 7) 写 JSON：保留原始记录、固定比较有效集审计与使用共同有效集得到的风格汇总。
    write_json(args.output, {"rho_grid": rhos, "evaluation_sampling": evaluation_sampler.report(),
                             "style_validity_audit": validity_audit, "records": records,
                             "scene_style_summary": summaries})
    print(f"Wrote {len(records)} paired rollout records and {len(summaries)} scene/style summaries to {args.output}")


if __name__ == "__main__": main()
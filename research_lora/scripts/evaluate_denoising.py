"""Fixed-x_t denoising gate: correct-vs-opposite adapter evidence.

固定 x_t 去噪门控评估：用"正确的 vs 相反的"风格适配器做证据对比。
核心思路：
1. 在固定加噪输入（固定的扩散时间 t=0.5 与固定的噪声）下评估 LoRA 适配器，
   确保所有对比共享完全相同的 x_t，差异只来源于适配器的风格方向；
2. 对每个数据批次，分别用"正确风格 rho"和"相反风格 rho"计算自车目标损失，
   若正确方向损失更小，则证明该风格适配器确实学到了目标风格（门控证据）。
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from research_lora.data.dataset import StyleManifestDataset, style_collate
from research_lora.evaluation.reports import write_json
from research_lora.model.checkpoint import load_adapter_checkpoint
from research_lora.runtime import load_baseline, load_plain_baseline, prepare_diffusion_batch
from research_lora.training.losses import build_noisy_inputs, style_diffusion_loss


def _loss_for_rho(planner, inputs, futures, config, rho, time, noise):
    """在指定风格强度 rho 下计算风格扩散损失。

    Args:
        planner: 风格 LoRA 规划器。
        inputs: 归一化后的输入字典。
        futures: (ego_future, neighbors_future, neighbor_future_mask) 真值三元组。
        config: 模型配置（提供 state_normalizer）。
        rho: 风格强度（正数=激进，负数=保守）。
        time: 固定的扩散时间 t（保证与其它 rho 评估共享同一加噪程度）。
        noise: 固定的高斯噪声（保证与其它 rho 评估共享同一噪声）。

    Returns:
        style_diffusion_loss 返回的损失字典（loss/ego_target_loss/neighbor_preserve_loss 等）。
    """
    # 先切换风格强度，再用同一组 time/noise 计算损失
    planner.set_strength(rho)
    return style_diffusion_loss(planner, planner, inputs, futures, planner.sde.marginal_prob, config.state_normalizer,
                                time=time, noise=noise)


def main() -> None:
    """命令行入口：固定 x_t 下去噪门控评估。

    参数说明：
        --args-file: 模型配置文件路径。
        --baseline-checkpoint: 基线权重 checkpoint 路径。
        --aggressive-adapter: 激进风格 LoRA 适配器权重路径。
        --conservative-adapter: 保守风格 LoRA 适配器权重路径。
        --manifest: 场景样本清单（StyleManifestDataset 用）。
        --cache-root: 数据缓存根目录。
        --style: 目标评估风格（aggressive / conservative）。
        --output: 结果 JSON 输出路径。
        --rank: LoRA 秩（默认 4）。
        --batch-size: 批次大小（默认 16）。
        --workers: DataLoader 工作进程数（默认 4）。
        --max-batches: 最多评估批次数（默认 20）。
        --device: 计算设备（默认 cuda）。
    """
    parser = argparse.ArgumentParser(description="Fixed-x_t denoising gate with correct-vs-opposite adapter evidence.")
    parser.add_argument("--args-file", required=True); parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--aggressive-adapter", required=True); parser.add_argument("--conservative-adapter", required=True)
    parser.add_argument("--manifest", required=True); parser.add_argument("--cache-root", required=True)
    parser.add_argument("--style", choices=("aggressive", "conservative"), required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=4); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--max-batches", type=int, default=20); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # 1) 加载带 LoRA 注入的风格规划器（含两个适配器），以及独立加载的参考基线
    #    reference 用于身份误差（identity error）校验：禁用适配器时应与基线逐位一致。
    planner, config = load_baseline(args.args_file, args.baseline_checkpoint, args.device, rank=args.rank)
    reference, _ = load_plain_baseline(args.args_file, args.baseline_checkpoint, args.device)

    # 2) 把激进/保守两个适配器权重都加载进规划器
    for adapter in (args.aggressive_adapter, args.conservative_adapter):
        load_adapter_checkpoint(adapter, planner, baseline_checkpoint=args.baseline_checkpoint, normalization_file=config.normalization_file_path)

    # 3) 评估模式 + 按指定风格构建场景数据集与加载器
    planner.eval()
    dataset = StyleManifestDataset(args.manifest, root=args.cache_root, predicted_neighbor_num=config.predicted_neighbor_num, style=args.style)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=style_collate)

    # 4) 确定"正确"与"相反"的风格强度方向：
    #    评估 aggressive 时，正确 rho=+1.0（激进），相反 rho=-1.0（保守）；反之亦然。
    correct_rho, wrong_rho = (1.0, -1.0) if args.style == "aggressive" else (-1.0, 1.0)

    # 5) 累加器：正确方向自车损失 / 相反方向自车损失 / 邻居保持损失 / 身份误差列表
    totals, count, identity_errors = {"correct_ego_target_loss": 0.0, "opposite_ego_target_loss": 0.0, "neighbor_preserve_loss": 0.0}, 0, []
    for batch in loader:
        # 5.1) 数据准备：搬到设备并归一化
        inputs, futures = prepare_diffusion_batch(batch, torch.device(args.device), config.observation_normalizer)

        # 5.2) 固定扩散时间 t=0.5 与固定噪声：
        #      · time 全批次取 0.5（中度加噪），保证可复现；
        #      · noise 用与未来真值同形状的标准正态噪声，之后所有 rho 评估共用同一组。
        time = torch.full((futures[0].shape[0],), 0.5, device=args.device)
        noise = torch.randn_like(torch.cat((futures[0].unsqueeze(1), futures[1]), dim=1))

        # 5.3) 用同一组 time/noise 构造固定 x_t，专供 rho=0 identity 比较。
        #      这避免完整 diffusion sampling 在内部重新采样初始噪声。
        fixed_noisy_inputs, _, _ = build_noisy_inputs(
            inputs, futures, planner.sde.marginal_prob, config.state_normalizer, time=time, noise=noise
        )

        # 5.4) 分别在"正确风格方向"与"相反风格方向"下计算损失（共享同一 x_t 与 t）
        correct = _loss_for_rho(planner, inputs, futures, config, correct_rho, time, noise)
        opposite = _loss_for_rho(planner, inputs, futures, config, wrong_rho, time, noise)

        # 5.5) 禁用适配器（rho=0），在固定 x_t 上与独立参考基线比较身份误差。
        planner.set_strength(0.0)
        identity_errors.append(planner.base_identity_error(fixed_noisy_inputs, reference))

        # 5.6) 累加各统计量；达到 max_batches 则提前停止
        totals["correct_ego_target_loss"] += float(correct["ego_target_loss"])
        totals["opposite_ego_target_loss"] += float(opposite["ego_target_loss"])
        totals["neighbor_preserve_loss"] += float(correct["neighbor_preserve_loss"])
        count += 1
        if count >= args.max_batches: break

    # 6) 批次数为 0 时视为无效评估
    if not count: raise ValueError("No evaluation batches")

    # 7) 汇总：各项损失取批平均，并给出"正确方向是否优于相反方向"的门控结论
    result = {key: value / count for key, value in totals.items()}
    result.update({"style": args.style, "batches": count, "identity_error": max(identity_errors),
                   "correct_beats_opposite": result["correct_ego_target_loss"] < result["opposite_ego_target_loss"]})

    # 8) 写 JSON 报告并打印
    write_json(args.output, result); print(result)


if __name__ == "__main__": main()

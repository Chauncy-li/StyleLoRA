from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from research_lora.data.dataset import StyleManifestDataset, style_collate
from research_lora.data.manifest import manifest_hash
from research_lora.data.sampler import SceneBalancedSampler
from research_lora.model.checkpoint import save_adapter_checkpoint, sha256_file
from research_lora.runtime import load_baseline, prepare_diffusion_batch
from research_lora.training.trainer import StyleLoRATrainer
from research_lora.training.style_prototypes import SceneStylePrototypeTable
from research_lora.config import apply_yaml_defaults


def _start_swanlab(args):
    """按需创建 SwanLab 实验；未指定项目时完全不引入 SwanLab 依赖。"""
    if not args.swanlab_project:
        return None
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError("已指定 --swanlab-project，但环境未安装 swanlab；请先执行 `pip install swanlab`") from exc
    init_kwargs = {
        "project": args.swanlab_project,
        "experiment_name": args.swanlab_experiment_name,
        "mode": args.swanlab_mode,
        "tags": ["research_lora", args.style, args.scope, args.method],
        "config": vars(args).copy(),
    }
    if args.swanlab_workspace:
        init_kwargs["workspace"] = args.swanlab_workspace
    if args.swanlab_logdir:
        init_kwargs["logdir"] = args.swanlab_logdir
    return swanlab.init(**init_kwargs)


def _seed_training(seed: int) -> None:
    """固定本次训练的初始化、扩散噪声与数据工作进程的主随机状态。

    CUDA 的部分算子仍可能存在极小的非确定性；这里保证不同 ``--seed`` 是独立实验，
    同一 seed 在相同软件与硬件环境下具有可复现的随机序列。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_: int) -> None:
    """让每个 DataLoader worker 的 Python/NumPy 随机状态跟随其 PyTorch 种子。"""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one shared aggressive or conservative ego-masked LoRA.")
    parser.add_argument("--args-file", required=True, help="Baseline args.json")
    parser.add_argument("--config", default=None, help="Optional research_lora YAML defaults; explicit CLI values win.")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--style", required=True, choices=("aggressive", "conservative"))
    parser.add_argument("--scope", default="shared", choices=("shared",))
    parser.add_argument("--method", default="lora", choices=("lora",))
    parser.add_argument("--target-modules", default="mlp_output", choices=("mlp_output",))
    parser.add_argument("--rank", type=int, default=4, choices=(4, 8))
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=None,
                        help="训练总步数；未显式指定时读取 --config 中的 training.steps。")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--neighbor-weight", type=float, default=1.0)
    parser.add_argument("--lora-reg-weight", type=float, default=0.0)
    parser.add_argument("--style-prototype-file", default=None,
                        help="训练集生成的 scene × style 三轴原型 JSON；省略时保持纯 MSE 基线。")
    parser.add_argument("--prototype-weight", type=float, default=0.0,
                        help="正确风格原型距离的辅助权重；0 表示不启用方案 B。")
    parser.add_argument("--prototype-margin-weight", type=float, default=0.0,
                        help="正确原型优于相反原型的间隔损失权重。")
    parser.add_argument("--prototype-margin", type=float, default=0.20,
                        help="标准化三轴空间中的正确/相反原型最小距离间隔。")
    parser.add_argument("--seed", type=int, default=17, help="训练随机种子；每个正式重复实验使用不同值。")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="每隔多少训练 step 保存一个 LoRA 候选 checkpoint；0 表示只保存 best 与 last。")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--swanlab-project", default=None, help="启用 SwanLab 后使用的项目名；省略则不记录 SwanLab。")
    parser.add_argument("--swanlab-experiment-name", default=None, help="SwanLab 实验名。")
    parser.add_argument("--swanlab-workspace", default=None, help="可选 SwanLab 团队 workspace。")
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "local"), default="online",
                        help="SwanLab 记录模式，默认 online。")
    parser.add_argument("--swanlab-logdir", default=None, help="可选 SwanLab 本地日志目录。")
    args = apply_yaml_defaults(parser, parser.parse_args())
    if args.steps is None:
        parser.error("请通过 --steps 或 --config 的 training.steps 指定训练总步数")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every 必须为非负整数")
    if args.prototype_weight < 0 or args.prototype_margin_weight < 0 or args.prototype_margin < 0:
        parser.error("原型权重和 --prototype-margin 必须为非负数")
    if (args.prototype_weight > 0 or args.prototype_margin_weight > 0) and not args.style_prototype_file:
        parser.error("启用方案 B 时必须传入 --style-prototype-file（且该文件只能由 train manifest 构建）")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu only for a small functional smoke test")
    # 在创建 LoRA 参数、DataLoader 与扩散训练噪声前固定随机状态。
    _seed_training(args.seed)
    planner, config = load_baseline(args.args_file, args.baseline_checkpoint, args.device, rank=args.rank,
                                    alpha=args.alpha, dropout=args.dropout)
    planner.set_style(args.style).set_strength(1.0 if args.style == "aggressive" else -1.0)
    dataset = StyleManifestDataset(args.manifest, root=args.cache_root, predicted_neighbor_num=config.predicted_neighbor_num)
    records_hash = manifest_hash(dataset.samples)
    prototype_table = (SceneStylePrototypeTable.from_json(args.style_prototype_file, device=torch.device(args.device))
                       if args.style_prototype_file else None)
    if prototype_table is not None and prototype_table.source_manifest_hash != records_hash:
        raise ValueError("Style-prototype artifact was not built from this exact training manifest")
    # 训练：有放回补齐少数场景，确保两个场景的总采样数严格 1:1。
    sampler = SceneBalancedSampler(dataset.samples, args.style, seed=args.seed)
    train_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        pin_memory=args.device.startswith("cuda"), drop_last=True, collate_fn=style_collate,
                        worker_init_fn=_seed_worker, generator=train_generator)
    val_dataset = StyleManifestDataset(args.val_manifest, root=args.cache_root, predicted_neighbor_num=config.predicted_neighbor_num,
                                      style=args.style)
    # 验证：固定无放回的 1:1 子集。少数场景样本全部保留，另一场景等量随机抽取。
    # 不调用 val_sampler.set_epoch，因此所有验证轮次使用同一批样本，checkpoint 可公平比较。
    val_sampler = SceneBalancedSampler(val_dataset.samples, args.style, seed=args.seed, replacement=False)
    val_generator = torch.Generator().manual_seed(args.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers,
                            pin_memory=args.device.startswith("cuda"), collate_fn=style_collate,
                            worker_init_fn=_seed_worker, generator=val_generator)
    trainer = StyleLoRATrainer(
        planner, normalizer=config.state_normalizer, device=args.device, learning_rate=args.lr,
        prototype_table=prototype_table, prototype_weight=args.prototype_weight,
        prototype_margin_weight=args.prototype_margin_weight, prototype_margin=args.prototype_margin,
    )
    # 周期 checkpoint 共享兼容性哈希，避免每次保存都重新读取完整基线权重文件。
    checkpoint_hashes = {"baseline_sha256": sha256_file(args.baseline_checkpoint),
                         "normalization_sha256": sha256_file(config.normalization_file_path)}
    output_path = Path(args.output)

    def save_checkpoint(path: Path, *, best_validation=None) -> None:
        """保存当前激活风格分支，并复用本次训练已计算的兼容性哈希。"""
        save_adapter_checkpoint(path, planner, style=args.style, baseline_checkpoint=args.baseline_checkpoint,
                                manifest_hash=records_hash, normalization_file=config.normalization_file_path,
                                training_config=vars(args), best_validation=best_validation, **checkpoint_hashes)

    def save_periodic_checkpoint(step: int) -> None:
        """保存中间候选模型；末步由后续的 ``_last`` checkpoint 单独保存。"""
        if not args.checkpoint_every or step >= args.steps or step % args.checkpoint_every:
            return
        path = output_path.with_name(f"{output_path.stem}_step_{step:07d}{output_path.suffix}")
        save_checkpoint(path)
        if swanlab_run is not None:
            swanlab_run.log({"checkpoint/periodic_step": step}, step=step)

    swanlab_run = _start_swanlab(args)
    def log_metrics(phase, step, metrics):
        if swanlab_run is not None:
            swanlab_run.log({f"{phase}/{key}": value for key, value in metrics.items()}, step=step)
        # train_step 已完成参数更新；此时保存的是该 step 对应的当前 LoRA 权重。
        if phase == "train":
            save_periodic_checkpoint(step)
    if swanlab_run is not None:
        report = planner.trainable_parameter_report()
        swanlab_run.log({"experiment/seed": args.seed, "data/train_samples": len(dataset),
                          "data/validation_samples": len(val_dataset), "data/balanced_validation_samples": len(val_sampler),
                          "style_prototype/enabled": int(prototype_table is not None),
                          "model/trainable_parameters": report["trainable_parameters"],
                          "model/trainable_ratio": report["trainable_ratio"]}, step=0)

    def prepare(batch, device):
        return prepare_diffusion_batch(batch, device, config.observation_normalizer,
                                       return_style_context=prototype_table is not None)
    try:
        result = trainer.fit(loader, prepare, max_steps=args.steps, validation_loader=val_loader,
                             validate_every=args.validate_every, neighbor_weight=args.neighbor_weight,
                             lora_reg_weight=args.lora_reg_weight, metrics_callback=log_metrics)
        result["sampling"] = sampler.epoch_report()
        result["validation_sampling"] = val_sampler.epoch_report()
        planner.assert_frozen_base_unchanged()
        last_path = output_path.with_name(f"{output_path.stem}_last{output_path.suffix}")
        save_checkpoint(last_path, best_validation=result.get("best_validation", {}))
        if result["best_adapter_state"] is not None:
            planner.load_adapter_state_dict(result["best_adapter_state"], args.style)
        save_checkpoint(output_path, best_validation=result.get("best_validation", {}))
        if swanlab_run is not None:
            swanlab_run.log({"summary/seconds": result["seconds"], "summary/steps_per_second": result["steps_per_second"],
                              "summary/peak_cuda_bytes": result["peak_cuda_bytes"],
                              **{f"sampling/train_{key}": value for key, value in result["sampling"].items()},
                              **{f"sampling/validation_{key}": value for key, value in result["validation_sampling"].items()}},
                             step=result["steps"])
        print(result)
    finally:
        if swanlab_run is not None:
            swanlab_run.finish()


if __name__ == "__main__":
    main()

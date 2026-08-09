"""Train the CSPQ preference encoder.

主训练入口：
- 加载弱偏好 manifest + 冻结 h_c + 专家轨迹（PreferenceEncoderDataset）；
- 场景平衡采样（每批两场景各半，free 全量 + car 随机等量下采样）；
- 支持核心消融开关：--disable-cross-scene-rnc（跨场景统一是否成立的证据）；
- 训练结束后保存 best/last checkpoint（含模型配置、manifest/特征 hash）；
- 可选接入 SwanLab 监控（--swanlab-project 开启）。

CSPQ 是本工作提出的组合架构（非已有方法名称）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stylelora.data.encoder_dataset import (
    PreferenceEncoderDataset,
    SceneBalancedBatchSampler,
    encoder_collate,
)
from stylelora.model.preference_encoder import CSPQPreferenceEncoder
from stylelora.paths import (
    DEFAULT_ENCODER_CHECKPOINT,
    DEFAULT_ENCODER_LAST_CHECKPOINT,
    DEFAULT_FEATURE_INDEX,
    DEFAULT_FEATURE_NPY,
    DEFAULT_FEATURE_VAL_INDEX,
    DEFAULT_FEATURE_VAL_NPY,
    DEFAULT_PREFERENCE_MANIFEST,
    DEFAULT_PREFERENCE_VAL_MANIFEST,
    ensure_repo_on_path,
)
from stylelora.training.encoder_trainer import CSPQTrainer


def _seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_: int) -> None:
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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
        "tags": ["research_lora_2", "cspq_encoder"],
        "config": vars(args).copy(),
    }
    if args.swanlab_workspace:
        init_kwargs["workspace"] = args.swanlab_workspace
    return swanlab.init(**init_kwargs)


def main() -> None:
    ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="Train CSPQ preference encoder (cross-scene unified latent).")
    parser.add_argument("--train-manifest", default=str(DEFAULT_PREFERENCE_MANIFEST))
    parser.add_argument("--val-manifest", default=str(DEFAULT_PREFERENCE_VAL_MANIFEST))
    parser.add_argument("--train-feature-npy", default=str(DEFAULT_FEATURE_NPY))
    parser.add_argument("--train-feature-index", default=str(DEFAULT_FEATURE_INDEX))
    parser.add_argument("--val-feature-npy", default=str(DEFAULT_FEATURE_VAL_NPY))
    parser.add_argument("--val-feature-index", default=str(DEFAULT_FEATURE_VAL_INDEX))
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", default=str(DEFAULT_ENCODER_CHECKPOINT), help="best checkpoint 输出路径。")
    parser.add_argument("--output-last", default=str(DEFAULT_ENCODER_LAST_CHECKPOINT), help="last checkpoint 输出路径。")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="每个 epoch 最多训练步数（受批数约束）。")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="每 N 训练步保存一个中间候选 checkpoint；0=关闭（默认）。")
    parser.add_argument("--checkpoint-max-k", type=int, default=5,
                        help="最多保留的中间候选 checkpoint 数（超出删除最旧）；仅当 checkpoint-every>0 时生效。")
    parser.add_argument("--batch-size", type=int, default=64, help="每批样本数（需为偶数；每场景各半）。")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--z-dim", type=int, default=8)
    parser.add_argument("--query-rank", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lambda-cross", type=float, default=0.1)
    parser.add_argument("--lambda-r", type=float, default=1.0)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--disable-cross-scene-rnc", action="store_true",
                        help="消融：关闭跨场景 RNC（跨场景统一是否成立的证据）。")
    # SwanLab 监控参数（完全可选）
    parser.add_argument("--swanlab-project", default=None, help="启用 SwanLab 后使用的项目名；省略则不记录。")
    parser.add_argument("--swanlab-experiment-name", default=None, help="SwanLab 实验名。")
    parser.add_argument("--swanlab-workspace", default=None, help="可选 SwanLab 团队 workspace。")
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "local"), default="online")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.batch_size % 2 != 0:
        parser.error("--batch-size 必须为正偶数（每场景各半）")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；可先用 --device cpu 做小规模冒烟。")

    _seed(args.seed)
    swanlab_run = _start_swanlab(args)

    # ---------- 数据集 ----------
    train_ds = PreferenceEncoderDataset(
        args.train_manifest, args.train_feature_npy, args.train_feature_index, args.cache_root,
    )
    val_ds = PreferenceEncoderDataset(
        args.val_manifest, args.val_feature_npy, args.val_feature_index, args.cache_root,
    )
    if train_ds.missing > 0 or val_ds.missing > 0:
        raise ValueError(
            f"存在特征缺失样本（train missing={train_ds.missing}, val missing={val_ds.missing}）；"
            "研究数据不应静默减少，请先生成完整的对齐特征索引再训练"
        )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("训练/验证数据集为空；请检查 manifest 与特征对齐（含过滤全无效样本后）")
    print(f"[data] train={len(train_ds)} (missing features={train_ds.missing}), "
          f"free={len(train_ds.free_indices)}, car={len(train_ds.car_indices)}")
    print(f"[data] val={len(val_ds)} (missing features={val_ds.missing}), "
          f"free={len(val_ds.free_indices)}, car={len(val_ds.car_indices)}")

    # ---------- 平衡采样 ----------
    gen = torch.Generator().manual_seed(args.seed)
    train_sampler = SceneBalancedBatchSampler(train_ds, args.batch_size, generator=gen)
    val_sampler = SceneBalancedBatchSampler(val_ds, args.batch_size, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=args.workers,
                              pin_memory=args.device.startswith("cuda"), collate_fn=encoder_collate,
                              worker_init_fn=_seed_worker)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=args.workers,
                            pin_memory=args.device.startswith("cuda"), collate_fn=encoder_collate,
                            worker_init_fn=_seed_worker)

    # ---------- 模型 ----------
    model = CSPQPreferenceEncoder(
        trajectory_dim=6, hc_dim=train_ds._features.shape[1], d_model=args.d_model,
        heads=args.heads, z_dim=args.z_dim, query_rank=args.query_rank,
    )
    trainer = CSPQTrainer(
        model, device=args.device, learning_rate=args.lr,
        lambda_cross=args.lambda_cross, lambda_r=args.lambda_r, lambda_a=args.lambda_a,
        temperature=args.temperature, disable_cross_scene=args.disable_cross_scene_rnc,
    )

    if swanlab_run is not None:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        swanlab_run.log({
            "data/train_samples": len(train_ds),
            "data/free": len(train_ds.free_indices),
            "data/car": len(train_ds.car_indices),
            "model/trainable_parameters": trainable,
            "sampling/batches_per_epoch": len(train_sampler),
            "sampling/seed": args.seed,
        }, step=0)

    # 每个 epoch 重置 car-follow 采样种子（不同 epoch 轮换）
    def on_epoch_start(epoch: int) -> None:
        train_sampler.generator = torch.Generator().manual_seed(args.seed + epoch)

    # 周期保存：全局 step 偏移（跨 epoch 连续编号）+ 已保留中间候选路径
    global_step_offset = 0
    periodic_paths: list = []
    periodic_config = vars(args).copy()

    def log_metrics(phase: str, step: int, metrics) -> None:
        """训练/验证指标回调：SwanLab 记录 + 周期 checkpoint 保存（含 k 上限清理）。"""
        nonlocal global_step_offset
        # 周期保存：train 阶段、checkpoint-every>0、step>0 且命中步数间隔
        if phase == "train" and args.checkpoint_every > 0 and step > 0 and step % args.checkpoint_every == 0:
            global_step = global_step_offset + step
            base = Path(args.output)
            ckpt_path = base.with_name(f"{base.stem}_step_{global_step:07d}{base.suffix}")
            trainer.save_checkpoint(
                ckpt_path, manifest_path=args.train_manifest,
                feature_index_path=args.train_feature_index,
                training_config=periodic_config, best_validation=best_overall,
            )
            periodic_paths.append(str(ckpt_path))
            # k 上限：超出 checkpoint-max-k 时删除最旧候选
            if args.checkpoint_max_k > 0 and len(periodic_paths) > args.checkpoint_max_k:
                old = Path(periodic_paths.pop(0))
                if old.exists():
                    old.unlink()
            print(f"[periodic] saved {ckpt_path} (kept {len(periodic_paths)})")
        # SwanLab 记录
        if swanlab_run is None:
            return
        payload = {}
        for key, value in metrics.items():
            if key == "scene_losses" and isinstance(value, dict):
                for scene_name, scene_vals in value.items():
                    for sub_key, sub_val in scene_vals.items():
                        payload[f"scene/{scene_name}/{sub_key}"] = float(sub_val)
            elif isinstance(value, (int, float)):
                payload[key] = float(value)
        if payload:
            swanlab_run.log({f"{phase}/{k}": v for k, v in payload.items()}, step=step)

    trained_steps = 0
    best_overall = {"loss": float("inf")}
    best_state_all = None
    for epoch in range(args.epochs):
        # 每个 epoch 重置 trainer.step，保证后续 epoch 继续训练
        trainer.step = 0
        on_epoch_start(epoch)
        print(f"[train] epoch {epoch + 1}/{args.epochs}")
        result = trainer.fit(
            train_loader,
            max_steps=args.steps_per_epoch,
            validation_loader=val_loader,
            validate_every=max(1, args.steps_per_epoch // 2),
            checkpoint_callback=log_metrics,
        )
        trained_steps += result["step"]
        global_step_offset += result["step"]  # 跨 epoch 累计 step，保证周期编号不重复
        if result["best_validation"]["loss"] < best_overall["loss"]:
            best_overall = result["best_validation"]
            best_state_all = result["best_state"]

    # ---------- 保存 checkpoint ----------
    config_vars = vars(args).copy()
    # 先保存 last：记录"最后 epoch 的最终模型状态"（当前模型即最后训练后的状态）
    trainer.save_checkpoint(Path(args.output_last), manifest_path=args.train_manifest,
                            feature_index_path=args.train_feature_index, training_config=config_vars,
                            best_validation=best_overall)
    # 再载入 best_state 并保存 best，确保 best 与 last 是两个不同 checkpoint
    if best_state_all is not None:
        trainer.model.load_state_dict(best_state_all)
    trainer.save_checkpoint(Path(args.output), manifest_path=args.train_manifest,
                            feature_index_path=args.train_feature_index, training_config=config_vars,
                            best_validation=best_overall)
    print(f"[save] best -> {args.output}")
    print(f"[save] last -> {args.output_last}")
    print(f"[result] scenario balance: per_scene={train_sampler.per_scene} batches/epoch={len(train_sampler)}")

    if swanlab_run is not None:
        swanlab_run.log({
            "summary/steps": trained_steps,
            "summary/best_loss": float(best_overall["loss"]),
            "summary/batches_per_epoch": len(train_sampler),
        }, step=trained_steps)
        swanlab_run.finish()


if __name__ == "__main__":
    main()


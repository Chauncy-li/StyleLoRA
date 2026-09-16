"""训练场景与目标风格条件的连续 LoRA 路由。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stylelora.data.counterfactual_preference_dataset import (
    CounterfactualPreferenceDataset,
    counterfactual_preference_collate,
)
from stylelora.data.preference_lora_dataset import (
    PreferenceLoRADataset,
    SceneBalancedLoRASampler,
    preference_lora_collate,
)
from stylelora.lora.model.checkpoint import load_adapter_checkpoint, save_adapter_checkpoint
from stylelora.lora.model.style_lora_planner import StyleLoRAPlanner
from stylelora.lora.runtime import load_plain_baseline
from stylelora.model.conditional_lora_router import (
    ConditionalLoRARouter,
    load_conditional_router_checkpoint,
    save_conditional_router_checkpoint,
)
from stylelora.training.conditional_preference_lora import ConditionalPreferenceLoRATrainer
from stylelora.training.preference_lora import load_frozen_cspq


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _rows_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _style_prototypes(dataset: PreferenceLoRADataset) -> dict[str, torch.Tensor]:
    """从完整训练集三段 rank 计算归一化风格原型。"""
    values = torch.as_tensor(dataset._latent[np.asarray(dataset.latent_rows)]).float()
    ranks = torch.tensor([sample.preference_rank for sample in dataset.samples]).float()
    masks = {
        "low": ranks <= 0.2,
        "neutral": (ranks >= 0.4) & (ranks <= 0.6),
        "high": ranks >= 0.8,
    }
    prototypes = {}
    for name, mask in masks.items():
        if not bool(mask.any()):
            raise ValueError(f"训练集无法计算 {name} 风格原型")
        prototypes[name] = F.normalize(values[mask].mean(dim=0), dim=0, eps=1e-6)
    return prototypes


def _state_for_best(planner: StyleLoRAPlanner) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in planner.state_dict().items()
        if ".aggressive.lora_" in key
        or ".conservative.lora_" in key
        or key.startswith("conditional_router.")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conditional dynamic preference LoRA.")
    parser.add_argument("--args-file", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--init-adapter-high", required=True)
    parser.add_argument("--init-adapter-low", required=True)
    parser.add_argument(
        "--init-router",
        default=None,
        help="可选：从已有 Bounded V2 条件路由继续训练；省略时保持原有随机初始化行为。",
    )
    parser.add_argument("--cspq-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--latent-bank", required=True)
    parser.add_argument("--latent-bank-index", required=True)
    parser.add_argument("--feature-npy", required=True)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--val-latent-bank")
    parser.add_argument("--val-latent-bank-index")
    parser.add_argument("--val-feature-npy")
    parser.add_argument("--val-feature-index")
    parser.add_argument("--cf-pair-bank")
    parser.add_argument("--cf-pair-index")
    parser.add_argument("--val-cf-pair-bank")
    parser.add_argument("--val-cf-pair-index")
    parser.add_argument("--output-high", required=True)
    parser.add_argument("--output-low", required=True)
    parser.add_argument("--output-router", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--router-hidden-dim", type=int, default=128)
    parser.add_argument("--lambda-n", type=float, default=1.0)
    parser.add_argument("--lambda-z", type=float, default=1.0)
    parser.add_argument("--lambda-s", type=float, default=1.0)
    parser.add_argument("--lambda-q", type=float, default=1.0)
    parser.add_argument("--lambda-dyn", type=float, default=0.1)
    parser.add_argument("--lambda-lat", type=float, default=1.0)
    parser.add_argument("--lambda-order", type=float, default=1.0)
    parser.add_argument(
        "--lambda-feasibility", type=float, default=0.0,
        help="不可行候选的可微 baseline 相对惩罚；0 保持旧 V2/V3 行为。",
    )
    parser.add_argument(
        "--lambda-response",
        type=float,
        default=0.0,
        help="相邻 rho 的纵向 progress 间隔损失；0 完全保持现有训练行为。",
    )
    parser.add_argument("--response-margin-m-per-rho", type=float, default=1.0)
    parser.add_argument("--response-min-baseline-progress-m", type=float, default=3.0)
    parser.add_argument("--order-margin-scale", type=float, default=0.25)
    parser.add_argument("--min-rho-gap", type=float, default=0.25)
    parser.add_argument(
        "--order-pair-mode", choices=("cross", "mixed_adjacent"), default="cross",
        help="cross 复现 V2；mixed_adjacent 混合跨方向与连续网格相邻排序。",
    )
    parser.add_argument(
        "--order-rho-grid",
        default="-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1",
    )
    parser.add_argument("--local-order-ratio", type=float, default=0.75)
    parser.add_argument("--lateral-tolerance", type=float, default=0.3)
    parser.add_argument("--lateral-topk-ratio", type=float, default=0.2)
    parser.add_argument("--lateral-smooth-weight", type=float, default=0.1)
    parser.add_argument("--feasible-max-mean-accel-degradation", type=float, default=0.5)
    parser.add_argument("--feasible-max-mean-jerk-degradation", type=float, default=2.0)
    parser.add_argument("--feasible-max-progress-loss-m", type=float, default=2.0)
    parser.add_argument(
        "--feasible-max-mean-lateral-deviation-m",
        "--feasible-max-mean-path-deviation-m",
        dest="feasible_max_mean_lateral_deviation_m",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--lambda-cf",
        type=float,
        default=0.0,
        help="V3 可行反事实配对辅助损失权重；默认 0，保持 Bounded V2 训练路径。",
    )
    parser.add_argument("--cf-margin", type=float, default=0.001)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        args.order_rho_grid = tuple(
            float(item) for item in args.order_rho_grid.split(",") if item.strip()
        )
    except ValueError as exc:
        parser.error(f"--order-rho-grid 解析失败：{exc}")

    if args.batch_size <= 0 or args.batch_size % 2:
        parser.error("--batch-size 必须是正偶数")
    if args.steps <= 0 or args.val_every <= 0 or args.val_batches <= 0:
        parser.error("steps/val-every/val-batches 必须为正")
    if args.lambda_cf < 0 or args.cf_margin < 0:
        parser.error("lambda-cf 和 cf-margin 不能为负")
    if args.lambda_feasibility < 0:
        parser.error("--lambda-feasibility 不能为负")
    if (
        args.lambda_response < 0
        or args.response_margin_m_per_rho < 0
        or args.response_min_baseline_progress_m < 0
    ):
        parser.error("纵向响应损失权重、间隔和最小 baseline progress 不能为负")
    if not 0 <= args.local_order_ratio <= 1:
        parser.error("--local-order-ratio 必须位于 [0,1]")
    if (
        len(args.order_rho_grid) < 3
        or any(b <= a for a, b in zip(args.order_rho_grid, args.order_rho_grid[1:]))
        or args.order_rho_grid[0] < -1
        or args.order_rho_grid[-1] > 1
        or not any(abs(value) < 1e-9 for value in args.order_rho_grid)
    ):
        parser.error("--order-rho-grid 必须在 [-1,1] 内严格递增并包含 0")
    if args.lambda_cf > 0 and not all((args.cf_pair_bank, args.cf_pair_index)):
        parser.error("启用反事实辅助损失时必须提供 cf-pair-bank 和 cf-pair-index")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    _seed(args.seed)

    print("[stage 1/5] 加载完整连续偏好数据", flush=True)
    dataset = PreferenceLoRADataset(
        args.manifest,
        args.cache_root,
        args.latent_bank,
        args.latent_bank_index,
        args.feature_npy,
        args.feature_index,
        direction="conditional",
        rank_low=0.0,
        rank_high=1.0,
    )
    if not dataset.free_indices or not dataset.car_indices:
        raise ValueError("完整训练集必须包含两类现有场景采样来源")
    prototypes = _style_prototypes(dataset)
    sampler = SceneBalancedLoRASampler(
        dataset, args.batch_size, generator=torch.Generator().manual_seed(args.seed)
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=preference_lora_collate,
    )
    print(f"[data] train={len(dataset)} missing={dataset.missing}", flush=True)

    cf_loader = None
    cf_sample_count = 0
    if args.lambda_cf > 0:
        cf_dataset = CounterfactualPreferenceDataset(
            pair_bank=args.cf_pair_bank,
            pair_index=args.cf_pair_index,
            manifest=args.manifest,
            cache_root=args.cache_root,
            latent_bank=args.latent_bank,
            latent_bank_index=args.latent_bank_index,
            feature_npy=args.feature_npy,
            feature_index=args.feature_index,
        )
        cf_sample_count = len(cf_dataset)
        cf_loader = DataLoader(
            cf_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.device.startswith("cuda"),
            collate_fn=counterfactual_preference_collate,
            drop_last=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"[data] feasible_style_pairs={len(cf_dataset)}", flush=True)

    print("[stage 2/5] 加载冻结 baseline、现有 High/Low LoRA 与 CSPQ", flush=True)
    model, config = load_plain_baseline(args.args_file, args.baseline_checkpoint, args.device)
    planner = StyleLoRAPlanner(model, rank=args.rank, alpha=args.alpha).to(args.device)
    load_adapter_checkpoint(
        args.init_adapter_high,
        planner,
        baseline_checkpoint=args.baseline_checkpoint,
        normalization_file=config.normalization_file_path,
    )
    load_adapter_checkpoint(
        args.init_adapter_low,
        planner,
        baseline_checkpoint=args.baseline_checkpoint,
        normalization_file=config.normalization_file_path,
    )
    cspq = load_frozen_cspq(args.cspq_checkpoint, args.device)
    if args.init_router:
        router, checkpoint_prototypes, _ = load_conditional_router_checkpoint(
            args.init_router, args.device
        )
        if router.use_diffusion_time:
            raise ValueError("--init-router 必须是不使用扩散时间输入的 Bounded V2 checkpoint")
        if tuple(router.layer_names) != tuple(planner.report.layers):
            raise ValueError("--init-router 的 LoRA 层列表与当前 planner 不一致")
        if router.hc_dim != int(dataset._features.shape[1]) or router.z_dim != int(dataset._latent.shape[1]):
            raise ValueError("--init-router 的场景或风格特征维度与当前数据不一致")
        prototypes = checkpoint_prototypes
        router = router.to(args.device)
        print(f"[model] 从 Bounded V2 路由继续训练：{args.init_router}", flush=True)
    else:
        router = ConditionalLoRARouter(
            planner.report.layers,
            hc_dim=int(dataset._features.shape[1]),
            z_dim=int(dataset._latent.shape[1]),
            hidden_dim=args.router_hidden_dim,
        ).to(args.device)
    planner.attach_conditional_router(
        router,
        prototypes,
        enabled=True,
        trainable=True,
    )

    loss_keys = (
        "lambda_n", "lambda_z", "lambda_s", "lambda_q", "lambda_dyn", "lambda_lat",
        "lambda_order", "lambda_feasibility", "lambda_response",
        "response_margin_m_per_rho", "response_min_baseline_progress_m",
        "order_margin_scale", "min_rho_gap",
        "order_pair_mode", "order_rho_grid", "local_order_ratio",
        "lateral_tolerance", "lateral_topk_ratio", "lateral_smooth_weight",
        "feasible_max_mean_accel_degradation", "feasible_max_mean_jerk_degradation",
        "feasible_max_progress_loss_m", "feasible_max_mean_lateral_deviation_m",
    )
    trainer = ConditionalPreferenceLoRATrainer(
        planner,
        cspq,
        observation_normalizer=config.observation_normalizer,
        state_normalizer=config.state_normalizer,
        device=args.device,
        learning_rate=args.lr,
        lambda_cf=args.lambda_cf,
        cf_margin=args.cf_margin,
        **{key: getattr(args, key) for key in loss_keys},
    )

    val_loader = None
    val_cf_loader = None
    val_sample_count = 0
    if args.val_manifest:
        required = (
            args.val_latent_bank,
            args.val_latent_bank_index,
            args.val_feature_npy,
            args.val_feature_index,
        )
        if not all(required):
            raise ValueError("提供验证 manifest 时必须提供全部验证 latent/feature 文件")
        val_dataset = PreferenceLoRADataset(
            args.val_manifest,
            args.cache_root,
            args.val_latent_bank,
            args.val_latent_bank_index,
            args.val_feature_npy,
            args.val_feature_index,
            direction="conditional",
            rank_low=0.0,
            rank_high=1.0,
        )
        val_sampler = SceneBalancedLoRASampler(
            val_dataset, args.batch_size, generator=torch.Generator().manual_seed(args.seed)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=args.workers,
            pin_memory=args.device.startswith("cuda"),
            collate_fn=preference_lora_collate,
        )
        val_sample_count = len(val_dataset)
        print(f"[data] val={len(val_dataset)} missing={val_dataset.missing}", flush=True)
        if args.lambda_cf > 0:
            if not all((args.val_cf_pair_bank, args.val_cf_pair_index)):
                raise ValueError("启用验证集时必须提供 val-cf-pair-bank 和 val-cf-pair-index")
            val_cf_dataset = CounterfactualPreferenceDataset(
                pair_bank=args.val_cf_pair_bank,
                pair_index=args.val_cf_pair_index,
                manifest=args.val_manifest,
                cache_root=args.cache_root,
                latent_bank=args.val_latent_bank,
                latent_bank_index=args.val_latent_bank_index,
                feature_npy=args.val_feature_npy,
                feature_index=args.val_feature_index,
            )
            val_cf_loader = DataLoader(
                val_cf_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=args.device.startswith("cuda"),
                collate_fn=counterfactual_preference_collate,
            )
            print(f"[data] val_feasible_style_pairs={len(val_cf_dataset)}", flush=True)

    print("[stage 3/5] 动态条件 LoRA 训练", flush=True)
    best_loss = float("inf")
    best_state = None
    best_metrics: dict[str, float] | None = None
    step = 0
    last_val_step = -1
    cf_iterator = iter(cf_loader) if cf_loader is not None else None
    while step < args.steps:
        for batch in loader:
            cf_batch = None
            if cf_iterator is not None:
                try:
                    cf_batch = next(cf_iterator)
                except StopIteration:
                    cf_iterator = iter(cf_loader)
                    cf_batch = next(cf_iterator)
            metrics = trainer.train_step(batch, cf_batch)
            step += 1
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(
                    f"[train] {step}/{args.steps} loss={metrics['loss']:.4f} "
                    f"ego={metrics['ego_denoise']:.4f} rank={metrics['rank_huber']:.4f} "
                    f"latent={metrics['mmd_z']:.4f} factors={metrics['factor_huber']:.4f} "
                    f"lat={metrics['lateral']:.4f} order={metrics['order']:.4f} "
                    f"response={metrics['response']:.4f} "
                    f"response_gap={metrics['response_mean_gap_m']:.3f} "
                    f"local_vio={metrics['local_order_violation_rate']:.3f} "
                    f"feasible={metrics['feasible_pair_rate']:.3f} "
                    f"feas_loss={metrics['feasibility']:.4f} cf={metrics['cf_loss']:.4f} "
                    f"cf_acc={metrics['cf_accuracy']:.3f}",
                    flush=True,
                )
            candidate = None
            if val_loader is not None and (step % args.val_every == 0 or step == args.steps):
                val_sampler.generator.manual_seed(args.seed)
                candidate = trainer.validate(
                    val_loader,
                    style_loader=val_cf_loader,
                    max_batches=args.val_batches,
                    seed=args.seed,
                )
                last_val_step = step
                print(
                    f"[val] step={step} loss={candidate['loss']:.4f} "
                    f"rank={candidate['rank_huber']:.4f} "
                    f"latent={candidate['mmd_z']:.4f} "
                    f"lat={candidate['lateral']:.4f} order={candidate['order']:.4f} "
                    f"response={candidate['response']:.4f} "
                    f"response_gap={candidate['response_mean_gap_m']:.3f} "
                    f"local_vio={candidate['local_order_violation_rate']:.3f} "
                    f"feasible={candidate['feasible_pair_rate']:.3f} "
                    f"feas_loss={candidate['feasibility']:.4f} cf={candidate['cf_loss']:.4f} "
                    f"cf_acc={candidate['cf_accuracy']:.3f}",
                    flush=True,
                )
            # 有验证集时只能用固定验证损失选择最佳模型，不能混入单个训练 batch。
            if candidate is not None and candidate["loss"] < best_loss:
                best_loss = candidate["loss"]
                best_state = _state_for_best(planner)
                best_metrics = dict(candidate)
            elif val_loader is None and metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                best_state = _state_for_best(planner)
                best_metrics = dict(metrics)
            if step >= args.steps:
                break

    # 正常配置会在最后一步验证；保留显式兜底，避免以后调整触发条件后无 best。
    if val_loader is not None and last_val_step != args.steps:
        val_sampler.generator.manual_seed(args.seed)
        candidate = trainer.validate(
            val_loader,
            style_loader=val_cf_loader,
            max_batches=args.val_batches,
            seed=args.seed,
        )
        if candidate["loss"] < best_loss:
            best_loss = candidate["loss"]
            best_state = _state_for_best(planner)
            best_metrics = dict(candidate)

    print("[stage 4/5] 恢复最佳验证参数并检查冻结基座", flush=True)
    if best_state is None:
        raise RuntimeError("训练没有产生可保存参数")
    planner.load_state_dict(best_state, strict=False)
    planner.assert_frozen_base_unchanged()

    print("[stage 5/5] 分别保存 High、Low 和条件路由 checkpoint", flush=True)
    manifest_hash = _rows_hash(Path(args.manifest))
    training_config = vars(args)
    best_validation = best_metrics or {"loss": best_loss}
    save_adapter_checkpoint(
        args.output_high,
        planner,
        style="aggressive",
        baseline_checkpoint=args.baseline_checkpoint,
        manifest_hash=manifest_hash,
        normalization_file=config.normalization_file_path,
        training_config=training_config,
        best_validation=best_validation,
    )
    save_adapter_checkpoint(
        args.output_low,
        planner,
        style="conservative",
        baseline_checkpoint=args.baseline_checkpoint,
        manifest_hash=manifest_hash,
        normalization_file=config.normalization_file_path,
        training_config=training_config,
        best_validation=best_validation,
    )
    save_conditional_router_checkpoint(
        args.output_router,
        planner.conditional_router,
        prototypes=prototypes,
        training_config=training_config,
        best_validation=best_validation,
    )
    report_path = Path(args.output_router).with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "method": (
                    "longitudinal_response_tuned_v5"
                    if args.lambda_response > 0
                    else (
                        "ordered_feasible_conditional_style_v4"
                        if args.order_pair_mode == "mixed_adjacent" and args.lambda_feasibility > 0
                        else "bounded_conditional_style_with_feasible_pairs_v3"
                    )
                ),
                "train_samples": len(dataset),
                "feasible_style_pairs": cf_sample_count,
                "validation_samples": val_sample_count,
                "best_validation": best_validation,
                "training_config": training_config,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] high={args.output_high}", flush=True)
    print(f"[done] low={args.output_low}", flush=True)
    print(f"[done] router={args.output_router}", flush=True)
    print(f"[done] report={report_path}", flush=True)


if __name__ == "__main__":
    main()

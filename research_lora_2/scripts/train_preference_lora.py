"""Train continuous-preference LoRA (high/low direction).

唯一 LoRA 训练入口：
- 复用 research_lora 的 StyleLoRAPlanner（双分支 LoRA + rho 路由）+ checkpoint 存取，
  内部 aggressive/conservative 槽位兼容旧 checkpoint；对外只暴露 high/low 方向。
- 冻结 DiffPlanner 基座与 CSPQ 编码器；预测轨迹过冻结 CSPQ（保留梯度）算偏好损失。
- 按 rank 区间取高/低偏好数据（high 默认 [0.8,1.0]，low 默认 [0,0.2]），两场景 1:1。
- 默认启动 SwanLab 监控（--swanlab-project 默认 "preference-lora"）；传 --no-swanlab 关闭。
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from research_lora.model.checkpoint import save_adapter_checkpoint
from research_lora.model.style_lora_planner import StyleLoRAPlanner
from research_lora.runtime import load_plain_baseline

from research_lora_2.data.preference_lora_dataset import (
    PreferenceLoRADataset,
    SceneBalancedLoRASampler,
    preference_lora_collate,
)
from research_lora_2.paths import (
    DEFAULT_ENCODER_CHECKPOINT,
    DEFAULT_FEATURE_INDEX,
    DEFAULT_FEATURE_NPY,
    DEFAULT_PREFERENCE_MANIFEST,
    ensure_repo_on_path,
)
from research_lora_2.training.preference_lora import PreferenceLoRATrainer, load_frozen_cspq

# 方向 -> (rank区间, 内部LoRA槽位)
DIRECTION_META = {
    "high": (0.8, 1.0, "aggressive"),
    "low": (0.0, 0.2, "conservative"),
}


def _rows_hash(path: Path) -> str:
    """偏好 manifest 的简单行哈希（用于 checkpoint 记录数据版本）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _start_swanlab(args):
    """默认启动 SwanLab；--no-swanlab 显式关闭。

    与项目其它训练脚本（train_preference_encoder / train_style_lora）保持同一
    swanlab.init 风格；默认 project 为 "preference-lora"，避免误传参数导致静默不记录。
    """
    if getattr(args, "no_swanlab", False):
        return None
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "默认启动 SwanLab 但环境未安装 swanlab；请先 `pip install swanlab`，"
            "或显式传 --no-swanlab 关闭"
        ) from exc
    init_kwargs = {
        "project": args.swanlab_project,
        "experiment_name": args.swanlab_experiment_name,
        "mode": args.swanlab_mode,
        "tags": ["research_lora_2", f"preference_lora_{args.direction}"],
        "config": vars(args),
    }
    if getattr(args, "swanlab_workspace", None):
        init_kwargs["workspace"] = args.swanlab_workspace
    if getattr(args, "swanlab_logdir", None):
        init_kwargs["logdir"] = args.swanlab_logdir
    return swanlab.init(**init_kwargs)


def _swanlab_log(swanlab_run, metrics: dict, step: int) -> None:
    """统一 SwanLab 记录入口；swanlab_run 为 None 时静默跳过。"""
    if swanlab_run is None:
        return
    try:
        swanlab_run.log({f"{phase}/{key}": value for phase, kv in metrics.items()
                         for key, value in kv.items()}, step=step)
    except TypeError:
        # 兼容旧版 swanlab（不接受 step 关键字）
        for phase, kv in metrics.items():
            swanlab_run.log({f"{phase}/{key}": value for key, value in kv.items()})


def main() -> None:
    ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="Train continuous-preference LoRA (high/low direction).")
    parser.add_argument("--args-file", required=True, help="Baseline args.json.")
    parser.add_argument("--baseline-checkpoint", required=True, help="Untouched DiffPlanner checkpoint.")
    parser.add_argument("--direction", required=True, choices=("high", "low"),
                        help="偏好方向：high=rank 高区间（激进），low=rank 低区间（保守）。")
    parser.add_argument("--manifest", default=str(DEFAULT_PREFERENCE_MANIFEST), help="弱偏好 manifest（训练拆分）。")
    parser.add_argument("--val-manifest", default=None, help="弱偏好 manifest（验证拆分）；省略则不验证（先跑通小步数用）。")
    parser.add_argument("--cache-root", required=True, help="DiffPlanner 缓存根目录。")
    parser.add_argument("--latent-bank", required=True, help="冻结编码器导出的 latent bank npy。")
    parser.add_argument("--latent-bank-index", required=True, help="latent bank 索引 jsonl。")
    parser.add_argument("--cspq-checkpoint", default=str(DEFAULT_ENCODER_CHECKPOINT), help="冻结 CSPQ checkpoint。")
    parser.add_argument("--feature-npy", default=str(DEFAULT_FEATURE_NPY), help="冻结 h_c npy。")
    parser.add_argument("--feature-index", default=str(DEFAULT_FEATURE_INDEX), help="冻结 h_c 索引。")
    parser.add_argument("--val-latent-bank", default=None, help="验证 latent bank npy（--val-manifest 时必需）。")
    parser.add_argument("--val-latent-bank-index", default=None, help="验证 latent bank 索引 jsonl。")
    parser.add_argument("--val-feature-npy", default=None, help="验证 h_c npy。")
    parser.add_argument("--val-feature-index", default=None, help="验证 h_c 索引。")
    parser.add_argument("--output", required=True, help="输出 adapter checkpoint（.pt）。")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32, help="需为偶数；每场景各半。")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-n", type=float, default=1.0)
    parser.add_argument("--lambda-z", type=float, default=1.0)
    parser.add_argument("--lambda-s", type=float, default=1.0)
    parser.add_argument("--lambda-dyn", type=float, default=0.0,
                        help="Ego 动力学一致性损失权重（加速度+jerk Huber）；首次实验建议 0.1。")
    parser.add_argument("--lambda-q", type=float, default=0.0,
                        help="CSPQ 三因子对齐损失权重（q_hat vs axis_percentiles Huber）；首次实验建议 1.0。")
    parser.add_argument("--rank", type=int, default=4, help="LoRA 秩。")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    # SwanLab 监控（默认启动；--no-swanlab 关闭）
    parser.add_argument("--no-swanlab", action="store_true", help="关闭 SwanLab 记录（默认开启）。")
    parser.add_argument("--swanlab-project", default="preference-lora", help="SwanLab 项目名（默认 preference-lora）。")
    parser.add_argument("--swanlab-experiment-name", default=None, help="SwanLab 实验名（默认自动生成）。")
    parser.add_argument("--swanlab-workspace", default=None, help="可选 SwanLab 团队 workspace。")
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "local"), default="online",
                        help="SwanLab 记录模式，默认 online。")
    parser.add_argument("--swanlab-logdir", default=None, help="可选 SwanLab 本地日志目录。")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.batch_size % 2 != 0:
        parser.error("--batch-size 必须为正偶数（每场景各半）")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；可先用 --device cpu 做小规模冒烟。")

    _seed(args.seed)
    rank_low, rank_high, _slot = DIRECTION_META[args.direction]

    # ---------- 数据 ----------
    ds = PreferenceLoRADataset(
        args.manifest, args.cache_root, args.latent_bank, args.latent_bank_index,
        args.feature_npy, args.feature_index, direction=args.direction,
        rank_low=rank_low, rank_high=rank_high,
    )
    if len(ds) == 0:
        raise ValueError(
            f"direction={args.direction} rank∈[{rank_low},{rank_high}] 无样本；"
            "请检查 manifest / latent bank / 特征对齐")
    if not ds.free_indices or not ds.car_indices:
        raise ValueError(
            f"direction={args.direction} rank窗口内场景不完整："
            f"free={len(ds.free_indices)}, car={len(ds.car_indices)}；"
            "SceneBalancedLoRASampler 任一场景为空会产出空迭代器，训练会陷入无限循环")
    print(f"[data] direction={args.direction} samples={len(ds)} "
          f"(free={len(ds.free_indices)}, car={len(ds.car_indices)}, missing={ds.missing})")
    sampler = SceneBalancedLoRASampler(ds, args.batch_size, generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=args.device.startswith("cuda"), collate_fn=preference_lora_collate)

    # ---------- 模型：冻结 DiffPlanner + 注入 LoRA；冻结 CSPQ ----------
    model, config = load_plain_baseline(args.args_file, args.baseline_checkpoint, args.device)
    planner = StyleLoRAPlanner(model, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    planner = planner.to(torch.device(args.device))
    cspq = load_frozen_cspq(args.cspq_checkpoint, args.device)
    # 只让当前方向分支激活（另一分支参数不更新）
    style = DIRECTION_META[args.direction][2]
    planner.set_style(style).set_strength(1.0 if args.direction == "high" else -1.0)

    trainer = PreferenceLoRATrainer(
        planner, cspq, observation_normalizer=config.observation_normalizer,
        state_normalizer=config.state_normalizer, device=args.device,
        direction=args.direction, learning_rate=args.lr,
        lambda_n=args.lambda_n, lambda_z=args.lambda_z, lambda_s=args.lambda_s,
        lambda_dyn=args.lambda_dyn, lambda_q=args.lambda_q,
    )

    # ---------- 可选验证集 ----------
    val_loader = None
    val_ds = None
    if args.val_manifest:
        if not (args.val_latent_bank and args.val_latent_bank_index and args.val_feature_npy and args.val_feature_index):
            raise ValueError("提供 --val-manifest 时必须同时给 --val-latent-bank/--val-latent-bank-index/--val-feature-npy/--val-feature-index")
        val_ds = PreferenceLoRADataset(
            args.val_manifest, args.cache_root, args.val_latent_bank, args.val_latent_bank_index,
            args.val_feature_npy, args.val_feature_index, direction=args.direction,
            rank_low=rank_low, rank_high=rank_high,
        )
        if len(val_ds) == 0:
            raise ValueError("验证集方向窗口里无样本")
        if not val_ds.free_indices or not val_ds.car_indices:
            raise ValueError(
                f"验证集方向窗口场景不完整：free={len(val_ds.free_indices)}, car={len(val_ds.car_indices)}")
        print(f"[data] val samples={len(val_ds)} (missing={val_ds.missing})")
        val_sampler = SceneBalancedLoRASampler(val_ds, args.batch_size,
                                               generator=torch.Generator().manual_seed(args.seed))
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=args.workers,
                                pin_memory=args.device.startswith("cuda"), collate_fn=preference_lora_collate)

        def _validate_fixed() -> dict:
            # 验证集必须真正固定：重置采样器 generator，保证每次验证抽到相同样本。
            # 否则 DataLoader 每次迭代会让 generator 状态前进，不同验证轮次的
            # best checkpoint 基于不同样本集比较，不可严格复现。
            val_sampler.generator.manual_seed(args.seed)
            return trainer.validate(val_loader, seed=args.seed)

    # ---------- SwanLab（默认启动；必须放在 val_loader/val_ds 定义之后）----------
    swanlab_run = _start_swanlab(args)
    if swanlab_run is not None:
        trainable = sum(p.numel() for p in planner.parameters() if p.requires_grad)
        _swanlab_log(swanlab_run, {"data": {"train_samples": len(ds),
                                            "val_samples": len(val_ds) if val_loader is not None else 0,
                                            "free": len(ds.free_indices), "car": len(ds.car_indices),
                                            "trainable_parameters": trainable}}, step=0)
        print(f"[swanlab] project={args.swanlab_project} mode={args.swanlab_mode} trainable={trainable}")

    # ---------- 训练（有验证集则按期验证并选 best；否则退回单批 loss 近似）----------
    print(f"[train] direction={args.direction} steps={args.steps}")
    best_loss = float("inf")
    best_state = None
    step = 0
    last_val_step = -10**9
    while step < args.steps:
        for batch in loader:
            metrics = trainer.train_step(batch)
            step += 1
            if step % 10 == 0 or step == args.steps:
                print(f"  step {step}/{args.steps} "
                      f"loss={metrics['loss']:.4f} ego={metrics['ego_denoise']:.4f} "
                      f"mmd={metrics['mmd_z']:.4f} rank={metrics['rank_huber']:.4f} "
                      f"dyn={metrics['dynamics']:.4f} q={metrics['factor_huber']:.4f}")
                _swanlab_log(swanlab_run, {"train": {
                    "loss": metrics["loss"], "ego_denoise": metrics["ego_denoise"],
                    "neighbor": metrics["neighbor"], "mmd_z": metrics["mmd_z"],
                    "rank_huber": metrics["rank_huber"], "s_mean": metrics["s_mean"],
                    "s_std": metrics["s_std"], "dynamics": metrics["dynamics"],
                    "acceleration": metrics["acceleration"], "jerk": metrics["jerk"],
                    "factor_huber": metrics["factor_huber"]}}, step=step)
            if val_loader is not None and step % 100 == 0:
                candidate = _validate_fixed()
                last_val_step = step
                if candidate["loss"] < best_loss:
                    best_loss = candidate["loss"]
                    best_state = {k: v.detach().cpu().clone() for k, v in planner.state_dict().items()
                                  if f".{style}.lora_" in k}
                print(f"    [val] avg_loss={candidate['loss']:.4f}")
                _swanlab_log(swanlab_run, {"val": {
                    "loss": candidate["loss"], "ego_denoise": candidate["ego_denoise"],
                    "neighbor": candidate["neighbor"], "mmd_z": candidate["mmd_z"],
                    "rank_huber": candidate["rank_huber"], "dynamics": candidate["dynamics"],
                    "acceleration": candidate["acceleration"], "jerk": candidate["jerk"],
                    "factor_huber": candidate["factor_huber"]}}, step=step)
            elif val_loader is None and metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in planner.state_dict().items()
                              if f".{style}.lora_" in k}
            if step >= args.steps:
                break
    # 末次强制验证：最后 100 步内未验证过则补一次（固定 seed + 固定验证样本集）
    if val_loader is not None and args.steps - last_val_step >= 100:
        candidate = _validate_fixed()
        if candidate["loss"] < best_loss:
            best_loss = candidate["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in planner.state_dict().items()
                          if f".{style}.lora_" in k}
        print(f"    [val-final] avg_loss={candidate['loss']:.4f}")
        _swanlab_log(swanlab_run, {"val_final": {
            "loss": candidate["loss"], "ego_denoise": candidate["ego_denoise"],
            "neighbor": candidate["neighbor"], "mmd_z": candidate["mmd_z"],
            "rank_huber": candidate["rank_huber"], "dynamics": candidate["dynamics"],
            "acceleration": candidate["acceleration"], "jerk": candidate["jerk"],
            "factor_huber": candidate["factor_huber"]}}, step=args.steps)

    # ---------- 保存 adapter checkpoint（内部 aggressive/conservative 槽位，对外 high/low）----------
    if best_state is not None:
        planner.load_state_dict({k: v for k, v in best_state.items()}, strict=False)
    manifest_hash = _rows_hash(Path(args.manifest))
    save_adapter_checkpoint(
        args.output, planner, style=style,
        baseline_checkpoint=args.baseline_checkpoint,
        manifest_hash=manifest_hash,
        normalization_file=config.normalization_file_path,
        training_config=vars(args),
        best_validation={"loss": best_loss},
    )
    print(f"[save] adapter ({args.direction}) -> {args.output}  best_loss={best_loss:.4f}")
    if swanlab_run is not None:
        _swanlab_log(swanlab_run, {"summary": {"best_loss": best_loss, "steps": step}}, step=step)
        swanlab_run.finish()


if __name__ == "__main__":
    main()
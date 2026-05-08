# -*- coding: utf-8 -*-
"""
Diffusion Planner 训练主脚本（Hydra 入口）。

功能说明：
1. 读取 Hydra 配置并构建训练/验证数据；
2. 执行 diffusion 训练循环（含可选 EMA）；
3. 记录日志并保存最佳/周期模型。

维护约定：
- 本文件保留原训练逻辑，不改损失定义和训练流程。
- baseline 版本仅保留 diffusion 训练主线。
"""

import os
import sys
import logging
import torch
import hydra
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, Union, List

from torch import optim, nn
from torch.utils.data import Subset, DataLoader
from timm.utils import ModelEma
from mmengine.fileio import dump
from omegaconf import DictConfig
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 环境设置与路径修补
# -----------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HYDRA_FULL_ERROR"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# NuPlan Devkit 路径
sys.path.append("/home/lisw/programs/Nuplan-Baseline-3090/nuplan-devkit")
sys.path.append(str(REPO_ROOT))

HYDRA_CONFIG_PATH = str((SCRIPT_DIR / "config").resolve())
if not os.path.isdir(HYDRA_CONFIG_PATH):
    HYDRA_CONFIG_PATH = "/home/lisw/programs/Nuplan-Baseline-3090/nuplan_baseline/config"

# -----------------------------------------------------------------------------
# 本地模块导入
# -----------------------------------------------------------------------------
from baseline.train.train_utils import set_seed, resume_model
from baseline.utils.normalizer import ObservationNormalizer, StateNormalizer
from baseline.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from baseline.utils.logger import WandbLogger as Logger
from baseline.common.data_augmentation import StatePerturbation
from baseline.common.dataset import ClosedLoopPlannerData
from baseline.train.manage import flatten_config, save_model, manage_best_models

# 模型与 Loss
from baseline.model.diff_planner import diffusion_planner
from baseline.model.diff_planner.loss.diff_loss import diffusion_loss_func

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SKIP_VALIDATION = True  # 如果为 True，则跳过验证环节，并使用训练 Loss 保存最佳模型


# -----------------------------------------------------------------------------
# 1. Model Factory (模型构建工厂)
# -----------------------------------------------------------------------------
def build_model(args: Any) -> nn.Module:
    """
    根据配置名称构建对应的 Planner 模型实例。
    """
    if args.name == "diffusion-planner":
        return diffusion_planner.Diffusion_Planner(args)
    raise ValueError(
        f"Unknown model name: {args.name}. "
        "Current baseline only keeps diffusion-planner."
    )


# -----------------------------------------------------------------------------
# 2. Diffusion Data Preprocessing (数据预处理)
# -----------------------------------------------------------------------------
def _prepare_diffusion_batch(
        batch: Union[List, Tuple, Dict],
        args: Any,
        aug: Optional[StatePerturbation] = None
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    准备 Diffusion Planner 所需的输入数据。

    功能：
    1. 统一 List/Tuple 或 Dict 格式的 Batch 数据。
    2. 执行数据增强 (坐标系扰动)。
    3. 特征工程：将航向角 (heading) 转换为 (cos, sin)。
    4. 执行观测数据归一化 (Observation Normalization)。

    Args:
        batch: DataLoader 输出的原始 batch 数据。
        args: 全局配置参数。
        aug: 数据增强模块实例。

    Returns:
        inputs: 归一化后的模型输入字典。
        ego_future: 自车未来轨迹 GT [B, T, 4] (x, y, cos, sin)。
        neighbors_future: 他车未来轨迹 GT [B, N, T, 4] (x, y, cos, sin)。
        mask: 他车有效性 Mask [B, N]。
    """
    device = args.device if isinstance(args.device, str) else str(args.device)

    # --- A. 数据解包与设备移动 ---
    if isinstance(batch, (list, tuple)):
        # 兼容 ClosedLoopPlannerData 的 Tuple 返回格式
        inputs = {
            "ego_current_state": batch[0].to(device),
            "neighbor_agents_past": batch[2].to(device),
            "lanes": batch[4].to(device),
            "lanes_speed_limit": batch[5].to(device),
            "lanes_has_speed_limit": batch[6].to(device),
            "route_lanes": batch[7].to(device),
            "route_lanes_speed_limit": batch[8].to(device),
            "route_lanes_has_speed_limit": batch[9].to(device),
            "static_objects": batch[10].to(device),
        }
        ego_future = batch[1].to(device)  # Raw Meters
        neighbors_future = batch[3].to(device)  # Raw Meters

    elif isinstance(batch, dict):
        # 兼容 Dict 格式的 Dataset
        def _get(*keys):
            for k in keys:
                if k in batch: return batch[k]
            raise KeyError(f"Missing keys {keys} in batch. Keys: {list(batch.keys())[:5]}...")

        inputs = {
            "ego_current_state": _get("ego_current_state", "ego_current").to(device),
            "neighbor_agents_past": _get("neighbor_agents_past", "neighbors_past").to(device),
            "lanes": _get("lanes").to(device),
            "lanes_speed_limit": _get("lanes_speed_limit").to(device),
            "lanes_has_speed_limit": _get("lanes_has_speed_limit").to(device),
            "route_lanes": _get("route_lanes").to(device),
            "route_lanes_speed_limit": _get("route_lanes_speed_limit").to(device),
            "route_lanes_has_speed_limit": _get("route_lanes_has_speed_limit").to(device),
            "static_objects": _get("static_objects").to(device),
        }
        ego_future = _get("ego_future_gt", "ego_future").to(device)
        neighbors_future = _get("neighbors_future_gt", "neighbors_future").to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # --- B. 数据增强 (Augmentation) ---
    # 如果启用，会对所有空间坐标（History, Map, GT）进行随机旋转和平移，增强模型鲁棒性
    if aug is not None:
        inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    # --- C. 目标构建 (Target Construction) ---
    # Diffusion 模型通常预测 (x, y, cos, sin)，避免直接预测角度带来的周期性问题

    # 1. Ego Future: [B, T, 3] (x, y, h) -> [B, T, 4] (x, y, cos, sin)
    ego_future = torch.cat(
        [
            ego_future[..., :2],
            torch.stack([ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1),
        ],
        dim=-1,
    )

    # 2. Neighbors Future: 处理同上，并生成 Mask
    # Mask逻辑：如果 (x, y, h) 全为0，则视为无效 Agent
    mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0

    neighbors_future = torch.cat(
        [
            neighbors_future[..., :2],
            torch.stack([neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1),
        ],
        dim=-1,
    )
    # 将无效 Agent 的值强制置零
    neighbors_future[mask] = 0.0

    # --- D. 输入归一化 (Input Normalization) ---
    # 使用统计得到的均值/方差对输入特征进行标准化
    inputs = args.observation_normalizer(inputs)

    return inputs, ego_future, neighbors_future, mask


# -----------------------------------------------------------------------------
# 3. Training & Validation Loops (训练与验证循环)
# -----------------------------------------------------------------------------
def train_epoch_diffusion(
        data_loader: DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        args: Any,
        ema: Optional[ModelEma],
        aug: Optional[StatePerturbation] = None
) -> Tuple[Dict[str, float], float]:
    """
    Diffusion 模型的一个训练 Epoch。
    """
    model.train()
    epoch_losses = []

    # 使用 tqdm 包装 data_loader 以显示进度条
    pbar = tqdm(data_loader, desc="Training", dynamic_ncols=True)

    for batch in pbar:
        # 1. 数据准备
        inputs, ego_future, neighbors_future, mask = _prepare_diffusion_batch(batch, args, aug)

        # 2. 清零梯度
        optimizer.zero_grad(set_to_none=True)

        # 3. 前向传播与 Loss 计算
        # diffusion_loss_func 内部包含模型 Forward 过程 (加噪 -> 预测去噪 -> 计算误差)
        loss_dict = {}
        loss_dict, _ = diffusion_loss_func(
            model,
            inputs,
            model.sde.marginal_prob,  # SDE 边缘概率函数
            (ego_future, neighbors_future, mask),
            args.state_normalizer,  # 状态归一化器 (用于 Loss 计算时的反归一化或 GT 归一化)
            loss_dict,
            args.diffusion_model_type,  # 'score' or 'x_start'
        )

        # 4. Loss 聚合
        # 总 Loss = 邻居预测 Loss + alpha * 自车规划 Loss
        loss_dict["loss"] = loss_dict["neighbor_prediction_loss"] + args.alpha_planning_loss * loss_dict[
            "ego_planning_loss"]

        # 5. 反向传播
        loss_dict["loss"].backward()

        # 6. 梯度裁剪 (防止梯度爆炸)
        clip_norm = getattr(args, "grad_clip_norm", 5.0)
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        # 7. 参数更新与 EMA
        optimizer.step()
        if ema is not None:
            ema.update(model)

        # 8. 记录 Batch Loss (Detach from graph to save memory)
        current_loss = loss_dict["loss"].item()

        epoch_losses.append(
            {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in loss_dict.items()})

        pbar.set_postfix({"loss": f"{current_loss:.4f}"})

    mean_stats = {}
    if len(epoch_losses) > 0:
        keys = epoch_losses[0].keys()
        for k in keys:
            mean_stats[k] = sum(d.get(k, 0.0) for d in epoch_losses) / len(epoch_losses)

    return mean_stats, mean_stats.get("loss", 0.0)


@torch.no_grad()
def validate_epoch_diffusion(
        val_loader: DataLoader,
        model: nn.Module,
        args: Any,
        aug: Optional[StatePerturbation] = None
) -> Dict[str, float]:
    """
    Diffusion 模型的验证循环。
    注意：这里主要计算验证集上的 Loss 作为性能指标，而非采样后的 ADE/FDE (采样速度慢)。
    """
    model.eval()
    epoch_losses = []

    for batch in val_loader:
        inputs, ego_future, neighbors_future, mask = _prepare_diffusion_batch(batch, args, aug=None)

        loss_dict = {}
        loss_dict, _ = diffusion_loss_func(
            model,
            inputs,
            model.sde.marginal_prob,
            (ego_future, neighbors_future, mask),
            args.state_normalizer,
            loss_dict,
            args.diffusion_model_type,
        )

        loss_dict["loss"] = loss_dict["neighbor_prediction_loss"] + args.alpha_planning_loss * loss_dict[
            "ego_planning_loss"]

        epoch_losses.append(
            {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in loss_dict.items()})

    mean_stats = {}
    if len(epoch_losses) > 0:
        keys = epoch_losses[0].keys()
        for k in keys:
            mean_stats[k] = sum(d.get(k, 0.0) for d in epoch_losses) / len(epoch_losses)

    mean_stats.setdefault("brier_fde", mean_stats.get("loss", float("inf")))
    return mean_stats


# -----------------------------------------------------------------------------
# 4. Main Execution (主入口)
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=HYDRA_CONFIG_PATH, config_name="train")
def model_training(cfg: DictConfig):
    """
    主训练函数，由 Hydra 驱动。
    """
    cfg.distributed.ddp = False
    args = flatten_config(cfg)

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)

    global_rank = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = str(device)

    if global_rank == 0:
        if args.resume_model_path is None:
            current_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
            save_path = f"{cfg.save_dir}/train_log/{cfg.name}/{current_time}/"
        else:
            save_path = cfg.model.resume_model_path

        os.makedirs(save_path, exist_ok=True)

        args_dict = vars(args)
        args_dict = {k: (v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict())
                     for k, v in args_dict.items()}
        dump(args_dict, os.path.join(save_path, "args.json"), file_format="json", indent=4)
    else:
        save_path = None

    set_seed(cfg.seed)

    def prepare_dataset(is_train=True):
        dataset_name = cfg.data.train_set if is_train else getattr(cfg.data, "val_set", cfg.data.train_set)
        dataset_list = cfg.data.train_set_list if is_train else getattr(cfg.data, "val_set_list",
                                                                        cfg.data.train_set_list)

        ds = ClosedLoopPlannerData(
            dataset_name,
            dataset_list,
            cfg.data.agent_num,
            args.predicted_neighbor_num,
            cfg.data.future_len,
        )

        prefix = "train" if is_train else "val"
        start = getattr(cfg.training, f"{prefix}_start_index", 0 if is_train else 12)
        num = getattr(cfg.training, f"{prefix}_num_samples", None)

        if start > 0 or num is not None:
            end = min(start + num, len(ds)) if num else len(ds)
            ds = Subset(ds, list(range(start, end)))
        return ds

    train_set = prepare_dataset(True)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
        pin_memory=True
    )

    if not SKIP_VALIDATION:
        val_set = prepare_dataset(False)
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            drop_last=False,
            pin_memory=True
        )
    else:
        val_loader = None

    planner = build_model(args).to(device)
    model_ema = ModelEma(planner, decay=0.999, device=device) if getattr(args, "use_ema", False) else None

    optimizer = optim.AdamW([{"params": planner.parameters(), "lr": args.learning_rate}])
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, cfg.training.train_epochs, args.warm_up_epoch)

    init_epoch, wandb_id, best_score = 0, None, float("inf")
    if args.resume_model_path:
        planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(
            args.resume_model_path, planner, optimizer, scheduler, model_ema, device
        )

    wandb_logger = Logger(args.name, args.notes, args, wandb_resume_id=wandb_id, save_path=save_path, rank=global_rank)

    aug = StatePerturbation(augment_prob=cfg.data.augment_prob,
                            device=args.device) if cfg.data.use_data_augment else None

    train_epoch_fn = train_epoch_diffusion

    for epoch in range(init_epoch, cfg.training.train_epochs):
        if global_rank == 0:
            logger.info(f"Epoch {epoch + 1}/{cfg.training.train_epochs}")

        train_log, train_total_loss = train_epoch_fn(train_loader, planner, optimizer, args, model_ema, aug)

        if not SKIP_VALIDATION:
            val_metrics = validate_epoch_diffusion(val_loader, planner, args, aug=None)
        else:
            val_metrics = {}

        if global_rank == 0:
            wandb_logger.log_metrics({f"train/{k}": v for k, v in train_log.items()}, step=epoch + 1)
            if not SKIP_VALIDATION:
                wandb_logger.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch + 1)
            wandb_logger.log_metrics({"lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

            if SKIP_VALIDATION:
                cur_score = train_total_loss
                score_name = "train_loss"
            else:
                cur_score = val_metrics.get("brier_fde", float("inf"))
                score_name = "score"

            if cur_score < best_score:
                best_score = cur_score
                save_model(
                    planner,
                    optimizer,
                    scheduler,
                    save_path,
                    epoch,
                    train_total_loss,
                    wandb_logger.id,
                    model_ema.ema if model_ema else None,
                    filename=f"best_model-epoch_{epoch}-{score_name}_{best_score:.4f}.pth",
                )
                manage_best_models(save_path, max_num=5)

            periodic_interval = getattr(cfg.training, "save_periodic_interval", 20)
            if (epoch + 1) % periodic_interval == 0:
                save_model(
                    planner,
                    optimizer,
                    scheduler,
                    os.path.join(save_path, "periodic_models"),
                    epoch,
                    train_total_loss,
                    wandb_logger.id,
                    model_ema.ema if model_ema else None,
                    filename=f"periodic_epoch_{epoch + 1:03d}.pth",
                )

        scheduler.step()


def main() -> None:
    """脚本入口。"""
    model_training()


if __name__ == "__main__":
    main()

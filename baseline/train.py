# -*- coding: utf-8 -*-
"""
Baseline 训练主脚本（注册制）。

功能说明：
1. 通过 Hydra 加载训练配置；
2. 通过注册表按 `method.name` 动态选择模型与 train/val 循环；
3. 保持 Diffusion 原训练逻辑不变，同时支持 Wayformer 训练接入。

当前支持：
- diffusion-planner
- wayformer

训练主流程（从上到下）：
1. 解析配置并构造 normalizer；
2. 按 method 注册表构建模型与 train/val epoch 函数；
3. 构建数据集与 dataloader；
4. 执行 epoch 循环，记录日志并保存 best/periodic checkpoint；
5. 统一释放日志后端资源（tensorboard / swanlab / wandb）。
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import torch
from mmengine.fileio import dump
from omegaconf import DictConfig
from timm.utils import ModelEma
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 环境设置与路径修补
# -----------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["HYDRA_FULL_ERROR"] = "0"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _append_if_exists(path: Path) -> None:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


# 优先使用当前工程内的 nuplan-devkit
_append_if_exists(REPO_ROOT / "nuplan-devkit")
_append_if_exists(REPO_ROOT)

HYDRA_CONFIG_PATH = str((SCRIPT_DIR / "config").resolve())

# -----------------------------------------------------------------------------
# 本地模块导入
# -----------------------------------------------------------------------------
from baseline.common.data_augmentation import StatePerturbation
from baseline.common.dataset import ClosedLoopPlannerData
from baseline.core.register import Registry
from baseline.model.diff_planner import diffusion_planner
from baseline.model.diff_planner.loss.diff_loss import diffusion_loss_func
from baseline.model.style_planner import diffusion_planner as style_diffusion_planner
from baseline.model.wayformer.wayf_planner import WayFormer
from baseline.train.manage import flatten_config, manage_best_models, save_model
from baseline.train.train_utils import resume_model, set_seed
from baseline.utils.logger import WandbLogger as Logger
from baseline.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from baseline.utils.normalizer import ObservationNormalizer, StateNormalizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 验证开关（建议保持 False）：
# - False: 每轮执行验证并基于 val 指标保存 best，结果更稳健，适合正式训练。
# - True:  跳过验证，按 train loss 保存 best，速度更快，适合快速冒烟调试。
SKIP_VALIDATION = False


@dataclass(frozen=True)
class TrainMethodSpec:
    """不同方法对应的训练入口描述。"""

    model_builder: Any
    train_epoch_fn: Any
    validate_epoch_fn: Any


TRAIN_METHOD_REGISTRY: Registry[TrainMethodSpec] = Registry("train_method")


def _batch_to_common_inputs(
    batch: Union[List, Tuple, Dict[str, torch.Tensor]],
    device: str,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """将 DataLoader batch 统一解包为基础输入 + GT。"""

    if isinstance(batch, (list, tuple)):
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
        ego_future = batch[1].to(device)
        neighbors_future = batch[3].to(device)
    elif isinstance(batch, dict):
        def _get(*keys):
            for key in keys:
                if key in batch:
                    return batch[key]
            raise KeyError(f"Missing keys {keys}. Batch keys: {list(batch.keys())[:8]} ...")

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

    return inputs, ego_future, neighbors_future


# -----------------------------------------------------------------------------
# Diffusion 数据预处理
# -----------------------------------------------------------------------------
def _prepare_diffusion_batch(
    batch: Union[List, Tuple, Dict[str, torch.Tensor]],
    args: Any,
    aug: Optional[StatePerturbation] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    准备 Diffusion Planner 所需输入，保持与历史训练逻辑一致。
    """
    device = args.device if isinstance(args.device, str) else str(args.device)
    inputs, ego_future, neighbors_future = _batch_to_common_inputs(batch, device)

    if aug is not None:
        inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    ego_future = torch.cat(
        [
            ego_future[..., :2],
            torch.stack([ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1),
        ],
        dim=-1,
    )

    mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
    neighbors_future = torch.cat(
        [
            neighbors_future[..., :2],
            torch.stack([neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1),
        ],
        dim=-1,
    )
    neighbors_future[mask] = 0.0

    inputs = args.observation_normalizer(inputs)
    return inputs, ego_future, neighbors_future, mask


def _prepare_wayformer_batch(
    batch: Union[List, Tuple, Dict[str, torch.Tensor]],
    args: Any,
    aug: Optional[StatePerturbation] = None,
) -> Dict[str, torch.Tensor]:
    """
    准备 Wayformer 所需输入。

    关键点：
    - 先执行增强与归一化；
    - 再把米制 GT 覆盖回 inputs，供 compute_loss / compute_metrics 正确计算。
    """
    device = args.device if isinstance(args.device, str) else str(args.device)
    inputs, ego_future_gt, neighbors_future_gt = _batch_to_common_inputs(batch, device)

    if isinstance(batch, (list, tuple)):
        if len(batch) <= 15:
            raise ValueError(
                "Wayformer training requires dataset fields [11..15] "
                "(ego_agent_past / masks). Current batch length is too short."
            )
        inputs.update(
            {
                "ego_agent_past": batch[11].to(device),
                "neighbor_agents_past_mask": batch[12].to(device),
                "neighbor_agents_future_mask": batch[13].to(device),
                "lanes_mask": batch[14].to(device),
                "route_lanes_mask": batch[15].to(device),
            }
        )
    elif isinstance(batch, dict):
        def _try_get(key: str):
            return batch[key].to(device) if key in batch else None

        for key in [
            "ego_agent_past",
            "neighbor_agents_past_mask",
            "neighbor_agents_future_mask",
            "lanes_mask",
            "route_lanes_mask",
        ]:
            value = _try_get(key)
            if value is not None:
                inputs[key] = value

        if "ego_agent_past" not in inputs:
            raise ValueError("Wayformer training requires `ego_agent_past` in batch inputs.")
        if "neighbor_agents_future_mask" not in inputs:
            # 兜底：如果数据里没有该 mask，则按坐标非零生成一个
            valid = torch.sum(torch.abs(neighbors_future_gt[..., :2]), dim=-1) > 1e-4
            inputs["neighbor_agents_future_mask"] = valid.float()
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if aug is not None:
        inputs, ego_future_gt, neighbors_future_gt = aug(inputs, ego_future_gt, neighbors_future_gt)

    inputs["ego_future_gt"] = ego_future_gt
    inputs["neighbors_future_gt"] = neighbors_future_gt
    if "ego_future_mask" not in inputs:
        bsz, tlen, _ = ego_future_gt.shape
        inputs["ego_future_mask"] = torch.ones((bsz, tlen), dtype=torch.float32, device=device)

    inputs = args.observation_normalizer(inputs)

    # 恢复米制 GT（compute_loss / metrics / safety 相关逻辑依赖真实物理单位）
    inputs["ego_future_gt"] = ego_future_gt
    inputs["neighbors_future_gt"] = neighbors_future_gt
    inputs["neighbor_agents_future"] = neighbors_future_gt
    return inputs


# -----------------------------------------------------------------------------
# 训练与验证循环：Diffusion
# -----------------------------------------------------------------------------
def train_epoch_diffusion(
    data_loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    args: Any,
    ema: Optional[ModelEma],
    aug: Optional[StatePerturbation] = None,
) -> Tuple[Dict[str, float], float]:
    model.train()
    epoch_losses = []
    pbar = tqdm(data_loader, desc="Training", dynamic_ncols=True)

    for batch in pbar:
        inputs, ego_future, neighbors_future, mask = _prepare_diffusion_batch(batch, args, aug)

        optimizer.zero_grad(set_to_none=True)
        loss_dict: Dict[str, Any] = {}
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
            "ego_planning_loss"
        ]
        loss_dict["loss"].backward()

        clip_norm = getattr(args, "grad_clip_norm", 5.0)
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        epoch_losses.append(
            {key: float(val.detach().cpu()) if torch.is_tensor(val) else float(val) for key, val in loss_dict.items()}
        )
        pbar.set_postfix({"loss": f"{loss_dict['loss'].item():.4f}"})

    mean_stats: Dict[str, float] = {}
    if epoch_losses:
        for key in epoch_losses[0].keys():
            mean_stats[key] = sum(loss_item.get(key, 0.0) for loss_item in epoch_losses) / len(epoch_losses)

    return mean_stats, mean_stats.get("loss", 0.0)


@torch.no_grad()
def validate_epoch_diffusion(
    val_loader: DataLoader,
    model: nn.Module,
    args: Any,
    aug: Optional[StatePerturbation] = None,
) -> Dict[str, float]:
    model.eval()
    epoch_losses = []

    for batch in val_loader:
        inputs, ego_future, neighbors_future, mask = _prepare_diffusion_batch(batch, args, aug=None)
        loss_dict: Dict[str, Any] = {}
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
            "ego_planning_loss"
        ]
        epoch_losses.append(
            {key: float(val.detach().cpu()) if torch.is_tensor(val) else float(val) for key, val in loss_dict.items()}
        )

    mean_stats: Dict[str, float] = {}
    if epoch_losses:
        for key in epoch_losses[0].keys():
            mean_stats[key] = sum(loss_item.get(key, 0.0) for loss_item in epoch_losses) / len(epoch_losses)
    mean_stats.setdefault("brier_fde", mean_stats.get("loss", float("inf")))
    return mean_stats


# -----------------------------------------------------------------------------
# 训练与验证循环：Wayformer
# -----------------------------------------------------------------------------
def train_epoch_wayformer(
    data_loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    args: Any,
    ema: Optional[ModelEma],
    aug: Optional[StatePerturbation] = None,
) -> Tuple[Dict[str, float], float]:
    model.train()
    epoch_logs: List[Dict[str, float]] = []
    pbar = tqdm(data_loader, desc="Training", dynamic_ncols=True)

    for batch in pbar:
        inputs = _prepare_wayformer_batch(batch, args, aug)
        optimizer.zero_grad(set_to_none=True)

        model_output = model(inputs)
        loss_dict = model.compute_loss(model_output, inputs)
        total_loss = loss_dict["loss"]
        total_loss.backward()

        clip_norm = getattr(args, "grad_clip_norm", 5.0)
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        log_dict = {key: float(val.detach().cpu()) if torch.is_tensor(val) else float(val) for key, val in loss_dict.items()}
        if hasattr(model, "compute_metrics"):
            metric_dict = model.compute_metrics(model_output, inputs)
            metric_dict = {
                key: float(val.detach().cpu()) if torch.is_tensor(val) else float(val)
                for key, val in metric_dict.items()
            }
            log_dict.update(metric_dict)

        epoch_logs.append(log_dict)
        pbar.set_postfix({"loss": f"{float(total_loss.detach().cpu()):.4f}"})

    mean_stats: Dict[str, float] = {}
    if epoch_logs:
        for key in epoch_logs[0].keys():
            mean_stats[key] = sum(item.get(key, 0.0) for item in epoch_logs) / len(epoch_logs)

    return mean_stats, mean_stats.get("loss", 0.0)


@torch.no_grad()
def validate_epoch_wayformer(
    val_loader: DataLoader,
    model: nn.Module,
    args: Any,
    aug: Optional[StatePerturbation] = None,
) -> Dict[str, float]:
    model.eval()
    epoch_logs: List[Dict[str, float]] = []

    for batch in val_loader:
        inputs = _prepare_wayformer_batch(batch, args, aug=None)
        model_output = model(inputs)
        loss_dict = model.compute_loss(model_output, inputs)
        metric_dict = model.compute_metrics(model_output, inputs) if hasattr(model, "compute_metrics") else {}

        merged = {}
        merged.update(
            {key: float(val.detach().cpu()) if torch.is_tensor(val) else float(val) for key, val in loss_dict.items()}
        )
        merged.update(
            {key: float(val.detach().cpu()) if torch.is_tensor(val) else float(val) for key, val in metric_dict.items()}
        )
        epoch_logs.append(merged)

    mean_stats: Dict[str, float] = {}
    if epoch_logs:
        for key in epoch_logs[0].keys():
            mean_stats[key] = sum(item.get(key, 0.0) for item in epoch_logs) / len(epoch_logs)
    mean_stats.setdefault("brier_fde", mean_stats.get("loss", float("inf")))
    return mean_stats


def _register_train_methods() -> None:
    """集中注册可训练方法，避免主流程里写分支判断。"""
    if len(TRAIN_METHOD_REGISTRY) > 0:
        return

    TRAIN_METHOD_REGISTRY.register(
        "diffusion-planner",
        TrainMethodSpec(
            model_builder=lambda args: diffusion_planner.Diffusion_Planner(args),
            train_epoch_fn=train_epoch_diffusion,
            validate_epoch_fn=validate_epoch_diffusion,
        ),
    )
    TRAIN_METHOD_REGISTRY.register(
        "style-planner",
        TrainMethodSpec(
            model_builder=lambda args: style_diffusion_planner.Diffusion_Planner(args),
            train_epoch_fn=train_epoch_diffusion,
            validate_epoch_fn=validate_epoch_diffusion,
        ),
    )
    TRAIN_METHOD_REGISTRY.register(
        "wayformer",
        TrainMethodSpec(
            model_builder=lambda args: WayFormer(args),
            train_epoch_fn=train_epoch_wayformer,
            validate_epoch_fn=validate_epoch_wayformer,
        ),
    )


def _build_method_spec(args: Any) -> TrainMethodSpec:
    _register_train_methods()
    return TRAIN_METHOD_REGISTRY.get(args.name)


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_PATH, config_name="train")
def model_training(cfg: DictConfig):
    """
    主训练函数（Hydra 入口）。

    说明：
    - 本函数不区分具体模型实现，模型差异通过注册表解耦；
    - `SKIP_VALIDATION` 会影响 best 模型选取依据（train loss vs val score）。
    """
    cfg.distributed.ddp = False
    args = flatten_config(cfg)

    # 从 normalization JSON 构造状态/观测归一化器
    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)

    global_rank = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = str(device)
    args.online_logger = str(getattr(args, "online_logger", "swanlab")).strip().lower()
    args.use_online_logger = bool(getattr(args, "use_online_logger", getattr(args, "use_wandb", True)))
    # 兼容 utils/logger.py 的历史字段读取逻辑
    args.use_wandb = args.use_online_logger

    # 根据 method.name 选择模型构建函数与 train/val epoch 函数
    method_spec = _build_method_spec(args)
    if args.name == "wayformer" and getattr(args, "agent_num", None) != getattr(args, "predicted_neighbor_num", None):
        raise ValueError(
            "Wayformer expects `data.agent_num == method.model.predicted_neighbor_num` "
            "for neighbor prediction supervision. "
            f"Got agent_num={getattr(args, 'agent_num', None)}, "
            f"predicted_neighbor_num={getattr(args, 'predicted_neighbor_num', None)}."
        )

    # rank0 负责创建输出目录并落盘完整训练参数快照
    if global_rank == 0:
        if args.resume_model_path is None:
            current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            save_path = f"{cfg.save_dir}/train_log/{cfg.name}/{current_time}/"
        else:
            save_path = cfg.model.resume_model_path

        os.makedirs(save_path, exist_ok=True)

        args_dict = vars(args)
        args_dict = {
            key: (val if not isinstance(val, (StateNormalizer, ObservationNormalizer)) else val.to_dict())
            for key, val in args_dict.items()
        }
        dump(args_dict, os.path.join(save_path, "args.json"), file_format="json", indent=4)
    else:
        save_path = None

    set_seed(cfg.seed)

    def prepare_dataset(is_train: bool = True):
        """按 train/val 切分参数构建数据集，并支持区间采样。"""
        dataset_name = cfg.data.train_set if is_train else getattr(cfg.data, "val_set", cfg.data.train_set)
        dataset_list = cfg.data.train_set_list if is_train else getattr(cfg.data, "val_set_list", cfg.data.train_set_list)

        dataset = ClosedLoopPlannerData(
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
            end = min(start + num, len(dataset)) if num else len(dataset)
            dataset = Subset(dataset, list(range(start, end)))
        return dataset

    # 训练集 dataloader：shuffle=True, drop_last=True（保证 batch 形状稳定）
    train_set = prepare_dataset(True)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # 验证集 dataloader：仅在不跳过验证时构建，避免无效开销
    if not SKIP_VALIDATION:
        val_set = prepare_dataset(False)
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            drop_last=False,
            pin_memory=True,
        )
    else:
        val_loader = None

    # 构建模型与优化组件（EMA 可选）
    planner = method_spec.model_builder(args).to(device)
    model_ema = ModelEma(planner, decay=0.999, device=device) if getattr(args, "use_ema", False) else None

    optimizer = optim.AdamW([{"params": planner.parameters(), "lr": args.learning_rate}])
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, cfg.training.train_epochs, args.warm_up_epoch)

    # checkpoint 恢复：包含模型、优化器、调度器、EMA、线上日志 run id
    init_epoch, wandb_id, best_score = 0, None, float("inf")
    if args.resume_model_path:
        planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(
            args.resume_model_path, planner, optimizer, scheduler, model_ema, device
        )

    wandb_logger = Logger(args.name, args.notes, args, wandb_resume_id=wandb_id, save_path=save_path, rank=global_rank)
    aug = StatePerturbation(augment_prob=cfg.data.augment_prob, device=args.device) if cfg.data.use_data_augment else None

    # 取出当前方法绑定的 train/val 执行函数
    train_epoch_fn = method_spec.train_epoch_fn
    validate_epoch_fn = method_spec.validate_epoch_fn

    try:
        for epoch in range(init_epoch, cfg.training.train_epochs):
            if global_rank == 0:
                logger.info("Epoch %s/%s", epoch + 1, cfg.training.train_epochs)

            # 1) Train
            train_log, train_total_loss = train_epoch_fn(train_loader, planner, optimizer, args, model_ema, aug)
            # 2) Validation（可选）
            val_metrics = validate_epoch_fn(val_loader, planner, args, aug=None) if not SKIP_VALIDATION else {}

            if global_rank == 0:
                wandb_logger.log_metrics({f"train/{key}": val for key, val in train_log.items()}, step=epoch + 1)
                if not SKIP_VALIDATION:
                    wandb_logger.log_metrics({f"val/{key}": val for key, val in val_metrics.items()}, step=epoch + 1)
                wandb_logger.log_metrics({"lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

                # 3) 定义 best 选取指标：
                #    - 跳过验证：使用训练 loss
                #    - 执行验证：使用 val brier_fde
                if SKIP_VALIDATION:
                    cur_score = train_total_loss
                    score_name = "train_loss"
                else:
                    cur_score = val_metrics.get("brier_fde", float("inf"))
                    score_name = "score"

                # 4) 保存 best checkpoint（越小越好）
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

                # 5) 周期性保存 checkpoint，便于中途回溯实验状态
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

            # 6) 每个 epoch 结束后推进学习率调度器
            scheduler.step()
    finally:
        # 统一收尾，确保 wandb/swanlab/tensorboard 句柄被正确关闭
        wandb_logger.finish()


def main() -> None:
    """脚本入口。"""
    model_training()


if __name__ == "__main__":
    main()

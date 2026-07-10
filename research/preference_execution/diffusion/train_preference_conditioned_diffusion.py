"""Standalone training entrypoint for preference-conditioned diffusion planning."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from timm.utils import ModelEma
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
for _path in (REPO_ROOT, DEVKIT_ROOT):
    _path_str = str(_path)
    if _path.exists() and _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from research._runtime import ensure_repo_on_path

ensure_repo_on_path()

from baseline.model.style_planner.diffusion_planner import Diffusion_Planner
from baseline.model.style_planner.layer.decoder import Decoder
from baseline.model.style_planner.loss.diff_loss import diffusion_loss_func
from baseline.train.manage import save_model
from baseline.train.train_utils import resume_model, set_seed
from baseline.utils.logger import WandbLogger as Logger
from baseline.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from baseline.utils.normalizer import ObservationNormalizer, StateNormalizer
from research._runtime import DEFAULT_CACHE_TRAIN_VAL_DIR, DEFAULT_NUM_WORKERS, DEFAULT_RECORD_ROOT
from research.preference_execution.diffusion.dataset import PreferenceConditionedPlannerData
from research.preference_execution.diffusion.preference_loss import compute_preference_aux_losses
from research.preference_execution.diffusion.style_condition import (
    STYLE_CONDITION_FEATURE_SET_CHOICES,
    global_style_condition_dim,
    phase_style_condition_dim,
    phase_style_flat_dim,
    phase_style_num_phases,
    style_condition_dim,
    use_temporal_style_gate,
    validate_style_condition_args,
)
from research.preference_execution.diffusion.training import (
    build_experiment_dir,
    prepare_preference_conditioned_batch,
    serializable_args_dict,
    write_json,
)
from research.style_scene_split.defaults import (
    DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR,
    DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR,
)

EXPERIMENT_PRESET_MANUAL = "manual"
EXPERIMENT_PRESET_BASELINE_9D = "baseline_9d"
EXPERIMENT_PRESET_EXEC_V2_CONDITION_ONLY = "exec_v2_condition_only"
EXPERIMENT_PRESET_PHASEWISE_EXEC_V1_CONDITION_ONLY = "phasewise_exec_v1_condition_only"
EXPERIMENT_PRESET_TWO_STAGE_EXEC_V1_PREF_LOSS = "two_stage_exec_v1_pref_loss"
EXPERIMENT_PRESET_SAMPLER_ONLY_MILD = "sampler_only_mild"
EXPERIMENT_PRESET_EXEC_V2_MILD_SAMPLER = "exec_v2_mild_sampler"

EXPERIMENT_PRESET_CHOICES = (
    EXPERIMENT_PRESET_MANUAL,
    EXPERIMENT_PRESET_BASELINE_9D,
    EXPERIMENT_PRESET_EXEC_V2_CONDITION_ONLY,
    EXPERIMENT_PRESET_PHASEWISE_EXEC_V1_CONDITION_ONLY,
    EXPERIMENT_PRESET_TWO_STAGE_EXEC_V1_PREF_LOSS,
    EXPERIMENT_PRESET_SAMPLER_ONLY_MILD,
    EXPERIMENT_PRESET_EXEC_V2_MILD_SAMPLER,
)

PRESET_DEFAULT_EXPERIMENT_NAMES = {
    EXPERIMENT_PRESET_BASELINE_9D: "effective_preference_global_vec_baseline_9d",
    EXPERIMENT_PRESET_EXEC_V2_CONDITION_ONLY: "effective_preference_global_vec_exec_v2_condition_only",
    EXPERIMENT_PRESET_PHASEWISE_EXEC_V1_CONDITION_ONLY: "effective_preference_global_vec_phasewise_exec_v1_condition_only",
    EXPERIMENT_PRESET_TWO_STAGE_EXEC_V1_PREF_LOSS: "effective_preference_global_vec_two_stage_exec_v1_pref_loss",
    EXPERIMENT_PRESET_SAMPLER_ONLY_MILD: "effective_preference_global_vec_sampler_only_mild",
    EXPERIMENT_PRESET_EXEC_V2_MILD_SAMPLER: "effective_preference_global_vec_exec_v2_mild_sampler",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train preference-conditioned diffusion planner.")
    parser.add_argument("--train_split_root", default=DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR)
    parser.add_argument("--val_split_root", default=DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR)
    parser.add_argument("--train_cache_dir", default=str(DEFAULT_CACHE_TRAIN_VAL_DIR))
    parser.add_argument("--val_cache_dir", default=str(DEFAULT_CACHE_TRAIN_VAL_DIR))
    parser.add_argument(
        "--normalization_file_path",
        default=str(REPO_ROOT / "baseline" / "resources" / "normalization_train.json"),
    )
    parser.add_argument(
        "--save_dir",
        default=str(Path(DEFAULT_RECORD_ROOT) / "research_train" / "preference_conditioned_diffusion"),
    )
    parser.add_argument("--experiment_name", default="effective_preference_global_vec")
    parser.add_argument(
        "--experiment_preset",
        default=EXPERIMENT_PRESET_MANUAL,
        choices=EXPERIMENT_PRESET_CHOICES,
        help="Canonical preset used to reproduce baseline/ablation runs without manually retyping every flag.",
    )
    parser.add_argument(
        "--condition_field",
        default="effective_preference_global_vec",
        choices=(
            "effective_preference_global_vec",
            "safe_preference_global_vec",
            "target_preference_global_vec",
        ),
    )
    parser.add_argument(
        "--style_condition_feature_set",
        default="global_only",
        choices=STYLE_CONDITION_FEATURE_SET_CHOICES,
        help="Feature layout passed into the diffusion decoder for preference control.",
    )
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.15)
    parser.add_argument("--cfg_guidance_scale", type=float, default=1.5)
    parser.add_argument("--two_stage_split_ratio", type=float, default=0.45)
    parser.add_argument("--two_stage_transition_ratio", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--notes", default="")
    parser.add_argument("--online_logger", default="swanlab", choices=("swanlab", "wandb", "disabled"))
    parser.add_argument("--use_online_logger", action="store_true", default=True)
    parser.add_argument("--disable_online_logger", action="store_true")
    parser.add_argument("--online_project_name", default="Style-Planner")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warm_up_epoch", type=int, default=5)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--alpha_planning_loss", type=float, default=1.0)
    parser.add_argument("--preference_aux_loss_weight", type=float, default=0.0)
    parser.add_argument("--temporal_near_loss_weight", type=float, default=0.0)
    parser.add_argument("--temporal_gate_loss_weight", type=float, default=0.0)
    parser.add_argument("--temporal_gate_target_loss_weight", type=float, default=0.0)
    parser.add_argument("--temporal_gate_order_loss_weight", type=float, default=0.0)
    parser.add_argument("--temporal_gate_order_margin", type=float, default=0.02)
    parser.add_argument("--preference_loss_dt", type=float, default=0.1)
    parser.add_argument("--temporal_gate_hidden_dim", type=int, default=192)
    parser.add_argument("--two_stage_far_recovery_mix", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--use_data_augment", action="store_true", default=False)
    parser.add_argument("--augment_prob", type=float, default=0.0)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--disable_ema", action="store_true")
    parser.add_argument("--resume_model_path", default=None)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--train_start_index", type=int, default=0)
    parser.add_argument("--train_num_samples", type=int, default=None)
    parser.add_argument("--val_start_index", type=int, default=0)
    parser.add_argument("--val_num_samples", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument(
        "--balance_scene_buckets",
        action="store_true",
        default=False,
        help="Use a weighted sampler to upweight underrepresented scene buckets during training.",
    )
    parser.add_argument("--scene_weight_free_drive", type=float, default=6.0)
    parser.add_argument("--scene_weight_car_follow", type=float, default=1.5)
    parser.add_argument("--scene_weight_lane_change", type=float, default=1.0)
    parser.add_argument("--scene_weight_default", type=float, default=1.0)

    parser.add_argument("--time_len", type=int, default=21)
    parser.add_argument("--future_len", type=int, default=80)
    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--predicted_neighbor_num", type=int, default=10)
    parser.add_argument("--static_objects_num", type=int, default=5)
    parser.add_argument("--static_objects_state_dim", type=int, default=10)
    parser.add_argument("--lane_len", type=int, default=20)
    parser.add_argument("--lane_num", type=int, default=70)
    parser.add_argument("--route_num", type=int, default=25)

    parser.add_argument("--encoder_depth", type=int, default=3)
    parser.add_argument("--decoder_depth", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--encoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--decoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--diffusion_model_type", default="x_start", choices=("score", "x_start"))
    return parser


def _apply_experiment_preset(
    args: argparse.Namespace,
    *,
    default_experiment_name: str,
) -> None:
    preset = str(args.experiment_preset)
    if preset == EXPERIMENT_PRESET_MANUAL:
        return

    preset_overrides: Dict[str, Any]
    if preset == EXPERIMENT_PRESET_BASELINE_9D:
        preset_overrides = {
            "style_condition_feature_set": "global_only",
            "balance_scene_buckets": False,
            "scene_weight_free_drive": 1.0,
            "scene_weight_car_follow": 1.0,
            "scene_weight_lane_change": 1.0,
            "scene_weight_default": 1.0,
        }
    elif preset == EXPERIMENT_PRESET_EXEC_V2_CONDITION_ONLY:
        preset_overrides = {
            "style_condition_feature_set": "exec_v2_effective_gap",
            "balance_scene_buckets": False,
            "scene_weight_free_drive": 1.0,
            "scene_weight_car_follow": 1.0,
            "scene_weight_lane_change": 1.0,
            "scene_weight_default": 1.0,
        }
    elif preset == EXPERIMENT_PRESET_PHASEWISE_EXEC_V1_CONDITION_ONLY:
        preset_overrides = {
            "style_condition_feature_set": "phasewise_exec_v1",
            "balance_scene_buckets": False,
            "scene_weight_free_drive": 1.0,
            "scene_weight_car_follow": 1.0,
            "scene_weight_lane_change": 1.0,
            "scene_weight_default": 1.0,
        }
    elif preset == EXPERIMENT_PRESET_TWO_STAGE_EXEC_V1_PREF_LOSS:
        preset_overrides = {
            "style_condition_feature_set": "two_stage_exec_v1",
            "balance_scene_buckets": False,
            "scene_weight_free_drive": 1.0,
            "scene_weight_car_follow": 1.0,
            "scene_weight_lane_change": 1.0,
            "scene_weight_default": 1.0,
            "preference_aux_loss_weight": 0.20,
            "temporal_near_loss_weight": 0.05,
            "temporal_gate_loss_weight": 0.05,
            "temporal_gate_target_loss_weight": 0.05,
            "temporal_gate_order_loss_weight": 0.02,
            "temporal_gate_order_margin": 0.02,
            "two_stage_split_ratio": 0.45,
            "two_stage_transition_ratio": 0.18,
            "two_stage_far_recovery_mix": 0.50,
        }
    elif preset == EXPERIMENT_PRESET_SAMPLER_ONLY_MILD:
        preset_overrides = {
            "style_condition_feature_set": "global_only",
            "balance_scene_buckets": True,
            "scene_weight_free_drive": 2.5,
            "scene_weight_car_follow": 1.25,
            "scene_weight_lane_change": 1.0,
            "scene_weight_default": 1.0,
        }
    elif preset == EXPERIMENT_PRESET_EXEC_V2_MILD_SAMPLER:
        preset_overrides = {
            "style_condition_feature_set": "exec_v2_effective_gap",
            "balance_scene_buckets": True,
            "scene_weight_free_drive": 2.5,
            "scene_weight_car_follow": 1.25,
            "scene_weight_lane_change": 1.0,
            "scene_weight_default": 1.0,
        }
    else:
        raise ValueError(f"Unsupported experiment_preset={preset!r}.")

    for key, value in preset_overrides.items():
        setattr(args, key, value)

    if args.experiment_name == default_experiment_name:
        args.experiment_name = PRESET_DEFAULT_EXPERIMENT_NAMES[preset]


def _build_args() -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args()
    default_experiment_name = parser.get_default("experiment_name")
    if args.disable_pin_memory:
        args.pin_memory = False
    if args.disable_ema:
        args.use_ema = False
    if args.disable_online_logger:
        args.use_online_logger = False
    _apply_experiment_preset(args, default_experiment_name=default_experiment_name)
    validate_style_condition_args(args.condition_field, args.style_condition_feature_set)
    args.name = "style-planner"
    args.guidance_fn = None
    args.global_style_condition_dim = global_style_condition_dim(args.style_condition_feature_set, base_global_dim=9)
    args.phase_style_condition_dim = phase_style_condition_dim(args.style_condition_feature_set)
    args.phase_style_num_phases = phase_style_num_phases(args.style_condition_feature_set)
    args.phase_style_flat_dim = phase_style_flat_dim(args.style_condition_feature_set)
    args.style_value_dim = style_condition_dim(args.style_condition_feature_set, base_global_dim=9)
    args.use_style_condition = True
    args.use_phase_style_condition = bool(args.phase_style_flat_dim > 0)
    args.use_temporal_style_gate = use_temporal_style_gate(args.style_condition_feature_set)
    args.use_wandb = args.use_online_logger
    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    return args


def _scene_bucket_weight(scene_bucket: str, args: argparse.Namespace) -> float:
    if scene_bucket == "straight_free_drive":
        return float(args.scene_weight_free_drive)
    if scene_bucket == "straight_car_follow":
        return float(args.scene_weight_car_follow)
    if scene_bucket == "straight_lane_change":
        return float(args.scene_weight_lane_change)
    return float(args.scene_weight_default)


def _build_train_sampler(
    dataset: PreferenceConditionedPlannerData,
    args: argparse.Namespace,
) -> WeightedRandomSampler | None:
    if not args.balance_scene_buckets:
        return None

    weights = [
        _scene_bucket_weight(str(record.get("scene_bucket", "")), args)
        for record in dataset.records
    ]
    if not weights:
        return None

    weight_tensor = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weight_tensor,
        num_samples=len(weights),
        replacement=True,
    )


def _prepare_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset = PreferenceConditionedPlannerData(
        cache_dir=args.train_cache_dir,
        split_root=args.train_split_root,
        condition_field=args.condition_field,
        start_index=args.train_start_index,
        num_samples=args.train_num_samples,
    )
    val_dataset = PreferenceConditionedPlannerData(
        cache_dir=args.val_cache_dir,
        split_root=args.val_split_root,
        condition_field=args.condition_field,
        start_index=args.val_start_index,
        num_samples=args.val_num_samples,
    )
    train_sampler = _build_train_sampler(train_dataset, args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.pin_memory,
    )
    return train_loader, val_loader


def _train_epoch(
    data_loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    ema: ModelEma | None,
    aug: Any | None,
) -> Dict[str, float]:
    model.train()
    logs = []
    pbar = tqdm(data_loader, desc="Train", dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar, start=1):
        inputs, ego_future, neighbors_future, mask, prep_log = prepare_preference_conditioned_batch(
            batch,
            args,
            train=True,
            aug=aug,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_dict: Dict[str, Any] = {}
        loss_dict, decoder_output = diffusion_loss_func(
            model,
            inputs,
            model.sde.marginal_prob,
            (ego_future, neighbors_future, mask),
            args.state_normalizer,
            loss_dict,
            args.diffusion_model_type,
        )
        diffusion_base_loss = (
            loss_dict["neighbor_prediction_loss"] + args.alpha_planning_loss * loss_dict["ego_planning_loss"]
        )
        loss_dict["diffusion_base_loss"] = diffusion_base_loss
        pref_metrics = compute_preference_aux_losses(
            decoder_output=decoder_output,
            inputs=inputs,
            neighbors_future=neighbors_future,
            neighbor_future_mask=mask,
            state_normalizer=args.state_normalizer,
            model_type=args.diffusion_model_type,
            dt=float(args.preference_loss_dt),
        )
        for key, value in pref_metrics.items():
            loss_dict[key] = value
        loss_dict["loss"] = diffusion_base_loss
        if float(args.preference_aux_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(args.preference_aux_loss_weight) * loss_dict["preference_proxy_loss"]
        if float(args.temporal_near_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(args.temporal_near_loss_weight) * loss_dict["temporal_near_condition_loss"]
        if float(args.temporal_gate_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(args.temporal_gate_loss_weight) * loss_dict["temporal_far_condition_loss"]
        if float(args.temporal_gate_target_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(args.temporal_gate_target_loss_weight) * (
                loss_dict["temporal_near_gate_target_loss"] + loss_dict["temporal_far_gate_target_loss"]
            )
        if float(args.temporal_gate_order_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(args.temporal_gate_order_loss_weight) * loss_dict["temporal_gate_order_loss"]
        loss_dict["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        log_item = {
            key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            for key, value in loss_dict.items()
        }
        log_item.update(prep_log)
        logs.append(log_item)
        pbar.set_postfix(loss=f"{log_item['loss']:.4f}", cond=f"{prep_log['style_condition_used_ratio']:.2f}")
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
    return _mean_logs(logs)


@torch.no_grad()
def _validate_epoch(
    data_loader: DataLoader,
    model: nn.Module,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    logs = []
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Val", dynamic_ncols=True), start=1):
        inputs, ego_future, neighbors_future, mask, prep_log = prepare_preference_conditioned_batch(
            batch,
            args,
            train=False,
            aug=None,
        )
        loss_dict: Dict[str, Any] = {}
        loss_dict, decoder_output = diffusion_loss_func(
            model,
            inputs,
            model.sde.marginal_prob,
            (ego_future, neighbors_future, mask),
            args.state_normalizer,
            loss_dict,
            args.diffusion_model_type,
        )
        diffusion_base_loss = (
            loss_dict["neighbor_prediction_loss"] + args.alpha_planning_loss * loss_dict["ego_planning_loss"]
        )
        loss_dict["diffusion_base_loss"] = diffusion_base_loss
        pref_metrics = compute_preference_aux_losses(
            decoder_output=decoder_output,
            inputs=inputs,
            neighbors_future=neighbors_future,
            neighbor_future_mask=mask,
            state_normalizer=args.state_normalizer,
            model_type=args.diffusion_model_type,
            dt=float(args.preference_loss_dt),
        )
        for key, value in pref_metrics.items():
            loss_dict[key] = value
        loss_dict["loss"] = diffusion_base_loss
        if float(args.preference_aux_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(args.preference_aux_loss_weight) * loss_dict["preference_proxy_loss"]
        if float(args.temporal_near_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(args.temporal_near_loss_weight) * loss_dict["temporal_near_condition_loss"]
        if float(args.temporal_gate_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(args.temporal_gate_loss_weight) * loss_dict["temporal_far_condition_loss"]
        if float(args.temporal_gate_target_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(args.temporal_gate_target_loss_weight) * (
                loss_dict["temporal_near_gate_target_loss"] + loss_dict["temporal_far_gate_target_loss"]
            )
        if float(args.temporal_gate_order_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(args.temporal_gate_order_loss_weight) * loss_dict["temporal_gate_order_loss"]
        log_item = {
            key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            for key, value in loss_dict.items()
        }
        log_item.update(prep_log)
        logs.append(log_item)
        if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
            break
    return _mean_logs(logs)


def _mean_logs(logs: list[Dict[str, float]]) -> Dict[str, float]:
    if not logs:
        return {}
    keys = logs[0].keys()
    return {
        key: float(sum(item.get(key, 0.0) for item in logs) / max(len(logs), 1))
        for key in keys
    }


def _append_epoch_log(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _log_runtime_imports() -> None:
    planner_path = inspect.getsourcefile(Diffusion_Planner) or "unknown"
    decoder_path = inspect.getsourcefile(Decoder) or "unknown"
    loss_path = inspect.getsourcefile(diffusion_loss_func) or "unknown"
    print(f"[PrefCondDiffusion] planner_module={planner_path}")
    print(f"[PrefCondDiffusion] decoder_module={decoder_path}")
    print(f"[PrefCondDiffusion] diffusion_loss_module={loss_path}")


def _assert_runtime_files_are_patched() -> None:
    decoder_path = inspect.getsourcefile(Decoder)
    loss_path = inspect.getsourcefile(diffusion_loss_func)
    if not decoder_path or not loss_path:
        raise RuntimeError("Failed to resolve runtime source files for decoder/loss.")

    with open(decoder_path, "r", encoding="utf-8") as file_obj:
        decoder_source = file_obj.read()
    with open(loss_path, "r", encoding="utf-8") as file_obj:
        loss_source = file_obj.read()

    decoder_markers = [
        'is_diffusion_loss_pass = ("sampled_trajectories" in inputs) and ("diffusion_time" in inputs)',
        '"x_start": denoised',
    ]
    loss_markers = [
        "def _extract_diffusion_prediction(",
        'candidate_keys = ["x_start", "score"] if model_type == "x_start" else ["score"]',
    ]

    missing_decoder = [marker for marker in decoder_markers if marker not in decoder_source]
    missing_loss = [marker for marker in loss_markers if marker not in loss_source]
    if missing_decoder or missing_loss:
        raise RuntimeError(
            "Patched diffusion runtime files are not active. "
            f"decoder_missing={missing_decoder}, loss_missing={missing_loss}, "
            f"decoder_path={decoder_path}, loss_path={loss_path}"
        )


def main() -> None:
    args = _build_args()
    set_seed(args.seed)
    _log_runtime_imports()
    _assert_runtime_files_are_patched()

    train_loader, val_loader = _prepare_dataloaders(args)
    save_path = build_experiment_dir(args.save_dir, args.experiment_name)
    write_json(os.path.join(save_path, "args.json"), serializable_args_dict(args))

    model = Diffusion_Planner(args).to(args.device)
    ema = ModelEma(model, decay=0.999, device=args.device) if args.use_ema else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, args.train_epochs, args.warm_up_epoch)

    init_epoch = 0
    wandb_id = None
    if args.resume_model_path:
        model, optimizer, scheduler, init_epoch, wandb_id, ema = resume_model(
            args.resume_model_path,
            model,
            optimizer,
            scheduler,
            ema,
            args.device,
        )

    aug = None
    if args.use_data_augment and args.augment_prob > 0:
        from baseline.common.data_augmentation import StatePerturbation

        aug = StatePerturbation(augment_prob=args.augment_prob, device=args.device)
    online_logger = Logger(
        args.experiment_name,
        args.notes,
        args,
        wandb_resume_id=wandb_id,
        save_path=save_path,
        proj_name=args.online_project_name,
        rank=0,
    )
    wandb_id = online_logger.id or wandb_id

    best_val_loss = float("inf")
    history_path = os.path.join(save_path, "epoch_metrics.jsonl")

    try:
        for epoch in range(init_epoch, args.train_epochs):
            train_stats = _train_epoch(train_loader, model, optimizer, args, ema, aug)
            val_stats = _validate_epoch(val_loader, model, args)
            scheduler.step()

            current_val_loss = float(val_stats.get("loss", float("inf")))
            epoch_payload = {
                "epoch": epoch + 1,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train": train_stats,
                "val": val_stats,
            }
            _append_epoch_log(history_path, epoch_payload)
            online_logger.log_metrics(
                {f"train/{key}": value for key, value in train_stats.items()},
                step=epoch + 1,
            )
            online_logger.log_metrics(
                {f"val/{key}": value for key, value in val_stats.items()},
                step=epoch + 1,
            )
            online_logger.log_metrics(
                {"lr": float(optimizer.param_groups[0]["lr"])},
                step=epoch + 1,
            )

            save_model(
                model,
                optimizer,
                scheduler,
                save_path,
                epoch,
                float(train_stats.get("loss", current_val_loss)),
                wandb_id,
                ema,
            )
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                save_model(
                    model,
                    optimizer,
                    scheduler,
                    save_path,
                    epoch,
                    float(train_stats.get("loss", current_val_loss)),
                    wandb_id,
                    ema,
                    filename=f"best_pref_cond_epoch_{epoch + 1}_valloss_{current_val_loss:.4f}.pth",
                )
            if (epoch + 1) % args.save_every == 0:
                save_model(
                    model,
                    optimizer,
                    scheduler,
                    save_path,
                    epoch,
                    float(train_stats.get("loss", current_val_loss)),
                    wandb_id,
                    ema,
                    filename=f"checkpoint_epoch_{epoch + 1}.pth",
                )

            print(
                "[PrefCondDiffusion] "
                f"epoch={epoch + 1}/{args.train_epochs} "
                f"train_loss={train_stats.get('loss', 0.0):.4f} "
                f"val_loss={current_val_loss:.4f} "
                f"cond_used={train_stats.get('style_condition_used_ratio', 0.0):.3f}"
            )
    finally:
        online_logger.finish()

    write_json(
        os.path.join(save_path, "train_summary.json"),
        {
            "best_val_loss": best_val_loss,
            "save_path": save_path,
            "train_split_root": args.train_split_root,
            "val_split_root": args.val_split_root,
            "condition_field": args.condition_field,
            "online_logger": args.online_logger,
            "use_online_logger": bool(args.use_online_logger),
        },
    )
    print(f"[PrefCondDiffusion] training finished. save_path={save_path}")


if __name__ == "__main__":
    main()

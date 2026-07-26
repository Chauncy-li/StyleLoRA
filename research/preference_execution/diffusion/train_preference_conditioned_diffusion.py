"""Standalone training entrypoint for preference-conditioned diffusion planning."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import torch
from timm.utils import ModelEma
from torch import nn, optim
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
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
from baseline.model.style_planner.guidance.preference_energy import (
    ConditionalPreferenceEnergy,
)
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
from research.preference_execution.diffusion.v6_losses import (
    compute_exogenous_neighbor_invariance_loss,
    compute_normal_anchor_consistency_loss,
    compute_normal_reference_prediction,
    compute_normal_relative_axis_loss,
    compute_signed_pair_monotonic_loss,
    compute_signed_raw_axis_loss,
    decoder_reported_v6_losses,
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
EXPERIMENT_PRESET_V6_SIGNED_ROUTER_STAGE_A = "v6_signed_router_stage_a"
EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A = (
    "v6_signed_router_ncqt_stage_a"
)
EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A2 = (
    "v6_signed_router_ncqt_stage_a2"
)
EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_KINEMATIC = (
    "v6_signed_router_ncqt_stage_a3_kinematic"
)
EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_2_TERMINAL = (
    "v6_signed_router_ncqt_stage_a3_2_terminal"
)
EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_3_ROLLOUT = (
    "v6_signed_router_ncqt_stage_a3_3_rollout"
)
EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_4_GLOBAL_PAIR = (
    "v6_signed_router_ncqt_stage_a3_4_global_pair"
)
EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_5_NORMAL_OPPORTUNITY = (
    "v6_signed_router_ncqt_stage_a3_5_normal_opportunity"
)
EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_6_WORST_AXIS = (
    "v6_signed_router_ncqt_stage_a3_6_worst_axis"
)
EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_7_AXIS_TEMPORAL = (
    "v6_signed_router_ncqt_stage_a3_7_axis_temporal"
)
EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_8_TERMINAL_EXECUTOR = (
    "v6_signed_router_ncqt_stage_a3_8_terminal_executor"
)

EXPERIMENT_PRESET_CHOICES = (
    EXPERIMENT_PRESET_MANUAL,
    EXPERIMENT_PRESET_BASELINE_9D,
    EXPERIMENT_PRESET_EXEC_V2_CONDITION_ONLY,
    EXPERIMENT_PRESET_PHASEWISE_EXEC_V1_CONDITION_ONLY,
    EXPERIMENT_PRESET_TWO_STAGE_EXEC_V1_PREF_LOSS,
    EXPERIMENT_PRESET_SAMPLER_ONLY_MILD,
    EXPERIMENT_PRESET_EXEC_V2_MILD_SAMPLER,
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_STAGE_A,
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A,
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A2,
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_KINEMATIC,
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_2_TERMINAL,
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_3_ROLLOUT,
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_4_GLOBAL_PAIR,
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_5_NORMAL_OPPORTUNITY,
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_6_WORST_AXIS,
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_7_AXIS_TEMPORAL,
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_8_TERMINAL_EXECUTOR,
)

PRESET_DEFAULT_EXPERIMENT_NAMES = {
    EXPERIMENT_PRESET_BASELINE_9D: "effective_preference_global_vec_baseline_9d",
    EXPERIMENT_PRESET_EXEC_V2_CONDITION_ONLY: "effective_preference_global_vec_exec_v2_condition_only",
    EXPERIMENT_PRESET_PHASEWISE_EXEC_V1_CONDITION_ONLY: "effective_preference_global_vec_phasewise_exec_v1_condition_only",
    EXPERIMENT_PRESET_TWO_STAGE_EXEC_V1_PREF_LOSS: "effective_preference_global_vec_two_stage_exec_v1_pref_loss",
    EXPERIMENT_PRESET_SAMPLER_ONLY_MILD: "effective_preference_global_vec_sampler_only_mild",
    EXPERIMENT_PRESET_EXEC_V2_MILD_SAMPLER: "effective_preference_global_vec_exec_v2_mild_sampler",
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_STAGE_A: "continuous_style_v6_signed_router_stage_a",
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A: "continuous_style_v6_signed_router_ncqt_stage_a",
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A2: "continuous_style_v6_signed_router_ncqt_stage_a2",
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_KINEMATIC: "continuous_style_v6_signed_router_ncqt_stage_a3_kinematic",
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_2_TERMINAL: "continuous_style_v6_signed_router_ncqt_stage_a3_2_terminal",
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_3_ROLLOUT: "continuous_style_v6_signed_router_ncqt_stage_a3_3_rollout",
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_4_GLOBAL_PAIR: "continuous_style_v6_signed_router_ncqt_stage_a3_4_global_pair",
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_5_NORMAL_OPPORTUNITY: "continuous_style_v6_signed_router_ncqt_stage_a3_5_normal_opportunity",
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_6_WORST_AXIS: "continuous_style_v6_signed_router_ncqt_stage_a3_6_worst_axis",
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_7_AXIS_TEMPORAL: "continuous_style_v6_signed_router_ncqt_stage_a3_7_axis_temporal",
    EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_8_TERMINAL_EXECUTOR: "continuous_style_v6_signed_router_ncqt_stage_a3_8_terminal_executor",
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
            "style_value_condition",
        ),
    )
    parser.add_argument(
        "--base_style_condition_dim",
        type=int,
        default=9,
        help="Dimension of the raw condition field before feature-set expansion; V6 direct axes use 12.",
    )
    parser.add_argument(
        "--train_conditioning_index_override",
        default="",
        help="Optional JSONL condition sidecar. Use V6 v6_direct_axis_conditions.jsonl for direct-axis training.",
    )
    parser.add_argument(
        "--val_conditioning_index_override",
        default="",
        help="Optional validation JSONL condition sidecar matching --train_conditioning_index_override schema.",
    )
    parser.add_argument(
        "--style_condition_feature_set",
        default="global_only",
        choices=STYLE_CONDITION_FEATURE_SET_CHOICES,
        help="Feature layout passed into the diffusion decoder for preference control.",
    )
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.15)
    parser.add_argument("--cfg_guidance_scale", type=float, default=1.5)
    parser.add_argument(
        "--style_condition_encoder",
        default="mlp",
        choices=("mlp", "axis_router_v1", "axis_router_v2_signed"),
    )
    parser.add_argument("--axis_router_token_dim", type=int, default=64)
    parser.add_argument(
        "--signed_router_injection_mode",
        default="global_adaln",
        choices=(
            "global_adaln",
            "ego_output_residual",
            "ego_kinematic_residual",
            "ego_axis_temporal_residual",
        ),
        help=(
            "Keep legacy global adaLN injection, use the reversible Stage-A2 "
            "ego-only output adapter, the Stage-A3.1 smooth kinematic adapter, "
            "or the A3.7 free-drive axis-temporal kinematic adapter."
        ),
    )
    parser.add_argument(
        "--signed_router_diffusion_gate_mode",
        default="all_steps",
        choices=("all_steps", "free_drive_terminal_only"),
        help=(
            "A3.8 keeps the legacy all-step path by default; "
            "free_drive_terminal_only applies the free-drive axis-temporal "
            "residual only at the final DPM denoise-to-zero call."
        ),
    )
    parser.add_argument(
        "--signed_router_terminal_t_max",
        type=float,
        default=0.0011,
        help="Inclusive diffusion-time threshold for the A3.8 terminal executor.",
    )
    parser.add_argument("--kinematic_ego_basis_count", type=int, default=6)
    parser.add_argument(
        "--preference_axis_reference_mode",
        default="self_generated",
        choices=("self_generated", "normal_neighbor"),
        help="Choose whether car-follow axes use styled neighbors or the detached rho=0 neighbor forecast.",
    )
    parser.add_argument(
        "--free_drive_accel_support_mode",
        default="self_generated",
        choices=("self_generated", "normal_anchor"),
        help=(
            "Measure acceleration willingness on each generated trajectory's "
            "own opportunity support, or on a detached rho=0 opportunity support."
        ),
    )
    parser.add_argument("--axis_router_activity_loss_weight", type=float, default=0.0)
    parser.add_argument("--axis_router_min_active_gate_fraction", type=float, default=0.10)
    parser.add_argument("--normal_anchor_cfg_enabled", action="store_true", default=False)
    parser.add_argument("--normal_anchor_loss_weight", type=float, default=0.0)
    parser.add_argument("--signed_raw_axis_loss_weight", type=float, default=0.0)
    parser.add_argument("--signed_raw_axis_loss_beta", type=float, default=0.08)
    parser.add_argument("--normal_relative_axis_loss_weight", type=float, default=0.0)
    parser.add_argument("--exogenous_neighbor_loss_weight", type=float, default=0.0)
    parser.add_argument("--signed_monotonic_loss_weight", type=float, default=0.0)
    parser.add_argument("--signed_symmetry_loss_weight", type=float, default=0.0)
    parser.add_argument("--signed_monotonic_delta", type=float, default=0.20)
    parser.add_argument("--signed_monotonic_margin", type=float, default=0.05)
    parser.add_argument("--signed_monotonic_batch_fraction", type=float, default=0.25)
    parser.add_argument("--signed_monotonic_max_pairs", type=int, default=32)
    parser.add_argument("--signed_monotonic_terminal_t_min", type=float, default=0.0)
    parser.add_argument("--signed_monotonic_terminal_t_max", type=float, default=0.0)
    parser.add_argument(
        "--signed_monotonic_aggregation_mode",
        default="axis_mean",
        choices=("axis_mean", "soft_worst_per_sample"),
        help=(
            "Preserve the historical mean over sampled (sample, axis) pairs, "
            "or select samples and apply a smooth worst-axis constraint within "
            "each sample."
        ),
    )
    parser.add_argument(
        "--signed_monotonic_worst_axis_temperature",
        type=float,
        default=0.05,
        help="LogSumExp temperature for per-sample soft worst-axis aggregation.",
    )
    parser.add_argument(
        "--signed_monotonic_command_mode",
        default="axis_isolated",
        choices=("axis_isolated", "global_rho"),
        help=(
            "Change one selected axis per pair, or move every causally active "
            "axis together along the inference-time global-rho command manifold."
        ),
    )
    parser.add_argument(
        "--signed_monotonic_short_rollout_steps",
        type=int,
        default=0,
        help=(
            "Number of stop-gradient first-order DPM-Solver++ state advances "
            "before the differentiable signed-pair terminal response. Zero "
            "preserves the Stage-A3.2 one-pass objective."
        ),
    )
    parser.add_argument("--preference_energy_normalization_path", default="")
    parser.add_argument("--preference_energy_rank_model_path", default="")
    parser.add_argument("--preference_energy_neighbours", type=int, default=64)
    parser.add_argument("--preference_energy_min_shared_features", type=int, default=3)
    parser.add_argument("--preference_energy_cdf_temperature", type=float, default=0.04)
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
    parser.add_argument("--backbone_lr_scale", type=float, default=1.0)
    parser.add_argument("--style_adapter_lr_scale", type=float, default=1.0)
    parser.add_argument("--freeze_planner_backbone", action="store_true", default=False)
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
    parser.add_argument(
        "--pretrained_model_path",
        default="",
        help="Optional checkpoint file used only for flexible weight initialization.",
    )
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
    parser.add_argument(
        "--use_controlled_preservation_sampler",
        action="store_true",
        default=False,
        help="Build each StylePlanner batch from explicit controlled and preservation pools.",
    )
    parser.add_argument("--controlled_batch_fraction", type=float, default=0.70)
    parser.add_argument(
        "--checkpoint_selection_metric",
        default="total_loss",
        choices=("total_loss", "signed_control", "signed_control_worst_scene"),
    )

    parser.add_argument("--time_len", type=int, default=21)
    parser.add_argument("--future_len", type=int, default=80)
    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--predicted_neighbor_num", type=int, default=10)
    parser.add_argument("--static_objects_num", type=int, default=5)
    parser.add_argument("--static_objects_state_dim", type=int, default=10)
    parser.add_argument("--lane_len", type=int, default=20)
    parser.add_argument("--lane_num", type=int, default=70)
    parser.add_argument("--route_len", type=int, default=20)
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
    elif preset == EXPERIMENT_PRESET_V6_SIGNED_ROUTER_STAGE_A:
        # Retained verbatim as the absolute-target Stage-A ablation used by the
        # first signed-router smoke experiment.
        preset_overrides = {
            "condition_field": "style_value_condition",
            "base_style_condition_dim": 12,
            "style_condition_feature_set": "global_only",
            "style_condition_encoder": "axis_router_v2_signed",
            "signed_router_injection_mode": "global_adaln",
            "preference_axis_reference_mode": "self_generated",
            "axis_router_token_dim": 64,
            "axis_router_activity_loss_weight": 0.0,
            "normal_anchor_cfg_enabled": False,
            "normal_anchor_loss_weight": 0.0,
            "cfg_dropout_prob": 0.0,
            "cfg_guidance_scale": 1.0,
            "signed_raw_axis_loss_weight": 0.50,
            "normal_relative_axis_loss_weight": 0.0,
            "exogenous_neighbor_loss_weight": 0.0,
            "signed_raw_axis_loss_beta": 0.08,
            "signed_monotonic_loss_weight": 0.10,
            "signed_symmetry_loss_weight": 0.0,
            "signed_monotonic_delta": 0.20,
            "signed_monotonic_margin": 0.05,
            "signed_monotonic_batch_fraction": 0.25,
            "signed_monotonic_max_pairs": 32,
            "freeze_planner_backbone": True,
            "use_ema": False,
            "use_controlled_preservation_sampler": True,
            "controlled_batch_fraction": 0.70,
            "checkpoint_selection_metric": "signed_control",
            "balance_scene_buckets": False,
            "preference_aux_loss_weight": 0.0,
            "temporal_near_loss_weight": 0.0,
            "temporal_gate_loss_weight": 0.0,
            "temporal_gate_target_loss_weight": 0.0,
            "temporal_gate_order_loss_weight": 0.0,
        }
    elif preset == EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A:
        preset_overrides = {
            "condition_field": "style_value_condition",
            "base_style_condition_dim": 12,
            "style_condition_feature_set": "global_only",
            "style_condition_encoder": "axis_router_v2_signed",
            "signed_router_injection_mode": "global_adaln",
            "preference_axis_reference_mode": "self_generated",
            "axis_router_token_dim": 64,
            # Signed controllability replaces the old unsupervised activity
            # floor as the primary router supervision.
            "axis_router_activity_loss_weight": 0.0,
            "normal_anchor_cfg_enabled": False,
            "normal_anchor_loss_weight": 0.0,
            "cfg_dropout_prob": 0.0,
            "cfg_guidance_scale": 1.0,
            # NCQT learns a conditional quantile displacement around the exact
            # pretrained normal trajectory instead of an incompatible absolute
            # percentile target.
            "signed_raw_axis_loss_weight": 0.0,
            "normal_relative_axis_loss_weight": 0.50,
            "exogenous_neighbor_loss_weight": 0.0,
            "signed_raw_axis_loss_beta": 0.08,
            "signed_monotonic_loss_weight": 0.50,
            "signed_symmetry_loss_weight": 0.20,
            "signed_monotonic_delta": 0.20,
            "signed_monotonic_margin": 0.05,
            "signed_monotonic_batch_fraction": 1.0,
            "signed_monotonic_max_pairs": 32,
            "freeze_planner_backbone": True,
            # Validate and checkpoint-select the actual signed adapter. A
            # 0.999 EMA lags badly during short router-only smoke runs.
            "use_ema": False,
            "use_controlled_preservation_sampler": True,
            "controlled_batch_fraction": 0.70,
            "checkpoint_selection_metric": "signed_control",
            "balance_scene_buckets": False,
            "preference_aux_loss_weight": 0.0,
            "temporal_near_loss_weight": 0.0,
            "temporal_gate_loss_weight": 0.0,
            "temporal_gate_target_loss_weight": 0.0,
            "temporal_gate_order_loss_weight": 0.0,
        }
    elif preset == EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A2:
        preset_overrides = {
            "condition_field": "style_value_condition",
            "base_style_condition_dim": 12,
            "style_condition_feature_set": "global_only",
            "style_condition_encoder": "axis_router_v2_signed",
            "axis_router_token_dim": 64,
            # Stage A2 is isolated behind two explicit switches. Returning to
            # the Stage-A preset reconstructs the previous architecture and
            # objective without reverting source files.
            "signed_router_injection_mode": "ego_output_residual",
            "preference_axis_reference_mode": "normal_neighbor",
            "axis_router_activity_loss_weight": 0.0,
            "normal_anchor_cfg_enabled": False,
            "normal_anchor_loss_weight": 0.0,
            "cfg_dropout_prob": 0.0,
            "cfg_guidance_scale": 1.0,
            "signed_raw_axis_loss_weight": 0.0,
            "normal_relative_axis_loss_weight": 0.50,
            "exogenous_neighbor_loss_weight": 0.05,
            "signed_raw_axis_loss_beta": 0.08,
            "signed_monotonic_loss_weight": 0.50,
            "signed_symmetry_loss_weight": 0.20,
            "signed_monotonic_delta": 0.20,
            "signed_monotonic_margin": 0.05,
            "signed_monotonic_batch_fraction": 1.0,
            "signed_monotonic_max_pairs": 32,
            "freeze_planner_backbone": True,
            "use_ema": False,
            "use_controlled_preservation_sampler": True,
            "controlled_batch_fraction": 0.70,
            "checkpoint_selection_metric": "signed_control_worst_scene",
            "balance_scene_buckets": False,
            "preference_aux_loss_weight": 0.0,
            "temporal_near_loss_weight": 0.0,
            "temporal_gate_loss_weight": 0.0,
            "temporal_gate_target_loss_weight": 0.0,
            "temporal_gate_order_loss_weight": 0.0,
        }
    elif preset == EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_KINEMATIC:
        # Stage A3.1 changes only the ego residual parameterization. Router,
        # NCQT supervision, sampling, and all frozen-planner settings remain
        # identical to Stage A2 so the physical effect is isolated cleanly.
        preset_overrides = {
            "condition_field": "style_value_condition",
            "base_style_condition_dim": 12,
            "style_condition_feature_set": "global_only",
            "style_condition_encoder": "axis_router_v2_signed",
            "axis_router_token_dim": 64,
            "signed_router_injection_mode": "ego_kinematic_residual",
            "kinematic_ego_basis_count": 6,
            "preference_axis_reference_mode": "normal_neighbor",
            "axis_router_activity_loss_weight": 0.0,
            "normal_anchor_cfg_enabled": False,
            "normal_anchor_loss_weight": 0.0,
            "cfg_dropout_prob": 0.0,
            "cfg_guidance_scale": 1.0,
            "signed_raw_axis_loss_weight": 0.0,
            "normal_relative_axis_loss_weight": 0.50,
            "exogenous_neighbor_loss_weight": 0.05,
            "signed_raw_axis_loss_beta": 0.08,
            "signed_monotonic_loss_weight": 0.50,
            "signed_symmetry_loss_weight": 0.20,
            "signed_monotonic_delta": 0.20,
            "signed_monotonic_margin": 0.05,
            "signed_monotonic_batch_fraction": 1.0,
            "signed_monotonic_max_pairs": 32,
            "freeze_planner_backbone": True,
            "use_ema": False,
            "use_controlled_preservation_sampler": True,
            "controlled_batch_fraction": 0.70,
            "checkpoint_selection_metric": "signed_control_worst_scene",
            "balance_scene_buckets": False,
            "preference_aux_loss_weight": 0.0,
            "temporal_near_loss_weight": 0.0,
            "temporal_gate_loss_weight": 0.0,
            "temporal_gate_target_loss_weight": 0.0,
            "temporal_gate_order_loss_weight": 0.0,
        }
    elif preset == EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_2_TERMINAL:
        # Stage A3.2 keeps the complete A3.1 model and objective weights. Only
        # the existing paired monotonic/symmetry branch is re-noised at a
        # low-noise terminal diffusion time; the base and NCQT passes retain
        # their original full-range random timestep.
        preset_overrides = {
            "condition_field": "style_value_condition",
            "base_style_condition_dim": 12,
            "style_condition_feature_set": "global_only",
            "style_condition_encoder": "axis_router_v2_signed",
            "axis_router_token_dim": 64,
            "signed_router_injection_mode": "ego_kinematic_residual",
            "kinematic_ego_basis_count": 6,
            "preference_axis_reference_mode": "normal_neighbor",
            "axis_router_activity_loss_weight": 0.0,
            "normal_anchor_cfg_enabled": False,
            "normal_anchor_loss_weight": 0.0,
            "cfg_dropout_prob": 0.0,
            "cfg_guidance_scale": 1.0,
            "signed_raw_axis_loss_weight": 0.0,
            "normal_relative_axis_loss_weight": 0.50,
            "exogenous_neighbor_loss_weight": 0.05,
            "signed_raw_axis_loss_beta": 0.08,
            "signed_monotonic_loss_weight": 0.50,
            "signed_symmetry_loss_weight": 0.20,
            "signed_monotonic_delta": 0.20,
            "signed_monotonic_margin": 0.05,
            "signed_monotonic_batch_fraction": 1.0,
            "signed_monotonic_max_pairs": 32,
            "signed_monotonic_terminal_t_min": 0.01,
            "signed_monotonic_terminal_t_max": 0.10,
            "freeze_planner_backbone": True,
            "use_ema": False,
            "use_controlled_preservation_sampler": True,
            "controlled_batch_fraction": 0.70,
            "checkpoint_selection_metric": "signed_control_worst_scene",
            "balance_scene_buckets": False,
            "preference_aux_loss_weight": 0.0,
            "temporal_near_loss_weight": 0.0,
            "temporal_gate_loss_weight": 0.0,
            "temporal_gate_target_loss_weight": 0.0,
            "temporal_gate_order_loss_weight": 0.0,
        }
    elif preset == EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_3_ROLLOUT:
        # Stage A3.3 changes only the existing A3.2 paired training branch.
        # Plus/normal/minus share the recovered forward noise, then traverse a
        # short stop-gradient DPM-Solver++ chain. The final terminal response is
        # differentiable, while Router, ego adapter, base loss, NCQT, inference,
        # CFG settings remain exactly those of Stage A3.2.
        preset_overrides = {
            "condition_field": "style_value_condition",
            "base_style_condition_dim": 12,
            "style_condition_feature_set": "global_only",
            "style_condition_encoder": "axis_router_v2_signed",
            "axis_router_token_dim": 64,
            "signed_router_injection_mode": "ego_kinematic_residual",
            "kinematic_ego_basis_count": 6,
            "preference_axis_reference_mode": "normal_neighbor",
            "axis_router_activity_loss_weight": 0.0,
            "normal_anchor_cfg_enabled": False,
            "normal_anchor_loss_weight": 0.0,
            "cfg_dropout_prob": 0.0,
            "cfg_guidance_scale": 1.0,
            "signed_raw_axis_loss_weight": 0.0,
            "normal_relative_axis_loss_weight": 0.50,
            "exogenous_neighbor_loss_weight": 0.05,
            "signed_raw_axis_loss_beta": 0.08,
            "signed_monotonic_loss_weight": 0.50,
            "signed_symmetry_loss_weight": 0.20,
            "signed_monotonic_delta": 0.20,
            "signed_monotonic_margin": 0.05,
            "signed_monotonic_batch_fraction": 1.0,
            "signed_monotonic_max_pairs": 32,
            "signed_monotonic_terminal_t_min": 0.01,
            "signed_monotonic_terminal_t_max": 0.10,
            "signed_monotonic_short_rollout_steps": 2,
            "freeze_planner_backbone": True,
            "use_ema": False,
            "use_controlled_preservation_sampler": True,
            "controlled_batch_fraction": 0.70,
            "checkpoint_selection_metric": "signed_control_worst_scene",
            "balance_scene_buckets": False,
            "preference_aux_loss_weight": 0.0,
            "temporal_near_loss_weight": 0.0,
            "temporal_gate_loss_weight": 0.0,
            "temporal_gate_target_loss_weight": 0.0,
            "temporal_gate_order_loss_weight": 0.0,
        }
    elif preset == EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_4_GLOBAL_PAIR:
        # Stage A3.4 returns to the stronger A3.2 terminal objective and changes
        # only how its existing minus/normal/plus commands are constructed.
        # All causally active axes now move together exactly as they do under a
        # scalar rho sweep at inference. Per-axis response is still measured
        # separately, exposing and penalizing cross-axis gradient interference.
        preset_overrides = {
            "condition_field": "style_value_condition",
            "base_style_condition_dim": 12,
            "style_condition_feature_set": "global_only",
            "style_condition_encoder": "axis_router_v2_signed",
            "axis_router_token_dim": 64,
            "signed_router_injection_mode": "ego_kinematic_residual",
            "kinematic_ego_basis_count": 6,
            "preference_axis_reference_mode": "normal_neighbor",
            "axis_router_activity_loss_weight": 0.0,
            "normal_anchor_cfg_enabled": False,
            "normal_anchor_loss_weight": 0.0,
            "cfg_dropout_prob": 0.0,
            "cfg_guidance_scale": 1.0,
            "signed_raw_axis_loss_weight": 0.0,
            "normal_relative_axis_loss_weight": 0.50,
            "exogenous_neighbor_loss_weight": 0.05,
            "signed_raw_axis_loss_beta": 0.08,
            "signed_monotonic_loss_weight": 0.50,
            "signed_symmetry_loss_weight": 0.20,
            "signed_monotonic_delta": 0.20,
            "signed_monotonic_margin": 0.05,
            "signed_monotonic_batch_fraction": 1.0,
            "signed_monotonic_max_pairs": 32,
            "signed_monotonic_terminal_t_min": 0.01,
            "signed_monotonic_terminal_t_max": 0.10,
            "signed_monotonic_command_mode": "global_rho",
            "signed_monotonic_short_rollout_steps": 0,
            "freeze_planner_backbone": True,
            "use_ema": False,
            "use_controlled_preservation_sampler": True,
            "controlled_batch_fraction": 0.70,
            "checkpoint_selection_metric": "signed_control_worst_scene",
            "balance_scene_buckets": False,
            "preference_aux_loss_weight": 0.0,
            "temporal_near_loss_weight": 0.0,
            "temporal_gate_loss_weight": 0.0,
            "temporal_gate_target_loss_weight": 0.0,
            "temporal_gate_order_loss_weight": 0.0,
        }
    elif preset == EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_5_NORMAL_OPPORTUNITY:
        # Stage A3.5 keeps the complete A3.4 command manifold. Its sole
        # training change is to measure free-drive acceleration willingness on
        # the detached semantic-normal opportunity support shared by
        # commanded/plus/minus trajectories. This removes command-dependent
        # measurement support without changing the axis target or planner.
        preset_overrides = {
            "condition_field": "style_value_condition",
            "base_style_condition_dim": 12,
            "style_condition_feature_set": "global_only",
            "style_condition_encoder": "axis_router_v2_signed",
            "axis_router_token_dim": 64,
            "signed_router_injection_mode": "ego_kinematic_residual",
            "kinematic_ego_basis_count": 6,
            "preference_axis_reference_mode": "normal_neighbor",
            "free_drive_accel_support_mode": "normal_anchor",
            "axis_router_activity_loss_weight": 0.0,
            "normal_anchor_cfg_enabled": False,
            "normal_anchor_loss_weight": 0.0,
            "cfg_dropout_prob": 0.0,
            "cfg_guidance_scale": 1.0,
            "signed_raw_axis_loss_weight": 0.0,
            "normal_relative_axis_loss_weight": 0.50,
            "exogenous_neighbor_loss_weight": 0.05,
            "signed_raw_axis_loss_beta": 0.08,
            "signed_monotonic_loss_weight": 0.50,
            "signed_symmetry_loss_weight": 0.20,
            "signed_monotonic_delta": 0.20,
            "signed_monotonic_margin": 0.05,
            "signed_monotonic_batch_fraction": 1.0,
            "signed_monotonic_max_pairs": 32,
            "signed_monotonic_terminal_t_min": 0.01,
            "signed_monotonic_terminal_t_max": 0.10,
            "signed_monotonic_command_mode": "global_rho",
            "signed_monotonic_short_rollout_steps": 0,
            "freeze_planner_backbone": True,
            "use_ema": False,
            "use_controlled_preservation_sampler": True,
            "controlled_batch_fraction": 0.70,
            "checkpoint_selection_metric": "signed_control_worst_scene",
            "balance_scene_buckets": False,
            "preference_aux_loss_weight": 0.0,
            "temporal_near_loss_weight": 0.0,
            "temporal_gate_loss_weight": 0.0,
            "temporal_gate_target_loss_weight": 0.0,
            "temporal_gate_order_loss_weight": 0.0,
        }
    elif preset == EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_6_WORST_AXIS:
        # Stage A3.6 is a strict A3.4 branch. It keeps the global-rho command,
        # self-generated acceleration support, low-noise terminal pair, frozen
        # planner, NCQT/symmetry weights, Router, and kinematic adapter. Its only
        # objective change groups all valid axes by sample and smooth-maxes the
        # monotonic hinge so an already-correct easy axis cannot dilute the
        # acceleration-willingness violation.
        preset_overrides = {
            "condition_field": "style_value_condition",
            "base_style_condition_dim": 12,
            "style_condition_feature_set": "global_only",
            "style_condition_encoder": "axis_router_v2_signed",
            "axis_router_token_dim": 64,
            "signed_router_injection_mode": "ego_kinematic_residual",
            "kinematic_ego_basis_count": 6,
            "preference_axis_reference_mode": "normal_neighbor",
            "free_drive_accel_support_mode": "self_generated",
            "axis_router_activity_loss_weight": 0.0,
            "normal_anchor_cfg_enabled": False,
            "normal_anchor_loss_weight": 0.0,
            "cfg_dropout_prob": 0.0,
            "cfg_guidance_scale": 1.0,
            "signed_raw_axis_loss_weight": 0.0,
            "normal_relative_axis_loss_weight": 0.50,
            "exogenous_neighbor_loss_weight": 0.05,
            "signed_raw_axis_loss_beta": 0.08,
            "signed_monotonic_loss_weight": 0.50,
            "signed_symmetry_loss_weight": 0.20,
            "signed_monotonic_delta": 0.20,
            "signed_monotonic_margin": 0.05,
            "signed_monotonic_batch_fraction": 1.0,
            # In soft-worst mode this cap is the maximum eligible sample count;
            # axis_mean retains the historical maximum (sample, axis) row count.
            "signed_monotonic_max_pairs": 32,
            "signed_monotonic_terminal_t_min": 0.01,
            "signed_monotonic_terminal_t_max": 0.10,
            "signed_monotonic_command_mode": "global_rho",
            "signed_monotonic_aggregation_mode": "soft_worst_per_sample",
            "signed_monotonic_worst_axis_temperature": 0.05,
            "signed_monotonic_short_rollout_steps": 0,
            "freeze_planner_backbone": True,
            "use_ema": False,
            "use_controlled_preservation_sampler": True,
            "controlled_batch_fraction": 0.70,
            "checkpoint_selection_metric": "signed_control_worst_scene",
            "balance_scene_buckets": False,
            "preference_aux_loss_weight": 0.0,
            "temporal_near_loss_weight": 0.0,
            "temporal_gate_loss_weight": 0.0,
            "temporal_gate_target_loss_weight": 0.0,
            "temporal_gate_order_loss_weight": 0.0,
        }
    elif preset == EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_7_AXIS_TEMPORAL:
        # Stage A3.7 inherits the complete A3.6 objective and changes only the
        # ego residual parameterization. The signed Router still owns the same
        # global-rho semantics and gates; free-drive axis contributions remain
        # separate until three fixed-scale acceleration banks are summed in
        # trajectory space. Car-follow keeps the legacy kinematic path.
        preset_overrides = {
            "condition_field": "style_value_condition",
            "base_style_condition_dim": 12,
            "style_condition_feature_set": "global_only",
            "style_condition_encoder": "axis_router_v2_signed",
            "axis_router_token_dim": 64,
            "signed_router_injection_mode": "ego_axis_temporal_residual",
            "signed_router_diffusion_gate_mode": "all_steps",
            "signed_router_terminal_t_max": 0.0011,
            "kinematic_ego_basis_count": 6,
            "preference_axis_reference_mode": "normal_neighbor",
            "free_drive_accel_support_mode": "self_generated",
            "axis_router_activity_loss_weight": 0.0,
            "normal_anchor_cfg_enabled": False,
            "normal_anchor_loss_weight": 0.0,
            "cfg_dropout_prob": 0.0,
            "cfg_guidance_scale": 1.0,
            "signed_raw_axis_loss_weight": 0.0,
            "normal_relative_axis_loss_weight": 0.50,
            "exogenous_neighbor_loss_weight": 0.05,
            "signed_raw_axis_loss_beta": 0.08,
            "signed_monotonic_loss_weight": 0.50,
            "signed_symmetry_loss_weight": 0.20,
            "signed_monotonic_delta": 0.20,
            "signed_monotonic_margin": 0.05,
            "signed_monotonic_batch_fraction": 1.0,
            "signed_monotonic_max_pairs": 32,
            "signed_monotonic_terminal_t_min": 0.01,
            "signed_monotonic_terminal_t_max": 0.10,
            "signed_monotonic_command_mode": "global_rho",
            "signed_monotonic_aggregation_mode": "soft_worst_per_sample",
            "signed_monotonic_worst_axis_temperature": 0.05,
            "signed_monotonic_short_rollout_steps": 0,
            "freeze_planner_backbone": True,
            "use_ema": False,
            "use_controlled_preservation_sampler": True,
            "controlled_batch_fraction": 0.70,
            "checkpoint_selection_metric": "signed_control_worst_scene",
            "balance_scene_buckets": False,
            "preference_aux_loss_weight": 0.0,
            "temporal_near_loss_weight": 0.0,
            "temporal_gate_loss_weight": 0.0,
            "temporal_gate_target_loss_weight": 0.0,
            "temporal_gate_order_loss_weight": 0.0,
        }
    elif preset == EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_8_TERMINAL_EXECUTOR:
        # Stage A3.8 changes exactly one sampling mechanism relative to A3.7:
        # all three free-drive axis-temporal residual branches execute only at
        # the final DPM denoise-to-zero call. The signed Router, one global rho,
        # six-axis/data contract, fixed trajectory-time banks, ego-only scope,
        # bias-free zero residual, and car-follow all-step path are unchanged.
        # The paired objective is evaluated in the same terminal-time window so
        # its local supervision is applied where the residual will be executed.
        preset_overrides = {
            "condition_field": "style_value_condition",
            "base_style_condition_dim": 12,
            "style_condition_feature_set": "global_only",
            "style_condition_encoder": "axis_router_v2_signed",
            "axis_router_token_dim": 64,
            "signed_router_injection_mode": "ego_axis_temporal_residual",
            "signed_router_diffusion_gate_mode": "free_drive_terminal_only",
            "signed_router_terminal_t_max": 0.0011,
            "kinematic_ego_basis_count": 6,
            "preference_axis_reference_mode": "normal_neighbor",
            "free_drive_accel_support_mode": "self_generated",
            "axis_router_activity_loss_weight": 0.0,
            "normal_anchor_cfg_enabled": False,
            "normal_anchor_loss_weight": 0.0,
            "cfg_dropout_prob": 0.0,
            "cfg_guidance_scale": 1.0,
            "signed_raw_axis_loss_weight": 0.0,
            "normal_relative_axis_loss_weight": 0.50,
            "exogenous_neighbor_loss_weight": 0.05,
            "signed_raw_axis_loss_beta": 0.08,
            "signed_monotonic_loss_weight": 0.50,
            "signed_symmetry_loss_weight": 0.20,
            "signed_monotonic_delta": 0.20,
            "signed_monotonic_margin": 0.05,
            "signed_monotonic_batch_fraction": 1.0,
            "signed_monotonic_max_pairs": 32,
            "signed_monotonic_terminal_t_min": 0.001,
            "signed_monotonic_terminal_t_max": 0.0011,
            "signed_monotonic_command_mode": "global_rho",
            "signed_monotonic_aggregation_mode": "soft_worst_per_sample",
            "signed_monotonic_worst_axis_temperature": 0.05,
            "signed_monotonic_short_rollout_steps": 0,
            "freeze_planner_backbone": True,
            "use_ema": False,
            "use_controlled_preservation_sampler": True,
            "controlled_batch_fraction": 0.70,
            "checkpoint_selection_metric": "signed_control_worst_scene",
            "balance_scene_buckets": False,
            "preference_aux_loss_weight": 0.0,
            "temporal_near_loss_weight": 0.0,
            "temporal_gate_loss_weight": 0.0,
            "temporal_gate_target_loss_weight": 0.0,
            "temporal_gate_order_loss_weight": 0.0,
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
    if int(args.route_len) <= 0:
        raise ValueError("--route_len must be positive")
    if int(args.base_style_condition_dim) <= 0:
        raise ValueError("--base_style_condition_dim must be positive")
    if float(args.backbone_lr_scale) <= 0.0:
        raise ValueError("--backbone_lr_scale must be positive")
    if float(args.style_adapter_lr_scale) <= 0.0:
        raise ValueError("--style_adapter_lr_scale must be positive")
    args.global_style_condition_dim = global_style_condition_dim(
        args.style_condition_feature_set,
        base_global_dim=int(args.base_style_condition_dim),
    )
    args.phase_style_condition_dim = phase_style_condition_dim(args.style_condition_feature_set)
    args.phase_style_num_phases = phase_style_num_phases(args.style_condition_feature_set)
    args.phase_style_flat_dim = phase_style_flat_dim(args.style_condition_feature_set)
    args.style_value_dim = style_condition_dim(
        args.style_condition_feature_set,
        base_global_dim=int(args.base_style_condition_dim),
    )
    args.use_style_condition = True
    args.use_phase_style_condition = bool(args.phase_style_flat_dim > 0)
    args.use_temporal_style_gate = use_temporal_style_gate(args.style_condition_feature_set)
    if args.style_condition_encoder in {
        "axis_router_v1",
        "axis_router_v2_signed",
    }:
        if args.condition_field != "style_value_condition":
            raise ValueError(
                f"{args.style_condition_encoder} requires "
                "--condition_field style_value_condition"
            )
        if int(args.global_style_condition_dim) != 12:
            raise ValueError(
                f"{args.style_condition_encoder} requires a 12-dimensional "
                "V6 global condition"
            )
    if (
        args.signed_router_injection_mode
        in {
            "ego_output_residual",
            "ego_kinematic_residual",
            "ego_axis_temporal_residual",
        }
        and args.style_condition_encoder != "axis_router_v2_signed"
    ):
        raise ValueError(
            "ego-only signed router injection requires "
            "--style_condition_encoder axis_router_v2_signed"
        )
    if int(args.kinematic_ego_basis_count) < 2:
        raise ValueError("--kinematic_ego_basis_count must be at least 2")
    if not 0.0 < float(args.signed_router_terminal_t_max) <= 1.0:
        raise ValueError("--signed_router_terminal_t_max must be in (0, 1]")
    if args.signed_router_diffusion_gate_mode == "free_drive_terminal_only":
        if args.signed_router_injection_mode != "ego_axis_temporal_residual":
            raise ValueError(
                "--signed_router_diffusion_gate_mode free_drive_terminal_only "
                "requires --signed_router_injection_mode "
                "ego_axis_temporal_residual"
            )
        if not 0.001 <= float(args.signed_router_terminal_t_max) <= 0.0011:
            raise ValueError(
                "A3.8 terminal-only execution requires "
                "--signed_router_terminal_t_max in [0.001, 0.0011]"
            )
    if (
        args.preference_axis_reference_mode == "normal_neighbor"
        and float(args.normal_relative_axis_loss_weight) <= 0.0
    ):
        raise ValueError(
            "--preference_axis_reference_mode normal_neighbor requires the "
            "normal-relative axis objective"
        )
    if args.normal_anchor_cfg_enabled and args.condition_field != "style_value_condition":
        raise ValueError(
            "Normal-Anchor CFG requires the V6 style_value_condition contract"
        )
    signed_objective_enabled = (
        float(args.signed_raw_axis_loss_weight) > 0.0
        or float(args.normal_relative_axis_loss_weight) > 0.0
        or float(args.signed_monotonic_loss_weight) > 0.0
        or float(args.signed_symmetry_loss_weight) > 0.0
    )
    if float(args.signed_raw_axis_loss_weight) < 0.0:
        raise ValueError("--signed_raw_axis_loss_weight must be non-negative")
    if float(args.signed_monotonic_loss_weight) < 0.0:
        raise ValueError("--signed_monotonic_loss_weight must be non-negative")
    if float(args.normal_relative_axis_loss_weight) < 0.0:
        raise ValueError("--normal_relative_axis_loss_weight must be non-negative")
    if float(args.signed_symmetry_loss_weight) < 0.0:
        raise ValueError("--signed_symmetry_loss_weight must be non-negative")
    if float(args.exogenous_neighbor_loss_weight) < 0.0:
        raise ValueError("--exogenous_neighbor_loss_weight must be non-negative")
    if signed_objective_enabled:
        if args.style_condition_encoder != "axis_router_v2_signed":
            raise ValueError(
                "Signed raw-axis/monotonic losses require "
                "--style_condition_encoder axis_router_v2_signed"
            )
        if args.diffusion_model_type != "x_start":
            raise ValueError("Signed V6 objectives require --diffusion_model_type x_start")
        if not args.preference_energy_normalization_path:
            raise ValueError(
                "--preference_energy_normalization_path is required as the "
                "train-only raw-axis normalization reference"
            )
        if not args.preference_energy_rank_model_path:
            raise ValueError(
                "--preference_energy_rank_model_path is required as the "
                "train-only conditional inverse-CDF reference"
            )
        if not 0.0 < float(args.signed_monotonic_batch_fraction) <= 1.0:
            raise ValueError("--signed_monotonic_batch_fraction must be in (0, 1]")
        if not 0.0 < float(args.signed_monotonic_delta) <= 0.5:
            raise ValueError("--signed_monotonic_delta must be in (0, 0.5]")
        if float(args.signed_raw_axis_loss_beta) <= 0.0:
            raise ValueError("--signed_raw_axis_loss_beta must be positive")
        if float(args.signed_monotonic_margin) <= 0.0:
            raise ValueError("--signed_monotonic_margin must be positive")
        if int(args.signed_monotonic_max_pairs) <= 0:
            raise ValueError("--signed_monotonic_max_pairs must be positive")
        if float(args.signed_monotonic_worst_axis_temperature) <= 0.0:
            raise ValueError(
                "--signed_monotonic_worst_axis_temperature must be positive"
            )
        if (
            args.signed_monotonic_aggregation_mode == "soft_worst_per_sample"
            and args.signed_monotonic_command_mode != "global_rho"
        ):
            raise ValueError(
                "--signed_monotonic_aggregation_mode soft_worst_per_sample "
                "requires --signed_monotonic_command_mode global_rho"
            )
        terminal_t_min = float(args.signed_monotonic_terminal_t_min)
        terminal_t_max = float(args.signed_monotonic_terminal_t_max)
        terminal_disabled = terminal_t_min == 0.0 and terminal_t_max == 0.0
        if not terminal_disabled and not 0.0 < terminal_t_min < terminal_t_max <= 1.0:
            raise ValueError(
                "signed monotonic terminal time must be disabled as 0/0 or "
                "satisfy 0 < t_min < t_max <= 1"
            )
        if (
            args.signed_router_diffusion_gate_mode
            == "free_drive_terminal_only"
            and terminal_t_max > float(args.signed_router_terminal_t_max) + 1e-12
        ):
            raise ValueError(
                "A3.8 paired supervision must remain inside the terminal "
                "residual execution window"
            )
        short_rollout_steps = int(args.signed_monotonic_short_rollout_steps)
        if not 0 <= short_rollout_steps <= 4:
            raise ValueError(
                "--signed_monotonic_short_rollout_steps must be in [0, 4]"
            )
        if short_rollout_steps > 0 and terminal_disabled:
            raise ValueError(
                "short signed-pair rollout requires a non-zero terminal time range"
            )
    if args.freeze_planner_backbone and args.style_condition_encoder != "axis_router_v2_signed":
        raise ValueError(
            "--freeze_planner_backbone is currently reserved for the signed "
            "StylePlanner router preset"
        )
    if args.use_controlled_preservation_sampler and not 0.0 < float(
        args.controlled_batch_fraction
    ) < 1.0:
        raise ValueError("--controlled_batch_fraction must be in (0, 1)")
    if args.resume_model_path and args.pretrained_model_path:
        raise ValueError(
            "Use either --resume_model_path or --pretrained_model_path, not both"
        )
    if (
        args.experiment_preset
        in {
            EXPERIMENT_PRESET_V6_SIGNED_ROUTER_STAGE_A,
            EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A,
            EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A2,
            EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_KINEMATIC,
            EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_2_TERMINAL,
            EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_3_ROLLOUT,
            EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_4_GLOBAL_PAIR,
            EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_5_NORMAL_OPPORTUNITY,
            EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_6_WORST_AXIS,
        }
        and not args.resume_model_path
        and not args.pretrained_model_path
    ):
        raise ValueError(
            "The V6 StylePlanner preset must initialize from a verified "
            "Diffusion/StylePlanner checkpoint via --pretrained_model_path, "
            "or resume an existing V6 run. This is required to preserve the "
            "rho=0 and lane-change base-planning anchor."
        )
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


class ControlledPreservationBatchSampler(Sampler[list[int]]):
    """Build fixed-ratio batches without changing the cached V6 dataset.

    Controlled rows are the causally executable car-follow/free-drive rows
    whose 12-D contract has at least one active axis.  Empty, lane-change, and
    all other rows form the preservation stream.  Sampling with replacement
    keeps the epoch length stable while ensuring that a zero-initialized router
    receives useful signed supervision in every batch.
    """

    def __init__(
        self,
        dataset: PreferenceConditionedPlannerData,
        *,
        batch_size: int,
        controlled_fraction: float,
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        self.controlled_fraction = float(controlled_fraction)
        if self.batch_size < 2:
            raise ValueError("controlled/preservation batches require batch_size >= 2")

        controlled_by_scene: Dict[str, list[int]] = {
            "straight_free_drive": [],
            "straight_car_follow": [],
        }
        preservation: list[int] = []
        for index, record in enumerate(dataset.records):
            condition = list(record.get("style_value_condition", []))
            scene = str(record.get("scene_bucket", ""))
            active_axis = (
                len(condition) >= 6
                and any(float(value) > 0.5 for value in condition[3:6])
            )
            if active_axis and scene in controlled_by_scene:
                controlled_by_scene[scene].append(index)
            else:
                preservation.append(index)

        self.controlled_pools = [
            pool for pool in controlled_by_scene.values() if pool
        ]
        self.preservation_pool = preservation
        if not self.controlled_pools:
            raise RuntimeError(
                "controlled/preservation sampler found no causally executable "
                "car-follow/free-drive rows"
            )
        if not self.preservation_pool:
            raise RuntimeError(
                "controlled/preservation sampler found no preservation rows"
            )

        self.controlled_count = min(
            max(int(round(self.batch_size * self.controlled_fraction)), 1),
            self.batch_size - 1,
        )
        self.preservation_count = self.batch_size - self.controlled_count
        self.num_batches = max(len(dataset) // self.batch_size, 1)
        controlled_total = sum(len(pool) for pool in self.controlled_pools)
        print(
            "[PrefCondDiffusion] controlled_preservation_sampler "
            f"controlled_rows={controlled_total} "
            f"preservation_rows={len(self.preservation_pool)} "
            f"per_batch={self.controlled_count}/{self.preservation_count}"
        )

    @staticmethod
    def _draw(pool: list[int], count: int) -> list[int]:
        if count <= 0:
            return []
        selected = torch.randint(len(pool), (count,)).tolist()
        return [pool[index] for index in selected]

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.num_batches):
            if len(self.controlled_pools) == 1:
                controlled = self._draw(
                    self.controlled_pools[0],
                    self.controlled_count,
                )
            else:
                first_count = self.controlled_count // 2
                second_count = self.controlled_count - first_count
                controlled = self._draw(self.controlled_pools[0], first_count)
                controlled.extend(
                    self._draw(self.controlled_pools[1], second_count)
                )
            batch = controlled + self._draw(
                self.preservation_pool,
                self.preservation_count,
            )
            order = torch.randperm(len(batch)).tolist()
            yield [batch[index] for index in order]

    def __len__(self) -> int:
        return self.num_batches


def _prepare_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset = PreferenceConditionedPlannerData(
        cache_dir=args.train_cache_dir,
        split_root=args.train_split_root,
        condition_field=args.condition_field,
        conditioning_index_override=args.train_conditioning_index_override or None,
        start_index=args.train_start_index,
        num_samples=args.train_num_samples,
    )
    val_dataset = PreferenceConditionedPlannerData(
        cache_dir=args.val_cache_dir,
        split_root=args.val_split_root,
        condition_field=args.condition_field,
        conditioning_index_override=args.val_conditioning_index_override or None,
        start_index=args.val_start_index,
        num_samples=args.val_num_samples,
    )
    if args.use_controlled_preservation_sampler:
        batch_sampler = ControlledPreservationBatchSampler(
            train_dataset,
            batch_size=int(args.batch_size),
            controlled_fraction=float(args.controlled_batch_fraction),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
    else:
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


def _build_signed_axis_objective(
    args: argparse.Namespace,
) -> ConditionalPreferenceEnergy | None:
    if (
        float(args.signed_raw_axis_loss_weight) <= 0.0
        and float(args.normal_relative_axis_loss_weight) <= 0.0
        and float(args.signed_monotonic_loss_weight) <= 0.0
        and float(args.signed_symmetry_loss_weight) <= 0.0
    ):
        return None
    # The V5 artifacts are reused only as frozen train-split references for
    # inverse-CDF raw-axis targets.
    return ConditionalPreferenceEnergy(
        normalization_path=str(args.preference_energy_normalization_path),
        conditional_rank_model_path=str(args.preference_energy_rank_model_path),
        state_normalizer=args.state_normalizer,
        neighbours=int(args.preference_energy_neighbours),
        min_shared_condition_features=int(
            args.preference_energy_min_shared_features
        ),
        cdf_temperature=float(args.preference_energy_cdf_temperature),
        free_drive_accel_support_mode=str(
            args.free_drive_accel_support_mode
        ),
        dt=float(args.preference_loss_dt),
    )


def _compute_v6_losses(
    *,
    model: nn.Module,
    decoder_output: Dict[str, torch.Tensor],
    inputs: Dict[str, Any],
    args: argparse.Namespace,
    axis_objective: ConditionalPreferenceEnergy | None,
    compute_monotonic: bool,
    monotonic_random_subset: bool,
) -> Dict[str, torch.Tensor]:
    losses = decoder_reported_v6_losses(
        decoder_output=decoder_output,
        inputs=inputs,
        min_active_gate_fraction=float(args.axis_router_min_active_gate_fraction),
    )
    if (
        bool(args.normal_anchor_cfg_enabled)
        and float(args.normal_anchor_loss_weight) > 0.0
    ):
        losses.update(
            compute_normal_anchor_consistency_loss(
                model=model,
                decoder_output=decoder_output,
                inputs=inputs,
                model_type=args.diffusion_model_type,
            )
        )
    else:
        zero = inputs["style_value_condition"].new_zeros(())
        losses.update(
            {
                "normal_anchor_consistency_loss": zero,
                "normal_anchor_active_ratio": zero,
                "normal_anchor_prediction_l1": zero,
            }
        )
    normal_prediction = None
    if (
        float(args.normal_relative_axis_loss_weight) > 0.0
        or float(args.exogenous_neighbor_loss_weight) > 0.0
        or compute_monotonic
    ):
        normal_prediction = compute_normal_reference_prediction(
            model=model,
            decoder_output=decoder_output,
            inputs=inputs,
            model_type=args.diffusion_model_type,
        )
    losses.update(
        compute_signed_raw_axis_loss(
            axis_objective=(
                axis_objective
                if float(args.signed_raw_axis_loss_weight) > 0.0
                else None
            ),
            decoder_output=decoder_output,
            inputs=inputs,
            beta=float(args.signed_raw_axis_loss_beta),
        )
    )
    if float(args.exogenous_neighbor_loss_weight) > 0.0:
        losses.update(
            compute_exogenous_neighbor_invariance_loss(
                decoder_output=decoder_output,
                normal_prediction=normal_prediction,
                inputs=inputs,
            )
        )
    else:
        zero = inputs["style_value_condition"].new_zeros(())
        losses.update(
            {
                "exogenous_neighbor_invariance_loss": zero,
                "exogenous_neighbor_invariance_l1": zero,
                "exogenous_neighbor_active_ratio": zero,
                "exogenous_neighbor_active_count": zero,
            }
        )
    losses.update(
        compute_normal_relative_axis_loss(
            axis_objective=(
                axis_objective
                if float(args.normal_relative_axis_loss_weight) > 0.0
                else None
            ),
            decoder_output=decoder_output,
            normal_prediction=normal_prediction,
            inputs=inputs,
            beta=float(args.signed_raw_axis_loss_beta),
            fixed_normal_neighbors=(
                args.preference_axis_reference_mode == "normal_neighbor"
            ),
        )
    )
    losses.update(
        compute_signed_pair_monotonic_loss(
            model=model,
            axis_objective=(
                axis_objective
                if compute_monotonic
                and (
                    float(args.signed_monotonic_loss_weight) > 0.0
                    or float(args.signed_symmetry_loss_weight) > 0.0
                )
                else None
            ),
            decoder_output=decoder_output,
            normal_prediction=normal_prediction,
            inputs=inputs,
            model_type=args.diffusion_model_type,
            delta=float(args.signed_monotonic_delta),
            margin=float(args.signed_monotonic_margin),
            max_pairs=int(args.signed_monotonic_max_pairs),
            random_subset=monotonic_random_subset,
            fixed_normal_neighbors=(
                args.preference_axis_reference_mode == "normal_neighbor"
            ),
            terminal_t_min=float(args.signed_monotonic_terminal_t_min),
            terminal_t_max=float(args.signed_monotonic_terminal_t_max),
            short_rollout_steps=int(args.signed_monotonic_short_rollout_steps),
            command_mode=str(args.signed_monotonic_command_mode),
            aggregation_mode=str(args.signed_monotonic_aggregation_mode),
            worst_axis_temperature=float(
                args.signed_monotonic_worst_axis_temperature
            ),
        )
    )
    return losses


def _load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str,
    *,
    minimum_coverage: float = 0.80,
) -> None:
    """Initialize matching backbone weights without requiring optimizer compatibility."""

    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "latest.pth"
    if not path.is_file():
        raise FileNotFoundError(f"pretrained checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model") or checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported pretrained checkpoint payload: {type(state_dict)!r}")
    source = {str(key).replace("module.", ""): value for key, value in state_dict.items()}
    target = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    missing, unexpected = model.load_state_dict(matched, strict=False)
    matched_numel = sum(int(target[key].numel()) for key in matched)
    total_numel = sum(int(value.numel()) for value in target.values())
    coverage = matched_numel / max(total_numel, 1)
    print(
        "[PrefCondDiffusion] pretrained initialization "
        f"path={path} coverage={coverage:.2%} matched={len(matched)} "
        f"missing={len(missing)} ignored_source={len(source) - len(matched)} "
        f"unexpected={len(unexpected)}"
    )
    required_coverage = float(minimum_coverage)
    if coverage < required_coverage:
        raise RuntimeError(
            "Pretrained parameter coverage is too low: "
            f"{coverage:.2%} < {required_coverage:.2%}"
        )


def _style_router_module(model: nn.Module) -> nn.Module:
    planner = model.module if hasattr(model, "module") else model
    router = planner.decoder.decoder.dit.style_condition_proj
    if router is None:
        raise RuntimeError("StylePlanner has no style condition router")
    return router


def _style_adapter_modules(model: nn.Module) -> list[nn.Module]:
    planner = model.module if hasattr(model, "module") else model
    dit = planner.decoder.decoder.dit
    modules = [_style_router_module(model)]
    ego_adapter = getattr(dit, "ego_style_output_proj", None)
    if ego_adapter is not None:
        modules.append(ego_adapter)
    return modules


def _configure_trainable_parameters(
    model: nn.Module,
    args: argparse.Namespace,
) -> None:
    if not args.freeze_planner_backbone:
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in _style_adapter_modules(model):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    trainable = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    if trainable <= 0:
        raise RuntimeError("signed StylePlanner router has no trainable parameters")
    print(
        "[PrefCondDiffusion] router_only_training "
        f"trainable_numel={trainable} frozen_numel={frozen}"
    )


def _set_train_mode(model: nn.Module, args: argparse.Namespace) -> None:
    if not args.freeze_planner_backbone:
        model.train()
        return
    # Keep the frozen planner in inference mode so dropout/batch-state cannot
    # create an apparent rho=0 drift. Only the signed router enters train mode.
    model.eval()
    for module in _style_adapter_modules(model):
        module.train()


def _build_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
) -> optim.Optimizer:
    """Use a conservative planner LR and a faster V6 style-adapter LR."""

    if args.style_condition_encoder not in {
        "axis_router_v1",
        "axis_router_v2_signed",
    }:
        return optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    style_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            "decoder.decoder.dit.style_condition_proj" in name
            or "decoder.decoder.dit.ego_style_output_proj" in name
        ):
            style_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    if not style_parameters:
        raise RuntimeError(
            f"{args.style_condition_encoder} optimizer found no style adapter parameters"
        )

    backbone_lr = float(args.learning_rate) * float(args.backbone_lr_scale)
    style_lr = float(args.learning_rate) * float(args.style_adapter_lr_scale)
    print(
        "[PrefCondDiffusion] optimizer_groups "
        f"backbone_params={len(backbone_parameters)} "
        f"backbone_lr={backbone_lr:.3e} "
        f"style_params={len(style_parameters)} style_lr={style_lr:.3e}"
    )
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_lr,
                "group_name": "planner_backbone",
            }
        )
    parameter_groups.append(
        {
            "params": style_parameters,
            "lr": style_lr,
            "group_name": "preference_axis_router",
        }
    )
    return optim.AdamW(
        parameter_groups,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )


def _train_epoch(
    data_loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    ema: ModelEma | None,
    aug: Any | None,
    axis_objective: ConditionalPreferenceEnergy | None,
) -> Dict[str, float]:
    _set_train_mode(model, args)
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
        compute_monotonic = (
            (
                float(args.signed_monotonic_loss_weight) > 0.0
                or float(args.signed_symmetry_loss_weight) > 0.0
            )
            and float(torch.rand(()).item())
            < float(args.signed_monotonic_batch_fraction)
        )
        for key, value in _compute_v6_losses(
            model=model,
            decoder_output=decoder_output,
            inputs=inputs,
            args=args,
            axis_objective=axis_objective,
            compute_monotonic=compute_monotonic,
            monotonic_random_subset=True,
        ).items():
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
        if float(args.axis_router_activity_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.axis_router_activity_loss_weight
            ) * loss_dict["axis_router_activity_loss"]
        if float(args.normal_anchor_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.normal_anchor_loss_weight
            ) * loss_dict["normal_anchor_consistency_loss"]
        if float(args.signed_raw_axis_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.signed_raw_axis_loss_weight
            ) * loss_dict["signed_raw_axis_loss"]
        if float(args.normal_relative_axis_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.normal_relative_axis_loss_weight
            ) * loss_dict["normal_relative_axis_loss"]
        if float(args.exogenous_neighbor_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.exogenous_neighbor_loss_weight
            ) * loss_dict["exogenous_neighbor_invariance_loss"]
        if float(args.signed_monotonic_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.signed_monotonic_loss_weight
            ) * loss_dict["signed_monotonic_loss"]
        if float(args.signed_symmetry_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.signed_symmetry_loss_weight
            ) * loss_dict["signed_symmetry_loss"]
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
    axis_objective: ConditionalPreferenceEnergy | None,
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
        for key, value in _compute_v6_losses(
            model=model,
            decoder_output=decoder_output,
            inputs=inputs,
            args=args,
            axis_objective=axis_objective,
            compute_monotonic=(
                float(args.signed_monotonic_loss_weight) > 0.0
                or float(args.signed_symmetry_loss_weight) > 0.0
            ),
            monotonic_random_subset=False,
        ).items():
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
        if float(args.axis_router_activity_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.axis_router_activity_loss_weight
            ) * loss_dict["axis_router_activity_loss"]
        if float(args.normal_anchor_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.normal_anchor_loss_weight
            ) * loss_dict["normal_anchor_consistency_loss"]
        if float(args.signed_raw_axis_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.signed_raw_axis_loss_weight
            ) * loss_dict["signed_raw_axis_loss"]
        if float(args.normal_relative_axis_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.normal_relative_axis_loss_weight
            ) * loss_dict["normal_relative_axis_loss"]
        if float(args.exogenous_neighbor_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.exogenous_neighbor_loss_weight
            ) * loss_dict["exogenous_neighbor_invariance_loss"]
        if float(args.signed_monotonic_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.signed_monotonic_loss_weight
            ) * loss_dict["signed_monotonic_loss"]
        if float(args.signed_symmetry_loss_weight) > 0.0:
            loss_dict["loss"] = loss_dict["loss"] + float(
                args.signed_symmetry_loss_weight
            ) * loss_dict["signed_symmetry_loss"]
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
    weight_by_metric = {
        "normal_relative_axis_loss": "normal_relative_axis_active_count",
        "normal_relative_axis_mae": "normal_relative_axis_active_count",
        "signed_monotonic_loss": "signed_monotonic_pair_count",
        "signed_monotonic_order_accuracy": "signed_monotonic_pair_count",
        "signed_monotonic_response": "signed_monotonic_pair_count",
        "signed_symmetry_loss": "signed_monotonic_pair_count",
        "signed_monotonic_worst_axis_loss": "signed_monotonic_sample_count",
        "signed_monotonic_diffusion_time_mean": "signed_monotonic_pair_count",
        "signed_monotonic_diffusion_time_min": "signed_monotonic_pair_count",
        "signed_monotonic_diffusion_time_max": "signed_monotonic_pair_count",
        "signed_monotonic_short_rollout_steps": "signed_monotonic_pair_count",
        "signed_monotonic_rollout_state_l1": "signed_monotonic_pair_count",
        "exogenous_neighbor_invariance_loss": "exogenous_neighbor_active_count",
        "exogenous_neighbor_invariance_l1": "exogenous_neighbor_active_count",
    }
    for scene in ("straight_free_drive", "straight_car_follow"):
        relative_count = f"normal_relative_axis_active_count_{scene}"
        pair_count = f"signed_monotonic_pair_count_{scene}"
        weight_by_metric[f"normal_relative_axis_mae_{scene}"] = relative_count
        weight_by_metric[f"signed_monotonic_order_accuracy_{scene}"] = pair_count
        weight_by_metric[f"signed_monotonic_response_{scene}"] = pair_count
    for axis_name in (
        "free_speed_utilization",
        "free_accel_willingness",
        "free_speed_response",
        "car_headway_tightness_from_h",
        "car_ttc_tightness",
        "car_closing_tolerance",
    ):
        axis_count = f"{axis_name}_pair_count"
        weight_by_metric[f"{axis_name}_order"] = axis_count
        weight_by_metric[f"{axis_name}_mean_response"] = axis_count
        weight_by_metric[f"{axis_name}_worst_ratio"] = axis_count
    means: Dict[str, float] = {}
    for key in keys:
        weight_key = weight_by_metric.get(key)
        if weight_key is None:
            means[key] = float(
                sum(item.get(key, 0.0) for item in logs) / max(len(logs), 1)
            )
            continue
        total_weight = sum(max(item.get(weight_key, 0.0), 0.0) for item in logs)
        if total_weight <= 0.0:
            means[key] = 0.0
            continue
        means[key] = float(
            sum(
                item.get(key, 0.0) * max(item.get(weight_key, 0.0), 0.0)
                for item in logs
            )
            / total_weight
        )
    return means


def _append_epoch_log(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _checkpoint_selection_score(
    val_stats: Dict[str, float],
    args: argparse.Namespace,
) -> float:
    if args.checkpoint_selection_metric == "total_loss":
        return float(val_stats.get("loss", float("inf")))
    if args.checkpoint_selection_metric == "signed_control_worst_scene":
        scene_scores = []
        for scene in ("straight_free_drive", "straight_car_follow"):
            relative_count = float(
                val_stats.get(f"normal_relative_axis_active_count_{scene}", 0.0)
            )
            pair_count = float(
                val_stats.get(f"signed_monotonic_pair_count_{scene}", 0.0)
            )
            if relative_count <= 0.0 or pair_count <= 0.0:
                return float("inf")
            mae = float(
                val_stats.get(
                    f"normal_relative_axis_mae_{scene}",
                    float("inf"),
                )
            )
            order_accuracy = float(
                val_stats.get(
                    f"signed_monotonic_order_accuracy_{scene}",
                    0.0,
                )
            )
            scene_scores.append(mae + max(0.0, 1.0 - order_accuracy))
        return max(scene_scores) if scene_scores else float("inf")
    relative_enabled = float(args.normal_relative_axis_loss_weight) > 0.0
    active_ratio = float(
        val_stats.get(
            "normal_relative_axis_active_ratio"
            if relative_enabled
            else "signed_raw_axis_active_ratio",
            0.0,
        )
    )
    pair_count = float(val_stats.get("signed_monotonic_pair_count", 0.0))
    if active_ratio <= 0.0 or pair_count <= 0.0:
        return float("inf")
    raw_mae = float(
        val_stats.get(
            "normal_relative_axis_mae"
            if relative_enabled
            else "signed_raw_axis_mae",
            float("inf"),
        )
    )
    order_accuracy = float(
        val_stats.get("signed_monotonic_order_accuracy", 0.0)
    )
    # Both terms live on an interpretable unit scale: normalized raw-axis error
    # plus order error. This prevents diffusion reconstruction from selecting a
    # checkpoint that has already lost signed controllability.
    return raw_mae + max(0.0, 1.0 - order_accuracy)


def _log_runtime_imports() -> None:
    planner_path = inspect.getsourcefile(Diffusion_Planner) or "unknown"
    decoder_path = inspect.getsourcefile(Decoder) or "unknown"
    loss_path = inspect.getsourcefile(diffusion_loss_func) or "unknown"
    v6_loss_path = inspect.getsourcefile(compute_signed_pair_monotonic_loss) or "unknown"
    axis_objective_path = (
        inspect.getsourcefile(ConditionalPreferenceEnergy) or "unknown"
    )
    print(f"[PrefCondDiffusion] planner_module={planner_path}")
    print(f"[PrefCondDiffusion] decoder_module={decoder_path}")
    print(f"[PrefCondDiffusion] diffusion_loss_module={loss_path}")
    print(f"[PrefCondDiffusion] v6_loss_module={v6_loss_path}")
    print(f"[PrefCondDiffusion] conditional_axis_objective_module={axis_objective_path}")


def _assert_runtime_files_are_patched(args: argparse.Namespace) -> None:
    decoder_path = inspect.getsourcefile(Decoder)
    loss_path = inspect.getsourcefile(diffusion_loss_func)
    v6_loss_path = inspect.getsourcefile(compute_signed_pair_monotonic_loss)
    axis_objective_path = inspect.getsourcefile(ConditionalPreferenceEnergy)
    if (
        not decoder_path
        or not loss_path
        or not v6_loss_path
        or not axis_objective_path
    ):
        raise RuntimeError("Failed to resolve runtime source files for decoder/loss.")

    with open(decoder_path, "r", encoding="utf-8") as file_obj:
        decoder_source = file_obj.read()
    with open(loss_path, "r", encoding="utf-8") as file_obj:
        loss_source = file_obj.read()
    with open(v6_loss_path, "r", encoding="utf-8") as file_obj:
        v6_loss_source = file_obj.read()
    with open(axis_objective_path, "r", encoding="utf-8") as file_obj:
        axis_objective_source = file_obj.read()

    decoder_markers = [
        'is_diffusion_loss_pass = ("sampled_trajectories" in inputs) and ("diffusion_time" in inputs)',
        '"x_start": denoised',
    ]
    if args.style_condition_encoder == "axis_router_v1":
        decoder_markers.extend(
            [
                "PreferenceAxisRouter",
                "normal_anchor_style_value_condition",
            ]
        )
    elif args.style_condition_encoder == "axis_router_v2_signed":
        decoder_markers.extend(
            [
                "SignedPreferenceAxisRouter",
                "axis_router_v2_signed",
            ]
        )
        if args.signed_router_injection_mode == "ego_output_residual":
            decoder_markers.extend(
                [
                    "EgoSignedOutputAdapter",
                    "ego_style_output_proj",
                ]
            )
        elif args.signed_router_injection_mode == "ego_kinematic_residual":
            decoder_markers.extend(
                [
                    "KinematicEgoSignedOutputAdapter",
                    "ego_kinematic_residual",
                ]
            )
        elif args.signed_router_injection_mode == "ego_axis_temporal_residual":
            decoder_markers.extend(
                [
                    "AxisTemporalKinematicEgoSignedOutputAdapter",
                    "ego_axis_temporal_residual",
                    "_axis_router_axis_residual",
                ]
            )
            if (
                args.signed_router_diffusion_gate_mode
                == "free_drive_terminal_only"
            ):
                decoder_markers.extend(
                    [
                        "free_drive_terminal_only",
                        "axis_temporal_diffusion_gate",
                    ]
                )
    loss_markers = [
        "def _extract_diffusion_prediction(",
        'candidate_keys = ["x_start", "score"] if model_type == "x_start" else ["score"]',
    ]
    v6_loss_markers = []
    axis_objective_markers = []
    if float(args.signed_monotonic_terminal_t_max) > 0.0:
        loss_markers.append('_training_clean_trajectories')
    if int(args.signed_monotonic_short_rollout_steps) > 0:
        v6_loss_markers.extend(
            [
                "_dpmpp_first_order_state_step",
                "short_rollout_steps",
            ]
        )
    if args.signed_monotonic_command_mode == "global_rho":
        v6_loss_markers.extend(
            [
                "command_mode",
                'pair_command_mode == "global_rho"',
            ]
        )
    if args.signed_monotonic_aggregation_mode == "soft_worst_per_sample":
        v6_loss_markers.extend(
            [
                "_soft_worst_axis_aggregate",
                "soft_worst_per_sample",
                "signed_monotonic_worst_axis_temperature",
            ]
        )
    if args.free_drive_accel_support_mode == "normal_anchor":
        decoder_markers.append("free_drive_accel_support_mode")
        v6_loss_markers.append("accel_opportunity_reference_output")
        axis_objective_markers.extend(
            [
                "accel_opportunity_reference_future",
                "preference_ego_reference_future",
            ]
        )

    missing_decoder = [marker for marker in decoder_markers if marker not in decoder_source]
    missing_loss = [marker for marker in loss_markers if marker not in loss_source]
    missing_v6_loss = [
        marker for marker in v6_loss_markers if marker not in v6_loss_source
    ]
    missing_axis_objective = [
        marker
        for marker in axis_objective_markers
        if marker not in axis_objective_source
    ]
    if (
        missing_decoder
        or missing_loss
        or missing_v6_loss
        or missing_axis_objective
    ):
        raise RuntimeError(
            "Patched diffusion runtime files are not active. "
            f"decoder_missing={missing_decoder}, loss_missing={missing_loss}, "
            f"v6_loss_missing={missing_v6_loss}, "
            f"axis_objective_missing={missing_axis_objective}, "
            f"decoder_path={decoder_path}, "
            f"loss_path={loss_path}, v6_loss_path={v6_loss_path}, "
            f"axis_objective_path={axis_objective_path}"
        )


def main() -> None:
    args = _build_args()
    set_seed(args.seed)
    _log_runtime_imports()
    _assert_runtime_files_are_patched(args)

    train_loader, val_loader = _prepare_dataloaders(args)
    save_path = build_experiment_dir(args.save_dir, args.experiment_name)
    write_json(os.path.join(save_path, "args.json"), serializable_args_dict(args))

    model = Diffusion_Planner(args).to(args.device)
    _load_pretrained_weights(
        model,
        str(args.pretrained_model_path or ""),
        minimum_coverage=(
            0.95
            if args.experiment_preset
            in {
                EXPERIMENT_PRESET_V6_SIGNED_ROUTER_STAGE_A,
                EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A,
                EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A2,
                EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_KINEMATIC,
                EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_2_TERMINAL,
                EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_3_ROLLOUT,
                EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_4_GLOBAL_PAIR,
                EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_5_NORMAL_OPPORTUNITY,
                EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_6_WORST_AXIS,
                EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_7_AXIS_TEMPORAL,
                EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_A3_8_TERMINAL_EXECUTOR,
            }
            else 0.80
        ),
    )
    _configure_trainable_parameters(model, args)
    ema = ModelEma(model, decay=0.999, device=args.device) if args.use_ema else None
    optimizer = _build_optimizer(model, args)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, args.train_epochs, args.warm_up_epoch)
    axis_objective = _build_signed_axis_objective(args)

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
    best_checkpoint_score = float("inf")
    history_path = os.path.join(save_path, "epoch_metrics.jsonl")

    try:
        for epoch in range(init_epoch, args.train_epochs):
            train_stats = _train_epoch(
                train_loader,
                model,
                optimizer,
                args,
                ema,
                aug,
                axis_objective,
            )
            val_stats = _validate_epoch(
                val_loader,
                model,
                args,
                axis_objective,
            )
            scheduler.step()

            current_val_loss = float(val_stats.get("loss", float("inf")))
            selection_score = _checkpoint_selection_score(val_stats, args)
            best_val_loss = min(best_val_loss, current_val_loss)
            epoch_payload = {
                "epoch": epoch + 1,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "checkpoint_selection_metric": args.checkpoint_selection_metric,
                "checkpoint_selection_score": selection_score,
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
            if selection_score < best_checkpoint_score:
                best_checkpoint_score = selection_score
                metric_tag = (
                    "signedcontrol_worstscene"
                    if args.checkpoint_selection_metric
                    == "signed_control_worst_scene"
                    else "signedcontrol"
                    if args.checkpoint_selection_metric == "signed_control"
                    else "valloss"
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
                    filename=(
                        f"best_pref_cond_epoch_{epoch + 1}_{metric_tag}_"
                        f"{selection_score:.4f}.pth"
                    ),
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
                f"selection_score={selection_score:.4f} "
                f"order_acc={val_stats.get('signed_monotonic_order_accuracy', 0.0):.3f} "
                f"free_order={val_stats.get('signed_monotonic_order_accuracy_straight_free_drive', 0.0):.3f} "
                f"car_order={val_stats.get('signed_monotonic_order_accuracy_straight_car_follow', 0.0):.3f} "
                f"pair_mode={args.signed_monotonic_command_mode} "
                f"accel_support={args.free_drive_accel_support_mode} "
                f"pair_t={val_stats.get('signed_monotonic_diffusion_time_mean', 0.0):.3f} "
                f"pair_unroll={val_stats.get('signed_monotonic_short_rollout_steps', 0.0):.0f} "
                f"rollout_state_l1={val_stats.get('signed_monotonic_rollout_state_l1', 0.0):.6f} "
                f"exo_l1={val_stats.get('exogenous_neighbor_invariance_l1', 0.0):.6f} "
                f"cond_used={train_stats.get('style_condition_used_ratio', 0.0):.3f}"
            )
    finally:
        online_logger.finish()

    write_json(
        os.path.join(save_path, "train_summary.json"),
        {
            "best_val_loss": best_val_loss,
            "best_checkpoint_score": best_checkpoint_score,
            "checkpoint_selection_metric": args.checkpoint_selection_metric,
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

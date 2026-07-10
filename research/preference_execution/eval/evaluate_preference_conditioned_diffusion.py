"""Open-loop controllability evaluation for preference-conditioned diffusion planning.

This script evaluates whether a trained preference-conditioned diffusion planner
responds to the calibrated preference sweep in the intended direction.

Default paths intentionally point to the server-side experiment/data layout.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
for _path in (REPO_ROOT, DEVKIT_ROOT):
    _path_str = str(_path)
    if _path.exists() and _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from research._runtime import DEFAULT_CACHE_TRAIN_VAL_DIR, ensure_repo_on_path

ensure_repo_on_path()

from baseline.model.style_planner.diffusion_planner import Diffusion_Planner
from baseline.utils.normalizer import ObservationNormalizer, StateNormalizer
from research.preference_execution.calibration.builder import STYLE_SWEEP_ORDER
from research.preference_execution.calibration.schema import calibration_index_path, calibration_output_dir
from research.preference_execution.diffusion.dataset import PreferenceConditionedPlannerData
from research.preference_execution.diffusion.style_condition import (
    STYLE_CONDITION_FEATURE_SET_EXEC_V2,
    STYLE_CONDITION_FEATURE_SET_PHASEWISE_EXEC_V1,
    STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1,
    build_style_condition_feature,
    global_style_condition_dim,
    phase_style_condition_dim,
    phase_style_flat_dim,
    phase_style_num_phases,
    resolve_style_condition_feature_set,
    style_condition_dim,
    style_condition_valid_mask,
    use_temporal_style_gate,
    validate_style_condition_args,
)
from research.preference_execution.diffusion.training import prepare_preference_conditioned_batch, write_json
from research.preference_execution.interaction_state.schema import AXIS_GATE_ORDER
from research.style_scene_split.defaults import DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR, PRIMARY_SCENE_BUCKETS
from research.style_scene_split.schema_v2 import style_axis_names_for_scene

DEFAULT_EXPERIMENT_DIR = (
    "/mnt/mydata/lishangwen/Nuplan-Baseline-Record/"
    "research_train/preference_conditioned_diffusion/"
    "effective_preference_global_vec/2026_06_19-22_57_29"
)
CONDITION_SOURCE_TO_FIELD = {
    "effective": "effective_preference_global_vec",
    "safe": "safe_preference_global_vec",
    "target": "target_preference_global_vec",
}
CONDITION_SOURCE_TO_SCENE_KEY = {
    "effective": "effective_scene_vec",
    "safe": "safe_scene_vec",
    "target": "target_scene_vec",
}
BEST_CHECKPOINT_PATTERN = re.compile(
    r"best_pref_cond_epoch_(?P<epoch>\d+)_valloss_(?P<val>[0-9.]+)\.pth$"
)


@dataclass
class ProxyBaseResult:
    """Minimal scene-style proxy bundle used to score predicted trajectories."""

    scene_bucket: str
    ego_speed_ratio_to_limit: float | None
    ego_accel_peak: float
    ego_jerk_p90: float
    ego_brake_peak: float
    following_min_thw: float | None
    following_min_gap: float | None
    event_brake_peak: float
    event_speed_drop_ratio: float
    merge_min_gap: float | None
    ego_lateral_onset_step: float | None
    ego_lateral_speed_peak: float
    ego_heading_change: float
    following_context_valid: bool
    merge_context_valid: bool
    lateral_motion_valid: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate preference-conditioned diffusion controllability on the validation split."
    )
    parser.add_argument("--experiment_dir", default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--split_root", default=DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR)
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_TRAIN_VAL_DIR))
    parser.add_argument("--calibration_index_path", default=None)
    parser.add_argument(
        "--condition_source",
        default="effective",
        choices=tuple(CONDITION_SOURCE_TO_FIELD.keys()),
        help="Which calibrated scene-vector family to sweep during evaluation.",
    )
    parser.add_argument(
        "--scene_bucket",
        default=None,
        choices=(None,) + tuple(PRIMARY_SCENE_BUCKETS),
        help="Optionally restrict evaluation to one scene bucket.",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1024,
        help="Safety cap for evaluation cost. Set to 0 or a negative value for the full split.",
    )
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--num_rollouts_per_condition", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cfg_guidance_scale", type=float, default=None)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--traj_smooth_window",
        type=int,
        default=5,
        help="Centered moving-average window for predicted trajectories before kinematic proxy extraction.",
    )
    parser.add_argument(
        "--kinematics_stride",
        type=int,
        default=3,
        help="Finite-difference stride (in frames) used for speed/accel/jerk proxy estimation.",
    )
    parser.add_argument("--monotonic_tol", type=float, default=1e-4)
    parser.add_argument("--prefer_ema", action="store_true", default=True)
    parser.add_argument("--disable_prefer_ema", action="store_true")
    parser.add_argument("--detail_jsonl_path", default=None)
    parser.add_argument("--summary_json_path", default=None)
    return parser


def _parse_args() -> argparse.Namespace:
    args = _build_parser().parse_args()
    if args.disable_prefer_ema:
        args.prefer_ema = False
    if args.sample_stride <= 0:
        raise ValueError(f"sample_stride must be > 0, got {args.sample_stride}")
    if args.num_rollouts_per_condition <= 0:
        raise ValueError(
            f"num_rollouts_per_condition must be > 0, got {args.num_rollouts_per_condition}"
        )
    if args.traj_smooth_window <= 0:
        raise ValueError(f"traj_smooth_window must be > 0, got {args.traj_smooth_window}")
    if args.kinematics_stride <= 0:
        raise ValueError(f"kinematics_stride must be > 0, got {args.kinematics_stride}")
    if args.max_samples is not None and args.max_samples <= 0:
        args.max_samples = None
    experiment_dir = Path(args.experiment_dir)
    if args.detail_jsonl_path is None:
        args.detail_jsonl_path = str(
            experiment_dir
            / f"eval_preference_conditioned_diffusion_{args.condition_source}_details.jsonl"
        )
    if args.summary_json_path is None:
        args.summary_json_path = str(
            experiment_dir
            / f"eval_preference_conditioned_diffusion_{args.condition_source}_summary.json"
        )
    if args.calibration_index_path is None:
        args.calibration_index_path = calibration_index_path(calibration_output_dir(args.split_root))
    return args


def _set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                yield json.loads(line)


def _resolve_checkpoint_path(experiment_dir: str, checkpoint_path: str | None) -> str:
    if checkpoint_path:
        return checkpoint_path

    exp_dir = Path(experiment_dir)
    best_candidates: List[tuple[float, int, Path]] = []
    for path_obj in exp_dir.glob("best_pref_cond_epoch_*_valloss_*.pth"):
        match = BEST_CHECKPOINT_PATTERN.match(path_obj.name)
        if match is None:
            continue
        best_candidates.append(
            (
                float(match.group("val")),
                int(match.group("epoch")),
                path_obj,
            )
        )
    if best_candidates:
        best_candidates.sort(key=lambda item: (item[0], item[1]))
        return str(best_candidates[0][2])

    latest_path = exp_dir / "latest.pth"
    if latest_path.exists():
        return str(latest_path)
    raise FileNotFoundError(
        f"Could not resolve checkpoint under {experiment_dir}. "
        "Expected a best_pref_cond_epoch_*.pth or latest.pth file."
    )


def _resolve_normalization_path(train_args: Namespace) -> str:
    candidate = Path(str(train_args.normalization_file_path))
    if candidate.exists():
        return str(candidate)
    fallback = REPO_ROOT / "baseline" / "resources" / "normalization_train.json"
    if fallback.exists():
        return str(fallback)
    return str(candidate)


def _load_model_args(experiment_dir: str, cli_args: argparse.Namespace) -> Namespace:
    args_path = Path(experiment_dir) / "args.json"
    train_payload = _load_json(args_path)
    model_args = Namespace(**train_payload)
    model_args.device = cli_args.device
    model_args.condition_field = CONDITION_SOURCE_TO_FIELD[cli_args.condition_source]
    model_args.cfg_guidance_scale = (
        float(cli_args.cfg_guidance_scale)
        if cli_args.cfg_guidance_scale is not None
        else float(getattr(model_args, "cfg_guidance_scale", 1.5))
    )
    model_args.style_condition_feature_set = resolve_style_condition_feature_set(model_args)
    validate_style_condition_args(model_args.condition_field, model_args.style_condition_feature_set)
    model_args.global_style_condition_dim = global_style_condition_dim(
        model_args.style_condition_feature_set,
        base_global_dim=len(AXIS_GATE_ORDER),
    )
    model_args.phase_style_condition_dim = phase_style_condition_dim(model_args.style_condition_feature_set)
    model_args.phase_style_num_phases = phase_style_num_phases(model_args.style_condition_feature_set)
    model_args.phase_style_flat_dim = phase_style_flat_dim(model_args.style_condition_feature_set)
    model_args.style_value_dim = style_condition_dim(
        model_args.style_condition_feature_set,
        base_global_dim=len(AXIS_GATE_ORDER),
    )
    model_args.use_style_condition = True
    model_args.use_phase_style_condition = bool(model_args.phase_style_flat_dim > 0)
    model_args.use_temporal_style_gate = use_temporal_style_gate(model_args.style_condition_feature_set)
    model_args.guidance_fn = None
    model_args.normalization_file_path = _resolve_normalization_path(model_args)
    model_args.state_normalizer = StateNormalizer.from_json(model_args)
    model_args.observation_normalizer = ObservationNormalizer.from_json(model_args)
    return model_args


def _load_model(
    model_args: Namespace,
    checkpoint_path: str,
    *,
    prefer_ema: bool,
) -> tuple[Diffusion_Planner, Dict[str, Any]]:
    device = torch.device(model_args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    weight_source = "model"
    state_dict = checkpoint.get("model", checkpoint)
    if prefer_ema and checkpoint.get("ema_state_dict") is not None:
        state_dict = checkpoint["ema_state_dict"]
        weight_source = "ema_state_dict"

    model = Diffusion_Planner(model_args).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    meta = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "loss": float(checkpoint.get("loss", 0.0)),
        "weight_source": weight_source,
    }
    return model, meta


def _load_calibration_map(calibration_path: str) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for record in _iter_jsonl(calibration_path):
        sample_id = str(record.get("sample_id", "")).strip()
        if sample_id:
            records[sample_id] = record
    if not records:
        raise RuntimeError(f"No calibration records found in {calibration_path}")
    return records


def _tensor_batch_from_sample(sample: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
    return batch


def _clone_inputs(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _to_xycs(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states[None, :]
    if states.shape[-1] >= 4:
        return states[..., :4].astype(np.float32)
    if states.shape[-1] >= 3:
        heading = states[..., 2]
        cos_sin = np.stack([np.cos(heading), np.sin(heading)], axis=-1).astype(np.float32)
        return np.concatenate([states[..., :2], cos_sin], axis=-1).astype(np.float32)
    raise ValueError(f"Expected state dim >= 3, got shape {states.shape}")


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _score_rising(value: float, low: float, high: float) -> float:
    if high <= low:
        return float(value >= high)
    return _clip01((float(value) - low) / (high - low))


def _score_falling(value: float, low: float, high: float) -> float:
    if high <= low:
        return float(value <= low)
    return _clip01((high - float(value)) / (high - low))


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _optional_value(value: float | None, default: float, *, large_invalid: bool = False) -> float:
    if value is None:
        return float(default)
    resolved = float(value)
    if not math.isfinite(resolved):
        return float(default)
    if large_invalid and abs(resolved) >= 1e5:
        return float(default)
    return resolved


def _safe_percentile(values: np.ndarray, percentile: float, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, percentile))


def _safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    a_array = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    b_array = np.nan_to_num(np.asarray(b, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if a_array.size == 0 or b_array.size == 0:
        return 0.0
    std_a = float(np.std(a_array))
    std_b = float(np.std(b_array))
    if (not math.isfinite(std_a)) or (not math.isfinite(std_b)) or std_a <= 1e-8 or std_b <= 1e-8:
        return 0.0
    corr = float(np.corrcoef(a_array, b_array)[0, 1])
    if not math.isfinite(corr):
        return 0.0
    return corr


def _is_non_decreasing(values: Sequence[float], tolerance: float) -> bool:
    seq = np.asarray(values, dtype=np.float32)
    if seq.size <= 1:
        return True
    return bool(np.all(np.diff(seq) >= -float(tolerance)))


def _make_global_vec(axis_names: Sequence[str], local_values: Sequence[float]) -> np.ndarray:
    global_vec = np.zeros(len(AXIS_GATE_ORDER), dtype=np.float32)
    for axis_name, value in zip(axis_names, local_values):
        global_vec[AXIS_GATE_ORDER.index(str(axis_name))] = float(value)
    return global_vec


def _route_speed_limit_mps(sample: Mapping[str, Any]) -> float | None:
    def _extract(limit_key: str, mask_key: str) -> np.ndarray:
        limits = np.asarray(_tensor_to_numpy(sample[limit_key]), dtype=np.float32).reshape(-1)
        mask = np.asarray(_tensor_to_numpy(sample[mask_key]), dtype=bool).reshape(-1)
        if limits.size == 0 or mask.size == 0:
            return np.zeros((0,), dtype=np.float32)
        size = min(limits.size, mask.size)
        valid = limits[:size][mask[:size]]
        valid = valid[np.isfinite(valid)]
        valid = valid[valid > 0]
        return valid

    route_valid = _extract("route_lanes_speed_limit", "route_lanes_has_speed_limit")
    if route_valid.size > 0:
        return float(np.mean(route_valid))
    lane_valid = _extract("lanes_speed_limit", "lanes_has_speed_limit")
    if lane_valid.size > 0:
        return float(np.mean(lane_valid))
    return None


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    series = np.asarray(values, dtype=np.float32)
    if series.size == 0 or int(window) <= 1:
        return series.astype(np.float32)
    window = min(int(window), int(series.size))
    if window <= 1:
        return series.astype(np.float32)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(series, (pad_left, pad_right), mode="edge")
    kernel = np.full((window,), 1.0 / float(window), dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _moving_average_matrix(values: np.ndarray, window: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or int(window) <= 1:
        return matrix.astype(np.float32)
    return np.stack(
        [_moving_average_1d(matrix[:, column], window) for column in range(matrix.shape[1])],
        axis=1,
    ).astype(np.float32)


def _smooth_trajectory_xycs(traj: np.ndarray, window: int) -> np.ndarray:
    smoothed = np.asarray(traj, dtype=np.float32).copy()
    if smoothed.ndim != 2 or smoothed.shape[0] <= 1 or int(window) <= 1:
        return smoothed
    smoothed[:, :2] = _moving_average_matrix(smoothed[:, :2], window)
    if smoothed.shape[1] >= 4:
        cos_sin = _moving_average_matrix(smoothed[:, 2:4], window)
        norm = np.linalg.norm(cos_sin, axis=-1, keepdims=True)
        cos_sin = np.divide(
            cos_sin,
            np.maximum(norm, 1e-6),
            out=np.zeros_like(cos_sin),
            where=norm > 1e-6,
        )
        smoothed[:, 2:4] = cos_sin
    smoothed[0] = np.asarray(traj, dtype=np.float32)[0]
    return smoothed


def _speed_from_xy(ego_xy: np.ndarray, dt: float, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    if ego_xy.shape[0] <= stride:
        return np.zeros((0,), dtype=np.float32)
    delta = ego_xy[stride:] - ego_xy[:-stride]
    speed_core = (np.linalg.norm(delta, axis=-1) / max(dt * stride, 1e-6)).astype(np.float32)
    if stride > 1 and speed_core.size > 0:
        speed_core = np.concatenate(
            [np.full((stride - 1,), speed_core[0], dtype=np.float32), speed_core],
            axis=0,
        )
    return speed_core.astype(np.float32)


def _finite_difference_1d(values: np.ndarray, dt: float, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    series = np.asarray(values, dtype=np.float32)
    if series.size <= stride:
        return np.zeros((0,), dtype=np.float32)
    diff_core = ((series[stride:] - series[:-stride]) / max(dt * stride, 1e-6)).astype(np.float32)
    if stride > 1 and diff_core.size > 0:
        diff_core = np.concatenate(
            [np.full((stride - 1,), diff_core[0], dtype=np.float32), diff_core],
            axis=0,
        )
    return diff_core.astype(np.float32)


def _accel_from_speed(speed: np.ndarray, dt: float, stride: int) -> np.ndarray:
    return _finite_difference_1d(speed, dt, stride)


def _jerk_from_accel(accel: np.ndarray, dt: float, stride: int) -> np.ndarray:
    return _finite_difference_1d(accel, dt, stride)


def _robust_positive_peak(values: np.ndarray, percentile: float = 90.0) -> float:
    series = np.asarray(values, dtype=np.float32)
    positive = series[series > 0.0]
    return _safe_percentile(positive, percentile, default=0.0)


def _robust_negative_peak(values: np.ndarray, percentile: float = 90.0) -> float:
    series = np.asarray(values, dtype=np.float32)
    negative_mag = (-series[series < 0.0]).astype(np.float32)
    return _safe_percentile(negative_mag, percentile, default=0.0)


def _robust_speed_drop_ratio(speed: np.ndarray) -> float:
    series = np.asarray(speed, dtype=np.float32)
    if series.size == 0:
        return 0.0
    high = _safe_percentile(series, 90.0, default=float(np.max(series)))
    low = _safe_percentile(series, 10.0, default=float(np.min(series)))
    if high <= 1e-3:
        return 0.0
    return _clip01((high - low) / max(high, 1e-3))


def _build_neighbor_future_context(
    sample: Mapping[str, Any],
    predicted_neighbor_num: int,
) -> tuple[np.ndarray, np.ndarray]:
    neighbors_current = _tensor_to_numpy(sample["neighbor_agents_past"])[:predicted_neighbor_num, -1, :4]
    neighbors_future = _tensor_to_numpy(sample["neighbors_future_gt"])[:predicted_neighbor_num]
    neighbors_future_xycs = _to_xycs(neighbors_future)
    neighbors_full = np.concatenate([neighbors_current[:, None, :4], neighbors_future_xycs], axis=1)
    valid_mask = np.linalg.norm(neighbors_full[:, :, :2], axis=-1) > 1e-4
    return neighbors_full.astype(np.float32), valid_mask


def _lead_follow_metrics(
    ego_full_xycs: np.ndarray,
    speed_with_current: np.ndarray,
    neighbor_full_xycs: np.ndarray,
    neighbor_valid: np.ndarray,
) -> tuple[float | None, float | None]:
    gap_values: List[float] = []
    thw_values: List[float] = []
    time_steps = min(ego_full_xycs.shape[0], neighbor_full_xycs.shape[1])
    for step in range(time_steps):
        rel_xy = neighbor_full_xycs[:, step, :2] - ego_full_xycs[step, :2]
        valid = neighbor_valid[:, step] & (rel_xy[:, 0] > 0.0) & (np.abs(rel_xy[:, 1]) < 4.0)
        if not np.any(valid):
            continue
        lead_gap = float(np.min(rel_xy[valid, 0]))
        speed_now = float(speed_with_current[min(step, speed_with_current.shape[0] - 1)])
        thw = lead_gap / max(speed_now, 0.1)
        gap_values.append(lead_gap)
        thw_values.append(thw)
    if not gap_values:
        return None, None
    return (
        _safe_percentile(np.asarray(gap_values, dtype=np.float32), 10.0, default=float(gap_values[0])),
        _safe_percentile(np.asarray(thw_values, dtype=np.float32), 10.0, default=float(thw_values[0])),
    )


def _lane_change_metrics(
    ego_full_xycs: np.ndarray,
    neighbor_full_xycs: np.ndarray,
    neighbor_valid: np.ndarray,
    dt: float,
) -> tuple[float | None, float, float | None, float]:
    lateral_offset = np.abs(ego_full_xycs[:, 1] - ego_full_xycs[0, 1])
    onset_candidates = np.where(lateral_offset >= 0.5)[0]
    onset_candidates = onset_candidates[onset_candidates > 0]
    onset_step = float(onset_candidates[0]) if onset_candidates.size > 0 else None

    dy = np.diff(ego_full_xycs[:, 1], axis=0)
    lateral_speed_series = (np.abs(dy) / max(dt, 1e-6)).astype(np.float32) if dy.size > 0 else np.zeros((0,), dtype=np.float32)
    lateral_speed_peak = _safe_percentile(lateral_speed_series, 90.0, default=0.0)

    heading = np.arctan2(ego_full_xycs[:, 3], ego_full_xycs[:, 2]).astype(np.float32)
    heading_change = _wrap_angle(float(heading[-1] - heading[0]))

    merge_gap: float | None = None
    if onset_step is not None:
        step = min(int(onset_step), neighbor_full_xycs.shape[1] - 1)
        rel_xy = neighbor_full_xycs[:, step, :2] - ego_full_xycs[step, :2]
        valid = neighbor_valid[:, step] & (np.abs(rel_xy[:, 1]) < 4.5)
        if np.any(valid):
            merge_gap = float(np.min(np.abs(rel_xy[valid, 0])))

    return merge_gap, lateral_speed_peak, onset_step, heading_change


def _proxy_base_from_prediction(
    sample: Mapping[str, Any],
    ego_future_xycs: np.ndarray,
    *,
    dt: float,
    traj_smooth_window: int,
    kinematics_stride: int,
    predicted_neighbor_num: int,
) -> tuple[ProxyBaseResult, Dict[str, float]]:
    ego_future_xycs = np.nan_to_num(
        np.asarray(ego_future_xycs, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    ego_current_xycs = _tensor_to_numpy(sample["ego_current_state"])[:4].astype(np.float32)
    ego_full_xycs = np.concatenate([ego_current_xycs[None, :], ego_future_xycs], axis=0).astype(np.float32)
    ego_full_xycs = _smooth_trajectory_xycs(ego_full_xycs, traj_smooth_window)
    ego_xy = ego_full_xycs[:, :2]

    speed = _speed_from_xy(ego_xy, dt, kinematics_stride)
    speed_with_current = (
        np.concatenate([[speed[0]], speed], axis=0)
        if speed.size > 0
        else np.zeros((ego_full_xycs.shape[0],), dtype=np.float32)
    )
    accel = _accel_from_speed(speed, dt, kinematics_stride)
    jerk = _jerk_from_accel(accel, dt, kinematics_stride)

    route_speed_limit_mps = _route_speed_limit_mps(sample)
    mean_speed = float(np.mean(speed)) if speed.size > 0 else 0.0
    speed_ratio = None
    if route_speed_limit_mps is not None and route_speed_limit_mps > 1e-3:
        speed_ratio = mean_speed / float(route_speed_limit_mps)

    neighbor_full_xycs, neighbor_valid = _build_neighbor_future_context(sample, predicted_neighbor_num)
    following_min_gap, following_min_thw = _lead_follow_metrics(
        ego_full_xycs,
        speed_with_current,
        neighbor_full_xycs,
        neighbor_valid,
    )
    merge_min_gap, lateral_speed_peak, lateral_onset_step, heading_change = _lane_change_metrics(
        ego_full_xycs,
        neighbor_full_xycs,
        neighbor_valid,
        dt,
    )

    accel_peak = _robust_positive_peak(accel, percentile=90.0)
    brake_peak = _robust_negative_peak(accel, percentile=90.0)
    speed_drop_ratio = _robust_speed_drop_ratio(speed)
    lateral_motion_valid = bool(
        (lateral_onset_step is not None)
        or (lateral_speed_peak >= 0.35)
        or (abs(heading_change) >= 0.03)
    )
    following_context_valid = bool((following_min_gap is not None) or (following_min_thw is not None))
    merge_context_valid = bool(merge_min_gap is not None)

    proxy = ProxyBaseResult(
        scene_bucket=str(sample["scene_bucket"]),
        ego_speed_ratio_to_limit=speed_ratio,
        ego_accel_peak=max(accel_peak, 0.0),
        ego_jerk_p90=_safe_percentile(np.abs(jerk), 90.0, default=0.0),
        ego_brake_peak=max(brake_peak, 0.0),
        following_min_thw=following_min_thw,
        following_min_gap=following_min_gap,
        event_brake_peak=max(brake_peak, 0.0),
        event_speed_drop_ratio=max(speed_drop_ratio, 0.0),
        merge_min_gap=merge_min_gap,
        ego_lateral_onset_step=lateral_onset_step,
        ego_lateral_speed_peak=max(lateral_speed_peak, 0.0),
        ego_heading_change=heading_change,
        following_context_valid=following_context_valid,
        merge_context_valid=merge_context_valid,
        lateral_motion_valid=lateral_motion_valid,
    )
    aux = {
        "ego_mean_speed": mean_speed,
        "route_speed_limit_mps": float(route_speed_limit_mps) if route_speed_limit_mps is not None else -1.0,
        "ego_accel_peak": proxy.ego_accel_peak,
        "ego_brake_peak": proxy.ego_brake_peak,
        "ego_jerk_p90": proxy.ego_jerk_p90,
        "following_min_gap": float(proxy.following_min_gap) if proxy.following_min_gap is not None else -1.0,
        "following_min_thw": float(proxy.following_min_thw) if proxy.following_min_thw is not None else -1.0,
        "merge_min_gap": float(proxy.merge_min_gap) if proxy.merge_min_gap is not None else -1.0,
        "ego_lateral_onset_step": float(proxy.ego_lateral_onset_step)
        if proxy.ego_lateral_onset_step is not None
        else -1.0,
        "ego_lateral_speed_peak": proxy.ego_lateral_speed_peak,
        "ego_heading_change": proxy.ego_heading_change,
        "event_speed_drop_ratio": proxy.event_speed_drop_ratio,
        "event_brake_peak": proxy.event_brake_peak,
        "following_context_valid": float(following_context_valid),
        "merge_context_valid": float(merge_context_valid),
        "lateral_motion_valid": float(lateral_motion_valid),
    }
    return proxy, aux


def _axis_valid_mask(scene_bucket: str, proxy: ProxyBaseResult) -> np.ndarray:
    if scene_bucket == "straight_free_drive":
        return np.asarray([True, True, True], dtype=bool)
    if scene_bucket == "straight_car_follow":
        return np.asarray([proxy.following_context_valid] * 3, dtype=bool)
    if scene_bucket == "straight_lane_change":
        return np.asarray(
            [
                proxy.merge_context_valid,
                proxy.lateral_motion_valid,
                proxy.lateral_motion_valid,
            ],
            dtype=bool,
        )
    return np.asarray([False, False, False], dtype=bool)


def _style_proxy_vec(base_result: ProxyBaseResult) -> np.ndarray:
    scene_bucket = str(base_result.scene_bucket)
    if scene_bucket == "straight_free_drive":
        speed_ratio = _optional_value(base_result.ego_speed_ratio_to_limit, default=0.70)
        speed_preference = _score_rising(speed_ratio, 0.55, 0.95)
        longitudinal_intensity = _clip01(
            0.55 * _score_rising(base_result.ego_accel_peak, 0.6, 1.8)
            + 0.45 * _score_rising(base_result.ego_jerk_p90, 12.0, 45.0)
        )
        smoothness = _clip01(
            0.60 * _score_falling(base_result.ego_jerk_p90, 12.0, 45.0)
            + 0.40 * _score_falling(base_result.ego_brake_peak, 0.6, 3.0)
        )
        return np.asarray([speed_preference, longitudinal_intensity, smoothness], dtype=np.float32)

    if scene_bucket == "straight_car_follow":
        min_thw = _optional_value(base_result.following_min_thw, default=2.0)
        min_gap = _optional_value(base_result.following_min_gap, default=16.0)
        headway_margin = _clip01(
            0.55 * _score_rising(min_thw, 1.1, 3.0)
            + 0.45 * _score_rising(min_gap, 8.0, 26.0)
        )
        response_decisiveness = _clip01(
            0.50 * _score_rising(base_result.event_brake_peak, 0.5, 2.4)
            + 0.50 * _score_rising(base_result.event_speed_drop_ratio, 0.03, 0.22)
        )
        response_smoothness = _clip01(
            0.55 * _score_falling(base_result.event_brake_peak, 0.5, 2.4)
            + 0.45 * _score_falling(base_result.event_speed_drop_ratio, 0.03, 0.22)
        )
        return np.asarray(
            [headway_margin, response_decisiveness, response_smoothness],
            dtype=np.float32,
        )

    if scene_bucket == "straight_lane_change":
        merge_gap = _optional_value(base_result.merge_min_gap, default=18.0, large_invalid=True)
        onset_step = _optional_value(base_result.ego_lateral_onset_step, default=28.0)
        gap_acceptance = _score_falling(merge_gap, 10.0, 28.0)
        lateral_commitment = _clip01(
            0.55 * _score_falling(onset_step, 8.0, 35.0)
            + 0.45 * _score_rising(base_result.ego_lateral_speed_peak, 0.5, 1.8)
        )
        execution_smoothness = _clip01(
            0.60 * _score_falling(base_result.ego_lateral_speed_peak, 0.6, 1.8)
            + 0.40 * _score_falling(abs(base_result.ego_heading_change), 0.03, 0.18)
        )
        return np.asarray(
            [gap_acceptance, lateral_commitment, execution_smoothness],
            dtype=np.float32,
        )

    return np.zeros((3,), dtype=np.float32)


def _predict_ego_trajectory(model: Diffusion_Planner, inputs: Mapping[str, Any]) -> np.ndarray:
    with torch.no_grad():
        _, decoder_output = model(inputs)
    prediction = decoder_output.get("prediction")
    if prediction is None:
        raise KeyError(f"Expected inference decoder output to contain 'prediction', got {sorted(decoder_output.keys())}")
    return _tensor_to_numpy(prediction[0, 0]).astype(np.float32)


def _ade_fde(pred_xycs: np.ndarray, gt_future: torch.Tensor) -> tuple[float, float]:
    pred_xy = np.asarray(pred_xycs[:, :2], dtype=np.float32)
    gt_xy = _tensor_to_numpy(gt_future)[..., :2].astype(np.float32)
    steps = min(pred_xy.shape[0], gt_xy.shape[0])
    if steps <= 0:
        return 0.0, 0.0
    errors = np.linalg.norm(pred_xy[:steps] - gt_xy[:steps], axis=-1)
    return float(np.mean(errors)), float(errors[-1])


def _mean_list(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _rollout_seed(
    base_seed: int,
    *,
    dataset_index: int,
    style_offset: int,
    rollout_index: int,
) -> int:
    return int(base_seed + dataset_index * 10007 + style_offset * 257 + rollout_index)


def _build_eval_style_condition(
    *,
    axis_names: Sequence[str],
    calibration_record: Mapping[str, Any],
    sample: Mapping[str, Any],
    style_label: str,
    model_args: Namespace,
    eval_args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    if model_args.style_condition_feature_set in (
        STYLE_CONDITION_FEATURE_SET_EXEC_V2,
        STYLE_CONDITION_FEATURE_SET_PHASEWISE_EXEC_V1,
        STYLE_CONDITION_FEATURE_SET_TWO_STAGE_EXEC_V1,
    ):
        if eval_args.condition_source != "effective":
            raise ValueError(
                "phase-aware effective condition features currently require "
                "--condition_source effective during evaluation."
            )

        target_scene_vec = torch.as_tensor(
            calibration_record["sweep"]["target_scene_vecs"][style_label],
            dtype=torch.float32,
            device=device,
        )
        effective_scene_vec = torch.as_tensor(
            calibration_record["sweep"]["effective_scene_vecs"][style_label],
            dtype=torch.float32,
            device=device,
        )
        safe_scene_vec = torch.as_tensor(
            calibration_record["sweep"]["safe_scene_vecs"][style_label],
            dtype=torch.float32,
            device=device,
        )
        local_axis_gate_values = torch.as_tensor(
            sample["local_axis_gate_values"],
            dtype=torch.float32,
            device=device,
        )
        base_global_vec = torch.as_tensor(
            _make_global_vec(axis_names, _tensor_to_numpy(effective_scene_vec)),
            dtype=torch.float32,
            device=device,
        )
        return build_style_condition_feature(
            base_global_vec,
            feature_set=model_args.style_condition_feature_set,
            target_scene_vec=target_scene_vec,
            safe_scene_vec=safe_scene_vec,
            effective_scene_vec=effective_scene_vec,
            local_axis_gate_values=local_axis_gate_values,
            target_global_vec=torch.as_tensor(
                _make_global_vec(axis_names, _tensor_to_numpy(target_scene_vec)),
                dtype=torch.float32,
                device=device,
            ),
            safe_global_vec=torch.as_tensor(
                _make_global_vec(axis_names, _tensor_to_numpy(safe_scene_vec)),
                dtype=torch.float32,
                device=device,
            ),
            scene_buckets=str(sample["scene_bucket"]),
        )

    sweep_scene_key = f"{eval_args.condition_source}_scene_vecs"
    target_local_vec = np.asarray(calibration_record["sweep"][sweep_scene_key][style_label], dtype=np.float32)
    target_global_vec = torch.as_tensor(
        _make_global_vec(axis_names, target_local_vec),
        dtype=torch.float32,
        device=device,
    )
    return build_style_condition_feature(
        target_global_vec,
        feature_set=model_args.style_condition_feature_set,
    )


def _evaluate_sample(
    sample: Mapping[str, Any],
    calibration_record: Mapping[str, Any],
    model: Diffusion_Planner,
    model_args: Namespace,
    eval_args: argparse.Namespace,
    *,
    dataset_index: int,
) -> Dict[str, Any]:
    batch = _tensor_batch_from_sample(sample)
    batch["scene_bucket"] = str(sample["scene_bucket"])
    base_inputs, _, _, _, _ = prepare_preference_conditioned_batch(batch, model_args, train=False, aug=None)

    observed_rollout_vecs: List[np.ndarray] = []
    observed_ades: List[float] = []
    observed_fdes: List[float] = []
    observed_aux: Dict[str, List[float]] = defaultdict(list)
    observed_valid_masks: List[np.ndarray] = []
    for rollout_index in range(eval_args.num_rollouts_per_condition):
        _set_eval_seed(
            _rollout_seed(
                eval_args.seed,
                dataset_index=dataset_index,
                style_offset=0,
                rollout_index=rollout_index,
            )
        )
        pred_observed = _predict_ego_trajectory(model, base_inputs)
        proxy_observed, aux_observed = _proxy_base_from_prediction(
            sample,
            pred_observed,
            dt=eval_args.dt,
            traj_smooth_window=eval_args.traj_smooth_window,
            kinematics_stride=eval_args.kinematics_stride,
            predicted_neighbor_num=model_args.predicted_neighbor_num,
        )
        observed_rollout_vecs.append(_style_proxy_vec(proxy_observed))
        observed_valid_masks.append(_axis_valid_mask(str(sample["scene_bucket"]), proxy_observed))
        ade, fde = _ade_fde(pred_observed, sample["ego_future_gt"])
        observed_ades.append(ade)
        observed_fdes.append(fde)
        for key, value in aux_observed.items():
            observed_aux[key].append(float(value))

    observed_vec = np.nan_to_num(
        np.mean(np.stack(observed_rollout_vecs, axis=0), axis=0).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    observed_aux_mean = {key: _mean_list(value_list) for key, value_list in observed_aux.items()}

    axis_names = list(calibration_record["scene_axis_names"])
    scene_bucket = str(calibration_record["scene_bucket"])
    current_scene_key = CONDITION_SOURCE_TO_SCENE_KEY[eval_args.condition_source]
    observed_target_vec = np.asarray(
        calibration_record["current"][current_scene_key],
        dtype=np.float32,
    )

    sweep_scene_key = f"{eval_args.condition_source}_scene_vecs"
    sweep_targets_by_style = {
        style_label: np.asarray(calibration_record["sweep"][sweep_scene_key][style_label], dtype=np.float32)
        for style_label in STYLE_SWEEP_ORDER
    }

    sweep_pred_by_style: Dict[str, np.ndarray] = {}
    sweep_aux_by_style: Dict[str, Dict[str, float]] = {}
    sweep_valid_mask_by_style: Dict[str, np.ndarray] = {}
    for style_offset, style_label in enumerate(STYLE_SWEEP_ORDER, start=1):
        rollout_vecs: List[np.ndarray] = []
        rollout_aux: Dict[str, List[float]] = defaultdict(list)
        rollout_valid_masks: List[np.ndarray] = []
        for rollout_index in range(eval_args.num_rollouts_per_condition):
            style_inputs = _clone_inputs(base_inputs)
            style_condition = _build_eval_style_condition(
                axis_names=axis_names,
                calibration_record=calibration_record,
                sample=sample,
                style_label=style_label,
                model_args=model_args,
                eval_args=eval_args,
                device=style_inputs["ego_current_state"].device,
            )
            style_valid = style_condition_valid_mask(style_condition).to(style_inputs["ego_current_state"].device)
            style_inputs["style_value_condition"] = style_condition.unsqueeze(0)
            style_inputs["style_feature_valid"] = style_valid.float()
            style_inputs["style_condition_used"] = style_valid.float()
            _set_eval_seed(
                _rollout_seed(
                    eval_args.seed,
                    dataset_index=dataset_index,
                    style_offset=style_offset,
                    rollout_index=rollout_index,
                )
            )
            pred_style = _predict_ego_trajectory(model, style_inputs)
            proxy_style, aux_style = _proxy_base_from_prediction(
                sample,
                pred_style,
                dt=eval_args.dt,
                traj_smooth_window=eval_args.traj_smooth_window,
                kinematics_stride=eval_args.kinematics_stride,
                predicted_neighbor_num=model_args.predicted_neighbor_num,
            )
            rollout_vecs.append(_style_proxy_vec(proxy_style))
            rollout_valid_masks.append(_axis_valid_mask(scene_bucket, proxy_style))
            for key, value in aux_style.items():
                rollout_aux[key].append(float(value))
        sweep_pred_by_style[style_label] = np.nan_to_num(
            np.mean(np.stack(rollout_vecs, axis=0), axis=0).astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        sweep_valid_mask_by_style[style_label] = np.all(
            np.stack(rollout_valid_masks, axis=0),
            axis=0,
        ).astype(bool)
        sweep_aux_by_style[style_label] = {
            key: _mean_list(value_list) for key, value_list in rollout_aux.items()
        }

    axis_metrics: Dict[str, Dict[str, float]] = {}
    monotonic_flags: Dict[str, bool] = {}
    axis_valid_flags: Dict[str, bool] = {}
    for axis_index, axis_name in enumerate(axis_names):
        reference_order = list(
            calibration_record["sweep"]["reference_order_by_axis"].get(axis_name, list(STYLE_SWEEP_ORDER))
        )
        axis_valid = bool(
            np.all(
                [
                    bool(sweep_valid_mask_by_style[style_label][axis_index])
                    for style_label in reference_order
                ]
            )
        )
        axis_valid_flags[axis_name] = axis_valid
        pred_values = [
            float(np.nan_to_num(sweep_pred_by_style[style_label][axis_index], nan=0.0, posinf=0.0, neginf=0.0))
            for style_label in reference_order
        ]
        target_values = [
            float(np.nan_to_num(sweep_targets_by_style[style_label][axis_index], nan=0.0, posinf=0.0, neginf=0.0))
            for style_label in reference_order
        ]
        monotonic = _is_non_decreasing(pred_values, eval_args.monotonic_tol)
        monotonic_flags[axis_name] = monotonic
        axis_metrics[axis_name] = {
            "valid": float(axis_valid),
            "monotonic": float(monotonic),
            "target_mae": (
                float(np.mean(np.abs(np.asarray(pred_values) - np.asarray(target_values))))
                if axis_valid
                else None
            ),
            "target_corr": _safe_corr(pred_values, target_values) if axis_valid else None,
            "response_span": (
                float(pred_values[-1] - pred_values[0]) if axis_valid and len(pred_values) >= 2 else None
            ),
        }

    return {
        "sample_id": str(sample["sample_id"]),
        "filename": str(sample["filename"]),
        "scene_bucket": scene_bucket,
        "style_label": str(sample["style_label"]),
        "target_style_label": str(sample["target_style_label"]),
        "axis_names": axis_names,
        "observed_ade": _mean_list(observed_ades),
        "observed_fde": _mean_list(observed_fdes),
        "observed_proxy_vec": observed_vec.tolist(),
        "observed_target_vec": observed_target_vec.tolist(),
        "observed_condition_mae_vec": np.abs(observed_vec - observed_target_vec).astype(np.float32).tolist(),
        "observed_proxy_aux": observed_aux_mean,
        "observed_axis_valid_flags": {
            axis_name: bool(
                np.all([bool(mask[axis_index]) for mask in observed_valid_masks])
            )
            for axis_index, axis_name in enumerate(axis_names)
        },
        "sweep_proxy_vecs": {
            style_label: sweep_pred_by_style[style_label].tolist()
            for style_label in STYLE_SWEEP_ORDER
        },
        "sweep_target_vecs": {
            style_label: sweep_targets_by_style[style_label].tolist()
            for style_label in STYLE_SWEEP_ORDER
        },
        "sweep_proxy_aux": sweep_aux_by_style,
        "sweep_axis_valid_by_style": {
            style_label: {
                axis_name: bool(sweep_valid_mask_by_style[style_label][axis_index])
                for axis_index, axis_name in enumerate(axis_names)
            }
            for style_label in STYLE_SWEEP_ORDER
        },
        "axis_metrics": axis_metrics,
        "monotonic_flags": monotonic_flags,
        "axis_valid_flags": axis_valid_flags,
        "reference_order_by_axis": calibration_record["sweep"]["reference_order_by_axis"],
    }


def _candidate_indices(
    dataset: PreferenceConditionedPlannerData,
    calibration_map: Mapping[str, Mapping[str, Any]],
    eval_args: argparse.Namespace,
) -> List[int]:
    indices: List[int] = []
    records = dataset.records
    start = max(0, int(eval_args.start_index))
    for dataset_index in range(start, len(records), int(eval_args.sample_stride)):
        record = records[dataset_index]
        sample_id = str(record.get("sample_id", ""))
        calibration_record = calibration_map.get(sample_id)
        if calibration_record is None:
            continue
        scene_bucket = str(calibration_record.get("scene_bucket", "none"))
        if scene_bucket not in PRIMARY_SCENE_BUCKETS:
            continue
        if eval_args.scene_bucket is not None and scene_bucket != eval_args.scene_bucket:
            continue
        indices.append(dataset_index)
        if eval_args.max_samples is not None and len(indices) >= int(eval_args.max_samples):
            break
    return indices


def _summarize_results(
    per_sample_results: Sequence[Mapping[str, Any]],
    *,
    eval_args: argparse.Namespace,
    checkpoint_meta: Mapping[str, Any],
    checkpoint_path: str,
    model_args: Namespace,
    candidate_sample_count: int,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for result in per_sample_results:
        grouped[str(result["scene_bucket"])].append(result)

    scene_metrics: Dict[str, Any] = {}
    overall_axis_monotonic: List[float] = []
    overall_axis_mae: List[float] = []
    overall_axis_corr: List[float] = []
    overall_axis_span: List[float] = []
    overall_ade: List[float] = []
    overall_fde: List[float] = []

    for scene_bucket in PRIMARY_SCENE_BUCKETS:
        scene_results = grouped.get(scene_bucket, [])
        axis_names = list(style_axis_names_for_scene(scene_bucket))
        style_metric_sums = {
            style_label: np.zeros((3,), dtype=np.float64) for style_label in STYLE_SWEEP_ORDER
        }
        style_target_sums = {
            style_label: np.zeros((3,), dtype=np.float64) for style_label in STYLE_SWEEP_ORDER
        }
        observed_mae_sums = np.zeros((3,), dtype=np.float64)
        axis_monotonic = {axis_name: [] for axis_name in axis_names}
        axis_mae = {axis_name: [] for axis_name in axis_names}
        axis_corr = {axis_name: [] for axis_name in axis_names}
        axis_span = {axis_name: [] for axis_name in axis_names}
        axis_valid_count = {axis_name: 0 for axis_name in axis_names}
        observed_ades = []
        observed_fdes = []

        for result in scene_results:
            observed_ades.append(float(result["observed_ade"]))
            observed_fdes.append(float(result["observed_fde"]))
            observed_mae_sums += np.asarray(result["observed_condition_mae_vec"], dtype=np.float64)
            for style_label in STYLE_SWEEP_ORDER:
                style_metric_sums[style_label] += np.asarray(
                    result["sweep_proxy_vecs"][style_label], dtype=np.float64
                )
                style_target_sums[style_label] += np.asarray(
                    result["sweep_target_vecs"][style_label], dtype=np.float64
                )
            for axis_name in axis_names:
                metrics = result["axis_metrics"][axis_name]
                if float(metrics.get("valid", 0.0)) > 0.5:
                    axis_valid_count[axis_name] += 1
                    axis_monotonic[axis_name].append(float(metrics["monotonic"]))
                    if metrics["target_mae"] is not None:
                        axis_mae[axis_name].append(float(metrics["target_mae"]))
                    if metrics["target_corr"] is not None:
                        axis_corr[axis_name].append(float(metrics["target_corr"]))
                    if metrics["response_span"] is not None:
                        axis_span[axis_name].append(float(metrics["response_span"]))

        count = len(scene_results)
        if count > 0:
            overall_ade.extend(observed_ades)
            overall_fde.extend(observed_fdes)
            for axis_name in axis_names:
                overall_axis_monotonic.extend(axis_monotonic[axis_name])
                overall_axis_mae.extend(axis_mae[axis_name])
                overall_axis_corr.extend(axis_corr[axis_name])
                overall_axis_span.extend(axis_span[axis_name])

        scene_metrics[scene_bucket] = {
            "sample_count": count,
            "axis_names": axis_names,
            "mean_observed_ade": _mean_list(observed_ades),
            "mean_observed_fde": _mean_list(observed_fdes),
            "mean_observed_condition_mae_by_axis": {
                axis_name: float(observed_mae_sums[index] / max(count, 1))
                for index, axis_name in enumerate(axis_names)
            },
            "valid_count_by_axis": {
                axis_name: int(axis_valid_count[axis_name]) for axis_name in axis_names
            },
            "valid_rate_by_axis": {
                axis_name: float(axis_valid_count[axis_name] / max(count, 1)) for axis_name in axis_names
            },
            "monotonic_rate_by_axis": {
                axis_name: _mean_list(axis_monotonic[axis_name]) for axis_name in axis_names
            },
            "mean_target_mae_by_axis": {
                axis_name: _mean_list(axis_mae[axis_name]) for axis_name in axis_names
            },
            "mean_target_corr_by_axis": {
                axis_name: _mean_list(axis_corr[axis_name]) for axis_name in axis_names
            },
            "mean_response_span_by_axis": {
                axis_name: _mean_list(axis_span[axis_name]) for axis_name in axis_names
            },
            "mean_sweep_proxy_vecs": {
                style_label: {
                    axis_name: float(style_metric_sums[style_label][axis_index] / max(count, 1))
                    for axis_index, axis_name in enumerate(axis_names)
                }
                for style_label in STYLE_SWEEP_ORDER
            },
            "mean_sweep_target_vecs": {
                style_label: {
                    axis_name: float(style_target_sums[style_label][axis_index] / max(count, 1))
                    for axis_index, axis_name in enumerate(axis_names)
                }
                for style_label in STYLE_SWEEP_ORDER
            },
        }

    summary = {
        "experiment_dir": str(eval_args.experiment_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_meta.get("epoch", -1)),
        "checkpoint_loss": float(checkpoint_meta.get("loss", 0.0)),
        "checkpoint_weight_source": str(checkpoint_meta.get("weight_source", "model")),
        "split_root": str(eval_args.split_root),
        "cache_dir": str(eval_args.cache_dir),
        "calibration_index_path": str(eval_args.calibration_index_path),
        "condition_source": str(eval_args.condition_source),
        "condition_field": str(model_args.condition_field),
        "experiment_preset": str(getattr(model_args, "experiment_preset", "manual")),
        "style_condition_feature_set": str(model_args.style_condition_feature_set),
        "cfg_guidance_scale": float(model_args.cfg_guidance_scale),
        "scene_bucket_filter": eval_args.scene_bucket,
        "max_samples": eval_args.max_samples,
        "sample_stride": int(eval_args.sample_stride),
        "num_rollouts_per_condition": int(eval_args.num_rollouts_per_condition),
        "proxy_config": {
            "traj_smooth_window": int(eval_args.traj_smooth_window),
            "kinematics_stride": int(eval_args.kinematics_stride),
            "kinematics_peak_percentile": 90.0,
            "gap_headway_percentile": 10.0,
            "speed_drop_percentiles": [10.0, 90.0],
        },
        "candidate_sample_count": int(candidate_sample_count),
        "evaluated_sample_count": int(len(per_sample_results)),
        "overall": {
            "mean_observed_ade": _mean_list(overall_ade),
            "mean_observed_fde": _mean_list(overall_fde),
            "mean_axis_monotonic_rate": _mean_list(overall_axis_monotonic),
            "mean_axis_target_mae": _mean_list(overall_axis_mae),
            "mean_axis_target_corr": _mean_list(overall_axis_corr),
            "mean_axis_response_span": _mean_list(overall_axis_span),
        },
        "scene_metrics": scene_metrics,
    }
    return summary


def main() -> None:
    eval_args = _parse_args()
    checkpoint_path = _resolve_checkpoint_path(eval_args.experiment_dir, eval_args.checkpoint_path)
    model_args = _load_model_args(eval_args.experiment_dir, eval_args)
    _set_eval_seed(eval_args.seed)
    model, checkpoint_meta = _load_model(
        model_args,
        checkpoint_path,
        prefer_ema=bool(eval_args.prefer_ema),
    )

    dataset = PreferenceConditionedPlannerData(
        eval_args.cache_dir,
        eval_args.split_root,
        condition_field=CONDITION_SOURCE_TO_FIELD[eval_args.condition_source],
    )
    calibration_map = _load_calibration_map(eval_args.calibration_index_path)
    candidate_indices = _candidate_indices(dataset, calibration_map, eval_args)
    if not candidate_indices:
        raise RuntimeError("No candidate samples matched the requested evaluation filters.")

    detail_path = Path(eval_args.detail_jsonl_path)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    per_sample_results: List[Dict[str, Any]] = []

    with open(detail_path, "w", encoding="utf-8") as detail_file:
        for dataset_index in tqdm(candidate_indices, desc="Eval", dynamic_ncols=True):
            sample = dataset[dataset_index]
            calibration_record = calibration_map[str(sample["sample_id"])]
            result = _evaluate_sample(
                sample,
                calibration_record,
                model,
                model_args,
                eval_args,
                dataset_index=dataset_index,
            )
            per_sample_results.append(result)
            detail_file.write(json.dumps(result, ensure_ascii=False))
            detail_file.write("\n")

    summary = _summarize_results(
        per_sample_results,
        eval_args=eval_args,
        checkpoint_meta=checkpoint_meta,
        checkpoint_path=checkpoint_path,
        model_args=model_args,
        candidate_sample_count=len(candidate_indices),
    )
    write_json(eval_args.summary_json_path, summary)

    print(f"[PrefCondEval] checkpoint={checkpoint_path}")
    print(f"[PrefCondEval] candidates={len(candidate_indices)}, evaluated={len(per_sample_results)}")
    print(f"[PrefCondEval] detail_jsonl={eval_args.detail_jsonl_path}")
    print(f"[PrefCondEval] summary_json={eval_args.summary_json_path}")
    print(
        "[PrefCondEval] overall="
        f"ade={summary['overall']['mean_observed_ade']:.4f}, "
        f"fde={summary['overall']['mean_observed_fde']:.4f}, "
        f"axis_monotonic={summary['overall']['mean_axis_monotonic_rate']:.4f}, "
        f"axis_target_mae={summary['overall']['mean_axis_target_mae']:.4f}, "
        f"axis_target_corr={summary['overall']['mean_axis_target_corr']:.4f}"
    )


if __name__ == "__main__":
    main()

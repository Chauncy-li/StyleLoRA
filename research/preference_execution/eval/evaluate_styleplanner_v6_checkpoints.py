"""Model-aware open-loop validation for trained V6 StylePlanner checkpoints.

This is the missing first stage of the V6 post-training evaluator.  It loads
planner-cache validation samples, runs fixed-noise rho sweeps, exports the
generated conditional-percentile axes, and then invokes the model-agnostic V6
aggregator.  Lane-change rows keep an empty style condition and are audited
only for preservation of the pretrained planner behavior.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
for _path in (REPO_ROOT, DEVKIT_ROOT):
    _value = str(_path)
    if _path.exists() and _value not in sys.path:
        sys.path.insert(0, _value)

from baseline.model.style_planner.diffusion_planner import Diffusion_Planner
from baseline.utils.normalizer import ObservationNormalizer, StateNormalizer
from research.continuous_style.v6 import (
    build_rho_style_command,
    evaluate_v6_rho_sweep,
)
from research.preference_execution.diffusion.dataset import (
    PreferenceConditionedPlannerData,
)
from research.preference_execution.diffusion.style_condition import (
    style_condition_valid_mask,
)
from research.preference_execution.diffusion.training import (
    prepare_preference_conditioned_batch,
)
from research.preference_execution.eval.summarize_styleplanner_v6_multiseed import (
    COMPARISON_NAME as MULTISEED_COMPARISON_NAME,
    summarize_evaluation_root,
)
from research.preference_execution.eval.summarize_styleplanner_v6_stage_b_scales import (
    SCALE_AXIS_TABLE_NAME,
    SCALE_DIAGNOSTIC_NAME,
    summarize_stage_b_scale_sweep,
)


CONTROLLED_SCENES = ("straight_free_drive", "straight_car_follow")
LANE_CHANGE_SCENE = "straight_lane_change"
CHECKPOINT_EPOCH_PATTERN = re.compile(r"(?:model|checkpoint)_epoch_(\d+)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate V6 StylePlanner checkpoints with fixed-noise rho sweeps, "
            "normal-anchor retention, and empty-condition lane-change retention."
        )
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--checkpoint-epochs",
        default="10,17,20,24",
        help="Comma-separated epochs resolved under --experiment-dir.",
    )
    parser.add_argument(
        "--checkpoint-paths",
        nargs="*",
        default=None,
        help="Explicit checkpoint paths; when provided, --checkpoint-epochs is ignored.",
    )
    parser.add_argument("--base-checkpoint-path", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--condition-index-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--normalization-path",
        default="",
        help="Optional override for the train-frozen V5 normalization artifact.",
    )
    parser.add_argument(
        "--conditional-rank-model-path",
        default="",
        help="Optional override for the train-frozen V5 conditional-rank model.",
    )
    parser.add_argument("--rho-values", default="-0.8,-0.4,0,0.4,0.8")
    parser.add_argument(
        "--variants",
        default="full",
        help="Comma-separated subset of router_only,anchor_cfg,full.",
    )
    parser.add_argument(
        "--max-samples-per-controlled-scene",
        type=int,
        default=64,
        help=(
            "Eligible samples retained for each controlled scene. "
            "Use 0 to evaluate every eligible candidate."
        ),
    )
    parser.add_argument(
        "--max-lane-change-samples",
        type=int,
        default=64,
        help="Lane-change preservation samples; use 0 for every available sample.",
    )
    parser.add_argument(
        "--eligible-search-multiplier",
        type=int,
        default=4,
        help="Draw extra controlled candidates, then retain a fixed number whose rho=0 axes are measurable.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help=(
            "Paired diffusion rollouts per fixed sample/rho command. "
            "Per-seed and worst-seed summaries are written automatically."
        ),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cfg-guidance-scale", type=float, default=None)
    parser.add_argument(
        "--cfg-guidance-scales",
        default="",
        help=(
            "Optional comma-separated Normal-Anchor CFG scales evaluated in one "
            "paired run, for example 1.1,1.2. Router-only is executed once and "
            "each scale gets an isolated anchor_cfg_s<scale> output. This is "
            "mutually exclusive with --cfg-guidance-scale."
        ),
    )
    parser.add_argument("--energy-guidance-scale", type=float, default=None)
    parser.add_argument(
        "--require-stage-b-contract",
        action="store_true",
        help=(
            "Require router_only and anchor_cfg together and fail after writing "
            "diagnostics unless the semantic-normal/rho-zero/unit-scale "
            "contracts pass."
        ),
    )
    parser.add_argument("--prefer-ema", action="store_true", default=True)
    parser.add_argument("--disable-prefer-ema", action="store_true")
    parser.add_argument("--dt", type=float, default=0.1)
    return parser


def _parse_float_list(raw: str) -> list[float]:
    values = [float(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    if any(not math.isfinite(value) or value < -1.0 or value > 1.0 for value in values):
        raise ValueError("rho values must be finite and in [-1, 1]")
    if len(set(values)) != len(values):
        raise ValueError("rho values must not contain duplicates")
    return sorted(values)


def _parse_int_list(raw: str) -> list[int]:
    values = [int(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("checkpoint epochs must be positive integers")
    return values


def _parse_variants(raw: str) -> list[str]:
    allowed = {"router_only", "anchor_cfg", "full"}
    values = [value.strip() for value in str(raw).split(",") if value.strip()]
    unknown = sorted(set(values) - allowed)
    if not values or unknown:
        raise ValueError(f"variants must be selected from {sorted(allowed)}, got {unknown}")
    unique = list(dict.fromkeys(values))
    # Router-only must run first when present so Stage-B/C variants can be
    # audited against the exact same-sample, same-noise trajectory.
    return sorted(unique, key=lambda value: (value != "router_only", unique.index(value)))


def _parse_guidance_scales(raw: str) -> list[float]:
    values = [float(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not values:
        return []
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("CFG guidance scales must be finite and positive")
    if len(set(values)) != len(values):
        raise ValueError("CFG guidance scales must not contain duplicates")
    return sorted(values)


def _guidance_scale_tag(scale: float) -> str:
    value = format(float(scale), ".8g")
    return value.replace("-", "m").replace("+", "").replace(".", "p")


def _build_variant_specs(
    variants: Sequence[str],
    *,
    cfg_default: float,
    energy_default: float,
    cfg_scale_sweep: Sequence[float],
) -> list[Dict[str, Any]]:
    """Resolve canonical variants into collision-free runtime/output branches."""

    multi_scale = bool(cfg_scale_sweep)
    scales = list(cfg_scale_sweep) if multi_scale else [float(cfg_default)]
    specs: list[Dict[str, Any]] = []
    for variant in variants:
        if variant == "router_only":
            specs.append(
                {
                    "key": "router_only",
                    "variant": "router_only",
                    "cfg_scale": 1.0,
                    "energy_scale": 0.0,
                }
            )
            continue
        for scale in scales:
            key = (
                f"{variant}_s{_guidance_scale_tag(scale)}"
                if multi_scale
                else variant
            )
            specs.append(
                {
                    "key": key,
                    "variant": variant,
                    "cfg_scale": float(scale),
                    "energy_scale": (
                        0.0 if variant == "anchor_cfg" else float(energy_default)
                    ),
                }
            )
    keys = [str(spec["key"]) for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Resolved variant output names are not unique: {keys}")
    return specs


def _read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)


def _resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint_paths:
        resolved = [Path(path).expanduser() for path in args.checkpoint_paths]
    else:
        experiment_dir = Path(args.experiment_dir)
        resolved = []
        for epoch in _parse_int_list(args.checkpoint_epochs):
            candidates = sorted(experiment_dir.glob(f"model_epoch_{epoch}_trainloss_*.pth"))
            if not candidates:
                candidates = sorted(experiment_dir.glob(f"checkpoint_epoch_{epoch}.pth"))
            if not candidates:
                raise FileNotFoundError(
                    f"No model_epoch_{epoch}_trainloss_*.pth or checkpoint_epoch_{epoch}.pth "
                    f"under {experiment_dir}"
                )
            resolved.append(candidates[0])
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint files not found: {missing}")
    return resolved


def _checkpoint_tag(path: Path, payload: Mapping[str, Any]) -> str:
    match = CHECKPOINT_EPOCH_PATTERN.search(path.name)
    epoch = int(match.group(1)) if match else int(payload.get("epoch", -1))
    return f"epoch_{epoch}" if epoch >= 0 else path.stem


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    payload = _read_json(Path(args.experiment_dir) / "args.json")
    model_args = argparse.Namespace(**payload)
    model_args.device = str(args.device)
    model_args.condition_field = "style_value_condition"
    model_args.style_condition_feature_set = "global_only"
    model_args.base_style_condition_dim = 12
    model_args.style_value_dim = 12
    model_args.global_style_condition_dim = 12
    model_args.phase_style_condition_dim = 0
    model_args.phase_style_num_phases = 0
    model_args.phase_style_flat_dim = 0
    model_args.use_style_condition = True
    model_args.use_phase_style_condition = False
    model_args.use_temporal_style_gate = False
    model_args.guidance_fn = None
    if args.normalization_path:
        model_args.preference_energy_normalization_path = str(args.normalization_path)
    if args.conditional_rank_model_path:
        model_args.preference_energy_rank_model_path = str(args.conditional_rank_model_path)
    if args.cfg_guidance_scale is not None:
        model_args.cfg_guidance_scale = float(args.cfg_guidance_scale)
    if args.energy_guidance_scale is not None:
        model_args.preference_energy_guidance_scale = float(args.energy_guidance_scale)
    if str(getattr(model_args, "style_condition_encoder", "")) == "axis_router_v2_signed":
        if not getattr(model_args, "preference_energy_normalization_path", ""):
            raise ValueError(
                "Signed-router evaluation requires the frozen normalization reference"
            )
        if not getattr(model_args, "preference_energy_rank_model_path", ""):
            raise ValueError(
                "Signed-router evaluation requires the frozen conditional-rank reference"
            )
        # Stage A disables energy during training and sampling. The evaluator
        # still instantiates the frozen reference object so it can audit the
        # final trajectory in the common V6 percentile space. Variant scale 0
        # guarantees that this measurement cannot modify the trajectory.
        model_args.preference_energy_enabled = True
    model_args.state_normalizer = StateNormalizer.from_json(model_args)
    model_args.observation_normalizer = ObservationNormalizer.from_json(model_args)
    return model_args


def _checkpoint_state(
    path: str | Path,
    *,
    prefer_ema: bool,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint payload {type(payload)!r}: {path}")
    source_name = "model"
    state = payload.get("model", payload)
    if prefer_ema and payload.get("ema_state_dict") is not None:
        state = payload["ema_state_dict"]
        source_name = "ema_state_dict"
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint state is not a mapping: {path}")
    normalized = {
        str(key).replace("module.", ""): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    return normalized, {
        "epoch": int(payload.get("epoch", -1)),
        "saved_loss": float(payload.get("loss", 0.0)),
        "weight_source": source_name,
    }


def _load_model(
    model_args: argparse.Namespace,
    checkpoint_path: str | Path,
    *,
    prefer_ema: bool,
    flexible: bool,
) -> tuple[Diffusion_Planner, Dict[str, Any]]:
    model = Diffusion_Planner(copy.copy(model_args))
    state, meta = _checkpoint_state(checkpoint_path, prefer_ema=prefer_ema)
    target = model.state_dict()
    matched = {
        key: value
        for key, value in state.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    matched_numel = sum(int(target[key].numel()) for key in matched)
    total_numel = sum(int(value.numel()) for value in target.values())
    coverage = matched_numel / max(total_numel, 1)
    if flexible:
        model.load_state_dict(matched, strict=False)
        if coverage < 0.95:
            raise RuntimeError(
                f"Base checkpoint coverage too low: {coverage:.2%} for {checkpoint_path}"
            )
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected or coverage < 0.999999:
            raise RuntimeError(
                "V6 checkpoint does not exactly match args.json: "
                f"coverage={coverage:.2%}, missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
    model.eval().to(torch.device(model_args.device))
    meta.update(
        {
            "path": str(checkpoint_path),
            "coverage": coverage,
            "matched_parameter_tensors": len(matched),
        }
    )
    return model, meta


def _batch_from_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
    batch["scene_bucket"] = [str(sample["scene_bucket"])]
    return batch


def _clone_inputs(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _prediction(
    model: Diffusion_Planner,
    inputs: Mapping[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    with torch.no_grad():
        _, output = model(inputs)
    prediction = output.get("prediction")
    if prediction is None:
        raise KeyError(f"Inference output has no prediction: {sorted(output)}")
    diagnostics: Dict[str, Any] = {}
    for key, value in output.items():
        if key == "prediction" or not torch.is_tensor(value):
            continue
        detached = value.detach().cpu()
        if detached.ndim == 0:
            diagnostics[key] = bool(detached) if detached.dtype == torch.bool else float(detached)
        else:
            item = detached[0]
            if item.numel() == 1:
                diagnostics[key] = bool(item) if item.dtype == torch.bool else float(item)
            else:
                caster = bool if item.dtype == torch.bool else float
                diagnostics[key] = [caster(entry) for entry in item.reshape(-1).tolist()]
    return prediction[0].detach().cpu().numpy().astype(np.float32), diagnostics


def _trajectory_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    count = min(int(left.shape[0]), int(right.shape[0]))
    if count <= 0:
        return 0.0, 0.0
    error = np.linalg.norm(left[:count, :2] - right[:count, :2], axis=-1)
    return float(np.mean(error)), float(error[-1])


def _neighbor_trajectory_distance(
    styled: np.ndarray,
    normal: np.ndarray,
    sample: Mapping[str, Any],
) -> tuple[float, float]:
    neighbor_count = min(int(styled.shape[0]), int(normal.shape[0])) - 1
    if neighbor_count <= 0:
        return 0.0, 0.0
    valid_now = (
        sample["neighbor_agents_past_mask"][:, -1]
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
    )[:neighbor_count]
    if not np.any(valid_now):
        return 0.0, 0.0
    styled_neighbor = styled[1 : 1 + neighbor_count][valid_now]
    normal_neighbor = normal[1 : 1 + neighbor_count][valid_now]
    step_count = min(styled_neighbor.shape[1], normal_neighbor.shape[1])
    if step_count <= 0:
        return 0.0, 0.0
    error = np.linalg.norm(
        styled_neighbor[:, :step_count, :2]
        - normal_neighbor[:, :step_count, :2],
        axis=-1,
    )
    return float(np.mean(error)), float(np.mean(error[:, -1]))


def _with_normal_neighbors(
    styled: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    composed = styled.copy()
    neighbor_count = min(int(styled.shape[0]), int(normal.shape[0])) - 1
    if neighbor_count > 0:
        composed[1 : 1 + neighbor_count] = normal[1 : 1 + neighbor_count]
    return composed


def _route_speed_limit(sample: Mapping[str, Any]) -> float | None:
    values = sample["route_lanes_speed_limit"].detach().cpu().numpy().reshape(-1)
    valid = sample["route_lanes_has_speed_limit"].detach().cpu().numpy().astype(bool).reshape(-1)
    count = min(values.size, valid.size)
    selected = values[:count][valid[:count]]
    selected = selected[np.isfinite(selected) & (selected > 0.5)]
    return float(np.median(selected)) if selected.size else None


def _trajectory_metrics(
    prediction: np.ndarray,
    sample: Mapping[str, Any],
    *,
    dt: float,
) -> Dict[str, float]:
    ego = prediction[0]
    gt = sample["ego_future_gt"].detach().cpu().numpy()
    ade, fde = _trajectory_distance(ego, gt)
    speed = (
        np.linalg.norm(np.diff(ego[:, :2], axis=0), axis=-1) / max(float(dt), 1e-6)
        if ego.shape[0] >= 2
        else np.zeros((0,), dtype=np.float32)
    )
    acceleration = (
        np.diff(speed) / max(float(dt), 1e-6)
        if speed.size >= 2
        else np.zeros((0,), dtype=np.float32)
    )
    jerk = (
        np.diff(acceleration) / max(float(dt), 1e-6)
        if acceleration.size >= 2
        else np.zeros((0,), dtype=np.float32)
    )
    speed_limit = _route_speed_limit(sample)
    overspeed_rate = (
        float(np.mean(speed > 1.05 * speed_limit))
        if speed.size and speed_limit is not None
        else 0.0
    )

    collision_rate = 0.0
    min_ellipse_clearance = -1.0
    if prediction.shape[0] > 1:
        neighbor = prediction[1:]
        valid_now = (
            sample["neighbor_agents_past_mask"][:, -1]
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )[: neighbor.shape[0]]
        neighbor = neighbor[: valid_now.shape[0]][valid_now]
        if neighbor.size:
            steps = min(ego.shape[0], neighbor.shape[1])
            relative = ego[None, :steps, :2] - neighbor[:, :steps, :2]
            clearance = np.sqrt(
                np.square(relative[..., 0] / 5.0)
                + np.square(relative[..., 1] / 2.2)
            )
            min_ellipse_clearance = float(np.min(clearance))
            collision_rate = float(np.mean(np.any(clearance < 1.0, axis=0)))
    return {
        "ade": ade,
        "fde": fde,
        "mean_speed_mps": float(np.mean(speed)) if speed.size else 0.0,
        "max_abs_accel_mps2": float(np.max(np.abs(acceleration))) if acceleration.size else 0.0,
        "jerk_p90_mps3": float(np.percentile(np.abs(jerk), 90.0)) if jerk.size else 0.0,
        "overspeed_rate": overspeed_rate,
        "collision_rate": collision_rate,
        "min_ellipse_clearance": min_ellipse_clearance,
    }


def _mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else None


def _maximum(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.max(array)) if array.size else None


def _select_indices(
    dataset: PreferenceConditionedPlannerData,
    *,
    per_controlled_scene: int,
    eligible_search_multiplier: int,
    lane_count: int,
    seed: int,
) -> tuple[Dict[str, list[int]], list[int]]:
    controlled: Dict[str, list[int]] = {scene: [] for scene in CONTROLLED_SCENES}
    lane: list[int] = []
    for index, record in enumerate(dataset.records):
        causal_scene = str(record.get("scene_bucket", "none"))
        offline_scene = str(record.get("offline_scene_bucket", "none"))
        if causal_scene in controlled:
            controlled[causal_scene].append(index)
        elif offline_scene == LANE_CHANGE_SCENE:
            lane.append(index)

    rng = random.Random(int(seed))

    def choose(values: list[int], limit: int) -> list[int]:
        if limit <= 0 or len(values) <= limit:
            return list(values)
        return sorted(rng.sample(values, limit))

    controlled = {
        scene: choose(
            indices,
            int(per_controlled_scene) * max(int(eligible_search_multiplier), 1),
        )
        for scene, indices in controlled.items()
    }
    lane = choose(lane, int(lane_count))
    if not any(controlled.values()):
        raise RuntimeError("No causally controlled free-drive/car-follow validation samples found")
    return controlled, lane


def _write_condition_subset(
    *,
    source_path: str | Path,
    output_path: str | Path,
    sample_ids: set[str],
) -> int:
    """Materialize only evaluated rows so every sweep aggregation stays cheap."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    remaining = set(sample_ids)
    with open(source_path, "r", encoding="utf-8") as source, open(
        target, "w", encoding="utf-8"
    ) as output:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))
            if sample_id not in remaining:
                continue
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            remaining.remove(sample_id)
            written += 1
    if written != len(sample_ids):
        raise RuntimeError(
            "Failed to recover every selected V6 condition row: "
            f"wanted={len(sample_ids)}, written={written}"
        )
    return written


def _variant_scales(
    variant: str,
    *,
    cfg_scale: float,
    energy_scale: float,
) -> tuple[float, float]:
    if variant == "router_only":
        return 1.0, 0.0
    if variant == "anchor_cfg":
        return float(cfg_scale), 0.0
    if variant == "full":
        return float(cfg_scale), float(energy_scale)
    raise ValueError(variant)


def _configure_variant_runtime(
    model: Diffusion_Planner,
    *,
    variant: str,
    energy_scale: float,
) -> None:
    """Select one inference module without mutating checkpoint parameters.

    Stage-A checkpoints deliberately save ``normal_anchor_cfg_enabled=False``.
    Stage B reuses those exact weights and enables the semantic rho=0
    reference only for the ``anchor_cfg`` branch. Merely changing the guidance
    scale is insufficient because it would otherwise use the all-zero/empty
    reference and would not be Normal-Anchor CFG.
    """

    decoder = model.decoder.decoder
    decoder._normal_anchor_cfg_enabled = variant in {"anchor_cfg", "full"}
    decoder._preference_energy_guidance_scale = float(energy_scale)


def _populate_base_cache(
    *,
    base_model: Diffusion_Planner,
    dataset: PreferenceConditionedPlannerData,
    dataset_indices: Sequence[int],
    model_args: argparse.Namespace,
    args: argparse.Namespace,
) -> Dict[tuple[int, int], tuple[np.ndarray, Dict[str, float]]]:
    cache: Dict[tuple[int, int], tuple[np.ndarray, Dict[str, float]]] = {}
    for dataset_index in tqdm(
        dataset_indices,
        desc="pretrained-base",
        dynamic_ncols=True,
    ):
        sample = dataset[dataset_index]
        batch = _batch_from_sample(sample)
        inputs, _, _, _, _ = prepare_preference_conditioned_batch(
            batch,
            model_args,
            train=False,
            aug=None,
        )
        inputs["style_value_condition"] = torch.zeros_like(
            inputs["style_value_condition"]
        )
        inputs["normal_anchor_style_value_condition"] = torch.zeros_like(
            inputs["normal_anchor_style_value_condition"]
        )
        for seed_offset in range(int(args.num_seeds)):
            rollout_seed = int(args.seed + dataset_index * 10007 + seed_offset)
            _set_seed(rollout_seed)
            prediction, _ = _prediction(base_model, inputs)
            cache[(dataset_index, rollout_seed)] = (
                prediction,
                _trajectory_metrics(prediction, sample, dt=args.dt),
            )
    return cache


def _run_checkpoint(
    *,
    checkpoint_path: Path,
    model_args: argparse.Namespace,
    dataset: PreferenceConditionedPlannerData,
    controlled_indices: Mapping[str, Sequence[int]],
    lane_indices: Sequence[int],
    rho_values: Sequence[float],
    variant_specs: Sequence[Mapping[str, Any]],
    evaluation_condition_index_path: str,
    args: argparse.Namespace,
    base_cache: Dict[tuple[int, int], tuple[np.ndarray, Dict[str, float]]],
) -> Dict[str, Any]:
    model, checkpoint_meta = _load_model(
        model_args,
        checkpoint_path,
        prefer_ema=bool(args.prefer_ema),
        flexible=False,
    )
    checkpoint_tag = _checkpoint_tag(checkpoint_path, checkpoint_meta)
    checkpoint_root = Path(args.output_root) / checkpoint_tag
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    variant_keys = [str(spec["key"]) for spec in variant_specs]
    spec_by_key = {str(spec["key"]): spec for spec in variant_specs}
    detail_paths = {
        variant_key: checkpoint_root
        / variant_key
        / "v6_generated_axis_sweep.jsonl"
        for variant_key in variant_keys
    }
    for path in detail_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    retention: Dict[str, Dict[str, list[float]]] = {
        variant_key: defaultdict(list) for variant_key in variant_keys
    }
    generated_counts = {variant_key: 0 for variant_key in variant_keys}
    eligible_counts = {scene: 0 for scene in CONTROLLED_SCENES}
    fixed_normal_neighbors = (
        str(
            getattr(
                model_args,
                "preference_axis_reference_mode",
                "self_generated",
            )
        )
        == "normal_neighbor"
    )

    with ExitStack() as stack:
        detail_files = {
            variant_key: stack.enter_context(open(path, "w", encoding="utf-8"))
            for variant_key, path in detail_paths.items()
        }
        all_controlled = [
            (scene, index)
            for scene, indices in controlled_indices.items()
            for index in indices
        ]
        for scene, dataset_index in tqdm(
            all_controlled,
            desc=f"{checkpoint_tag}:controlled",
            dynamic_ncols=True,
        ):
            requested_count = int(args.max_samples_per_controlled_scene)
            if requested_count > 0 and eligible_counts[scene] >= requested_count:
                continue
            sample = dataset[dataset_index]
            batch = _batch_from_sample(sample)
            base_inputs, _, _, _, _ = prepare_preference_conditioned_batch(
                batch,
                model_args,
                train=False,
                aug=None,
            )
            causal_mask = (
                sample["local_axis_gate_values"].detach().cpu().numpy() > 0.5
            )
            scene_gate = sample["scene_gate_values"].detach().cpu().numpy()
            normal_condition = build_rho_style_command(
                scene_bucket=scene,
                rho=0.0,
                causal_axis_mask=causal_mask,
                scene_gate_values=scene_gate,
            ).style_value_condition()

            eligibility_inputs = _clone_inputs(base_inputs)
            eligibility_condition = torch.as_tensor(
                normal_condition,
                device=eligibility_inputs["ego_current_state"].device,
                dtype=torch.float32,
            ).unsqueeze(0)
            eligibility_inputs["style_value_condition"] = eligibility_condition
            eligibility_inputs[
                "normal_anchor_style_value_condition"
            ] = eligibility_condition.clone()
            eligibility_inputs["cfg_guidance_scale"] = 1.0
            _configure_variant_runtime(
                model,
                variant="router_only",
                energy_scale=0.0,
            )
            eligibility_seed = int(args.seed + dataset_index * 10007)
            _set_seed(eligibility_seed)
            normal_prediction, eligibility_diagnostics = _prediction(
                model,
                eligibility_inputs,
            )
            runtime_valid = np.asarray(
                eligibility_diagnostics.get(
                    "preference_generated_axis_valid_mask",
                    [False, False, False],
                ),
                dtype=bool,
            )
            fixed_axis_valid = causal_mask & runtime_valid
            if not np.any(fixed_axis_valid):
                continue
            eligible_counts[scene] += 1

            for seed_offset in range(int(args.num_seeds)):
                rollout_seed = int(args.seed + dataset_index * 10007 + seed_offset)
                if seed_offset == 0:
                    rollout_normal_prediction = normal_prediction
                else:
                    _set_seed(rollout_seed)
                    rollout_normal_prediction, _ = _prediction(
                        model,
                        eligibility_inputs,
                    )
                cache_key = (dataset_index, rollout_seed)
                if cache_key not in base_cache:
                    raise KeyError(f"Missing pretrained-base cache entry {cache_key}")
                base_prediction, base_metrics = base_cache[cache_key]

                empty_inputs = _clone_inputs(base_inputs)
                empty_inputs["style_value_condition"] = torch.zeros_like(
                    empty_inputs["style_value_condition"]
                )
                empty_inputs["normal_anchor_style_value_condition"] = torch.zeros_like(
                    empty_inputs["normal_anchor_style_value_condition"]
                )
                _set_seed(rollout_seed)
                checkpoint_empty, _ = _prediction(model, empty_inputs)
                empty_metrics = _trajectory_metrics(checkpoint_empty, sample, dt=args.dt)
                empty_base_ade, empty_base_fde = _trajectory_distance(
                    checkpoint_empty[0], base_prediction[0]
                )

                router_reference_by_rho: Dict[float, np.ndarray] = {}
                for spec in variant_specs:
                    variant_key = str(spec["key"])
                    variant = str(spec["variant"])
                    cfg_scale = float(spec["cfg_scale"])
                    energy_scale = float(spec["energy_scale"])
                    _configure_variant_runtime(
                        model,
                        variant=variant,
                        energy_scale=energy_scale,
                    )
                    variant_retention = retention[variant_key]
                    variant_retention["controlled_base_ade"].append(base_metrics["ade"])
                    variant_retention["controlled_checkpoint_empty_ade"].append(empty_metrics["ade"])
                    variant_retention["controlled_empty_to_base_ade"].append(empty_base_ade)
                    variant_retention["controlled_empty_to_base_fde"].append(empty_base_fde)

                    for rho in rho_values:
                        command = build_rho_style_command(
                            scene_bucket=scene,
                            rho=float(rho),
                            causal_axis_mask=causal_mask,
                            scene_gate_values=scene_gate,
                        )
                        style_inputs = _clone_inputs(base_inputs)
                        condition = torch.as_tensor(
                            command.style_value_condition(),
                            device=style_inputs["ego_current_state"].device,
                            dtype=torch.float32,
                        ).unsqueeze(0)
                        normal = torch.as_tensor(
                            normal_condition,
                            device=condition.device,
                            dtype=condition.dtype,
                        ).unsqueeze(0)
                        style_inputs["style_value_condition"] = condition
                        style_inputs["normal_anchor_style_value_condition"] = normal
                        if fixed_normal_neighbors:
                            style_inputs["preference_neighbor_reference_future"] = (
                                torch.as_tensor(
                                    rollout_normal_prediction[1:],
                                    device=style_inputs["ego_current_state"].device,
                                    dtype=style_inputs["ego_current_state"].dtype,
                                ).unsqueeze(0)
                            )
                        if (
                            str(
                                getattr(
                                    model_args,
                                    "free_drive_accel_support_mode",
                                    "self_generated",
                                )
                            )
                            == "normal_anchor"
                        ):
                            # Keep the acceleration-opportunity support fixed
                            # across the entire same-sample rho sweep. The
                            # reference is physical because decoder inference
                            # outputs have already been inverse-normalized.
                            style_inputs["preference_ego_reference_future"] = (
                                torch.as_tensor(
                                    rollout_normal_prediction[0],
                                    device=style_inputs["ego_current_state"].device,
                                    dtype=style_inputs["ego_current_state"].dtype,
                                ).unsqueeze(0)
                            )
                        valid = style_condition_valid_mask(condition)
                        style_inputs["style_feature_valid"] = valid.float()
                        style_inputs["style_condition_used"] = valid.float()
                        style_inputs["cfg_guidance_scale"] = cfg_scale
                        _set_seed(rollout_seed)
                        prediction, diagnostics = _prediction(model, style_inputs)
                        rho_key = float(rho)
                        if variant == "router_only":
                            router_reference_by_rho[rho_key] = prediction.copy()
                            to_router_ade = 0.0
                            to_router_fde = 0.0
                            to_router_max_abs = 0.0
                        elif rho_key in router_reference_by_rho:
                            router_prediction = router_reference_by_rho[rho_key]
                            to_router_ade, to_router_fde = _trajectory_distance(
                                prediction[0],
                                router_prediction[0],
                            )
                            to_router_max_abs = float(
                                np.max(np.abs(prediction - router_prediction))
                            )
                        else:
                            to_router_ade = None
                            to_router_fde = None
                            to_router_max_abs = None
                        joint_metrics = _trajectory_metrics(
                            prediction,
                            sample,
                            dt=args.dt,
                        )
                        audit_prediction = (
                            _with_normal_neighbors(
                                prediction,
                                rollout_normal_prediction,
                            )
                            if fixed_normal_neighbors
                            else prediction
                        )
                        metrics = _trajectory_metrics(
                            audit_prediction,
                            sample,
                            dt=args.dt,
                        )
                        neighbor_drift_ade, neighbor_drift_fde = (
                            _neighbor_trajectory_distance(
                                prediction,
                                rollout_normal_prediction,
                                sample,
                            )
                        )
                        to_base_ade, to_base_fde = _trajectory_distance(
                            prediction[0], base_prediction[0]
                        )
                        to_empty_ade, to_empty_fde = _trajectory_distance(
                            prediction[0], checkpoint_empty[0]
                        )
                        axis_percentile = diagnostics.get(
                            "preference_generated_axis_percentile",
                            [0.5, 0.5, 0.5],
                        )
                        axis_valid = diagnostics.get(
                            "preference_generated_axis_valid_mask",
                            [False, False, False],
                        )
                        row = {
                            "sample_id": str(sample["sample_id"]),
                            "filename": str(sample["filename"]),
                            "checkpoint": checkpoint_tag,
                            "variant": variant_key,
                            "variant_base": variant,
                            "scene_bucket": scene,
                            "rho_requested": float(rho),
                            "seed": int(rollout_seed),
                            # rollout_seed also contains dataset_index so it
                            # cannot identify the same replica across samples.
                            # seed_index is the comparable multi-seed cohort.
                            "seed_index": int(seed_offset),
                            "seed_base": int(args.seed),
                            "generated_axis_percentile_vec": axis_percentile,
                            "generated_axis_canonical_vec": diagnostics.get(
                                "preference_generated_axis_canonical",
                                [0.0, 0.0, 0.0],
                            ),
                            # Eligibility is fixed once at the same-sample rho=0
                            # trajectory. The command cannot change which axes
                            # enter the sweep statistics.
                            "generated_axis_valid_mask": fixed_axis_valid.tolist(),
                            "generated_axis_runtime_confidence_mask": axis_valid,
                            "target_axis_percentile_vec": command.target_executed.tolist(),
                            "trajectory_metrics": metrics,
                            "joint_trajectory_metrics": joint_metrics,
                            "neighbor_to_normal_ade": neighbor_drift_ade,
                            "neighbor_to_normal_fde": neighbor_drift_fde,
                            "trajectory_to_pretrained_base_ade": to_base_ade,
                            "trajectory_to_pretrained_base_fde": to_base_fde,
                            "trajectory_to_checkpoint_empty_ade": to_empty_ade,
                            "trajectory_to_checkpoint_empty_fde": to_empty_fde,
                            "trajectory_to_router_only_ade": to_router_ade,
                            "trajectory_to_router_only_fde": to_router_fde,
                            "trajectory_to_router_only_max_abs": to_router_max_abs,
                            "checkpoint_empty_metrics": empty_metrics,
                            "pretrained_base_metrics": base_metrics,
                            "router_gate": diagnostics.get("axis_router_gate", []),
                            "router_axis_residual_l2": diagnostics.get(
                                "axis_router_axis_residual_l2", []
                            ),
                            "axis_temporal_free_drive_used": diagnostics.get(
                                "axis_temporal_free_drive_used", False
                            ),
                            "axis_temporal_diffusion_gate": diagnostics.get(
                                "axis_temporal_diffusion_gate", 1.0
                            ),
                            "axis_temporal_terminal_only_used": diagnostics.get(
                                "axis_temporal_terminal_only_used", False
                            ),
                            "axis_temporal_terminal_active": diagnostics.get(
                                "axis_temporal_terminal_active", False
                            ),
                            "axis_temporal_diffusion_time": diagnostics.get(
                                "axis_temporal_diffusion_time", 0.0
                            ),
                            "axis_temporal_coefficient_l2": diagnostics.get(
                                "axis_temporal_coefficient_l2", []
                            ),
                            "axis_temporal_acceleration_rms": diagnostics.get(
                                "axis_temporal_acceleration_rms", []
                            ),
                            "axis_temporal_acceleration_early_mean": diagnostics.get(
                                "axis_temporal_acceleration_early_mean", []
                            ),
                            "axis_temporal_acceleration_mid_mean": diagnostics.get(
                                "axis_temporal_acceleration_mid_mean", []
                            ),
                            "axis_temporal_acceleration_late_mean": diagnostics.get(
                                "axis_temporal_acceleration_late_mean", []
                            ),
                            "axis_temporal_profile_cosine": diagnostics.get(
                                "axis_temporal_profile_cosine", []
                            ),
                            "axis_temporal_total_acceleration_rms": diagnostics.get(
                                "axis_temporal_total_acceleration_rms", 0.0
                            ),
                            "accel_opportunity_anchor_used": diagnostics.get(
                                "preference_accel_opportunity_anchor_used",
                                False,
                            ),
                            "normal_anchor_cfg_used": diagnostics.get(
                                "normal_anchor_cfg_used", False
                            ),
                            "normal_anchor_cfg_requested": variant
                            in {"anchor_cfg", "full"},
                            "normal_anchor_cfg_scale": float(cfg_scale),
                            "empty_cfg_reference_used": diagnostics.get(
                                "empty_cfg_reference_used", False
                            ),
                            "preference_energy_guidance_used": diagnostics.get(
                                "preference_energy_guidance_used", False
                            ),
                            "preference_energy": diagnostics.get("preference_energy", 0.0),
                            "preference_axis_energy": diagnostics.get(
                                "preference_axis_energy", 0.0
                            ),
                            "preference_safety_energy": diagnostics.get(
                                "preference_safety_energy", 0.0
                            ),
                        }
                        detail_files[variant_key].write(
                            json.dumps(row, ensure_ascii=False) + "\n"
                        )
                        generated_counts[variant_key] += 1
                        variant_retention[f"rho_{rho:g}_ade"].append(metrics["ade"])
                        variant_retention[f"rho_{rho:g}_fde"].append(metrics["fde"])
                        variant_retention[f"rho_{rho:g}_collision_rate"].append(
                            metrics["collision_rate"]
                        )
                        variant_retention[f"rho_{rho:g}_overspeed_rate"].append(
                            metrics["overspeed_rate"]
                        )
                        variant_retention[
                            f"rho_{rho:g}_neighbor_to_normal_ade"
                        ].append(neighbor_drift_ade)
                        variant_retention[
                            f"rho_{rho:g}_neighbor_to_normal_fde"
                        ].append(neighbor_drift_fde)
                        variant_retention["normal_anchor_cfg_used"].append(
                            float(
                                bool(
                                    diagnostics.get(
                                        "normal_anchor_cfg_used",
                                        False,
                                    )
                                )
                            )
                        )
                        variant_retention["empty_cfg_reference_used"].append(
                            float(
                                bool(
                                    diagnostics.get(
                                        "empty_cfg_reference_used",
                                        False,
                                    )
                                )
                            )
                        )
                        if to_router_ade is not None:
                            variant_retention[
                                "trajectory_to_router_only_ade"
                            ].append(float(to_router_ade))
                            variant_retention[
                                "trajectory_to_router_only_fde"
                            ].append(float(to_router_fde))
                            variant_retention[
                                "trajectory_to_router_only_max_abs"
                            ].append(float(to_router_max_abs))
                        if abs(float(rho)) <= 1e-8:
                            variant_retention["normal_to_empty_ade"].append(to_empty_ade)
                            variant_retention["normal_to_empty_fde"].append(to_empty_fde)
                            if to_router_ade is not None:
                                variant_retention[
                                    "rho_zero_to_router_only_ade"
                                ].append(float(to_router_ade))
                                variant_retention[
                                    "rho_zero_to_router_only_fde"
                                ].append(float(to_router_fde))

        requested_count = int(args.max_samples_per_controlled_scene)
        missing_eligible = (
            {
                scene: requested_count - count
                for scene, count in eligible_counts.items()
                if count < requested_count
            }
            if requested_count > 0
            else {}
        )
        if missing_eligible:
            print(
                "[V6Eval] warning: controlled candidate pool did not fill "
                f"the requested eligible sample count: {missing_eligible}"
            )

        for dataset_index in tqdm(
            lane_indices,
            desc=f"{checkpoint_tag}:lane-empty",
            dynamic_ncols=True,
        ):
            sample = dataset[dataset_index]
            batch = _batch_from_sample(sample)
            base_inputs, _, _, _, _ = prepare_preference_conditioned_batch(
                batch,
                model_args,
                train=False,
                aug=None,
            )
            empty_inputs = _clone_inputs(base_inputs)
            empty_inputs["style_value_condition"] = torch.zeros_like(
                empty_inputs["style_value_condition"]
            )
            empty_inputs["normal_anchor_style_value_condition"] = torch.zeros_like(
                empty_inputs["normal_anchor_style_value_condition"]
            )
            for seed_offset in range(int(args.num_seeds)):
                rollout_seed = int(args.seed + dataset_index * 10007 + seed_offset)
                cache_key = (dataset_index, rollout_seed)
                if cache_key not in base_cache:
                    raise KeyError(f"Missing pretrained-base cache entry {cache_key}")
                base_prediction, base_metrics = base_cache[cache_key]
                _set_seed(rollout_seed)
                checkpoint_prediction, _ = _prediction(model, empty_inputs)
                checkpoint_metrics = _trajectory_metrics(
                    checkpoint_prediction, sample, dt=args.dt
                )
                drift_ade, drift_fde = _trajectory_distance(
                    checkpoint_prediction[0], base_prediction[0]
                )
                for variant_key in variant_keys:
                    lane_retention = retention[variant_key]
                    lane_retention["lane_base_ade"].append(base_metrics["ade"])
                    lane_retention["lane_checkpoint_ade"].append(
                        checkpoint_metrics["ade"]
                    )
                    lane_retention["lane_checkpoint_collision_rate"].append(
                        checkpoint_metrics["collision_rate"]
                    )
                    lane_retention["lane_empty_to_base_ade"].append(drift_ade)
                    lane_retention["lane_empty_to_base_fde"].append(drift_fde)

    variant_reports: Dict[str, Any] = {}
    for variant_key in variant_keys:
        spec = spec_by_key[variant_key]
        variant = str(spec["variant"])
        variant_root = checkpoint_root / variant_key
        sweep_report = evaluate_v6_rho_sweep(
            condition_index_path=str(evaluation_condition_index_path),
            generated_axis_path=str(detail_paths[variant_key]),
            output_dir=str(variant_root),
            amplitude=0.25,
            normal_relative=True,
        )
        retention_report = {
            key: _mean(values) for key, values in retention[variant_key].items()
        }
        for key in (
            "trajectory_to_router_only_ade",
            "trajectory_to_router_only_fde",
            "trajectory_to_router_only_max_abs",
            "rho_zero_to_router_only_ade",
            "rho_zero_to_router_only_fde",
        ):
            values = retention[variant_key].get(key, [])
            retention_report[f"{key}_max"] = _maximum(values)
        cfg_scale = float(spec["cfg_scale"])
        energy_scale = float(spec["energy_scale"])
        retention_report.update(
            {
                "controlled_sample_count": int(
                    sum(eligible_counts.values())
                ),
                "eligible_sample_counts_by_scene": dict(eligible_counts),
                "controlled_candidate_count": int(
                    sum(len(values) for values in controlled_indices.values())
                ),
                "lane_change_sample_count": int(len(lane_indices)),
                "num_seeds": int(args.num_seeds),
                "generated_rho_rows": int(generated_counts[variant_key]),
                "variant": variant_key,
                "variant_base": variant,
                "normal_anchor_cfg_requested": variant
                in {"anchor_cfg", "full"},
                "cfg_guidance_scale": float(cfg_scale),
                "energy_guidance_scale": float(energy_scale),
                "preference_axis_reference_mode": (
                    "normal_neighbor"
                    if fixed_normal_neighbors
                    else "self_generated"
                ),
                "free_drive_accel_support_mode": str(
                    getattr(
                        model_args,
                        "free_drive_accel_support_mode",
                        "self_generated",
                    )
                ),
            }
        )
        _write_json(variant_root / "v6_open_loop_retention.json", retention_report)
        variant_reports[variant_key] = {
            "rho_sweep": sweep_report,
            "retention": retention_report,
            "generated_axis_path": str(detail_paths[variant_key]),
            "variant_base": variant,
            "cfg_guidance_scale": cfg_scale,
            "energy_guidance_scale": energy_scale,
        }

    stage_b_contracts: Dict[str, Any] = {}
    router_available = "router_only" in variant_reports
    for spec in variant_specs:
        variant_key = str(spec["key"])
        if str(spec["variant"]) != "anchor_cfg" or not router_available:
            continue
        anchor_retention = variant_reports[variant_key]["retention"]
        trajectory_max_abs = anchor_retention.get(
            "trajectory_to_router_only_max_abs_max"
        )
        rho_zero_ade = anchor_retention.get(
            "rho_zero_to_router_only_ade_max"
        )
        normal_anchor_fraction = anchor_retention.get(
            "normal_anchor_cfg_used"
        )
        empty_reference_fraction = anchor_retention.get(
            "empty_cfg_reference_used"
        )
        tolerance = 1e-6
        cfg_scale = float(spec["cfg_scale"])
        unit_scale = abs(cfg_scale - 1.0) <= 1e-12
        reference_contract = bool(
            normal_anchor_fraction is not None
            and normal_anchor_fraction >= 1.0 - 1e-12
            and empty_reference_fraction is not None
            and empty_reference_fraction <= 1e-12
        )
        rho_zero_contract = bool(
            rho_zero_ade is not None
            and rho_zero_ade <= tolerance
        )
        unit_scale_contract = (
            bool(
                trajectory_max_abs is not None
                and trajectory_max_abs <= tolerance
            )
            if unit_scale
            else None
        )
        contract = {
            "artifact": "styleplanner_v6_stage_b_normal_anchor_cfg_contract",
            "checkpoint_tag": checkpoint_tag,
            "variant": variant_key,
            "cfg_guidance_scale": cfg_scale,
            "normal_anchor_cfg_used_fraction": normal_anchor_fraction,
            "empty_cfg_reference_used_fraction": empty_reference_fraction,
            "trajectory_to_router_only_max_abs": trajectory_max_abs,
            "rho_zero_to_router_only_ade_max": rho_zero_ade,
            "tolerance": tolerance,
            "semantic_normal_reference_exclusive": reference_contract,
            "rho_zero_matches_router_only": rho_zero_contract,
            "unit_scale_expected_exact": unit_scale,
            "unit_scale_matches_router_only": unit_scale_contract,
            "passes_contract": bool(
                reference_contract
                and rho_zero_contract
                and (
                    unit_scale_contract
                    if unit_scale_contract is not None
                    else True
                )
            ),
        }
        stage_b_contracts[variant_key] = contract
        _write_json(
            checkpoint_root / variant_key / "v6_stage_b_contract.json",
            contract,
        )

    stage_b_contract = (
        next(iter(stage_b_contracts.values()))
        if len(stage_b_contracts) == 1
        else None
    )
    if stage_b_contract is not None:
        # Preserve the original single-scale artifact path and report field.
        _write_json(
            checkpoint_root / "v6_stage_b_contract.json",
            stage_b_contract,
        )
    elif stage_b_contracts:
        _write_json(
            checkpoint_root / "v6_stage_b_contracts.json",
            {
                "artifact": (
                    "styleplanner_v6_stage_b_normal_anchor_cfg_contracts"
                ),
                "checkpoint_tag": checkpoint_tag,
                "passes_all_contracts": bool(
                    all(
                        contract.get("passes_contract") is True
                        for contract in stage_b_contracts.values()
                    )
                ),
                "contracts": stage_b_contracts,
            },
        )

    scale_diagnostics = summarize_stage_b_scale_sweep(
        checkpoint_root,
        variant_specs=variant_specs,
    )
    report = {
        "artifact": "styleplanner_v6_checkpoint_open_loop_validation",
        "checkpoint": checkpoint_meta,
        "checkpoint_tag": checkpoint_tag,
        "variants": variant_reports,
        "stage_b_contract": stage_b_contract,
        "stage_b_contracts": stage_b_contracts,
        "stage_b_scale_diagnostics_path": (
            str(checkpoint_root / SCALE_DIAGNOSTIC_NAME)
            if scale_diagnostics is not None
            else None
        ),
        "stage_b_scale_axis_table_path": (
            str(checkpoint_root / SCALE_AXIS_TABLE_NAME)
            if scale_diagnostics is not None
            else None
        ),
    }
    _write_json(checkpoint_root / "v6_checkpoint_validation_summary.json", report)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def main() -> None:
    args = _parser().parse_args()
    if args.disable_prefer_ema:
        args.prefer_ema = False
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if args.max_samples_per_controlled_scene < 0:
        raise ValueError("--max-samples-per-controlled-scene must be non-negative")
    if args.max_lane_change_samples < 0:
        raise ValueError("--max-lane-change-samples must be non-negative")
    if args.eligible_search_multiplier <= 0:
        raise ValueError("--eligible-search-multiplier must be positive")
    if args.cfg_guidance_scales and args.cfg_guidance_scale is not None:
        raise ValueError(
            "--cfg-guidance-scales is mutually exclusive with "
            "--cfg-guidance-scale"
        )
    rho_values = _parse_float_list(args.rho_values)
    variants = _parse_variants(args.variants)
    cfg_scale_sweep = _parse_guidance_scales(args.cfg_guidance_scales)
    if cfg_scale_sweep and any(scale <= 1.0 for scale in cfg_scale_sweep):
        raise ValueError(
            "--cfg-guidance-scales is the post-contract effect sweep and "
            "requires every scale to be greater than 1.0"
        )
    if cfg_scale_sweep and not {"anchor_cfg", "full"}.intersection(variants):
        raise ValueError(
            "--cfg-guidance-scales requires anchor_cfg or full in --variants"
        )
    if args.require_stage_b_contract and not {
        "router_only",
        "anchor_cfg",
    }.issubset(variants):
        raise ValueError(
            "--require-stage-b-contract requires "
            "--variants router_only,anchor_cfg"
        )
    checkpoints = _resolve_checkpoints(args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model_args = _model_args(args)
    cfg_default = float(getattr(model_args, "cfg_guidance_scale", 1.5))
    energy_default = float(
        getattr(model_args, "preference_energy_guidance_scale", 0.0)
    )
    variant_specs = _build_variant_specs(
        variants,
        cfg_default=cfg_default,
        energy_default=energy_default,
        cfg_scale_sweep=cfg_scale_sweep,
    )
    variant_keys = [str(spec["key"]) for spec in variant_specs]
    dataset = PreferenceConditionedPlannerData(
        cache_dir=str(args.cache_dir),
        split_root=str(args.split_root),
        condition_field="style_value_condition",
        conditioning_index_override=str(args.condition_index_path),
    )
    controlled_indices, lane_indices = _select_indices(
        dataset,
        per_controlled_scene=int(args.max_samples_per_controlled_scene),
        eligible_search_multiplier=int(args.eligible_search_multiplier),
        lane_count=int(args.max_lane_change_samples),
        seed=int(args.seed),
    )
    selected_indices = {
        index
        for indices in controlled_indices.values()
        for index in indices
    } | set(lane_indices)
    selected_sample_ids = {
        str(dataset.records[index]["sample_id"])
        for index in selected_indices
    }
    evaluation_condition_index_path = output_root / "selected_v6_conditions.jsonl"
    _write_condition_subset(
        source_path=args.condition_index_path,
        output_path=evaluation_condition_index_path,
        sample_ids=selected_sample_ids,
    )

    base_args = copy.copy(model_args)
    # Empty-condition baseline predictions do not need the expensive frozen
    # conditional-rank energy module.
    base_args.preference_energy_enabled = False
    base_model, base_meta = _load_model(
        base_args,
        args.base_checkpoint_path,
        prefer_ema=bool(args.prefer_ema),
        flexible=True,
    )
    base_cache = _populate_base_cache(
        base_model=base_model,
        dataset=dataset,
        dataset_indices=sorted(selected_indices),
        model_args=model_args,
        args=args,
    )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reports = []
    for checkpoint_path in checkpoints:
        reports.append(
            _run_checkpoint(
                checkpoint_path=checkpoint_path,
                model_args=model_args,
                dataset=dataset,
                controlled_indices=controlled_indices,
                lane_indices=lane_indices,
                rho_values=rho_values,
                variant_specs=variant_specs,
                evaluation_condition_index_path=str(
                    evaluation_condition_index_path
                ),
                args=args,
                base_cache=base_cache,
            )
        )

    if args.require_stage_b_contract:
        failures = []
        for report in reports:
            contracts = report.get("stage_b_contracts", {})
            if not (
                isinstance(contracts, Mapping)
                and contracts
                and all(
                    isinstance(contract, Mapping)
                    and contract.get("passes_contract") is True
                    for contract in contracts.values()
                )
            ):
                failures.append(report["checkpoint_tag"])
        if failures:
            raise RuntimeError(
                "Stage-B Normal-Anchor CFG contract failed for "
                f"{failures}; inspect per-variant v6_stage_b_contract.json"
            )

    checkpoint_tags = [report["checkpoint_tag"] for report in reports]
    multiseed_comparison = summarize_evaluation_root(
        output_root,
        checkpoint_tags=checkpoint_tags,
        variants=variant_keys,
    )
    comparison = {
        "artifact": "styleplanner_v6_checkpoint_comparison",
        "experiment_dir": str(args.experiment_dir),
        "base_checkpoint": base_meta,
        "condition_index_path": str(args.condition_index_path),
        "evaluation_condition_index_path": str(
            evaluation_condition_index_path
        ),
        "rho_values": rho_values,
        "requested_variants": variants,
        "executed_variants": variant_keys,
        "variant_specs": variant_specs,
        "cfg_guidance_scales": (
            list(cfg_scale_sweep) if cfg_scale_sweep else [cfg_default]
        ),
        "num_seeds": int(args.num_seeds),
        "seed_base": int(args.seed),
        "candidate_sample_counts": {
            **{scene: len(indices) for scene, indices in controlled_indices.items()},
            LANE_CHANGE_SCENE: len(lane_indices),
        },
        "requested_eligible_samples_per_controlled_scene": int(
            args.max_samples_per_controlled_scene
        ),
        "max_lane_change_samples": int(args.max_lane_change_samples),
        "eligible_search_multiplier": int(args.eligible_search_multiplier),
        "multiseed_comparison_path": str(
            output_root / MULTISEED_COMPARISON_NAME
        ),
        "multiseed_report_count": int(len(multiseed_comparison["reports"])),
        "checkpoints": reports,
    }
    comparison_path = output_root / "v6_checkpoint_comparison.json"
    _write_json(comparison_path, comparison)
    print(f"[V6Eval] comparison={comparison_path}")
    for report in reports:
        print(
            f"[V6Eval] {report['checkpoint_tag']} -> "
            f"{output_root / report['checkpoint_tag']}"
        )
        if report.get("stage_b_scale_diagnostics_path"):
            print(
                "[V6Eval] stage_b_scale_diagnostics="
                f"{report['stage_b_scale_diagnostics_path']}"
            )


if __name__ == "__main__":
    main()

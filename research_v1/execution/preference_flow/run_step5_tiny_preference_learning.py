"""Step-5 frozen-base, single-diffusion-phase Preference Flow smoke training.

This is intentionally a tiny proof, not full planner training: it uses a fixed
8--16 sample cohort, one fixed noisy diffusion phase per sample, and updates
only the Preference Flow vector field.  Future expert targets appear only in
losses; the adapter condition contains neutral x0, preference xq, q/log-SNR,
and causal router metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as functional

from baseline.model.style_planner.preference_flow import (
    CLEAN_PREDICTION_EDITOR_DISABLED,
    PreferenceFlowConfig,
    PreferenceFlowConditionEncoder,
    PreferenceFlowTrainingAdapter,
    SmoothLongitudinalTrajectoryResidualDecoder,
)
from baseline.utils.io import opendata
from research_v1.execution.preference_flow.run_step1_base_regression import (
    _build_model_args,
    _checkpoint_state,
    _existing_file,
    _load_frozen_styleplanner,
    _model_args_path,
    _write_report,
)


_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
_SCENE_NAMES = ("straight_free_drive", "straight_car_follow")
_INPUT_KEYS = (
    "ego_current_state",
    "neighbor_agents_past",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
    "static_objects",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed-cohort Step-5 Preference Flow learning proof."
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--model-args", default=None)
    parser.add_argument("--normalization-file-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefer-ema", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Use the whole fixed tiny cohort for stable overfit optimization.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--lambda-identity", type=float, default=1.0)
    parser.add_argument("--lambda-content", type=float, default=0.02)
    parser.add_argument("--lambda-smooth", type=float, default=0.02)
    parser.add_argument("--flow-checkpoint-name", default="step5_tiny_flow.pt")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="preference-flow-step5")
    parser.add_argument("--swanlab-run-name", default="tiny-overfit")
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline"),
        default="online",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _max_abs(value: torch.Tensor) -> float:
    return 0.0 if value.numel() == 0 else float(value.detach().abs().max().cpu().item())


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum().item())
    return math.sqrt(total)


def _all_finite_gradients(parameters: Iterable[torch.nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
    )


def _rho_on_grid(value: float) -> bool:
    return any(abs(float(value) - candidate) <= 1e-6 for candidate in _GRID)


def _cohort_entries(path: Path, cache_root: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_samples = payload.get("samples") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_samples, list):
        raise TypeError("--cohort-file must contain a samples list")
    if not 8 <= len(raw_samples) <= 16:
        raise ValueError("Step-5 cohort must contain 8--16 samples")
    entries: List[Dict[str, Any]] = []
    scene_counts = {scene: 0 for scene in _SCENE_NAMES}
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, Mapping):
            raise TypeError(f"cohort sample {index} is not an object")
        filename = str(raw.get("filename", "")).strip()
        scene = str(raw.get("causal_scene_bucket", ""))
        rho = float(raw.get("training_rho", float("nan")))
        features = raw.get("task_features")
        if scene not in _SCENE_NAMES or not filename or not _rho_on_grid(rho):
            raise ValueError(f"invalid Step-5 cohort sample {index}")
        if not isinstance(features, list) or len(features) != 9:
            raise ValueError(f"cohort sample {index} needs exactly 9 causal task features")
        cache_path = Path(filename)
        cache_path = cache_path.resolve() if cache_path.is_absolute() else (cache_root / cache_path).resolve()
        try:
            cache_path.relative_to(cache_root)
        except ValueError as error:
            raise ValueError(f"cohort sample {index} escapes --cache-root") from error
        if not cache_path.is_file():
            raise FileNotFoundError(f"cohort sample {index} cache missing: {cache_path}")
        entries.append(
            {
                "cache_path": cache_path,
                "filename": str(cache_path.relative_to(cache_root)),
                "scene": scene,
                "rho": rho,
                "task_features": [float(value) for value in features],
            }
        )
        scene_counts[scene] += 1
    if any(count == 0 for count in scene_counts.values()):
        raise RuntimeError("cohort must include both free-drive and car-follow samples")
    for scene in _SCENE_NAMES:
        values = [entry["rho"] for entry in entries if entry["scene"] == scene]
        if not any(value < 0.0 for value in values) or not any(value > 0.0 for value in values):
            raise RuntimeError(f"{scene} cohort rows need both negative and positive training rho")
    return entries


def _load_sample(entry: Mapping[str, Any]) -> Dict[str, Any]:
    cache = opendata(str(entry["cache_path"]))
    required = set(_INPUT_KEYS) | {"ego_agent_future", "neighbor_agents_future"}
    try:
        missing = sorted(required.difference(cache.keys()))
        if missing:
            raise KeyError(f"cache {entry['cache_path']} missing {missing}")
        def _cache_tensor(value: Any, *, boolean: bool = False) -> torch.Tensor:
            tensor = torch.as_tensor(value)
            return tensor.to(torch.bool) if boolean else tensor.to(torch.float32)

        inputs = {
            key: _cache_tensor(
                cache[key],
                boolean=key in {"lanes_has_speed_limit", "route_lanes_has_speed_limit"},
            )
            for key in _INPUT_KEYS
        }
        sample = {
            "inputs": inputs,
            "ego_future": torch.tensor(cache["ego_agent_future"], dtype=torch.float32),
            "neighbors_future": torch.tensor(cache["neighbor_agents_future"], dtype=torch.float32),
            "task_features": torch.tensor(entry["task_features"], dtype=torch.float32),
            "rho": float(entry["rho"]),
            "scene": str(entry["scene"]),
            "filename": str(entry["filename"]),
        }
    finally:
        cache.close()
    return sample


def _attach_fixed_phases(
    samples: Sequence[Dict[str, Any]],
    *,
    predicted_neighbors: int,
    future_len: int,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    total = len(samples)
    for index, sample in enumerate(samples):
        sample["diffusion_time"] = 0.15 + 0.70 * (index + 0.5) / float(total)
        sample["noise"] = torch.randn(
            (1 + int(predicted_neighbors), int(future_len), 4),
            generator=generator,
            dtype=torch.float32,
        )


def _batch(
    samples: Sequence[Mapping[str, Any]], indices: Sequence[int], device: torch.device
) -> Dict[str, Any]:
    selected = [samples[index] for index in indices]
    return {
        "inputs": {
            key: torch.stack([item["inputs"][key] for item in selected]).to(device)
            for key in _INPUT_KEYS
        },
        "ego_future": torch.stack([item["ego_future"] for item in selected]).to(device),
        "neighbors_future": torch.stack([item["neighbors_future"] for item in selected]).to(device),
        "task_features": torch.stack([item["task_features"] for item in selected]).to(device),
        "rho": torch.tensor([item["rho"] for item in selected], device=device),
        "diffusion_time": torch.tensor(
            [item["diffusion_time"] for item in selected], device=device
        ),
        "noise": torch.stack([item["noise"] for item in selected]).to(device),
        "scene": [str(item["scene"]) for item in selected],
    }


def _xycs(trajectory: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            trajectory[..., :2],
            torch.stack((trajectory[..., 2].cos(), trajectory[..., 2].sin()), dim=-1),
        ),
        dim=-1,
    )


def _prepare_batch(
    raw: Mapping[str, Any], model_args: Any
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = dict(raw["inputs"])
    ego_future = _xycs(raw["ego_future"])
    predicted_neighbors = int(model_args.predicted_neighbor_num)
    raw_neighbors_future = raw["neighbors_future"][:, :predicted_neighbors]
    neighbor_mask = torch.sum(torch.ne(raw_neighbors_future[..., :3], 0), dim=-1) == 0
    neighbors_future = _xycs(raw_neighbors_future)
    neighbors_future = neighbors_future.masked_fill(neighbor_mask[..., None], 0.0)
    inputs = model_args.observation_normalizer(inputs)

    ego_current = inputs["ego_current_state"][:, :4]
    neighbor_current = inputs["neighbor_agents_past"][:, :predicted_neighbors, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbor_current, 0), dim=-1) == 0
    joint_neighbor_mask = torch.cat((neighbor_current_mask[..., None], neighbor_mask), dim=-1)
    future = torch.cat((ego_future[:, None], neighbors_future), dim=1)
    current = torch.cat((ego_current[:, None], neighbor_current), dim=1)
    all_gt = torch.cat((current[:, :, None], model_args.state_normalizer(future)), dim=2)
    all_gt[:, 1:][joint_neighbor_mask] = 0.0
    return inputs, all_gt, ego_future, neighbors_future, neighbor_mask


def _noisy_state(
    model: torch.nn.Module, all_gt: torch.Tensor, diffusion_time: torch.Tensor, noise: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, std = model.sde.marginal_prob(all_gt[:, :, 1:, :], diffusion_time)
    xq = torch.cat((all_gt[:, :, :1, :], mean + std * noise), dim=2)
    probe = torch.ones(
        (int(all_gt.shape[0]), 1, 1, 1), device=all_gt.device, dtype=all_gt.dtype
    )
    alpha, sigma = model.sde.marginal_prob(probe, diffusion_time)
    alpha = alpha.reshape(int(all_gt.shape[0]), -1)[:, 0].abs().clamp_min(1e-6)
    sigma = sigma.reshape(int(all_gt.shape[0]), -1)[:, 0].clamp_min(1e-6)
    log_snr = torch.log(alpha) - torch.log(sigma)
    return xq, log_snr


def _neutral_clean_prediction(
    model: torch.nn.Module,
    inputs: Mapping[str, torch.Tensor],
    xq: torch.Tensor,
    diffusion_time: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        _encoding, output = model(
            {**inputs, "sampled_trajectories": xq, "diffusion_time": diffusion_time}
        )
    clean = output.get("x_start")
    if not torch.is_tensor(clean) or tuple(clean.shape) != tuple(xq.shape):
        raise RuntimeError("Step-5 requires an x_start StylePlanner clean prediction")
    return clean.detach().reshape(int(clean.shape[0]), int(clean.shape[1]), -1)


def _physical_ego_future(clean_prediction: torch.Tensor, model_args: Any) -> torch.Tensor:
    joint = clean_prediction.reshape(
        int(clean_prediction.shape[0]), int(clean_prediction.shape[1]), -1, 4
    )
    mean = model_args.state_normalizer.mean[0, 0].to(clean_prediction.device, clean_prediction.dtype)
    std = model_args.state_normalizer.std[0, 0].to(clean_prediction.device, clean_prediction.dtype)
    return joint[:, 0, 1:, :] * std + mean


def _speed_proxy(current: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    positions = torch.cat((current[:, None, :2], future[..., :2]), dim=1)
    return torch.linalg.vector_norm(positions[:, 1:] - positions[:, :-1], dim=-1).mean(dim=-1)


def _headway_tightness_proxy(
    ego_future: torch.Tensor, neighbors_future: torch.Tensor, neighbor_mask: torch.Tensor
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(
        ego_future[:, None, :, :2] - neighbors_future[..., :2], dim=-1
    )
    distance = distance.masked_fill(neighbor_mask, float("inf"))
    minimum = distance.amin(dim=(1, 2))
    if bool(torch.isinf(minimum).any().item()):
        raise RuntimeError("car-follow cohort contains a row with no valid future neighbor")
    return -minimum


def _path_residual_components(
    current: torch.Tensor,
    neutral_future: torch.Tensor,
    edited_future: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return signed longitudinal and lateral physical XY residuals."""

    baseline = torch.cat((current[:, None, :2], neutral_future[..., :2]), dim=1)
    tangent = baseline[:, 1:] - baseline[:, :-1]
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-6)
    residual = edited_future[..., :2] - neutral_future[..., :2]
    longitudinal = (residual * tangent).sum(dim=-1)
    lateral = residual[..., 0] * tangent[..., 1] - residual[..., 1] * tangent[..., 0]
    return longitudinal, lateral


def _absolute_statistics(values: torch.Tensor) -> Dict[str, float]:
    """Small JSON-safe summary of physical residual magnitudes."""

    flat = values.detach().abs().reshape(-1).float().cpu()
    if flat.numel() == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
    }


def _task_behavior_terms(
    preference_metric: torch.Tensor,
    positive_metric: torch.Tensor,
    neutral_metric: torch.Tensor,
    negative_metric: torch.Tensor,
    target_metric: torch.Tensor,
    *,
    target_scale: float,
    order_margin: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep target fitting and +/- rho ordering as separate reportable terms."""

    target_fit = 0.02 * functional.smooth_l1_loss(
        (preference_metric - target_metric) / float(target_scale),
        torch.zeros_like(preference_metric),
    )
    positive_order = torch.relu(
        float(order_margin) - (positive_metric - neutral_metric)
    ).mean()
    negative_order = torch.relu(
        float(order_margin) - (neutral_metric - negative_metric)
    ).mean()
    return target_fit, positive_order, negative_order


def _loss_terms(
    *,
    output: Any,
    positive_direction: Any,
    negative_direction: Any,
    neutral: torch.Tensor,
    identity: Any,
    all_gt: torch.Tensor,
    ego_future: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbor_mask: torch.Tensor,
    task_features: torch.Tensor,
    model_args: Any,
) -> Dict[str, torch.Tensor]:
    preference_future = _physical_ego_future(output.clean_prediction, model_args)
    positive_future = _physical_ego_future(
        positive_direction.clean_prediction, model_args
    )
    negative_future = _physical_ego_future(
        negative_direction.clean_prediction, model_args
    )
    neutral_future = _physical_ego_future(neutral, model_args)
    current = all_gt[:, 0, 0, :]
    free_mask = task_features[:, 0] > 0.5
    car_mask = task_features[:, 1] > 0.5
    target_fit_parts: List[torch.Tensor] = []
    positive_order_parts: List[torch.Tensor] = []
    negative_order_parts: List[torch.Tensor] = []
    task_behavior: Dict[str, torch.Tensor] = {}
    if bool(free_mask.any().item()):
        pref_metric = _speed_proxy(current[free_mask], preference_future[free_mask])
        positive_metric = _speed_proxy(current[free_mask], positive_future[free_mask])
        negative_metric = _speed_proxy(current[free_mask], negative_future[free_mask])
        neutral_metric = _speed_proxy(current[free_mask], neutral_future[free_mask])
        target_metric = _speed_proxy(current[free_mask], ego_future[free_mask])
        target_fit, positive_order, negative_order = _task_behavior_terms(
            pref_metric,
            positive_metric,
            neutral_metric,
            negative_metric,
            target_metric,
            target_scale=0.5,
            order_margin=0.005,
        )
        target_fit_parts.append(target_fit)
        positive_order_parts.append(positive_order)
        negative_order_parts.append(negative_order)
        task_behavior["free"] = target_fit + positive_order + negative_order
    if bool(car_mask.any().item()):
        pref_metric = _headway_tightness_proxy(
            preference_future[car_mask], neighbors_future[car_mask], neighbor_mask[car_mask]
        )
        positive_metric = _headway_tightness_proxy(
            positive_future[car_mask], neighbors_future[car_mask], neighbor_mask[car_mask]
        )
        negative_metric = _headway_tightness_proxy(
            negative_future[car_mask], neighbors_future[car_mask], neighbor_mask[car_mask]
        )
        neutral_metric = _headway_tightness_proxy(
            neutral_future[car_mask], neighbors_future[car_mask], neighbor_mask[car_mask]
        )
        target_metric = _headway_tightness_proxy(
            ego_future[car_mask], neighbors_future[car_mask], neighbor_mask[car_mask]
        )
        target_fit, positive_order, negative_order = _task_behavior_terms(
            pref_metric,
            positive_metric,
            neutral_metric,
            negative_metric,
            target_metric,
            target_scale=2.0,
            order_margin=0.02,
        )
        target_fit_parts.append(target_fit)
        positive_order_parts.append(positive_order)
        negative_order_parts.append(negative_order)
        task_behavior["car"] = target_fit + positive_order + negative_order
    if not task_behavior:
        raise RuntimeError("batch has no supported causal task")
    target_fit = torch.stack(target_fit_parts).mean()
    positive_order = torch.stack(positive_order_parts).mean()
    negative_order = torch.stack(negative_order_parts).mean()
    behavior = target_fit + positive_order + negative_order
    xy_residual = preference_future[..., :2] - neutral_future[..., :2]
    content = xy_residual.square().mean()
    smooth = (
        (xy_residual[:, 2:] - 2.0 * xy_residual[:, 1:-1] + xy_residual[:, :-2])
        .square()
        .mean()
        if int(xy_residual.shape[1]) >= 3
        else xy_residual.new_zeros(())
    )
    identity_loss = (identity.clean_prediction - neutral).square().mean()
    return {
        "behavior": behavior,
        "target_fit": target_fit,
        "order": positive_order + negative_order,
        "order_positive": positive_order,
        "order_negative": negative_order,
        "free_behavior": task_behavior.get("free", behavior.new_zeros(())),
        "car_behavior": task_behavior.get("car", behavior.new_zeros(())),
        "identity": identity_loss,
        "content": content,
        "smooth": smooth,
        "preference_future": preference_future,
        "neutral_future": neutral_future,
    }


def _forward(
    *,
    model: torch.nn.Module,
    adapter: PreferenceFlowTrainingAdapter,
    raw: Mapping[str, Any],
    model_args: Any,
    lambdas: Mapping[str, float],
) -> Dict[str, Any]:
    inputs, all_gt, ego_future, neighbors_future, neighbor_mask = _prepare_batch(raw, model_args)
    xq, log_snr = _noisy_state(model, all_gt, raw["diffusion_time"], raw["noise"])
    neutral = _neutral_clean_prediction(model, inputs, xq, raw["diffusion_time"])
    preference = adapter(
        neutral,
        xq.reshape(int(xq.shape[0]), int(xq.shape[1]), -1),
        raw["diffusion_time"],
        log_snr,
        raw["task_features"],
        raw["rho"],
    )
    positive_direction = adapter(
        neutral,
        xq.reshape(int(xq.shape[0]), int(xq.shape[1]), -1),
        raw["diffusion_time"],
        log_snr,
        raw["task_features"],
        torch.ones_like(raw["rho"]),
    )
    negative_direction = adapter(
        neutral,
        xq.reshape(int(xq.shape[0]), int(xq.shape[1]), -1),
        raw["diffusion_time"],
        log_snr,
        raw["task_features"],
        -torch.ones_like(raw["rho"]),
    )
    identity = adapter(
        neutral,
        xq.reshape(int(xq.shape[0]), int(xq.shape[1]), -1),
        raw["diffusion_time"],
        log_snr,
        raw["task_features"],
        torch.zeros_like(raw["rho"]),
    )
    losses = _loss_terms(
        output=preference,
        positive_direction=positive_direction,
        negative_direction=negative_direction,
        neutral=neutral,
        identity=identity,
        all_gt=all_gt,
        ego_future=ego_future,
        neighbors_future=neighbors_future,
        neighbor_mask=neighbor_mask,
        task_features=raw["task_features"],
        model_args=model_args,
    )
    total = (
        losses["behavior"]
        + float(lambdas["identity"]) * losses["identity"]
        + float(lambdas["content"]) * losses["content"]
        + float(lambdas["smooth"]) * losses["smooth"]
    )
    return {
        "total": total,
        "losses": losses,
        "neutral": neutral,
        "preference": preference,
        "positive_direction": positive_direction,
        "negative_direction": negative_direction,
        "identity": identity,
        "all_gt": all_gt,
        "xq": xq,
        "log_snr": log_snr,
    }


def _evaluate(
    *,
    model: torch.nn.Module,
    adapter: PreferenceFlowTrainingAdapter,
    samples: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    model_args: Any,
    lambdas: Mapping[str, float],
) -> Dict[str, float | bool]:
    adapter_was_training = adapter.training
    adapter.eval()
    loss_keys = (
        "total",
        "behavior",
        "target_fit",
        "order",
        "order_positive",
        "order_negative",
        "free_behavior",
        "car_behavior",
        "identity",
        "content",
        "smooth",
    )
    totals = {key: 0.0 for key in loss_keys}
    count = 0
    rho_zero_exact = True
    max_latent = max_residual = max_non_ego = max_current = max_lateral = 0.0
    all_outputs_finite = True
    longitudinal_residuals: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            raw = _batch(samples, list(range(start, min(start + batch_size, len(samples)))), device)
            result = _forward(
                model=model,
                adapter=adapter,
                raw=raw,
                model_args=model_args,
                lambdas=lambdas,
            )
            weight = int(raw["rho"].shape[0])
            totals["total"] += float(result["total"].item()) * weight
            for key in loss_keys[1:]:
                totals[key] += float(result["losses"][key].item()) * weight
            identity = result["identity"].clean_prediction
            neutral = result["neutral"]
            rho_zero_exact = rho_zero_exact and bool(torch.equal(identity, neutral))
            direct = result["preference"].clean_prediction - neutral
            max_latent = max(
                max_latent,
                _max_abs(result["preference"].latent_end - result["preference"].latent_start),
            )
            max_residual = max(max_residual, _max_abs(result["preference"].ego_future_residual))
            max_non_ego = max(max_non_ego, _max_abs(direct[:, 1:, :]))
            max_current = max(max_current, _max_abs(direct[:, 0, :4]))
            longitudinal, lateral = _path_residual_components(
                result["all_gt"][:, 0, 0, :],
                result["losses"]["neutral_future"],
                result["losses"]["preference_future"],
            )
            longitudinal_residuals.append(longitudinal)
            max_lateral = max(max_lateral, _max_abs(lateral))
            all_outputs_finite = all_outputs_finite and bool(
                torch.isfinite(result["preference"].clean_prediction).all().item()
            )
            count += weight
    if adapter_was_training:
        adapter.train()
    physical_longitudinal = _absolute_statistics(torch.cat(longitudinal_residuals, dim=0))
    return {
        **{key: value / float(count) for key, value in totals.items()},
        "rho_zero_exact": rho_zero_exact,
        "max_latent_displacement": max_latent,
        "max_ego_future_residual": max_residual,
        "max_non_ego_direct_residual": max_non_ego,
        "max_ego_current_direct_residual": max_current,
        "max_lateral_residual": max_lateral,
        "physical_longitudinal_residual_abs_mean": physical_longitudinal["mean"],
        "physical_longitudinal_residual_abs_p95": physical_longitudinal["p95"],
        "physical_longitudinal_residual_abs_max": physical_longitudinal["max"],
        "all_outputs_finite": all_outputs_finite,
    }


class _SwanLabMonitor:
    def __init__(
        self, args: argparse.Namespace, config: Mapping[str, Any], logdir: Path
    ) -> None:
        self.enabled = bool(args.use_swanlab)
        self._swanlab = None
        if self.enabled:
            try:
                import swanlab  # type: ignore
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "--use-swanlab was requested but swanlab is not installed; "
                    "install it in mdsn_py39 or rerun without --use-swanlab"
                ) from error
            self._swanlab = swanlab
            swanlab.init(
                project=str(args.swanlab_project),
                experiment_name=str(args.swanlab_run_name),
                config=dict(config),
                mode=str(args.swanlab_mode),
                logdir=str(logdir),
            )

    def log(self, values: Mapping[str, float], step: int) -> None:
        if self._swanlab is not None:
            try:
                self._swanlab.log(dict(values), step=int(step))
            except TypeError:  # compatibility with older SwanLab releases
                self._swanlab.log(dict(values))

    def finish(self) -> None:
        if self._swanlab is not None:
            self._swanlab.finish()


def _new_adapter(model_args: Any, device: torch.device, dtype: torch.dtype) -> PreferenceFlowTrainingAdapter:
    config = PreferenceFlowConfig(condition_dim=24)
    decoder = SmoothLongitudinalTrajectoryResidualDecoder(
        config.latent_dim,
        ego_mean=model_args.state_normalizer.mean[0, 0],
        ego_std=model_args.state_normalizer.std[0, 0],
    )
    condition_encoder = PreferenceFlowConditionEncoder(
        config.condition_dim,
        task_feature_dim=9,
        include_diffusion_features=True,
    )
    return PreferenceFlowTrainingAdapter(
        config=config,
        trajectory_decoder=decoder,
        condition_encoder=condition_encoder,
    ).to(device=device, dtype=dtype)


def _base_snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def _base_change(model: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]) -> float:
    return max(
        _max_abs(parameter.detach().cpu() - snapshot[name])
        for name, parameter in model.named_parameters()
    )


def _adapter_snapshot(adapter: PreferenceFlowTrainingAdapter) -> Dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in adapter.state_dict().items()}


def _loss_gradient_vector(
    loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]
) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )


def _task_gradient_alignment(
    *,
    model: torch.nn.Module,
    adapter: PreferenceFlowTrainingAdapter,
    samples: Sequence[Mapping[str, Any]],
    device: torch.device,
    model_args: Any,
    lambdas: Mapping[str, float],
) -> Dict[str, float | bool | None]:
    """Diagnose whether free-drive and car-follow behavior gradients conflict."""

    was_training = adapter.training
    adapter.eval()
    raw = _batch(samples, list(range(len(samples))), device)
    result = _forward(
        model=model, adapter=adapter, raw=raw, model_args=model_args, lambdas=lambdas
    )
    parameters = list(adapter.vector_field.parameters())
    free_gradient = _loss_gradient_vector(result["losses"]["free_behavior"], parameters)
    car_gradient = _loss_gradient_vector(result["losses"]["car_behavior"], parameters)
    free_norm = float(torch.linalg.vector_norm(free_gradient).detach().cpu().item())
    car_norm = float(torch.linalg.vector_norm(car_gradient).detach().cpu().item())
    cosine = None
    if free_norm > 0.0 and car_norm > 0.0:
        cosine = float(
            (torch.dot(free_gradient, car_gradient) / (free_norm * car_norm)).detach().cpu().item()
        )
    if was_training:
        adapter.train()
    return {
        "vector_field_free_grad_norm": free_norm,
        "vector_field_car_grad_norm": car_norm,
        "vector_field_free_car_cosine": cosine,
        "gradient_conflict": bool(cosine is not None and cosine < 0.0),
    }


def _smoke_task(
    *,
    scene: str,
    samples: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    adapter: PreferenceFlowTrainingAdapter,
    device: torch.device,
    model_args: Any,
) -> Dict[str, Any]:
    indices = [index for index, sample in enumerate(samples) if sample["scene"] == scene]
    if not indices:
        raise RuntimeError(f"Step-5 smoke has no {scene} samples")
    task_samples = [samples[index] for index in indices]
    raw = _batch(samples, indices, device)
    inputs, all_gt, _ego_future, neighbors_future, neighbor_mask = _prepare_batch(raw, model_args)
    xq, log_snr = _noisy_state(model, all_gt, raw["diffusion_time"], raw["noise"])
    neutral = _neutral_clean_prediction(model, inputs, xq, raw["diffusion_time"])
    outputs = {}
    with torch.no_grad():
        for rho in (-1.0, 0.0, 1.0):
            outputs[rho] = adapter(
                neutral,
                xq.reshape(int(xq.shape[0]), int(xq.shape[1]), -1),
                raw["diffusion_time"],
                log_snr,
                raw["task_features"],
                torch.full_like(raw["rho"], rho),
            )
    current = all_gt[:, 0, 0, :]
    future = {rho: _physical_ego_future(output.clean_prediction, model_args) for rho, output in outputs.items()}
    if scene == "straight_free_drive":
        values = {rho: _speed_proxy(current, item) for rho, item in future.items()}
        metric_name = "mean_step_speed_proxy"
    else:
        values = {
            rho: _headway_tightness_proxy(item, neighbors_future, neighbor_mask)
            for rho, item in future.items()
        }
        metric_name = "headway_tightness_proxy"
    lateral_values: List[torch.Tensor] = []
    longitudinal_values: List[torch.Tensor] = []
    max_non_ego = max_current = 0.0
    for rho in (-1.0, 1.0):
        longitudinal, lateral = _path_residual_components(current, future[0.0], future[rho])
        longitudinal_values.append(longitudinal)
        lateral_values.append(lateral)
        direct = outputs[rho].clean_prediction - neutral
        max_non_ego = max(max_non_ego, _max_abs(direct[:, 1:, :]))
        max_current = max(max_current, _max_abs(direct[:, 0, :4]))
    rows: List[Dict[str, Any]] = []
    for index, sample in enumerate(task_samples):
        rho_minus = float(values[-1.0][index].item())
        rho_zero = float(values[0.0][index].item())
        rho_plus = float(values[1.0][index].item())
        minus_margin = rho_zero - rho_minus
        plus_margin = rho_plus - rho_zero
        rows.append(
            {
                "filename": str(sample["filename"]),
                "rho_minus_one": rho_minus,
                "rho_zero": rho_zero,
                "rho_plus_one": rho_plus,
                "minus_margin": minus_margin,
                "plus_margin": plus_margin,
                "direction_passed": bool(minus_margin > 0.0 and plus_margin > 0.0),
                "rho_zero_exact": bool(
                    torch.equal(
                        outputs[0.0].clean_prediction[index : index + 1],
                        neutral[index : index + 1],
                    )
                ),
            }
        )
    longitudinal = _absolute_statistics(torch.cat(longitudinal_values, dim=0))
    average_minus = sum(row["rho_minus_one"] for row in rows) / len(rows)
    average_zero = sum(row["rho_zero"] for row in rows) / len(rows)
    average_plus = sum(row["rho_plus_one"] for row in rows) / len(rows)
    return {
        "scene": scene,
        "metric": metric_name,
        "scene_count": len(rows),
        "passed_count": sum(bool(row["direction_passed"]) for row in rows),
        "pass_rate": sum(bool(row["direction_passed"]) for row in rows) / len(rows),
        "average_rho_minus_one": average_minus,
        "average_rho_zero": average_zero,
        "average_rho_plus_one": average_plus,
        "average_direction_passed": bool(
            average_minus < average_zero and average_zero < average_plus
        ),
        "minimum_minus_margin": min(float(row["minus_margin"]) for row in rows),
        "minimum_plus_margin": min(float(row["plus_margin"]) for row in rows),
        "mean_minus_margin": sum(float(row["minus_margin"]) for row in rows) / len(rows),
        "mean_plus_margin": sum(float(row["plus_margin"]) for row in rows) / len(rows),
        "rho_zero_exact": all(bool(row["rho_zero_exact"]) for row in rows),
        "max_lateral_residual": _max_abs(torch.cat(lateral_values, dim=0)),
        "max_non_ego_direct_residual": max_non_ego,
        "max_ego_current_direct_residual": max_current,
        "physical_longitudinal_residual_abs_mean": longitudinal["mean"],
        "physical_longitudinal_residual_abs_p95": longitudinal["p95"],
        "physical_longitudinal_residual_abs_max": longitudinal["max"],
        "all_outputs_finite": all(
            bool(torch.isfinite(output.clean_prediction).all().item())
            for output in outputs.values()
        ),
        "scenes": rows,
    }


def run(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    if int(args.steps) <= 0 or int(args.batch_size) <= 0 or int(args.log_interval) <= 0:
        raise ValueError("--steps, --batch-size, and --log-interval must be positive")
    if float(args.learning_rate) <= 0.0:
        raise ValueError("--learning-rate must be positive")
    _seed_everything(int(args.seed))
    checkpoint_path = _existing_file(args.base_checkpoint, "--base-checkpoint")
    cache_root = Path(args.cache_root).expanduser().resolve()
    cohort_path = _existing_file(args.cohort_file, "--cohort-file")
    if not cache_root.is_dir():
        raise NotADirectoryError(f"--cache-root is not a directory: {cache_root}")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    args_path = _model_args_path(checkpoint_path, args.model_args)
    model_args = _build_model_args(
        args_path,
        mode=CLEAN_PREDICTION_EDITOR_DISABLED,
        device=str(device),
        normalization_file_override=args.normalization_file_path,
    )
    if str(getattr(model_args, "diffusion_model_type", "")) != "x_start":
        raise RuntimeError("Step-5 requires an x_start base StylePlanner checkpoint")
    state, checkpoint_meta = _checkpoint_state(checkpoint_path, prefer_ema=bool(args.prefer_ema))
    model, model_meta = _load_frozen_styleplanner(model_args, state, checkpoint_meta)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    frozen_snapshot = _base_snapshot(model)

    entries = _cohort_entries(cohort_path, cache_root)
    samples = [_load_sample(entry) for entry in entries]
    if int(args.batch_size) != len(samples):
        raise ValueError(
            "Step-5 tiny overfit uses one deterministic full-cohort batch; "
            f"set --batch-size {len(samples)}"
        )
    _attach_fixed_phases(
        samples,
        predicted_neighbors=int(model_args.predicted_neighbor_num),
        future_len=int(model_args.future_len),
        seed=int(args.seed),
    )
    base_dtype = next(model.parameters()).dtype
    adapter = _new_adapter(model_args, device, base_dtype)
    flow_parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not flow_parameters:
        raise RuntimeError("Preference Flow has no trainable parameters")
    optimizer = torch.optim.Adam(flow_parameters, lr=float(args.learning_rate))
    base_parameter_ids = {id(parameter) for parameter in model.parameters()}
    if any(id(parameter) in base_parameter_ids for group in optimizer.param_groups for parameter in group["params"]):
        raise RuntimeError("frozen base planner leaked into the Flow optimizer")
    lambdas = {
        "identity": float(args.lambda_identity),
        "content": float(args.lambda_content),
        "smooth": float(args.lambda_smooth),
    }
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    gradient_path = output_dir / "step5_gradient_bootstrap.json"
    tiny_path = output_dir / "step5_tiny_overfit.json"
    smoke_path = output_dir / "step5_rho_direction_smoke.json"
    checkpoint_output = output_dir / str(args.flow_checkpoint_name)
    if checkpoint_output.exists() and not bool(args.overwrite):
        raise FileExistsError(f"refusing to overwrite Flow checkpoint: {checkpoint_output}")
    monitor = _SwanLabMonitor(
        args,
        {
            "seed": int(args.seed),
            "cohort_size": len(samples),
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "rho_grid": list(_GRID),
            "base_checkpoint": str(checkpoint_path),
        },
        output_dir / "swanlog",
    )
    try:
        initial = _evaluate(
            model=model,
            adapter=adapter,
            samples=samples,
            device=device,
            batch_size=int(args.batch_size),
            model_args=model_args,
            lambdas=lambdas,
        )
        bootstrap_raw = _batch(samples, list(range(min(int(args.batch_size), len(samples)))), device)
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        bootstrap = _forward(
            model=model, adapter=adapter, raw=bootstrap_raw, model_args=model_args, lambdas=lambdas
        )
        bootstrap["total"].backward()
        vector_grad = _grad_norm(adapter.vector_field.parameters())
        decoder_parameters = list(adapter.trajectory_decoder.parameters())
        encoder_parameters = list(adapter.condition_encoder.parameters())
        gradient_report = {
            "schema_version": "preference_flow_step5_gradient_bootstrap_v1",
            "base_grad_norm": _grad_norm(model.parameters()),
            "vector_field_grad_norm": vector_grad,
            "trajectory_decoder_grad_norm": _grad_norm(decoder_parameters),
            "trajectory_decoder_trainable": bool(decoder_parameters),
            "condition_encoder_grad_norm": _grad_norm(encoder_parameters),
            "condition_encoder_trainable": bool(encoder_parameters),
            "all_trainable_gradients_finite": _all_finite_gradients(flow_parameters),
            "rho_nonzero_present": bool(torch.any(bootstrap_raw["rho"] != 0.0).item()),
            "double_zero_fixed_by": "fixed_nonzero_longitudinal_decoder_jacobian",
            "passed": bool(
                _grad_norm(model.parameters()) == 0.0
                and vector_grad > 0.0
                and _all_finite_gradients(flow_parameters)
            ),
        }
        _write_report(gradient_path, gradient_report, overwrite=bool(args.overwrite))
        optimizer.zero_grad(set_to_none=True)
        if not gradient_report["passed"]:
            raise AssertionError(f"Step-5 gradient bootstrap failed: {gradient_path}")

        full_raw = _batch(samples, list(range(len(samples))), device)
        history: List[Dict[str, float]] = []
        best_state = _adapter_snapshot(adapter)
        best_total = float(initial["total"])
        best_step = 0
        for step in range(1, int(args.steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            result = _forward(
                model=model, adapter=adapter, raw=full_raw, model_args=model_args, lambdas=lambdas
            )
            current_total = float(result["total"].detach().item())
            if current_total < best_total:
                best_total = current_total
                best_step = step - 1
                best_state = _adapter_snapshot(adapter)
            result["total"].backward()
            torch.nn.utils.clip_grad_norm_(flow_parameters, max_norm=10.0)
            optimizer.step()
            point = {
                "step": float(step),
                "total": float(result["total"].detach().item()),
                "behavior": float(result["losses"]["behavior"].detach().item()),
                "target_fit": float(result["losses"]["target_fit"].detach().item()),
                "order": float(result["losses"]["order"].detach().item()),
                "order_positive": float(result["losses"]["order_positive"].detach().item()),
                "order_negative": float(result["losses"]["order_negative"].detach().item()),
                "free_behavior": float(result["losses"]["free_behavior"].detach().item()),
                "car_behavior": float(result["losses"]["car_behavior"].detach().item()),
                "identity": float(result["losses"]["identity"].detach().item()),
                "content": float(result["losses"]["content"].detach().item()),
                "smooth": float(result["losses"]["smooth"].detach().item()),
                "vector_field_grad_norm": _grad_norm(adapter.vector_field.parameters()),
            }
            history.append(point)
            if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
                monitor.log({f"train/{key}": value for key, value in point.items()}, step)

        terminal = _evaluate(
            model=model,
            adapter=adapter,
            samples=samples,
            device=device,
            batch_size=int(args.batch_size),
            model_args=model_args,
            lambdas=lambdas,
        )
        if float(terminal["total"]) < best_total:
            best_state = _adapter_snapshot(adapter)
            best_total = float(terminal["total"])
            best_step = int(args.steps)
        adapter.load_state_dict(best_state, strict=True)
        final = _evaluate(
            model=model,
            adapter=adapter,
            samples=samples,
            device=device,
            batch_size=int(args.batch_size),
            model_args=model_args,
            lambdas=lambdas,
        )
        torch.save(
            {
                "schema_version": "preference_flow_step5_tiny_checkpoint_v1",
                "flow_config": asdict(adapter.config),
                "state_dict": adapter.state_dict(),
                "base_checkpoint": str(checkpoint_path),
            },
            checkpoint_output,
        )
        reloaded = _new_adapter(model_args, device, base_dtype).eval()
        checkpoint_payload = torch.load(checkpoint_output, map_location=device)
        reloaded.load_state_dict(checkpoint_payload["state_dict"], strict=True)
        reload_raw = full_raw
        with torch.no_grad():
            original = _forward(
                model=model, adapter=adapter, raw=reload_raw, model_args=model_args, lambdas=lambdas
            )
            restored = _forward(
                model=model, adapter=reloaded, raw=reload_raw, model_args=model_args, lambdas=lambdas
            )
        reload_error = _max_abs(
            original["preference"].clean_prediction - restored["preference"].clean_prediction
        )
        base_change = _base_change(model, frozen_snapshot)
        behavior_reduction = 1.0 - float(final["behavior"]) / max(float(initial["behavior"]), 1e-12)
        free_smoke = _smoke_task(
            scene="straight_free_drive", samples=samples, model=model, adapter=adapter,
            device=device, model_args=model_args,
        )
        car_smoke = _smoke_task(
            scene="straight_car_follow", samples=samples, model=model, adapter=adapter,
            device=device, model_args=model_args,
        )
        direction_passed_count = int(free_smoke["passed_count"]) + int(car_smoke["passed_count"])
        average_direction_passed = bool(
            free_smoke["average_direction_passed"]
            and car_smoke["average_direction_passed"]
        )
        smoke_report = {
            "schema_version": "preference_flow_step5_rho_direction_smoke_v2",
            "seed": int(args.seed),
            "device": str(device),
            "free_drive": free_smoke,
            "car_follow": car_smoke,
            "scene_count": len(samples),
            "direction_passed_count": direction_passed_count,
            "direction_pass_rate": direction_passed_count / float(len(samples)),
            "average_direction_passed": average_direction_passed,
            "rho_zero_exact": bool(free_smoke["rho_zero_exact"] and car_smoke["rho_zero_exact"]),
            "max_lateral_residual": max(
                float(free_smoke["max_lateral_residual"]), float(car_smoke["max_lateral_residual"])
            ),
            "max_non_ego_direct_residual": max(
                float(free_smoke["max_non_ego_direct_residual"]), float(car_smoke["max_non_ego_direct_residual"])
            ),
            "max_ego_current_direct_residual": max(
                float(free_smoke["max_ego_current_direct_residual"]),
                float(car_smoke["max_ego_current_direct_residual"]),
            ),
            "max_physical_longitudinal_residual_abs": max(
                float(free_smoke["physical_longitudinal_residual_abs_max"]),
                float(car_smoke["physical_longitudinal_residual_abs_max"]),
            ),
            "all_outputs_finite": bool(
                free_smoke["all_outputs_finite"] and car_smoke["all_outputs_finite"]
            ),
            "passed": bool(
                direction_passed_count == len(samples)
                and average_direction_passed
                and free_smoke["rho_zero_exact"]
                and car_smoke["rho_zero_exact"]
                # Longitudinality is structural; this tolerance only absorbs
                # float32 tangent/cross-product roundoff in the report check.
                and float(free_smoke["max_lateral_residual"]) <= 1e-5
                and float(car_smoke["max_lateral_residual"]) <= 1e-5
                and float(free_smoke["max_non_ego_direct_residual"]) == 0.0
                and float(car_smoke["max_non_ego_direct_residual"]) == 0.0
                and float(free_smoke["max_ego_current_direct_residual"]) == 0.0
                and float(car_smoke["max_ego_current_direct_residual"]) == 0.0
                and bool(free_smoke["all_outputs_finite"])
                and bool(car_smoke["all_outputs_finite"])
            ),
        }
        gradient_alignment = _task_gradient_alignment(
            model=model,
            adapter=adapter,
            samples=samples,
            device=device,
            model_args=model_args,
            lambdas=lambdas,
        )
        optimization_passed = bool(
            float(final["total"]) < float(initial["total"])
            and behavior_reduction >= 0.5
            and bool(final["rho_zero_exact"])
            and bool(final["all_outputs_finite"])
            and float(final["max_latent_displacement"]) > 0.0
            and float(final["max_ego_future_residual"]) > 0.0
            and float(final["max_non_ego_direct_residual"]) == 0.0
            and float(final["max_ego_current_direct_residual"]) == 0.0
            and float(final["max_lateral_residual"]) <= 1e-5
            and base_change == 0.0
            and reload_error == 0.0
        )
        tiny_report = {
            "schema_version": "preference_flow_step5_tiny_overfit_v2",
            "base_checkpoint": str(checkpoint_path),
            "model_args": str(args_path),
            "cache_root": str(cache_root),
            "cohort_file": str(cohort_path),
            "flow_checkpoint": str(checkpoint_output),
            "rho_direction_smoke": str(smoke_path),
            "seed": int(args.seed),
            "device": str(device),
            "weight_source": str(model_meta["weight_source"]),
            "checkpoint_coverage": float(model_meta["coverage"]),
            "cohort_size": len(samples),
            "scene_counts": {scene: sum(item["scene"] == scene for item in samples) for scene in _SCENE_NAMES},
            "rho_grid": list(_GRID),
            "steps": int(args.steps),
            "full_cohort_batch": True,
            "flow_initialization": "fresh_zero_initialized",
            "selected_checkpoint_step": best_step,
            "selected_checkpoint_total": best_total,
            "terminal_before_selection": terminal,
            "optimizer_parameter_count": sum(parameter.numel() for parameter in flow_parameters),
            "optimizer_contains_base_parameter": False,
            "base_requires_grad_false": all(not parameter.requires_grad for parameter in model.parameters()),
            "frozen_base_max_abs_change": base_change,
            "initial": initial,
            "final": final,
            "behavior_loss_relative_reduction": behavior_reduction,
            "checkpoint_reload_max_abs_error": reload_error,
            "free_car_vector_field_gradient_alignment": gradient_alignment,
            "optimization_passed": optimization_passed,
            "history": history,
            "passed": bool(gradient_report["passed"] and optimization_passed and smoke_report["passed"]),
        }
        _write_report(tiny_path, tiny_report, overwrite=bool(args.overwrite))
        _write_report(smoke_path, smoke_report, overwrite=bool(args.overwrite))
        monitor.log(
            {
                "final/total": float(final["total"]),
                "final/behavior": float(final["behavior"]),
                "final/target_fit": float(final["target_fit"]),
                "final/order": float(final["order"]),
                "final/behavior_reduction": behavior_reduction,
                "final/direction_passed_count": float(direction_passed_count),
                "final/max_physical_longitudinal_residual": float(
                    smoke_report["max_physical_longitudinal_residual_abs"]
                ),
                "final/smoke_passed": float(smoke_report["passed"]),
            },
            int(args.steps),
        )
        if not tiny_report["passed"] or not smoke_report["passed"]:
            raise AssertionError(
                "Step-5 tiny proof failed; inspect " f"{tiny_path} and {smoke_path}"
            )
    finally:
        monitor.finish()
    return gradient_path, tiny_path, smoke_path, checkpoint_output


def main() -> None:
    gradient_path, tiny_path, smoke_path, checkpoint_path = run(_parser().parse_args())
    print(f"Step-5 gradient bootstrap passed: {gradient_path}")
    print(f"Step-5 tiny overfit passed: {tiny_path}")
    print(f"Step-5 rho direction smoke passed: {smoke_path}")
    print(f"Step-5 Flow checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

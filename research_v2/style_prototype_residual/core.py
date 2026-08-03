"""Frozen-base residual extraction, compact models, and single-stream editor.

The only planner integration is a runtime assignment to the existing verified
clean-prediction callback.  No file under ``baseline/`` is modified.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from baseline.model.style_planner.preference_flow import (
    CLEAN_PREDICTION_EDITOR_DISABLED,
    CleanPredictionEditContext,
)
from baseline.utils.io import opendata
from research_v1.execution.preference_flow.run_step1_base_regression import (
    _build_model_args,
    _cache_inputs,
    _checkpoint_state,
    _existing_file,
    _load_frozen_styleplanner,
    _model_args_path,
)
from research_v1.execution.preference_flow.run_step5_tiny_preference_learning import (
    _noisy_state,
    _prepare_batch,
)
from research_v2.style_prototype_residual.data import STYLE_NAMES, STYLE_TO_INDEX, StyleEntry


INPUT_KEYS: Tuple[str, ...] = (
    "ego_current_state", "neighbor_agents_past", "lanes", "lanes_speed_limit",
    "lanes_has_speed_limit", "route_lanes", "route_lanes_speed_limit",
    "route_lanes_has_speed_limit", "static_objects",
)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def max_abs(value: torch.Tensor) -> float:
    return 0.0 if value.numel() == 0 else float(value.detach().abs().max().cpu().item())


def grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    return math.sqrt(sum(float(parameter.grad.detach().square().sum().item()) for parameter in parameters if parameter.grad is not None))


def all_finite(parameters: Iterable[nn.Parameter]) -> bool:
    return all(
        bool(torch.isfinite(parameter).all().item())
        and (parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()))
        for parameter in parameters
    )


def frozen_snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def frozen_change(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> float:
    values = [max_abs(parameter.detach().cpu() - snapshot[name]) for name, parameter in model.named_parameters()]
    return max(values, default=0.0)


def load_frozen_base(
    *, checkpoint: str, model_args: Optional[str], normalization_file_path: Optional[str], device: str, prefer_ema: bool
) -> Tuple[nn.Module, SimpleNamespace, Dict[str, Any]]:
    checkpoint_path = _existing_file(checkpoint, "--base-checkpoint")
    args = _build_model_args(
        _model_args_path(checkpoint_path, model_args),
        mode=CLEAN_PREDICTION_EDITOR_DISABLED,
        device=device,
        normalization_file_override=normalization_file_path,
    )
    state, metadata = _checkpoint_state(checkpoint_path, prefer_ema=prefer_ema)
    model, metadata = _load_frozen_styleplanner(args, state, metadata)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, args, metadata


def _cache_tensor(value: Any, *, boolean: bool = False) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    return tensor.to(torch.bool) if boolean else tensor.to(torch.float32)


def load_batch(entries: Sequence[StyleEntry], model_args: SimpleNamespace, device: torch.device) -> Dict[str, Any]:
    """Load raw cache data; official observation normalization occurs later once."""
    if not entries:
        raise ValueError("cannot load an empty batch")
    samples: List[Dict[str, Any]] = []
    agent_num = int(model_args.agent_num)
    required = set(INPUT_KEYS) | {"ego_agent_future", "neighbor_agents_future"}
    for entry in entries:
        cache = opendata(entry.cache_path)
        try:
            missing = sorted(required.difference(cache.keys()))
            if missing:
                raise KeyError(f"cache {entry.cache_path} missing {missing}")
            inputs = {
                key: _cache_tensor(
                    cache[key][:agent_num] if key == "neighbor_agents_past" else cache[key],
                    boolean=key in {"lanes_has_speed_limit", "route_lanes_has_speed_limit"},
                )
                for key in INPUT_KEYS
            }
            samples.append({
                "inputs": inputs,
                "ego_future": _cache_tensor(cache["ego_agent_future"]),
                "neighbors_future": _cache_tensor(cache["neighbor_agents_future"]),
                "entry": entry,
            })
        finally:
            cache.close()
    return {
        "inputs": {key: torch.stack([sample["inputs"][key] for sample in samples]).to(device) for key in INPUT_KEYS},
        "ego_future": torch.stack([sample["ego_future"] for sample in samples]).to(device),
        "neighbors_future": torch.stack([sample["neighbors_future"] for sample in samples]).to(device),
        "style_index": torch.tensor([STYLE_TO_INDEX[str(sample["entry"].style)] for sample in samples], device=device),
        "scene": [sample["entry"].scene for sample in samples],
        "entry": [sample["entry"] for sample in samples],
    }


@dataclass(frozen=True)
class FixedQView:
    base_ego: torch.Tensor
    target_ego: torch.Tensor
    residual: torch.Tensor
    scene_feature: torch.Tensor
    diffusion_time: torch.Tensor
    xq: torch.Tensor


def _pooled_scene_feature(encoder_outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    encoding = encoder_outputs.get("encoding")
    if not torch.is_tensor(encoding) or encoding.ndim != 3:
        raise RuntimeError("frozen StylePlanner encoder did not return [B, token, hidden] encoding")
    return encoding.mean(dim=1)


def fixed_q_view(
    model: nn.Module,
    model_args: SimpleNamespace,
    batch: Mapping[str, Any],
    diffusion_time: torch.Tensor,
    noise: torch.Tensor,
) -> FixedQView:
    """Exact teacher-forced xq construction reused from the prior Step-5 path."""
    raw = {"inputs": batch["inputs"], "ego_future": batch["ego_future"], "neighbors_future": batch["neighbors_future"]}
    normalized_inputs, all_gt, _ego, _neighbors, _mask = _prepare_batch(raw, model_args)
    xq, _log_snr = _noisy_state(model, all_gt, diffusion_time, noise)
    with torch.no_grad():
        encoder_outputs, outputs = model({**normalized_inputs, "sampled_trajectories": xq, "diffusion_time": diffusion_time})
    clean = outputs.get("x_start")
    if not torch.is_tensor(clean) or tuple(clean.shape) != tuple(xq.shape):
        raise RuntimeError("base planner did not return same-shape clean x0")
    base_ego = clean[:, 0, 1:, :].detach()
    target_ego = all_gt[:, 0, 1:, :].detach()
    return FixedQView(
        base_ego=base_ego,
        target_ego=target_ego,
        residual=(target_ego - base_ego).detach(),
        scene_feature=_pooled_scene_feature(encoder_outputs).detach(),
        diffusion_time=diffusion_time.detach(), xq=xq.detach(),
    )


def random_view_parameters(
    *, batch_size: int, predicted_neighbors: int, future_len: int, seed: int, device: torch.device,
    q_min: float = 0.15, q_max: float = 0.85,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < q_min < q_max < 1.0:
        raise ValueError("q range must lie strictly inside (0, 1)")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    time = q_min + (q_max - q_min) * torch.rand((batch_size,), generator=generator)
    noise = torch.randn((batch_size, 1 + predicted_neighbors, future_len, 4), generator=generator)
    return time.to(device=device, dtype=torch.float32), noise.to(device=device, dtype=torch.float32)


@dataclass(frozen=True)
class NetworkConfig:
    future_len: int
    scene_dim: int
    latent_dim: int = 8
    hidden_dim: int = 256
    layers: int = 3

    @property
    def residual_dim(self) -> int:
        return int(self.future_len) * 4

    def to_dict(self) -> Dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


def _mlp(dimensions: Sequence[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(int(left), int(right)))
        if index + 1 < len(dimensions) - 1:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class ResidualStyleEncoder(nn.Module):
    """Small encoder E(residual, frozen scene feature, q) -> z."""
    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.network = _mlp((config.residual_dim + config.scene_dim + 1, config.hidden_dim, config.hidden_dim, config.latent_dim))

    def forward(self, residual: torch.Tensor, scene_feature: torch.Tensor, diffusion_time: torch.Tensor) -> torch.Tensor:
        batch = int(residual.shape[0])
        if tuple(residual.shape[1:]) != (self.config.future_len, 4):
            raise ValueError("residual shape disagrees with encoder future length")
        if tuple(scene_feature.shape) != (batch, self.config.scene_dim):
            raise ValueError("scene feature shape disagrees with encoder config")
        time = diffusion_time.reshape(batch, 1).to(dtype=residual.dtype)
        return self.network(torch.cat((residual.reshape(batch, -1), scene_feature.to(dtype=residual.dtype), time), dim=-1))


class RawResidualExecutor(nn.Module):
    """A_raw(h,c,q,u), intentionally deterministic: no dropout or stochastic layers."""
    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        dims = (config.residual_dim + config.scene_dim + 1 + config.latent_dim, config.hidden_dim, config.hidden_dim, config.residual_dim)
        self.network = _mlp(dims)

    def raw(self, base_ego: torch.Tensor, scene_feature: torch.Tensor, diffusion_time: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        batch = int(base_ego.shape[0])
        if tuple(base_ego.shape[1:]) != (self.config.future_len, 4):
            raise ValueError("base clean prediction shape disagrees with executor future length")
        if tuple(scene_feature.shape) != (batch, self.config.scene_dim) or tuple(control.shape) != (batch, self.config.latent_dim):
            raise ValueError("executor condition shape mismatch")
        features = torch.cat((
            base_ego.reshape(batch, -1), scene_feature.to(dtype=base_ego.dtype),
            diffusion_time.reshape(batch, 1).to(dtype=base_ego.dtype), control.to(dtype=base_ego.dtype),
        ), dim=-1)
        return self.network(features).reshape(batch, self.config.future_len, 4)

    def control(self, base_ego: torch.Tensor, scene_feature: torch.Tensor, diffusion_time: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Structural zero: A_raw(h,c,q,u) - A_raw(h,c,q,0), exactly."""
        return self.raw(base_ego, scene_feature, diffusion_time, control) - self.raw(
            base_ego, scene_feature, diffusion_time, torch.zeros_like(control)
        )


class DirectEmbeddingControl(nn.Module):
    """Fair aggr/norm/cons direct-embedding control baseline."""
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(3, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def for_labels(self, style_index: torch.Tensor) -> torch.Tensor:
        normal = self.embedding.weight[STYLE_TO_INDEX["norm"]]
        return self.embedding(style_index) - normal

    def for_rho(self, rho: torch.Tensor) -> torch.Tensor:
        normal = self.embedding.weight[STYLE_TO_INDEX["norm"]]
        aggr = self.embedding.weight[STYLE_TO_INDEX["aggr"]] - normal
        cons = self.embedding.weight[STYLE_TO_INDEX["cons"]] - normal
        value = rho.reshape(-1, 1)
        return torch.where(value >= 0.0, value * aggr, (-value) * cons)

    def prototypes(self) -> torch.Tensor:
        normal = self.embedding.weight[STYLE_TO_INDEX["norm"]]
        return self.embedding.weight - normal


class PrototypeControl(nn.Module):
    """Saved residual prototypes; deployment needs no ResidualStyleEncoder."""
    def __init__(self, prototypes: torch.Tensor) -> None:
        super().__init__()
        if tuple(prototypes.shape[:1]) != (3,):
            raise ValueError("prototypes must have shape [3, latent_dim]")
        self.register_buffer("prototype", prototypes.detach().clone())

    def for_labels(self, style_index: torch.Tensor) -> torch.Tensor:
        return self.prototype[style_index] - self.prototype[STYLE_TO_INDEX["norm"]]

    def for_rho(self, rho: torch.Tensor) -> torch.Tensor:
        normal = self.prototype[STYLE_TO_INDEX["norm"]]
        aggr = self.prototype[STYLE_TO_INDEX["aggr"]] - normal
        cons = self.prototype[STYLE_TO_INDEX["cons"]] - normal
        value = rho.reshape(-1, 1)
        return torch.where(value >= 0.0, value * aggr, (-value) * cons)


def prototype_means(encoder: ResidualStyleEncoder, views: Iterable[Tuple[FixedQView, torch.Tensor]]) -> torch.Tensor:
    totals = [None, None, None]
    counts = [0, 0, 0]
    encoder.eval()
    with torch.no_grad():
        for view, labels in views:
            codes = encoder(view.residual, view.scene_feature, view.diffusion_time)
            for index in range(3):
                selected = codes[labels == index]
                if selected.numel():
                    total = selected.sum(dim=0)
                    totals[index] = total if totals[index] is None else totals[index] + total
                    counts[index] += int(selected.shape[0])
    if any(value == 0 for value in counts):
        raise RuntimeError(f"cannot form all style prototypes; counts={counts}")
    return torch.stack([totals[index] / counts[index] for index in range(3)], dim=0)


class SingleStreamResidualEditor:
    """Runtime-only clean-x0 editor: writes ego future and nothing else."""
    def __init__(
        self, executor: RawResidualExecutor, controller: nn.Module, scene_feature: torch.Tensor,
        rho: float, agent_count: int, future_len: int,
    ) -> None:
        self.executor = executor.eval()
        self.controller = controller.eval()
        self.scene_feature = scene_feature.detach()
        self.rho = float(rho)
        self.agent_count = int(agent_count)
        self.future_len = int(future_len)
        self.records: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.records.clear()

    def __call__(self, clean_prediction: torch.Tensor, context: CleanPredictionEditContext) -> torch.Tensor:
        batch = int(clean_prediction.shape[0])
        expected = (batch, self.agent_count, self.future_len + 1, 4)
        if clean_prediction.numel() != math.prod(expected):
            raise ValueError(f"clean prediction cannot be reshaped to {expected}")
        joint = clean_prediction.reshape(expected)
        feature = self.scene_feature.to(device=clean_prediction.device, dtype=clean_prediction.dtype)
        if int(feature.shape[0]) == 1 and batch > 1:
            feature = feature.expand(batch, -1)
        if tuple(feature.shape) != (batch, self.executor.config.scene_dim):
            raise ValueError("frozen scene feature batch does not match DPM batch")
        rho = torch.full((batch,), self.rho, device=clean_prediction.device, dtype=clean_prediction.dtype)
        control = self.controller.for_rho(rho).to(dtype=clean_prediction.dtype)
        residual = self.executor.control(joint[:, 0, 1:, :], feature, context.diffusion_time, control)
        zero_control = bool(torch.count_nonzero(control).item() == 0)
        if zero_control:
            if torch.count_nonzero(residual).item() != 0:
                raise RuntimeError("centered executor violated exact u=0 identity")
            edited = clean_prediction
        else:
            edited = clean_prediction.contiguous().clone()
            edited.reshape(expected)[:, 0, 1:, :] += residual
        self.records.append({
            "evaluation_index": int(context.model_evaluation_index),
            "diffusion_time": float(context.diffusion_time.detach()[0].cpu().item()),
            "rho": self.rho, "zero_control": zero_control,
            "ego_future_residual_abs_max": max_abs(residual),
            "ego_current_residual_abs_max": 0.0, "non_ego_residual_abs_max": 0.0,
        })
        return edited


@contextlib.contextmanager
def installed_single_stream_editor(model: nn.Module, editor: Optional[SingleStreamResidualEditor]) -> Iterator[None]:
    """Use the existing Step-1 clean-x0 slot without modifying baseline source."""
    decoder = model.decoder.decoder
    original = decoder._clean_prediction_editor
    decoder._clean_prediction_editor = editor
    try:
        yield
    finally:
        decoder._clean_prediction_editor = original


def full_dpm_inputs(cache_path: str | Path, model_args: SimpleNamespace, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    raw = _cache_inputs(Path(cache_path), model_args, device)
    normalized = model_args.observation_normalizer({key: value.clone() for key, value in raw.items()})
    return raw, normalized


def frozen_scene_feature(model: nn.Module, normalized_inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        return _pooled_scene_feature(model.encoder(dict(normalized_inputs))).detach()


def full_dpm_rollout(
    model: nn.Module, normalized_inputs: Mapping[str, torch.Tensor], *, seed: int,
    editor: Optional[SingleStreamResidualEditor] = None, observer: Any = None,
) -> torch.Tensor:
    seed_everything(seed)
    inputs = {key: value.clone() for key, value in normalized_inputs.items()}
    if observer is not None:
        inputs["clean_prediction_observer"] = observer
    with installed_single_stream_editor(model, editor), torch.no_grad():
        _encoding, outputs = model(inputs)
    prediction = outputs.get("prediction")
    if not torch.is_tensor(prediction) or prediction.ndim != 4:
        raise RuntimeError("full-DPM StylePlanner rollout did not return [B,P,T,4] prediction")
    return prediction.detach()


def physical_motion_metrics(current_xy: torch.Tensor, future_xy: torch.Tensor, dt_seconds: float = 0.1) -> Dict[str, float]:
    positions = torch.cat((current_xy.reshape(1, 2), future_xy[:, :2]), dim=0)
    speed = torch.linalg.vector_norm(positions[1:] - positions[:-1], dim=-1) / dt_seconds
    acceleration = torch.diff(speed) / dt_seconds if speed.numel() > 1 else speed.new_empty((0,))
    jerk = torch.diff(acceleration) / dt_seconds if acceleration.numel() > 1 else speed.new_empty((0,))
    def average(value: torch.Tensor) -> float:
        return 0.0 if not value.numel() else float(value.mean().cpu().item())
    return {"speed_mean_mps": average(speed), "speed_max_mps": 0.0 if not speed.numel() else float(speed.max().cpu().item()),
            "acceleration_abs_mean_mps2": average(acceleration.abs()), "jerk_abs_mean_mps3": average(jerk.abs())}

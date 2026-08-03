"""CLI for the FC-first same-state residual prototype feasibility study.

Commands deliberately stay separate so FC can be reviewed before FCL.  They
all write beneath an explicit output directory on the server.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from baseline.model.style_planner.preference_flow import CleanPredictionTraceRecorder
from baseline.utils.io import opendata
from research_v2.style_prototype_residual.core import (
    DirectEmbeddingControl,
    FixedQView,
    NetworkConfig,
    PrototypeControl,
    RawResidualExecutor,
    ResidualStyleEncoder,
    SingleStreamResidualEditor,
    all_finite,
    fixed_q_view,
    frozen_change,
    frozen_scene_feature,
    frozen_snapshot,
    full_dpm_inputs,
    full_dpm_rollout,
    grad_norm,
    load_batch,
    load_frozen_base,
    max_abs,
    physical_motion_metrics,
    prototype_means,
    random_view_parameters,
    seed_everything,
    write_json,
)
from research_v2.style_prototype_residual.data import (
    FC_SCENES,
    FCL_SCENES,
    STYLE_NAMES,
    STYLE_TO_INDEX,
    StyleEntry,
    audit_entries,
    balanced_indices,
    group_histogram,
    load_entries,
    read_manifest,
    write_manifest,
)


SCHEMA = "style_prototype_residual_pilot_v1"
NORM_INDEX = STYLE_TO_INDEX["norm"]
AGGR_INDEX = STYLE_TO_INDEX["aggr"]
CONS_INDEX = STYLE_TO_INDEX["cons"]


@dataclass
class SwanMonitor:
    enabled: bool
    module: Any = None

    @classmethod
    def create(cls, args: argparse.Namespace, config: Mapping[str, Any], output_dir: Path) -> "SwanMonitor":
        if not bool(getattr(args, "use_swanlab", False)):
            return cls(False)
        try:
            import swanlab  # type: ignore
        except ModuleNotFoundError as error:
            raise RuntimeError("--use-swanlab requested but swanlab is unavailable in mdsn_py39") from error
        swanlab.init(
            project=str(args.swanlab_project), experiment_name=str(args.swanlab_run_name),
            config=dict(config), mode=str(args.swanlab_mode), logdir=str(output_dir / "swanlog"),
        )
        return cls(True, swanlab)

    def log(self, values: Mapping[str, float], step: int) -> None:
        if self.module is not None:
            try:
                self.module.log(dict(values), step=int(step))
            except TypeError:
                self.module.log(dict(values))

    def finish(self) -> None:
        if self.module is not None:
            self.module.finish()


def _scenes(mode: str) -> Tuple[str, ...]:
    if mode == "fc":
        return FC_SCENES
    if mode == "fcl":
        return FCL_SCENES
    raise ValueError(f"unsupported mode {mode!r}")


def _output_dir(args: argparse.Namespace) -> Path:
    path = Path(args.output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_paths(output_dir: Path, mode: str) -> Tuple[Path, Path]:
    folder = output_dir / "manifests"
    return folder / f"{mode}_train.jsonl", folder / f"{mode}_val.jsonl"


def _require_fc_gate(args: argparse.Namespace) -> None:
    if getattr(args, "mode", "fc") != "fcl":
        return
    summary_path = str(getattr(args, "fc_summary", "")).strip()
    if not summary_path:
        raise ValueError("FCL is blocked until FC passes; provide --fc-summary")
    with Path(summary_path).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not bool(summary.get("fc_full_dpm_supported", False)):
        raise RuntimeError("FCL is blocked because FC full-DPM is not supported")


def _trainable(entries: Sequence[StyleEntry], maximum: int = 0) -> List[StyleEntry]:
    result = [entry for entry in entries if entry.trainable]
    if maximum > 0:
        result = result[:maximum]
    if not result:
        raise RuntimeError("manifest has no trainable samples")
    return result


def _chunks(values: Sequence[StyleEntry], size: int) -> Iterator[List[StyleEntry]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _sampling_balance(entries: Sequence[StyleEntry], schedule: Sequence[int]) -> Dict[str, Dict[str, int]]:
    return {"before": group_histogram(entries), "after": group_histogram(entries, schedule)}


def _postprocess_progress(label: str, current: int, total: int, started: float, interval: int) -> None:
    if current != 1 and current != total and current % max(int(interval), 1) != 0:
        return
    elapsed = time.monotonic() - started
    rate = current / max(elapsed, 1e-6)
    remaining = (total - current) / max(rate, 1e-6)
    print(
        f"[{label}] batch {current}/{total} ({100.0 * current / total:.1f}%), "
        f"elapsed {elapsed / 60.0:.1f} min, ETA {remaining / 60.0:.1f} min",
        flush=True,
    )


def _shuffled_labels(entries: Sequence[StyleEntry], *, seed: int) -> Dict[str, int]:
    labels = [STYLE_TO_INDEX[str(entry.style)] for entry in entries]
    rng = np.random.default_rng(seed)
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return {entry.filename: int(label) for entry, label in zip(entries, shuffled)}


def _labels(batch: Mapping[str, Any], shuffled: Optional[Mapping[str, int]] = None) -> torch.Tensor:
    labels = batch["style_index"]
    if shuffled is None:
        return labels
    return torch.tensor([shuffled[entry.filename] for entry in batch["entry"]], device=labels.device, dtype=torch.long)


def _new_networks(view: FixedQView, args: argparse.Namespace) -> Tuple[NetworkConfig, RawResidualExecutor]:
    config = NetworkConfig(
        future_len=int(view.base_ego.shape[1]), scene_dim=int(view.scene_feature.shape[1]),
        latent_dim=int(args.latent_dim), hidden_dim=int(args.hidden_dim), layers=3,
    )
    return config, RawResidualExecutor(config)


def _view_for_batch(
    model: nn.Module, model_args: Any, batch: Mapping[str, Any], *, seed: int, q_min: float, q_max: float
) -> FixedQView:
    time, noise = random_view_parameters(
        batch_size=int(batch["style_index"].shape[0]), predicted_neighbors=int(model_args.predicted_neighbor_num),
        future_len=int(model_args.future_len), seed=seed, device=batch["style_index"].device,
        q_min=q_min, q_max=q_max,
    )
    return fixed_q_view(model, model_args, batch, time, noise)


def _zero_identity(executor: RawResidualExecutor, view: FixedQView) -> float:
    zero = torch.zeros((int(view.base_ego.shape[0]), executor.config.latent_dim), device=view.base_ego.device, dtype=view.base_ego.dtype)
    return max_abs(executor.control(view.base_ego, view.scene_feature, view.diffusion_time, zero))


def _style_group_losses(
    executor: RawResidualExecutor, view: FixedQView, labels: torch.Tensor,
    sample_control: torch.Tensor, prototype_control: torch.Tensor, scenes: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Sample-code and scene×style prototype losses have intentionally distinct targets."""
    sample_prediction = executor.control(view.base_ego, view.scene_feature, view.diffusion_time, sample_control)
    active = labels != NORM_INDEX
    sample_loss = (
        functional.smooth_l1_loss(view.base_ego[active] + sample_prediction[active], view.target_ego[active])
        if bool(active.any()) else view.residual.new_zeros(())
    )
    prototype_prediction = executor.control(view.base_ego, view.scene_feature, view.diffusion_time, prototype_control)
    prototype_terms: List[torch.Tensor] = []
    magnitude_terms: List[torch.Tensor] = []
    counts: Dict[str, int] = {}
    for scene in sorted(set(scenes)):
        for style_index in (AGGR_INDEX, CONS_INDEX):
            rows = torch.tensor([item == scene for item in scenes], device=labels.device, dtype=torch.bool) & (labels == style_index)
            count = int(rows.sum().item())
            if count == 0:
                continue
            key = f"{scene}/{STYLE_NAMES[style_index]}"
            counts[key] = count
            mean_prediction = prototype_prediction[rows].mean(dim=0)
            mean_target = view.residual[rows].mean(dim=0)
            prototype_terms.append(functional.smooth_l1_loss(mean_prediction, mean_target))
            magnitude_terms.append(functional.smooth_l1_loss(
                torch.linalg.vector_norm(mean_prediction.reshape(-1), ord=2),
                torch.linalg.vector_norm(mean_target.reshape(-1), ord=2),
            ))
    prototype_loss = torch.stack(prototype_terms).mean() if prototype_terms else view.residual.new_zeros(())
    magnitude_loss = torch.stack(magnitude_terms).mean() if magnitude_terms else view.residual.new_zeros(())
    return sample_loss, prototype_loss, magnitude_loss, counts


def _fixed_q_evaluation(
    model: nn.Module, model_args: Any, entries: Sequence[StyleEntry], executor: RawResidualExecutor,
    controller: nn.Module, *, device: torch.device, batch_size: int, seed: int, maximum: int,
    encoder: Optional[ResidualStyleEncoder] = None,
) -> Dict[str, Any]:
    rows = _trainable(entries, maximum)
    totals: Dict[str, List[float]] = defaultdict(list)
    identity = 0.0
    group_counts: Counter = Counter()
    executor.eval(); controller.eval()
    if encoder is not None:
        encoder.eval()
    with torch.no_grad():
        for offset, subset in enumerate(_chunks(rows, batch_size)):
            batch = load_batch(subset, model_args, device)
            view = _view_for_batch(model, model_args, batch, seed=seed + offset, q_min=0.15, q_max=0.85)
            labels = batch["style_index"]
            correct = controller.for_labels(labels)
            opposite_labels = labels.clone()
            opposite_labels[labels == AGGR_INDEX] = CONS_INDEX
            opposite_labels[labels == CONS_INDEX] = AGGR_INDEX
            opposite = controller.for_labels(opposite_labels)
            base_error = functional.smooth_l1_loss(view.base_ego, view.target_ego, reduction="none").mean(dim=(1, 2))
            correct_error = functional.smooth_l1_loss(
                view.base_ego + executor.control(view.base_ego, view.scene_feature, view.diffusion_time, correct),
                view.target_ego, reduction="none",
            ).mean(dim=(1, 2))
            opposite_error = functional.smooth_l1_loss(
                view.base_ego + executor.control(view.base_ego, view.scene_feature, view.diffusion_time, opposite),
                view.target_ego, reduction="none",
            ).mean(dim=(1, 2))
            totals["base_error"].extend(base_error.cpu().tolist())
            totals["correct_prototype_error"].extend(correct_error.cpu().tolist())
            totals["opposite_prototype_error"].extend(opposite_error.cpu().tolist())
            identity = max(identity, _zero_identity(executor, view))
            for entry in subset:
                group_counts[f"{entry.scene}/{entry.style}"] += 1
            if encoder is not None:
                code = encoder(view.residual, view.scene_feature, view.diffusion_time)
                prototype = controller.prototype if isinstance(controller, PrototypeControl) else controller.prototypes()
                sample = code - prototype[NORM_INDEX]
                sample_error = functional.smooth_l1_loss(
                    view.base_ego + executor.control(view.base_ego, view.scene_feature, view.diffusion_time, sample),
                    view.target_ego, reduction="none",
                ).mean(dim=(1, 2))
                totals["sample_code_error"].extend(sample_error.cpu().tolist())
    summary = {key: float(np.mean(value)) if value else None for key, value in totals.items()}
    return {
        "sample_count": len(rows), "group_counts": dict(group_counts), "rho_zero_exact_error": identity,
        **summary,
        "correct_beats_base": None if summary.get("correct_prototype_error") is None else bool(summary["correct_prototype_error"] < summary["base_error"]),
        "correct_beats_opposite": None if summary.get("correct_prototype_error") is None else bool(summary["correct_prototype_error"] < summary["opposite_prototype_error"]),
    }


def _save_execution_checkpoint(
    path: Path, *, kind: str, config: NetworkConfig, executor: RawResidualExecutor,
    controller: Optional[DirectEmbeddingControl] = None, prototypes: Optional[torch.Tensor] = None,
    encoder_checkpoint: Optional[str] = None, base_checkpoint: str,
) -> None:
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA, "kind": kind, "network_config": config.to_dict(),
        "executor_state_dict": executor.state_dict(), "base_checkpoint": str(base_checkpoint),
        "encoder_checkpoint": encoder_checkpoint,
    }
    if controller is not None:
        payload["controller_state_dict"] = controller.state_dict()
    if prototypes is not None:
        payload["prototypes"] = prototypes.detach().cpu()
    torch.save(payload, path)


def _load_execution_checkpoint(path: str | Path, device: torch.device) -> Tuple[NetworkConfig, RawResidualExecutor, nn.Module, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA:
        raise ValueError("not a style_prototype_residual execution checkpoint")
    config = NetworkConfig(**{key: int(value) for key, value in dict(payload["network_config"]).items()})
    executor = RawResidualExecutor(config).to(device)
    executor.load_state_dict(payload["executor_state_dict"], strict=True)
    if payload["kind"] == "direct":
        controller = DirectEmbeddingControl(config.latent_dim).to(device)
        controller.load_state_dict(payload["controller_state_dict"], strict=True)
    elif payload["kind"] == "prototype":
        controller = PrototypeControl(torch.as_tensor(payload["prototypes"], dtype=torch.float32)).to(device)
    else:
        raise ValueError(f"unknown execution checkpoint kind {payload['kind']!r}")
    return config, executor.eval(), controller.eval(), dict(payload)


def _execution_reload_error(
    path: Path, executor: RawResidualExecutor, controller: nn.Module, view: FixedQView,
    labels: torch.Tensor, *, device: torch.device,
) -> float:
    """Exact reload check for the deployed executor and three saved controls."""
    config, restored_executor, restored_controller, _metadata = _load_execution_checkpoint(path, device)
    if config.to_dict() != executor.config.to_dict():
        return float("inf")
    with torch.no_grad():
        left = executor.control(view.base_ego, view.scene_feature, view.diffusion_time, controller.for_labels(labels))
        right = restored_executor.control(
            view.base_ego, view.scene_feature, view.diffusion_time, restored_controller.for_labels(labels)
        )
    return max_abs(left - right)


def command_data_audit(args: argparse.Namespace) -> None:
    output = _output_dir(args)
    scenes = _scenes(args.mode)
    train, train_skipped = load_entries(
        args.train_index, args.cache_root, split="train", allowed_scenes=scenes,
        low_threshold=args.cons_threshold, high_threshold=args.aggr_threshold, normal_half_width=args.normal_half_width,
    )
    val, val_skipped = load_entries(
        args.val_index, args.cache_root, split="val", allowed_scenes=scenes,
        low_threshold=args.cons_threshold, high_threshold=args.aggr_threshold, normal_half_width=args.normal_half_width,
    )
    train_manifest, val_manifest = _manifest_paths(output, args.mode)
    write_manifest(train_manifest, train)
    write_manifest(val_manifest, val)
    report = audit_entries(train, val, dt_seconds=args.dt_seconds, max_cache_audit=args.max_cache_audit)
    report.update({
        "mode": args.mode, "allowed_scenes": list(scenes), "train_index": str(Path(args.train_index).resolve()),
        "val_index": str(Path(args.val_index).resolve()), "cache_root": str(Path(args.cache_root).resolve()),
        "train_manifest": str(train_manifest), "val_manifest": str(val_manifest),
        "weak_label_rule": {
            "aggr_all_active_ge": args.aggr_threshold, "cons_all_active_le": args.cons_threshold,
            "norm_all_active_abs_from_0.5_le": args.normal_half_width,
            "mixed_axes": "unknown; never averaged into a personality scalar",
        }, "source_skipped": {"train": dict(train_skipped), "val": dict(val_skipped)},
    })
    write_json(output / "data_audit.json", report)
    print(f"Data audit written: {output / 'data_audit.json'}")
    print(f"Manifests written: {train_manifest}, {val_manifest}")


def command_same_state_contract(args: argparse.Namespace) -> None:
    _require_fc_gate(args)
    output = _output_dir(args)
    entries = _trainable(read_manifest(args.train_manifest), args.max_samples)
    device = torch.device(args.device)
    seed_everything(args.seed)
    model, model_args, checkpoint_meta = load_frozen_base(
        checkpoint=args.base_checkpoint, model_args=args.model_args, normalization_file_path=args.normalization_file_path,
        device=args.device, prefer_ema=args.prefer_ema,
    )
    snapshot = frozen_snapshot(model)
    rows: List[Dict[str, Any]] = []
    for offset, subset in enumerate(_chunks(entries, args.batch_size)):
        batch = load_batch(subset, model_args, device)
        first = _view_for_batch(model, model_args, batch, seed=args.seed + 2 * offset, q_min=args.q_min, q_max=args.q_max)
        second = _view_for_batch(model, model_args, batch, seed=args.seed + 2 * offset + 1, q_min=args.q_min, q_max=args.q_max)
        for index, entry in enumerate(subset):
            rows.append({
                "filename": entry.filename, "scene": entry.scene, "style": entry.style,
                "q1": float(first.diffusion_time[index].cpu().item()), "q2": float(second.diffusion_time[index].cpu().item()),
                "residual_q1_abs_mean": float(first.residual[index].abs().mean().cpu().item()),
                "residual_q2_abs_mean": float(second.residual[index].abs().mean().cpu().item()),
                "residual_q1_abs_max": max_abs(first.residual[index]), "residual_q2_abs_max": max_abs(second.residual[index]),
                "same_shape": tuple(first.residual[index].shape) == tuple(second.residual[index].shape),
                "finite": bool(torch.isfinite(first.residual[index]).all().item() and torch.isfinite(second.residual[index]).all().item()),
                "ego_only": tuple(first.residual[index].shape) == (int(model_args.future_len), 4),
            })
    report = {
        "schema_version": SCHEMA, "stage": "same_state_residual_contract", "sample_count": len(rows),
        "base_checkpoint_metadata": checkpoint_meta, "base_parameter_max_abs_change": frozen_change(model, snapshot),
        "uses_production_observation_normalizer": True, "uses_same_state_xq": True,
        "uses_free_running_final_trajectory_difference": False,
        "residual_definition": "expert normalized ego x0 - frozen base clean x0 at the same xq",
        "all_finite": all(row["finite"] for row in rows), "all_ego_only": all(row["ego_only"] for row in rows),
        "rows": rows,
    }
    write_json(output / "same_state_residual_contract.json", report)
    print(f"Same-state residual contract written: {output / 'same_state_residual_contract.json'}")


def _training_setup(args: argparse.Namespace) -> Tuple[Path, List[StyleEntry], List[StyleEntry], torch.device, nn.Module, Any, Dict[str, Any], Dict[str, torch.Tensor]]:
    _require_fc_gate(args)
    output = _output_dir(args)
    train = _trainable(read_manifest(args.train_manifest), args.max_train_samples)
    val = _trainable(read_manifest(args.val_manifest), args.max_val_samples)
    device = torch.device(args.device)
    seed_everything(args.seed)
    model, model_args, metadata = load_frozen_base(
        checkpoint=args.base_checkpoint, model_args=args.model_args, normalization_file_path=args.normalization_file_path,
        device=args.device, prefer_ema=args.prefer_ema,
    )
    return output, train, val, device, model, model_args, metadata, frozen_snapshot(model)


def command_train_direct(args: argparse.Namespace) -> None:
    output, train, val, device, model, model_args, metadata, snapshot = _training_setup(args)
    schedule = balanced_indices(train, args.steps * args.batch_size, seed=args.seed, batch_size=args.batch_size)
    initial_batch = load_batch([train[index] for index in schedule[:args.batch_size]], model_args, device)
    initial_view = _view_for_batch(model, model_args, initial_batch, seed=args.seed, q_min=args.q_min, q_max=args.q_max)
    config, executor = _new_networks(initial_view, args)
    executor = executor.to(device)
    controller = DirectEmbeddingControl(config.latent_dim).to(device)
    optimizer = torch.optim.AdamW(list(executor.parameters()) + list(controller.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    monitor = SwanMonitor.create(args, {"kind": "direct", "mode": args.mode, "network": config.to_dict(), "steps": args.steps}, output)
    history: List[Dict[str, float]] = []
    try:
        for step in range(args.steps):
            chosen = schedule[step * args.batch_size : (step + 1) * args.batch_size]
            batch = load_batch([train[index] for index in chosen], model_args, device)
            view = _view_for_batch(model, model_args, batch, seed=args.seed + step, q_min=args.q_min, q_max=args.q_max)
            labels = batch["style_index"]
            control = controller.for_labels(labels)
            sample, prototype, magnitude, groups = _style_group_losses(
                executor, view, labels, control, control, batch["scene"],
            )
            loss = sample + args.lambda_prototype * prototype + args.lambda_magnitude * magnitude
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all_finite(list(executor.parameters()) + list(controller.parameters())):
                raise FloatingPointError("non-finite direct-embedding gradient")
            optimizer.step()
            row = {"step": float(step + 1), "loss": float(loss.detach().cpu().item()), "sample": float(sample.detach().cpu().item()),
                   "prototype": float(prototype.detach().cpu().item()), "magnitude": float(magnitude.detach().cpu().item()),
                   "grad_norm": grad_norm(list(executor.parameters()) + list(controller.parameters())), "groups": float(len(groups))}
            if step == 0 or (step + 1) % args.log_interval == 0 or step + 1 == args.steps:
                history.append(row); monitor.log({key: value for key, value in row.items() if key not in {"step", "groups"}}, step + 1)
    finally:
        monitor.finish()
    checkpoint = output / "direct_embedding.pt"
    _save_execution_checkpoint(checkpoint, kind="direct", config=config, executor=executor, controller=controller, base_checkpoint=args.base_checkpoint)
    fixed = _fixed_q_evaluation(
        model, model_args, val, executor, controller, device=device, batch_size=args.batch_size,
        seed=args.seed + 100000, maximum=args.max_eval_samples,
    )
    reload_error = _execution_reload_error(
        checkpoint, executor, controller, initial_view, initial_batch["style_index"], device=device,
    )
    report = {
        "schema_version": SCHEMA, "stage": "direct_embedding_fixed_q", "mode": args.mode,
        "checkpoint": str(checkpoint), "network_config": config.to_dict(), "base_checkpoint_metadata": metadata,
        "base_parameter_max_abs_change": frozen_change(model, snapshot), "training_history": history,
        "fixed_q": fixed, "normal_control_exact_error": max_abs(controller.for_labels(torch.tensor([NORM_INDEX], device=device))),
        "uses_residual_encoder": False, "same_executor_architecture": True,
        "optimizer": "AdamW", "seed": args.seed, "steps": args.steps,
        "scene_style_sampling": _sampling_balance(train, schedule),
        "checkpoint_reload_control_error": reload_error,
    }
    write_json(output / "direct_embedding_baseline.json", report)
    print(f"Direct embedding checkpoint: {checkpoint}")
    print(f"Direct fixed-q report: {output / 'direct_embedding_baseline.json'}")


def _prototype_view_stream(
    model: nn.Module, model_args: Any, entries: Sequence[StyleEntry], *, device: torch.device,
    batch_size: int, seed: int, maximum: int, label_map: Optional[Mapping[str, int]] = None,
    progress_label: str = "", progress_interval: int = 25,
) -> Iterator[Tuple[FixedQView, torch.Tensor]]:
    rows = _trainable(entries, maximum)
    total = math.ceil(len(rows) / batch_size)
    started = time.monotonic()
    for offset, subset in enumerate(_chunks(rows, batch_size)):
        batch = load_batch(subset, model_args, device)
        if progress_label:
            _postprocess_progress(progress_label, offset + 1, total, started, progress_interval)
        yield (
            _view_for_batch(model, model_args, batch, seed=seed + offset, q_min=0.15, q_max=0.85),
            _labels(batch, label_map),
        )


def _balanced_accuracy(prediction: torch.Tensor, target: torch.Tensor, class_count: int) -> Optional[float]:
    values: List[float] = []
    for index in range(class_count):
        selected = target == index
        if bool(selected.any()):
            values.append(float((prediction[selected] == target[selected]).float().mean().cpu().item()))
    return None if not values else float(np.mean(values))


def _linear_probe(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, *, classes: int, seed: int) -> Dict[str, Optional[float]]:
    observed = set(train_y.cpu().tolist())
    if len(observed) < 2 or val_x.numel() == 0:
        return {"accuracy": None, "balanced_accuracy": None, "class_count": len(observed)}
    seed_everything(seed)
    probe = nn.Linear(int(train_x.shape[1]), classes).to(train_x.device)
    counts = torch.bincount(train_y, minlength=classes).float().clamp_min(1.0)
    weights = (counts.sum() / counts).to(train_x.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        functional.cross_entropy(probe(train_x), train_y, weight=weights).backward()
        optimizer.step()
    with torch.no_grad():
        prediction = probe(val_x).argmax(dim=-1)
    return {
        "accuracy": float((prediction == val_y).float().mean().cpu().item()),
        "balanced_accuracy": _balanced_accuracy(prediction, val_y, classes), "class_count": len(observed),
    }


def _collect_codes(
    model: nn.Module, model_args: Any, encoder: ResidualStyleEncoder, entries: Sequence[StyleEntry], *,
    device: torch.device, batch_size: int, seed: int, maximum: int,
    progress_label: str, progress_interval: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    codes: List[torch.Tensor] = []; labels: List[torch.Tensor] = []; scenes: List[torch.Tensor] = []
    same_distances: List[float] = []; different_distances: List[float] = []
    scene_to_index = {scene: index for index, scene in enumerate(sorted({entry.scene for entry in entries}))}
    rows = _trainable(entries, maximum)
    total = math.ceil(len(rows) / batch_size)
    started = time.monotonic()
    encoder.eval()
    with torch.no_grad():
        for offset, subset in enumerate(_chunks(rows, batch_size)):
            _postprocess_progress(progress_label, offset + 1, total, started, progress_interval)
            batch = load_batch(subset, model_args, device)
            first = _view_for_batch(model, model_args, batch, seed=seed + 2 * offset, q_min=0.15, q_max=0.85)
            second = _view_for_batch(model, model_args, batch, seed=seed + 2 * offset + 1, q_min=0.15, q_max=0.85)
            z1 = encoder(first.residual, first.scene_feature, first.diffusion_time)
            z2 = encoder(second.residual, second.scene_feature, second.diffusion_time)
            same_distances.extend(torch.linalg.vector_norm(z1 - z2, dim=-1).cpu().tolist())
            if int(z1.shape[0]) > 1:
                different_distances.extend(torch.linalg.vector_norm(z1 - torch.roll(z1, 1, dims=0), dim=-1).cpu().tolist())
            codes.append(z1.cpu()); labels.append(batch["style_index"].cpu())
            scenes.append(torch.tensor([scene_to_index[scene] for scene in batch["scene"]], dtype=torch.long))
    return torch.cat(codes), torch.cat(labels), torch.cat(scenes), float(np.mean(same_distances)), float(np.mean(different_distances))


def _representation_report(
    model: nn.Module, model_args: Any, encoder: ResidualStyleEncoder, prototypes: torch.Tensor,
    train: Sequence[StyleEntry], val: Sequence[StyleEntry], *, device: torch.device, args: argparse.Namespace,
    label_shuffle: bool,
) -> Dict[str, Any]:
    train_z, train_y, train_scene, train_same, train_diff = _collect_codes(
        model, model_args, encoder, train, device=device, batch_size=args.batch_size,
        seed=args.seed + 200000, maximum=args.max_eval_samples,
        progress_label="representation/train", progress_interval=args.postprocess_progress_interval,
    )
    val_z, val_y, val_scene, val_same, val_diff = _collect_codes(
        model, model_args, encoder, val, device=device, batch_size=args.batch_size,
        seed=args.seed + 300000, maximum=args.max_eval_samples,
        progress_label="representation/val", progress_interval=args.postprocess_progress_interval,
    )
    train_z = train_z.to(device); train_y = train_y.to(device); train_scene = train_scene.to(device)
    val_z = val_z.to(device); val_y = val_y.to(device); val_scene = val_scene.to(device)
    canonical = prototypes.to(device)
    nearest = torch.cdist(val_z, canonical).argmin(dim=-1)
    style_probe = _linear_probe(train_z, train_y, val_z, val_y, classes=3, seed=args.seed + 1)
    scene_classes = max(int(train_scene.max().item()), int(val_scene.max().item())) + 1
    scene_probe = _linear_probe(train_z, train_scene, val_z, val_scene, classes=scene_classes, seed=args.seed + 2)
    means = []
    within: List[float] = []
    for index in range(3):
        selected = train_z[train_y == index]
        if selected.numel():
            mean = selected.mean(dim=0); means.append(mean)
            within.append(float(torch.linalg.vector_norm(selected - mean, dim=-1).mean().cpu().item()))
    between = [float(torch.linalg.vector_norm(left - right).cpu().item()) for pos, left in enumerate(means) for right in means[pos + 1 :]]
    majority = float(torch.bincount(val_y, minlength=3).max().item() / max(int(val_y.numel()), 1))
    return {
        "schema_version": SCHEMA, "stage": "representation", "label_shuffle": bool(label_shuffle),
        "train_sample_count": int(train_y.numel()), "val_sample_count": int(val_y.numel()),
        "prototype_nearest_balanced_accuracy": _balanced_accuracy(nearest, val_y, 3),
        "majority_class_accuracy": majority, "style_probe": style_probe, "scene_probe": scene_probe,
        "style_probe_minus_scene_probe": None if style_probe["balanced_accuracy"] is None or scene_probe["balanced_accuracy"] is None else float(style_probe["balanced_accuracy"] - scene_probe["balanced_accuracy"]),
        "same_sample_cross_q_noise_distance": {"train": train_same, "val": val_same},
        "different_sample_distance": {"train": train_diff, "val": val_diff},
        "within_class_distance_mean": float(np.mean(within)) if within else None,
        "between_class_distance_mean": float(np.mean(between)) if between else None,
        "scene_style_balanced_sampling": True,
        "prototype": prototypes.detach().cpu(),
    }


def command_train_encoder(args: argparse.Namespace) -> None:
    output, train, val, device, model, model_args, metadata, snapshot = _training_setup(args)
    schedule = balanced_indices(train, args.steps * args.batch_size, seed=args.seed, batch_size=args.batch_size)
    shuffled = _shuffled_labels(train, seed=args.seed + 918) if args.label_shuffle else None
    first_batch = load_batch([train[index] for index in schedule[:args.batch_size]], model_args, device)
    first_view = _view_for_batch(model, model_args, first_batch, seed=args.seed, q_min=args.q_min, q_max=args.q_max)
    config, _executor_unused = _new_networks(first_view, args)
    encoder = ResidualStyleEncoder(config).to(device)
    learned_prototypes = nn.Parameter(torch.randn((3, config.latent_dim), device=device) * 0.02)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + [learned_prototypes], lr=args.learning_rate, weight_decay=args.weight_decay)
    monitor = SwanMonitor.create(args, {"kind": "encoder", "mode": args.mode, "network": config.to_dict(), "label_shuffle": args.label_shuffle}, output)
    history: List[Dict[str, float]] = []
    try:
        for step in range(args.steps):
            chosen = schedule[step * args.batch_size : (step + 1) * args.batch_size]
            batch = load_batch([train[index] for index in chosen], model_args, device)
            labels = _labels(batch, shuffled)
            first = _view_for_batch(model, model_args, batch, seed=args.seed + 2 * step, q_min=args.q_min, q_max=args.q_max)
            second = _view_for_batch(model, model_args, batch, seed=args.seed + 2 * step + 1, q_min=args.q_min, q_max=args.q_max)
            z1 = encoder(first.residual, first.scene_feature, first.diffusion_time)
            z2 = encoder(second.residual, second.scene_feature, second.diffusion_time)
            logits1 = -torch.cdist(z1, learned_prototypes).square() / args.prototype_temperature
            logits2 = -torch.cdist(z2, learned_prototypes).square() / args.prototype_temperature
            representation = 0.5 * (functional.cross_entropy(logits1, labels) + functional.cross_entropy(logits2, labels))
            view_loss = functional.mse_loss(z1, z2)
            loss = representation + args.lambda_view * view_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not all_finite(list(encoder.parameters()) + [learned_prototypes]):
                raise FloatingPointError("non-finite residual-encoder gradient")
            optimizer.step()
            row = {"step": float(step + 1), "loss": float(loss.detach().cpu().item()), "repr": float(representation.detach().cpu().item()),
                   "view": float(view_loss.detach().cpu().item()), "grad_norm": grad_norm(encoder.parameters())}
            if step == 0 or (step + 1) % args.log_interval == 0 or step + 1 == args.steps:
                history.append(row); monitor.log({key: value for key, value in row.items() if key != "step"}, step + 1)
    finally:
        monitor.finish()
    prototypes = prototype_means(
        encoder,
        _prototype_view_stream(
            model, model_args, train, device=device, batch_size=args.batch_size,
            seed=args.seed + 400000, maximum=args.max_prototype_samples,
            label_map=shuffled,
            progress_label="prototype/means", progress_interval=args.postprocess_progress_interval,
        ),
    )
    checkpoint = output / ("residual_encoder_label_shuffle.pt" if args.label_shuffle else "residual_encoder.pt")
    torch.save({"schema_version": SCHEMA, "network_config": config.to_dict(), "encoder_state_dict": encoder.state_dict(),
                "prototypes": prototypes.detach().cpu(), "learned_prototypes": learned_prototypes.detach().cpu(),
                "label_shuffle": bool(args.label_shuffle), "base_checkpoint": args.base_checkpoint}, checkpoint)
    report = _representation_report(model, model_args, encoder, prototypes, train, val, device=device, args=args, label_shuffle=args.label_shuffle)
    report.update({"mode": args.mode, "checkpoint": str(checkpoint), "base_checkpoint_metadata": metadata,
                   "base_parameter_max_abs_change": frozen_change(model, snapshot), "training_history": history,
                   "weak_supervision": "scene-by-style balanced three-class labels; no behavior-axis loss", "deployment_uses_encoder": False})
    report["scene_style_sampling"] = _sampling_balance(train, schedule)
    name = "representation_label_shuffle.json" if args.label_shuffle else f"representation_{args.mode}.json"
    write_json(output / name, report)
    print(f"Residual encoder checkpoint: {checkpoint}")
    print(f"Representation report: {output / name}")


def _load_encoder_checkpoint(path: str | Path, device: torch.device) -> Tuple[NetworkConfig, ResidualStyleEncoder, torch.Tensor, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA:
        raise ValueError("not a style_prototype_residual encoder checkpoint")
    config = NetworkConfig(**{key: int(value) for key, value in dict(payload["network_config"]).items()})
    encoder = ResidualStyleEncoder(config).to(device)
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    return config, encoder.eval(), torch.as_tensor(payload["prototypes"], device=device, dtype=torch.float32), dict(payload)


def command_train_executor(args: argparse.Namespace) -> None:
    output, train, val, device, model, model_args, metadata, snapshot = _training_setup(args)
    config, encoder, prototypes, encoder_meta = _load_encoder_checkpoint(args.encoder_checkpoint, device)
    schedule = balanced_indices(train, args.steps * args.batch_size, seed=args.seed, batch_size=args.batch_size)
    executor = RawResidualExecutor(config).to(device)
    controller = PrototypeControl(prototypes).to(device)
    optimizer = torch.optim.AdamW(executor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    monitor = SwanMonitor.create(args, {"kind": "prototype_executor", "mode": args.mode, "network": config.to_dict(), "steps": args.steps}, output)
    history: List[Dict[str, float]] = []
    try:
        for step in range(args.steps):
            chosen = schedule[step * args.batch_size : (step + 1) * args.batch_size]
            batch = load_batch([train[index] for index in chosen], model_args, device)
            view = _view_for_batch(model, model_args, batch, seed=args.seed + step, q_min=args.q_min, q_max=args.q_max)
            labels = batch["style_index"]
            with torch.no_grad():
                sample_control = encoder(view.residual, view.scene_feature, view.diffusion_time) - prototypes[NORM_INDEX]
            prototype_control = controller.for_labels(labels)
            sample, prototype, magnitude, groups = _style_group_losses(
                executor, view, labels, sample_control, prototype_control, batch["scene"],
            )
            loss = sample + args.lambda_prototype * prototype + args.lambda_magnitude * magnitude
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not all_finite(executor.parameters()):
                raise FloatingPointError("non-finite prototype-executor gradient")
            optimizer.step()
            row = {"step": float(step + 1), "loss": float(loss.detach().cpu().item()), "sample": float(sample.detach().cpu().item()),
                   "prototype": float(prototype.detach().cpu().item()), "magnitude": float(magnitude.detach().cpu().item()),
                   "grad_norm": grad_norm(executor.parameters()), "groups": float(len(groups))}
            if step == 0 or (step + 1) % args.log_interval == 0 or step + 1 == args.steps:
                history.append(row); monitor.log({key: value for key, value in row.items() if key not in {"step", "groups"}}, step + 1)
    finally:
        monitor.finish()
    checkpoint = output / "prototype_executor.pt"
    _save_execution_checkpoint(
        checkpoint, kind="prototype", config=config, executor=executor, prototypes=prototypes,
        encoder_checkpoint=str(args.encoder_checkpoint), base_checkpoint=args.base_checkpoint,
    )
    fixed = _fixed_q_evaluation(
        model, model_args, val, executor, controller, device=device, batch_size=args.batch_size,
        seed=args.seed + 500000, maximum=args.max_eval_samples, encoder=encoder,
    )
    check_batch = load_batch(train[: min(len(train), args.batch_size)], model_args, device)
    check_view = _view_for_batch(model, model_args, check_batch, seed=args.seed + 700000, q_min=args.q_min, q_max=args.q_max)
    reload_error = _execution_reload_error(
        checkpoint, executor, controller, check_view, check_batch["style_index"], device=device,
    )
    report = {
        "schema_version": SCHEMA, "stage": "prototype_executor_fixed_q", "mode": args.mode,
        "checkpoint": str(checkpoint), "encoder_checkpoint": str(args.encoder_checkpoint), "network_config": config.to_dict(),
        "base_checkpoint_metadata": metadata, "encoder_training_label_shuffle": bool(encoder_meta.get("label_shuffle", False)),
        "base_parameter_max_abs_change": frozen_change(model, snapshot), "training_history": history, "fixed_q": fixed,
        "normal_control_exact_error": max_abs(controller.for_labels(torch.tensor([NORM_INDEX], device=device))),
        "sample_code_supervision": "individual aggr/cons residual reconstruction", "prototype_supervision": "scene-by-style group mean only",
        "deployment_uses_residual_encoder": False,
        "scene_style_sampling": _sampling_balance(train, schedule),
        "checkpoint_reload_control_error": reload_error,
    }
    write_json(output / "executor_fixed_q.json", report)
    print(f"Prototype executor checkpoint: {checkpoint}")
    print(f"Prototype fixed-q report: {output / 'executor_fixed_q.json'}")


def _parse_rho_grid(text: str) -> List[float]:
    try:
        values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--rho-grid must be a comma-separated finite float list") from error
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("--rho-grid must contain at least one finite value")
    if 0.0 not in values:
        values.append(0.0)
    return sorted(set(values))


def _trace_max_error(reference: Sequence[torch.Tensor], candidate: Sequence[torch.Tensor]) -> float:
    if len(reference) != len(candidate):
        return float("inf")
    return max((max_abs(left - right) for left, right in zip(reference, candidate)), default=0.0)


def _trace_time_error(reference: CleanPredictionTraceRecorder, candidate: CleanPredictionTraceRecorder) -> float:
    left = reference.records()
    right = candidate.records()
    if len(left) != len(right):
        return float("inf")
    values: List[float] = []
    for first, second in zip(left, right):
        values.extend((max_abs(first.diffusion_time - second.diffusion_time), max_abs(first.log_snr - second.log_snr)))
    return max(values, default=0.0)


def _trajectory_change_metrics(current_xy: torch.Tensor, base: torch.Tensor, edited: torch.Tensor) -> Dict[str, float]:
    """Physical diagnostics only; they are never used as an optimizer loss."""
    difference = edited - base
    xy_difference = difference[:, :2]
    positions = torch.cat((current_xy.reshape(1, 2), base[:, :2]), dim=0)
    tangent = positions[1:] - positions[:-1]
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-6)
    normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=-1)
    longitudinal = (xy_difference * tangent).sum(dim=-1)
    lateral = (xy_difference * normal).sum(dim=-1)
    return {
        "trajectory_abs_mean_m": float(torch.linalg.vector_norm(xy_difference, dim=-1).mean().cpu().item()),
        "trajectory_abs_p95_m": float(torch.quantile(torch.linalg.vector_norm(xy_difference, dim=-1), 0.95).cpu().item()),
        "trajectory_abs_max_m": float(torch.linalg.vector_norm(xy_difference, dim=-1).max().cpu().item()),
        "longitudinal_abs_mean_m": float(longitudinal.abs().mean().cpu().item()),
        "lateral_abs_max_m": float(lateral.abs().max().cpu().item()),
        "lateral_signed_final_m": float(lateral[-1].cpu().item()),
    }


def _future_reference_after_rollout(cache_path: str | Path, *, device: torch.device) -> torch.Tensor:
    """Read expert future strictly after every deployment rollout has finished."""
    cache = opendata(str(cache_path))
    try:
        if "ego_agent_future" not in cache:
            raise KeyError(f"cache {cache_path} has no ego_agent_future for evaluation")
        return torch.as_tensor(cache["ego_agent_future"], device=device, dtype=torch.float32).detach()
    finally:
        cache.close()


def _ade_fde(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    length = min(int(prediction.shape[0]), int(target.shape[0]))
    if length <= 0:
        return {"ade_m": float("nan"), "fde_m": float("nan")}
    distance = torch.linalg.vector_norm(prediction[:length, :2] - target[:length, :2], dim=-1)
    return {"ade_m": float(distance.mean().cpu().item()), "fde_m": float(distance[-1].cpu().item())}


def _style_rho(style: Optional[str]) -> float:
    if style == "aggr":
        return 1.0
    if style == "cons":
        return -1.0
    return 0.0


def _full_report_name(kind: str, mode: str) -> str:
    stem = "direct_embedding_full_dpm" if kind == "direct" else "executor_full_dpm"
    return f"{stem}{'_fcl' if mode == 'fcl' else ''}.json"


def command_full_dpm(args: argparse.Namespace) -> None:
    """Validate deployment with a single DPM stream and no future-derived input."""
    _require_fc_gate(args)
    output = _output_dir(args)
    entries = _trainable(read_manifest(args.val_manifest), args.max_full_dpm_samples)
    device = torch.device(args.device)
    seed_everything(args.seed)
    model, model_args, metadata = load_frozen_base(
        checkpoint=args.base_checkpoint, model_args=args.model_args,
        normalization_file_path=args.normalization_file_path, device=args.device,
        prefer_ema=args.prefer_ema,
    )
    base_snapshot = frozen_snapshot(model)
    config, executor, controller, execution_meta = _load_execution_checkpoint(args.execution_checkpoint, device)
    if int(config.future_len) != int(model_args.future_len):
        raise ValueError("execution checkpoint future length does not match frozen base planner")
    rho_grid = _parse_rho_grid(args.rho_grid)
    rows: List[Dict[str, Any]] = []
    summary_values: Dict[str, List[float]] = defaultdict(list)
    maximum_x0_identity = 0.0
    maximum_xq_identity = 0.0
    maximum_time_identity = 0.0
    maximum_final_identity = 0.0
    maximum_non_ego_direct = 0.0
    maximum_current_direct = 0.0
    all_finite = True
    expected_evaluations: Optional[int] = None
    lane_checks: List[Dict[str, Any]] = []
    executor.eval(); controller.eval()
    for index, entry in enumerate(entries):
        # This input path has observations only.  The expert future is opened below,
        # after all base/edited DPM runs, solely to calculate evaluation metrics.
        raw_inputs, normalized_inputs = full_dpm_inputs(entry.cache_path, model_args, device)
        scene_feature = frozen_scene_feature(model, normalized_inputs)
        base_trace = CleanPredictionTraceRecorder()
        base_prediction = full_dpm_rollout(
            model, normalized_inputs, seed=args.seed + index, observer=base_trace,
        )
        zero_trace = CleanPredictionTraceRecorder()
        zero_editor = SingleStreamResidualEditor(
            executor, controller, scene_feature, 0.0,
            agent_count=1 + int(model_args.predicted_neighbor_num), future_len=int(model_args.future_len),
        )
        zero_prediction = full_dpm_rollout(
            model, normalized_inputs, seed=args.seed + index, editor=zero_editor, observer=zero_trace,
        )
        x0_identity = _trace_max_error(base_trace.snapshots(), zero_trace.snapshots())
        xq_identity = _trace_max_error(base_trace.current_state_snapshots(), zero_trace.current_state_snapshots())
        time_identity = _trace_time_error(base_trace, zero_trace)
        final_identity = max_abs(base_prediction - zero_prediction)
        maximum_x0_identity = max(maximum_x0_identity, x0_identity)
        maximum_xq_identity = max(maximum_xq_identity, xq_identity)
        maximum_time_identity = max(maximum_time_identity, time_identity)
        maximum_final_identity = max(maximum_final_identity, final_identity)
        if expected_evaluations is None:
            expected_evaluations = len(base_trace.records())
        elif expected_evaluations != len(base_trace.records()):
            raise RuntimeError("full-DPM evaluation count changed across samples")
        if int(expected_evaluations) != int(args.expected_evaluations):
            raise AssertionError(
                f"expected {args.expected_evaluations} DPM evaluations, got {expected_evaluations}; "
                "do not compare a different solver schedule"
            )
        outputs: Dict[float, torch.Tensor] = {0.0: zero_prediction}
        edit_records: Dict[float, List[Dict[str, Any]]] = {0.0: list(zero_editor.records)}
        for rho in rho_grid:
            if rho == 0.0:
                continue
            editor = SingleStreamResidualEditor(
                executor, controller, scene_feature, rho,
                agent_count=1 + int(model_args.predicted_neighbor_num), future_len=int(model_args.future_len),
            )
            outputs[rho] = full_dpm_rollout(model, normalized_inputs, seed=args.seed + index, editor=editor)
            edit_records[rho] = list(editor.records)
        # Deliberately delayed: no expert future can influence this rollout or editor.
        expert = _future_reference_after_rollout(entry.cache_path, device=device)
        current_xy = raw_inputs["ego_current_state"][0, :2].detach()
        base_ego = base_prediction[0, 0]
        per_rho: Dict[str, Any] = {}
        for rho in rho_grid:
            prediction = outputs[rho][0, 0]
            direct = edit_records[rho]
            non_ego_direct = max((float(record["non_ego_residual_abs_max"]) for record in direct), default=0.0)
            current_direct = max((float(record["ego_current_residual_abs_max"]) for record in direct), default=0.0)
            maximum_non_ego_direct = max(maximum_non_ego_direct, non_ego_direct)
            maximum_current_direct = max(maximum_current_direct, current_direct)
            finite = bool(torch.isfinite(prediction).all().item())
            all_finite = all_finite and finite
            change = _trajectory_change_metrics(current_xy, base_ego, prediction)
            motion = physical_motion_metrics(current_xy, prediction)
            target_error = _ade_fde(prediction, expert)
            per_rho[f"{rho:+.3f}"] = {
                "rho": rho, "finite": finite, "target_error": target_error,
                "change_from_base": change, "motion": motion,
                "editor_evaluation_count": len(direct),
                "editor_ego_future_residual_abs_max": max((float(record["ego_future_residual_abs_max"]) for record in direct), default=0.0),
                "editor_non_ego_direct_residual_abs_max": non_ego_direct,
                "editor_ego_current_direct_residual_abs_max": current_direct,
            }
            summary_values[f"ade/{rho:+.3f}"].append(target_error["ade_m"])
            summary_values[f"change/{rho:+.3f}"].append(change["trajectory_abs_mean_m"])
        label_rho = _style_rho(entry.style)
        correct_ade = per_rho[f"{label_rho:+.3f}"]["target_error"]["ade_m"] if label_rho in outputs else None
        base_ade = per_rho["+0.000"]["target_error"]["ade_m"]
        if entry.scene == "straight_lane_change":
            threshold = float(args.lane_change_min_lateral_m)
            expert_delta = float((expert[-1, 1] - current_xy[1]).cpu().item())
            expected_sign = 0 if abs(expert_delta) < threshold else (1 if expert_delta > 0.0 else -1)
            predicted_delta = {
                f"{rho:+.3f}": float((outputs[rho][0, 0, -1, 1] - current_xy[1]).cpu().item())
                for rho in rho_grid
            }
            direction_ok = all(
                (abs(value) < threshold if expected_sign == 0 else value * expected_sign >= threshold)
                for value in predicted_delta.values()
            )
            lane_checks.append({
                "filename": entry.filename, "expert_final_lateral_delta_m": expert_delta,
                "expected_direction": expected_sign, "predicted_final_lateral_delta_m": predicted_delta,
                "direction_preserved": direction_ok,
            })
        rows.append({
            "filename": entry.filename, "scene": entry.scene, "style": entry.style,
            "dpm_evaluations": len(base_trace.records()),
            "base_vs_rho_zero": {
                "x0_max_abs_error": x0_identity, "xq_max_abs_error": xq_identity,
                "time_log_snr_max_abs_error": time_identity, "final_max_abs_error": final_identity,
            },
            "base_ade_m": base_ade, "style_matched_rho": label_rho,
            "style_matched_ade_m": correct_ade,
            "style_matched_minus_base_ade_m": None if correct_ade is None else correct_ade - base_ade,
            "rho": per_rho,
        })
    pair_differences: List[float] = []
    for row in rows:
        negative = row["rho"].get("-1.000")
        positive = row["rho"].get("+1.000")
        if negative is not None and positive is not None:
            pair_differences.append(abs(
                negative["change_from_base"]["trajectory_abs_mean_m"]
                - positive["change_from_base"]["trajectory_abs_mean_m"]
            ))
    aggregate = {key: float(np.mean(value)) for key, value in summary_values.items() if value}
    response_present = any(value > float(args.min_response_m) for key, values in summary_values.items() if key.startswith("change/") for value in values)
    base_change = frozen_change(model, base_snapshot)
    report = {
        "schema_version": SCHEMA, "stage": "single_stream_full_dpm", "mode": args.mode,
        "kind": execution_meta["kind"], "execution_checkpoint": str(args.execution_checkpoint),
        "base_checkpoint_metadata": metadata, "base_parameter_max_abs_change": base_change,
        "rho_grid": rho_grid, "sample_count": len(rows), "evaluations_per_scene": expected_evaluations,
        "deployment_contract": {
            "single_dpm_stream": True, "uses_residual_encoder": False,
            "uses_future_or_offline_label_as_input": False,
            "expert_future_read_after_all_rollouts_for_metrics_only": True,
            "final_clean_prediction_formula": "base_clean_x0 + A_ctrl",
        },
        "rho_zero_identity": {
            "max_abs_clean_x0_error": maximum_x0_identity,
            "max_abs_dpm_state_error": maximum_xq_identity,
            "max_abs_time_log_snr_error": maximum_time_identity,
            "max_abs_final_trajectory_error": maximum_final_identity,
            "strict": maximum_x0_identity == 0.0 and maximum_xq_identity == 0.0 and maximum_time_identity == 0.0 and maximum_final_identity == 0.0,
        },
        "direct_edit_contract": {
            "max_non_ego_direct_residual": maximum_non_ego_direct,
            "max_ego_current_direct_residual": maximum_current_direct,
            "strict": maximum_non_ego_direct == 0.0 and maximum_current_direct == 0.0,
        },
        "aggregate": aggregate,
        "negative_positive_mean_change_difference_m": float(np.mean(pair_differences)) if pair_differences else None,
        "lane_change_checks": lane_checks,
        "lane_direction_preserved": bool(lane_checks) and all(bool(item["direction_preserved"]) for item in lane_checks),
        "response_present": bool(response_present), "all_finite": all_finite,
        "full_dpm_execution_supported": bool(
            all_finite and base_change == 0.0 and response_present and maximum_x0_identity == 0.0 and maximum_xq_identity == 0.0
            and maximum_time_identity == 0.0 and maximum_final_identity == 0.0
            and maximum_non_ego_direct == 0.0 and maximum_current_direct == 0.0
        ),
        "rows": rows,
    }
    report.pop("_unused", None)
    write_json(output / _full_report_name(str(execution_meta["kind"]), args.mode), report)
    print(f"Full-DPM report: {output / _full_report_name(str(execution_meta['kind']), args.mode)}")


def _load_json_optional(path: str) -> Optional[Dict[str, Any]]:
    if not str(path).strip():
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _fixed_report_supported(report: Optional[Mapping[str, Any]], *, sample_code: bool = False) -> bool:
    if report is None:
        return False
    fixed = report.get("fixed_q", {})
    if not isinstance(fixed, Mapping):
        return False
    if float(report.get("base_parameter_max_abs_change", float("inf"))) != 0.0:
        return False
    if float(report.get("checkpoint_reload_control_error", float("inf"))) != 0.0:
        return False
    if float(fixed.get("rho_zero_exact_error", float("inf"))) != 0.0:
        return False
    if sample_code:
        return "sample_code_error" in fixed and float(fixed["sample_code_error"]) < float(fixed.get("base_error", float("inf")))
    return bool(fixed.get("correct_beats_base", False) and fixed.get("correct_beats_opposite", False))


def command_summary(args: argparse.Namespace) -> None:
    """Write an explicit evidence summary; successful execution is not a research claim."""
    output = _output_dir(args)
    direct_fixed = _load_json_optional(args.direct_fixed_report)
    direct_full = _load_json_optional(args.direct_full_report)
    representation = _load_json_optional(args.representation_report)
    prototype_fixed = _load_json_optional(args.prototype_fixed_report)
    prototype_full = _load_json_optional(args.prototype_full_report)
    label_shuffle = _load_json_optional(args.label_shuffle_report)
    fcl_full = _load_json_optional(args.fcl_full_report)
    direct_supported = bool(
        _fixed_report_supported(direct_fixed)
        and direct_full is not None and bool(direct_full.get("full_dpm_execution_supported", False))
    )
    sample_supported = _fixed_report_supported(prototype_fixed, sample_code=True)
    prototype_supported = _fixed_report_supported(prototype_fixed)
    prototype_full_supported = bool(prototype_full is not None and prototype_full.get("full_dpm_execution_supported", False))
    direct_error = None if direct_fixed is None else direct_fixed.get("fixed_q", {}).get("correct_prototype_error")
    prototype_error = None if prototype_fixed is None else prototype_fixed.get("fixed_q", {}).get("correct_prototype_error")
    residual_beats_direct = bool(
        direct_error is not None and prototype_error is not None and float(prototype_error) < float(direct_error)
    )
    representation_supported = False
    if representation is not None:
        style = representation.get("style_probe", {}).get("balanced_accuracy")
        majority = representation.get("majority_class_accuracy")
        representation_supported = style is not None and majority is not None and float(style) > float(majority)
    fcl_attempted = fcl_full is not None
    lane_recommended = bool(
        fcl_full is not None and bool(fcl_full.get("full_dpm_execution_supported", False))
        and bool(fcl_full.get("lane_direction_preserved", False))
    )
    summary = {
        "schema_version": SCHEMA,
        "research_statement": "FC first: same-state residual prototypes edit frozen single-stream DPM clean predictions.",
        "direct_embedding_execution_supported": direct_supported,
        "residual_sample_code_supported": sample_supported,
        "prototype_execution_supported": prototype_supported,
        "residual_beats_direct_embedding": residual_beats_direct,
        "fc_full_dpm_supported": bool(prototype_supported and prototype_full_supported),
        "fcl_attempted": fcl_attempted,
        "lane_change_recommended": lane_recommended,
        "representation_supported_over_majority": representation_supported,
        "label_shuffle_control_present": label_shuffle is not None,
        "evidence": {
            "direct_fixed": args.direct_fixed_report or None,
            "direct_full": args.direct_full_report or None,
            "representation": args.representation_report or None,
            "prototype_fixed": args.prototype_fixed_report or None,
            "prototype_full": args.prototype_full_report or None,
            "label_shuffle": args.label_shuffle_report or None,
            "fcl_full": args.fcl_full_report or None,
        },
        "interpretation": {
            "sample_code": "Whether a training-only residual code improves individual residual reconstruction.",
            "prototype": "Whether class prototypes improve scene-style mean residuals; they are not required to reconstruct every expert future.",
            "deployment": "Uses saved aggr/norm/cons controls and a single DPM stream, never the Residual Encoder or expert future.",
            "warning": "A true flag records only the stated measured criterion; it is not evidence that six independent behavior axes were learned.",
        },
    }
    write_json(output / "summary.json", summary)
    print(f"Summary: {output / 'summary.json'}")


def _add_data_audit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--val-index", required=True)
    parser.add_argument("--mode", choices=("fc", "fcl"), default="fc")
    parser.add_argument("--cons-threshold", type=float, default=0.33)
    parser.add_argument("--aggr-threshold", type=float, default=0.67)
    parser.add_argument("--normal-half-width", type=float, default=0.08)
    parser.add_argument("--dt-seconds", type=float, default=0.1)
    parser.add_argument("--max-cache-audit", type=int, default=0)


def _add_runtime_arguments(parser: argparse.ArgumentParser, *, manifests: bool = True) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("fc", "fcl"), default="fc")
    parser.add_argument("--fc-summary", default="")
    if manifests:
        parser.add_argument("--train-manifest", required=True)
        parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--model-args", default=None)
    parser.add_argument("--normalization-file-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefer-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--q-min", type=float, default=0.15)
    parser.add_argument("--q-max", type=float, default=0.85)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="style-prototype-residual")
    parser.add_argument("--swanlab-run-name", default="")
    parser.add_argument("--swanlab-mode", default="cloud")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    _add_runtime_arguments(parser)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lambda-prototype", type=float, default=0.25)
    parser.add_argument("--lambda-magnitude", type=float, default=0.02)
    parser.add_argument("--lambda-view", type=float, default=0.1)
    parser.add_argument("--prototype-temperature", type=float, default=0.25)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--max-prototype-samples", type=int, default=0)
    parser.add_argument("--postprocess-progress-interval", type=int, default=25)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("data-audit", help="build FC/FCL manifests and audit weak labels")
    _add_data_audit_arguments(audit); audit.set_defaults(handler=command_data_audit)
    same_state = subparsers.add_parser("same-state-contract", help="audit Delta x0 at the same noisy state")
    _add_runtime_arguments(same_state)
    same_state.add_argument("--max-samples", type=int, default=0)
    same_state.set_defaults(handler=command_same_state_contract)
    direct = subparsers.add_parser("train-direct", help="fair direct aggr/norm/cons embedding baseline")
    _add_training_arguments(direct); direct.set_defaults(handler=command_train_direct)
    encoder = subparsers.add_parser("train-encoder", help="training-only residual encoder and prototypes")
    _add_training_arguments(encoder)
    encoder.add_argument("--label-shuffle", action="store_true")
    encoder.set_defaults(handler=command_train_encoder)
    executor = subparsers.add_parser("train-executor", help="sample-code and scene-style prototype executor")
    _add_training_arguments(executor)
    executor.add_argument("--encoder-checkpoint", required=True)
    executor.set_defaults(handler=command_train_executor)
    full = subparsers.add_parser("full-dpm", help="single-stream deployment validation")
    _add_runtime_arguments(full)
    full.add_argument("--execution-checkpoint", required=True)
    full.add_argument("--max-full-dpm-samples", type=int, default=0)
    full.add_argument("--rho-grid", default="-1,0,1")
    full.add_argument("--expected-evaluations", type=int, default=11)
    full.add_argument("--min-response-m", type=float, default=1e-5)
    full.add_argument("--lane-change-min-lateral-m", type=float, default=0.5)
    full.set_defaults(handler=command_full_dpm)
    summary = subparsers.add_parser("summary", help="write explicit scientific-evidence summary")
    summary.add_argument("--output-dir", required=True)
    summary.add_argument("--direct-fixed-report", default="")
    summary.add_argument("--direct-full-report", default="")
    summary.add_argument("--representation-report", default="")
    summary.add_argument("--prototype-fixed-report", default="")
    summary.add_argument("--prototype-full-report", default="")
    summary.add_argument("--label-shuffle-report", default="")
    summary.add_argument("--fcl-full-report", default="")
    summary.set_defaults(handler=command_summary)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

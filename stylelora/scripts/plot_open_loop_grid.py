"""Plot ten deterministic open-loop scenes over the standard nine rho values.

This is a visualization-only entry point.  It reuses the validation cache and
the exact StyleLoRA rollout path, so no training or full open-loop evaluation is
required.  One 3x3 PNG is written per selected scene.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from stylelora.data.encoder_dataset import SCENE_IDS
from stylelora.data.preference_lora_dataset import (
    PreferenceLoRADataset,
    preference_lora_collate,
)
from stylelora.lora.evaluation.rollout import rollout_with_rho
from stylelora.lora.model.checkpoint import load_adapter_checkpoint
from stylelora.lora.model.style_lora_planner import StyleLoRAPlanner
from stylelora.lora.runtime import load_plain_baseline, prepare_diffusion_batch
from stylelora.model.conditional_lora_router import load_conditional_router_checkpoint
from stylelora.paths import ensure_repo_on_path


RHO_GRID = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
SCENE_NAMES = {value: key for key, value in SCENE_IDS.items()}


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _select_scene_indices(
    dataset: PreferenceLoRADataset,
    count: int,
    seed: int,
    scene_type: str,
) -> list[int]:
    """Select reproducible random scenes, balanced or from one scene type."""
    if count <= 0:
        raise ValueError("--num-scenes must be positive")
    rng = np.random.default_rng(seed)
    if scene_type != "balanced":
        candidates = (
            dataset.free_indices
            if scene_type == "straight_free_drive"
            else dataset.car_indices
        )
        if len(candidates) < count:
            raise ValueError(
                f"{scene_type} has only {len(candidates)} scenes, requested {count}"
            )
        return [int(value) for value in rng.permutation(candidates)[:count]]

    groups = [list(dataset.free_indices), list(dataset.car_indices)]
    requested = [count // 2, count - count // 2]
    selected_groups: list[list[int]] = []
    for indices, take in zip(groups, requested):
        if indices:
            order = rng.permutation(indices)
            selected_groups.append(
                [int(value) for value in order[: min(take, len(order))]]
            )
        else:
            selected_groups.append([])
    selected = [
        group[position]
        for position in range(max(map(len, selected_groups), default=0))
        for group in selected_groups
        if position < len(group)
    ]
    if len(selected) < count:
        remaining = sorted(set(range(len(dataset))).difference(selected))
        order = rng.permutation(remaining)
        selected.extend(int(value) for value in order[: count - len(selected)])
    if len(selected) < count:
        raise ValueError(f"dataset has only {len(selected)} selectable scenes, requested {count}")
    return selected[:count]


def _prepare(batch: dict, device: torch.device, observation_normalizer):
    tensors = {key: value.to(device) for key, value in batch["tensors"].items()}
    metadata = [
        {"scene_type": SCENE_NAMES[int(scene_id)]}
        for scene_id in batch["scene_id"].cpu().tolist()
    ]
    return prepare_diffusion_batch(
        {"tensors": tensors, "metadata": metadata},
        device,
        observation_normalizer,
        return_style_context=False,
    )[0]


def _extract_ego_prediction(output: dict) -> np.ndarray:
    prediction = output.get("prediction", output.get("x_start"))
    if prediction is None or prediction.ndim not in (3, 4):
        shape = None if prediction is None else tuple(prediction.shape)
        raise RuntimeError(f"open-loop prediction missing or has unexpected shape {shape}")
    ego = prediction[:, 0] if prediction.ndim == 4 else prediction
    return ego.detach().cpu().numpy()


def _mask_for_polyline(mask: np.ndarray | None, row: int, point_count: int) -> np.ndarray:
    if mask is None:
        return np.ones(point_count, dtype=bool)
    value = np.asarray(mask)
    if value.ndim >= 2:
        result = value[row].reshape(-1).astype(bool)
        if result.size == point_count:
            return result
        return np.full(point_count, bool(result.any()), dtype=bool)
    if value.ndim == 1 and value.size > row:
        return np.full(point_count, bool(value[row]), dtype=bool)
    return np.ones(point_count, dtype=bool)


def _plot_map(ax, tensors: dict, index: int) -> None:
    """Draw cached lane centerlines/boundaries and highlight route lanes."""
    lanes = _numpy(tensors["lanes"][index])
    lane_mask = _numpy(tensors["lanes_mask"][index]) if "lanes_mask" in tensors else None
    for row, lane in enumerate(lanes):
        valid = _mask_for_polyline(lane_mask, row, lane.shape[0])
        valid &= np.isfinite(lane[:, 0]) & np.isfinite(lane[:, 1])
        if valid.sum() < 2:
            continue
        center = lane[valid, :2]
        ax.plot(center[:, 0], center[:, 1], color="#a8adb4", lw=0.55, ls="--", alpha=0.65)
        if lane.shape[1] >= 8:
            left = center + lane[valid, 4:6]
            right = center + lane[valid, 6:8]
            ax.plot(left[:, 0], left[:, 1], color="#c7cbd0", lw=0.45, alpha=0.55)
            ax.plot(right[:, 0], right[:, 1], color="#c7cbd0", lw=0.45, alpha=0.55)

    route = _numpy(tensors["route_lanes"][index])
    route_mask = (
        _numpy(tensors["route_lanes_mask"][index])
        if "route_lanes_mask" in tensors else None
    )
    for row, lane in enumerate(route):
        valid = _mask_for_polyline(route_mask, row, lane.shape[0])
        valid &= np.isfinite(lane[:, 0]) & np.isfinite(lane[:, 1])
        if valid.sum() < 2:
            continue
        center = lane[valid, :2]
        ax.plot(center[:, 0], center[:, 1], color="#3f7fc4", lw=1.35, alpha=0.55)


def _draw_box(ax, state: np.ndarray, *, color: str, alpha: float, zorder: int) -> None:
    if state.size < 2 or not np.isfinite(state[:2]).all():
        return
    x, y = float(state[0]), float(state[1])
    if state.size >= 4:
        heading = math.atan2(float(state[3]), float(state[2]))
    else:
        heading = 0.0
    width = float(state[6]) if state.size > 6 and state[6] > 0 else 1.8
    length = float(state[7]) if state.size > 7 and state[7] > 0 else 4.5
    local = np.asarray(
        [[length / 2, width / 2], [length / 2, -width / 2],
         [-length / 2, -width / 2], [-length / 2, width / 2]],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [[math.cos(heading), -math.sin(heading)],
         [math.sin(heading), math.cos(heading)]],
        dtype=np.float64,
    )
    corners = local @ rotation.T + np.asarray([x, y])
    ax.fill(corners[:, 0], corners[:, 1], facecolor=color, edgecolor=color,
            alpha=alpha, lw=0.6, zorder=zorder)


def _plot_actors(ax, tensors: dict, index: int) -> None:
    past = _numpy(tensors["neighbor_agents_past"][index])
    past_mask = (
        _numpy(tensors["neighbor_agents_past_mask"][index]).astype(bool)
        if "neighbor_agents_past_mask" in tensors else None
    )
    future = _numpy(tensors["neighbors_future_gt"][index])
    future_mask = (
        _numpy(tensors["neighbor_agents_future_mask"][index]).astype(bool)
        if "neighbor_agents_future_mask" in tensors else None
    )
    count = min(past.shape[0], future.shape[0])
    for agent in range(count):
        valid_past = (
            past_mask[agent]
            if past_mask is not None
            else np.linalg.norm(past[agent, :, :2], axis=-1) > 1e-6
        )
        valid_past &= np.isfinite(past[agent, :, :2]).all(axis=1)
        if not valid_past.any():
            continue
        history = past[agent, valid_past, :2]
        ax.plot(history[:, 0], history[:, 1], color="#6f7378", lw=0.75, alpha=0.7)
        current = past[agent, np.flatnonzero(valid_past)[-1]]
        _draw_box(ax, current, color="#62676d", alpha=0.65, zorder=4)
        valid_future = (
            future_mask[agent]
            if future_mask is not None
            else np.linalg.norm(future[agent, :, :2], axis=-1) > 1e-6
        )
        valid_future &= np.isfinite(future[agent, :, :2]).all(axis=1)
        if valid_future.any():
            path = np.concatenate((current[None, :2], future[agent, valid_future, :2]), axis=0)
            ax.plot(path[:, 0], path[:, 1], color="#85898e", lw=0.8, ls=":", alpha=0.72)

    static = _numpy(tensors["static_objects"][index])
    for state in static:
        if state.size < 6 or not np.isfinite(state[:6]).all() or state[4] <= 0 or state[5] <= 0:
            continue
        padded = np.zeros(8, dtype=np.float64)
        padded[:4] = state[:4]
        padded[6], padded[7] = state[4], state[5]
        _draw_box(ax, padded, color="#d58b32", alpha=0.65, zorder=3)

    if "ego_agent_past" in tensors:
        ego_past = _numpy(tensors["ego_agent_past"][index])
        valid = np.isfinite(ego_past[:, :2]).all(axis=1)
        if valid.any():
            ax.plot(ego_past[valid, 0], ego_past[valid, 1], color="#202124", lw=1.0, alpha=0.75)
    ego_state = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 4.8])
    _draw_box(ax, ego_state, color="#111111", alpha=0.9, zorder=7)


def _scene_limits(trajectories: dict[float, np.ndarray], ground_truth: np.ndarray) -> tuple:
    points = [np.asarray(value)[:, :2] for value in trajectories.values()]
    points.append(np.asarray(ground_truth)[:, :2])
    points.append(np.zeros((1, 2), dtype=np.float64))
    values = np.concatenate(points, axis=0)
    values = values[np.isfinite(values).all(axis=1)]
    low, high = values.min(axis=0), values.max(axis=0)
    center = 0.5 * (low + high)
    side = max(float(np.max(high - low)) * 1.24, 22.0)
    half = 0.5 * side
    return (center[0] - half, center[0] + half,
            center[1] - half, center[1] + half)


def _plot_scene(
    *, output: Path, key: str, scene_name: str, tensors: dict, index: int,
    trajectories: dict[float, np.ndarray], dpi: int,
) -> None:
    ground_truth = _numpy(tensors["ego_future_gt"][index])
    limits = _scene_limits(trajectories, ground_truth)
    figure, axes = plt.subplots(3, 3, figsize=(14.2, 13.0), constrained_layout=True)
    color_map = plt.get_cmap("coolwarm")
    baseline = trajectories[0.0]
    for ax, rho in zip(axes.flat, RHO_GRID):
        _plot_map(ax, tensors, index)
        _plot_actors(ax, tensors, index)
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], color="black", lw=1.5,
                ls="--", label="ego ground truth", zorder=8)
        ax.plot(baseline[:, 0], baseline[:, 1], color="#2ca25f", lw=1.55,
                ls=":", label="rho=0 baseline", zorder=9)
        color = color_map((rho + 1.0) / 2.0)
        trajectory = trajectories[rho]
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, lw=2.25,
                label=f"planned rho={rho:+.2f}", zorder=10)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color=[color], s=16, zorder=11)
        ax.set_title(f"rho = {rho:+.2f}", fontsize=11)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.grid(True, lw=0.35, alpha=0.25)
        ax.set_xlabel("longitudinal x (m)")
        ax.set_ylabel("lateral y (m)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(f"{scene_name} | {key}", fontsize=13)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="Plot 10 open-loop scenes as nine-rho grids.")
    parser.add_argument("--args-file", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--adapter-high", required=True)
    parser.add_argument("--adapter-low", required=True)
    parser.add_argument("--conditional-router-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--feature-npy", required=True)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--latent-bank", required=True)
    parser.add_argument("--latent-bank-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument(
        "--scene-type",
        choices=("balanced", "straight_free_drive", "straight_car_follow"),
        default="balanced",
        help="Randomly select one scene type, or use an approximately balanced set.",
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PreferenceLoRADataset(
        args.manifest, args.cache_root, args.latent_bank, args.latent_bank_index,
        args.feature_npy, args.feature_index, direction="high", rank_low=0.0, rank_high=1.0,
    )
    indices = _select_scene_indices(
        dataset, args.num_scenes, args.seed, args.scene_type
    )
    batch = preference_lora_collate([dataset[index] for index in indices])

    baseline, config = load_plain_baseline(
        args.args_file, args.baseline_checkpoint, args.device
    )
    planner = StyleLoRAPlanner(
        baseline, rank=args.rank, alpha=args.alpha, dropout=0.0
    ).to(device)
    for checkpoint in (args.adapter_high, args.adapter_low):
        load_adapter_checkpoint(
            checkpoint, planner, baseline_checkpoint=args.baseline_checkpoint,
            normalization_file=config.normalization_file_path, strict_hash=True,
        )
    router, prototypes, _ = load_conditional_router_checkpoint(
        args.conditional_router_checkpoint, device
    )
    planner.attach_conditional_router(router, prototypes, enabled=True, trainable=False)
    planner.eval()
    model_inputs = _prepare(batch, device, config.observation_normalizer)

    predictions: dict[float, np.ndarray] = {}
    for rho in RHO_GRID:
        print(f"[rollout] rho={rho:+.2f}", flush=True)
        with torch.inference_mode():
            output, _ = rollout_with_rho(planner, model_inputs, rho, seed=args.seed)
        predictions[rho] = _extract_ego_prediction(output)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_records = []
    for plot_index, dataset_index in enumerate(indices, start=1):
        scene_id = int(batch["scene_id"][plot_index - 1])
        scene_name = SCENE_NAMES[scene_id]
        key = str(batch["key"][plot_index - 1])
        output = output_dir / f"scene_{plot_index:02d}_{scene_name}.png"
        trajectories = {
            rho: prediction[plot_index - 1] for rho, prediction in predictions.items()
        }
        _plot_scene(
            output=output, key=key, scene_name=scene_name,
            tensors=batch["tensors"], index=plot_index - 1,
            trajectories=trajectories, dpi=args.dpi,
        )
        selected_records.append({
            "figure": output.name,
            "key": key,
            "scene_type": scene_name,
            "dataset_index": int(dataset_index),
        })
        print(f"[saved {plot_index}/{len(indices)}] {output}", flush=True)

    with (output_dir / "selected_scenes.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"seed": args.seed, "rho_grid": list(RHO_GRID), "scenes": selected_records},
            handle, ensure_ascii=False, indent=2,
        )
    print(f"[done] wrote {len(selected_records)} figures to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

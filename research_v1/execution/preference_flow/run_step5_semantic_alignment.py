"""Step 5-S1: identity-anchored, five-point Preference Flow alignment.

This is a fixed 12-scene proof, not Step 6.  The base StylePlanner remains
frozen and the Flow starts fresh.  Neutral x0 fixes both the content trajectory
and the interaction reference; all rho branches share xq, noise, lead ID and
valid time masks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as functional

from baseline.model.style_planner.preference_flow import CLEAN_PREDICTION_EDITOR_DISABLED
from research_v1.execution.preference_flow.differentiable_behavior_alignment import (
    DifferentiableBehaviorBridge,
    FrozenBehaviorCalibration,
    NeutralAnchoredLeadReference,
    existing_formal_metric_for_trajectory,
    legacy_min_distance_proxy,
    longitudinal_perturbation,
    pathwise_order_loss,
    trajectory_tangents,
)
from research_v1.execution.preference_flow.run_step1_base_regression import (
    _build_model_args,
    _checkpoint_state,
    _existing_file,
    _load_frozen_styleplanner,
    _model_args_path,
    _write_report,
)
from research_v1.execution.preference_flow.run_step5_tiny_preference_learning import (
    _SwanLabMonitor,
    _absolute_statistics,
    _adapter_snapshot,
    _all_finite_gradients,
    _attach_fixed_phases,
    _base_change,
    _base_snapshot,
    _batch,
    _cohort_entries,
    _grad_norm,
    _load_sample,
    _max_abs,
    _new_adapter,
    _noisy_state,
    _neutral_clean_prediction,
    _physical_ego_future,
    _prepare_batch,
    _seed_everything,
    _speed_proxy,
)


_RHO_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
_SCENES = ("straight_free_drive", "straight_car_follow")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Step-5-S1 identity-anchored pathwise Preference Flow alignment."
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--model-args", default=None)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-file", required=True)
    parser.add_argument("--behavior-normalization-path", required=True)
    parser.add_argument("--normalization-file-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefer-ema", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--lambda-content", type=float, default=0.02)
    parser.add_argument("--lambda-smooth", type=float, default=0.02)
    parser.add_argument("--path-margin-per-rho", type=float, default=0.02)
    parser.add_argument("--positive-alignment-span", type=float, default=0.15)
    parser.add_argument("--negative-alignment-span", type=float, default=0.15)
    parser.add_argument("--flow-checkpoint-name", default="step5_semantic_alignment_flow.pt")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="preference-flow-step5")
    parser.add_argument("--swanlab-run-name", default="semantic-alignment")
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline"), default="online")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _masked_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _physical_grid(outputs: Sequence[Any], model_args: Any) -> torch.Tensor:
    return torch.stack([_physical_ego_future(output.clean_prediction, model_args) for output in outputs])


def _semantic_context(
    *,
    model: torch.nn.Module,
    raw: Mapping[str, Any],
    model_args: Any,
    selector: NeutralAnchoredLeadReference,
    bridge: DifferentiableBehaviorBridge,
) -> Dict[str, Any]:
    inputs, all_gt, ego_future, neighbors_future, neighbor_mask = _prepare_batch(raw, model_args)
    xq, log_snr = _noisy_state(model, all_gt, raw["diffusion_time"], raw["noise"])
    neutral = _neutral_clean_prediction(model, inputs, xq, raw["diffusion_time"])
    neutral_future = _physical_ego_future(neutral, model_args)
    raw_inputs = raw["inputs"]
    predicted_neighbors = int(model_args.predicted_neighbor_num)
    ego_current = raw_inputs["ego_current_state"][:, :4]
    neighbor_history = raw_inputs["neighbor_agents_past"][:, :predicted_neighbors]
    reference = selector.build(
        neutral_future=neutral_future,
        ego_current_state=ego_current,
        neighbor_history=neighbor_history,
    )
    speed_limit, speed_limit_valid = bridge.speed_limit_from_inputs(
        raw_inputs["route_lanes_speed_limit"],
        raw_inputs["route_lanes_has_speed_limit"],
        raw_inputs["lanes_speed_limit"],
        raw_inputs["lanes_has_speed_limit"],
    )
    task_features = raw["task_features"]
    return {
        "inputs": inputs,
        "all_gt": all_gt,
        "ego_future": ego_future,
        "neighbors_future": neighbors_future,
        "neighbor_mask": neighbor_mask,
        "xq": xq,
        "log_snr": log_snr,
        "neutral": neutral,
        "neutral_future": neutral_future,
        "ego_current": ego_current,
        "reference": reference,
        "speed_limit": speed_limit,
        "speed_limit_valid": speed_limit_valid,
        "free_mask": task_features[:, 0] > 0.5,
        "car_mask": task_features[:, 1] > 0.5,
    }


def _grid_outputs(
    *,
    adapter: torch.nn.Module,
    context: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> Tuple[List[Any], torch.Tensor]:
    grid = torch.tensor(_RHO_GRID, device=raw["rho"].device, dtype=raw["rho"].dtype)
    current = context["xq"].reshape(int(context["xq"].shape[0]), int(context["xq"].shape[1]), -1)
    outputs = [
        adapter(
            context["neutral"], current, raw["diffusion_time"], context["log_snr"], raw["task_features"],
            torch.full_like(raw["rho"], float(rho)),
        )
        for rho in _RHO_GRID
    ]
    return outputs, grid


def _legacy_expert_fit(
    *,
    grid_future: torch.Tensor,
    raw: Mapping[str, Any],
    context: Mapping[str, Any],
) -> torch.Tensor:
    """Old future-target proxy remains a diagnostic, not a training objective."""

    grid = torch.tensor(_RHO_GRID, device=raw["rho"].device, dtype=raw["rho"].dtype)
    selected_index = (raw["rho"][:, None] - grid[None, :]).abs().argmin(dim=1)
    batch = torch.arange(raw["rho"].shape[0], device=raw["rho"].device)
    selected = grid_future.permute(1, 0, 2, 3)[batch, selected_index]
    free_mask, car_mask = context["free_mask"], context["car_mask"]
    terms = []
    if bool(free_mask.any().item()):
        predicted = _speed_proxy(context["ego_current"][free_mask], selected[free_mask])
        target = _speed_proxy(context["ego_current"][free_mask], context["ego_future"][free_mask])
        terms.append(functional.smooth_l1_loss((predicted - target) / 0.5, torch.zeros_like(predicted)))
    if bool(car_mask.any().item()):
        predicted, _, _, valid = legacy_min_distance_proxy(
            selected[car_mask], context["neighbors_future"][car_mask], context["neighbor_mask"][car_mask]
        )
        target, _, _, _ = legacy_min_distance_proxy(
            context["ego_future"][car_mask], context["neighbors_future"][car_mask], context["neighbor_mask"][car_mask]
        )
        if bool(valid.any().item()):
            terms.append(functional.smooth_l1_loss((predicted[valid] - target[valid]) / 2.0, torch.zeros_like(predicted[valid])))
    return torch.stack(terms).mean() if terms else grid_future.new_zeros(())


def _semantic_losses(
    *,
    measurements: Sequence[Any],
    grid_future: torch.Tensor,
    context: Mapping[str, Any],
    raw: Mapping[str, Any],
    bridge: DifferentiableBehaviorBridge,
    rho_grid: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    neutral_measurement = measurements[2]
    scores = torch.stack([measurement.score for measurement in measurements])
    offline_scores = torch.stack([measurement.offline_score for measurement in measurements])
    valid = neutral_measurement.score_valid
    targets, target_valid = zip(
        *[
            bridge.desired_score(
                neutral_measurement,
                torch.full_like(raw["rho"], float(rho)),
                free_mask=context["free_mask"],
                car_mask=context["car_mask"],
                positive_span=float(args.positive_alignment_span),
                negative_span=float(args.negative_alignment_span),
            )
            for rho in _RHO_GRID
        ]
    )
    target_scores = torch.stack(list(targets))
    semantic_valid = torch.stack(list(target_valid)) & valid[None, :]
    alignment_values = functional.smooth_l1_loss(scores, target_scores, reduction="none")
    neutral_relative = _masked_average(alignment_values, semantic_valid)
    free_valid = semantic_valid & context["free_mask"][None, :]
    car_valid = semantic_valid & context["car_mask"][None, :]
    free_alignment = _masked_average(alignment_values, free_valid)
    car_alignment = _masked_average(alignment_values, car_valid)
    pathwise, deltas = pathwise_order_loss(
        scores, valid, rho_grid, margin_per_rho=float(args.path_margin_per_rho)
    )
    free_pathwise, _ = pathwise_order_loss(
        scores[:, context["free_mask"]], valid[context["free_mask"]], rho_grid,
        margin_per_rho=float(args.path_margin_per_rho),
    )
    car_pathwise, _ = pathwise_order_loss(
        scores[:, context["car_mask"]], valid[context["car_mask"]], rho_grid,
        margin_per_rho=float(args.path_margin_per_rho),
    )
    residual = grid_future - context["neutral_future"][None]
    content = residual[..., :2].square().mean()
    smooth = (
        (residual[:, :, 2:, :2] - 2.0 * residual[:, :, 1:-1, :2] + residual[:, :, :-2, :2])
        .square()
        .mean()
        if grid_future.shape[2] >= 3
        else residual.new_zeros(())
    )
    legacy = _legacy_expert_fit(grid_future=grid_future, raw=raw, context=context)
    return {
        "neutral_relative_alignment_loss": neutral_relative,
        "pathwise_order_loss": pathwise,
        "legacy_expert_target_fit": legacy,
        "free_neutral_relative_alignment_loss": free_alignment,
        "car_neutral_relative_alignment_loss": car_alignment,
        "free_pathwise_order_loss": free_pathwise,
        "car_pathwise_order_loss": car_pathwise,
        "content": content,
        "smooth": smooth,
        "scores": scores,
        "offline_scores": offline_scores,
        "score_valid": valid,
        "pathwise_deltas": deltas,
    }


def _forward(
    *,
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    raw: Mapping[str, Any],
    model_args: Any,
    selector: NeutralAnchoredLeadReference,
    bridge: DifferentiableBehaviorBridge,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    context = _semantic_context(
        model=model, raw=raw, model_args=model_args, selector=selector, bridge=bridge
    )
    outputs, rho_grid = _grid_outputs(adapter=adapter, context=context, raw=raw)
    grid_future = _physical_grid(outputs, model_args)
    measurements = [
        bridge.measure(
            edited_future=future,
            neutral_future=context["neutral_future"],
            ego_current_state=context["ego_current"],
            neighbor_future=context["neighbors_future"],
            neighbor_mask=context["neighbor_mask"],
            lead_reference=context["reference"],
            speed_limit_mps=context["speed_limit"],
            speed_limit_valid=context["speed_limit_valid"],
            free_mask=context["free_mask"],
            car_mask=context["car_mask"],
        )
        for future in grid_future
    ]
    losses = _semantic_losses(
        measurements=measurements,
        grid_future=grid_future,
        context=context,
        raw=raw,
        bridge=bridge,
        rho_grid=rho_grid,
        args=args,
    )
    total = (
        losses["neutral_relative_alignment_loss"]
        + losses["pathwise_order_loss"]
        + float(args.lambda_content) * losses["content"]
        + float(args.lambda_smooth) * losses["smooth"]
    )
    return {
        "total": total,
        "losses": losses,
        "context": context,
        "outputs": outputs,
        "grid_future": grid_future,
        "measurements": measurements,
        "rho_grid": rho_grid,
    }


def _physical_residual_summary(result: Mapping[str, Any]) -> Dict[str, float]:
    context, future = result["context"], result["grid_future"]
    residuals, lateral = [], []
    for index, rho in enumerate(_RHO_GRID):
        if rho == 0.0:
            continue
        tangent = trajectory_tangents(context["ego_current"], context["neutral_future"])
        xy = future[index, ..., :2] - context["neutral_future"][..., :2]
        residuals.append((xy * tangent).sum(dim=-1))
        lateral.append(xy[..., 0] * tangent[..., 1] - xy[..., 1] * tangent[..., 0])
    summary = _absolute_statistics(torch.cat(residuals, dim=0))
    return {
        "physical_longitudinal_residual_abs_mean": summary["mean"],
        "physical_longitudinal_residual_abs_p95": summary["p95"],
        "physical_longitudinal_residual_abs_max": summary["max"],
        "max_lateral_residual": _max_abs(torch.cat(lateral, dim=0)),
    }


def _result_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    losses, context, outputs = result["losses"], result["context"], result["outputs"]
    zero = outputs[2].clean_prediction
    neutral = context["neutral"]
    direct = [output.clean_prediction - neutral for index, output in enumerate(outputs) if index != 2]
    return {
        key: float(losses[key].detach().item())
        for key in (
            "neutral_relative_alignment_loss", "pathwise_order_loss", "legacy_expert_target_fit",
            "free_neutral_relative_alignment_loss", "car_neutral_relative_alignment_loss",
            "free_pathwise_order_loss", "car_pathwise_order_loss", "content", "smooth",
        )
    } | {
        "total": float(result["total"].detach().item()),
        "rho_zero_exact": bool(torch.equal(zero, neutral)),
        "score_valid_count": int(losses["score_valid"].sum().item()),
        "max_latent_displacement": max(
            _max_abs(output.latent_end - output.latent_start) for output in outputs
        ),
        "max_ego_future_residual": max(_max_abs(output.ego_future_residual) for output in outputs),
        "max_non_ego_direct_residual": max(_max_abs(item[:, 1:, :]) for item in direct),
        "max_ego_current_direct_residual": max(_max_abs(item[:, 0, :4]) for item in direct),
        "all_outputs_finite": all(bool(torch.isfinite(output.clean_prediction).all().item()) for output in outputs),
        **_physical_residual_summary(result),
    }


def _gradient_vector(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def _gradient_alignment(result: Mapping[str, Any], adapter: torch.nn.Module) -> Dict[str, Any]:
    parameters = list(adapter.vector_field.parameters())
    free = result["losses"]["free_neutral_relative_alignment_loss"] + result["losses"]["free_pathwise_order_loss"]
    car = result["losses"]["car_neutral_relative_alignment_loss"] + result["losses"]["car_pathwise_order_loss"]
    free_vector = _gradient_vector(free, parameters)
    car_vector = _gradient_vector(car, parameters)
    free_norm = float(torch.linalg.vector_norm(free_vector).detach().cpu().item())
    car_norm = float(torch.linalg.vector_norm(car_vector).detach().cpu().item())
    cosine = None
    if free_norm > 0.0 and car_norm > 0.0:
        cosine = float((torch.dot(free_vector, car_vector) / (free_norm * car_norm)).detach().cpu().item())
    return {
        "vector_field_free_grad_norm": free_norm,
        "vector_field_car_grad_norm": car_norm,
        "vector_field_free_car_cosine": cosine,
        "gradient_conflict": bool(cosine is not None and cosine < 0.0),
    }


def _existing_formal_sweep(
    grid_future: torch.Tensor,
    samples: Sequence[Mapping[str, Any]],
    calibration: FrozenBehaviorCalibration,
) -> List[Dict[str, Any]]:
    """Run the unchanged hard metric implementation after optimization only."""

    rows: List[Dict[str, Any]] = []
    with TemporaryDirectory(prefix="preference_flow_step5_s1_") as temporary:
        root = Path(temporary)
        for sample_index, sample in enumerate(samples):
            values, valid, raw_values, axis_valid, sources, failures = [], [], [], [], [], []
            for rho_index, _rho in enumerate(_RHO_GRID):
                formal = existing_formal_metric_for_trajectory(
                    cache_path=str(sample["cache_path"]),
                    temporary_cache_path=str(root / f"scene_{sample_index:02d}_{rho_index:02d}.npz"),
                    scene=str(sample["scene"]), ego_future=grid_future[rho_index, sample_index],
                    calibration=calibration,
                )
                values.append(float(formal.score))
                valid.append(bool(formal.valid))
                raw_values.append(list(formal.raw_values))
                axis_valid.append(list(formal.axis_valid_mask))
                sources.append(str(formal.metric_source))
                failures.append(formal.failure)
            rows.append({
                "values": values, "valid": valid, "raw_values": raw_values,
                "axis_valid_masks": axis_valid, "metric_sources": sources,
                "failures": failures,
            })
    return rows


def _direction_report(
    result: Mapping[str, Any], samples: Sequence[Mapping[str, Any]], formal_sweep: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    losses, context, outputs = result["losses"], result["context"], result["outputs"]
    scores = losses["scores"].detach().cpu()
    hard_reference = losses["offline_scores"].detach().cpu()
    valid = losses["score_valid"].detach().cpu()
    if len(formal_sweep) != len(samples):
        raise RuntimeError("formal sweep/sample count mismatch")
    rows = []
    rho_zero_exact = torch.equal(outputs[2].clean_prediction, context["neutral"])
    for index, sample in enumerate(samples):
        proxy = [float(value) for value in scores[:, index].tolist()]
        fixed_lead_hard = [float(value) for value in hard_reference[:, index].tolist()]
        formal_record = formal_sweep[index]
        formal = [float(value) for value in formal_record["values"]]
        formal_valid = [bool(value) for value in formal_record["valid"]]
        proxy_deltas = [proxy[item + 1] - proxy[item] for item in range(len(proxy) - 1)]
        fixed_lead_hard_deltas = [
            fixed_lead_hard[item + 1] - fixed_lead_hard[item]
            for item in range(len(fixed_lead_hard) - 1)
        ]
        formal_deltas = [formal[item + 1] - formal[item] for item in range(len(formal) - 1)]
        interval_valid = [
            bool(valid[index].item()) and formal_valid[item] and formal_valid[item + 1]
            for item in range(len(proxy_deltas))
        ]
        agreement = [
            bool(is_valid and ((left > 0.0) == (right > 0.0)))
            for left, right, is_valid in zip(proxy_deltas, formal_deltas, interval_valid)
        ]
        reference = context["reference"]
        rows.append({
            "filename": str(sample["filename"]),
            "scene": str(sample["scene"]),
            "lead_index_neutral": int(reference.lead_indices[index].detach().cpu().item()),
            "lead_valid_step_count": int(reference.valid_time_mask[index].sum().detach().cpu().item()),
            "score_valid": bool(valid[index].item()),
            "rho_grid": list(_RHO_GRID),
            "differentiable_proxy": proxy,
            "fixed_lead_hard_reference": fixed_lead_hard,
            "existing_formal_metric": formal,
            "existing_formal_metric_valid": formal_valid,
            "existing_formal_metric_raw_values": formal_record["raw_values"],
            "existing_formal_metric_axis_valid_masks": formal_record["axis_valid_masks"],
            "existing_formal_metric_sources": formal_record["metric_sources"],
            "existing_formal_metric_failures": formal_record["failures"],
            "proxy_adjacent_margins": proxy_deltas,
            "fixed_lead_hard_adjacent_margins": fixed_lead_hard_deltas,
            "offline_adjacent_margins": formal_deltas,
            "proxy_path_passed": bool(valid[index].item() and all(value > 0.0 for value in proxy_deltas)),
            "offline_path_passed": bool(all(interval_valid) and all(value > 0.0 for value in formal_deltas)),
            "proxy_offline_direction_agreement_rate": sum(agreement) / len(agreement),
            "proxy_offline_direction_consistent": bool(all(agreement)),
        })
    grouped = {}
    for scene in _SCENES:
        scene_rows = [row for row in rows if row["scene"] == scene]
        grouped[scene] = {
            "scene_count": len(scene_rows),
            "path_passed_count": sum(bool(row["proxy_path_passed"]) for row in scene_rows),
            "offline_path_passed_count": sum(bool(row["offline_path_passed"]) for row in scene_rows),
            "proxy_offline_direction_consistent_count": sum(
                bool(row["proxy_offline_direction_consistent"]) for row in scene_rows
            ),
            "mean_proxy_offline_direction_agreement": (
                sum(float(row["proxy_offline_direction_agreement_rate"]) for row in scene_rows) / len(scene_rows)
                if scene_rows else 0.0
            ),
        }
    passed_count = sum(bool(row["proxy_path_passed"]) for row in rows)
    return {
        "schema_version": "preference_flow_step5_s1_all_scene_direction_v1",
        "rho_grid": list(_RHO_GRID),
        "scene_count": len(rows),
        "path_passed_count": passed_count,
        "path_pass_rate": passed_count / float(len(rows)),
        "rho_zero_exact": bool(rho_zero_exact),
        "by_scene": grouped,
        "scenes": rows,
        "passed": bool(
            passed_count == len(rows)
            and all(grouped[scene]["path_passed_count"] == grouped[scene]["scene_count"] for scene in _SCENES)
            and all(grouped[scene]["offline_path_passed_count"] == grouped[scene]["scene_count"] for scene in _SCENES)
            and all(
                grouped[scene]["proxy_offline_direction_consistent_count"] == grouped[scene]["scene_count"]
                for scene in _SCENES
            )
            and rho_zero_exact
        ),
    }


def _proxy_sensitivity_audit(result: Mapping[str, Any], samples: Sequence[Mapping[str, Any]], bridge: DifferentiableBehaviorBridge) -> Dict[str, Any]:
    context = result["context"]
    neutral = context["neutral_future"]
    legacy, min_neighbor, min_time, min_valid = legacy_min_distance_proxy(
        neutral, context["neighbors_future"], context["neighbor_mask"]
    )
    rows = []
    key = lambda meters: f"{float(meters):+.2f}"
    for index, sample in enumerate(samples):
        if sample["scene"] != "straight_car_follow":
            continue
        perturbations = {}
        for meters in (-0.20, -0.05, 0.05, 0.20):
            edited = longitudinal_perturbation(neutral, context["ego_current"], neutral, meters)
            measurement = bridge.measure(
                edited_future=edited,
                neutral_future=neutral,
                ego_current_state=context["ego_current"],
                neighbor_future=context["neighbors_future"],
                neighbor_mask=context["neighbor_mask"],
                lead_reference=context["reference"],
                speed_limit_mps=context["speed_limit"],
                speed_limit_valid=context["speed_limit_valid"],
                free_mask=context["free_mask"],
                car_mask=context["car_mask"],
            )
            min_value, _, _, _ = legacy_min_distance_proxy(edited, context["neighbors_future"], context["neighbor_mask"])
            perturbations[key(meters)] = {
                "legacy_min_distance_proxy": float(min_value[index].detach().cpu().item()),
                "headway_raw_s": float(measurement.raw_headway[index].detach().cpu().item()),
                "headway_valid": bool(measurement.headway_valid[index].item()),
                "ttc_raw_s": float(measurement.raw_ttc[index].detach().cpu().item()),
                "ttc_valid": bool(measurement.ttc_valid[index].item()),
            }
        negative, positive = perturbations[key(-0.20)], perturbations[key(0.20)]
        base = perturbations[key(0.05)]
        rows.append({
            "filename": str(sample["filename"]),
            "legacy_min_distance_neighbor_index": int(min_neighbor[index].detach().cpu().item()),
            "legacy_min_distance_time_index": int(min_time[index].detach().cpu().item()),
            "legacy_min_distance_valid": bool(min_valid[index].item()),
            "neutral_anchored_lead_index": int(context["reference"].lead_indices[index].detach().cpu().item()),
            "neutral_anchored_valid_step_count": int(context["reference"].valid_time_mask[index].sum().detach().cpu().item()),
            "same_neighbor_as_legacy_min": bool(
                context["reference"].lead_indices[index].detach().cpu().item() == min_neighbor[index].detach().cpu().item()
            ),
            "neutral_legacy_min_distance_proxy": float(legacy[index].detach().cpu().item()),
            "perturbations": perturbations,
            "sensitivity_detected": {
                "legacy_min_distance": abs(positive["legacy_min_distance_proxy"] - negative["legacy_min_distance_proxy"]) > 1e-6,
                "headway": abs(positive["headway_raw_s"] - negative["headway_raw_s"]) > 1e-6,
                "ttc": bool(base["ttc_valid"]) and abs(positive["ttc_raw_s"] - negative["ttc_raw_s"]) > 1e-6,
            },
        })
    return {
        "schema_version": "preference_flow_step5_s1_proxy_sensitivity_audit_v1",
        "perturbations_m": [-0.20, -0.05, 0.05, 0.20],
        "car_follow_scene_count": len(rows),
        "lead_matches_legacy_min_count": sum(bool(row["same_neighbor_as_legacy_min"]) for row in rows),
        "sensitivity_counts": {
            key: sum(bool(row["sensitivity_detected"][key]) for row in rows)
            for key in ("legacy_min_distance", "headway", "ttc")
        },
        "scenes": rows,
    }


def _check_args(args: argparse.Namespace) -> None:
    if int(args.steps) <= 0 or int(args.batch_size) <= 0 or int(args.log_interval) <= 0:
        raise ValueError("--steps, --batch-size, and --log-interval must be positive")
    if float(args.learning_rate) <= 0.0 or float(args.path_margin_per_rho) <= 0.0:
        raise ValueError("learning rate and path margin must be positive")
    if float(args.positive_alignment_span) <= 0.0 or float(args.negative_alignment_span) <= 0.0:
        raise ValueError("alignment spans must be positive")


def run(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, Path]:
    _check_args(args)
    _seed_everything(int(args.seed))
    checkpoint_path = _existing_file(args.base_checkpoint, "--base-checkpoint")
    cache_root = Path(args.cache_root).expanduser().resolve()
    cohort_path = _existing_file(args.cohort_file, "--cohort-file")
    calibration_path = _existing_file(args.behavior_normalization_path, "--behavior-normalization-path")
    if not cache_root.is_dir():
        raise NotADirectoryError(f"--cache-root is not a directory: {cache_root}")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    args_path = _model_args_path(checkpoint_path, args.model_args)
    model_args = _build_model_args(
        args_path, mode=CLEAN_PREDICTION_EDITOR_DISABLED, device=str(device),
        normalization_file_override=args.normalization_file_path,
    )
    if str(getattr(model_args, "diffusion_model_type", "")) != "x_start":
        raise RuntimeError("Step-5-S1 requires an x_start StylePlanner checkpoint")
    state, checkpoint_meta = _checkpoint_state(checkpoint_path, prefer_ema=bool(args.prefer_ema))
    model, model_meta = _load_frozen_styleplanner(model_args, state, checkpoint_meta)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    frozen_snapshot = _base_snapshot(model)

    entries = _cohort_entries(cohort_path, cache_root)
    samples = [
        {**_load_sample(entry), "cache_path": str(entry["cache_path"])}
        for entry in entries
    ]
    scene_counts = {scene: sum(item["scene"] == scene for item in samples) for scene in _SCENES}
    if len(samples) != 12 or any(scene_counts[scene] != 6 for scene in _SCENES):
        raise ValueError("Step-5-S1 requires exactly six free-drive and six car-follow scenes")
    if int(args.batch_size) != len(samples):
        raise ValueError(f"Step-5-S1 uses the complete deterministic cohort; set --batch-size {len(samples)}")
    _attach_fixed_phases(
        samples, predicted_neighbors=int(model_args.predicted_neighbor_num),
        future_len=int(model_args.future_len), seed=int(args.seed),
    )
    calibration = FrozenBehaviorCalibration.from_json(str(calibration_path))
    selector = NeutralAnchoredLeadReference()
    bridge = DifferentiableBehaviorBridge(calibration)
    base_dtype = next(model.parameters()).dtype
    adapter = _new_adapter(model_args, device, base_dtype)
    flow_parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(flow_parameters, lr=float(args.learning_rate))
    base_ids = {id(parameter) for parameter in model.parameters()}
    if any(id(parameter) in base_ids for group in optimizer.param_groups for parameter in group["params"]):
        raise RuntimeError("frozen base planner leaked into the Flow optimizer")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "step5_proxy_sensitivity_audit.json"
    pathwise_path = output_dir / "step5_pathwise_alignment.json"
    direction_path = output_dir / "step5_all_scene_direction.json"
    gradient_path = output_dir / "step5_gradient_alignment_history.json"
    checkpoint_output = output_dir / str(args.flow_checkpoint_name)
    if checkpoint_output.exists() and not bool(args.overwrite):
        raise FileExistsError(f"refusing to overwrite Flow checkpoint: {checkpoint_output}")
    monitor = _SwanLabMonitor(
        args,
        {"seed": int(args.seed), "cohort_size": len(samples), "rho_grid": list(_RHO_GRID),
         "calibration": str(calibration_path), "base_checkpoint": str(checkpoint_path)},
        output_dir / "swanlog",
    )
    try:
        full_raw = _batch(samples, list(range(len(samples))), device)
        adapter.eval()
        with torch.no_grad():
            initial_result = _forward(
                model=model, adapter=adapter, raw=full_raw, model_args=model_args,
                selector=selector, bridge=bridge, args=args,
            )
        initial = _result_summary(initial_result)
        audit = _proxy_sensitivity_audit(initial_result, samples, bridge)
        _write_report(audit_path, audit, overwrite=bool(args.overwrite))

        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        bootstrap = _forward(
            model=model, adapter=adapter, raw=full_raw, model_args=model_args,
            selector=selector, bridge=bridge, args=args,
        )
        bootstrap["total"].backward()
        bootstrap_report = {
            "base_grad_norm": _grad_norm(model.parameters()),
            "vector_field_grad_norm": _grad_norm(adapter.vector_field.parameters()),
            "all_trainable_gradients_finite": _all_finite_gradients(flow_parameters),
            "passed": bool(_grad_norm(model.parameters()) == 0.0 and _grad_norm(adapter.vector_field.parameters()) > 0.0 and _all_finite_gradients(flow_parameters)),
        }
        optimizer.zero_grad(set_to_none=True)
        if not bootstrap_report["passed"]:
            raise AssertionError("Step-5-S1 gradient bootstrap failed")

        adapter.eval()
        initial_gradient_result = _forward(
            model=model, adapter=adapter, raw=full_raw, model_args=model_args,
            selector=selector, bridge=bridge, args=args,
        )
        gradient_history = {"initial": _gradient_alignment(initial_gradient_result, adapter)}
        best_state = _adapter_snapshot(adapter)
        best_total = float(initial["total"])
        best_step = 0
        history: List[Dict[str, float]] = []
        midpoint = max(int(args.steps) // 2, 1)
        for step in range(1, int(args.steps) + 1):
            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            result = _forward(
                model=model, adapter=adapter, raw=full_raw, model_args=model_args,
                selector=selector, bridge=bridge, args=args,
            )
            current_total = float(result["total"].detach().item())
            if current_total < best_total:
                best_total, best_step, best_state = current_total, step - 1, _adapter_snapshot(adapter)
            result["total"].backward()
            torch.nn.utils.clip_grad_norm_(flow_parameters, max_norm=10.0)
            optimizer.step()
            point = {
                "step": float(step), "total": current_total,
                "neutral_relative_alignment_loss": float(result["losses"]["neutral_relative_alignment_loss"].detach().item()),
                "pathwise_order_loss": float(result["losses"]["pathwise_order_loss"].detach().item()),
                "free_neutral_relative_alignment_loss": float(result["losses"]["free_neutral_relative_alignment_loss"].detach().item()),
                "car_neutral_relative_alignment_loss": float(result["losses"]["car_neutral_relative_alignment_loss"].detach().item()),
                "free_pathwise_order_loss": float(result["losses"]["free_pathwise_order_loss"].detach().item()),
                "car_pathwise_order_loss": float(result["losses"]["car_pathwise_order_loss"].detach().item()),
                "legacy_expert_target_fit": float(result["losses"]["legacy_expert_target_fit"].detach().item()),
                "content": float(result["losses"]["content"].detach().item()),
                "smooth": float(result["losses"]["smooth"].detach().item()),
            }
            history.append(point)
            if step == midpoint:
                adapter.eval()
                mid_result = _forward(
                    model=model, adapter=adapter, raw=full_raw, model_args=model_args,
                    selector=selector, bridge=bridge, args=args,
                )
                gradient_history["midpoint"] = _gradient_alignment(mid_result, adapter)
            if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
                monitor.log({f"train/{key}": value for key, value in point.items()}, step)

        adapter.eval()
        with torch.no_grad():
            terminal_result = _forward(
                model=model, adapter=adapter, raw=full_raw, model_args=model_args,
                selector=selector, bridge=bridge, args=args,
            )
        terminal = _result_summary(terminal_result)
        if float(terminal["total"]) < best_total:
            best_total, best_step, best_state = float(terminal["total"]), int(args.steps), _adapter_snapshot(adapter)
        adapter.load_state_dict(best_state, strict=True)
        adapter.eval()
        with torch.no_grad():
            final_result = _forward(
                model=model, adapter=adapter, raw=full_raw, model_args=model_args,
                selector=selector, bridge=bridge, args=args,
            )
        final = _result_summary(final_result)
        gradient_history["final"] = _gradient_alignment(
            _forward(model=model, adapter=adapter, raw=full_raw, model_args=model_args,
                     selector=selector, bridge=bridge, args=args), adapter
        )
        gradient_history["optimizer_modified_for_conflict"] = False

        torch.save({
            "schema_version": "preference_flow_step5_s1_checkpoint_v1",
            "flow_config": asdict(adapter.config), "state_dict": adapter.state_dict(),
            "base_checkpoint": str(checkpoint_path), "behavior_normalization_path": str(calibration_path),
        }, checkpoint_output)
        reloaded = _new_adapter(model_args, device, base_dtype).eval()
        reloaded.load_state_dict(torch.load(checkpoint_output, map_location=device)["state_dict"], strict=True)
        with torch.no_grad():
            reload_result = _forward(
                model=model, adapter=reloaded, raw=full_raw, model_args=model_args,
                selector=selector, bridge=bridge, args=args,
            )
        reload_error = max(
            _max_abs(original.clean_prediction - restored.clean_prediction)
            for original, restored in zip(final_result["outputs"], reload_result["outputs"])
        )
        base_change = _base_change(model, frozen_snapshot)
        formal_sweep = _existing_formal_sweep(final_result["grid_future"], samples, calibration)
        direction = _direction_report(final_result, samples, formal_sweep)
        direction.update(_physical_residual_summary(final_result))
        direction.update({
            "max_non_ego_direct_residual": float(final["max_non_ego_direct_residual"]),
            "max_ego_current_direct_residual": float(final["max_ego_current_direct_residual"]),
            "all_outputs_finite": bool(final["all_outputs_finite"]),
        })
        alignment_reduction = 1.0 - float(final["neutral_relative_alignment_loss"]) / max(float(initial["neutral_relative_alignment_loss"]), 1e-12)
        passed = bool(
            bootstrap_report["passed"]
            and direction["passed"]
            and alignment_reduction >= 0.5
            and bool(final["rho_zero_exact"])
            and bool(final["all_outputs_finite"])
            and float(final["max_non_ego_direct_residual"]) == 0.0
            and float(final["max_ego_current_direct_residual"]) == 0.0
            and float(final["max_lateral_residual"]) <= 1e-5
            and base_change == 0.0 and reload_error == 0.0
        )
        pathwise_report = {
            "schema_version": "preference_flow_step5_s1_pathwise_alignment_v1",
            "base_checkpoint": str(checkpoint_path), "model_args": str(args_path),
            "cohort_file": str(cohort_path), "behavior_normalization_path": str(calibration_path),
            "calibration_ranges": calibration.summary(), "seed": int(args.seed), "device": str(device),
            "checkpoint_coverage": float(model_meta["coverage"]), "cohort_size": len(samples),
            "scene_counts": scene_counts,
            "rho_grid": list(_RHO_GRID), "full_cohort_batch": True,
            "flow_initialization": "fresh_zero_initialized", "selected_checkpoint_step": best_step,
            "selected_checkpoint_total": best_total, "terminal_before_selection": terminal,
            "initial": initial, "final": final, "history": history,
            "gradient_bootstrap": bootstrap_report,
            "neutral_relative_alignment_loss_relative_reduction": alignment_reduction,
            "legacy_expert_target_fit_report_only": True,
            "frozen_base_max_abs_change": base_change, "checkpoint_reload_max_abs_error": reload_error,
            "direction_report": str(direction_path), "proxy_sensitivity_audit": str(audit_path),
            "passed": passed,
        }
        _write_report(pathwise_path, pathwise_report, overwrite=bool(args.overwrite))
        _write_report(direction_path, direction, overwrite=bool(args.overwrite))
        _write_report(gradient_path, gradient_history, overwrite=bool(args.overwrite))
        monitor.log({
            "final/neutral_relative_alignment_loss": float(final["neutral_relative_alignment_loss"]),
            "final/pathwise_order_loss": float(final["pathwise_order_loss"]),
            "final/alignment_reduction": alignment_reduction,
            "final/direction_passed_count": float(direction["path_passed_count"]),
            "final/passed": float(passed),
        }, int(args.steps))
        if not passed:
            raise AssertionError(f"Step-5-S1 failed; inspect {pathwise_path}, {direction_path}, and {gradient_path}")
    finally:
        monitor.finish()
    return audit_path, pathwise_path, direction_path, gradient_path, checkpoint_output


def main() -> None:
    audit, pathwise, direction, gradients, checkpoint = run(_parser().parse_args())
    print(f"Step-5-S1 proxy sensitivity audit passed: {audit}")
    print(f"Step-5-S1 pathwise alignment passed: {pathwise}")
    print(f"Step-5-S1 all-scene direction passed: {direction}")
    print(f"Step-5-S1 gradient history: {gradients}")
    print(f"Step-5-S1 Flow checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

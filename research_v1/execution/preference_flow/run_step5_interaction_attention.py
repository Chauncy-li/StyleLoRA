"""Step 5-S2: neutral-anchored soft interaction attention.

This remains a fixed-cohort, one-diffusion-phase learning proof.  The frozen
StylePlanner produces neutral x0 once; its ego/neighbor predictions determine
one attention map that every rho endpoint reuses.  Ground-truth futures appear
only in differentiable losses and the post-training formal audit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as functional

from baseline.model.style_planner.preference_flow import (
    CLEAN_PREDICTION_EDITOR_DISABLED,
    NeutralAnchoredInteractionAttention,
    PreferenceFlowConfig,
    PreferenceFlowConditionEncoder,
    PreferenceFlowTrainingAdapter,
    SmoothLongitudinalTrajectoryResidualDecoder,
)
from research_v1.execution.preference_flow.differentiable_behavior_alignment import (
    DifferentiableBehaviorBridge,
    FrozenBehaviorCalibration,
    NeutralAnchoredLeadReference,
    feasible_pathwise_loss,
)
from research_v1.execution.preference_flow.run_step1_base_regression import (
    _build_model_args,
    _checkpoint_state,
    _existing_file,
    _load_frozen_styleplanner,
    _model_args_path,
    _write_report,
)
from research_v1.execution.preference_flow.run_step5_semantic_alignment import (
    _existing_formal_sweep,
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
)


_RHO_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
_SCENES = ("straight_free_drive", "straight_car_follow")
_INTERACTION_DIM = 16


@dataclass
class _Branch:
    name: str
    adapter: PreferenceFlowTrainingAdapter
    attention: Optional[NeutralAnchoredInteractionAttention]

    def modules(self) -> Tuple[torch.nn.Module, ...]:
        return (self.adapter,) if self.attention is None else (self.adapter, self.attention)

    def parameters(self) -> List[torch.nn.Parameter]:
        return [parameter for module in self.modules() for parameter in module.parameters() if parameter.requires_grad]

    def train(self, mode: bool) -> None:
        for module in self.modules():
            module.train(mode)

    def snapshot(self) -> Dict[str, Dict[str, torch.Tensor]]:
        result = {"adapter": _adapter_snapshot(self.adapter)}
        if self.attention is not None:
            result["attention"] = {name: value.detach().clone() for name, value in self.attention.state_dict().items()}
        return result

    def restore(self, state: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        self.adapter.load_state_dict(state["adapter"], strict=True)
        if self.attention is not None:
            self.attention.load_state_dict(state["attention"], strict=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step-5-S2 neutral-anchored interaction attention.")
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
    parser.add_argument("--lambda-labeled-target", type=float, default=0.10)
    parser.add_argument("--positive-alignment-span", type=float, default=0.15)
    parser.add_argument("--negative-alignment-span", type=float, default=0.15)
    parser.add_argument("--active-epsilon", type=float, default=1e-5)
    parser.add_argument("--active-fraction", type=float, default=0.5)
    parser.add_argument("--flow-checkpoint-name", default="step5_s2_interaction_attention_flow.pt")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="preference-flow-step5")
    parser.add_argument("--swanlab-run-name", default="interaction-attention")
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline"), default="online")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _masked_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _physical_neighbor_future(clean_prediction: torch.Tensor, model_args: Any) -> torch.Tensor:
    joint = clean_prediction.reshape(int(clean_prediction.shape[0]), int(clean_prediction.shape[1]), -1, 4)
    mean = model_args.state_normalizer.mean[1, 0].to(clean_prediction.device, clean_prediction.dtype)
    std = model_args.state_normalizer.std[1, 0].to(clean_prediction.device, clean_prediction.dtype)
    return joint[:, 1:, 1:, :] * std + mean


def _new_attention_branch(model_args: Any, device: torch.device, dtype: torch.dtype) -> _Branch:
    config = PreferenceFlowConfig(condition_dim=24 + _INTERACTION_DIM)
    adapter = PreferenceFlowTrainingAdapter(
        config=config,
        trajectory_decoder=SmoothLongitudinalTrajectoryResidualDecoder(
            config.latent_dim,
            ego_mean=model_args.state_normalizer.mean[0, 0],
            ego_std=model_args.state_normalizer.std[0, 0],
        ),
        condition_encoder=PreferenceFlowConditionEncoder(
            config.condition_dim,
            task_feature_dim=9,
            include_diffusion_features=True,
            interaction_feature_dim=_INTERACTION_DIM,
        ),
    ).to(device=device, dtype=dtype)
    attention = NeutralAnchoredInteractionAttention(_INTERACTION_DIM).to(device=device, dtype=dtype)
    return _Branch("neutral_anchored_soft_attention", adapter, attention)


def _new_hard_branch(model_args: Any, device: torch.device, dtype: torch.dtype) -> _Branch:
    return _Branch("hard_single_lead_reference", _new_adapter(model_args, device, dtype), None)


def _build_context(
    *,
    model: torch.nn.Module,
    raw: Mapping[str, Any],
    model_args: Any,
    selector: NeutralAnchoredLeadReference,
    bridge: DifferentiableBehaviorBridge,
) -> Dict[str, Any]:
    """Freeze the common neutral x0 and causal scene geometry for both methods."""

    inputs, all_gt, ego_future, neighbor_future, neighbor_mask = _prepare_batch(raw, model_args)
    xq, log_snr = _noisy_state(model, all_gt, raw["diffusion_time"], raw["noise"])
    neutral = _neutral_clean_prediction(model, inputs, xq, raw["diffusion_time"])
    neutral_future = _physical_ego_future(neutral, model_args)
    neutral_neighbors = _physical_neighbor_future(neutral, model_args)
    raw_inputs = raw["inputs"]
    predicted_neighbors = int(model_args.predicted_neighbor_num)
    ego_current = raw_inputs["ego_current_state"][:, :4].to(neutral.dtype)
    neighbor_history = raw_inputs["neighbor_agents_past"][:, :predicted_neighbors].to(neutral.dtype)
    neighbor_current, _neighbor_velocity, observed = selector._last_observed(neighbor_history)
    reference = selector.build(
        neutral_future=neutral_future,
        ego_current_state=ego_current,
        neighbor_history=neighbor_history,
    )
    speed_limit, speed_limit_valid = bridge.speed_limit_from_inputs(
        raw_inputs["route_lanes_speed_limit"], raw_inputs["route_lanes_has_speed_limit"],
        raw_inputs["lanes_speed_limit"], raw_inputs["lanes_has_speed_limit"],
    )
    return {
        "xq": xq.detach(),
        "preference_xq": xq.detach().reshape(int(xq.shape[0]), int(xq.shape[1]), -1),
        "log_snr": log_snr.detach(),
        "neutral": neutral.detach(),
        "neutral_future": neutral_future.detach(),
        "neutral_neighbors": neutral_neighbors.detach(),
        "ego_current": ego_current.detach(),
        "neighbor_current": neighbor_current.detach(),
        "agent_valid": observed.detach().to(torch.bool),
        "neighbor_future": neighbor_future.detach(),
        "neighbor_mask": neighbor_mask.detach(),
        "ego_future": ego_future.detach(),
        "reference": reference,
        "speed_limit": speed_limit.detach().to(neutral.dtype),
        "speed_limit_valid": speed_limit_valid.detach(),
        "free_mask": (raw["task_features"][:, 0] > 0.5).detach(),
        "car_mask": (raw["task_features"][:, 1] > 0.5).detach(),
    }


def _attention_output(branch: _Branch, context: Mapping[str, Any], raw: Mapping[str, Any]) -> Any:
    if branch.attention is None:
        return None
    return branch.attention(
        context["neutral_future"], context["neutral_neighbors"],
        ego_current_state=context["ego_current"],
        neighbor_current_state=context["neighbor_current"],
        agent_valid_mask=context["agent_valid"],
        diffusion_time=raw["diffusion_time"],
        log_snr=context["log_snr"],
    )


def _grid_outputs(branch: _Branch, context: Mapping[str, Any], raw: Mapping[str, Any], attention: Any) -> List[Any]:
    interaction = None if attention is None else attention.interaction_context
    return [
        branch.adapter(
            context["neutral"], context["preference_xq"], raw["diffusion_time"], context["log_snr"],
            raw["task_features"], torch.full_like(raw["rho"], float(rho)),
            interaction_context=interaction,
            physical_ego_current_state=context["ego_current"],
        )
        for rho in _RHO_GRID
    ]


def _measurements(
    branch: _Branch,
    outputs: Sequence[Any],
    context: Mapping[str, Any],
    model_args: Any,
    bridge: DifferentiableBehaviorBridge,
    attention: Any,
) -> Tuple[List[Any], torch.Tensor]:
    futures = torch.stack([_physical_ego_future(output.clean_prediction, model_args) for output in outputs])
    shared = {
        "neutral_future": context["neutral_future"],
        "ego_current_state": context["ego_current"],
        "neighbor_future": context["neighbor_future"],
        "neighbor_mask": context["neighbor_mask"],
        "speed_limit_mps": context["speed_limit"],
        "speed_limit_valid": context["speed_limit_valid"],
        "free_mask": context["free_mask"],
        "car_mask": context["car_mask"],
    }
    if attention is None:
        measurements = [
            bridge.measure(
                edited_future=future, lead_reference=context["reference"],
                base_tangent=output.base_tangent, **shared,
            )
            for future, output in zip(futures, outputs)
        ]
    else:
        measurements = [
            bridge.measure_neutral_attention(
                edited_future=future,
                neighbor_current_state=context["neighbor_current"],
                agent_valid_mask=context["agent_valid"],
                neutral_attention_weights=attention.neighbor_time_attention_weights,
                interaction_confidence=attention.interaction_confidence,
                base_tangent=output.base_tangent,
                **shared,
            )
            for future, output in zip(futures, outputs)
        ]
    return measurements, futures


def _losses(
    *,
    outputs: Sequence[Any],
    measurements: Sequence[Any],
    grid_future: torch.Tensor,
    context: Mapping[str, Any],
    raw: Mapping[str, Any],
    bridge: DifferentiableBehaviorBridge,
    attention: Any,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    neutral_measurement = measurements[2]
    scores = torch.stack([item.score for item in measurements])
    valid = neutral_measurement.score_valid
    target_pairs = [
        bridge.desired_score(
            neutral_measurement, torch.full_like(raw["rho"], float(rho)),
            free_mask=context["free_mask"], car_mask=context["car_mask"],
            positive_span=float(args.positive_alignment_span),
            negative_span=float(args.negative_alignment_span),
        )
        for rho in _RHO_GRID
    ]
    target_scores = torch.stack([item[0] for item in target_pairs])
    target_valid = torch.stack([item[1] for item in target_pairs]) & valid[None]
    alignment = _masked_average(functional.smooth_l1_loss(scores, target_scores, reduction="none"), target_valid)
    path = feasible_pathwise_loss(
        scores, target_scores, valid,
        active_epsilon=float(args.active_epsilon), active_fraction=float(args.active_fraction),
    )
    chosen = (raw["rho"][:, None] - torch.tensor(_RHO_GRID, device=raw["rho"].device, dtype=raw["rho"].dtype)[None]).abs().argmin(dim=1)
    batch = torch.arange(raw["rho"].shape[0], device=raw["rho"].device)
    chosen_scores = scores.permute(1, 0)[batch, chosen]
    shared = {
        "neutral_future": context["neutral_future"], "ego_current_state": context["ego_current"],
        "neighbor_future": context["neighbor_future"], "neighbor_mask": context["neighbor_mask"],
        "speed_limit_mps": context["speed_limit"], "speed_limit_valid": context["speed_limit_valid"],
        "free_mask": context["free_mask"], "car_mask": context["car_mask"],
    }
    if attention is None:
        target_measurement = bridge.measure(
            edited_future=context["ego_future"], lead_reference=context["reference"],
            base_tangent=outputs[2].base_tangent, **shared,
        )
    else:
        target_measurement = bridge.measure_neutral_attention(
            edited_future=context["ego_future"], neighbor_current_state=context["neighbor_current"],
            agent_valid_mask=context["agent_valid"],
            neutral_attention_weights=attention.neighbor_time_attention_weights,
            interaction_confidence=attention.interaction_confidence,
            base_tangent=outputs[2].base_tangent, **shared,
        )
    labeled_valid = valid & target_measurement.score_valid
    labeled_target = _masked_average(
        functional.smooth_l1_loss(chosen_scores, target_measurement.score.detach(), reduction="none"), labeled_valid
    )
    residual = grid_future[..., :2] - context["neutral_future"][None, ..., :2]
    smooth = (
        (residual[:, :, 2:] - 2.0 * residual[:, :, 1:-1] + residual[:, :, :-2]).square().mean()
        if int(residual.shape[2]) >= 3 else residual.new_zeros(())
    )
    free_valid = target_valid & context["free_mask"][None]
    car_valid = target_valid & context["car_mask"][None]
    return {
        "alignment": alignment,
        "path": path.loss,
        "labeled_target": labeled_target,
        "content": residual.square().mean(),
        "smooth": smooth,
        "scores": scores,
        "score_valid": valid,
        "target_scores": target_scores,
        "target_valid": target_valid,
        "path_deltas": path.deltas,
        "target_deltas": path.target_deltas,
        "active_mask": path.active_mask,
        "saturated_mask": path.saturated_mask,
        "free_alignment": _masked_average(functional.smooth_l1_loss(scores, target_scores, reduction="none"), free_valid),
        "car_alignment": _masked_average(functional.smooth_l1_loss(scores, target_scores, reduction="none"), car_valid),
        "free_path": _masked_average(torch.relu(-path.deltas), path.active_mask & context["free_mask"][None]),
        "car_path": _masked_average(torch.relu(-path.deltas), path.active_mask & context["car_mask"][None]),
        "labeled_valid": labeled_valid,
    }


def _forward(
    *,
    branch: _Branch,
    context: Mapping[str, Any],
    raw: Mapping[str, Any],
    model_args: Any,
    bridge: DifferentiableBehaviorBridge,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    attention = _attention_output(branch, context, raw)
    outputs = _grid_outputs(branch, context, raw, attention)
    measurements, grid_future = _measurements(branch, outputs, context, model_args, bridge, attention)
    losses = _losses(
        outputs=outputs, measurements=measurements, grid_future=grid_future,
        context=context, raw=raw, bridge=bridge, attention=attention, args=args,
    )
    total = (
        losses["alignment"] + losses["path"]
        + float(args.lambda_labeled_target) * losses["labeled_target"]
        + float(args.lambda_content) * losses["content"]
        + float(args.lambda_smooth) * losses["smooth"]
    )
    return {
        "total": total, "losses": losses, "outputs": outputs, "measurements": measurements,
        "grid_future": grid_future, "attention": attention,
    }


def _result_summary(result: Mapping[str, Any], context: Mapping[str, Any]) -> Dict[str, Any]:
    outputs, losses = result["outputs"], result["losses"]
    nonzero = [output.clean_prediction - context["neutral"] for index, output in enumerate(outputs) if index != 2]
    longitudinal = torch.cat([
        (output.physical_xy_residual * output.base_tangent).sum(dim=-1)
        for index, output in enumerate(outputs) if index != 2
    ], dim=0)
    lateral = torch.cat([
        output.physical_xy_residual[..., 0] * output.base_tangent[..., 1]
        - output.physical_xy_residual[..., 1] * output.base_tangent[..., 0]
        for index, output in enumerate(outputs) if index != 2
    ], dim=0)
    return {
        "total": float(result["total"].detach().item()),
        **{key: float(losses[key].detach().item()) for key in ("alignment", "path", "labeled_target", "content", "smooth", "free_alignment", "car_alignment")},
        "rho_zero_exact": bool(torch.equal(outputs[2].clean_prediction, context["neutral"])),
        "score_valid_count": int(losses["score_valid"].sum().item()),
        "labeled_target_valid_count": int(losses["labeled_valid"].sum().item()),
        "max_latent_displacement": max(_max_abs(output.latent_end - output.latent_start) for output in outputs),
        "max_ego_future_residual": max(_max_abs(output.ego_future_residual) for output in outputs),
        "max_non_ego_direct_residual": max(_max_abs(value[:, 1:, :]) for value in nonzero),
        "max_ego_current_direct_residual": max(_max_abs(value[:, 0, :4]) for value in nonzero),
        "max_lateral_residual": _max_abs(lateral),
        "physical_longitudinal_residual_abs_mean": _absolute_statistics(longitudinal)["mean"],
        "physical_longitudinal_residual_abs_p95": _absolute_statistics(longitudinal)["p95"],
        "physical_longitudinal_residual_abs_max": _absolute_statistics(longitudinal)["max"],
        "all_outputs_finite": all(bool(torch.isfinite(output.clean_prediction).all().item()) for output in outputs),
    }


def _gradient_alignment(result: Mapping[str, Any], branch: _Branch) -> Dict[str, Any]:
    parameters = list(branch.adapter.vector_field.parameters())
    losses = result["losses"]
    free = losses["free_alignment"] + losses["free_path"]
    car = losses["car_alignment"] + losses["car_path"]

    def vector(loss: torch.Tensor) -> torch.Tensor:
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        return torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1) for parameter, gradient in zip(parameters, gradients)])

    free_vector, car_vector = vector(free), vector(car)
    free_norm = float(torch.linalg.vector_norm(free_vector).detach().cpu().item())
    car_norm = float(torch.linalg.vector_norm(car_vector).detach().cpu().item())
    return {
        "vector_field_free_grad_norm": free_norm,
        "vector_field_car_grad_norm": car_norm,
        "vector_field_free_car_cosine": None if free_norm == 0.0 or car_norm == 0.0 else float(torch.dot(free_vector, car_vector).detach().cpu().item() / (free_norm * car_norm)),
        "optimizer_modified_for_conflict": False,
        "context_reference": "neutral_only",
        "rho_is_not_an_attention_input": True,
    }


def _attention_diagnostics(result: Mapping[str, Any], context: Mapping[str, Any]) -> Dict[str, Any]:
    attention = result["attention"]
    if attention is None:
        return {"enabled": False}
    weights = attention.neighbor_time_attention_weights.detach()
    total_error = _max_abs(weights.sum(dim=(1, 2)) + attention.null_interaction_weight.detach() - 1.0)
    masked = weights.masked_select(~context["agent_valid"][:, :, None].expand_as(weights))
    top = weights.sum(dim=-1).argmax(dim=1)
    conditions = [output.condition[:, -_INTERACTION_DIM:] for output in result["outputs"]]
    shared_error = max(_max_abs(value - conditions[0]) for value in conditions[1:])
    return {
        "enabled": True,
        "weights_plus_null_max_abs_error": total_error,
        "masked_agent_weight_abs_max": _max_abs(masked),
        "interaction_confidence": [float(value) for value in attention.interaction_confidence.detach().cpu().tolist()],
        "null_interaction_weight": [float(value) for value in attention.null_interaction_weight.detach().cpu().tolist()],
        "top_neighbor_indices": [int(value) for value in top.detach().cpu().tolist()],
        "unique_top_neighbor_count": int(torch.unique(top).numel()),
        "shared_attention_condition_max_abs_error": shared_error,
        "all_rho_share_the_same_neutral_attention": bool(shared_error == 0.0),
        "masked_agents_exact_zero": bool(_max_abs(masked) == 0.0),
        "weights_normalized": bool(total_error <= 1e-6),
        "all_attention_outputs_finite": bool(
            torch.isfinite(attention.interaction_context).all().item()
            and torch.isfinite(weights).all().item()
            and torch.isfinite(attention.null_interaction_weight).all().item()
            and torch.isfinite(attention.interaction_confidence).all().item()
        ),
    }


def _direction_report(
    *,
    result: Mapping[str, Any],
    context: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    formal_sweep: Sequence[Mapping[str, Any]],
    has_attention: bool,
) -> Dict[str, Any]:
    losses, measurements = result["losses"], result["measurements"]
    scores = losses["scores"].detach().cpu()
    targets = losses["target_scores"].detach().cpu()
    valid = losses["score_valid"].detach().cpu()
    active = losses["active_mask"].detach().cpu()
    saturated = losses["saturated_mask"].detach().cpu()
    rows: List[Dict[str, Any]] = []
    active_correct = active_total = saturated_ok = saturated_total = reversal_count = 0
    formal_agree = formal_correct = formal_total = 0
    coverage_values: List[float] = []
    formal_headway_valid = attention_covered = car_proxy_covered = 0
    for index, sample in enumerate(samples):
        proxy = [float(value) for value in scores[:, index].tolist()]
        target = [float(value) for value in targets[:, index].tolist()]
        formal = [float(value) for value in formal_sweep[index]["values"]]
        formal_valid = [bool(value) for value in formal_sweep[index]["valid"]]
        proxy_delta = [proxy[item + 1] - proxy[item] for item in range(4)]
        target_delta = [target[item + 1] - target[item] for item in range(4)]
        formal_delta = [formal[item + 1] - formal[item] for item in range(4)]
        interval_rows = []
        for step in range(4):
            active_interval = bool(active[step, index].item())
            saturated_interval = bool(saturated[step, index].item())
            proxy_ok = proxy_delta[step] > 0.0 if active_interval else proxy_delta[step] >= -1e-6
            formal_ok = formal_delta[step] > 0.0 if active_interval else formal_delta[step] >= -1e-6
            formal_is_valid = bool(valid[index].item()) and formal_valid[step] and formal_valid[step + 1]
            if active_interval:
                active_total += 1
                active_correct += int(proxy_ok)
            if saturated_interval:
                saturated_total += 1
                saturated_ok += int(proxy_ok)
            reversal_count += int(bool(valid[index].item()) and proxy_delta[step] < -1e-6)
            if formal_is_valid:
                formal_total += 1
                same_direction = proxy_ok == formal_ok
                formal_agree += int(same_direction)
                formal_correct += int(formal_ok)
            interval_rows.append({
                "rho_interval": [_RHO_GRID[step], _RHO_GRID[step + 1]], "target_delta": target_delta[step],
                "proxy_delta": proxy_delta[step], "formal_delta": formal_delta[step],
                "active": active_interval, "saturated": saturated_interval,
                "proxy_direction_correct": proxy_ok, "formal_direction_correct": formal_ok,
                "formal_valid": formal_is_valid,
            })
        row = {
            "filename": str(sample["filename"]), "scene": str(sample["scene"]), "rho_grid": list(_RHO_GRID),
            "differentiable_proxy": proxy, "neutral_relative_target": target,
            "existing_formal_metric": formal, "existing_formal_metric_valid": formal_valid,
            "existing_formal_metric_raw_values": formal_sweep[index]["raw_values"],
            "existing_formal_metric_axis_valid_masks": formal_sweep[index]["axis_valid_masks"],
            "existing_formal_metric_sources": formal_sweep[index]["metric_sources"],
            "existing_formal_metric_failures": formal_sweep[index]["failures"],
            "intervals": interval_rows,
            "full_path_reversal_count": sum(delta < -1e-6 for delta in proxy_delta),
        }
        if row["scene"] == "straight_car_follow":
            car_proxy_covered += int(bool(valid[index].item()))
        if has_attention:
            coverage = float(measurements[2].attention_coverage[index].detach().cpu().item())
            row["neutral_attention_coverage"] = coverage
            row["neutral_interaction_confidence"] = float(measurements[2].interaction_confidence[index].detach().cpu().item())
            coverage_values.append(coverage)
            if row["scene"] == "straight_car_follow" and bool(formal_sweep[index]["axis_valid_masks"][2][0]):
                formal_headway_valid += 1
                attention_covered += int(coverage > 1e-6)
        rows.append(row)
    by_scene = {}
    for scene in _SCENES:
        scene_rows = [row for row in rows if row["scene"] == scene]
        by_scene[scene] = {
            "scene_count": len(scene_rows),
            "full_path_reversal_count": sum(int(row["full_path_reversal_count"]) for row in scene_rows),
        }
    return {
        "schema_version": "preference_flow_step5_s2_direction_v1",
        "rho_grid": list(_RHO_GRID), "scene_count": len(rows), "scenes": rows, "by_scene": by_scene,
        "active_interval_accuracy": active_correct / float(max(active_total, 1)),
        "active_interval_correct_count": active_correct, "active_interval_count": active_total,
        "saturated_interval_non_reversal_accuracy": saturated_ok / float(max(saturated_total, 1)),
        "saturated_interval_non_reversal_count": saturated_ok, "saturated_interval_count": saturated_total,
        "full_path_reversal_count": reversal_count,
        "proxy_formal_direction_agreement": formal_agree / float(max(formal_total, 1)),
        "proxy_formal_direction_agreement_count": formal_agree, "proxy_formal_direction_interval_count": formal_total,
        "formal_direction_correct_count": formal_correct,
        "car_follow_formal_headway_valid_count": formal_headway_valid,
        "car_follow_proxy_interaction_coverage_count": car_proxy_covered,
        "car_follow_attention_coverage_count": attention_covered,
        "car_follow_attention_coverage_rate": attention_covered / float(max(formal_headway_valid, 1)),
        "mean_attention_coverage": sum(coverage_values) / float(max(len(coverage_values), 1)),
        "passed": bool(
            active_total > 0 and active_correct == active_total and saturated_ok == saturated_total and reversal_count == 0
            and formal_agree == formal_total and formal_correct == formal_total
            and (not has_attention or (formal_headway_valid == 6 and attention_covered == 6))
        ),
    }


def _train_branch(
    *,
    branch: _Branch,
    context: Mapping[str, Any],
    raw: Mapping[str, Any],
    model_args: Any,
    bridge: DifferentiableBehaviorBridge,
    args: argparse.Namespace,
    monitor: _SwanLabMonitor,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, float]], Dict[str, Any]]:
    parameters = branch.parameters()
    optimizer = torch.optim.Adam(parameters, lr=float(args.learning_rate))
    branch.train(False)
    with torch.no_grad():
        initial_result = _forward(branch=branch, context=context, raw=raw, model_args=model_args, bridge=bridge, args=args)
    initial = _result_summary(initial_result, context)
    branch.train(True)
    optimizer.zero_grad(set_to_none=True)
    bootstrap = _forward(branch=branch, context=context, raw=raw, model_args=model_args, bridge=bridge, args=args)
    bootstrap["total"].backward()
    bootstrap_report = {
        "vector_field_grad_norm": _grad_norm(branch.adapter.vector_field.parameters()),
        "all_trainable_gradients_finite": _all_finite_gradients(parameters),
    }
    bootstrap_report["passed"] = bool(bootstrap_report["vector_field_grad_norm"] > 0.0 and bootstrap_report["all_trainable_gradients_finite"])
    optimizer.zero_grad(set_to_none=True)
    if not bootstrap_report["passed"]:
        raise AssertionError(f"{branch.name} vector-field gradient bootstrap failed")
    gradient_history = {"initial": _gradient_alignment(
        _forward(branch=branch, context=context, raw=raw, model_args=model_args, bridge=bridge, args=args), branch
    )}
    best_state, best_total, best_step = branch.snapshot(), float(initial["total"]), 0
    history: List[Dict[str, float]] = []
    midpoint = max(int(args.steps) // 2, 1)
    for step in range(1, int(args.steps) + 1):
        branch.train(True)
        optimizer.zero_grad(set_to_none=True)
        result = _forward(branch=branch, context=context, raw=raw, model_args=model_args, bridge=bridge, args=args)
        current_total = float(result["total"].detach().item())
        if current_total < best_total:
            best_state, best_total, best_step = branch.snapshot(), current_total, step - 1
        result["total"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
        optimizer.step()
        row = {"step": float(step), **{key: float(result["losses"][key].detach().item()) for key in ("alignment", "path", "labeled_target", "content", "smooth")}, "total": current_total}
        history.append(row)
        if step == midpoint:
            branch.train(False)
            gradient_history["midpoint"] = _gradient_alignment(
                _forward(branch=branch, context=context, raw=raw, model_args=model_args, bridge=bridge, args=args), branch
            )
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            monitor.log({f"{branch.name}/train/{key}": value for key, value in row.items()}, step)
    branch.train(False)
    with torch.no_grad():
        terminal = _forward(branch=branch, context=context, raw=raw, model_args=model_args, bridge=bridge, args=args)
    terminal_summary = _result_summary(terminal, context)
    if terminal_summary["total"] < best_total:
        best_state, best_total, best_step = branch.snapshot(), float(terminal_summary["total"]), int(args.steps)
    branch.restore(best_state)
    branch.train(False)
    with torch.no_grad():
        final_result = _forward(branch=branch, context=context, raw=raw, model_args=model_args, bridge=bridge, args=args)
    final = _result_summary(final_result, context)
    gradient_history["final"] = _gradient_alignment(
        _forward(branch=branch, context=context, raw=raw, model_args=model_args, bridge=bridge, args=args), branch
    )
    gradient_history["optimizer_modified_for_conflict"] = False
    return initial, final, history, {
        "result": final_result, "bootstrap": bootstrap_report, "gradient_history": gradient_history,
        "selected_step": best_step, "selected_total": best_total, "terminal_before_selection": terminal_summary,
    }


def _check_args(args: argparse.Namespace) -> None:
    if int(args.steps) <= 0 or int(args.batch_size) <= 0 or int(args.log_interval) <= 0:
        raise ValueError("--steps, --batch-size, and --log-interval must be positive")
    if float(args.learning_rate) <= 0.0 or float(args.lambda_labeled_target) < 0.0:
        raise ValueError("learning rate must be positive and target-loss weight non-negative")
    if float(args.positive_alignment_span) <= 0.0 or float(args.negative_alignment_span) <= 0.0:
        raise ValueError("alignment spans must be positive")
    if float(args.active_epsilon) < 0.0 or not 0.0 < float(args.active_fraction) <= 1.0:
        raise ValueError("invalid active interval settings")


def run(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, Path]:
    _check_args(args)
    _seed_everything(int(args.seed))
    checkpoint_path = _existing_file(args.base_checkpoint, "--base-checkpoint")
    cache_root, cohort_path = Path(args.cache_root).expanduser().resolve(), _existing_file(args.cohort_file, "--cohort-file")
    calibration_path = _existing_file(args.behavior_normalization_path, "--behavior-normalization-path")
    if not cache_root.is_dir():
        raise NotADirectoryError(f"--cache-root is not a directory: {cache_root}")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    args_path = _model_args_path(checkpoint_path, args.model_args)
    model_args = _build_model_args(args_path, mode=CLEAN_PREDICTION_EDITOR_DISABLED, device=str(device), normalization_file_override=args.normalization_file_path)
    state, checkpoint_meta = _checkpoint_state(checkpoint_path, prefer_ema=bool(args.prefer_ema))
    model, model_meta = _load_frozen_styleplanner(model_args, state, checkpoint_meta)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    frozen_snapshot = _base_snapshot(model)
    entries = _cohort_entries(cohort_path, cache_root)
    samples = [{**_load_sample(entry), "cache_path": str(entry["cache_path"])} for entry in entries]
    counts = {scene: sum(item["scene"] == scene for item in samples) for scene in _SCENES}
    if len(samples) != 12 or any(counts[scene] != 6 for scene in _SCENES) or int(args.batch_size) != 12:
        raise ValueError("Step-5-S2 requires exactly six free-drive and six car-follow scenes with --batch-size 12")
    _attach_fixed_phases(samples, predicted_neighbors=int(model_args.predicted_neighbor_num), future_len=int(model_args.future_len), seed=int(args.seed))
    calibration = FrozenBehaviorCalibration.from_json(str(calibration_path))
    selector, bridge = NeutralAnchoredLeadReference(), DifferentiableBehaviorBridge(calibration)
    raw = _batch(samples, list(range(len(samples))), device)
    with torch.no_grad():
        context = _build_context(model=model, raw=raw, model_args=model_args, selector=selector, bridge=bridge)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    hard_path = output_dir / "step5_s2_hard_single_lead.json"
    soft_path = output_dir / "step5_s2_soft_interaction_attention.json"
    comparison_path = output_dir / "step5_s2_comparison.json"
    diagnostics_path = output_dir / "step5_s2_attention_diagnostics.json"
    checkpoint_output = output_dir / str(args.flow_checkpoint_name)
    for path in (hard_path, soft_path, comparison_path, diagnostics_path, checkpoint_output):
        if path.exists() and not bool(args.overwrite):
            raise FileExistsError(f"refusing to overwrite: {path}")
    monitor = _SwanLabMonitor(args, {"seed": int(args.seed), "cohort_size": len(samples), "rho_grid": list(_RHO_GRID), "base_checkpoint": str(checkpoint_path)}, output_dir / "swanlog")
    try:
        base_dtype = next(model.parameters()).dtype
        _seed_everything(int(args.seed))
        hard = _new_hard_branch(model_args, device, base_dtype)
        hard_initial, hard_final, hard_history, hard_extra = _train_branch(branch=hard, context=context, raw=raw, model_args=model_args, bridge=bridge, args=args, monitor=monitor)
        hard_formal = _existing_formal_sweep(hard_extra["result"]["grid_future"], samples, calibration)
        hard_direction = _direction_report(result=hard_extra["result"], context=context, samples=samples, formal_sweep=hard_formal, has_attention=False)
        _write_report(hard_path, {"schema_version": "preference_flow_step5_s2_hard_reference_v1", "method": hard.name, "checkpoint_coverage": float(model_meta["coverage"]), "initial": hard_initial, "final": hard_final, "history": hard_history, "gradient_bootstrap": hard_extra["bootstrap"], "gradient_history": hard_extra["gradient_history"], "direction": hard_direction, "selected_step": hard_extra["selected_step"], "selected_total": hard_extra["selected_total"], "same_seed": int(args.seed), "same_steps": int(args.steps)}, overwrite=True)

        _seed_everything(int(args.seed))
        soft = _new_attention_branch(model_args, device, base_dtype)
        soft_initial, soft_final, soft_history, soft_extra = _train_branch(branch=soft, context=context, raw=raw, model_args=model_args, bridge=bridge, args=args, monitor=monitor)
        soft_formal = _existing_formal_sweep(soft_extra["result"]["grid_future"], samples, calibration)
        soft_direction = _direction_report(result=soft_extra["result"], context=context, samples=samples, formal_sweep=soft_formal, has_attention=True)
        attention = _attention_diagnostics(soft_extra["result"], context)
        alignment_reduction = 1.0 - soft_final["alignment"] / max(soft_initial["alignment"], 1e-12)
        torch.save({"schema_version": "preference_flow_step5_s2_checkpoint_v1", "flow_config": asdict(soft.adapter.config), "adapter_state_dict": soft.adapter.state_dict(), "attention_state_dict": soft.attention.state_dict(), "base_checkpoint": str(checkpoint_path), "behavior_normalization_path": str(calibration_path)}, checkpoint_output)
        _seed_everything(int(args.seed))
        reloaded = _new_attention_branch(model_args, device, base_dtype)
        reload_payload = torch.load(checkpoint_output, map_location=device)
        reloaded.adapter.load_state_dict(reload_payload["adapter_state_dict"], strict=True)
        reloaded.attention.load_state_dict(reload_payload["attention_state_dict"], strict=True)
        reloaded.train(False)
        with torch.no_grad():
            reload_result = _forward(branch=reloaded, context=context, raw=raw, model_args=model_args, bridge=bridge, args=args)
        reload_error = max(_max_abs(left.clean_prediction - right.clean_prediction) for left, right in zip(soft_extra["result"]["outputs"], reload_result["outputs"]))
        base_change = _base_change(model, frozen_snapshot)
        passed = bool(
            soft_extra["bootstrap"]["passed"] and soft_direction["passed"] and alignment_reduction >= 0.5
            and soft_final["rho_zero_exact"] and soft_final["all_outputs_finite"]
            and soft_final["max_non_ego_direct_residual"] == 0.0 and soft_final["max_ego_current_direct_residual"] == 0.0
            and soft_final["max_lateral_residual"] <= 1e-5 and base_change == 0.0 and reload_error == 0.0
            and attention["weights_normalized"] and attention["masked_agents_exact_zero"]
            and attention["all_attention_outputs_finite"] and attention["unique_top_neighbor_count"] > 1
            and attention["all_rho_share_the_same_neutral_attention"]
        )
        soft_report = {
            "schema_version": "preference_flow_step5_s2_soft_attention_v1", "method": soft.name,
            "checkpoint_coverage": float(model_meta["coverage"]),
            "initial": soft_initial, "final": soft_final, "history": soft_history,
            "gradient_bootstrap": soft_extra["bootstrap"], "gradient_history": soft_extra["gradient_history"],
            "direction": soft_direction, "attention": attention, "selected_step": soft_extra["selected_step"],
            "selected_total": soft_extra["selected_total"], "neutral_relative_alignment_loss_relative_reduction": alignment_reduction,
            "frozen_base_max_abs_change": base_change, "checkpoint_reload_max_abs_error": reload_error,
            "attention_inputs": "neutral ego/neighbor clean predictions, current observed geometry, q/log-SNR only",
            "attention_rho_input": False, "attention_weights_detached_for_behavior_loss": True,
            "attention_disabled_step1_to_step4_contract": "unchanged: no sampler, decoder, or base-planner path invokes this optional module",
            "passed": passed,
        }
        _write_report(soft_path, soft_report, overwrite=True)
        _write_report(diagnostics_path, {"schema_version": "preference_flow_step5_s2_attention_diagnostics_v1", **attention, "direction": soft_direction, "passed": passed}, overwrite=True)
        comparison = {
            "schema_version": "preference_flow_step5_s2_comparison_v1", "same_base_checkpoint": str(checkpoint_path),
            "same_cohort": str(cohort_path), "same_seed": int(args.seed), "same_training_steps": int(args.steps),
            "hard_single_lead": {"score_valid_count": hard_final["score_valid_count"], "car_follow_proxy_interaction_coverage_count": hard_direction["car_follow_proxy_interaction_coverage_count"], "alignment": hard_final["alignment"], "direction": hard_direction, "physical_longitudinal_residual": {key: hard_final[key] for key in ("physical_longitudinal_residual_abs_mean", "physical_longitudinal_residual_abs_p95", "physical_longitudinal_residual_abs_max")}},
            "neutral_anchored_soft_attention": {"score_valid_count": soft_final["score_valid_count"], "car_follow_proxy_interaction_coverage_count": soft_direction["car_follow_proxy_interaction_coverage_count"], "alignment": soft_final["alignment"], "direction": soft_direction, "physical_longitudinal_residual": {key: soft_final[key] for key in ("physical_longitudinal_residual_abs_mean", "physical_longitudinal_residual_abs_p95", "physical_longitudinal_residual_abs_max")}},
            "comparison_focus": ["car_follow_coverage", "formal_direction_agreement", "full_path_reversal_count", "neutral_relative_alignment", "physical_trajectory_residual"],
            "passed": passed,
        }
        _write_report(comparison_path, comparison, overwrite=True)
        monitor.log({"final/soft_alignment_reduction": alignment_reduction, "final/soft_passed": float(passed), "final/soft_car_coverage": soft_direction["car_follow_attention_coverage_rate"]}, int(args.steps))
        if not passed:
            raise AssertionError(f"Step-5-S2 failed; inspect {soft_path}, {comparison_path}, and {diagnostics_path}")
    finally:
        monitor.finish()
    return hard_path, soft_path, comparison_path, diagnostics_path, checkpoint_output


def main() -> None:
    hard, soft, comparison, diagnostics, checkpoint = run(_parser().parse_args())
    print(f"Step-5-S2 hard single-lead reference: {hard}")
    print(f"Step-5-S2 soft interaction attention: {soft}")
    print(f"Step-5-S2 comparison: {comparison}")
    print(f"Step-5-S2 attention diagnostics: {diagnostics}")
    print(f"Step-5-S2 Flow checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

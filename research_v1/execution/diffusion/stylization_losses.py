"""Losses specific to V6 axis routing and normal anchoring."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from baseline.model.style_planner.guidance.preference_energy import (
    ConditionalPreferenceEnergy,
)
from baseline.model.style_planner.library.dpm_solver_pytorch import NoiseScheduleVP
from baseline.model.style_planner.loss.diff_loss import _extract_diffusion_prediction


CONTROLLED_SCENES = (
    "straight_free_drive",
    "straight_car_follow",
)

PAIR_AXIS_DIAGNOSTIC_NAMES = {
    "straight_free_drive": (
        "free_speed_utilization",
        "free_accel_willingness",
        "free_speed_response",
    ),
    "straight_car_follow": (
        "car_headway_tightness_from_h",
        "car_ttc_tightness",
        "car_closing_tolerance",
    ),
}


def _scene_names(inputs: Dict[str, Any], batch_size: int) -> list[str]:
    raw = inputs.get("scene_bucket", [])
    if isinstance(raw, str):
        return [raw] * batch_size
    names = [str(item) for item in raw]
    if len(names) < batch_size:
        names.extend(["none"] * (batch_size - len(names)))
    return names[:batch_size]


def _empty_relative_scene_metrics(zero: torch.Tensor) -> Dict[str, torch.Tensor]:
    metrics: Dict[str, torch.Tensor] = {}
    for scene in CONTROLLED_SCENES:
        metrics[f"normal_relative_axis_mae_{scene}"] = zero
        metrics[f"normal_relative_axis_active_count_{scene}"] = zero
    return metrics


def _empty_pair_scene_metrics(zero: torch.Tensor) -> Dict[str, torch.Tensor]:
    metrics: Dict[str, torch.Tensor] = {}
    for scene in CONTROLLED_SCENES:
        metrics[f"signed_monotonic_order_accuracy_{scene}"] = zero
        metrics[f"signed_monotonic_response_{scene}"] = zero
        metrics[f"signed_monotonic_pair_count_{scene}"] = zero
        for axis_name in PAIR_AXIS_DIAGNOSTIC_NAMES[scene]:
            metrics[f"{axis_name}_order"] = zero
            metrics[f"{axis_name}_mean_response"] = zero
            metrics[f"{axis_name}_pair_count"] = zero
            metrics[f"{axis_name}_worst_ratio"] = zero
    return metrics


def _empty_signed_pair_metrics(zero: torch.Tensor) -> Dict[str, torch.Tensor]:
    metrics = {
        "signed_monotonic_loss": zero,
        "signed_monotonic_order_accuracy": zero,
        "signed_monotonic_response": zero,
        "signed_symmetry_loss": zero,
        "signed_monotonic_pair_count": zero,
        "signed_monotonic_sample_count": zero,
        "signed_monotonic_axis_pair_count": zero,
        "signed_monotonic_worst_axis_loss": zero,
        "signed_monotonic_worst_axis_temperature": zero,
        "signed_monotonic_soft_worst_enabled": zero,
        "signed_monotonic_diffusion_time_mean": zero,
        "signed_monotonic_diffusion_time_min": zero,
        "signed_monotonic_diffusion_time_max": zero,
        "signed_monotonic_short_rollout_steps": zero,
        "signed_monotonic_rollout_state_l1": zero,
    }
    metrics.update(_empty_pair_scene_metrics(zero))
    return metrics


def _logsnr_short_rollout_times(
    reference: torch.Tensor,
    *,
    t_start: float,
    t_end: float,
    steps: int,
) -> torch.Tensor:
    """Return the same descending log-SNR grid used by StylePlanner inference."""

    if int(steps) <= 0:
        raise ValueError("short rollout steps must be positive")
    if not 0.0 < float(t_end) < float(t_start) <= 1.0:
        raise ValueError("short rollout times must satisfy 0 < t_end < t_start <= 1")
    schedule = NoiseScheduleVP(schedule="linear")
    endpoints = reference.new_tensor([float(t_start), float(t_end)])
    lambda_start = schedule.marginal_lambda(endpoints[:1])
    lambda_end = schedule.marginal_lambda(endpoints[1:])
    lambda_grid = torch.linspace(
        lambda_start.item(),
        lambda_end.item(),
        int(steps) + 1,
        device=reference.device,
        dtype=reference.dtype,
    )
    times = schedule.inverse_lambda(lambda_grid)
    times[0] = float(t_start)
    times[-1] = float(t_end)
    return times


def _dpmpp_first_order_state_step(
    state: torch.Tensor,
    x_start: torch.Tensor,
    *,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    fixed_current: torch.Tensor,
) -> torch.Tensor:
    """One differentiable DPM-Solver++ first-order (DDIM) state update.

    Only future states are advanced. The observed current state is re-applied in
    exactly the same way as StylePlanner's inference-time state constraint.
    """

    if state.shape != x_start.shape:
        raise ValueError(
            "short-rollout state/x_start shape mismatch: "
            f"{tuple(state.shape)} != {tuple(x_start.shape)}"
        )
    schedule = NoiseScheduleVP(schedule="linear")
    alpha_start = schedule.marginal_alpha(t_start).reshape(-1, 1, 1, 1)
    alpha_end = schedule.marginal_alpha(t_end).reshape(-1, 1, 1, 1)
    sigma_start = schedule.marginal_std(t_start).reshape(-1, 1, 1, 1).clamp_min(1e-6)
    sigma_end = schedule.marginal_std(t_end).reshape(-1, 1, 1, 1)
    state_weight = sigma_end / sigma_start
    x_start_weight = alpha_end - state_weight * alpha_start
    future = state_weight * state[..., 1:, :] + x_start_weight * x_start[..., 1:, :]
    return torch.cat([fixed_current, future], dim=-2)


def selftest_short_rollout_dpmpp() -> Dict[str, bool]:
    """Algebraic checks for the A3.3 generated-state transition."""

    dtype = torch.float64
    clean = torch.tensor(
        [[[[0.2, -0.4, 0.8, -0.1], [0.5, 0.3, -0.2, 0.7]]]],
        dtype=dtype,
    )
    noise = torch.tensor(
        [[[[0.1, 0.6, -0.3, 0.2], [-0.5, 0.4, 0.9, -0.7]]]],
        dtype=dtype,
    )
    t_start = torch.tensor([0.10], dtype=dtype)
    t_end = torch.tensor([0.01], dtype=dtype)
    schedule = NoiseScheduleVP(schedule="linear")
    alpha_start = schedule.marginal_alpha(t_start).reshape(-1, 1, 1, 1)
    sigma_start = schedule.marginal_std(t_start).reshape(-1, 1, 1, 1)
    alpha_end = schedule.marginal_alpha(t_end).reshape(-1, 1, 1, 1)
    sigma_end = schedule.marginal_std(t_end).reshape(-1, 1, 1, 1)
    state = alpha_start * clean + sigma_start * noise
    expected = alpha_end * clean + sigma_end * noise
    fixed_current = clean[..., :1, :]
    state = torch.cat([fixed_current, state[..., 1:, :]], dim=-2)
    expected = torch.cat([fixed_current, expected[..., 1:, :]], dim=-2)
    advanced = _dpmpp_first_order_state_step(
        state,
        clean,
        t_start=t_start,
        t_end=t_end,
        fixed_current=fixed_current,
    )
    grid = _logsnr_short_rollout_times(
        clean,
        t_start=0.10,
        t_end=0.01,
        steps=2,
    )
    return {
        "first_order_matches_shared_noise_path": bool(
            torch.allclose(advanced, expected, atol=1e-9, rtol=1e-9)
        ),
        "time_grid_has_exact_endpoints": bool(
            abs(float(grid[0]) - 0.10) < 1e-12
            and abs(float(grid[-1]) - 0.01) < 1e-12
        ),
        "time_grid_is_strictly_descending": bool(torch.all(grid[:-1] > grid[1:])),
        "current_state_is_fixed": bool(
            torch.equal(advanced[..., :1, :], fixed_current)
        ),
    }


def _reference_zero(
    decoder_output: Dict[str, torch.Tensor],
    fallback: torch.Tensor,
) -> torch.Tensor:
    for value in decoder_output.values():
        if torch.is_tensor(value):
            return value.new_zeros(())
    return fallback.new_zeros(())


def decoder_reported_v6_losses(
    *,
    decoder_output: Dict[str, torch.Tensor],
    inputs: Dict[str, Any],
    min_active_gate_fraction: float,
) -> Dict[str, torch.Tensor]:
    """Collect differentiable losses already produced inside the decoder."""

    style_condition = torch.as_tensor(inputs["style_value_condition"])
    zero = _reference_zero(decoder_output, style_condition)
    gate = decoder_output.get("axis_router_gate")
    availability = decoder_output.get("axis_router_availability")
    activity_loss = zero
    gate_mean = zero
    active_fraction = zero
    if gate is not None and availability is not None:
        active = availability > 1e-6
        active_fraction = active.float().mean()
        if bool(active.any()):
            minimum = float(min_active_gate_fraction) * availability
            activity_loss = torch.relu(minimum[active] - gate[active]).mean()
            gate_mean = gate[active].mean()

    reported = {
        "axis_router_activity_loss": activity_loss,
        "axis_router_gate_mean": gate_mean,
        "axis_router_active_fraction": active_fraction,
    }
    free_used = decoder_output.get("axis_temporal_free_drive_used")
    coefficient_l2 = decoder_output.get("axis_temporal_coefficient_l2")
    acceleration_rms = decoder_output.get("axis_temporal_acceleration_rms")
    if (
        torch.is_tensor(free_used)
        and torch.is_tensor(coefficient_l2)
        and torch.is_tensor(acceleration_rms)
        and coefficient_l2.ndim == 2
        and acceleration_rms.ndim == 2
        and coefficient_l2.shape[-1] == 3
        and acceleration_rms.shape[-1] == 3
    ):
        free_mask = free_used.to(dtype=torch.bool).reshape(-1)
        reported["axis_temporal_free_drive_ratio"] = free_mask.float().mean().detach()
        axis_names = (
            "speed_utilization",
            "accel_willingness",
            "speed_response",
        )
        for axis_index, axis_name in enumerate(axis_names):
            if bool(free_mask.any()):
                coefficient_mean = coefficient_l2[free_mask, axis_index].mean()
                acceleration_mean = acceleration_rms[free_mask, axis_index].mean()
            else:
                coefficient_mean = zero
                acceleration_mean = zero
            reported[
                f"axis_temporal_free_{axis_name}_coefficient_l2"
            ] = coefficient_mean.detach()
            reported[
                f"axis_temporal_free_{axis_name}_acceleration_rms"
            ] = acceleration_mean.detach()

        profile_cosine = decoder_output.get("axis_temporal_profile_cosine")
        if torch.is_tensor(profile_cosine) and profile_cosine.ndim == 2:
            pair_names = (
                "utilization_accel",
                "utilization_response",
                "accel_response",
            )
            for pair_index, pair_name in enumerate(pair_names):
                value = (
                    profile_cosine[free_mask, pair_index].mean()
                    if bool(free_mask.any())
                    else zero
                )
                reported[
                    f"axis_temporal_free_profile_cosine_{pair_name}"
                ] = value.detach()
    return reported


def _planner_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def compute_normal_reference_prediction(
    *,
    model: torch.nn.Module,
    decoder_output: Dict[str, torch.Tensor],
    inputs: Dict[str, Any],
    model_type: str,
) -> torch.Tensor | None:
    """Evaluate semantic normal on the exact same x_t, t, and scene context."""

    if model_type != "x_start":
        return None
    sampled = decoder_output.get("_training_sampled_trajectories")
    diffusion_time = decoder_output.get("_training_diffusion_time")
    context_encoding = decoder_output.get("_training_context_encoding")
    if sampled is None or diffusion_time is None or context_encoding is None:
        return None

    normal_condition = torch.as_tensor(inputs["style_value_condition"]).clone()
    normal_condition[:, 0:3] = 0.5
    normal_inputs = {
        **inputs,
        "sampled_trajectories": sampled,
        "diffusion_time": diffusion_time,
        "style_value_condition": normal_condition,
    }
    planner = _planner_module(model)
    with torch.no_grad():
        output = planner.decoder(
            {"encoding": context_encoding},
            normal_inputs,
        )
    return output.get("x_start")


def compute_signed_raw_axis_loss(
    *,
    axis_objective: ConditionalPreferenceEnergy | None,
    decoder_output: Dict[str, torch.Tensor],
    inputs: Dict[str, Any],
    beta: float,
) -> Dict[str, torch.Tensor]:
    """Apply the non-saturating V6 raw-axis objective to the main x-start pass."""

    style_condition = torch.as_tensor(inputs["style_value_condition"])
    zero = _reference_zero(decoder_output, style_condition)
    prediction = decoder_output.get("x_start")
    if axis_objective is None or prediction is None:
        return {
            "signed_raw_axis_loss": zero,
            "signed_raw_axis_mae": zero,
            "signed_raw_axis_active_ratio": zero,
        }
    terms = axis_objective.raw_axis_loss_terms(
        prediction,
        inputs,
        beta=float(beta),
    )
    return {
        "signed_raw_axis_loss": terms["signed_raw_axis_loss"],
        "signed_raw_axis_mae": terms["signed_raw_axis_mae"],
        "signed_raw_axis_active_ratio": terms["signed_raw_axis_active_ratio"],
    }


def compute_normal_relative_axis_loss(
    *,
    axis_objective: ConditionalPreferenceEnergy | None,
    decoder_output: Dict[str, torch.Tensor],
    normal_prediction: torch.Tensor | None,
    inputs: Dict[str, Any],
    beta: float,
    fixed_normal_neighbors: bool = False,
) -> Dict[str, torch.Tensor]:
    """Apply Normal-referenced Conditional Quantile Transport (NCQT)."""

    style_condition = torch.as_tensor(inputs["style_value_condition"])
    zero = _reference_zero(decoder_output, style_condition)
    prediction = decoder_output.get("x_start")
    if axis_objective is None or prediction is None or normal_prediction is None:
        empty = {
            "normal_relative_axis_loss": zero,
            "normal_relative_axis_mae": zero,
            "normal_relative_axis_active_ratio": zero,
            "normal_relative_axis_active_count": zero,
        }
        empty.update(_empty_relative_scene_metrics(zero))
        return empty
    terms = axis_objective.normal_relative_axis_loss_terms(
        prediction,
        normal_prediction,
        inputs,
        beta=float(beta),
        fixed_normal_neighbors=bool(fixed_normal_neighbors),
    )
    result = {
        "normal_relative_axis_loss": terms["normal_relative_axis_loss"],
        "normal_relative_axis_mae": terms["normal_relative_axis_mae"],
        "normal_relative_axis_active_ratio": terms[
            "normal_relative_axis_active_ratio"
        ],
        "normal_relative_axis_active_count": terms[
            "normal_relative_axis_active_count"
        ],
    }
    generated_delta = terms["normal_relative_generated_delta"]
    target_delta = terms["normal_relative_target_delta"]
    valid_weight = terms["normal_relative_valid_weight"]
    valid_sample = valid_weight.sum(dim=-1) > 1e-4
    denominator = valid_weight.sum(dim=-1).clamp_min(1e-6)
    sample_mae = (
        (generated_delta - target_delta).abs() * valid_weight
    ).sum(dim=-1) / denominator
    names = _scene_names(inputs, int(generated_delta.shape[0]))
    for scene in CONTROLLED_SCENES:
        scene_mask = torch.as_tensor(
            [name == scene for name in names],
            device=generated_delta.device,
            dtype=torch.bool,
        ) & valid_sample
        count = scene_mask.sum()
        result[f"normal_relative_axis_active_count_{scene}"] = count.to(
            dtype=generated_delta.dtype
        )
        result[f"normal_relative_axis_mae_{scene}"] = (
            sample_mae[scene_mask].mean() if bool(scene_mask.any()) else zero
        )
    return result


def compute_exogenous_neighbor_invariance_loss(
    *,
    decoder_output: Dict[str, torch.Tensor],
    normal_prediction: torch.Tensor | None,
    inputs: Dict[str, Any],
    beta: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Measure direct style leakage into predicted non-ego trajectories.

    Both branches share the exact same noisy joint state.  Consequently this
    term penalizes only direct conditioning leakage at the denoiser, while
    rollout-time interaction responses remain observable in the evaluator.
    """

    style_condition = torch.as_tensor(inputs["style_value_condition"])
    zero = _reference_zero(decoder_output, style_condition)
    prediction = decoder_output.get("x_start")
    if prediction is None or normal_prediction is None or prediction.shape[1] <= 1:
        return {
            "exogenous_neighbor_invariance_loss": zero,
            "exogenous_neighbor_invariance_l1": zero,
            "exogenous_neighbor_active_ratio": zero,
            "exogenous_neighbor_active_count": zero,
        }

    signed_active = (
        ((style_condition[:, 0:3] - 0.5).abs() > 1e-6)
        & (style_condition[:, 3:6] > 0.5)
    ).any(dim=-1)
    neighbor_mask = inputs.get("neighbor_current_mask")
    if neighbor_mask is None:
        neighbor_current = inputs["neighbor_agents_past"][:, :, -1, :4]
        neighbor_mask = torch.sum(neighbor_current != 0, dim=-1) == 0
    valid_neighbor = ~torch.as_tensor(
        neighbor_mask,
        device=prediction.device,
        dtype=torch.bool,
    )[:, : prediction.shape[1] - 1]
    active_entry = signed_active[:, None] & valid_neighbor
    active_sample = active_entry.any(dim=-1)
    if not bool(active_entry.any()):
        return {
            "exogenous_neighbor_invariance_loss": zero,
            "exogenous_neighbor_invariance_l1": zero,
            "exogenous_neighbor_active_ratio": active_sample.float().mean(),
            "exogenous_neighbor_active_count": zero,
        }

    styled_neighbor = prediction[:, 1:, 1:, :]
    normal_neighbor = normal_prediction[:, 1:, 1:, :].detach()
    entry_weight = active_entry[:, :, None, None].to(styled_neighbor.dtype)
    element_count = entry_weight.sum() * styled_neighbor.shape[2] * styled_neighbor.shape[3]
    element_count = element_count.clamp_min(1.0)
    difference = styled_neighbor - normal_neighbor
    loss = (
        F.smooth_l1_loss(
            styled_neighbor,
            normal_neighbor,
            reduction="none",
            beta=max(float(beta), 1e-4),
        )
        * entry_weight
    ).sum() / element_count
    l1 = (difference.abs() * entry_weight).sum() / element_count
    return {
        "exogenous_neighbor_invariance_loss": loss,
        "exogenous_neighbor_invariance_l1": l1,
        "exogenous_neighbor_active_ratio": active_sample.float().mean(),
        "exogenous_neighbor_active_count": active_sample.sum().to(
            dtype=styled_neighbor.dtype
        ),
    }


def _paired_tensor(
    value: Any,
    sample_indices: torch.Tensor,
    *,
    source_batch_size: int,
) -> Any:
    if not torch.is_tensor(value) or value.ndim == 0:
        return value
    if value.shape[0] != source_batch_size:
        return value
    selected = value[sample_indices]
    return torch.cat([selected, selected], dim=0)


def _soft_worst_axis_aggregate(
    hinge_error: torch.Tensor,
    axis_weight: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate valid axes within each sample using a weighted smooth maximum."""

    if hinge_error.ndim != 2 or hinge_error.shape != axis_weight.shape:
        raise ValueError(
            "soft worst-axis aggregation expects matching [sample, axis] tensors"
        )
    smooth_temperature = float(temperature)
    if smooth_temperature <= 0.0:
        raise ValueError("worst-axis temperature must be positive")

    valid_axis = axis_weight > 1e-4
    valid_sample = valid_axis.any(dim=-1)
    if not bool(valid_sample.any()):
        zero = hinge_error.new_zeros(())
        empty_index = torch.zeros(
            hinge_error.shape[0],
            device=hinge_error.device,
            dtype=torch.long,
        )
        return zero, hinge_error.new_zeros((hinge_error.shape[0],)), valid_sample, empty_index

    selected_error = hinge_error[valid_sample]
    selected_weight = axis_weight[valid_sample]
    selected_valid = valid_axis[valid_sample]
    negative_infinity = torch.finfo(hinge_error.dtype).min
    log_weight = torch.where(
        selected_valid,
        torch.log(selected_weight.clamp_min(1e-12)),
        selected_weight.new_full(selected_weight.shape, negative_infinity),
    )
    per_valid_sample = smooth_temperature * (
        torch.logsumexp(
            log_weight + selected_error / smooth_temperature,
            dim=-1,
        )
        - torch.logsumexp(log_weight, dim=-1)
    )
    per_sample = hinge_error.new_zeros((hinge_error.shape[0],))
    per_sample[valid_sample] = per_valid_sample
    masked_error = torch.where(
        valid_axis,
        hinge_error,
        hinge_error.new_full(hinge_error.shape, negative_infinity),
    )
    worst_axis_index = torch.argmax(masked_error, dim=-1)
    return per_valid_sample.mean(), per_sample, valid_sample, worst_axis_index


def selftest_soft_worst_axis_aggregation() -> Dict[str, bool]:
    """Check sample grouping, invalid-axis exclusion, and gradients for A3.6."""

    hinge_error = torch.tensor(
        [[0.00, 0.40, 50.0], [50.0, 0.10, 0.30]],
        dtype=torch.float64,
        requires_grad=True,
    )
    axis_weight = torch.tensor(
        [[1.0, 0.5, 0.0], [0.0, 0.8, 0.4]],
        dtype=torch.float64,
    )
    loss, per_sample, valid_sample, worst_axis = _soft_worst_axis_aggregate(
        hinge_error,
        axis_weight,
        temperature=0.05,
    )
    loss.backward()
    gradient = hinge_error.grad
    assert gradient is not None

    changed_invalid = hinge_error.detach().clone()
    changed_invalid[0, 2] = 5000.0
    changed_invalid[1, 0] = 5000.0
    changed_loss, _, _, _ = _soft_worst_axis_aggregate(
        changed_invalid,
        axis_weight,
        temperature=0.05,
    )
    return {
        "all_samples_retained": bool(valid_sample.tolist() == [True, True]),
        "worst_axis_is_per_sample": bool(worst_axis.tolist() == [1, 2]),
        "invalid_axes_do_not_change_loss": bool(
            torch.allclose(loss.detach(), changed_loss, atol=1e-12, rtol=1e-12)
        ),
        "invalid_axes_have_zero_gradient": bool(
            gradient[0, 2].item() == 0.0 and gradient[1, 0].item() == 0.0
        ),
        "smooth_max_is_bounded_by_hard_max": bool(
            per_sample[0] <= hinge_error.detach()[0, 1]
            and per_sample[1] <= hinge_error.detach()[1, 2]
        ),
    }


def compute_signed_pair_monotonic_loss(
    *,
    model: torch.nn.Module,
    axis_objective: ConditionalPreferenceEnergy | None,
    decoder_output: Dict[str, torch.Tensor],
    normal_prediction: torch.Tensor | None,
    inputs: Dict[str, Any],
    model_type: str,
    delta: float,
    margin: float,
    max_pairs: int,
    random_subset: bool,
    fixed_normal_neighbors: bool = False,
    terminal_t_min: float = 0.0,
    terminal_t_max: float = 0.0,
    short_rollout_steps: int = 0,
    command_mode: str = "axis_isolated",
    aggregation_mode: str = "axis_mean",
    worst_axis_temperature: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Compare plus/minus commands on the same observation and noisy state.

    One pair is created for every eligible (sample, axis) entry, capped by
    ``max_pairs``.  Both branches reuse exactly the same x_t, diffusion time,
    and frozen encoder context, so the only changing variable is the selected
    signed preference coordinate. When ``terminal_t_max > terminal_t_min``, the
    same clean trajectory and recovered noise are re-composed at a low-noise
    terminal time for this paired branch only. A positive
    ``short_rollout_steps`` additionally advances plus/normal/minus through a
    stop-gradient first-order DPM-Solver++ chain before the differentiable
    terminal response pass. This exposes the loss to model-generated states
    without backpropagating through the complete inference solver.

    ``axis_isolated`` changes only the selected target coordinate. In
    ``global_rho`` mode every causally active target moves by the same signed
    delta, matching the scalar-rho command manifold used by evaluation and
    runtime while retaining a separate response loss for each active axis.

    ``axis_mean`` preserves the historical eligible-(sample, axis) sampling and
    weighted mean exactly. ``soft_worst_per_sample`` instead selects eligible
    samples, evaluates all of each sample's causally valid axes in one shared
    global-rho pair, and applies a confidence-weighted LogSumExp smooth maximum
    only across those axes. Symmetry remains a weighted axis mean in both modes.
    """

    style_condition = torch.as_tensor(inputs["style_value_condition"])
    zero = _reference_zero(decoder_output, style_condition)
    sampled = decoder_output.get("_training_sampled_trajectories")
    diffusion_time = decoder_output.get("_training_diffusion_time")
    clean_trajectories = decoder_output.get("_training_clean_trajectories")
    context_encoding = decoder_output.get("_training_context_encoding")
    terminal_pair_enabled = (
        float(terminal_t_max) > float(terminal_t_min) > 0.0
    )
    rollout_steps = max(int(short_rollout_steps), 0)
    pair_command_mode = str(command_mode)
    pair_aggregation_mode = str(aggregation_mode)
    if pair_command_mode not in {"axis_isolated", "global_rho"}:
        raise ValueError(
            "signed pair command_mode must be 'axis_isolated' or 'global_rho'"
        )
    if pair_aggregation_mode not in {"axis_mean", "soft_worst_per_sample"}:
        raise ValueError(
            "signed pair aggregation_mode must be 'axis_mean' or "
            "'soft_worst_per_sample'"
        )
    if (
        pair_aggregation_mode == "soft_worst_per_sample"
        and pair_command_mode != "global_rho"
    ):
        raise ValueError(
            "soft_worst_per_sample aggregation requires global_rho commands"
        )
    if float(worst_axis_temperature) <= 0.0:
        raise ValueError("signed pair worst-axis temperature must be positive")
    if rollout_steps > 0 and not terminal_pair_enabled:
        raise ValueError(
            "short signed-pair rollout requires an enabled low-noise terminal range"
        )
    if (
        axis_objective is None
        or sampled is None
        or diffusion_time is None
        or context_encoding is None
        or normal_prediction is None
        or model_type != "x_start"
    ):
        return _empty_signed_pair_metrics(zero)

    pair_cap = max(int(max_pairs), 1)
    causal_axis_mask = style_condition[:, 3:6] > 0.5
    if pair_aggregation_mode == "soft_worst_per_sample":
        eligible_samples = torch.nonzero(
            causal_axis_mask.any(dim=-1),
            as_tuple=False,
        ).flatten()
        if eligible_samples.numel() == 0:
            return _empty_signed_pair_metrics(zero)
        if eligible_samples.shape[0] > pair_cap:
            if random_subset:
                order = torch.randperm(
                    eligible_samples.shape[0],
                    device=eligible_samples.device,
                )[:pair_cap]
                eligible_samples = eligible_samples[order]
            else:
                eligible_samples = eligible_samples[:pair_cap]
        sample_indices = eligible_samples
        axis_indices = None
    else:
        # Historical path: preserve eligible (sample, axis) selection and cap.
        active_positions = torch.nonzero(causal_axis_mask, as_tuple=False)
        if active_positions.numel() == 0:
            return _empty_signed_pair_metrics(zero)
        if active_positions.shape[0] > pair_cap:
            if random_subset:
                order = torch.randperm(
                    active_positions.shape[0],
                    device=active_positions.device,
                )[:pair_cap]
                active_positions = active_positions[order]
            else:
                active_positions = active_positions[:pair_cap]
        sample_indices = active_positions[:, 0]
        axis_indices = active_positions[:, 1]
    pair_count = int(sample_indices.shape[0])
    base_condition = style_condition[sample_indices].clone()
    base_condition[:, 0:3] = 0.5
    minus_condition = base_condition.clone()
    plus_condition = base_condition.clone()
    command_delta = min(max(float(delta), 1e-3), 0.5)
    row_index = torch.arange(pair_count, device=style_condition.device)
    if pair_command_mode == "global_rho":
        active_pair_mask = base_condition[:, 3:6] > 0.5
        minus_targets = torch.full_like(base_condition[:, 0:3], 0.5)
        plus_targets = torch.full_like(base_condition[:, 0:3], 0.5)
        minus_targets = torch.where(
            active_pair_mask,
            minus_targets - command_delta,
            minus_targets,
        )
        plus_targets = torch.where(
            active_pair_mask,
            plus_targets + command_delta,
            plus_targets,
        )
        minus_condition[:, 0:3] = minus_targets
        plus_condition[:, 0:3] = plus_targets
    else:
        assert axis_indices is not None
        minus_condition[row_index, axis_indices] = 0.5 - command_delta
        plus_condition[row_index, axis_indices] = 0.5 + command_delta
    paired_condition = torch.cat([minus_condition, plus_condition], dim=0)

    planner = _planner_module(model)
    pair_sampled = sampled[sample_indices]
    pair_diffusion_time = diffusion_time[sample_indices]
    pair_normal_prediction = normal_prediction[sample_indices]
    if terminal_pair_enabled:
        if clean_trajectories is None:
            raise RuntimeError(
                "terminal signed-pair supervision requires the private clean "
                "trajectory exported by the StylePlanner diffusion loss"
            )
        pair_clean = clean_trajectories[sample_indices]
        original_mean, original_std = planner.sde.marginal_prob(
            pair_clean[..., 1:, :],
            pair_diffusion_time,
        )
        original_std = original_std.view(
            -1,
            *([1] * (pair_clean[..., 1:, :].ndim - 1)),
        ).clamp_min(1e-6)
        recovered_noise = (
            pair_sampled[..., 1:, :] - original_mean
        ) / original_std

        if rollout_steps > 0:
            # A3.3 begins every truncated chain at the upper terminal bound.
            # Randomness still comes from the recovered, shared forward noise.
            pair_diffusion_time = torch.full(
                (pair_count,),
                float(terminal_t_max),
                device=diffusion_time.device,
                dtype=diffusion_time.dtype,
            )
        elif random_subset:
            pair_diffusion_time = torch.rand(
                pair_count,
                device=diffusion_time.device,
                dtype=diffusion_time.dtype,
            )
            pair_diffusion_time = pair_diffusion_time * (
                float(terminal_t_max) - float(terminal_t_min)
            ) + float(terminal_t_min)
        else:
            pair_diffusion_time = torch.full(
                (pair_count,),
                0.5 * (float(terminal_t_min) + float(terminal_t_max)),
                device=diffusion_time.device,
                dtype=diffusion_time.dtype,
            )
        terminal_mean, terminal_std = planner.sde.marginal_prob(
            pair_clean[..., 1:, :],
            pair_diffusion_time,
        )
        terminal_std = terminal_std.view(
            -1,
            *([1] * (pair_clean[..., 1:, :].ndim - 1)),
        )
        terminal_future = terminal_mean + terminal_std * recovered_noise
        pair_sampled = torch.cat(
            [pair_clean[..., :1, :], terminal_future],
            dim=-2,
        ).detach()

    source_batch_size = int(style_condition.shape[0])
    paired_inputs: Dict[str, Any] = {}
    passthrough_keys = (
        "ego_current_state",
        "neighbor_agents_past",
        "route_lanes",
        "phase_time_mask",
        "preference_ego_current_state_raw",
        "preference_ego_agent_past_raw",
        "preference_neighbor_agents_past_raw",
        "preference_neighbor_agents_past_mask_raw",
        "preference_lanes_raw",
        "preference_lanes_mask_raw",
        "preference_lanes_speed_limit_raw",
        "preference_lanes_has_speed_limit_raw",
        "preference_route_lanes_raw",
        "preference_route_lanes_mask_raw",
        "preference_route_lanes_speed_limit_raw",
        "preference_route_lanes_has_speed_limit_raw",
    )
    for key in passthrough_keys:
        if key in inputs:
            paired_inputs[key] = _paired_tensor(
                inputs[key],
                sample_indices,
                source_batch_size=source_batch_size,
            )
    paired_encoder = {
        "encoding": torch.cat(
            [context_encoding[sample_indices], context_encoding[sample_indices]],
            dim=0,
        )
    }

    normal_pair_inputs = {
        key: value[sample_indices]
        if torch.is_tensor(value)
        and value.ndim > 0
        and value.shape[0] == source_batch_size
        else value
        for key, value in inputs.items()
    }
    normal_pair_inputs.update(
        {
            "style_value_condition": base_condition,
        }
    )
    normal_encoder = {"encoding": context_encoding[sample_indices]}

    paired_sampled = torch.cat([pair_sampled, pair_sampled], dim=0)
    paired_diffusion_time = torch.cat(
        [pair_diffusion_time, pair_diffusion_time],
        dim=0,
    )
    normal_sampled = pair_sampled
    normal_diffusion_time = pair_diffusion_time
    rollout_state_l1 = zero

    if rollout_steps > 0:
        rollout_times = _logsnr_short_rollout_times(
            pair_sampled,
            t_start=float(terminal_t_max),
            t_end=float(terminal_t_min),
            steps=rollout_steps,
        )
        paired_fixed_current = paired_sampled[..., :1, :].detach()
        normal_fixed_current = normal_sampled[..., :1, :].detach()
        for rollout_index in range(rollout_steps):
            paired_diffusion_time = torch.full(
                (2 * pair_count,),
                float(rollout_times[rollout_index]),
                device=diffusion_time.device,
                dtype=diffusion_time.dtype,
            )
            normal_diffusion_time = torch.full(
                (pair_count,),
                float(rollout_times[rollout_index]),
                device=diffusion_time.device,
                dtype=diffusion_time.dtype,
            )
            paired_inputs.update(
                {
                    "sampled_trajectories": paired_sampled,
                    "diffusion_time": paired_diffusion_time,
                    "style_value_condition": paired_condition,
                }
            )
            normal_pair_inputs.update(
                {
                    "sampled_trajectories": normal_sampled,
                    "diffusion_time": normal_diffusion_time,
                }
            )
            # Earlier states act as a bootstrapped target. Only the final
            # terminal response pass below carries gradients, which bounds
            # memory while still matching inference-time state evolution.
            with torch.no_grad():
                paired_rollout_output = planner.decoder(
                    paired_encoder,
                    paired_inputs,
                )
                normal_rollout_output = planner.decoder(
                    normal_encoder,
                    normal_pair_inputs,
                )
                paired_rollout_x_start = paired_rollout_output.get("x_start")
                normal_rollout_x_start = normal_rollout_output.get("x_start")
                if paired_rollout_x_start is None or normal_rollout_x_start is None:
                    raise RuntimeError(
                        "short signed-pair rollout requires x_start from every decoder pass"
                    )
                next_pair_time = torch.full_like(
                    paired_diffusion_time,
                    float(rollout_times[rollout_index + 1]),
                )
                next_normal_time = torch.full_like(
                    normal_diffusion_time,
                    float(rollout_times[rollout_index + 1]),
                )
                paired_sampled = _dpmpp_first_order_state_step(
                    paired_sampled,
                    paired_rollout_x_start,
                    t_start=paired_diffusion_time,
                    t_end=next_pair_time,
                    fixed_current=paired_fixed_current,
                ).detach()
                normal_sampled = _dpmpp_first_order_state_step(
                    normal_sampled,
                    normal_rollout_x_start,
                    t_start=normal_diffusion_time,
                    t_end=next_normal_time,
                    fixed_current=normal_fixed_current,
                ).detach()

        paired_diffusion_time = torch.full(
            (2 * pair_count,),
            float(rollout_times[-1]),
            device=diffusion_time.device,
            dtype=diffusion_time.dtype,
        )
        normal_diffusion_time = torch.full(
            (pair_count,),
            float(rollout_times[-1]),
            device=diffusion_time.device,
            dtype=diffusion_time.dtype,
        )
        normal_for_pair = torch.cat([normal_sampled, normal_sampled], dim=0)
        rollout_state_l1 = (
            paired_sampled[:, 0, 1:, :] - normal_for_pair[:, 0, 1:, :]
        ).abs().mean()

    paired_inputs.update(
        {
            "sampled_trajectories": paired_sampled,
            "diffusion_time": paired_diffusion_time,
            "style_value_condition": paired_condition,
        }
    )
    if terminal_pair_enabled:
        normal_pair_inputs.update(
            {
                "sampled_trajectories": normal_sampled,
                "diffusion_time": normal_diffusion_time,
            }
        )
        with torch.no_grad():
            normal_pair_output = planner.decoder(
                normal_encoder,
                normal_pair_inputs,
            )
        pair_normal_prediction = normal_pair_output.get("x_start")
        if pair_normal_prediction is None:
            raise RuntimeError(
                "terminal signed-pair normal branch did not return x_start"
            )

    paired_output = planner.decoder(paired_encoder, paired_inputs)
    paired_prediction = paired_output.get("x_start")
    if paired_prediction is None:
        return _empty_signed_pair_metrics(zero)
    prepared = axis_objective.prepare(paired_inputs, paired_condition)
    coordinate, confidence, _ = axis_objective.raw_axis_coordinates(
        paired_prediction,
        prepared,
        neighbor_reference_output=(
            torch.cat(
                [
                    pair_normal_prediction,
                    pair_normal_prediction,
                ],
                dim=0,
            )
            if fixed_normal_neighbors
            else None
        ),
        accel_opportunity_reference_output=torch.cat(
            [
                pair_normal_prediction,
                pair_normal_prediction,
            ],
            dim=0,
        ),
    )
    minus_coordinate = coordinate[:pair_count]
    plus_coordinate = coordinate[pair_count:]

    normal_inputs = {
        key: value[sample_indices]
        if torch.is_tensor(value)
        and value.ndim > 0
        and value.shape[0] == source_batch_size
        else value
        for key, value in inputs.items()
    }
    normal_inputs["style_value_condition"] = base_condition
    normal_prepared = axis_objective.prepare(normal_inputs, base_condition)
    normal_coordinate, normal_confidence, _ = axis_objective.raw_axis_coordinates(
        pair_normal_prediction,
        normal_prepared,
        neighbor_reference_output=(
            pair_normal_prediction if fixed_normal_neighbors else None
        ),
        accel_opportunity_reference_output=pair_normal_prediction,
    )
    active_pair_mask = normal_prepared["axis_mask"].bool()
    if pair_aggregation_mode == "soft_worst_per_sample":
        response_matrix = plus_coordinate - minus_coordinate
        pair_weight_matrix = (
            normal_confidence.detach() * active_pair_mask.to(normal_confidence.dtype)
        )
        valid_matrix = pair_weight_matrix > 1e-4
        hinge_error = torch.relu(float(margin) - response_matrix)
        loss, _, valid_sample, worst_axis_index = _soft_worst_axis_aggregate(
            hinge_error,
            pair_weight_matrix,
            temperature=float(worst_axis_temperature),
        )
        if not bool(valid_sample.any()):
            return _empty_signed_pair_metrics(zero)
        valid_weight = pair_weight_matrix[valid_matrix]
        denominator = valid_weight.sum().clamp_min(1e-6)
        order_accuracy = (
            (response_matrix[valid_matrix] > 0.0).to(valid_weight.dtype)
            * valid_weight
        ).sum() / denominator
        response_mean = (
            response_matrix[valid_matrix] * valid_weight
        ).sum() / denominator
        positive_response_matrix = plus_coordinate - normal_coordinate.detach()
        negative_response_matrix = minus_coordinate - normal_coordinate.detach()
        symmetry = F.smooth_l1_loss(
            positive_response_matrix[valid_matrix],
            -negative_response_matrix[valid_matrix],
            reduction="none",
            beta=max(float(margin), 1e-4),
        )
        symmetry_loss = (symmetry * valid_weight).sum() / denominator
        sample_count = valid_sample.sum()
    else:
        # Historical axis-mean objective: keep all indexing and reductions
        # unchanged so A3.2--A3.5 presets/checkpoints retain their behavior.
        assert axis_indices is not None
        selected_minus = minus_coordinate[row_index, axis_indices]
        selected_plus = plus_coordinate[row_index, axis_indices]
        selected_normal = normal_coordinate[row_index, axis_indices].detach()
        pair_weight = normal_confidence[row_index, axis_indices].detach()
        valid = pair_weight > 1e-4
        if not bool(valid.any()):
            return _empty_signed_pair_metrics(zero)

        response = selected_plus - selected_minus
        valid_weight = pair_weight[valid]
        denominator = valid_weight.sum().clamp_min(1e-6)
        loss = (
            torch.relu(float(margin) - response[valid]) * valid_weight
        ).sum() / denominator
        order_accuracy = (
            (response[valid] > 0.0).to(valid_weight.dtype) * valid_weight
        ).sum() / denominator
        response_mean = (response[valid] * valid_weight).sum() / denominator
        positive_response = selected_plus - selected_normal
        negative_response = selected_minus - selected_normal
        symmetry = F.smooth_l1_loss(
            positive_response[valid],
            -negative_response[valid],
            reduction="none",
            beta=max(float(margin), 1e-4),
        )
        symmetry_loss = (symmetry * valid_weight).sum() / denominator
        response_matrix = response.new_zeros((pair_count, 3))
        pair_weight_matrix = pair_weight.new_zeros((pair_count, 3))
        valid_matrix = torch.zeros(
            (pair_count, 3),
            device=response.device,
            dtype=torch.bool,
        )
        response_matrix[row_index, axis_indices] = response
        pair_weight_matrix[row_index, axis_indices] = pair_weight
        valid_matrix[row_index, axis_indices] = valid
        valid_sample = valid_matrix.any(dim=-1)
        worst_axis_index = torch.zeros(
            pair_count,
            device=response.device,
            dtype=torch.long,
        )
        sample_count = torch.unique(sample_indices[valid]).numel()

    axis_pair_count = valid_matrix.sum()
    result = {
        "signed_monotonic_loss": loss,
        "signed_monotonic_order_accuracy": order_accuracy,
        "signed_monotonic_response": response_mean,
        "signed_symmetry_loss": symmetry_loss,
        "signed_monotonic_pair_count": axis_pair_count.to(dtype=zero.dtype),
        "signed_monotonic_sample_count": zero.new_tensor(float(sample_count)),
        "signed_monotonic_axis_pair_count": axis_pair_count.to(dtype=zero.dtype),
        "signed_monotonic_worst_axis_loss": (
            loss if pair_aggregation_mode == "soft_worst_per_sample" else zero
        ),
        "signed_monotonic_worst_axis_temperature": zero.new_tensor(
            float(worst_axis_temperature)
        ),
        "signed_monotonic_soft_worst_enabled": zero.new_tensor(
            float(pair_aggregation_mode == "soft_worst_per_sample")
        ),
        "signed_monotonic_diffusion_time_mean": normal_diffusion_time.mean(),
        "signed_monotonic_diffusion_time_min": normal_diffusion_time.min(),
        "signed_monotonic_diffusion_time_max": normal_diffusion_time.max(),
        "signed_monotonic_short_rollout_steps": zero.new_tensor(
            float(rollout_steps)
        ),
        "signed_monotonic_rollout_state_l1": rollout_state_l1,
    }
    pair_scene_index = normal_prepared["scene_index"]
    for scene_index, scene in enumerate(CONTROLLED_SCENES):
        scene_sample = pair_scene_index == scene_index
        scene_valid = valid_matrix & scene_sample[:, None]
        count = scene_valid.sum()
        result[f"signed_monotonic_pair_count_{scene}"] = count.to(
            dtype=valid_weight.dtype
        )
        if bool(scene_valid.any()):
            scene_weight = pair_weight_matrix[scene_valid]
            scene_denominator = scene_weight.sum().clamp_min(1e-6)
            result[f"signed_monotonic_order_accuracy_{scene}"] = (
                (response_matrix[scene_valid] > 0.0).to(scene_weight.dtype)
                * scene_weight
            ).sum() / scene_denominator
            result[f"signed_monotonic_response_{scene}"] = (
                response_matrix[scene_valid] * scene_weight
            ).sum() / scene_denominator
        else:
            result[f"signed_monotonic_order_accuracy_{scene}"] = zero
            result[f"signed_monotonic_response_{scene}"] = zero
        scene_valid_sample = valid_sample & scene_sample
        scene_valid_sample_count = scene_valid_sample.sum().clamp_min(1)
        for axis_index, axis_name in enumerate(
            PAIR_AXIS_DIAGNOSTIC_NAMES[scene]
        ):
            axis_valid = scene_valid[:, axis_index]
            result[f"{axis_name}_pair_count"] = axis_valid.sum().to(
                dtype=zero.dtype
            )
            if bool(axis_valid.any()):
                axis_weight = pair_weight_matrix[axis_valid, axis_index]
                axis_denominator = axis_weight.sum().clamp_min(1e-6)
                axis_response = response_matrix[axis_valid, axis_index]
                result[f"{axis_name}_order"] = (
                    (axis_response > 0.0).to(axis_weight.dtype) * axis_weight
                ).sum() / axis_denominator
                result[f"{axis_name}_mean_response"] = (
                    axis_response * axis_weight
                ).sum() / axis_denominator
            else:
                result[f"{axis_name}_order"] = zero
                result[f"{axis_name}_mean_response"] = zero
            if (
                pair_aggregation_mode == "soft_worst_per_sample"
                and bool(scene_valid_sample.any())
            ):
                result[f"{axis_name}_worst_ratio"] = (
                    (worst_axis_index[scene_valid_sample] == axis_index)
                    .to(zero.dtype)
                    .sum()
                    / scene_valid_sample_count.to(zero.dtype)
                )
            else:
                result[f"{axis_name}_worst_ratio"] = zero
    return result


def compute_normal_anchor_consistency_loss(
    *,
    model: torch.nn.Module,
    decoder_output: Dict[str, torch.Tensor],
    inputs: Dict[str, Any],
    model_type: str,
) -> Dict[str, torch.Tensor]:
    """Anchor semantic rho=0 to the empty/base planner on the same noisy state.

    Empty and normal conditions are evaluated together in one additional
    decoder pass. The empty branch is stop-gradient: the loss adapts only the
    semantic-normal/style path and cannot drag the base diffusion behavior.
    """

    style_condition = torch.as_tensor(inputs["style_value_condition"])
    zero = _reference_zero(decoder_output, style_condition)
    normal_condition = inputs.get("normal_anchor_style_value_condition")
    sampled = decoder_output.get("_training_sampled_trajectories")
    diffusion_time = decoder_output.get("_training_diffusion_time")
    context_encoding = decoder_output.get("_training_context_encoding")
    if (
        normal_condition is None
        or sampled is None
        or diffusion_time is None
        or context_encoding is None
    ):
        return {
            "normal_anchor_consistency_loss": zero,
            "normal_anchor_active_ratio": zero,
            "normal_anchor_prediction_l1": zero,
        }

    normal_condition = torch.as_tensor(
        normal_condition,
        device=sampled.device,
        dtype=sampled.dtype,
    )
    active = torch.any(normal_condition.abs() > 1e-6, dim=-1)
    if not bool(active.any()):
        return {
            "normal_anchor_consistency_loss": zero,
            "normal_anchor_active_ratio": active.float().mean(),
            "normal_anchor_prediction_l1": zero,
        }

    active_indices = torch.where(active)[0]
    active_normal_condition = normal_condition[active_indices]
    active_sampled = sampled[active_indices]
    active_diffusion_time = diffusion_time[active_indices]
    active_context_encoding = context_encoding[active_indices]
    active_batch_size = int(active_indices.shape[0])
    empty_condition = torch.zeros_like(active_normal_condition)
    paired_condition = torch.cat(
        [active_normal_condition, empty_condition],
        dim=0,
    )
    paired_inputs: Dict[str, Any] = {
        "ego_current_state": torch.cat(
            [inputs["ego_current_state"][active_indices]] * 2,
            dim=0,
        ),
        "neighbor_agents_past": torch.cat(
            [inputs["neighbor_agents_past"][active_indices]] * 2,
            dim=0,
        ),
        "route_lanes": torch.cat(
            [inputs["route_lanes"][active_indices]] * 2,
            dim=0,
        ),
        "sampled_trajectories": torch.cat([active_sampled] * 2, dim=0),
        "diffusion_time": torch.cat([active_diffusion_time] * 2, dim=0),
        "style_value_condition": paired_condition,
    }
    phase_time_mask = inputs.get("phase_time_mask")
    if torch.is_tensor(phase_time_mask):
        active_phase_time_mask = phase_time_mask[active_indices]
        paired_inputs["phase_time_mask"] = torch.cat(
            [active_phase_time_mask] * 2,
            dim=0,
        )
    paired_encoder = {
        "encoding": torch.cat([active_context_encoding] * 2, dim=0),
    }

    planner = _planner_module(model)
    if not hasattr(planner, "decoder"):
        raise TypeError("Normal-anchor loss requires the StylePlanner decoder wrapper")
    dit_module = getattr(getattr(planner.decoder, "decoder", None), "dit", None)
    style_modules = [
        getattr(dit_module, "style_condition_proj", None),
        getattr(dit_module, "ego_style_output_proj", None),
    ]
    router_parameter_ids = {
        id(parameter)
        for module in style_modules
        if module is not None
        for parameter in module.parameters()
    }
    temporarily_frozen = []
    for parameter in planner.parameters():
        if parameter.requires_grad and id(parameter) not in router_parameter_ids:
            parameter.requires_grad_(False)
            temporarily_frozen.append(parameter)
    try:
        paired_output = planner.decoder(paired_encoder, paired_inputs)
    finally:
        for parameter in temporarily_frozen:
            parameter.requires_grad_(True)
    prediction = _extract_diffusion_prediction(
        paired_output,
        model_type=model_type,
        future_steps=sampled.shape[2] - 1,
    )
    normal_prediction = prediction[:active_batch_size, 0]
    empty_prediction = prediction[active_batch_size:, 0].detach()
    difference = normal_prediction - empty_prediction
    loss = F.smooth_l1_loss(
        normal_prediction,
        empty_prediction,
        beta=0.05,
    )
    return {
        "normal_anchor_consistency_loss": loss,
        "normal_anchor_active_ratio": active.float().mean(),
        "normal_anchor_prediction_l1": difference.abs().mean(),
    }

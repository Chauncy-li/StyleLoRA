"""目标风格条件 LoRA 的事实监督训练。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from stylelora.lora.model.style_lora_planner import StyleLoRAPlanner
from stylelora.lora.safety.trajectory_acceptance import (
    cached_relative_feasibility_mask,
    differentiable_relative_feasibility_penalty,
)
from stylelora.lora.training.losses import build_noisy_inputs
from stylelora.model.preference_encoder import CSPQPreferenceEncoder
from stylelora.training.preference_lora import (
    _build_pred_tokens,
    _dynamics_consistency_loss,
    _lateral_residual_loss,
    _masked_factor_huber,
    _mmd_rbf_biased,
    _prediction,
)


def _weighted_rank_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    loss = F.huber_loss(prediction, target, reduction="none")
    weight = weight.clamp_min(0.0)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-12)


def ordered_pair_objective(
    low_s: torch.Tensor,
    high_s: torch.Tensor,
    confidence: torch.Tensor,
    feasible_pair: torch.Tensor,
    margin: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """在真正参与排序的 feasible 权重和上归一化有序间隔损失。"""
    weight = confidence.clamp_min(0.0) * feasible_pair.to(confidence)
    items = F.relu(margin - (high_s - low_s)).square()
    denominator = weight.sum().clamp_min(1e-12)
    loss = (items * weight).sum() / denominator
    violation = (((high_s - low_s) <= 0).to(weight) * weight).sum() / denominator
    return {
        "loss": loss,
        "violation_rate": violation.detach(),
        "valid_weight": weight.sum().detach(),
        "valid_count": feasible_pair.sum().to(confidence).detach(),
    }


def _route_aligned_progress(
    ego: torch.Tensor,
    baseline_ego: torch.Tensor,
    current_xy: torch.Tensor,
) -> torch.Tensor:
    """Differentiable longitudinal progress measured along the baseline heading."""
    if ego.shape != baseline_ego.shape or ego.ndim != 3 or ego.shape[-1] < 4:
        raise ValueError("ego and baseline_ego must match [B,T,D] with D >= 4")
    if current_xy.ndim != 3 or current_xy.shape[:2] != (ego.shape[0], 1):
        raise ValueError("current_xy must have shape [B,1,2]")
    tangent = F.normalize(baseline_ego[..., 2:4], dim=-1, eps=1e-6)
    xy = torch.cat((current_xy.to(ego), ego[..., :2]), dim=1)
    increments = torch.diff(xy, dim=1)
    return (increments * tangent).sum(dim=-1).sum(dim=-1)


def longitudinal_response_objective(
    low_ego: torch.Tensor,
    high_ego: torch.Tensor,
    baseline_ego: torch.Tensor,
    current_xy: torch.Tensor,
    confidence: torch.Tensor,
    feasible_pair: torch.Tensor,
    rho_gap: torch.Tensor,
    *,
    margin_m_per_rho: float,
    min_baseline_progress_m: float,
) -> dict[str, torch.Tensor]:
    """Require a visible physical progress gap between ordered rho predictions."""
    if margin_m_per_rho < 0 or min_baseline_progress_m < 0:
        raise ValueError("response margins must be non-negative")
    low_progress = _route_aligned_progress(low_ego, baseline_ego, current_xy)
    high_progress = _route_aligned_progress(high_ego, baseline_ego, current_xy)
    baseline_progress = _route_aligned_progress(
        baseline_ego, baseline_ego, current_xy
    ).detach()
    response_gap = high_progress - low_progress
    active = feasible_pair & (baseline_progress >= float(min_baseline_progress_m))
    weight = confidence.clamp_min(0.0) * active.to(confidence)
    denominator = weight.sum().clamp_min(1e-12)
    required_gap = float(margin_m_per_rho) * rho_gap
    items = F.relu(required_gap - response_gap).square()
    loss = (items * weight).sum() / denominator
    violation = ((response_gap <= 0).to(weight) * weight).sum() / denominator
    return {
        "loss": loss,
        "violation_rate": violation.detach(),
        "mean_gap_m": ((response_gap * weight).sum() / denominator).detach(),
        "active_rate": active.to(confidence).mean().detach(),
        "valid_count": active.sum().to(confidence).detach(),
        "valid_weight": weight.sum().detach(),
    }


def _sample_ordered_rho_pairs(
    reference: torch.Tensor,
    *,
    mode: str,
    rho_grid: tuple[float, ...],
    local_order_ratio: float,
    min_rho_gap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """采样跨 Low/High 或同一连续网格内的相邻有序强度对。"""
    half_gap = float(min_rho_gap) / 2.0
    if not 0.0 < min_rho_gap <= 2.0:
        raise ValueError("min_rho_gap 必须位于 (0,2]")
    cross_low = -(half_gap + (1.0 - half_gap) * torch.rand_like(reference))
    cross_high = half_gap + (1.0 - half_gap) * torch.rand_like(reference)
    if mode == "cross":
        return cross_low, cross_high, torch.zeros_like(reference, dtype=torch.bool)
    if mode != "mixed_adjacent":
        raise ValueError("order_pair_mode 必须是 cross 或 mixed_adjacent")
    if not 0.0 <= local_order_ratio <= 1.0:
        raise ValueError("local_order_ratio 必须位于 [0,1]")
    grid = torch.as_tensor(rho_grid, dtype=reference.dtype, device=reference.device)
    if grid.ndim != 1 or grid.numel() < 3:
        raise ValueError("order_rho_grid 至少包含三个严格递增强度")
    if not bool(torch.all(grid[1:] > grid[:-1])) or grid[0] < -1 or grid[-1] > 1:
        raise ValueError("order_rho_grid 必须在 [-1,1] 内严格递增")
    if not bool(torch.any(grid == 0)):
        raise ValueError("order_rho_grid 必须包含 rho=0")
    interval = torch.randint(0, grid.numel() - 1, reference.shape, device=reference.device)
    adjacent_low = grid[interval]
    adjacent_high = grid[interval + 1]
    use_local = torch.rand_like(reference) < float(local_order_ratio)
    return (
        torch.where(use_local, adjacent_low, cross_low),
        torch.where(use_local, adjacent_high, cross_high),
        use_local,
    )


def _masked_rate(condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=torch.float32)
    return (condition.to(weight) * weight).sum() / weight.sum().clamp_min(1.0)


def feasible_style_pair_objective(
    preferred_error: torch.Tensor,
    rejected_error: torch.Tensor,
    confidence: torch.Tensor,
    *,
    margin: float,
) -> dict[str, torch.Tensor]:
    """计算可行风格配对的锚定项与有限间隔排序项。"""
    weight = confidence.clamp_min(0.0)
    denominator = weight.sum().clamp_min(1e-12)
    anchor = (preferred_error * weight).sum() / denominator
    ranking_items = F.relu(float(margin) + preferred_error - rejected_error)
    ranking = (ranking_items * weight).sum() / denominator
    accuracy = (
        ((rejected_error - preferred_error) >= float(margin)).to(weight) * weight
    ).sum() / denominator
    return {
        "loss": anchor + ranking,
        "anchor": anchor,
        "ranking": ranking,
        "accuracy": accuracy.detach(),
        "gap": ((rejected_error - preferred_error) * weight).sum().detach() / denominator,
    }


def feasible_style_pair_loss(
    *,
    planner: StyleLoRAPlanner,
    inputs: dict,
    futures: tuple,
    marginal_prob,
    state_normalizer,
    batch: dict,
    margin: float,
) -> dict[str, torch.Tensor]:
    """用可行 style 轨迹提供小权重锚定与有限间隔排序监督。"""
    _, neighbors, neighbor_mask = futures
    preferred_futures = (batch["preferred_ego"], neighbors, neighbor_mask)
    rejected_futures = (batch["rejected_ego"], neighbors, neighbor_mask)
    batch_size = batch["requested_rho"].shape[0]
    time = torch.rand(batch_size, device=batch["requested_rho"].device) * (1.0 - 1e-3) + 1e-3
    full_future = torch.cat((batch["preferred_ego"][:, None], neighbors), dim=1)
    noise = torch.randn_like(full_future)
    preferred_inputs, preferred_target, _ = build_noisy_inputs(
        inputs, preferred_futures, marginal_prob, state_normalizer, time=time, noise=noise
    )
    rejected_inputs, rejected_target, _ = build_noisy_inputs(
        inputs, rejected_futures, marginal_prob, state_normalizer, time=time, noise=noise
    )

    planner.set_conditional_coordinate(batch["requested_rho"])
    _, preferred_output = planner(preferred_inputs)
    preferred_prediction = _prediction(preferred_output, preferred_target.shape[2])
    _, rejected_output = planner(rejected_inputs)
    rejected_prediction = _prediction(rejected_output, rejected_target.shape[2])
    preferred_error = (preferred_prediction[:, 0] - preferred_target[:, 0]).square().mean(dim=(1, 2))
    rejected_error = (rejected_prediction[:, 0] - rejected_target[:, 0]).square().mean(dim=(1, 2))
    return feasible_style_pair_objective(
        preferred_error,
        rejected_error,
        batch["pair_confidence"],
        margin=margin,
    )


def conditional_preference_lora_loss(
    *,
    planner: StyleLoRAPlanner,
    cspq: CSPQPreferenceEncoder,
    inputs: dict,
    futures: tuple,
    marginal_prob,
    state_normalizer,
    batch: dict,
    lambda_n: float = 1.0,
    lambda_z: float = 1.0,
    lambda_s: float = 1.0,
    lambda_q: float = 1.0,
    lambda_dyn: float = 0.1,
    lambda_lat: float = 1.0,
    lambda_order: float = 1.0,
    lambda_feasibility: float = 0.0,
    lambda_response: float = 0.0,
    order_margin_scale: float = 0.25,
    min_rho_gap: float = 0.25,
    order_pair_mode: str = "cross",
    order_rho_grid: tuple[float, ...] = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0),
    local_order_ratio: float = 0.75,
    lateral_tolerance: float = 0.3,
    lateral_topk_ratio: float = 0.2,
    lateral_smooth_weight: float = 0.1,
    feasible_max_mean_accel_degradation: float = 0.5,
    feasible_max_mean_jerk_degradation: float = 2.0,
    feasible_max_progress_loss_m: float = 2.0,
    feasible_max_mean_lateral_deviation_m: float = 1.5,
    response_margin_m_per_rho: float = 1.0,
    response_min_baseline_progress_m: float = 3.0,
) -> dict[str, torch.Tensor]:
    """用事实轨迹训练连续条件策略，并保持通用规划能力。"""
    noisy_inputs, target, neighbors_valid = build_noisy_inputs(
        inputs, futures, marginal_prob, state_normalizer
    )
    signed = (2.0 * (batch["rank"] - 0.5)).clamp(-1.0, 1.0)
    # 训练和推理统一由 Low/Neutral/High 原型构造目标条件。
    planner.set_conditional_coordinate(signed)
    _, adaptive_output = planner(noisy_inputs)
    adaptive = _prediction(adaptive_output, target.shape[2])

    with torch.no_grad():
        planner.disable_adapter()
        _, base_output = planner(noisy_inputs)
        planner.enable_adapter()
        base = _prediction(base_output, target.shape[2])

    physical_adaptive = state_normalizer.inverse(adaptive)
    physical_base = state_normalizer.inverse(base)
    physical_ego = physical_adaptive[:, 0]
    physical_base_ego = physical_base[:, 0]
    preference = cspq(_build_pred_tokens(physical_ego), batch["h_c"])
    z_pred = preference["z"]
    s_pred = preference["s"].squeeze(-1)

    ego_denoise = ((adaptive[:, 0] - target[:, 0]) ** 2).sum(dim=-1).mean()
    neighbor = (
        ((adaptive[:, 1:] - base[:, 1:]) ** 2).sum(dim=-1)[neighbors_valid].mean()
        if neighbors_valid.any()
        else adaptive.new_zeros(())
    )
    mmd_loss = _mmd_rbf_biased(z_pred, batch["z_target"])
    rank_huber = _weighted_rank_huber(s_pred, batch["rank"], batch["confidence"])
    factor_huber = _masked_factor_huber(
        preference["q_hat"], batch["q_vec"], batch["valid_mask"]
    )

    current_xy = batch["tensors"]["ego_current_state"][:, None, :2].to(physical_ego)
    expert_ego = futures[0].to(physical_ego.device)
    pred_xy = torch.cat((current_xy, physical_ego[..., :2]), dim=1)
    gt_xy = torch.cat((current_xy, expert_ego[..., :2]), dim=1)
    dynamics = _dynamics_consistency_loss(pred_xy, gt_xy)
    lateral = _lateral_residual_loss(
        physical_ego,
        physical_base_ego,
        expert_ego,
        current_xy,
        tolerance=lateral_tolerance,
        topk_ratio=lateral_topk_ratio,
        smooth_weight=lateral_smooth_weight,
    )
    adaptive_feasibility = differentiable_relative_feasibility_penalty(
        physical_ego,
        physical_base_ego,
        current_xy,
        max_mean_accel_degradation=feasible_max_mean_accel_degradation,
        max_mean_jerk_degradation=feasible_max_mean_jerk_degradation,
        max_progress_loss_m=feasible_max_progress_loss_m,
        max_mean_lateral_deviation_m=feasible_max_mean_lateral_deviation_m,
    )

    # V4 可在同一计算预算内混合跨 Low/High 与同方向相邻强度对。
    low_rho, high_rho, local_pair = _sample_ordered_rho_pairs(
        signed,
        mode=order_pair_mode,
        rho_grid=tuple(float(value) for value in order_rho_grid),
        local_order_ratio=local_order_ratio,
        min_rho_gap=min_rho_gap,
    )
    rho_gap = high_rho - low_rho

    planner.set_conditional_coordinate(low_rho)
    _, low_output = planner(noisy_inputs)
    low_prediction = _prediction(low_output, target.shape[2])
    planner.set_conditional_coordinate(high_rho)
    _, high_output = planner(noisy_inputs)
    high_prediction = _prediction(high_output, target.shape[2])
    planner.set_conditional_coordinate(signed)

    physical_low = state_normalizer.inverse(low_prediction)
    physical_high = state_normalizer.inverse(high_prediction)
    low_ego = physical_low[:, 0]
    high_ego = physical_high[:, 0]
    low_s = cspq(_build_pred_tokens(low_ego), batch["h_c"])["s"].squeeze(-1)
    high_s = cspq(_build_pred_tokens(high_ego), batch["h_c"])["s"].squeeze(-1)
    low_feasible = cached_relative_feasibility_mask(
        low_ego,
        physical_base_ego,
        current_xy,
        max_mean_accel_degradation=feasible_max_mean_accel_degradation,
        max_mean_jerk_degradation=feasible_max_mean_jerk_degradation,
        max_progress_loss_m=feasible_max_progress_loss_m,
        max_mean_lateral_deviation_m=feasible_max_mean_lateral_deviation_m,
    )
    high_feasible = cached_relative_feasibility_mask(
        high_ego,
        physical_base_ego,
        current_xy,
        max_mean_accel_degradation=feasible_max_mean_accel_degradation,
        max_mean_jerk_degradation=feasible_max_mean_jerk_degradation,
        max_progress_loss_m=feasible_max_progress_loss_m,
        max_mean_lateral_deviation_m=feasible_max_mean_lateral_deviation_m,
    )
    feasible_pair = low_feasible & high_feasible
    order_result = ordered_pair_objective(
        low_s,
        high_s,
        batch["confidence"],
        feasible_pair,
        float(order_margin_scale) * rho_gap,
    )
    order = order_result["loss"]
    order_violation = order_result["violation_rate"]
    if lambda_response > 0:
        response_result = longitudinal_response_objective(
            low_ego,
            high_ego,
            physical_base_ego,
            current_xy,
            batch["confidence"],
            feasible_pair,
            rho_gap,
            margin_m_per_rho=response_margin_m_per_rho,
            min_baseline_progress_m=response_min_baseline_progress_m,
        )
    else:
        # Preserve the V4 computation path exactly when the optional loss is disabled.
        zero = adaptive.new_zeros(())
        response_result = {
            "loss": zero,
            "violation_rate": zero.detach(),
            "mean_gap_m": zero.detach(),
            "active_rate": zero.detach(),
            "valid_count": zero.detach(),
            "valid_weight": zero.detach(),
        }
    low_feasibility = differentiable_relative_feasibility_penalty(
        low_ego,
        physical_base_ego,
        current_xy,
        max_mean_accel_degradation=feasible_max_mean_accel_degradation,
        max_mean_jerk_degradation=feasible_max_mean_jerk_degradation,
        max_progress_loss_m=feasible_max_progress_loss_m,
        max_mean_lateral_deviation_m=feasible_max_mean_lateral_deviation_m,
    )
    high_feasibility = differentiable_relative_feasibility_penalty(
        high_ego,
        physical_base_ego,
        current_xy,
        max_mean_accel_degradation=feasible_max_mean_accel_degradation,
        max_mean_jerk_degradation=feasible_max_mean_jerk_degradation,
        max_progress_loss_m=feasible_max_progress_loss_m,
        max_mean_lateral_deviation_m=feasible_max_mean_lateral_deviation_m,
    )
    feasibility = (
        adaptive_feasibility["loss"]
        + low_feasibility["loss"]
        + high_feasibility["loss"]
    ) / 3.0
    paired_low_lateral = _lateral_residual_loss(
        low_ego,
        physical_base_ego,
        expert_ego,
        current_xy,
        tolerance=lateral_tolerance,
        topk_ratio=lateral_topk_ratio,
        smooth_weight=lateral_smooth_weight,
    )["lateral"]
    paired_high_lateral = _lateral_residual_loss(
        high_ego,
        physical_base_ego,
        expert_ego,
        current_xy,
        tolerance=lateral_tolerance,
        topk_ratio=lateral_topk_ratio,
        smooth_weight=lateral_smooth_weight,
    )["lateral"]
    paired_lateral = 0.5 * (paired_low_lateral + paired_high_lateral)

    total = (
        ego_denoise
        + lambda_n * neighbor
        + lambda_z * mmd_loss
        + lambda_s * rank_huber
        + lambda_q * factor_huber
        + lambda_dyn * dynamics["dynamics"]
        + lambda_lat * (lateral["lateral"] + paired_lateral)
        + lambda_order * order
        + lambda_feasibility * feasibility
        + lambda_response * response_result["loss"]
    )
    local_valid = feasible_pair & local_pair
    cross_valid = feasible_pair & ~local_pair
    return {
        "loss": total,
        "ego_denoise": ego_denoise,
        "neighbor": neighbor,
        "mmd_z": mmd_loss,
        "rank_huber": rank_huber,
        "factor_huber": factor_huber,
        "dynamics": dynamics["dynamics"],
        "lateral": lateral["lateral"],
        "paired_lateral": paired_lateral,
        "order": order,
        "order_violation_rate": order_violation.detach(),
        "response": response_result["loss"],
        "response_violation_rate": response_result["violation_rate"],
        "response_mean_gap_m": response_result["mean_gap_m"],
        "response_active_rate": response_result["active_rate"],
        "response_valid_count": response_result["valid_count"],
        "response_valid_weight": response_result["valid_weight"],
        "local_order_violation_rate": _masked_rate(
            (high_s - low_s) <= 0, local_valid
        ).detach(),
        "cross_order_violation_rate": _masked_rate(
            (high_s - low_s) <= 0, cross_valid
        ).detach(),
        "local_order_pair_rate": local_pair.to(signed).mean().detach(),
        "valid_order_pair_count": order_result["valid_count"],
        "valid_order_pair_weight": order_result["valid_weight"],
        "feasible_pair_rate": feasible_pair.to(signed).mean().detach(),
        "saturated_pair_rate": (~feasible_pair).to(signed).mean().detach(),
        "feasibility": feasibility,
        "feasibility_accel": (
            adaptive_feasibility["accel"] + low_feasibility["accel"] + high_feasibility["accel"]
        ) / 3.0,
        "feasibility_jerk": (
            adaptive_feasibility["jerk"] + low_feasibility["jerk"] + high_feasibility["jerk"]
        ) / 3.0,
        "feasibility_progress": (
            adaptive_feasibility["progress"] + low_feasibility["progress"] + high_feasibility["progress"]
        ) / 3.0,
        "feasibility_lateral": (
            adaptive_feasibility["lateral"] + low_feasibility["lateral"] + high_feasibility["lateral"]
        ) / 3.0,
        "rho_pair_gap": rho_gap.mean().detach(),
        "paired_s_mean": (0.5 * (low_s + high_s)).mean().detach(),
        "s_mean": s_pred.mean(),
    }


class ConditionalPreferenceLoRATrainer:
    """冻结 DiffPlanner/CSPQ，仅训练两个 LoRA 分支和条件路由。"""

    def __init__(
        self,
        planner: StyleLoRAPlanner,
        cspq: CSPQPreferenceEncoder,
        *,
        observation_normalizer,
        state_normalizer,
        device: str,
        learning_rate: float = 1e-4,
        grad_clip_norm: float = 5.0,
        # V3 的反事实可行配对仅作为可选实验，默认关闭以保持 Bounded V2 主路径。
        lambda_cf: float = 0.0,
        cf_margin: float = 0.001,
        **loss_config,
    ) -> None:
        self.planner = planner
        self.cspq = cspq.eval()
        self.observation_normalizer = observation_normalizer
        self.state_normalizer = state_normalizer
        self.device = torch.device(device)
        self.loss_config = dict(loss_config)
        self.grad_clip_norm = float(grad_clip_norm)
        self.lambda_cf = float(lambda_cf)
        self.cf_margin = float(cf_margin)
        if self.lambda_cf < 0 or self.cf_margin < 0:
            raise ValueError("lambda_cf 和 cf_margin 不能为负")
        parameters = [parameter for parameter in planner.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("条件 LoRA 训练没有可训练参数")
        self.optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        self.step = 0

    def _prepare_batch(self, batch: dict) -> dict:
        prepared = {}
        for key, value in batch.items():
            if key == "tensors":
                continue
            prepared[key] = value if isinstance(value, list) else value.to(self.device)
        prepared["tensors"] = {
            key: value.to(self.device) for key, value in batch["tensors"].items()
        }
        return prepared

    def _prepare_diffusion(self, batch: dict):
        from stylelora.lora.runtime import prepare_diffusion_batch

        metadata = [
            {
                "scene_type": "straight_free_drive" if scene == 0 else "straight_car_follow",
                "cache_path": "",
                "log_name": "",
                "token": "",
            }
            for scene in batch["scene_id"].detach().cpu().tolist()
        ]
        return prepare_diffusion_batch(
            {"tensors": batch["tensors"], "metadata": metadata},
            self.device,
            self.observation_normalizer,
            return_style_context=False,
        )

    def _loss_from_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        prepared_inputs, futures = self._prepare_diffusion(batch)
        return conditional_preference_lora_loss(
            planner=self.planner,
            cspq=self.cspq,
            inputs=prepared_inputs,
            futures=futures,
            marginal_prob=self.planner.sde.marginal_prob,
            state_normalizer=self.state_normalizer,
            batch=batch,
            **self.loss_config,
        )

    def _style_loss_from_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        prepared_inputs, futures = self._prepare_diffusion(batch)
        return feasible_style_pair_loss(
            planner=self.planner,
            inputs=prepared_inputs,
            futures=futures,
            marginal_prob=self.planner.sde.marginal_prob,
            state_normalizer=self.state_normalizer,
            batch=batch,
            margin=self.cf_margin,
        )

    def train_step(self, batch: dict, style_batch: dict | None = None) -> dict[str, float]:
        self.planner.train(True)
        prepared = self._prepare_batch(batch)
        prepared_style = self._prepare_batch(style_batch) if style_batch is not None else None
        self.optimizer.zero_grad(set_to_none=True)
        metrics = self._loss_from_batch(prepared)
        factual_loss = metrics["loss"]
        if prepared_style is not None and self.lambda_cf > 0:
            style_metrics = self._style_loss_from_batch(prepared_style)
            metrics["loss"] = factual_loss + self.lambda_cf * style_metrics["loss"]
            metrics.update({f"cf_{key}": value for key, value in style_metrics.items()})
        else:
            zero = factual_loss.detach() * 0.0
            metrics.update({
                "cf_loss": zero,
                "cf_anchor": zero,
                "cf_ranking": zero,
                "cf_accuracy": zero,
                "cf_gap": zero,
            })
        metrics["factual_loss"] = factual_loss
        if not torch.isfinite(metrics["loss"]):
            raise FloatingPointError("条件 LoRA loss 出现 NaN/Inf")
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.planner.parameters() if parameter.requires_grad],
            self.grad_clip_norm,
        )
        self.optimizer.step()
        self.planner.assert_frozen_base_unchanged()
        self.step += 1
        return {key: float(value.detach().cpu()) for key, value in metrics.items()}

    @torch.no_grad()
    def validate(
        self,
        loader,
        *,
        style_loader=None,
        max_batches: int | None = None,
        seed: int | None = None,
    ) -> dict[str, float]:
        """计算验证集平均损失；提供 seed 时固定各 batch 的扩散随机量。"""
        self.planner.train(False)
        totals: dict[str, float] = {}
        count = 0
        style_iterator = iter(style_loader) if style_loader is not None else None
        fork_devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        for batch in loader:
            prepared = self._prepare_batch(batch)
            prepared_style = None
            if style_iterator is not None:
                try:
                    style_batch = next(style_iterator)
                except StopIteration:
                    style_iterator = iter(style_loader)
                    style_batch = next(style_iterator)
                prepared_style = self._prepare_batch(style_batch)
            if seed is None:
                metrics = self._loss_from_batch(prepared)
                style_metrics = (
                    self._style_loss_from_batch(prepared_style)
                    if prepared_style is not None and self.lambda_cf > 0 else None
                )
            else:
                with torch.random.fork_rng(devices=fork_devices):
                    torch.manual_seed(seed + count)
                    if fork_devices:
                        torch.cuda.manual_seed_all(seed + count)
                    metrics = self._loss_from_batch(prepared)
                    style_metrics = (
                        self._style_loss_from_batch(prepared_style)
                        if prepared_style is not None and self.lambda_cf > 0 else None
                    )
            factual_loss = metrics["loss"]
            if style_metrics is not None:
                metrics["loss"] = factual_loss + self.lambda_cf * style_metrics["loss"]
                metrics.update({f"cf_{key}": value for key, value in style_metrics.items()})
            else:
                zero = factual_loss * 0.0
                metrics.update({
                    "cf_loss": zero,
                    "cf_anchor": zero,
                    "cf_ranking": zero,
                    "cf_accuracy": zero,
                    "cf_gap": zero,
                })
            metrics["factual_loss"] = factual_loss
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            count += 1
            if max_batches is not None and count >= max_batches:
                break
        if count == 0:
            raise ValueError("验证 DataLoader 没有产生 batch")
        return {key: value / count for key, value in totals.items()}

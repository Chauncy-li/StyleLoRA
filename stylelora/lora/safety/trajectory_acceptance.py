"""训练、门控标签和闭环执行共用的风格候选可行性检查。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class CandidateAcceptanceResult:
    """一次候选验收的结果与可审计信息。"""

    accepted: bool
    failure_reasons: Tuple[str, ...]
    candidate_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    baseline_hard_valid: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "failure_reasons": list(self.failure_reasons),
            "candidate_metrics": dict(self.candidate_metrics),
            "baseline_metrics": dict(self.baseline_metrics),
            "baseline_hard_valid": self.baseline_hard_valid,
        }


def _trajectory_dynamics(
    ego: torch.Tensor,
    current_xy: torch.Tensor,
    *,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回逐样本速度、加速度、jerk 与行驶距离。"""
    xy = torch.cat((current_xy, ego[..., :2]), dim=1)
    speed = torch.linalg.vector_norm(torch.diff(xy, dim=1), dim=-1) / dt
    acceleration = torch.diff(speed, dim=1) / dt
    jerk = torch.diff(acceleration, dim=1) / dt
    progress = torch.linalg.vector_norm(torch.diff(xy, dim=1), dim=-1).sum(dim=1)
    return speed, acceleration, jerk, progress


def _mean_lateral_deviation(
    candidate_ego: torch.Tensor,
    baseline_ego: torch.Tensor,
) -> torch.Tensor:
    """计算候选轨迹相对 baseline 航向法向的平均横向偏移。"""
    if candidate_ego.shape[-1] < 4 or baseline_ego.shape[-1] < 4:
        raise ValueError("候选与 baseline 轨迹至少需要 [x,y,cos,sin] 四维状态")
    heading = baseline_ego[..., 2:4]
    heading = heading / torch.linalg.vector_norm(heading, dim=-1, keepdim=True).clamp_min(1e-6)
    normal = torch.stack((-heading[..., 1], heading[..., 0]), dim=-1)
    displacement = candidate_ego[..., :2] - baseline_ego[..., :2]
    return (displacement * normal).sum(dim=-1).abs().mean(dim=1)


def cached_hard_feasibility_mask(
    candidate_ego: torch.Tensor,
    current_xy: torch.Tensor,
    *,
    dt: float = 0.1,
    max_speed_mps: float = 35.0,
    max_abs_accel_mps2: float = 12.0,
    max_abs_jerk_mps3: float = 45.0,
) -> torch.Tensor:
    """检查缓存轨迹的数值有效性和绝对物理边界，不包含软质量退化。"""
    if candidate_ego.ndim != 3:
        raise ValueError("candidate_ego 必须是 [B,T,D]")
    if current_xy.ndim != 3 or current_xy.shape[:2] != (candidate_ego.shape[0], 1):
        raise ValueError("current_xy 必须是 [B,1,2]")
    candidate = candidate_ego.detach()
    finite = torch.isfinite(candidate).flatten(1).all(dim=1)
    speed, acceleration, jerk, _ = _trajectory_dynamics(
        candidate, current_xy.detach(), dt=dt
    )

    def _max_abs(value: torch.Tensor) -> torch.Tensor:
        return value.abs().amax(dim=1) if value.shape[1] else value.new_zeros(value.shape[0])

    return (
        finite
        & (_max_abs(speed) <= max_speed_mps)
        & (_max_abs(acceleration) <= max_abs_accel_mps2)
        & (_max_abs(jerk) <= max_abs_jerk_mps3)
    ).detach()


def cached_feasibility_mask(
    candidate_ego: torch.Tensor,
    baseline_ego: torch.Tensor,
    current_xy: torch.Tensor,
    *,
    dt: float = 0.1,
    max_speed_mps: float = 35.0,
    max_abs_accel_mps2: float = 12.0,
    max_abs_jerk_mps3: float = 45.0,
    max_mean_accel_degradation: float = 0.5,
    max_mean_jerk_degradation: float = 2.0,
    max_progress_loss_m: float = 2.0,
    max_mean_lateral_deviation_m: float = 1.5,
) -> torch.Tensor:
    """缓存数据可用的公共检查，返回停止梯度的逐样本可行性 mask。

    地图外形与车辆外形碰撞需要在线 map API 和车辆尺寸，因此由闭环验证器补充。
    """
    if candidate_ego.shape != baseline_ego.shape or candidate_ego.ndim != 3:
        raise ValueError("candidate_ego/baseline_ego 必须是形状相同的 [B,T,D]")
    if current_xy.ndim != 3 or current_xy.shape[:2] != (candidate_ego.shape[0], 1):
        raise ValueError("current_xy 必须为 [B,1,2]")
    hard_mask = cached_hard_feasibility_mask(
        candidate_ego,
        current_xy,
        dt=dt,
        max_speed_mps=max_speed_mps,
        max_abs_accel_mps2=max_abs_accel_mps2,
        max_abs_jerk_mps3=max_abs_jerk_mps3,
    )
    relative_mask = cached_relative_feasibility_mask(
        candidate_ego,
        baseline_ego,
        current_xy,
        dt=dt,
        max_mean_accel_degradation=max_mean_accel_degradation,
        max_mean_jerk_degradation=max_mean_jerk_degradation,
        max_progress_loss_m=max_progress_loss_m,
        max_mean_lateral_deviation_m=max_mean_lateral_deviation_m,
    )
    return (hard_mask & relative_mask).detach()


def cached_relative_feasibility_mask(
    candidate_ego: torch.Tensor,
    baseline_ego: torch.Tensor,
    current_xy: torch.Tensor,
    *,
    dt: float = 0.1,
    max_mean_accel_degradation: float = 0.5,
    max_mean_jerk_degradation: float = 2.0,
    max_progress_loss_m: float = 2.0,
    max_mean_lateral_deviation_m: float = 1.5,
) -> torch.Tensor:
    """仅比较候选与 baseline 的相对退化，用于构造离线反事实偏好对。

    离线缓存轨迹缺少地图与车辆外形，绝对安全条件由闭环候选验证负责；
    此处不再用固定加速度/jerk 阈值把可学习的风格候选全部判为失败。
    """
    if candidate_ego.shape != baseline_ego.shape or candidate_ego.ndim != 3:
        raise ValueError("candidate_ego/baseline_ego 必须是形状相同的 [B,T,D]")
    if current_xy.ndim != 3 or current_xy.shape[:2] != (candidate_ego.shape[0], 1):
        raise ValueError("current_xy 必须为 [B,1,2]")
    candidate = candidate_ego.detach()
    baseline = baseline_ego.detach()
    _, c_accel, c_jerk, c_progress = _trajectory_dynamics(
        candidate, current_xy.detach(), dt=dt
    )
    _, b_accel, b_jerk, b_progress = _trajectory_dynamics(
        baseline, current_xy.detach(), dt=dt
    )

    def _mean_abs(value: torch.Tensor) -> torch.Tensor:
        return value.abs().mean(dim=1) if value.shape[1] else value.new_zeros(value.shape[0])

    mean_lateral_deviation = _mean_lateral_deviation(candidate, baseline)
    finite = (
        torch.isfinite(candidate).flatten(1).all(dim=1)
        & torch.isfinite(baseline).flatten(1).all(dim=1)
    )
    mask = (
        finite
        & (_mean_abs(c_accel) <= _mean_abs(b_accel) + max_mean_accel_degradation)
        & (_mean_abs(c_jerk) <= _mean_abs(b_jerk) + max_mean_jerk_degradation)
        & (c_progress + max_progress_loss_m >= b_progress)
        & (mean_lateral_deviation <= max_mean_lateral_deviation_m)
    )
    return mask.detach()


def differentiable_relative_feasibility_penalty(
    candidate_ego: torch.Tensor,
    baseline_ego: torch.Tensor,
    current_xy: torch.Tensor,
    *,
    dt: float = 0.1,
    max_mean_accel_degradation: float = 0.5,
    max_mean_jerk_degradation: float = 2.0,
    max_progress_loss_m: float = 2.0,
    max_mean_lateral_deviation_m: float = 1.5,
) -> Dict[str, torch.Tensor]:
    """返回保持梯度的 baseline 相对可行性超额损失。

    与 :func:`cached_relative_feasibility_mask` 使用相同的四类约束，但不把
    超阈值候选直接屏蔽。每项先按允许退化量归一化，再平方惩罚，使不可行
    候选也能收到把轨迹拉回可行区域的梯度。布尔 mask 仍仅用于评测和审计。
    """
    if candidate_ego.shape != baseline_ego.shape or candidate_ego.ndim != 3:
        raise ValueError("candidate_ego/baseline_ego 必须是形状相同的 [B,T,D]")
    if current_xy.ndim != 3 or current_xy.shape[:2] != (candidate_ego.shape[0], 1):
        raise ValueError("current_xy 必须为 [B,1,2]")
    tolerances = (
        max_mean_accel_degradation,
        max_mean_jerk_degradation,
        max_progress_loss_m,
        max_mean_lateral_deviation_m,
    )
    if any(float(value) <= 0 for value in tolerances):
        raise ValueError("可微可行性惩罚的所有允许退化量必须为正")

    _, candidate_accel, candidate_jerk, candidate_progress = _trajectory_dynamics(
        candidate_ego, current_xy, dt=dt
    )
    # baseline 是冻结策略输出；显式 detach 避免为参照轨迹构造无意义的梯度。
    baseline = baseline_ego.detach()
    _, baseline_accel, baseline_jerk, baseline_progress = _trajectory_dynamics(
        baseline, current_xy.detach(), dt=dt
    )

    def _mean_abs(value: torch.Tensor) -> torch.Tensor:
        return value.abs().mean(dim=1) if value.shape[1] else value.new_zeros(value.shape[0])

    raw_excess = {
        "accel": _mean_abs(candidate_accel) - _mean_abs(baseline_accel)
        - float(max_mean_accel_degradation),
        "jerk": _mean_abs(candidate_jerk) - _mean_abs(baseline_jerk)
        - float(max_mean_jerk_degradation),
        "progress": baseline_progress - candidate_progress - float(max_progress_loss_m),
        "lateral": _mean_lateral_deviation(candidate_ego, baseline)
        - float(max_mean_lateral_deviation_m),
    }
    scales = {
        "accel": float(max_mean_accel_degradation),
        "jerk": float(max_mean_jerk_degradation),
        "progress": float(max_progress_loss_m),
        "lateral": float(max_mean_lateral_deviation_m),
    }
    normalized = {
        name: torch.nan_to_num(
            torch.relu(value) / scales[name], nan=1.0, posinf=10.0, neginf=0.0
        )
        for name, value in raw_excess.items()
    }
    per_sample = sum(value.square() for value in normalized.values())
    return {
        "loss": per_sample.mean(),
        "per_sample": per_sample,
        "accel": normalized["accel"].square().mean(),
        "jerk": normalized["jerk"].square().mean(),
        "progress": normalized["progress"].square().mean(),
        "lateral": normalized["lateral"].square().mean(),
        "within_budget_rate": (per_sample <= 0).to(per_sample).mean().detach(),
    }


class BaselineRelativeCandidateValidator:
    """使用在线地图、预测参与者与 baseline 参照验收最终风格候选。"""

    def __init__(self, config: Any, *, step_interval: float) -> None:
        # 复用仓库已有的车辆外形、可行驶区域、预测碰撞和动力学检查。
        from baseline.simulation.candidate_selector import SafetyCandidateSelector

        self.selector = SafetyCandidateSelector(config, step_interval=step_interval)
        self.max_mean_accel_degradation = float(
            getattr(config, "bounded_max_mean_accel_degradation", 0.5)
        )
        self.max_mean_jerk_degradation = float(
            getattr(config, "bounded_max_mean_jerk_degradation", 2.0)
        )
        self.max_progress_loss_m = float(
            getattr(config, "bounded_max_progress_loss_m", 2.0)
        )
        self.max_lateral_deviation_m = float(
            getattr(
                config,
                "bounded_max_lateral_deviation_m",
                getattr(config, "bounded_max_path_deviation_m", 1.5),
            )
        )

    @staticmethod
    def _joint_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError("在线候选验收只支持 batch size 1")
            array = array[0]
        if array.ndim != 3 or array.shape[-1] != 4:
            raise ValueError(f"联合轨迹必须为 [P,T,4]，实际为 {array.shape}")
        return array

    def validate(
        self,
        candidate: torch.Tensor | np.ndarray,
        baseline: torch.Tensor | np.ndarray,
        *,
        raw_inputs: Dict[str, torch.Tensor],
        ego_state: Any,
        map_api: Any,
    ) -> CandidateAcceptanceResult:
        """候选通过绝对检查且相对 baseline 退化受限时才接受。"""
        from baseline.simulation.anchor_generator import AnchorCandidate

        candidate_array = self._joint_array(candidate)
        baseline_array = self._joint_array(baseline)
        if candidate_array.shape != baseline_array.shape:
            raise ValueError("风格候选与 baseline 联合轨迹形状必须一致")
        baseline_ego = baseline_array[0]
        anchor = AnchorCandidate(
            intent="baseline",
            ego_future_local=baseline_ego,
            ego_future_global_xy=self.selector._local_to_global_xy(
                baseline_ego[:, :2], ego_state
            ),
            target_lane_id=None,
            diagnostics={},
        )
        baseline_eval = self.selector._evaluate(
            baseline_array, anchor, raw_inputs, ego_state, map_api
        )
        candidate_eval = self.selector._evaluate(
            candidate_array, anchor, raw_inputs, ego_state, map_api
        )
        reasons = list(candidate_eval.failure_reasons)
        candidate_metrics = dict(candidate_eval.metrics)
        baseline_metrics = dict(baseline_eval.metrics)
        candidate_tensor = torch.as_tensor(candidate_array[0], dtype=torch.float32)[None]
        baseline_tensor = torch.as_tensor(baseline_array[0], dtype=torch.float32)[None]
        candidate_metrics["baseline_lateral_deviation_m"] = float(
            _mean_lateral_deviation(candidate_tensor, baseline_tensor).item()
        )

        relative_checks = (
            (
                "mean_accel_degradation",
                "mean_abs_accel_mps2",
                self.max_mean_accel_degradation,
            ),
            (
                "mean_jerk_degradation",
                "mean_abs_jerk_mps3",
                self.max_mean_jerk_degradation,
            ),
        )
        for reason, name, tolerance in relative_checks:
            if (
                name in candidate_metrics
                and name in baseline_metrics
                and candidate_metrics[name] > baseline_metrics[name] + tolerance
            ):
                reasons.append(reason)
        if (
            "progress_m" in candidate_metrics
            and "progress_m" in baseline_metrics
            and candidate_metrics["progress_m"] + self.max_progress_loss_m
            < baseline_metrics["progress_m"]
        ):
            reasons.append("progress_degradation")
        if (
            candidate_metrics["baseline_lateral_deviation_m"]
            > self.max_lateral_deviation_m
        ):
            reasons.append("baseline_lateral_deviation")

        unique_reasons = tuple(dict.fromkeys(reasons))
        return CandidateAcceptanceResult(
            accepted=bool(candidate_eval.hard_valid and not unique_reasons),
            failure_reasons=unique_reasons,
            candidate_metrics=dict(candidate_metrics),
            baseline_metrics=dict(baseline_metrics),
            baseline_hard_valid=bool(baseline_eval.hard_valid),
        )

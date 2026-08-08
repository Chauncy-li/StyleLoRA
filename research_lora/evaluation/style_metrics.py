"""Scene-specific three-axis style measurements for evaluation and training.

The axes are calculated from generated ego trajectories and the frozen scene
context, then compared against expert trajectories from a reference manifest.
``scene_style_vector`` is the evaluation implementation; the paired
``differentiable_scene_style_vector`` is used only by the training auxiliary loss.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


AXES_BY_SCENE = {
    "straight_free_drive": ("speed_preference", "longitudinal_intensity", "smoothness"),
    "straight_car_follow": ("headway_margin", "response_decisiveness", "response_smoothness"),
}


def ade_fde(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    distances = torch.linalg.vector_norm(prediction[..., :2] - target[..., :2], dim=-1)
    return {"ade": float(distances.mean()), "fde": float(distances[..., -1].mean())}


def _clip01(value: torch.Tensor) -> torch.Tensor:
    return value.clamp(0.0, 1.0)


def _rising(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return _clip01((value - low) / (high - low))


def _falling(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return _clip01((high - value) / (high - low))


def _kinematics(trajectory_xy: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    speed = torch.linalg.vector_norm(trajectory_xy[1:] - trajectory_xy[:-1], dim=-1) / dt
    if speed.numel() < 2:
        zero = trajectory_xy.new_zeros((1,))
        return speed, zero, zero
    acceleration = (speed[1:] - speed[:-1]) / dt
    jerk = (acceleration[1:] - acceleration[:-1]) / dt if acceleration.numel() > 1 else acceleration.new_zeros((1,))
    return speed, acceleration, jerk


def _p90_positive(values: torch.Tensor) -> torch.Tensor:
    values = values[values > 0]
    return torch.quantile(values, 0.9) if values.numel() else values.new_zeros(())


def _p90_negative(values: torch.Tensor) -> torch.Tensor:
    values = -values[values < 0]
    return torch.quantile(values, 0.9) if values.numel() else values.new_zeros(())


def _soft_p90_positive(values: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """可微的正加速度 P90：用 softplus 代替训练时按符号筛选样本。"""
    if not values.numel():
        return values.new_zeros(())
    return torch.quantile(F.softplus(values / temperature) * temperature, 0.9)


def _soft_p90_negative(values: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """可微的制动峰值 P90：正值表示减速度幅度。"""
    if not values.numel():
        return values.new_zeros(())
    return torch.quantile(F.softplus(-values / temperature) * temperature, 0.9)


def _route_speed_limit(route_limits: torch.Tensor, route_has_limits: torch.Tensor,
                       lane_limits: torch.Tensor, lane_has_limits: torch.Tensor) -> torch.Tensor | None:
    for limits, mask in ((route_limits, route_has_limits), (lane_limits, lane_has_limits)):
        valid = limits.reshape(-1)[mask.reshape(-1).bool()]
        valid = valid[torch.isfinite(valid) & (valid > 0)]
        if valid.numel():
            return valid.mean()
    return None


def scene_style_vector(*, scene: str, ego_future: torch.Tensor, ego_current: torch.Tensor,
                       neighbors_past: torch.Tensor, neighbors_future: torch.Tensor,
                       route_limits: torch.Tensor, route_has_limits: torch.Tensor,
                       lane_limits: torch.Tensor, lane_has_limits: torch.Tensor, dt: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a three-axis vector and per-axis validity for one sample.

    Future states can be ``[x,y,heading]`` or ``[x,y,cos,sin]``; only positions
    are required for the proxy. Car-follow axes are invalid without a lead.
    """
    full_xy = torch.cat((ego_current[:2].reshape(1, 2), ego_future[:, :2]), dim=0)
    speed, acceleration, jerk = _kinematics(full_xy, dt)
    # 不按预测加速度的正负做离散索引，避免在零附近切断风格损失的梯度。
    accel_peak, brake_peak = _soft_p90_positive(acceleration), _soft_p90_negative(acceleration)
    jerk_p90 = torch.quantile(jerk.abs(), 0.9) if jerk.numel() else full_xy.new_zeros(())
    if scene == "straight_free_drive":
        limit = _route_speed_limit(route_limits, route_has_limits, lane_limits, lane_has_limits)
        speed_ratio = speed.mean() / limit if limit is not None else full_xy.new_tensor(0.70)
        axes = torch.stack((
            _rising(speed_ratio, 0.55, 0.95),
            _clip01(0.55 * _rising(accel_peak, 0.6, 1.8) + 0.45 * _rising(jerk_p90, 12.0, 45.0)),
            _clip01(0.60 * _falling(jerk_p90, 12.0, 45.0) + 0.40 * _falling(brake_peak, 0.6, 3.0)),
        ))
        return axes, torch.ones(3, dtype=torch.bool, device=axes.device)
    if scene != "straight_car_follow":
        raise KeyError(f"Unsupported LoRA evaluation scene {scene!r}")
    # baseline 保留 32 个历史邻车，但只预测前 predicted_neighbor_num（通常为 10）个。
    # 跟车指标只应使用与未来预测一一对应的前 N 个历史邻车，不能直接拼接 32 与 10。
    neighbor_count = min(neighbors_past.shape[0], neighbors_future.shape[0])
    neighbor_current = neighbors_past[:neighbor_count, -1, :2]
    neighbor_future = neighbors_future[:neighbor_count, :, :2]
    neighbor_xy = torch.cat((neighbor_current[:, None], neighbor_future), dim=1)
    horizon = min(full_xy.shape[0], neighbor_xy.shape[1])
    gaps, headways = [], []
    speed_full = torch.cat((speed[:1], speed), dim=0) if speed.numel() else full_xy.new_zeros((horizon,))
    for step in range(horizon):
        relative = neighbor_xy[:, step] - full_xy[step]
        valid = (relative[:, 0] > 0) & (relative[:, 1].abs() < 4.0) & (neighbor_xy[:, step].abs().sum(dim=-1) > 1e-4)
        if valid.any():
            gap = relative[valid, 0].min()
            gaps.append(gap); headways.append(gap / speed_full[min(step, speed_full.numel() - 1)].clamp_min(0.1))
    if not gaps:
        return torch.zeros(3, device=full_xy.device), torch.zeros(3, dtype=torch.bool, device=full_xy.device)
    min_gap = torch.quantile(torch.stack(gaps), 0.1)
    min_thw = torch.quantile(torch.stack(headways), 0.1)
    high = torch.quantile(speed, 0.9) if speed.numel() else full_xy.new_zeros(())
    low = torch.quantile(speed, 0.1) if speed.numel() else full_xy.new_zeros(())
    drop = ((high - low) / high.clamp_min(1e-3)).clamp_min(0)
    axes = torch.stack((
        _clip01(0.55 * _rising(min_thw, 1.1, 3.0) + 0.45 * _rising(min_gap, 8.0, 26.0)),
        _clip01(0.50 * _rising(brake_peak, 0.5, 2.4) + 0.50 * _rising(drop, 0.03, 0.22)),
        _clip01(0.55 * _falling(brake_peak, 0.5, 2.4) + 0.45 * _falling(drop, 0.03, 0.22)),
    ))
    return axes, torch.ones(3, dtype=torch.bool, device=axes.device)


def differentiable_scene_style_vector(*, scene: str, ego_future: torch.Tensor, ego_current: torch.Tensor,
                                      neighbors_past: torch.Tensor, neighbors_future: torch.Tensor,
                                      route_limits: torch.Tensor, route_has_limits: torch.Tensor,
                                      lane_limits: torch.Tensor, lane_has_limits: torch.Tensor,
                                      lead_reference_future: torch.Tensor | None = None,
                                      neighbor_future_valid_mask: torch.Tensor | None = None,
                                      dt: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a three-axis vector that remains connected to ``ego_future``'s graph.

    训练用可微三轴：不调用 ``detach``、NumPy 或离散轨迹分箱。free-drive 与
    评测定义相同；car-follow 的前车候选由专家 ``lead_reference_future`` 固定，
    然后以预测轨迹计算间距和车头时距。因此无有效前车时可安全屏蔽样本，同时
    不会让预测变化触发离散的前车身份切换。
    """
    full_xy = torch.cat((ego_current[:2].reshape(1, 2), ego_future[:, :2]), dim=0)
    speed, acceleration, jerk = _kinematics(full_xy, dt)
    accel_peak, brake_peak = _p90_positive(acceleration), _p90_negative(acceleration)
    jerk_p90 = torch.quantile(jerk.abs(), 0.9) if jerk.numel() else full_xy.new_zeros(())
    if scene == "straight_free_drive":
        limit = _route_speed_limit(route_limits, route_has_limits, lane_limits, lane_has_limits)
        speed_ratio = speed.mean() / limit if limit is not None else full_xy.new_tensor(0.70)
        axes = torch.stack((
            _rising(speed_ratio, 0.55, 0.95),
            _clip01(0.55 * _rising(accel_peak, 0.6, 1.8) + 0.45 * _rising(jerk_p90, 12.0, 45.0)),
            _clip01(0.60 * _falling(jerk_p90, 12.0, 45.0) + 0.40 * _falling(brake_peak, 0.6, 3.0)),
        ))
        return axes, torch.ones(3, dtype=torch.bool, device=axes.device)
    if scene != "straight_car_follow":
        raise KeyError(f"Unsupported LoRA evaluation scene {scene!r}")

    neighbor_count = min(neighbors_past.shape[0], neighbors_future.shape[0])
    neighbor_current = neighbors_past[:neighbor_count, -1, :2]
    neighbor_future = neighbors_future[:neighbor_count, :, :2]
    neighbor_xy = torch.cat((neighbor_current[:, None], neighbor_future), dim=1)
    if neighbor_future_valid_mask is None:
        future_valid = neighbor_future.abs().sum(dim=-1) > 1e-4
    else:
        future_valid = neighbor_future_valid_mask[:neighbor_count, :neighbor_future.shape[1]].bool()
    frame_valid = torch.cat((neighbor_current.abs().sum(dim=-1, keepdim=True) > 1e-4, future_valid), dim=1)
    reference_future = ego_future if lead_reference_future is None else lead_reference_future
    reference_xy = torch.cat((ego_current[:2].reshape(1, 2), reference_future[:, :2]), dim=0)
    horizon = min(full_xy.shape[0], neighbor_xy.shape[1], reference_xy.shape[0])
    speed_full = torch.cat((speed[:1], speed), dim=0) if speed.numel() else full_xy.new_zeros((horizon,))
    gaps, headways = [], []
    for step in range(horizon):
        # 前车集合只由专家参考轨迹和冻结场景决定；对预测轨迹的梯度不会被 bool 条件截断。
        reference_relative = neighbor_xy[:, step] - reference_xy[step]
        candidate = ((reference_relative[:, 0] > 0) & (reference_relative[:, 1].abs() < 4.0)
                     & frame_valid[:, step])
        if candidate.any():
            relative = neighbor_xy[candidate, step] - full_xy[step]
            gap = relative[:, 0].min()
            gaps.append(gap)
            headways.append(gap / speed_full[min(step, speed_full.numel() - 1)].clamp_min(0.1))
    if not gaps:
        return torch.zeros(3, device=full_xy.device), torch.zeros(3, dtype=torch.bool, device=full_xy.device)
    min_gap = torch.quantile(torch.stack(gaps), 0.1)
    min_thw = torch.quantile(torch.stack(headways), 0.1)
    high = torch.quantile(speed, 0.9) if speed.numel() else full_xy.new_zeros(())
    low = torch.quantile(speed, 0.1) if speed.numel() else full_xy.new_zeros(())
    drop = ((high - low) / high.clamp_min(1e-3)).clamp_min(0)
    axes = torch.stack((
        _clip01(0.55 * _rising(min_thw, 1.1, 3.0) + 0.45 * _rising(min_gap, 8.0, 26.0)),
        _clip01(0.50 * _rising(brake_peak, 0.5, 2.4) + 0.50 * _rising(drop, 0.03, 0.22)),
        _clip01(0.55 * _falling(brake_peak, 0.5, 2.4) + 0.45 * _falling(drop, 0.03, 0.22)),
    ))
    return axes, torch.ones(3, dtype=torch.bool, device=axes.device)


def wasserstein_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().sort().values, b.flatten().sort().values
    n = min(a.numel(), b.numel())
    if not n: return float("nan")
    index_a = torch.linspace(0, a.numel() - 1, n, device=a.device).long()
    index_b = torch.linspace(0, b.numel() - 1, n, device=b.device).long()
    return float((a[index_a] - b[index_b]).abs().mean())


def per_axis_wasserstein(predicted: torch.Tensor, reference: torch.Tensor, axis_names: Sequence[str]) -> dict[str, float]:
    if (predicted.ndim != 2 or reference.ndim != 2 or predicted.shape[1] != reference.shape[1]
            or predicted.shape[1] != len(axis_names)):
        raise ValueError("Wasserstein inputs must be [samples, axes] with one name per axis")
    return {name: wasserstein_1d(predicted[:, index], reference[:, index]) for index, name in enumerate(axis_names)}


def mmd_rbf(a: torch.Tensor, b: torch.Tensor, sigma: float = 1.0) -> float:
    def kernel(x, y): return torch.exp(-torch.cdist(x, y).square() / (2 * sigma * sigma))
    return float(kernel(a, a).mean() + kernel(b, b).mean() - 2 * kernel(a, b).mean())


def target_cluster_bounds(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.quantile(reference, 0.05, dim=0), torch.quantile(reference, 0.95, dim=0)


def aggregate_style_evaluation(records: Sequence[Mapping[str, object]], references: Mapping[tuple[str, str], torch.Tensor]) -> list[dict[str, object]]:
    grouped: Dict[tuple[float, str, str], list[torch.Tensor]] = defaultdict(list)
    for record in records:
        if bool(record.get("style_valid", False)):
            grouped[(float(record["rho"]), str(record["scene"]), str(record["target_style"]))].append(torch.tensor(record["style_vector"]))
    result = []
    for key, vectors in grouped.items():
        rho, scene, style = key; predicted = torch.stack(vectors); reference = references.get((scene, style))
        if reference is None or reference.numel() == 0:
            continue
        lower, upper = target_cluster_bounds(reference)
        hit = ((predicted >= lower) & (predicted <= upper)).all(dim=-1).float().mean()
        axes = AXES_BY_SCENE[scene]
        result.append({"rho": rho, "scene": scene, "target_style": style, "count": len(vectors),
                       "cluster_hit_rate": float(hit), "wasserstein_by_axis": per_axis_wasserstein(predicted, reference, axes),
                       "mmd_rbf": mmd_rbf(predicted, reference), "mean_style_vector": predicted.mean(dim=0).tolist(),
                       "target_mean_style_vector": reference.mean(dim=0).tolist(), "axis_names": axes,
                       "metric_space": "generated_trajectory_proxy_axes"})
    return result

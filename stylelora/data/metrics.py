"""Raw three-axis physical metric extraction from DiffPlanner caches.

从 DiffPlanner 缓存张量离线提取"越大越激进"的原始物理三轴指标。
只基于物理量与方向先验，不设置分类阈值，不与 style_metrics 的混合风格向量耦合。

每个场景输出三个独立物理量（顺序与 axes.AXES_BY_SCENE 一致）：
- straight_free_drive:
    axis0 speed_preference       平均速度 / 道路限速比（无量纲），越大越激进
    axis1 longitudinal_intensity 正向加速度 P90 与制动峰值 P90 的较大幅值（m/s^2），越大越激进
    axis2 smoothness             平顺性 = 1 / (jerk_p90 + eps)，越大越平顺（越大越保守）
- straight_car_follow:
    axis0 headway_margin         min(THW)（秒），越大 = 跟车越远 = 越保守
    axis1 response_decisiveness  正向加速度 P90 与制动峰值 P90 的较大幅值（m/s^2），越大越激进
    axis2 response_smoothness    平顺性 = 1 / (jerk_p90 + eps)，越大越平顺（越大越保守）

说明：本函数输出的是 AXES_BY_SCENE 轴名对应的语义量，方向符号统一由
axes.DIRECTION_BY_SCENE 决定（+1 越大越激进，-1 越大越保守）。
不要在 metrics 内预先反转方向，否则会造成二次反转。

axis_valid 判定：free-drive 若无有效限速则该轴无效；car-follow 若无有效前车
则三个轴都无效（三轴定义依赖跟车交互）。
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import torch

from stylelora.data.axes import AXES_BY_SCENE

# 缓存张量字段名（与 StyleManifestDataset 输出一致）
_REQUIRED_KEYS = (
    "ego_current_state", "ego_future_gt", "neighbor_agents_past", "neighbors_future_gt",
    "route_lanes_speed_limit", "route_lanes_has_speed_limit", "lanes_speed_limit", "lanes_has_speed_limit",
)
_CAR_FOLLOW_KEYS = ("neighbor_agents_future_mask",)


def _kinematics(full_xy: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从拼接的 [x,y] 轨迹计算速度、加速度与 jerk 序列。

    Args:
        full_xy: [T, 2] 位置序列（首帧为当前帧）。
        dt: 时间步长（秒）。

    Returns:
        (speed[T-1], accel[T-2], jerk[T-3])。
    """
    speed = torch.linalg.vector_norm(full_xy[1:] - full_xy[:-1], dim=-1) / dt
    if speed.numel() < 2:
        zero = full_xy.new_zeros((1,))
        return speed, zero, zero
    acceleration = (speed[1:] - speed[:-1]) / dt
    jerk = (acceleration[1:] - acceleration[:-1]) / dt if acceleration.numel() > 1 else acceleration.new_zeros((1,))
    return speed, acceleration, jerk


def _p90_positive(values: torch.Tensor) -> torch.Tensor:
    """正向冲击 P90：仅统计正值，无正值返回 0。"""
    values = values[values > 0]
    return torch.quantile(values, 0.9) if values.numel() else values.new_zeros(())


def _p90_negative(values: torch.Tensor) -> torch.Tensor:
    """制动峰值 P90：负值取绝对值后的 P90，无负值返回 0。"""
    values = -values[values < 0]
    return torch.quantile(values, 0.9) if values.numel() else values.new_zeros(())


def _route_speed_value(tensors: Mapping[str, torch.Tensor], index: int) -> torch.Tensor | None:
    """返回该样本可用的道路限速均值（米/秒）；无有效限速返回 None。"""
    for limits, has_limits in (
        (tensors["route_lanes_speed_limit"][index], tensors["route_lanes_has_speed_limit"][index]),
        (tensors["lanes_speed_limit"][index], tensors["lanes_has_speed_limit"][index]),
    ):
        flat_limits = limits.reshape(-1)
        flat_mask = has_limits.reshape(-1).bool()
        valid = flat_limits[flat_mask]
        valid = valid[torch.isfinite(valid) & (valid > 0)]
        if valid.numel():
            return valid.mean()
    return None


def _lead_candidates(*, ego_point: torch.Tensor, neighbor_xy: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
    """返回某时间步中位于自车前方（纵向 > 0、横向 < 4m、非零）的邻居掩码。

    Args:
        ego_point: [2] 自车当前位置。
        neighbor_xy: [P, 2] 邻居当前位置。
        frame_valid: [P] 该帧有效的邻居掩码。

    Returns:
        [P] bool 掩码。
    """
    relative = neighbor_xy - ego_point
    return (relative[:, 0] > 0) & (relative[:, 1].abs() < 4.0) & (relative[:, 0] < 60.0) & frame_valid


def scene_physics_axes(tensors: Mapping[str, torch.Tensor], index: int, scene: str,
                       *, dt: float = 0.1) -> Tuple[Sequence[float], Sequence[bool]]:
    """为一条样本计算三轴物理指标与有效性掩码。

    Args:
        tensors: 一个批次的缓存张量字典（同 StyleManifestDataset 输出）。
        index: 该样本在批次中的索引。
        scene: straight_free_drive / straight_car_follow。
        dt: 时间步长（秒）。

    Returns:
        (values[3], valid[3])：越大越激进的物理指标与各轴有效性。

    Raises:
        KeyError: 缓存张量缺少必要字段。
    """
    missing = [key for key in _REQUIRED_KEYS if key not in tensors]
    if missing:
        raise KeyError(f"Cache misses required keys: {missing}")

    ego_current = tensors["ego_current_state"][index]
    ego_future = tensors["ego_future_gt"][index]
    full_xy = torch.cat((ego_current[:2].reshape(1, 2).float(), ego_future[:, :2].float()), dim=0)
    speed, acceleration, jerk = _kinematics(full_xy, dt)
    accel_peak = _p90_positive(acceleration)
    brake_peak = _p90_negative(acceleration)
    jerk_p90 = torch.quantile(jerk.abs(), 0.9) if jerk.numel() else full_xy.new_zeros(())
    # axis1 longitudinal_intensity：最大纵向冲击幅值（正向加速峰值与制动峰值的较大者）
    axis1 = torch.maximum(accel_peak, brake_peak)
    # axis2 smoothness：平顺性，1/(jerk+eps)，越大越平顺；配合 axes 中 smoothness 的 -1 方向
    axis2 = 1.0 / (jerk_p90 + 1e-6)

    if scene == "straight_free_drive":
        limit = _route_speed_value(tensors, index)
        if limit is None or limit.item() <= 0:
            # 无有效限速：速度轴无效，轴1/轴2 仍有效
            return (float("nan"), float(axis1), float(axis2)), (False, True, True)
        speed_ratio = speed.mean() / limit
        return (float(speed_ratio), float(axis1), float(axis2)), (True, True, True)

    if scene != "straight_car_follow":
        raise KeyError(f"Unsupported scene {scene!r}; expected one of {sorted(AXES_BY_SCENE)}")

    # ---- car-follow：与 build_style_prototypes 相同的前置处理 ----
    neighbors_future = tensors["neighbors_future_gt"][index]
    if "neighbor_agents_future_mask" in tensors:
        # 缓存掩码 True 表示有效帧；失效帧先置零再找前车
        neighbors_future = neighbors_future.clone()
        neighbors_future[~tensors["neighbor_agents_future_mask"][index].bool()] = 0
    neighbors_past = tensors["neighbor_agents_past"][index]
    # 只使用与未来预测对应的历史邻车（同 style_metrics 约束）
    neighbor_count = min(neighbors_past.shape[0], neighbors_future.shape[0])
    neighbor_current = neighbors_past[:neighbor_count, -1, :2].float()
    neighbor_xy = torch.cat((neighbor_current[:, None], neighbors_future[:neighbor_count, :, :2].float()), dim=1)
    frame_valid_full = neighbor_xy.abs().sum(dim=-1) > 1e-4
    horizon = min(full_xy.shape[0], neighbor_xy.shape[1])
    speed_full = torch.cat((speed[:1], speed), dim=0) if speed.numel() else full_xy.new_zeros((horizon,))

    gaps, headways = [], []
    for step in range(horizon):
        candidate = _lead_candidates(
            ego_point=full_xy[step],
            neighbor_xy=neighbor_xy[:, step],
            frame_valid=frame_valid_full[:, step],
        )
        if not candidate.any():
            continue
        relative = neighbor_xy[candidate, step] - full_xy[step]
        gap = relative[:, 0].min()
        gaps.append(float(gap))
        headways.append(float(gap / max(float(speed_full[min(step, speed_full.numel() - 1)]), 0.1)))
    if not gaps:
        # 无有效前车：跟车三轴物理定义不成立，三个轴全部无效
        return (float("nan"), float("nan"), float("nan")), (False, False, False)

    min_thw = min(headways)
    # axis0 headway_margin：最小车头时距（秒），越大 = 跟车越远 = 越保守；
    # 配合 axes 中 headway_margin 的 -1 方向，不二次反转
    return (float(min_thw), float(axis1), float(axis2)), (True, True, True)


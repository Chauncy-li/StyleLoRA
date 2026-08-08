"""Physical direction priors for the three preference axes.

三轴物理方向先验：规定每个场景的三个轴"数值越大 = 越激进"的方向符号。
只使用物理方向先验，不设置任何分类阈值。

方向约定（+1: 越大越激进，-1: 越大越保守）：
- straight_free_drive:
    speed_preference          + 速度越快越激进
    longitudinal_intensity    + 纵向加/减速强度越大越激进
    smoothness                - 平滑性越高（越平稳）越保守 -> 平滑性越低越激进
- straight_car_follow:
    headway_margin            - 车头时距/间距越小越激进
    response_decisiveness     + 制动峰值/速度跌落越大越激进
    response_smoothness       - 响应越平顺越保守 -> 越不平顺越激进
"""

from __future__ import annotations

from typing import Mapping, Sequence

# 每个场景的轴名顺序（与 research_lora.evaluation.style_metrics.AXES_BY_SCENE 一致）
AXES_BY_SCENE: Mapping[str, Sequence[str]] = {
    "straight_free_drive": ("speed_preference", "longitudinal_intensity", "smoothness"),
    "straight_car_follow": ("headway_margin", "response_decisiveness", "response_smoothness"),
}

# 每个场景轴的物理方向符号（+1: 越大越激进，-1: 越大越保守）
DIRECTION_BY_SCENE: Mapping[str, Sequence[int]] = {
    "straight_free_drive": (+1, +1, -1),
    "straight_car_follow": (-1, +1, -1),
}

# 每个场景轴的物理单位描述（用于审计与文档，不影响计算）
UNITS_BY_SCENE: Mapping[str, Sequence[str]] = {
    "straight_free_drive": (
        "speed_ratio_to_limit",  # 平均速度 / 限速比，无量纲
        "longitudinal_aggressiveness",  # 加速度 P90 + jerk P90 综合，单位 m/s^2 量级
        "inverse_smoothness",  # jerk P90 + 制动峰值反向，越大越颠簸
    ),
    "straight_car_follow": (
        "inverse_headway",  # THW / gap 反向，越小间距越激进
        "response_aggressiveness",  # 制动峰值 + 速度跌落综合
        "inverse_response_smoothness",  # 制动/跌落反向
    ),
}


def axes_for(scene: str) -> tuple[str, str, str]:
    """返回场景的三个轴名（顺序固定）。"""
    return tuple(AXES_BY_SCENE[scene])


def direction_sign(scene: str, axis: str) -> int:
    """返回指定场景下某轴的物理方向符号（+1 越大越激进，-1 越大越保守）。

    Raises:
        KeyError: 场景或轴不在先验表中。
    """
    names = AXES_BY_SCENE.get(scene)
    if names is None:
        raise KeyError(f"Unknown scene {scene!r}; expected one of {sorted(AXES_BY_SCENE)}")
    if axis not in names:
        raise KeyError(f"Unknown axis {axis!r} for scene {scene!r}; expected {list(names)}")
    return DIRECTION_BY_SCENE[scene][names.index(axis)]


def direction_signs(scene: str) -> tuple[int, int, int]:
    """返回场景三个轴的完整方向符号元组。"""
    return tuple(DIRECTION_BY_SCENE[scene])
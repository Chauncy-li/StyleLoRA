"""Versioned, serialisable weak-preference records for research_lora_2.

research_lora_2 的弱排序偏好样本记录：一条样本的三轴物理指标、
场景内百分位、稳健聚合偏好排名与置信度。不使用 aggr/norm/cons 分类标签。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

from research_lora.data.schema import StyleSample

# 合法的场景取值（与 research_lora.data.schema 保持一致）
VALID_SCENES = ("straight_free_drive", "straight_car_follow")

# 弱排序产物版本；任何字段语义变更都必须升版本
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class PreferenceSample:
    """一条规范化后的弱排序偏好样本记录（不可变）。

    Attributes:
        scene_type: 场景类型（straight_free_drive / straight_car_follow）。
        axis_names: 三个轴的规范名称（顺序对应 axis_raw / axis_percentiles）。
        axis_raw: 三个轴的原始物理指标值 [3]（溯源用，不参与训练）。
        axis_valid: 三个轴是否有效 [3]（car-follow 无有效前车时轴无效）。
        axis_percentiles: 三个轴的场景内百分位 [3]，统一为"越大越激进"。
        preference_rank: 三轴加权中位数聚合的 [0,1] 相对激进程度。
        rank_confidence: 三轴一致度置信度 [0,1]，由 1 - MAD 得到。
        cache_path: 对应缓存文件路径（追溯用）。
        log_name / token: 场景日志名与令牌（跨拆分泄漏检测）。
        scene_feature: 冻结 DiffPlanner 场景特征 h_c 的引用 fid（可选）。
        source_index: 来源 manifest 文件路径（溯源）。
    """

    scene_type: str
    axis_names: Sequence[str]
    axis_raw: Sequence[float]
    axis_valid: Sequence[bool]
    axis_percentiles: Sequence[float]
    preference_rank: float
    rank_confidence: float
    cache_path: str = ""
    log_name: str = ""
    token: str = ""
    scene_feature: str = ""
    source_index: str = ""

    def __post_init__(self) -> None:
        """构造后校验：场景合法、数组长度一致、排名/置信度在合法范围。"""
        if self.scene_type not in VALID_SCENES:
            raise ValueError(f"Unknown scene_type {self.scene_type!r}; expected one of {VALID_SCENES}")
        if not (len(self.axis_names) == len(self.axis_raw) == len(self.axis_valid) == len(self.axis_percentiles) == 3):
            raise ValueError("axis_names/axis_raw/axis_valid/axis_percentiles must all have length 3")
        if not 0.0 <= float(self.preference_rank) <= 1.0:
            raise ValueError(f"preference_rank must be in [0, 1], got {self.preference_rank!r}")
        if not 0.0 <= float(self.rank_confidence) <= 1.0:
            raise ValueError(f"rank_confidence must be in [0, 1], got {self.rank_confidence!r}")
        for value in self.axis_percentiles:
            number = float(value)
            # NaN 表示该轴无效（由 axis_valid=False 标记），不参与聚合；允许存储
            if not math.isnan(number) and not 0.0 <= number <= 1.0:
                raise ValueError(f"axis_percentiles must be in [0, 1] or NaN, got {value!r}")

    @property
    def key(self) -> str:
        """稳定的泄漏检测键：优先 log_name:token，回退 cache_path。"""
        return f"{self.log_name}:{self.token}" if self.log_name or self.token else self.cache_path

    @property
    def valid_axes(self) -> int:
        """有效轴数量。"""
        return int(sum(bool(value) for value in self.axis_valid))

    def to_dict(self) -> Dict[str, Any]:
        """转成普通字典（用于 JSON 序列化）。"""
        return asdict(self)

    @classmethod
    def from_style_sample(cls, sample: StyleSample) -> "PreferenceSample":
        """从 research_lora 的 StyleSample 构造偏好样本的元信息骨架。

        只复制场景/追溯/来源字段；三轴物理指标与百分位由下游算法填充。
        """
        return cls(
            scene_type=sample.scene_type,
            axis_names=(),
            axis_raw=(),
            axis_valid=(),
            axis_percentiles=(),
            preference_rank=0.0,
            rank_confidence=0.0,
            cache_path=sample.cache_path,
            log_name=sample.log_name,
            token=sample.token,
            source_index=sample.source_index,
        )

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PreferenceSample":
        """从 JSON 行（字典）恢复偏好样本，保持数组类型稳定。"""
        return cls(
            scene_type=str(row["scene_type"]),
            axis_names=tuple(str(name) for name in row["axis_names"]),
            axis_raw=tuple(float(value) for value in row["axis_raw"]),
            axis_valid=tuple(bool(value) for value in row["axis_valid"]),
            axis_percentiles=tuple(float(value) for value in row["axis_percentiles"]),
            preference_rank=float(row["preference_rank"]),
            rank_confidence=float(row["rank_confidence"]),
            cache_path=str(row.get("cache_path", "")),
            log_name=str(row.get("log_name", "")),
            token=str(row.get("token", "")),
            scene_feature=str(row.get("scene_feature", "")),
            source_index=str(row.get("source_index", "")),
        )
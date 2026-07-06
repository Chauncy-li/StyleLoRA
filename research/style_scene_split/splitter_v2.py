"""style_scene_split v2 的增强版切分器。

设计目标：
1. 复用原始 splitter 的高精度场景净化与硬标签逻辑；
2. 在不影响原始代码的前提下，补充质量层、连续风格层和条件标签层；
3. 让新产物既能继续服务 retrieval memory，又能支持更强的统计分析。

实现策略：
- v2 直接继承原始 StyleSceneSplitter；
- 主场景选择、风格硬标签和 abstention 逻辑全部复用原实现；
- 新增逻辑仅在 super().split(...) 之后附加，不回写旧结构。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np

from .schema_v2 import StyleSceneSplitResultV2
from .splitter import (
    StyleSceneSplitter,
    _clip01,
    _nonzero_xy_mask,
    _resolve_neighbor_valid_mask,
    _score_falling,
    _score_rising,
)


class StyleSceneSplitterV2(StyleSceneSplitter):
    """在原始高精度 splitter 之上叠加质量、连续风格和条件标签。

    这一类的核心原则是：
    - 不改动旧逻辑的 decision boundary；
    - 只做“附加层”的增强；
    - 让旧的 scene/style split 仍然是主干，新的连续层是补充表达。
    """

    def __init__(
        self,
        time_delta: float = 0.1,
        min_scene_score: float = 0.58,
        min_scene_margin: float = 0.10,
        min_split_confidence: float = 0.55,
        min_quality_score: float = 0.55,
    ) -> None:
        super().__init__(
            time_delta=time_delta,
            min_scene_score=min_scene_score,
            min_scene_margin=min_scene_margin,
            min_split_confidence=min_split_confidence,
        )
        # 质量层的阈值和 split_valid 分离控制，便于区分“样本差”和“分类不稳”。
        self.min_quality_score = float(min_quality_score)

    def split(self, cache_data: Dict[str, np.ndarray]) -> StyleSceneSplitResultV2:
        """执行 v2 切分。

        流程分为四步：
        1. 先调用旧版 splitter，得到原始高精度结果；
        2. 计算样本质量层；
        3. 计算条件标签和连续风格向量；
        4. 组合成新的 v2 dataclass 输出。
        """

        base_result = super().split(cache_data)

        sample_quality_valid, quality_score, quality_reason, quality_vec = self._compute_sample_quality(
            cache_data=cache_data,
            base_result=base_result,
        )
        density_level, speed_regime, curvature_level = self._infer_condition_tags(base_result)
        style_performance_vec, style_performance_confidence = self._build_style_performance_vec(base_result)

        # memory_eligible 比 split_valid 更严格：需要样本可靠，且切分结果也可靠。
        memory_eligible = bool(sample_quality_valid and base_result.split_valid)

        # 如果样本质量不过关，则 v2 的 subset 仍然标记为 invalid，
        # 从而避免无意中把低质量样本送进下游 memory 构建流程。
        subset_id = base_result.subset_id if memory_eligible else "invalid"

        payload = asdict(base_result)
        payload.update(
            {
                "subset_id": subset_id,
                "sample_quality_valid": sample_quality_valid,
                "memory_eligible": memory_eligible,
                "quality_score": quality_score,
                "quality_reason": quality_reason,
                "quality_vec": quality_vec,
                "style_performance_vec": style_performance_vec,
                "style_performance_confidence": style_performance_confidence,
                "condition_density_level": density_level,
                "condition_speed_regime": speed_regime,
                "condition_curvature_level": curvature_level,
            }
        )
        return StyleSceneSplitResultV2(**payload)

    # ------------------------------------------------------------------
    # v2 新增：样本质量层
    # ------------------------------------------------------------------

    def _compute_sample_quality(
        self,
        cache_data: Dict[str, np.ndarray],
        base_result,
    ) -> Tuple[bool, float, str, np.ndarray]:
        """计算样本质量。

        这里故意不用极苛刻的规则，而是采用“宽松但可解释”的评分：
        - 目标不是再做一层 aggressive filtering；
        - 目标是把“样本本身是否可靠”单独显式化。
        """

        traj_completeness = self._trajectory_completeness_score(cache_data)
        neighbor_consistency = self._neighbor_consistency_score(cache_data)
        route_metadata = self._route_metadata_score(base_result)
        metric_sanity = self._metric_sanity_score(base_result)

        quality_vec = np.asarray(
            [
                traj_completeness,
                neighbor_consistency,
                route_metadata,
                metric_sanity,
            ],
            dtype=np.float32,
        )

        # 权重设计遵循“轨迹本身 + 邻车可用性”优先，再考虑 route 元数据和指标 sanity。
        quality_score = float(
            0.30 * traj_completeness
            + 0.25 * neighbor_consistency
            + 0.25 * route_metadata
            + 0.20 * metric_sanity
        )
        sample_quality_valid = bool(quality_score >= self.min_quality_score)

        low_parts = []
        if traj_completeness < 0.45:
            low_parts.append("ego_future_completeness_low")
        if neighbor_consistency < 0.35:
            low_parts.append("neighbor_future_mask_consistency_low")
        if route_metadata < 0.35:
            low_parts.append("route_metadata_support_weak")
        if metric_sanity < 0.45:
            low_parts.append("kinematic_metrics_out_of_broad_range")

        if sample_quality_valid:
            quality_reason = "sample_quality_is_sufficient"
        elif low_parts:
            quality_reason = "; ".join(low_parts)
        else:
            quality_reason = "quality_score_below_threshold"

        return sample_quality_valid, quality_score, quality_reason, quality_vec

    def _trajectory_completeness_score(self, cache_data: Dict[str, np.ndarray]) -> float:
        """衡量 ego future 轨迹是否足够完整。

        注意这里不用“是否运动明显”来判定质量，因为静止或低速样本并不一定是坏样本。
        我们只检查：
        - 数据形状是否合理；
        - 是否有限值完整；
        - 是否不是清一色异常填充。
        """

        ego_future = np.asarray(cache_data.get("ego_agent_future", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        if ego_future.ndim != 2 or ego_future.shape[0] == 0:
            return 0.0

        xy = ego_future[:, :2]
        finite_ratio = float(np.isfinite(xy).all(axis=1).mean()) if xy.size > 0 else 0.0
        # 允许部分静止点存在，只要不是整条轨迹都为极端异常填充即可。
        nonzero_ratio = float((np.linalg.norm(xy, axis=-1) > 1e-5).mean()) if xy.size > 0 else 0.0

        # 如果轨迹大部分点都有限，我们就给较高分；nonzero_ratio 只做轻微辅助项。
        return _clip01(0.80 * finite_ratio + 0.20 * max(nonzero_ratio, 0.25))

    def _neighbor_consistency_score(self, cache_data: Dict[str, np.ndarray]) -> float:
        """衡量 neighbor future mask 与非零轨迹的一致性。

        原始缓存中不同版本可能有：
        - 直接 valid mask；
        - 反向语义 mask；
        - 只有轨迹，没有 mask。
        这里不追求“精确恢复真值”，只看是否存在明显不一致。
        """

        neighbors_future = np.asarray(
            cache_data.get("neighbor_agents_future", np.zeros((0, 0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        if neighbors_future.ndim != 3:
            return 0.0
        if neighbors_future.shape[0] == 0:
            return 1.0

        nonzero_mask = _nonzero_xy_mask(neighbors_future)
        if nonzero_mask.size == 0:
            return 1.0

        raw_mask = cache_data.get("neighbor_agents_future_mask", None)
        if raw_mask is None:
            # 没有显式 mask 的情况下，默认给予中高分，但不判满分。
            return 0.75

        raw_mask = np.asarray(raw_mask, dtype=bool)
        if raw_mask.shape != nonzero_mask.shape:
            return 0.40

        direct_overlap = int(np.logical_and(raw_mask, nonzero_mask).sum())
        inverse_overlap = int(np.logical_and(~raw_mask, nonzero_mask).sum())
        denom = max(int(nonzero_mask.sum()), 1)
        overlap_ratio = max(direct_overlap, inverse_overlap) / denom

        # 如果 mask 与非零轨迹几乎完全一致，分数就很高；
        # 否则逐步下降，但不直接置零。
        return _clip01(overlap_ratio)

    def _route_metadata_score(self, base_result) -> float:
        """衡量 route / map 元数据是否足以支撑本次分析。

        这里不把 speed limit 缺失视为严重错误，
        因为某些场景本来就可能拿不到稳定限速信息。
        """

        route_lane_term = 1.0 if int(base_result.route_lane_count) > 0 else 0.0
        speed_limit_term = 1.0 if self._is_valid_optional(base_result.route_speed_limit_mps) else 0.65
        control_term = 0.85 if bool(base_result.route_has_control) else 1.0
        return _clip01(0.45 * route_lane_term + 0.35 * speed_limit_term + 0.20 * control_term)

    def _metric_sanity_score(self, base_result) -> float:
        """对核心动力学指标做宽松 sanity check。

        这里的目的不是判断风格，而是排除明显不合理的数值：
        - 例如过大的 jerk、负值异常、离谱距离等。
        """

        checks = [
            0.0 <= float(base_result.ego_mean_speed) <= 45.0,
            0.0 <= float(base_result.ego_accel_peak) <= 8.0,
            0.0 <= float(base_result.ego_brake_peak) <= 10.0,
            0.0 <= float(base_result.ego_jerk_p90) <= 150.0,
            0.0 <= float(base_result.ego_jerk_peak) <= 250.0,
            0.0 <= float(base_result.global_min_distance) <= 200.0,
            0.0 <= float(abs(base_result.ego_heading_change)) <= float(np.pi),
            0.0 <= float(base_result.ego_progress) <= 200.0,
        ]
        return float(np.mean(np.asarray(checks, dtype=np.float32)))

    # ------------------------------------------------------------------
    # v2 新增：条件标签层
    # ------------------------------------------------------------------

    def _infer_condition_tags(self, base_result) -> Tuple[str, str, str]:
        """根据已有结果推断附加条件标签。

        这些标签是“主场景桶之外的条件变量”，
        用于做 scene-conditioned style distribution analysis。
        """

        nearby_count = int(base_result.nearby_agent_count)
        if nearby_count <= 2:
            density_level = "sparse"
        elif nearby_count <= 7:
            density_level = "medium"
        else:
            density_level = "dense"

        speed_ref = base_result.route_speed_limit_mps
        if not self._is_valid_optional(speed_ref):
            speed_ref = float(base_result.ego_mean_speed)

        speed_ref = float(speed_ref) if speed_ref is not None else 0.0
        if speed_ref < 6.0:
            speed_regime = "slow"
        elif speed_ref < 12.0:
            speed_regime = "urban"
        elif speed_ref < 20.0:
            speed_regime = "suburban"
        else:
            speed_regime = "high_speed"

        heading_change = abs(float(base_result.ego_heading_change))
        if heading_change < 0.08:
            curvature_level = "low"
        elif heading_change < 0.18:
            curvature_level = "mild"
        else:
            curvature_level = "high"

        return density_level, speed_regime, curvature_level

    # ------------------------------------------------------------------
    # v2 新增：连续风格层
    # ------------------------------------------------------------------

    def _build_style_performance_vec(self, base_result) -> Tuple[np.ndarray, float]:
        """为每个有效场景桶构建 3 维连续风格表现向量。

        设计原则：
        - 不替代离散风格标签；
        - 只用当前 splitter 已经产出的可解释指标；
        - 维度含义随 scene bucket 变化，但都保持在 [0, 1]。
        """

        if base_result.scene_bucket == "straight_free_drive":
            style_vec = self._free_drive_style_vec(base_result)
        elif base_result.scene_bucket == "straight_car_follow":
            style_vec = self._car_follow_style_vec(base_result)
        elif base_result.scene_bucket == "straight_lane_change":
            style_vec = self._lane_change_style_vec(base_result)
        else:
            style_vec = np.zeros((3,), dtype=np.float32)

        # 连续向量置信度不强制依赖 hard label 是否有效，但必须依赖当前场景和指标的可信度。
        confidence = _clip01(
            0.45 * float(base_result.scene_confidence)
            + 0.35 * float(base_result.style_confidence)
            + 0.20 * float(base_result.split_confidence)
        )
        if base_result.scene_bucket == "none":
            confidence = 0.0

        return style_vec.astype(np.float32), float(confidence)

    def _free_drive_style_vec(self, base_result) -> np.ndarray:
        """构造直行自由驾驶场景下的连续风格向量。

        维度定义：
        1. speed_preference：更偏向高巡航速度
        2. longitudinal_intensity：更偏向强加速/高纵向动态
        3. smoothness：更平顺、更舒适
        """

        speed_ratio = self._optional_value(base_result.ego_speed_ratio_to_limit, default=0.70)
        speed_preference = _score_rising(speed_ratio, 0.55, 0.95)
        longitudinal_intensity = _clip01(
            0.55 * _score_rising(base_result.ego_accel_peak, 0.6, 1.8)
            + 0.45 * _score_rising(base_result.ego_jerk_p90, 12.0, 45.0)
        )
        smoothness = _clip01(
            0.60 * _score_falling(base_result.ego_jerk_p90, 12.0, 45.0)
            + 0.40 * _score_falling(base_result.ego_brake_peak, 0.6, 3.0)
        )
        return np.asarray([speed_preference, longitudinal_intensity, smoothness], dtype=np.float32)

    def _car_follow_style_vec(self, base_result) -> np.ndarray:
        """构造直行跟驰场景下的连续风格向量。

        维度定义：
        1. headway_margin：更偏向保守的大车头时距/间距
        2. response_decisiveness：面对跟驰事件时反应更明显
        3. response_smoothness：反应过程更平顺
        """

        min_thw = self._optional_value(base_result.following_min_thw, default=2.0)
        min_gap = self._optional_value(base_result.following_min_gap, default=16.0)

        headway_margin = _clip01(
            0.55 * _score_rising(min_thw, 1.1, 3.0)
            + 0.45 * _score_rising(min_gap, 8.0, 26.0)
        )
        response_decisiveness = _clip01(
            0.50 * _score_rising(base_result.event_brake_peak, 0.5, 2.4)
            + 0.50 * _score_rising(base_result.event_speed_drop_ratio, 0.03, 0.22)
        )
        response_smoothness = _clip01(
            0.55 * _score_falling(base_result.event_brake_peak, 0.5, 2.4)
            + 0.45 * _score_falling(base_result.event_speed_drop_ratio, 0.03, 0.22)
        )
        return np.asarray([headway_margin, response_decisiveness, response_smoothness], dtype=np.float32)

    def _lane_change_style_vec(self, base_result) -> np.ndarray:
        """构造直行换道场景下的连续风格向量。

        维度定义：
        1. gap_acceptance：更愿意接受较小的 merge gap
        2. lateral_commitment：横向动作更早、更果断
        3. execution_smoothness：换道执行更平顺
        """

        merge_gap = self._optional_value(base_result.merge_min_gap, default=18.0, large_invalid=True)
        onset_step = self._optional_value(base_result.ego_lateral_onset_step, default=28.0)

        gap_acceptance = _score_falling(merge_gap, 10.0, 28.0)
        lateral_commitment = _clip01(
            0.55 * _score_falling(onset_step, 8.0, 35.0)
            + 0.45 * _score_rising(base_result.ego_lateral_speed_peak, 0.5, 1.8)
        )
        execution_smoothness = _clip01(
            0.60 * _score_falling(base_result.ego_lateral_speed_peak, 0.6, 1.8)
            + 0.40 * _score_falling(abs(base_result.ego_heading_change), 0.03, 0.18)
        )
        return np.asarray([gap_acceptance, lateral_commitment, execution_smoothness], dtype=np.float32)

    # ------------------------------------------------------------------
    # 内部辅助函数
    # ------------------------------------------------------------------

    def _optional_value(
        self,
        value: Optional[float],
        default: float,
        large_invalid: bool = False,
    ) -> float:
        """将可选值转成安全可用的浮点数。

        - `large_invalid=True` 时，会把 1e6 这类“缺省占位极大值”视为无效；
        - 其余情况下，只要数值有限就直接使用。
        """

        if value is None:
            return float(default)
        value = float(value)
        if not np.isfinite(value):
            return float(default)
        if large_invalid and value >= 1e5:
            return float(default)
        if value <= -0.5:
            return float(default)
        return float(value)

    def _is_valid_optional(self, value: Optional[float]) -> bool:
        """判断一个可选浮点数是否是真正可用的数值。"""
        if value is None:
            return False
        value = float(value)
        if not np.isfinite(value):
            return False
        if value >= 1e5 or value <= -0.5:
            return False
        return True

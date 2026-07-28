"""Enhanced v2 scene splitter.

The implementation reuses the legacy high-precision scene filtering, hard
labels, and abstention boundaries. It only appends sample-quality,
continuous-style, and condition-tag layers after ``super().split(...)`` so
the established retrieval-memory behavior remains intact.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np

from .schema import StyleSceneSplitResultV2
from .legacy.splitter import (
    StyleSceneSplitter,
    _clip01,
    _nonzero_xy_mask,
    _resolve_neighbor_valid_mask,
    _score_falling,
    _score_rising,
)


class StyleSceneSplitterV2(StyleSceneSplitter):
    """Add quality, continuous-style, and condition layers to the legacy split.

    The legacy scene/style split remains authoritative; the added continuous
    representation does not alter any existing decision boundary.
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
        # Keep sample-quality and split-validity thresholds independent.
        self.min_quality_score = float(min_quality_score)

    def split(self, cache_data: Dict[str, np.ndarray]) -> StyleSceneSplitResultV2:
        """Run the v2 split and append the added metadata layers."""

        base_result = super().split(cache_data)

        sample_quality_valid, quality_score, quality_reason, quality_vec = self._compute_sample_quality(
            cache_data=cache_data,
            base_result=base_result,
        )
        density_level, speed_regime, curvature_level = self._infer_condition_tags(base_result)
        style_performance_vec, style_performance_confidence = self._build_style_performance_vec(base_result)

        # Memory eligibility requires both reliable data and a reliable split.
        memory_eligible = bool(sample_quality_valid and base_result.split_valid)

        # Invalid-quality samples must not enter downstream memory construction.
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
    # V2 addition: sample-quality layer
    # ------------------------------------------------------------------

    def _compute_sample_quality(
        self,
        cache_data: Dict[str, np.ndarray],
        base_result,
    ) -> Tuple[bool, float, str, np.ndarray]:
        """Compute a permissive, interpretable sample-reliability score."""

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

        # Prioritize trajectory and neighbor validity over route metadata.
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
        """Measure ego-future completeness without penalizing low-speed motion."""

        ego_future = np.asarray(cache_data.get("ego_agent_future", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        if ego_future.ndim != 2 or ego_future.shape[0] == 0:
            return 0.0

        xy = ego_future[:, :2]
        finite_ratio = float(np.isfinite(xy).all(axis=1).mean()) if xy.size > 0 else 0.0
        # Stationary points are valid unless the full trajectory is bad padding.
        nonzero_ratio = float((np.linalg.norm(xy, axis=-1) > 1e-5).mean()) if xy.size > 0 else 0.0

        # Finite-value coverage dominates; nonzero coverage is secondary.
        return _clip01(0.80 * finite_ratio + 0.20 * max(nonzero_ratio, 0.25))

    def _neighbor_consistency_score(self, cache_data: Dict[str, np.ndarray]) -> float:
        """Measure consistency between neighbor masks and nonzero trajectories.

        Cache variants may provide a direct mask, an inverted mask, or no mask.
        This score only detects clear inconsistencies rather than reconstructing
        ground-truth validity.
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
        # Without an explicit mask, assign a moderately high non-perfect score.
            return 0.75

        raw_mask = np.asarray(raw_mask, dtype=bool)
        if raw_mask.shape != nonzero_mask.shape:
            return 0.40

        direct_overlap = int(np.logical_and(raw_mask, nonzero_mask).sum())
        inverse_overlap = int(np.logical_and(~raw_mask, nonzero_mask).sum())
        denom = max(int(nonzero_mask.sum()), 1)
        overlap_ratio = max(direct_overlap, inverse_overlap) / denom

        # Decrease the score gradually as mask/trajectory agreement weakens.
        return _clip01(overlap_ratio)

    def _route_metadata_score(self, base_result) -> float:
        """Measure whether route/map metadata is adequate for analysis."""

        route_lane_term = 1.0 if int(base_result.route_lane_count) > 0 else 0.0
        speed_limit_term = 1.0 if self._is_valid_optional(base_result.route_speed_limit_mps) else 0.65
        control_term = 0.85 if bool(base_result.route_has_control) else 1.0
        return _clip01(0.45 * route_lane_term + 0.35 * speed_limit_term + 0.20 * control_term)

    def _metric_sanity_score(self, base_result) -> float:
        """Apply permissive sanity checks to core dynamic metrics."""

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
    # V2 addition: condition-tag layer
    # ------------------------------------------------------------------

    def _infer_condition_tags(self, base_result) -> Tuple[str, str, str]:
        """Infer auxiliary condition tags outside the primary scene bucket."""

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
    # V2 addition: continuous-style layer
    # ------------------------------------------------------------------

    def _build_style_performance_vec(self, base_result) -> Tuple[np.ndarray, float]:
        """Build a three-axis continuous-style vector for a valid scene bucket.

        The vector supplements rather than replaces discrete labels. It uses
        existing interpretable metrics and keeps every component in [0, 1].
        """

        if base_result.scene_bucket == "straight_free_drive":
            style_vec = self._free_drive_style_vec(base_result)
        elif base_result.scene_bucket == "straight_car_follow":
            style_vec = self._car_follow_style_vec(base_result)
        elif base_result.scene_bucket == "straight_lane_change":
            style_vec = self._lane_change_style_vec(base_result)
        else:
            style_vec = np.zeros((3,), dtype=np.float32)

        # Vector confidence depends on scene/metric reliability, not hard labels.
        confidence = _clip01(
            0.45 * float(base_result.scene_confidence)
            + 0.35 * float(base_result.style_confidence)
            + 0.20 * float(base_result.split_confidence)
        )
        if base_result.scene_bucket == "none":
            confidence = 0.0

        return style_vec.astype(np.float32), float(confidence)

    def _free_drive_style_vec(self, base_result) -> np.ndarray:
        """Build free-driving axes: speed, longitudinal intensity, smoothness."""

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
        """Build car-following axes: headway, decisiveness, and smoothness."""

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
        """Build lane-change axes: gap acceptance, commitment, and smoothness."""

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _optional_value(
        self,
        value: Optional[float],
        default: float,
        large_invalid: bool = False,
    ) -> float:
        """Convert an optional value to a safe finite float.

        With ``large_invalid=True``, large sentinel values such as 1e6 are
        rejected.
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
        """Return whether an optional float is genuinely usable."""
        if value is None:
            return False
        value = float(value)
        if not np.isfinite(value):
            return False
        if value >= 1e5 or value <= -0.5:
            return False
        return True

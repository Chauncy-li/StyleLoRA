"""Safety-first selection for anchor warm-start diffusion candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from baseline.simulation.anchor_generator import AnchorBundle, AnchorCandidate


@dataclass(frozen=True)
class CandidateEvaluation:
    intent: str
    hard_valid: bool
    score: float
    failure_reasons: Tuple[str, ...]
    metrics: Dict[str, float]


@dataclass(frozen=True)
class SelectionResult:
    selected_prediction: torch.Tensor  # [P, T, 4]
    selected_candidate_index: int  # -1 means deterministic brake fallback
    used_fallback: bool
    evaluations: Tuple[CandidateEvaluation, ...]


class SafetyCandidateSelector:
    """Reject unsafe candidates first, then rank the remaining trajectories."""

    def __init__(self, config, *, step_interval: float) -> None:
        self.dt = float(step_interval)
        self.max_speed_mps = float(getattr(config, "selector_max_speed_mps", 35.0))
        self.max_abs_accel_mps2 = float(getattr(config, "selector_max_abs_accel_mps2", 12.0))
        self.max_abs_jerk_mps3 = float(getattr(config, "selector_max_abs_jerk_mps3", 45.0))
        self.max_curvature_1pm = float(getattr(config, "selector_max_curvature_1pm", 0.55))
        self.max_offroad_fraction = float(getattr(config, "selector_max_offroad_fraction", 0.08))
        self.max_wrong_way_fraction = float(getattr(config, "selector_max_wrong_way_fraction", 0.10))
        self.collision_margin_m = float(getattr(config, "selector_collision_margin_m", 0.25))
        self.hard_min_clearance_m = float(getattr(config, "selector_hard_min_clearance_m", 0.0))
        self.drivable_check_stride = max(1, int(getattr(config, "selector_drivable_check_stride", 4)))
        self.anchor_deviation_weight = float(getattr(config, "selector_anchor_deviation_weight", 0.35))
        self.clearance_weight = float(getattr(config, "selector_clearance_weight", 3.0))
        self.comfort_weight = float(getattr(config, "selector_comfort_weight", 0.15))
        self.progress_weight = float(getattr(config, "selector_progress_weight", 0.05))
        self.ego_vehicle = get_pacifica_parameters()

    def select(
        self,
        prediction: torch.Tensor,
        anchors: Sequence[AnchorCandidate],
        bundle: AnchorBundle,
        raw_inputs: Dict[str, torch.Tensor],
        ego_state: EgoState,
        map_api: AbstractMap,
    ) -> SelectionResult:
        """Select from ``prediction[K, P, T, 4]`` without changing model output semantics."""

        if prediction.ndim != 4 or prediction.shape[0] != len(anchors):
            raise ValueError(
                "Candidate prediction must have shape [K, P, T, 4] matching anchors; "
                f"got prediction={tuple(prediction.shape)}, anchors={len(anchors)}"
            )
        prediction_np = prediction.detach().float().cpu().numpy()
        evaluations: List[CandidateEvaluation] = []
        for index, anchor in enumerate(anchors):
            evaluations.append(
                self._evaluate(
                    prediction_np[index],
                    anchor,
                    raw_inputs,
                    ego_state,
                    map_api,
                )
            )

        valid_indices = [index for index, item in enumerate(evaluations) if item.hard_valid]
        if valid_indices:
            selected_index = min(valid_indices, key=lambda index: evaluations[index].score)
            selected = prediction[selected_index].clone()
            return SelectionResult(
                selected_prediction=selected,
                selected_candidate_index=int(selected_index),
                used_fallback=False,
                evaluations=tuple(evaluations),
            )

        # Preserve candidate-specific neighbor prediction for logging/safety
        # analysis, but execute a deterministic in-lane braking ego trajectory.
        if len(anchors) > 0:
            keep_index = next((i for i, anchor in enumerate(anchors) if anchor.intent == "keep"), 0)
            selected = prediction[keep_index].clone()
        else:
            raise RuntimeError("Anchor generator returned no candidates; a keep anchor is required")
        fallback = torch.as_tensor(
            bundle.fallback_ego_future_local,
            dtype=selected.dtype,
            device=selected.device,
        )
        selected[0] = fallback
        return SelectionResult(
            selected_prediction=selected,
            selected_candidate_index=-1,
            used_fallback=True,
            evaluations=tuple(evaluations),
        )

    def _evaluate(
        self,
        joint_prediction: np.ndarray,
        anchor: AnchorCandidate,
        raw_inputs: Dict[str, torch.Tensor],
        ego_state: EgoState,
        map_api: AbstractMap,
    ) -> CandidateEvaluation:
        ego = np.asarray(joint_prediction[0], dtype=np.float64)
        failures: List[str] = []
        if ego.ndim != 2 or ego.shape[-1] != 4 or not np.isfinite(ego).all():
            return CandidateEvaluation(
                intent=anchor.intent,
                hard_valid=False,
                score=float("inf"),
                failure_reasons=("non_finite_trajectory",),
                metrics={},
            )

        dynamics = self._dynamic_metrics(ego, float(ego_state.dynamic_car_state.speed))
        if dynamics["max_speed_mps"] > self.max_speed_mps:
            failures.append("speed_limit")
        if dynamics["max_abs_accel_mps2"] > self.max_abs_accel_mps2:
            failures.append("acceleration_limit")
        if dynamics["max_abs_jerk_mps3"] > self.max_abs_jerk_mps3:
            failures.append("jerk_limit")
        if dynamics["max_curvature_1pm"] > self.max_curvature_1pm:
            failures.append("curvature_limit")

        offroad_fraction = self._offroad_fraction(ego, ego_state, map_api)
        if offroad_fraction > self.max_offroad_fraction:
            failures.append("off_drivable_area")

        wrong_way_fraction = self._wrong_way_fraction(ego, anchor.ego_future_local)
        if wrong_way_fraction > self.max_wrong_way_fraction:
            failures.append("maneuver_direction_lost")

        model_clearance = self._joint_min_clearance(joint_prediction, raw_inputs)
        cv_clearance = self._constant_velocity_min_clearance(ego, raw_inputs)
        min_clearance = min(model_clearance, cv_clearance)
        if min_clearance < self.hard_min_clearance_m:
            failures.append("predicted_collision")

        anchor_xy = np.asarray(anchor.ego_future_local[:, :2], dtype=np.float64)
        anchor_deviation = float(np.mean(np.linalg.norm(ego[:, :2] - anchor_xy, axis=-1)))
        progress = float(np.sum(np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 2)), ego[:, :2]], axis=0), axis=0), axis=-1)))
        clearance_cost = float(np.exp(-max(min_clearance, 0.0) / 2.0)) if np.isfinite(min_clearance) else 0.0
        comfort_cost = (
            0.05 * dynamics["mean_abs_accel_mps2"]
            + 0.02 * dynamics["mean_abs_jerk_mps3"]
            + dynamics["mean_curvature_1pm"]
        )
        score = (
            self.anchor_deviation_weight * anchor_deviation
            + self.clearance_weight * clearance_cost
            + self.comfort_weight * comfort_cost
            + 20.0 * offroad_fraction
            + 10.0 * wrong_way_fraction
            - self.progress_weight * progress
        )
        metrics = {
            **dynamics,
            "offroad_fraction": float(offroad_fraction),
            "wrong_way_fraction": float(wrong_way_fraction),
            "min_predicted_clearance_m": float(min_clearance),
            "min_model_neighbor_clearance_m": float(model_clearance),
            "min_constant_velocity_clearance_m": float(cv_clearance),
            "anchor_mean_deviation_m": anchor_deviation,
            "progress_m": progress,
        }
        return CandidateEvaluation(
            intent=anchor.intent,
            hard_valid=not failures,
            score=float(score),
            failure_reasons=tuple(failures),
            metrics=metrics,
        )

    def _dynamic_metrics(self, ego: np.ndarray, current_speed_mps: float) -> Dict[str, float]:
        xy = np.concatenate([np.zeros((1, 2), dtype=np.float64), ego[:, :2]], axis=0)
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / self.dt
        accel = np.diff(np.concatenate([[current_speed_mps], speed])) / self.dt
        jerk = np.diff(np.concatenate([[accel[0] if len(accel) else 0.0], accel])) / self.dt
        heading = np.unwrap(np.arctan2(ego[:, 3], ego[:, 2]))
        heading = np.concatenate([[0.0], heading])
        distance = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
        curvature = np.abs(np.diff(heading)) / np.maximum(distance, 0.2)
        return {
            "max_speed_mps": float(np.max(speed, initial=0.0)),
            "max_abs_accel_mps2": float(np.max(np.abs(accel), initial=0.0)),
            "mean_abs_accel_mps2": float(np.mean(np.abs(accel))) if len(accel) else 0.0,
            "max_abs_jerk_mps3": float(np.max(np.abs(jerk), initial=0.0)),
            "mean_abs_jerk_mps3": float(np.mean(np.abs(jerk))) if len(jerk) else 0.0,
            "max_curvature_1pm": float(np.max(curvature, initial=0.0)),
            "mean_curvature_1pm": float(np.mean(curvature)) if len(curvature) else 0.0,
        }

    def _offroad_fraction(self, ego: np.ndarray, ego_state: EgoState, map_api: AbstractMap) -> float:
        local_xy = ego[:, :2]
        global_xy = self._local_to_global_xy(local_xy, ego_state)
        heading_local = np.arctan2(ego[:, 3], ego[:, 2])
        heading_global = heading_local + float(ego_state.rear_axle.heading)
        half_width = 0.5 * float(self.ego_vehicle.width)
        lateral = np.stack([-np.sin(heading_global), np.cos(heading_global)], axis=-1)
        forward = np.stack([np.cos(heading_global), np.sin(heading_global)], axis=-1)
        longitudinal_offsets = (
            -float(self.ego_vehicle.rear_length),
            float(self.ego_vehicle.rear_axle_to_center),
            float(self.ego_vehicle.front_length),
        )
        lateral_offsets = (-half_width, 0.0, half_width)
        check_points = np.stack(
            [
                global_xy + longitudinal * forward + lateral_offset * lateral
                for longitudinal in longitudinal_offsets
                for lateral_offset in lateral_offsets
            ],
            axis=1,
        )
        indices = list(range(0, len(check_points), self.drivable_check_stride))
        if indices[-1] != len(check_points) - 1:
            indices.append(len(check_points) - 1)
        results: List[bool] = []
        try:
            for index in indices:
                for point_xy in check_points[index]:
                    results.append(
                        bool(
                            map_api.is_in_layer(
                                Point2D(float(point_xy[0]), float(point_xy[1])),
                                SemanticMapLayer.DRIVABLE_AREA,
                            )
                        )
                    )
        except Exception:
            return 0.0
        return float(1.0 - np.mean(results)) if results else 0.0

    @staticmethod
    def _wrong_way_fraction(ego: np.ndarray, anchor: np.ndarray) -> float:
        ego_xy = np.concatenate([np.zeros((1, 2)), ego[:, :2]], axis=0)
        anchor_xy = np.concatenate([np.zeros((1, 2)), anchor[:, :2]], axis=0)
        ego_delta = np.diff(ego_xy, axis=0)
        anchor_delta = np.diff(anchor_xy, axis=0)
        moving = (np.linalg.norm(ego_delta, axis=-1) > 0.05) & (
            np.linalg.norm(anchor_delta, axis=-1) > 0.05
        )
        if not np.any(moving):
            return 0.0
        dot = np.sum(ego_delta[moving] * anchor_delta[moving], axis=-1)
        return float(np.mean(dot < 0.0))

    def _joint_min_clearance(
        self, joint_prediction: np.ndarray, raw_inputs: Dict[str, torch.Tensor]
    ) -> float:
        if joint_prediction.shape[0] <= 1:
            return float("inf")
        neighbor_past = raw_inputs.get("neighbor_agents_past")
        if neighbor_past is None:
            return float("inf")
        current = neighbor_past[0, : joint_prediction.shape[0] - 1, -1].detach().float().cpu().numpy()
        valid = np.any(np.abs(current[:, :4]) > 1e-5, axis=-1)
        if not np.any(valid):
            return float("inf")

        neighbors = np.asarray(joint_prediction[1 : 1 + len(current)], dtype=np.float64)[valid]
        current = current[valid]
        if not np.isfinite(neighbors).all():
            return float("-inf")
        return self._min_clearance_to_neighbors(joint_prediction[0], neighbors, current)

    def _constant_velocity_min_clearance(
        self,
        ego: np.ndarray,
        raw_inputs: Dict[str, torch.Tensor],
    ) -> float:
        """Conservative second opinion independent of learned neighbor output."""

        neighbor_past = raw_inputs.get("neighbor_agents_past")
        if neighbor_past is None:
            return float("inf")
        current = neighbor_past[0, :, -1].detach().float().cpu().numpy()
        valid = np.any(np.abs(current[:, :4]) > 1e-5, axis=-1)
        if not np.any(valid):
            return float("inf")
        current = current[valid]
        time = np.arange(1, len(ego) + 1, dtype=np.float64) * self.dt
        xy = current[:, None, :2] + current[:, None, 4:6] * time[None, :, None]
        heading = np.broadcast_to(current[:, None, 2:4], (len(current), len(ego), 2))
        neighbors = np.concatenate([xy, heading], axis=-1)
        return self._min_clearance_to_neighbors(ego, neighbors, current)

    def _min_clearance_to_neighbors(
        self,
        ego: np.ndarray,
        neighbors: np.ndarray,
        current: np.ndarray,
    ) -> float:
        if len(neighbors) == 0:
            return float("inf")
        neighbor_width = np.maximum(current[:, 6], 0.5)
        neighbor_length = np.maximum(current[:, 7], neighbor_width)
        ego = np.asarray(ego, dtype=np.float64)
        ego_width = float(self.ego_vehicle.width)
        ego_length = float(self.ego_vehicle.length)

        ego_centers, ego_radius = self._vehicle_discs(
            ego,
            ego_length,
            ego_width,
            origin_to_center=float(self.ego_vehicle.rear_axle_to_center),
        )
        neighbor_centers = []
        neighbor_radii = []
        for index in range(len(neighbors)):
            centers, radius = self._vehicle_discs(
                neighbors[index], float(neighbor_length[index]), float(neighbor_width[index])
            )
            neighbor_centers.append(centers)
            neighbor_radii.append(radius)
        neighbor_centers_np = np.stack(neighbor_centers, axis=0)  # [N, T, 3, 2]
        neighbor_radii_np = np.asarray(neighbor_radii, dtype=np.float64)
        delta = ego_centers[None, :, :, None, :] - neighbor_centers_np[:, :, None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        clearance = distance - (
            ego_radius + neighbor_radii_np[:, None, None, None] + self.collision_margin_m
        )
        return float(np.min(clearance))

    @staticmethod
    def _vehicle_discs(
        states: np.ndarray,
        length: float,
        width: float,
        *,
        origin_to_center: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        heading = states[:, 2:4]
        norm = np.linalg.norm(heading, axis=-1, keepdims=True)
        heading = heading / np.maximum(norm, 1e-6)
        offset = max(0.5 * (length - width), 0.0)
        offsets = np.asarray([-offset, 0.0, offset], dtype=np.float64)
        vehicle_center = states[:, :2] + heading * float(origin_to_center)
        centers = vehicle_center[:, None, :] + heading[:, None, :] * offsets[None, :, None]
        return centers, 0.5 * width

    @staticmethod
    def _local_to_global_xy(local_xy: np.ndarray, ego_state: EgoState) -> np.ndarray:
        heading = float(ego_state.rear_axle.heading)
        c, s = np.cos(heading), np.sin(heading)
        origin = np.asarray([ego_state.rear_axle.x, ego_state.rear_axle.y], dtype=np.float64)
        delta = np.stack(
            [local_xy[:, 0] * c - local_xy[:, 1] * s, local_xy[:, 0] * s + local_xy[:, 1] * c],
            axis=-1,
        )
        return delta + origin[None, :]

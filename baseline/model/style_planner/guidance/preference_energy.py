"""Condition-calibrated behavior-axis objectives for StylePlanner."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from research_v1.stylization.soft_metrics import (
    safe_headway_series,
    safe_ttc_series,
    soft_high_quantile,
    soft_low_quantile,
)


SCENE_ORDER = (
    "straight_free_drive",
    "straight_car_follow",
    "straight_lane_change",
)
CONTROLLED_SCENES = frozenset(SCENE_ORDER[:2])
EGO_LENGTH_M = 5.176
CONDITIONAL_CONTEXT_EGO_LENGTH_M = 4.8
EXPECTED_CONDITION_NAMES = {
    "straight_free_drive": (
        "initial_speed_over_limit",
        "route_speed_limit_mps",
        "route_curvature_1pm",
    ),
    "straight_car_follow": (
        "initial_headway_s",
        "initial_closing_speed_mps",
        "initial_ego_long_speed_mps",
        "initial_lead_long_speed_mps",
        "initial_lead_long_accel_mps2",
    ),
}
def _empty_support_diagnostics(
    *,
    batch_size: int,
    device: torch.device,
    reason_code: int,
) -> Dict[str, torch.Tensor]:
    return {
        "support_reason_code": torch.full(
            (batch_size,),
            int(reason_code),
            dtype=torch.long,
            device=device,
        ),
        "shared_condition_count": torch.zeros(
            (batch_size,),
            dtype=torch.long,
            device=device,
        ),
        "speed_limit_source_code": torch.zeros(
            (batch_size,),
            dtype=torch.long,
            device=device,
        ),
        "speed_limit_valid": torch.zeros(
            (batch_size,),
            dtype=torch.bool,
            device=device,
        ),
        "route_curvature_valid": torch.zeros(
            (batch_size,),
            dtype=torch.bool,
            device=device,
        ),
        "free_drive_clear": torch.zeros(
            (batch_size,),
            dtype=torch.bool,
            device=device,
        ),
        "active_traffic_control": torch.zeros(
            (batch_size,),
            dtype=torch.bool,
            device=device,
        ),
        "reference_valid_axis_mask": torch.zeros(
            (batch_size, 3),
            dtype=torch.bool,
            device=device,
        ),
    }


def _read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return dict(json.load(file_obj))


def _resolve_reference_path(model_path: str | Path, reference_path: str) -> Path:
    candidate = Path(reference_path)
    if candidate.is_file():
        return candidate
    sibling = Path(model_path).resolve().parent / candidate.name
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        f"Conditional-percentile reference not found: {reference_path!r}; "
        f"also checked {str(sibling)!r}"
    )


def _as_batch_tensor(
    inputs: Mapping[str, Any],
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    value = inputs.get(key)
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device)
    if tensor.numel() == 0:
        return None
    return tensor.to(dtype=dtype)


def _finite_difference(values: torch.Tensor, dt: float) -> torch.Tensor:
    if values.shape[0] <= 1:
        return values.new_zeros((0,))
    return (values[1:] - values[:-1]) / max(float(dt), 1e-6)


def _speed_from_xy(xy: torch.Tensor, dt: float) -> torch.Tensor:
    if xy.shape[0] <= 1:
        return xy.new_zeros((0,))
    return torch.linalg.norm(xy[1:] - xy[:-1], dim=-1) / max(float(dt), 1e-6)


def _weighted_high_quantile(
    values: torch.Tensor,
    opportunity: torch.Tensor,
    *,
    temperature: float = 12.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.numel() == 0:
        zero = values.new_zeros(())
        return zero, zero
    opportunity = opportunity.clamp(0.0, 1.0)
    logits = max(float(temperature), 1e-3) * values + torch.log(opportunity.clamp_min(1e-6))
    weights = torch.softmax(logits, dim=0)
    value = torch.sum(weights * values)
    confidence = 1.0 - torch.exp(-opportunity.sum())
    return value, confidence.clamp(0.0, 1.0)


def _first_event_value(
    values: torch.Tensor,
    activation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.numel() == 0:
        zero = values.new_zeros(())
        return zero, zero
    activation = activation.clamp(0.0, 1.0)
    survival = torch.cumprod(
        torch.cat([activation.new_ones((1,)), (1.0 - activation[:-1]).clamp_min(1e-4)]),
        dim=0,
    )
    first_weight = activation * survival
    confidence = first_weight.sum().clamp(0.0, 1.0)
    value = torch.sum(first_weight * values) / first_weight.sum().clamp_min(1e-6)
    return value, confidence


class ConditionalPreferenceEnergy:
    """Provide frozen conditional references and differentiable behavior axes.

    The historical class name is retained because current training/evaluation
    scripts import it directly. Sampling-time energy guidance has been removed;
    this object now serves only NCQT, signed-axis losses, and read-only generated
    trajectory diagnostics.
    """

    def __init__(
        self,
        *,
        normalization_path: str,
        conditional_rank_model_path: str,
        state_normalizer: Any,
        neighbours: int = 64,
        min_shared_condition_features: int = 3,
        cdf_temperature: float = 0.04,
        free_drive_accel_support_mode: str = "self_generated",
        dt: float = 0.1,
    ) -> None:
        if not normalization_path or not conditional_rank_model_path:
            raise ValueError(
                "Conditional axis objective requires train v5_normalization.json and "
                "v5_conditional_rank_model.json paths"
            )
        self.normalization_path = str(normalization_path)
        self.conditional_rank_model_path = str(conditional_rank_model_path)
        self.state_normalizer = state_normalizer
        self.neighbours = max(int(neighbours), 1)
        self.min_shared_condition_features = max(int(min_shared_condition_features), 1)
        self.cdf_temperature = max(float(cdf_temperature), 1e-3)
        self.free_drive_accel_support_mode = str(
            free_drive_accel_support_mode
        )
        if self.free_drive_accel_support_mode not in {
            "self_generated",
            "normal_anchor",
        }:
            raise ValueError(
                "free_drive_accel_support_mode must be 'self_generated' or "
                "'normal_anchor'"
            )
        self.dt = max(float(dt), 1e-3)
        self._last_diagnostics: Dict[str, torch.Tensor] | None = None
        # Lazily built per (scene, available causal-context dimensions).
        # Full sorting over the 8k/178k train reference banks for every batch
        # sample is far too expensive for end-to-end training.
        self._knn_indexes: Dict[tuple[str, tuple[int, ...]], Dict[str, Any]] = {}

        normalization = _read_json(self.normalization_path)
        rank_model = _read_json(self.conditional_rank_model_path)
        if str(normalization.get("artifact", "")) != "normalization_model":
            raise ValueError(
                f"Not a V5 normalization model: {self.normalization_path}"
            )
        if str(normalization.get("fit_split", "")) != "train_only_input":
            raise ValueError(
                "Conditional axis objective requires a train-only V5 normalization "
                f"artifact, got fit_split={normalization.get('fit_split')!r}"
            )
        if str(rank_model.get("artifact", "")) != "conditional_rank_model":
            raise ValueError(
                "Not a V5 conditional-rank model: "
                f"{self.conditional_rank_model_path}"
            )
        if str(rank_model.get("fit_split", "")) != "train_only_oof":
            raise ValueError(
                "Conditional axis objective requires a train-only V5 conditional-rank "
                f"artifact, got fit_split={rank_model.get('fit_split')!r}"
            )
        self.min_effective_neighbours = max(
            float(rank_model.get("min_effective_neighbours", 32.0)),
            1.0,
        )
        self._normalization: Dict[str, Dict[str, np.ndarray]] = {}
        self._references: Dict[str, Dict[str, np.ndarray]] = {}
        for scene in CONTROLLED_SCENES:
            scene_norm = dict(dict(normalization.get("scenes", {})).get(scene, {}))
            axis_stats = list(scene_norm.get("axis_stats", []))
            if len(axis_stats) != 3:
                raise ValueError(f"Normalization model has no three-axis stats for {scene}")
            self._normalization[scene] = {
                "q_low": np.asarray([float(item["q_low"]) for item in axis_stats], dtype=np.float32),
                "q_high": np.asarray([float(item["q_high"]) for item in axis_stats], dtype=np.float32),
                "falling": np.asarray(
                    [str(item.get("direction", "rising")) == "falling" for item in axis_stats],
                    dtype=bool,
                ),
            }

            scene_rank = dict(dict(rank_model.get("scenes", {})).get(scene, {}))
            condition_names = tuple(
                str(value) for value in scene_rank.get("condition_names", [])
            )
            if condition_names != EXPECTED_CONDITION_NAMES[scene]:
                raise ValueError(
                    f"Conditional context contract mismatch for {scene}: "
                    f"expected={EXPECTED_CONDITION_NAMES[scene]}, "
                    f"got={condition_names}"
                )
            reference_path = _resolve_reference_path(
                self.conditional_rank_model_path,
                str(scene_rank.get("reference_npz", "")),
            )
            with np.load(reference_path, allow_pickle=False) as payload:
                self._references[scene] = {
                    "condition_values": np.asarray(payload["condition_values"], dtype=np.float32),
                    "condition_valid_mask": np.asarray(payload["condition_valid_mask"], dtype=bool),
                    "canonical_style_vec": np.asarray(payload["canonical_style_vec"], dtype=np.float32),
                    "canonical_axis_valid_mask": np.asarray(payload["canonical_axis_valid_mask"], dtype=bool),
                }

    def __deepcopy__(self, _memo):
        # ModelEma deep-copies the planner. Frozen NumPy reference banks are
        # immutable and can be safely shared instead of doubling host memory.
        return self

    @staticmethod
    def _route_speed_limit(
        speed_limit: np.ndarray,
        has_speed_limit: np.ndarray,
        route_mask: np.ndarray,
        route_lanes: np.ndarray | None = None,
    ) -> float | None:
        limits = speed_limit.reshape(-1)
        has_limit = has_speed_limit.reshape(-1).astype(bool)
        lane_count = min(limits.shape[0], has_limit.shape[0])
        if route_lanes is not None and route_lanes.ndim == 3:
            lane_count = min(lane_count, route_lanes.shape[0])
            best: tuple[float, float] | None = None
            for lane_index in range(lane_count):
                if not bool(has_limit[lane_index]):
                    continue
                limit = float(limits[lane_index])
                if not math.isfinite(limit) or limit <= 0.5:
                    continue
                valid = (
                    route_mask[lane_index].astype(bool)
                    if route_mask.ndim == 2 and lane_index < route_mask.shape[0]
                    else np.ones((route_lanes.shape[1],), dtype=bool)
                )
                xy = route_lanes[lane_index, valid, :2]
                if xy.size == 0 or not np.all(np.isfinite(xy)):
                    continue
                distance_sq = float(np.min(np.sum(xy**2, axis=-1)))
                candidate = (distance_sq, limit)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            if best is not None:
                return float(best[1])

        valid_route = (
            np.any(route_mask.astype(bool), axis=-1)
            if route_mask.ndim >= 2
            else route_mask.astype(bool).reshape(-1)
        )
        valid = valid_route[:lane_count] & has_limit[:lane_count]
        values = limits[:lane_count][valid]
        values = values[np.isfinite(values) & (values > 0.5)]
        return float(np.median(values)) if values.size else None

    @staticmethod
    def _route_curvature(route_lanes: np.ndarray, route_mask: np.ndarray) -> float | None:
        candidates: list[tuple[float, float]] = []
        for lane_index in range(route_lanes.shape[0]):
            valid = route_mask[lane_index].astype(bool)
            xy = route_lanes[lane_index, valid, :2]
            if xy.shape[0] < 3 or not np.all(np.isfinite(xy)):
                continue
            delta = np.diff(xy, axis=0)
            length = np.linalg.norm(delta, axis=-1)
            usable = length > 1e-3
            if int(np.sum(usable)) < 2:
                continue
            heading = np.unwrap(np.arctan2(delta[usable, 1], delta[usable, 0]))
            ds = length[usable]
            curvature = np.abs(np.diff(heading)) / np.maximum(
                0.5 * (ds[1:] + ds[:-1]),
                1e-3,
            )
            finite = curvature[np.isfinite(curvature)]
            if finite.size:
                origin_distance = float(np.min(np.sum(xy**2, axis=-1)))
                candidates.append(
                    (
                        origin_distance,
                        float(np.clip(np.median(finite), 0.0, 0.5)),
                    )
                )
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _route_has_active_control(
        route_lanes: np.ndarray,
        route_mask: np.ndarray,
    ) -> bool:
        """Detect causal yellow/red control encoded on valid route points."""

        if route_lanes.ndim != 3 or route_lanes.shape[-1] < 11:
            return False
        valid = route_mask.astype(bool)
        traffic = route_lanes[..., 8:12]
        yellow_or_red = (traffic[..., 1] > 0.5) | (traffic[..., 2] > 0.5)
        return bool(np.any(yellow_or_red & valid))

    @staticmethod
    def _free_drive_currently_clear(
        ego_current: np.ndarray,
        neighbor_past: np.ndarray,
        neighbor_mask: np.ndarray,
    ) -> bool:
        """Conservative current-frame version of the V5 free-drive lead mask."""

        if neighbor_past.ndim != 3 or neighbor_past.shape[0] == 0:
            return True
        current = neighbor_past[:, -1]
        valid = neighbor_mask[:, -1].astype(bool)
        if current.shape[-1] > 8:
            valid &= current[:, 8] > 0.5
        valid &= np.all(np.isfinite(current[:, :2]), axis=-1)
        candidate = valid & (current[:, 0] > 0.0) & (np.abs(current[:, 1]) < 1.9)
        if not np.any(candidate):
            return True
        indices = np.where(candidate)[0]
        lengths = (
            current[indices, 7]
            if current.shape[-1] > 7
            else np.full(indices.shape, 4.5)
        )
        gap = np.maximum(
            current[indices, 0]
            - 0.5 * (EGO_LENGTH_M + np.maximum(lengths, 0.0)),
            0.0,
        )
        ego_speed = abs(float(ego_current[4])) if ego_current.shape[0] > 4 else 0.0
        required_clearance = max(30.0, 3.0 * ego_speed)
        return bool(np.all(gap >= required_clearance))

    @staticmethod
    def _current_context(
        *,
        scene: str,
        ego_current: np.ndarray,
        neighbor_past: np.ndarray,
        neighbor_mask: np.ndarray,
        route_lanes: np.ndarray,
        route_mask: np.ndarray,
        route_speed_limit: np.ndarray,
        route_has_speed_limit: np.ndarray,
        resolved_speed_limit: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        speed_limit = resolved_speed_limit
        if (
            speed_limit is None
            or not math.isfinite(float(speed_limit))
            or float(speed_limit) <= 0.5
        ):
            speed_limit = ConditionalPreferenceEnergy._route_speed_limit(
                route_speed_limit,
                route_has_speed_limit,
                route_mask,
                route_lanes,
            )
        curvature = ConditionalPreferenceEnergy._route_curvature(route_lanes, route_mask)
        ego_speed = float(ego_current[4]) if ego_current.shape[0] > 4 else 0.0

        if scene == "straight_free_drive":
            ratio = (
                float(np.clip(ego_speed / speed_limit, 0.0, 1.5))
                if speed_limit is not None and speed_limit > 0.5
                else None
            )
            raw: Sequence[float | None] = (ratio, speed_limit, curvature)
            lead_index = -1
        else:
            current = neighbor_past[:, -1]
            valid = neighbor_mask[:, -1].astype(bool)
            finite_xy = np.all(np.isfinite(current[:, :2]), axis=-1)
            candidate = (
                valid
                & finite_xy
                & (current[:, 0] > 0.0)
                & (np.abs(current[:, 1]) < 1.9)
            )
            lead_index = -1
            headway = closing = lead_speed = lead_accel = None
            if np.any(candidate):
                indices = np.where(candidate)[0]
                lengths = (
                    current[indices, 7]
                    if current.shape[-1] > 7
                    else np.full(indices.shape, 4.5)
                )
                gap = np.maximum(
                    current[indices, 0]
                    - 0.5
                    * (
                        CONDITIONAL_CONTEXT_EGO_LENGTH_M
                        + np.maximum(lengths, 0.0)
                    ),
                    0.0,
                )
                lead_index = int(indices[int(np.argmin(gap))])
                selected_gap = float(np.min(gap))
                lead_speed = float(current[lead_index, 4]) if current.shape[-1] > 4 else 0.0
                headway = selected_gap / max(abs(ego_speed), 1.5)
                closing = max(ego_speed - lead_speed, 0.0)
                if neighbor_past.shape[1] >= 2 and neighbor_mask[lead_index, -2]:
                    previous_speed = float(neighbor_past[lead_index, -2, 4])
                    lead_accel = (lead_speed - previous_speed) / 0.1
            raw = (headway, closing, ego_speed, lead_speed, lead_accel)

        valid = np.asarray(
            [value is not None and math.isfinite(float(value)) for value in raw],
            dtype=bool,
        )
        values = np.asarray(
            [float(value) if ok else 0.0 for value, ok in zip(raw, valid)],
            dtype=np.float32,
        )
        return values, valid, lead_index

    def _local_references(
        self,
        scene: str,
        query: np.ndarray,
        query_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reference = self._references[scene]
        canonical = reference["canonical_style_vec"]
        axis_valid = reference["canonical_axis_valid_mask"]

        shared_dims = np.where(query_valid)[0]
        if shared_dims.size < self.min_shared_condition_features:
            return (
                np.zeros((3, self.neighbours), dtype=np.float32),
                np.zeros((3, self.neighbours), dtype=np.float32),
                np.zeros((3,), dtype=bool),
            )
        index = self._get_knn_index(scene, shared_dims)
        if index is None:
            return (
                np.zeros((3, self.neighbours), dtype=np.float32),
                np.zeros((3, self.neighbours), dtype=np.float32),
                np.zeros((3,), dtype=bool),
            )

        standardized_query = (
            query[shared_dims] - index["median"]
        ) / index["scale"]
        model = index["model"]
        if model is not None:
            distances, local_positions = model.kneighbors(
                standardized_query.reshape(1, -1),
                return_distance=True,
            )
            ordered_distance = distances[0].astype(np.float32, copy=False)
            order = index["eligible_indices"][local_positions[0]]
        else:
            standardized_reference = index["standardized_reference"]
            distance = np.linalg.norm(
                standardized_reference - standardized_query,
                axis=1,
            )
            pool_size = int(index["pool_size"])
            local_positions = np.argpartition(
                distance,
                kth=pool_size - 1,
            )[:pool_size]
            local_order = np.argsort(distance[local_positions])
            local_positions = local_positions[local_order]
            ordered_distance = distance[local_positions]
            order = index["eligible_indices"][local_positions]

        values = np.zeros((3, self.neighbours), dtype=np.float32)
        weights = np.zeros_like(values)
        valid_output = np.zeros((3,), dtype=bool)
        order_list = order.tolist()
        for axis in range(3):
            selected_positions = [
                position
                for position, reference_index in enumerate(order_list)
                if axis_valid[reference_index, axis]
            ][: self.neighbours]
            if not selected_positions:
                continue
            positions = np.asarray(selected_positions, dtype=np.int64)
            reference_indices = order[positions]
            local_distance = ordered_distance[positions]
            bandwidth = max(float(local_distance[-1]), 0.05)
            local_weight = np.exp(-0.5 * (local_distance / bandwidth) ** 2)
            local_weight = local_weight / max(float(np.sum(local_weight)), 1e-12)
            effective_neighbours = 1.0 / max(
                float(np.sum(local_weight**2)),
                1e-12,
            )
            count = len(reference_indices)
            values[axis, :count] = canonical[reference_indices, axis]
            weights[axis, :count] = local_weight.astype(np.float32)
            valid_output[axis] = bool(
                count >= min(self.neighbours, 8)
                and effective_neighbours >= self.min_effective_neighbours
            )
        return values, weights, valid_output

    def _get_knn_index(
        self,
        scene: str,
        shared_dims: np.ndarray,
    ) -> Dict[str, Any] | None:
        """Build and cache a robustly standardized causal-context KNN index."""

        key = (scene, tuple(int(value) for value in shared_dims.tolist()))
        cached = self._knn_indexes.get(key)
        if cached is not None:
            return cached

        reference = self._references[scene]
        condition = reference["condition_values"]
        condition_valid = reference["condition_valid_mask"]
        eligible = np.all(condition_valid[:, shared_dims], axis=1)
        eligible_indices = np.where(eligible)[0]
        if eligible_indices.size == 0:
            return None

        local_condition = condition[eligible_indices][:, shared_dims]
        median = np.median(local_condition, axis=0).astype(np.float32)
        q25, q75 = np.quantile(local_condition, [0.25, 0.75], axis=0)
        scale = np.maximum(q75 - q25, 1e-3).astype(np.float32)
        standardized = ((local_condition - median) / scale).astype(
            np.float32,
            copy=False,
        )
        pool_size = min(
            max(self.neighbours * 16, 512),
            int(eligible_indices.size),
        )
        model = None
        try:
            from sklearn.neighbors import NearestNeighbors  # type: ignore

            model = NearestNeighbors(
                n_neighbors=pool_size,
                algorithm="auto",
                metric="euclidean",
                n_jobs=1,
            )
            model.fit(standardized)
        except ImportError:
            if eligible_indices.size > 30000:
                raise RuntimeError(
                    "scikit-learn is required for differentiable conditional-axis "
                    "objectives with more than 30k reference rows"
                )

        payload: Dict[str, Any] = {
            "eligible_indices": eligible_indices,
            "median": median,
            "scale": scale,
            "pool_size": pool_size,
            "model": model,
            "standardized_reference": standardized if model is None else None,
        }
        self._knn_indexes[key] = payload
        return payload

    @staticmethod
    def _support_debug_from_prepared(
        prepared: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        key_map = {
            "support_reason_code": "preference_axis_reference_support_reason_code",
            "shared_condition_count": (
                "preference_axis_reference_shared_condition_count"
            ),
            "speed_limit_source_code": (
                "preference_axis_reference_speed_limit_source_code"
            ),
            "speed_limit_valid": "preference_axis_reference_speed_limit_valid",
            "route_curvature_valid": (
                "preference_axis_reference_route_curvature_valid"
            ),
            "free_drive_clear": "preference_axis_reference_free_drive_clear",
            "active_traffic_control": (
                "preference_axis_reference_active_traffic_control"
            ),
            "reference_valid_axis_mask": (
                "preference_axis_reference_valid_axis_mask"
            ),
        }
        diagnostics: Dict[str, torch.Tensor] = {}
        for prepared_key, output_key in key_map.items():
            value = prepared.get(prepared_key)
            if torch.is_tensor(value):
                diagnostics[output_key] = value.detach()
        return diagnostics

    def prepare(
        self,
        inputs: Mapping[str, Any],
        style_condition: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if style_condition.ndim == 1:
            style_condition = style_condition.unsqueeze(0)
        device = style_condition.device
        condition = style_condition.detach().to(dtype=torch.float32)
        batch_size = condition.shape[0]
        scene_one_hot = condition[:, 6:9]
        scene_index = torch.argmax(scene_one_hot, dim=-1)
        scene_present = scene_one_hot.sum(dim=-1) > 0.5
        axis_mask = condition[:, 3:6] > 0.5
        target = condition[:, 0:3].clamp(0.0, 1.0)
        if not bool((scene_present & axis_mask.any(dim=-1)).any()):
            prepared = {
                "enabled": torch.zeros(
                    (batch_size,),
                    dtype=torch.bool,
                    device=device,
                ),
                "target": target,
                "axis_mask": torch.zeros(
                    (batch_size, 3),
                    dtype=torch.bool,
                    device=device,
                ),
                **_empty_support_diagnostics(
                    batch_size=batch_size,
                    device=device,
                    reason_code=1,
                ),
            }
            self._last_diagnostics = self._support_debug_from_prepared(prepared)
            return prepared

        ego_current = _as_batch_tensor(
            inputs, "preference_ego_current_state_raw", device=device
        )
        neighbor_past = _as_batch_tensor(
            inputs, "preference_neighbor_agents_past_raw", device=device
        )
        neighbor_mask = _as_batch_tensor(
            inputs,
            "preference_neighbor_agents_past_mask_raw",
            device=device,
            dtype=torch.bool,
        )
        route_lanes = _as_batch_tensor(
            inputs, "preference_route_lanes_raw", device=device
        )
        route_mask = _as_batch_tensor(
            inputs,
            "preference_route_lanes_mask_raw",
            device=device,
            dtype=torch.bool,
        )
        route_speed_limit = _as_batch_tensor(
            inputs, "preference_route_lanes_speed_limit_raw", device=device
        )
        route_has_speed_limit = _as_batch_tensor(
            inputs,
            "preference_route_lanes_has_speed_limit_raw",
            device=device,
            dtype=torch.bool,
        )
        # The V5/V6 condition-label contract falls back from route lanes to
        # ordinary map lanes when route-lane speed limits are unavailable.
        # These inputs are optional so old callers and checkpoints retain their
        # previous behaviour whenever route speed metadata is valid.
        lanes = _as_batch_tensor(
            inputs, "preference_lanes_raw", device=device
        )
        lanes_mask = _as_batch_tensor(
            inputs,
            "preference_lanes_mask_raw",
            device=device,
            dtype=torch.bool,
        )
        lanes_speed_limit = _as_batch_tensor(
            inputs, "preference_lanes_speed_limit_raw", device=device
        )
        lanes_has_speed_limit = _as_batch_tensor(
            inputs,
            "preference_lanes_has_speed_limit_raw",
            device=device,
            dtype=torch.bool,
        )
        required = (
            ego_current,
            neighbor_past,
            neighbor_mask,
            route_lanes,
            route_mask,
            route_speed_limit,
            route_has_speed_limit,
        )
        if any(value is None for value in required):
            prepared = {
                "enabled": torch.zeros((batch_size,), dtype=torch.bool, device=device),
                "target": target,
                "axis_mask": torch.zeros((batch_size, 3), dtype=torch.bool, device=device),
                **_empty_support_diagnostics(
                    batch_size=batch_size,
                    device=device,
                    reason_code=2,
                ),
            }
            self._last_diagnostics = self._support_debug_from_prepared(prepared)
            return prepared
        predicted_neighbor_count = int(neighbor_past.shape[1])
        neighbor_current_mask = inputs.get("neighbor_current_mask")
        if torch.is_tensor(neighbor_current_mask) and neighbor_current_mask.ndim >= 2:
            predicted_neighbor_count = min(
                predicted_neighbor_count,
                int(neighbor_current_mask.shape[1]),
            )
        neighbor_past = neighbor_past[:, :predicted_neighbor_count]
        neighbor_mask = neighbor_mask[:, :predicted_neighbor_count]

        ref_values = np.zeros((batch_size, 3, self.neighbours), dtype=np.float32)
        ref_weights = np.zeros_like(ref_values)
        ref_valid = np.zeros((batch_size, 3), dtype=bool)
        q_low = np.zeros((batch_size, 3), dtype=np.float32)
        q_high = np.ones((batch_size, 3), dtype=np.float32)
        falling = np.zeros((batch_size, 3), dtype=bool)
        lead_index = np.full((batch_size,), -1, dtype=np.int64)
        speed_limit_values = np.zeros((batch_size,), dtype=np.float32)
        speed_limit_valid = np.zeros((batch_size,), dtype=bool)
        support_reason_code = np.full((batch_size,), 1, dtype=np.int64)
        shared_condition_count = np.zeros((batch_size,), dtype=np.int64)
        speed_limit_source_code = np.zeros((batch_size,), dtype=np.int64)
        route_curvature_valid = np.zeros((batch_size,), dtype=bool)
        free_drive_clear = np.zeros((batch_size,), dtype=bool)
        active_traffic_control = np.zeros((batch_size,), dtype=bool)

        ego_np = ego_current.detach().cpu().numpy()
        neighbor_np = neighbor_past.detach().cpu().numpy()
        neighbor_mask_np = neighbor_mask.detach().cpu().numpy().astype(bool)
        route_np = route_lanes.detach().cpu().numpy()
        route_mask_np = route_mask.detach().cpu().numpy().astype(bool)
        route_speed_np = route_speed_limit.detach().cpu().numpy()
        route_has_speed_np = route_has_speed_limit.detach().cpu().numpy().astype(bool)
        lanes_np = lanes.detach().cpu().numpy() if lanes is not None else None
        lanes_mask_np = (
            lanes_mask.detach().cpu().numpy().astype(bool)
            if lanes_mask is not None
            else None
        )
        lanes_speed_np = (
            lanes_speed_limit.detach().cpu().numpy()
            if lanes_speed_limit is not None
            else None
        )
        lanes_has_speed_np = (
            lanes_has_speed_limit.detach().cpu().numpy().astype(bool)
            if lanes_has_speed_limit is not None
            else None
        )

        for batch_index in range(batch_size):
            if not bool(scene_present[batch_index]):
                continue
            scene = SCENE_ORDER[int(scene_index[batch_index].item())]
            if scene not in CONTROLLED_SCENES or not bool(axis_mask[batch_index].any()):
                continue
            limit = self._route_speed_limit(
                route_speed_np[batch_index],
                route_has_speed_np[batch_index],
                route_mask_np[batch_index],
                route_np[batch_index],
            )
            if limit is not None and limit > 0.5:
                speed_limit_source_code[batch_index] = 1
            elif (
                lanes_mask_np is not None
                and lanes_speed_np is not None
                and lanes_has_speed_np is not None
            ):
                limit = self._route_speed_limit(
                    lanes_speed_np[batch_index],
                    lanes_has_speed_np[batch_index],
                    lanes_mask_np[batch_index],
                    (
                        lanes_np[batch_index]
                        if lanes_np is not None
                        else None
                    ),
                )
                if limit is not None and limit > 0.5:
                    speed_limit_source_code[batch_index] = 2
            speed_limit_values[batch_index] = float(limit or 0.0)
            speed_limit_valid[batch_index] = limit is not None and limit > 0.5

            query, query_valid, selected_lead = self._current_context(
                scene=scene,
                ego_current=ego_np[batch_index],
                neighbor_past=neighbor_np[batch_index],
                neighbor_mask=neighbor_mask_np[batch_index],
                route_lanes=route_np[batch_index],
                route_mask=route_mask_np[batch_index],
                route_speed_limit=route_speed_np[batch_index],
                route_has_speed_limit=route_has_speed_np[batch_index],
                resolved_speed_limit=limit,
            )
            shared_condition_count[batch_index] = int(np.sum(query_valid))
            if scene == "straight_free_drive":
                route_curvature_valid[batch_index] = bool(query_valid[2])
            local_values, local_weights, local_valid = self._local_references(
                scene,
                query,
                query_valid,
            )
            # The frozen free-drive labels explicitly excluded traffic-control
            # influenced frames. Do not extrapolate their speed-preference
            # reference through a currently yellow/red route segment.
            if scene == "straight_free_drive":
                active_traffic_control[batch_index] = (
                    self._route_has_active_control(
                        route_np[batch_index],
                        route_mask_np[batch_index],
                    )
                )
                free_drive_clear[batch_index] = (
                    self._free_drive_currently_clear(
                        ego_np[batch_index],
                        neighbor_np[batch_index],
                        neighbor_mask_np[batch_index],
                    )
                )
                if (
                    active_traffic_control[batch_index]
                    or not free_drive_clear[batch_index]
                ):
                    local_valid[:] = False

            if active_traffic_control[batch_index]:
                support_reason_code[batch_index] = 5
            elif (
                scene == "straight_free_drive"
                and not free_drive_clear[batch_index]
            ):
                support_reason_code[batch_index] = 6
            elif (
                shared_condition_count[batch_index]
                < self.min_shared_condition_features
            ):
                support_reason_code[batch_index] = 3
            elif not bool(np.any(local_valid)):
                support_reason_code[batch_index] = 4
            elif not bool(
                np.any(
                    local_valid
                    & axis_mask[batch_index].detach().cpu().numpy().astype(bool)
                )
            ):
                support_reason_code[batch_index] = 7
            else:
                support_reason_code[batch_index] = 0

            ref_values[batch_index] = local_values
            ref_weights[batch_index] = local_weights
            ref_valid[batch_index] = local_valid
            q_low[batch_index] = self._normalization[scene]["q_low"]
            q_high[batch_index] = self._normalization[scene]["q_high"]
            falling[batch_index] = self._normalization[scene]["falling"]
            lead_index[batch_index] = selected_lead

        reference_valid = torch.as_tensor(ref_valid, device=device)
        effective_mask = axis_mask & reference_valid
        enabled = scene_present & effective_mask.any(dim=-1)
        prepared = {
            "enabled": enabled,
            "target": target,
            "axis_mask": effective_mask,
            "scene_index": scene_index,
            "reference_values": torch.as_tensor(ref_values, device=device),
            "reference_weights": torch.as_tensor(ref_weights, device=device),
            "q_low": torch.as_tensor(q_low, device=device),
            "q_high": torch.as_tensor(q_high, device=device),
            "falling": torch.as_tensor(falling, device=device),
            "lead_index": torch.as_tensor(lead_index, device=device),
            "speed_limit_mps": torch.as_tensor(speed_limit_values, device=device),
            "speed_limit_valid": torch.as_tensor(speed_limit_valid, device=device),
            "support_reason_code": torch.as_tensor(
                support_reason_code,
                device=device,
            ),
            "shared_condition_count": torch.as_tensor(
                shared_condition_count,
                device=device,
            ),
            "speed_limit_source_code": torch.as_tensor(
                speed_limit_source_code,
                device=device,
            ),
            "route_curvature_valid": torch.as_tensor(
                route_curvature_valid,
                device=device,
            ),
            "free_drive_clear": torch.as_tensor(
                free_drive_clear,
                device=device,
            ),
            "active_traffic_control": torch.as_tensor(
                active_traffic_control,
                device=device,
            ),
            "reference_valid_axis_mask": reference_valid,
            "ego_current": ego_current,
            "neighbor_past": neighbor_past,
            "neighbor_mask": neighbor_mask,
        }
        self._last_diagnostics = self._support_debug_from_prepared(prepared)
        return prepared

    def _free_drive_axes(
        self,
        ego_future: torch.Tensor,
        ego_current: torch.Tensor,
        speed_limit: torch.Tensor,
        *,
        accel_opportunity_reference_future: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_xy = torch.cat([ego_current[None, :2], ego_future[:, :2]], dim=0)
        speed = _speed_from_xy(full_xy, self.dt)
        acceleration = _finite_difference(speed, self.dt)
        limit = speed_limit.clamp_min(1.0)
        speed_utilization = torch.clamp(speed / limit, min=0.0, max=1.5).mean()

        if acceleration.numel() > 0:
            opportunity_speed = speed
            anchored_accel_support = (
                self.free_drive_accel_support_mode == "normal_anchor"
                and accel_opportunity_reference_future is not None
            )
            if anchored_accel_support:
                reference_full_xy = torch.cat(
                    [
                        ego_current[None, :2],
                        accel_opportunity_reference_future[:, :2],
                    ],
                    dim=0,
                )
                opportunity_speed = _speed_from_xy(
                    reference_full_xy,
                    self.dt,
                ).detach()
            support_count = min(
                int(acceleration.shape[0]),
                max(int(opportunity_speed.shape[0]) - 1, 0),
            )
            headroom = (
                limit - opportunity_speed[:support_count]
            ).clamp_min(0.0)
            opportunity = torch.sigmoid((headroom - 2.0) / 0.35)
            if anchored_accel_support:
                opportunity = opportunity.detach()
            positive_accel = torch.relu(acceleration)
            accel_willingness, accel_confidence = _weighted_high_quantile(
                positive_accel[:support_count],
                opportunity,
            )
        else:
            accel_willingness = ego_future.new_zeros(())
            accel_confidence = ego_future.new_zeros(())

        response_values = []
        response_opportunity = []
        window = max(int(round(2.0 / self.dt)), 2)
        for start in range(max(speed.shape[0] - window, 0)):
            start_speed = speed[start]
            headroom = (limit - start_speed).clamp_min(0.0)
            future_max = soft_high_quantile(speed[start : start + window + 1], temperature=10.0)
            response_values.append((future_max - start_speed).clamp_min(0.0) / headroom.clamp_min(2.0))
            response_opportunity.append(torch.sigmoid((headroom - 2.0) / 0.35))
        if response_values:
            response, response_confidence = _weighted_high_quantile(
                torch.stack(response_values).clamp(0.0, 1.5),
                torch.stack(response_opportunity),
            )
        else:
            response = ego_future.new_zeros(())
            response_confidence = ego_future.new_zeros(())
        values = torch.stack([speed_utilization, accel_willingness, response])
        confidence = torch.stack(
            [
                ego_future.new_ones(()),
                accel_confidence,
                response_confidence,
            ]
        )
        return values, confidence

    def _car_follow_axes(
        self,
        *,
        ego_future: torch.Tensor,
        ego_current: torch.Tensor,
        neighbor_future: torch.Tensor,
        neighbor_current: torch.Tensor,
        lead_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if lead_index < 0 or lead_index >= neighbor_future.shape[0]:
            return ego_future.new_zeros((3,)), ego_future.new_zeros((3,))
        lead_future = neighbor_future[lead_index].detach()
        lead_current = neighbor_current[lead_index].detach()
        ego_xy = torch.cat([ego_current[None, :2], ego_future[:, :2]], dim=0)
        lead_xy = torch.cat([lead_current[None, :2], lead_future[:, :2]], dim=0)
        ego_speed = _speed_from_xy(ego_xy, self.dt)
        lead_speed = _speed_from_xy(lead_xy, self.dt)
        count = min(ego_speed.shape[0], lead_speed.shape[0], ego_future.shape[0])
        if count < 2:
            return ego_future.new_zeros((3,)), ego_future.new_zeros((3,))

        lead_length = lead_current[7].clamp_min(3.0) if lead_current.shape[0] > 7 else ego_future.new_tensor(4.8)
        gap = (
            lead_future[:count, 0]
            - ego_future[:count, 0]
            - 0.5 * EGO_LENGTH_M
            - 0.5 * lead_length
        ).clamp_min(0.05)
        headway = safe_headway_series(gap, ego_speed[:count])
        closing = ego_speed[:count] - lead_speed[:count]
        ttc = safe_ttc_series(gap, closing, ttc_cap_s=10.0)
        closing_opportunity = torch.sigmoid((closing - 0.25) / 0.08)
        h_value = soft_low_quantile(headway, temperature=12.0)
        ttc_value, ttc_confidence = _weighted_high_quantile(
            -ttc,
            closing_opportunity,
            temperature=12.0,
        )
        ttc_value = -ttc_value

        acceleration = _finite_difference(ego_speed[:count], self.dt)
        if acceleration.numel() > 0:
            closing_relief = (closing[:-1] - closing[1:]).clamp_min(0.0)
            strong_brake = torch.sigmoid((-acceleration - 0.40) / 0.08)
            mild_brake = torch.sigmoid((-acceleration - 0.15) / 0.05)
            relief = torch.sigmoid((closing_relief - 0.20) / 0.05)
            response = 1.0 - (1.0 - strong_brake) * (1.0 - mild_brake * relief)
            response_ttc, response_confidence = _first_event_value(ttc[1:], response)
        else:
            response_ttc = ego_future.new_zeros(())
            response_confidence = ego_future.new_zeros(())
        values = torch.stack([h_value, ttc_value, response_ttc])
        confidence = torch.stack(
            [
                ego_future.new_ones(()),
                ttc_confidence,
                response_confidence,
            ]
        )
        return values, confidence

    @staticmethod
    def _inverse_weighted_reference_cdf(
        values: torch.Tensor,
        weights: torch.Tensor,
        target_percentile: torch.Tensor,
    ) -> torch.Tensor:
        """Invert one local empirical CDF in canonical raw-axis coordinates."""

        valid = weights > 0.0
        if not bool(valid.any()):
            return target_percentile.detach()
        selected_values = values[valid]
        selected_weights = weights[valid]
        order = torch.argsort(selected_values)
        selected_values = selected_values[order]
        selected_weights = selected_weights[order]
        selected_weights = selected_weights / selected_weights.sum().clamp_min(1e-8)
        cumulative = torch.cumsum(selected_weights, dim=0)
        query = target_percentile.detach().clamp(0.0, 1.0)
        upper_index = torch.searchsorted(cumulative, query).clamp(
            max=selected_values.shape[0] - 1
        )
        upper_value = selected_values[upper_index]
        if int(upper_index.item()) == 0:
            return upper_value
        lower_index = upper_index - 1
        lower_value = selected_values[lower_index]
        lower_cdf = cumulative[lower_index]
        upper_cdf = cumulative[upper_index]
        mix = (query - lower_cdf) / (upper_cdf - lower_cdf).clamp_min(1e-8)
        return lower_value + mix.clamp(0.0, 1.0) * (upper_value - lower_value)

    def raw_axis_coordinates(
        self,
        model_output: torch.Tensor,
        prepared: Mapping[str, torch.Tensor],
        *,
        neighbor_reference_output: torch.Tensor | None = None,
        accel_opportunity_reference_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return unclamped canonical raw axes, confidence, and physical axes.

        Unlike bounded percentile diagnostics, the generated coordinates are not
        clamped and do not pass through a sigmoid CDF.  Gradients therefore
        remain available when a predicted trajectory lies outside the train
        reference support.
        """

        if model_output.ndim == 3:
            batch_size, agent_count, flat_dim = model_output.shape
            if flat_dim % 4 != 0:
                raise ValueError("Raw-axis objective expects flattened xy-cos-sin trajectories")
            model_output = model_output.reshape(batch_size, agent_count, -1, 4)
        if model_output.ndim != 4:
            raise ValueError(
                f"Raw-axis objective expects [B,P,T,4], got {tuple(model_output.shape)}"
            )

        batch_size = model_output.shape[0]
        enabled = prepared["enabled"].bool()
        if not bool(enabled.any()):
            canonical_batch = model_output.new_zeros((batch_size, 3))
            confidence_batch = model_output.new_zeros((batch_size, 3))
            raw_batch = model_output.new_zeros((batch_size, 3))
            return canonical_batch, confidence_batch, raw_batch

        future_normalized = model_output[:, :, 1:, :]
        future_physical = self.state_normalizer.inverse(future_normalized)
        neighbor_reference_physical = None
        if neighbor_reference_output is not None:
            if neighbor_reference_output.ndim == 3:
                reference_batch, reference_agents, reference_flat_dim = (
                    neighbor_reference_output.shape
                )
                if reference_flat_dim % 4 != 0:
                    raise ValueError(
                        "Neighbor reference expects flattened xy-cos-sin trajectories"
                    )
                neighbor_reference_output = neighbor_reference_output.reshape(
                    reference_batch,
                    reference_agents,
                    -1,
                    4,
                )
            if neighbor_reference_output.ndim != 4:
                raise ValueError(
                    "Neighbor reference expects [B,P,T,4], got "
                    f"{tuple(neighbor_reference_output.shape)}"
                )
            if neighbor_reference_output.shape[0] != batch_size:
                raise ValueError("Model output and neighbor reference batch sizes differ")
            neighbor_reference_physical = self.state_normalizer.inverse(
                neighbor_reference_output[:, :, 1:, :]
            ).detach()
        accel_opportunity_reference_physical = None
        if (
            self.free_drive_accel_support_mode == "normal_anchor"
            and accel_opportunity_reference_output is not None
        ):
            if accel_opportunity_reference_output.ndim == 3:
                reference_batch, reference_agents, reference_flat_dim = (
                    accel_opportunity_reference_output.shape
                )
                if reference_flat_dim % 4 != 0:
                    raise ValueError(
                        "Acceleration opportunity reference expects flattened "
                        "xy-cos-sin trajectories"
                    )
                accel_opportunity_reference_output = (
                    accel_opportunity_reference_output.reshape(
                        reference_batch,
                        reference_agents,
                        -1,
                        4,
                    )
                )
            if accel_opportunity_reference_output.ndim != 4:
                raise ValueError(
                    "Acceleration opportunity reference expects [B,P,T,4], got "
                    f"{tuple(accel_opportunity_reference_output.shape)}"
                )
            if accel_opportunity_reference_output.shape[0] != batch_size:
                raise ValueError(
                    "Model output and acceleration opportunity reference batch "
                    "sizes differ"
                )
            accel_opportunity_reference_physical = self.state_normalizer.inverse(
                accel_opportunity_reference_output[:, :, 1:, :]
            ).detach()
        canonical_rows = []
        confidence_rows = []
        raw_rows = []
        for batch_index in range(batch_size):
            if not bool(enabled[batch_index]):
                zero_row = model_output.new_zeros((3,))
                canonical_rows.append(zero_row)
                confidence_rows.append(zero_row)
                raw_rows.append(zero_row)
                continue
            scene = SCENE_ORDER[int(prepared["scene_index"][batch_index].item())]
            ego_future = future_physical[batch_index, 0]
            neighbor_future = (
                neighbor_reference_physical[batch_index, 1:]
                if neighbor_reference_physical is not None
                else future_physical[batch_index, 1:].detach()
            )
            ego_current = prepared["ego_current"][batch_index]
            neighbor_count = neighbor_future.shape[0]
            neighbor_current = prepared["neighbor_past"][
                batch_index, :neighbor_count, -1
            ]

            if scene == "straight_free_drive":
                raw_axis, metric_confidence = self._free_drive_axes(
                    ego_future,
                    ego_current,
                    prepared["speed_limit_mps"][batch_index].clamp_min(1.0),
                    accel_opportunity_reference_future=(
                        accel_opportunity_reference_physical[batch_index, 0]
                        if accel_opportunity_reference_physical is not None
                        else ego_future.detach()
                    ),
                )
            elif scene == "straight_car_follow":
                raw_axis, metric_confidence = self._car_follow_axes(
                    ego_future=ego_future,
                    ego_current=ego_current,
                    neighbor_future=neighbor_future,
                    neighbor_current=neighbor_current,
                    lead_index=int(prepared["lead_index"][batch_index].item()),
                )
            else:
                continue

            q_low = prepared["q_low"][batch_index]
            q_high = prepared["q_high"][batch_index]
            canonical = (raw_axis - q_low) / (q_high - q_low).clamp_min(1e-4)
            canonical = torch.where(
                prepared["falling"][batch_index],
                1.0 - canonical,
                canonical,
            )
            canonical_rows.append(canonical)
            # Confidence is a measurement-validity weight, not a control
            # target. Detaching it prevents the planner from reducing the loss
            # by learning to suppress its own opportunity/confidence score.
            confidence_rows.append(metric_confidence.detach())
            raw_rows.append(raw_axis)
        return (
            torch.stack(canonical_rows, dim=0),
            torch.stack(confidence_rows, dim=0),
            torch.stack(raw_rows, dim=0),
        )

    def raw_axis_loss_terms(
        self,
        model_output: torch.Tensor,
        inputs: Mapping[str, Any],
        prepared: Mapping[str, torch.Tensor] | None = None,
        *,
        beta: float = 0.08,
    ) -> Dict[str, torch.Tensor]:
        """Supervise generated raw axes against inverse conditional-CDF targets."""

        if prepared is None:
            style_condition = torch.as_tensor(inputs["style_value_condition"])
            prepared = self.prepare(inputs, style_condition)
        canonical, metric_confidence, raw_axis = self.raw_axis_coordinates(
            model_output,
            prepared,
        )
        zero = model_output.new_zeros(())
        enabled = prepared["enabled"].bool()
        target_rows = []
        valid_weight_rows = []
        sample_losses = []
        sample_mae = []

        for batch_index in range(canonical.shape[0]):
            if not bool(enabled[batch_index]):
                zero_row = canonical.detach().new_zeros((3,))
                target_rows.append(zero_row)
                valid_weight_rows.append(zero_row)
                continue
            axis_mask = prepared["axis_mask"][batch_index].float()
            axis_targets = []
            for axis in range(3):
                if bool(prepared["axis_mask"][batch_index, axis]):
                    axis_target = self._inverse_weighted_reference_cdf(
                        prepared["reference_values"][batch_index, axis],
                        prepared["reference_weights"][batch_index, axis],
                        prepared["target"][batch_index, axis],
                    )
                else:
                    axis_target = canonical.detach().new_zeros(())
                axis_targets.append(axis_target)
            target_row = torch.stack(axis_targets)
            weight = (axis_mask * metric_confidence[batch_index]).detach()
            target_rows.append(target_row)
            valid_weight_rows.append(weight)
            if not bool((weight > 1e-4).any()):
                continue
            error = F.smooth_l1_loss(
                canonical[batch_index],
                target_row,
                reduction="none",
                beta=max(float(beta), 1e-4),
            )
            absolute = (
                canonical[batch_index] - target_row
            ).abs()
            denominator = weight.sum().clamp_min(1.0)
            sample_losses.append((error * weight).sum() / denominator)
            sample_mae.append((absolute * weight).sum() / denominator)

        target_coordinate = torch.stack(target_rows, dim=0)
        valid_weight = torch.stack(valid_weight_rows, dim=0)

        return {
            "signed_raw_axis_loss": (
                torch.stack(sample_losses).mean() if sample_losses else zero
            ),
            "signed_raw_axis_mae": (
                torch.stack(sample_mae).mean() if sample_mae else zero
            ),
            "signed_raw_axis_active_ratio": enabled.float().mean(),
            "signed_raw_axis_coordinate": canonical,
            "signed_raw_axis_target": target_coordinate,
            "signed_raw_axis_valid_weight": valid_weight,
            "signed_generated_physical_axis": raw_axis,
        }

    def _inverse_reference_coordinates(
        self,
        prepared: Mapping[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Map percentile commands to frozen local canonical-axis coordinates."""

        rows = []
        axis_mask = prepared["axis_mask"].bool()
        for batch_index in range(target.shape[0]):
            axis_targets = []
            for axis in range(3):
                if bool(axis_mask[batch_index, axis]):
                    value = self._inverse_weighted_reference_cdf(
                        prepared["reference_values"][batch_index, axis],
                        prepared["reference_weights"][batch_index, axis],
                        target[batch_index, axis],
                    )
                else:
                    value = target.new_zeros(())
                axis_targets.append(value)
            rows.append(torch.stack(axis_targets))
        return torch.stack(rows, dim=0)

    def normal_relative_axis_loss_terms(
        self,
        model_output: torch.Tensor,
        normal_model_output: torch.Tensor,
        inputs: Mapping[str, Any],
        prepared: Mapping[str, torch.Tensor] | None = None,
        *,
        beta: float = 0.08,
        fixed_normal_neighbors: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Normal-referenced conditional quantile transport objective.

        The frozen reference defines a desired *displacement* from its local
        median, while the model displacement is measured from the exact
        semantic-normal/base trajectory produced from the same noisy state.
        This makes the preference target compatible with structural base
        preservation even when the pretrained planner's normal trajectory is
        not located at the empirical 0.5 percentile.
        """

        if prepared is None:
            style_condition = torch.as_tensor(inputs["style_value_condition"])
            prepared = self.prepare(inputs, style_condition)

        generated_coordinate, _, generated_raw = self.raw_axis_coordinates(
            model_output,
            prepared,
            neighbor_reference_output=(
                normal_model_output if fixed_normal_neighbors else None
            ),
            accel_opportunity_reference_output=normal_model_output,
        )
        normal_coordinate, normal_confidence, normal_raw = self.raw_axis_coordinates(
            normal_model_output,
            prepared,
            neighbor_reference_output=(
                normal_model_output if fixed_normal_neighbors else None
            ),
            accel_opportunity_reference_output=normal_model_output,
        )
        zero = model_output.new_zeros(())
        if not bool(prepared["enabled"].bool().any()):
            zero_axes = model_output.new_zeros((model_output.shape[0], 3))
            return {
                "normal_relative_axis_loss": zero,
                "normal_relative_axis_mae": zero,
                "normal_relative_axis_active_ratio": zero,
                "normal_relative_axis_active_count": zero,
                "normal_relative_generated_delta": zero_axes,
                "normal_relative_target_delta": zero_axes,
                "normal_relative_valid_weight": zero_axes,
                "normal_relative_generated_raw_axis": generated_raw,
                "normal_relative_anchor_raw_axis": normal_raw.detach(),
            }
        target_coordinate = self._inverse_reference_coordinates(
            prepared,
            prepared["target"],
        )
        neutral_target = torch.full_like(prepared["target"], 0.5)
        neutral_coordinate = self._inverse_reference_coordinates(
            prepared,
            neutral_target,
        )

        generated_delta = generated_coordinate - normal_coordinate.detach()
        target_delta = target_coordinate - neutral_coordinate
        # Fix opportunity/validity at the semantic-normal trajectory. Commanded
        # trajectories cannot lower their own supervision by changing metric
        # confidence.
        valid_weight = (
            prepared["axis_mask"].float() * normal_confidence.detach()
        )
        valid_sample = valid_weight.sum(dim=-1) > 1e-4
        if bool(valid_sample.any()):
            error = F.smooth_l1_loss(
                generated_delta,
                target_delta,
                reduction="none",
                beta=max(float(beta), 1e-4),
            )
            absolute = (generated_delta - target_delta).abs()
            sample_denominator = valid_weight.sum(dim=-1).clamp_min(1e-6)
            sample_loss = (error * valid_weight).sum(dim=-1) / sample_denominator
            sample_mae = (
                absolute * valid_weight
            ).sum(dim=-1) / sample_denominator
            loss = sample_loss[valid_sample].mean()
            mae = sample_mae[valid_sample].mean()
        else:
            loss = zero
            mae = zero

        return {
            "normal_relative_axis_loss": loss,
            "normal_relative_axis_mae": mae,
            "normal_relative_axis_active_ratio": valid_sample.float().mean(),
            "normal_relative_axis_active_count": zero.new_tensor(
                float(valid_sample.sum().item())
            ),
            "normal_relative_generated_delta": generated_delta,
            "normal_relative_target_delta": target_delta,
            "normal_relative_valid_weight": valid_weight,
            "normal_relative_generated_raw_axis": generated_raw,
            "normal_relative_anchor_raw_axis": normal_raw.detach(),
        }

    def axis_diagnostic_terms(
        self,
        model_output: torch.Tensor,
        inputs: Mapping[str, Any],
        prepared: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if model_output.ndim == 3:
            batch_size, agent_count, flat_dim = model_output.shape
            if flat_dim % 4 != 0:
                raise ValueError(
                    "Axis diagnostics expect flattened xy-cos-sin trajectories"
                )
            model_output = model_output.reshape(batch_size, agent_count, -1, 4)
        if model_output.ndim != 4:
            raise ValueError(
                "Axis diagnostics expect [B,P,T,4], got "
                f"{tuple(model_output.shape)}"
            )

        enabled = prepared["enabled"].bool()
        batch_size = model_output.shape[0]
        zero = model_output.new_zeros(())
        generated_percentile_debug = model_output.new_zeros((batch_size, 3))
        generated_raw_axis_debug = model_output.new_zeros((batch_size, 3))
        generated_canonical_axis_debug = model_output.new_zeros((batch_size, 3))
        generated_axis_valid_debug = torch.zeros(
            (batch_size, 3),
            dtype=torch.bool,
            device=model_output.device,
        )
        accel_opportunity_anchor_used_debug = torch.zeros(
            (batch_size,),
            dtype=torch.bool,
            device=model_output.device,
        )
        support_debug = self._support_debug_from_prepared(prepared)
        if not bool(enabled.any()):
            diagnostics = {
                **support_debug,
                "preference_command_strength": zero,
                "preference_axis_error": zero,
                "preference_axis_diagnostic_active_ratio": enabled.float().mean(),
                "preference_generated_axis_percentile": generated_percentile_debug,
                "preference_generated_raw_axis": generated_raw_axis_debug,
                "preference_generated_axis_canonical": generated_canonical_axis_debug,
                "preference_generated_axis_valid_mask": generated_axis_valid_debug,
                "preference_accel_opportunity_anchor_used": (
                    accel_opportunity_anchor_used_debug
                ),
                "preference_target_axis_percentile": prepared["target"].detach(),
            }
            self._last_diagnostics = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in diagnostics.items()
            }
            return diagnostics

        future_normalized = model_output[:, :, 1:, :]
        future_physical = self.state_normalizer.inverse(future_normalized)
        fixed_neighbor_future = _as_batch_tensor(
            inputs,
            "preference_neighbor_reference_future",
            device=model_output.device,
            dtype=model_output.dtype,
        )
        if fixed_neighbor_future is not None:
            if fixed_neighbor_future.ndim != 4:
                raise ValueError(
                    "preference_neighbor_reference_future must have shape "
                    "[B,N,T,4]"
                )
            if fixed_neighbor_future.shape[0] != batch_size:
                raise ValueError(
                    "preference_neighbor_reference_future batch size mismatch"
                )
            fixed_neighbor_future = fixed_neighbor_future.detach()
        fixed_ego_future = _as_batch_tensor(
            inputs,
            "preference_ego_reference_future",
            device=model_output.device,
            dtype=model_output.dtype,
        )
        if fixed_ego_future is not None:
            if fixed_ego_future.ndim != 3 or fixed_ego_future.shape[-1] < 2:
                raise ValueError(
                    "preference_ego_reference_future must have shape [B,T,>=2]"
                )
            if fixed_ego_future.shape[0] != batch_size:
                raise ValueError(
                    "preference_ego_reference_future batch size mismatch"
                )
            fixed_ego_future = fixed_ego_future.detach()
        strength_per_sample = []
        axis_error_values = []

        for batch_index in range(batch_size):
            if not bool(enabled[batch_index]):
                strength_per_sample.append(zero)
                continue
            scene = SCENE_ORDER[int(prepared["scene_index"][batch_index].item())]
            ego_future = future_physical[batch_index, 0]
            neighbor_future = (
                fixed_neighbor_future[batch_index]
                if fixed_neighbor_future is not None
                else future_physical[batch_index, 1:].detach()
            )
            ego_current = prepared["ego_current"][batch_index]
            neighbor_count = neighbor_future.shape[0]
            neighbor_current = prepared["neighbor_past"][
                batch_index, :neighbor_count, -1
            ]
            speed_limit = prepared["speed_limit_mps"][batch_index].clamp_min(1.0)

            if scene == "straight_free_drive":
                raw_axis, metric_confidence = self._free_drive_axes(
                    ego_future,
                    ego_current,
                    speed_limit,
                    accel_opportunity_reference_future=(
                        fixed_ego_future[batch_index]
                        if fixed_ego_future is not None
                        else ego_future.detach()
                    ),
                )
                accel_opportunity_anchor_used_debug[batch_index] = bool(
                    self.free_drive_accel_support_mode == "normal_anchor"
                    and fixed_ego_future is not None
                )
            elif scene == "straight_car_follow":
                raw_axis, metric_confidence = self._car_follow_axes(
                    ego_future=ego_future,
                    ego_current=ego_current,
                    neighbor_future=neighbor_future,
                    neighbor_current=neighbor_current,
                    lead_index=int(prepared["lead_index"][batch_index].item()),
                )
            else:
                raw_axis = ego_future.new_zeros((3,))
                metric_confidence = ego_future.new_zeros((3,))

            q_low = prepared["q_low"][batch_index]
            q_high = prepared["q_high"][batch_index]
            canonical_unclamped = (raw_axis - q_low) / (
                q_high - q_low
            ).clamp_min(1e-4)
            canonical_unclamped = torch.where(
                prepared["falling"][batch_index],
                1.0 - canonical_unclamped,
                canonical_unclamped,
            )
            canonical = canonical_unclamped.clamp(0.0, 1.0)
            reference = prepared["reference_values"][batch_index]
            reference_weight = prepared["reference_weights"][batch_index]
            percentile = (
                reference_weight
                * torch.sigmoid(
                    (canonical.unsqueeze(-1) - reference) / self.cdf_temperature
                )
            ).sum(dim=-1)

            axis_mask = prepared["axis_mask"][batch_index].float() * metric_confidence
            target = prepared["target"][batch_index]
            generated_percentile_debug[batch_index] = percentile.detach()
            generated_raw_axis_debug[batch_index] = raw_axis.detach()
            generated_canonical_axis_debug[batch_index] = canonical_unclamped.detach()
            generated_axis_valid_debug[batch_index] = axis_mask.detach() > 1e-4
            axis_error = F.smooth_l1_loss(percentile, target, reduction="none", beta=0.08)
            strength = (
                ((target - 0.5).abs() / 0.25).clamp(0.0, 1.0) * prepared["axis_mask"][batch_index].float()
            ).sum() / prepared["axis_mask"][batch_index].float().sum().clamp_min(1.0)
            strength_per_sample.append(strength)
            axis_error_values.append((axis_error * axis_mask).sum() / axis_mask.sum().clamp_min(1.0))

        strength_tensor = torch.stack(strength_per_sample)
        active_float = enabled.float()
        denominator = active_float.sum().clamp_min(1.0)
        diagnostics = {
            **support_debug,
            "preference_command_strength": (strength_tensor * active_float).sum() / denominator,
            "preference_axis_diagnostic_active_ratio": active_float.mean(),
            "preference_axis_error": (
                torch.stack(axis_error_values).mean() if axis_error_values else zero
            ),
            "preference_generated_axis_percentile": generated_percentile_debug,
            "preference_generated_raw_axis": generated_raw_axis_debug,
            "preference_generated_axis_canonical": generated_canonical_axis_debug,
            "preference_generated_axis_valid_mask": generated_axis_valid_debug,
            "preference_accel_opportunity_anchor_used": (
                accel_opportunity_anchor_used_debug
            ),
            "preference_target_axis_percentile": prepared["target"].detach(),
        }
        self._last_diagnostics = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in diagnostics.items()
        }
        return diagnostics

    def pop_last_diagnostics(self) -> Dict[str, torch.Tensor] | None:
        diagnostics = self._last_diagnostics
        self._last_diagnostics = None
        return diagnostics

    def reference_summary(self) -> Dict[str, Dict[str, object]]:
        return {
            scene: {
                "reference_count": int(payload["condition_values"].shape[0]),
                "condition_dim": int(payload["condition_values"].shape[1]),
                "axis_valid_counts": [
                    int(np.sum(payload["canonical_axis_valid_mask"][:, axis]))
                    for axis in range(3)
                ],
            }
            for scene, payload in self._references.items()
        }

    def reference_query_selftest(self) -> Dict[str, Dict[str, object]]:
        """Exercise the cached KNN path with a representative causal query."""

        report: Dict[str, Dict[str, object]] = {}
        for scene, payload in self._references.items():
            condition = payload["condition_values"]
            condition_valid = payload["condition_valid_mask"]
            valid_rate = condition_valid.mean(axis=0)
            shared_dims = np.argsort(-valid_rate)[: self.min_shared_condition_features]
            eligible = np.all(condition_valid[:, shared_dims], axis=1)
            if not np.any(eligible):
                report[scene] = {
                    "shared_dims": shared_dims.tolist(),
                    "eligible_count": 0,
                    "valid_axes": [False, False, False],
                    "weight_sums": [0.0, 0.0, 0.0],
                }
                continue
            query = np.zeros((condition.shape[1],), dtype=np.float32)
            query[shared_dims] = np.median(
                condition[eligible][:, shared_dims],
                axis=0,
            )
            query_valid = np.zeros((condition.shape[1],), dtype=bool)
            query_valid[shared_dims] = True
            _, weights, valid_axes = self._local_references(
                scene,
                query,
                query_valid,
            )
            report[scene] = {
                "shared_dims": shared_dims.tolist(),
                "eligible_count": int(np.sum(eligible)),
                "valid_axes": valid_axes.tolist(),
                "weight_sums": weights.sum(axis=-1).tolist(),
            }
        return report

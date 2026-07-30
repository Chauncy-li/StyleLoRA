"""Identity-anchored, differentiable behavior semantics for Step 5.

This module deliberately keeps interaction identity outside Preference Flow: a
neutral trajectory and observable neighbor history choose one persistent lead
ID and a detached validity mask.  Step 5-S2 adds a neutral soft-attention
measurement whose weights are likewise fixed before every rho behavior loss.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch


DEFAULT_DT = 0.1
DEFAULT_SAME_LANE_HALF_WIDTH_M = 1.9
DEFAULT_EGO_LENGTH_M = 5.2
DEFAULT_NEIGHBOR_LENGTH_M = 4.5
DEFAULT_TIME_GAP_VMIN_MPS = 1.5
DEFAULT_CLOSING_SPEED_MIN_MPS = 0.25
DEFAULT_TTC_MAX_S = 10.0

_AXIS_INDEX = {
    "speed_utilization": ("straight_free_drive", 0),
    "headway_tightness_from_h": ("straight_car_follow", 0),
    "ttc_tightness": ("straight_car_follow", 1),
}


class BehaviorAlignmentError(ValueError):
    """Raised when a Step-5-S1 behavior contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BehaviorAlignmentError(message)


def _finite(name: str, value: torch.Tensor) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all().item()), f"{name} must be finite")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = mask.to(dtype=values.dtype)
    count = weights.sum(dim=-1)
    return (values * weights).sum(dim=-1) / count.clamp_min(1.0), count > 0


def _masked_quantile(values: torch.Tensor, mask: torch.Tensor, quantile: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hard offline aggregation only; never used by the training loss."""

    rows = []
    valid_rows = []
    for row, row_mask in zip(values.detach(), mask.detach()):
        selected = row[row_mask]
        valid_rows.append(bool(selected.numel() > 0))
        rows.append(
            torch.quantile(selected.float(), float(quantile))
            if selected.numel() > 0
            else row.new_zeros(())
        )
    return torch.stack(rows).to(device=values.device, dtype=values.dtype), torch.tensor(
        valid_rows, device=values.device, dtype=torch.bool
    )


def trajectory_tangents(current_state: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    """Return one unit neutral-path tangent per future step."""

    _require(current_state.ndim == 2 and current_state.shape[-1] >= 4, "current_state must be [B, >=4]")
    _require(future.ndim == 3 and future.shape[-1] >= 2, "future must be [B, T, >=2]")
    _require(current_state.shape[0] == future.shape[0], "current/future batch mismatch")
    positions = torch.cat((current_state[:, None, :2], future[..., :2]), dim=1)
    deltas = positions[:, 1:] - positions[:, :-1]
    heading = current_state[:, None, 2:4].expand_as(deltas)
    fallback = torch.where(
        torch.linalg.vector_norm(heading, dim=-1, keepdim=True) > 1e-4,
        heading,
        torch.tensor([1.0, 0.0], device=future.device, dtype=future.dtype).view(1, 1, 2),
    )
    tangent = torch.where(
        torch.linalg.vector_norm(deltas, dim=-1, keepdim=True) > 1e-4,
        deltas,
        fallback,
    )
    return tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-6)


def longitudinal_perturbation(
    future: torch.Tensor, current_state: torch.Tensor, reference_future: torch.Tensor, meters: float
) -> torch.Tensor:
    """Apply a reporting-only longitudinal XY shift along the neutral tangent."""

    tangent = trajectory_tangents(current_state, reference_future)
    edited = future.clone()
    edited[..., :2] = edited[..., :2] + float(meters) * tangent
    return edited


@dataclass(frozen=True)
class LeadReference:
    """Detached neutral interaction identity shared by every rho branch."""

    lead_indices: torch.Tensor  # [B], -1 if unavailable
    has_lead: torch.Tensor  # [B]
    valid_time_mask: torch.Tensor  # [B, T]
    lead_current_xy: torch.Tensor  # [B, 2]
    lead_velocity_xy: torch.Tensor  # [B, 2]
    lead_length_m: torch.Tensor  # [B]
    initial_effective_gap_m: torch.Tensor  # [B]


class NeutralAnchoredLeadReference:
    """Choose one persistent lead from neutral path plus observable history.

    The selector never reads neighbor futures.  It chooses an ID once at the
    current frame and extrapolates its observed velocity only to derive a
    detached neutral validity mask.  Ground-truth neighbor futures are looked
    up later by the loss bridge for that already-fixed ID.
    """

    def __init__(
        self,
        *,
        dt: float = DEFAULT_DT,
        same_lane_half_width_m: float = DEFAULT_SAME_LANE_HALF_WIDTH_M,
        ego_length_m: float = DEFAULT_EGO_LENGTH_M,
        default_neighbor_length_m: float = DEFAULT_NEIGHBOR_LENGTH_M,
    ) -> None:
        self.dt = float(dt)
        self.same_lane_half_width_m = float(same_lane_half_width_m)
        self.ego_length_m = float(ego_length_m)
        self.default_neighbor_length_m = float(default_neighbor_length_m)
        _require(self.dt > 0.0, "dt must be positive")

    def _last_observed(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = torch.any(history[..., :6].abs() > 1e-6, dim=-1)
        steps = torch.arange(history.shape[2], device=history.device).view(1, 1, -1)
        last_index = torch.where(valid, steps, torch.full_like(steps, -1)).amax(dim=-1)
        has = last_index >= 0
        safe_index = last_index.clamp_min(0)
        gather = safe_index[..., None, None].expand(-1, -1, 1, history.shape[-1])
        current = history.gather(2, gather).squeeze(2)
        previous_index = (safe_index - 1).clamp_min(0)
        previous = history.gather(
            2, previous_index[..., None, None].expand(-1, -1, 1, history.shape[-1])
        ).squeeze(2)
        finite_velocity = torch.isfinite(current[..., 4:6]).all(dim=-1)
        observed_velocity = current[..., 4:6]
        fallback_velocity = (current[..., :2] - previous[..., :2]) / self.dt
        velocity = torch.where(
            finite_velocity[..., None], observed_velocity, fallback_velocity
        )
        velocity = torch.where(has[..., None], velocity, torch.zeros_like(velocity))
        return current, velocity, has

    def build(
        self,
        *,
        neutral_future: torch.Tensor,
        ego_current_state: torch.Tensor,
        neighbor_history: torch.Tensor,
    ) -> LeadReference:
        _require(neutral_future.ndim == 3 and neutral_future.shape[-1] >= 2, "neutral_future must be [B, T, >=2]")
        _require(ego_current_state.ndim == 2 and ego_current_state.shape[-1] >= 4, "ego_current_state must be [B, >=4]")
        _require(neighbor_history.ndim == 4 and neighbor_history.shape[-1] >= 6, "neighbor_history must be [B, N, H, >=6]")
        _require(neutral_future.shape[0] == ego_current_state.shape[0] == neighbor_history.shape[0], "lead reference batch mismatch")
        _finite("neutral_future", neutral_future)
        _finite("ego_current_state", ego_current_state)
        _finite("neighbor_history", neighbor_history)
        with torch.no_grad():
            neutral = neutral_future.detach()
            ego = ego_current_state.detach()
            history = neighbor_history.detach()
            current, velocity, observed = self._last_observed(history)
            initial_tangent = trajectory_tangents(ego, neutral)[:, 0]
            rel = current[..., :2] - ego[:, None, :2]
            longitudinal = (rel * initial_tangent[:, None]).sum(dim=-1)
            lateral = rel[..., 0] * initial_tangent[:, None, 1] - rel[..., 1] * initial_tangent[:, None, 0]
            # The cached neighbor layout is
            # [x, y, cos, sin, vx, vy, width, length].  Keep this independent
            # from the model's normalized representation.
            if history.shape[-1] > 7:
                lengths = history[..., 7].abs().amax(dim=2)
                lengths = torch.where(lengths > 1e-3, lengths, torch.full_like(lengths, self.default_neighbor_length_m))
            else:
                lengths = torch.full_like(longitudinal, self.default_neighbor_length_m)
            gap = longitudinal - 0.5 * (self.ego_length_m + lengths)
            candidates = observed & (longitudinal > 0.0) & (lateral.abs() < self.same_lane_half_width_m)
            selected = torch.where(candidates, gap, torch.full_like(gap, float("inf"))).argmin(dim=1)
            has = candidates.any(dim=1)
            safe = selected.clamp_min(0)
            batch = torch.arange(neutral.shape[0], device=neutral.device)
            lead_xy = current[batch, safe, :2]
            lead_velocity = velocity[batch, safe]
            lead_length = lengths[batch, safe]
            initial_gap = gap[batch, safe]
            time = torch.arange(1, neutral.shape[1] + 1, device=neutral.device, dtype=neutral.dtype).view(1, -1, 1) * self.dt
            extrapolated_xy = lead_xy[:, None, :] + lead_velocity[:, None, :] * time
            tangent = trajectory_tangents(ego, neutral)
            future_rel = extrapolated_xy - neutral[..., :2]
            future_longitudinal = (future_rel * tangent).sum(dim=-1)
            future_lateral = future_rel[..., 0] * tangent[..., 1] - future_rel[..., 1] * tangent[..., 0]
            valid = has[:, None] & (future_longitudinal > 0.0) & (future_lateral.abs() < self.same_lane_half_width_m)
            return LeadReference(
                lead_indices=torch.where(has, safe, torch.full_like(safe, -1)).detach(),
                has_lead=has.detach(),
                valid_time_mask=valid.detach(),
                lead_current_xy=lead_xy.detach(),
                lead_velocity_xy=lead_velocity.detach(),
                lead_length_m=lead_length.detach(),
                initial_effective_gap_m=torch.where(has, initial_gap, torch.zeros_like(initial_gap)).detach(),
            )


@dataclass(frozen=True)
class FrozenAxisCalibration:
    q_low: float
    q_high: float
    direction: str

    @property
    def span(self) -> float:
        return self.q_high - self.q_low


class FrozenBehaviorCalibration:
    """Frozen V5 ranges used to express target movement in canonical units."""

    def __init__(self, axes: Mapping[Tuple[str, int], FrozenAxisCalibration], source: str) -> None:
        self._axes = dict(axes)
        self.source = str(source)
        for key in _AXIS_INDEX.values():
            _require(key in self._axes, f"calibration misses {key}")

    @classmethod
    def from_json(cls, path: str) -> "FrozenBehaviorCalibration":
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        _require(str(payload.get("artifact", "")) == "normalization_model", "not a V5 normalization model")
        scenes = payload.get("scenes", {})
        _require(isinstance(scenes, Mapping), "normalization scenes must be an object")
        axes: Dict[Tuple[str, int], FrozenAxisCalibration] = {}
        for scene in {scene for scene, _ in _AXIS_INDEX.values()}:
            scene_payload = scenes.get(scene, {})
            stats = scene_payload.get("axis_stats", []) if isinstance(scene_payload, Mapping) else []
            _require(isinstance(stats, Sequence) and len(stats) >= 2, f"normalization misses {scene} axis stats")
            for index, raw in enumerate(stats):
                if not isinstance(raw, Mapping):
                    continue
                low, high = float(raw.get("q_low", float("nan"))), float(raw.get("q_high", float("nan")))
                direction = str(raw.get("direction", ""))
                _require(torch.isfinite(torch.tensor([low, high])).all().item() and high > low, f"invalid {scene} axis {index} range")
                _require(direction in {"rising", "falling"}, f"invalid {scene} axis {index} direction")
                axes[(scene, index)] = FrozenAxisCalibration(low, high, direction)
        return cls(axes, str(source))

    def canonical(self, axis_name: str, raw_value: torch.Tensor) -> torch.Tensor:
        scene, index = _AXIS_INDEX[axis_name]
        axis = self._axes[(scene, index)]
        normalized = (raw_value - axis.q_low) / axis.span
        canonical = normalized if axis.direction == "rising" else 1.0 - normalized
        return canonical.clamp(0.0, 1.0)

    def desired(
        self,
        neutral_canonical: torch.Tensor,
        rho: torch.Tensor,
        *,
        positive_span: float,
        negative_span: float,
    ) -> torch.Tensor:
        _require(positive_span > 0.0 and negative_span > 0.0, "alignment spans must be positive")
        span = torch.where(rho >= 0.0, torch.full_like(rho, positive_span), torch.full_like(rho, negative_span))
        return (neutral_canonical + rho * span).clamp(0.0, 1.0)

    def summary(self) -> Dict[str, Dict[str, float | str]]:
        return {
            f"{scene}:{index}": {"q_low": axis.q_low, "q_high": axis.q_high, "direction": axis.direction}
            for (scene, index), axis in self._axes.items()
        }


@dataclass(frozen=True)
class BehaviorMeasurement:
    speed: torch.Tensor
    speed_valid: torch.Tensor
    headway: torch.Tensor
    headway_valid: torch.Tensor
    ttc: torch.Tensor
    ttc_valid: torch.Tensor
    score: torch.Tensor
    score_valid: torch.Tensor
    offline_score: torch.Tensor
    offline_score_valid: torch.Tensor
    raw_speed: torch.Tensor
    raw_headway: torch.Tensor
    raw_ttc: torch.Tensor


@dataclass(frozen=True)
class AttentionBehaviorMeasurement:
    """Behavior proxy measured against fixed neutral-attention relations.

    ``attention_coverage`` is intentionally a detached audit value.  The
    attention map is computed from the neutral branch once and is never a
    differentiable escape route for any rho branch's behavior loss.
    """

    speed: torch.Tensor
    speed_valid: torch.Tensor
    headway: torch.Tensor
    headway_valid: torch.Tensor
    ttc: torch.Tensor
    ttc_valid: torch.Tensor
    score: torch.Tensor
    score_valid: torch.Tensor
    raw_speed: torch.Tensor
    raw_headway: torch.Tensor
    raw_ttc: torch.Tensor
    attention_coverage: torch.Tensor
    interaction_confidence: torch.Tensor


@dataclass(frozen=True)
class FeasiblePathwiseLoss:
    """Adjacent-rho feasibility-aware ordering tensors for one grid."""

    loss: torch.Tensor
    deltas: torch.Tensor
    target_deltas: torch.Tensor
    active_mask: torch.Tensor
    saturated_mask: torch.Tensor


@dataclass(frozen=True)
class ExistingFormalMetricResult:
    """One non-differentiable score from the unmodified formal metric code."""

    score: float
    valid: bool
    raw_values: Tuple[float, float, float]
    axis_valid_mask: Tuple[bool, bool, bool]
    metric_source: str
    failure: Optional[str]


def _scalar_canonical(calibration: "FrozenBehaviorCalibration", axis_name: str, raw_value: float) -> float:
    scene, index = _AXIS_INDEX[axis_name]
    axis = calibration._axes[(scene, index)]
    normalized = (float(raw_value) - axis.q_low) / axis.span
    canonical = normalized if axis.direction == "rising" else 1.0 - normalized
    return min(max(canonical, 0.0), 1.0)


def existing_formal_metric_for_trajectory(
    *,
    cache_path: str,
    temporary_cache_path: str,
    scene: str,
    ego_future: torch.Tensor,
    calibration: "FrozenBehaviorCalibration",
) -> ExistingFormalMetricResult:
    """Evaluate an edited ego future with the existing hard metric implementation.

    ``research_v1.stylization.metrics`` intentionally accepts cache files, not
    model tensors.  To keep that production metric unchanged, this function
    writes a short-lived cache clone with only the ego future replaced.  It is
    used after training for the rho-sweep audit only; it is never on the
    differentiable path and never exposes future data to the Flow.
    """

    try:
        import numpy as np

        from research_v1.stylization.metrics import build_behavior_metric_bundle

        source = Path(cache_path)
        target = Path(temporary_cache_path)
        if scene not in {"straight_free_drive", "straight_car_follow"}:
            raise BehaviorAlignmentError(f"unsupported formal scene: {scene}")
        if not source.is_file():
            raise FileNotFoundError(source)
        xy = ego_future.detach().cpu().to(torch.float32).numpy()
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise BehaviorAlignmentError("formal ego_future must be [T, >=2]")
        keys = (
            "neighbor_agents_past", "neighbor_agents_future",
            "neighbor_agents_past_mask", "neighbor_agents_future_mask",
            "route_lanes", "route_lanes_mask", "route_lanes_speed_limit",
            "route_lanes_has_speed_limit", "lanes", "lanes_mask",
            "lanes_speed_limit", "lanes_has_speed_limit",
        )
        with np.load(source, allow_pickle=False) as cached:
            payload = {key: np.asarray(cached[key]) for key in keys if key in cached.files}
        payload["ego_agent_future"] = np.asarray(xy, dtype=np.float32)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, **payload)
        bundle = build_behavior_metric_bundle({"scene_bucket": scene, "cache_path": str(target)})
        raw = tuple(float(value) for value in bundle.raw_metric_values.tolist())
        valid = tuple(bool(value) for value in bundle.axis_valid_mask.tolist())
        if len(raw) != 3 or len(valid) != 3 or bundle.metric_source != "cache":
            raise BehaviorAlignmentError(f"formal metric unavailable: {bundle.metric_source}")
        if scene == "straight_free_drive":
            return ExistingFormalMetricResult(
                score=_scalar_canonical(calibration, "speed_utilization", raw[0]),
                valid=valid[0], raw_values=raw, axis_valid_mask=valid,
                metric_source=str(bundle.metric_source), failure=None,
            )
        headway = _scalar_canonical(calibration, "headway_tightness_from_h", raw[0])
        ttc = _scalar_canonical(calibration, "ttc_tightness", raw[1])
        return ExistingFormalMetricResult(
            score=(headway + (ttc if valid[1] else 0.0)) / (2.0 if valid[1] else 1.0),
            valid=valid[0], raw_values=raw, axis_valid_mask=valid,
            metric_source=str(bundle.metric_source), failure=None,
        )
    except Exception as error:  # Keep the audit complete and make failure explicit.
        return ExistingFormalMetricResult(
            score=0.0, valid=False, raw_values=(0.0, 0.0, 0.0),
            axis_valid_mask=(False, False, False), metric_source="error",
            failure=f"{type(error).__name__}: {error}",
        )


def _combine_scores(
    *,
    speed: torch.Tensor,
    speed_valid: torch.Tensor,
    headway: torch.Tensor,
    headway_valid: torch.Tensor,
    ttc: torch.Tensor,
    ttc_valid: torch.Tensor,
    free_mask: torch.Tensor,
    car_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    car_weight = 1.0 + ttc_valid.to(dtype=speed.dtype)
    car_score = (headway + ttc * ttc_valid.to(dtype=speed.dtype)) / car_weight
    score = torch.where(free_mask, speed, torch.where(car_mask, car_score, torch.zeros_like(speed)))
    valid = torch.where(free_mask, speed_valid, torch.where(car_mask, headway_valid, torch.zeros_like(headway_valid)))
    return score, valid


class DifferentiableBehaviorBridge:
    """Torch proxy with an offline hard aggregation matching metric semantics."""

    def __init__(
        self,
        calibration: FrozenBehaviorCalibration,
        *,
        dt: float = DEFAULT_DT,
        ego_length_m: float = DEFAULT_EGO_LENGTH_M,
        min_speed_mps: float = DEFAULT_TIME_GAP_VMIN_MPS,
        closing_speed_min_mps: float = DEFAULT_CLOSING_SPEED_MIN_MPS,
        ttc_cap_s: float = DEFAULT_TTC_MAX_S,
    ) -> None:
        self.calibration = calibration
        self.dt = float(dt)
        self.ego_length_m = float(ego_length_m)
        self.min_speed_mps = float(min_speed_mps)
        self.closing_speed_min_mps = float(closing_speed_min_mps)
        self.ttc_cap_s = float(ttc_cap_s)

    @staticmethod
    def speed_limit_from_inputs(
        route_limits: torch.Tensor,
        route_has_limits: torch.Tensor,
        lane_limits: torch.Tensor,
        lane_has_limits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use a detached route-first scalar speed reference per sample."""

        def _reduce(limits: torch.Tensor, has_limits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            values = limits.detach().reshape(limits.shape[0], -1)
            flags = has_limits.detach()
            if flags.numel() != limits.numel():
                flags = flags.unsqueeze(-1).expand_as(limits)
            available = flags.reshape(flags.shape[0], -1).bool() & (values > 0.5)
            mean, valid = _masked_mean(values, available)
            return mean.detach(), valid.detach()

        route_value, route_valid = _reduce(route_limits, route_has_limits)
        lane_value, lane_valid = _reduce(lane_limits, lane_has_limits)
        return torch.where(route_valid, route_value, lane_value), route_valid | lane_valid

    @staticmethod
    def _gather_lead(
        neighbor_future: torch.Tensor, neighbor_mask: torch.Tensor, reference: LeadReference
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, neighbors, steps, _ = neighbor_future.shape
        _require(neighbors > 0, "neighbor_future must retain at least one predicted neighbor")
        safe_index = reference.lead_indices.clamp(min=0, max=neighbors - 1)
        gather = safe_index[:, None, None, None].expand(batch_size, 1, steps, neighbor_future.shape[-1])
        lead = neighbor_future.gather(1, gather).squeeze(1)
        invalid = neighbor_mask.gather(1, safe_index[:, None, None].expand(batch_size, 1, steps)).squeeze(1)
        valid = reference.has_lead[:, None] & reference.valid_time_mask & ~invalid
        return lead, valid.detach()

    def measure(
        self,
        *,
        edited_future: torch.Tensor,
        neutral_future: torch.Tensor,
        ego_current_state: torch.Tensor,
        neighbor_future: torch.Tensor,
        neighbor_mask: torch.Tensor,
        lead_reference: LeadReference,
        speed_limit_mps: torch.Tensor,
        speed_limit_valid: torch.Tensor,
        free_mask: torch.Tensor,
        car_mask: torch.Tensor,
        base_tangent: Optional[torch.Tensor] = None,
    ) -> BehaviorMeasurement:
        _require(edited_future.shape == neutral_future.shape and edited_future.ndim == 3, "edited/neutral futures must match [B,T,4]")
        _require(neighbor_future.ndim == 4 and neighbor_mask.shape == neighbor_future.shape[:3], "neighbor future/mask mismatch")
        _require(edited_future.shape[0] == neighbor_future.shape[0], "behavior batch mismatch")
        _finite("edited_future", edited_future)
        lead_future, lead_valid = self._gather_lead(neighbor_future, neighbor_mask, lead_reference)
        if base_tangent is None:
            tangent = trajectory_tangents(ego_current_state, neutral_future).detach()
        else:
            _require(
                tuple(base_tangent.shape) == (edited_future.shape[0], edited_future.shape[1], 2),
                "base_tangent must be [B,T,2]",
            )
            _finite("base_tangent", base_tangent)
            tangent = base_tangent.detach()
        edited_full = torch.cat((ego_current_state[:, None, :2], edited_future[..., :2]), dim=1)
        neutral_full = torch.cat((ego_current_state[:, None, :2], neutral_future[..., :2]), dim=1)
        edited_velocity = (edited_full[:, 1:] - edited_full[:, :-1]) / self.dt
        neutral_velocity = (neutral_full[:, 1:] - neutral_full[:, :-1]) / self.dt
        speed = torch.linalg.vector_norm(edited_velocity, dim=-1)
        ego_long = (edited_velocity * tangent).sum(dim=-1)
        neutral_long = (neutral_velocity * tangent).sum(dim=-1)
        lead_full = torch.cat((lead_reference.lead_current_xy[:, None, :], lead_future[..., :2]), dim=1)
        lead_velocity = (lead_full[:, 1:] - lead_full[:, :-1]) / self.dt
        lead_long = (lead_velocity * tangent).sum(dim=-1)

        relative = lead_future[..., :2] - edited_future[..., :2]
        neutral_relative = lead_future[..., :2] - neutral_future[..., :2]
        gap = ((relative * tangent).sum(dim=-1) - 0.5 * (self.ego_length_m + lead_reference.lead_length_m[:, None])).clamp_min(0.0)
        neutral_gap = ((neutral_relative * tangent).sum(dim=-1) - 0.5 * (self.ego_length_m + lead_reference.lead_length_m[:, None])).clamp_min(0.0)
        thw = gap / ego_long.abs().clamp_min(self.min_speed_mps)
        headway_raw, headway_valid = _masked_mean(thw, lead_valid)
        offline_headway_raw, offline_headway_valid = _masked_quantile(thw, lead_valid, 0.10)

        neutral_closing = neutral_long - lead_long
        ttc_valid_mask = (lead_valid & (neutral_closing >= self.closing_speed_min_mps)).detach()
        closing = ego_long - lead_long
        ttc = (gap / closing.clamp_min(1e-3)).clamp(max=self.ttc_cap_s)
        ttc_raw, ttc_valid = _masked_mean(ttc, ttc_valid_mask)
        offline_ttc_raw, offline_ttc_valid = _masked_quantile(ttc, ttc_valid_mask, 0.10)

        ratio = speed / speed_limit_mps[:, None].clamp_min(1e-3)
        speed_mask = speed_limit_valid[:, None].expand_as(ratio)
        speed_raw, speed_valid = _masked_mean(ratio, speed_mask)
        offline_speed_raw, offline_speed_valid = _masked_quantile(ratio, speed_mask, 0.50)

        speed_canonical = self.calibration.canonical("speed_utilization", speed_raw)
        headway_canonical = self.calibration.canonical("headway_tightness_from_h", headway_raw)
        ttc_canonical = self.calibration.canonical("ttc_tightness", ttc_raw)
        offline_speed = self.calibration.canonical("speed_utilization", offline_speed_raw)
        offline_headway = self.calibration.canonical("headway_tightness_from_h", offline_headway_raw)
        offline_ttc = self.calibration.canonical("ttc_tightness", offline_ttc_raw)
        score, score_valid = _combine_scores(
            speed=speed_canonical,
            speed_valid=speed_valid,
            headway=headway_canonical,
            headway_valid=headway_valid,
            ttc=ttc_canonical,
            ttc_valid=ttc_valid,
            free_mask=free_mask,
            car_mask=car_mask,
        )
        offline_score, offline_score_valid = _combine_scores(
            speed=offline_speed,
            speed_valid=offline_speed_valid,
            headway=offline_headway,
            headway_valid=offline_headway_valid,
            ttc=offline_ttc,
            ttc_valid=offline_ttc_valid,
            free_mask=free_mask,
            car_mask=car_mask,
        )
        return BehaviorMeasurement(
            speed=speed_canonical,
            speed_valid=speed_valid,
            headway=headway_canonical,
            headway_valid=headway_valid,
            ttc=ttc_canonical,
            ttc_valid=ttc_valid,
            score=score,
            score_valid=score_valid,
            offline_score=offline_score.detach(),
            offline_score_valid=offline_score_valid.detach(),
            raw_speed=speed_raw,
            raw_headway=headway_raw,
            raw_ttc=ttc_raw,
        )

    def measure_neutral_attention(
        self,
        *,
        edited_future: torch.Tensor,
        neutral_future: torch.Tensor,
        ego_current_state: torch.Tensor,
        neighbor_future: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_current_state: torch.Tensor,
        agent_valid_mask: torch.Tensor,
        neutral_attention_weights: torch.Tensor,
        interaction_confidence: torch.Tensor,
        base_tangent: torch.Tensor,
        speed_limit_mps: torch.Tensor,
        speed_limit_valid: torch.Tensor,
        free_mask: torch.Tensor,
        car_mask: torch.Tensor,
    ) -> AttentionBehaviorMeasurement:
        """Measure with detached neutral weights over all available neighbors.

        Neighbor futures are supervision-only inputs.  Neither the attention
        module nor the Preference Flow inference condition receives them.
        """

        _require(
            edited_future.shape == neutral_future.shape and edited_future.ndim == 3,
            "edited/neutral futures must match [B,T,4]",
        )
        _require(
            neighbor_future.ndim == 4 and neighbor_mask.shape == neighbor_future.shape[:3],
            "neighbor future/mask mismatch",
        )
        batch_size, neighbors, steps, _ = neighbor_future.shape
        _require(
            tuple(neutral_attention_weights.shape) == (batch_size, neighbors, steps),
            "neutral attention weights must be [B,N,T]",
        )
        _require(
            neutral_attention_weights.device == edited_future.device
            and neutral_attention_weights.dtype == edited_future.dtype
            and tuple(interaction_confidence.shape) == (batch_size,)
            and interaction_confidence.device == edited_future.device
            and interaction_confidence.dtype == edited_future.dtype,
            "neutral attention outputs must match behavior device/dtype",
        )
        _require(
            tuple(agent_valid_mask.shape) == (batch_size, neighbors)
            and agent_valid_mask.dtype == torch.bool,
            "agent_valid_mask must be bool [B,N]",
        )
        _require(
            neighbor_current_state.ndim == 3
            and neighbor_current_state.shape[:2] == (batch_size, neighbors)
            and neighbor_current_state.shape[-1] >= 6,
            "neighbor_current_state must be [B,N,>=6]",
        )
        _require(
            neighbor_current_state.device == edited_future.device
            and neighbor_current_state.dtype == edited_future.dtype,
            "neighbor_current_state must match behavior device/dtype",
        )
        _require(
            tuple(base_tangent.shape) == (batch_size, steps, 2),
            "base_tangent must be [B,T,2]",
        )
        _finite("edited_future", edited_future)
        _finite("neutral_future", neutral_future)
        _finite("neutral attention weights", neutral_attention_weights)
        _finite("base_tangent", base_tangent)

        # The weight map and validity logic are neutral-only references.  They
        # must have no behavior-loss gradient for any rho endpoint.
        weights = neutral_attention_weights.detach()
        confidence = interaction_confidence.detach()
        valid = ((~neighbor_mask) & agent_valid_mask[:, :, None]).detach()
        weights = weights * valid.to(dtype=weights.dtype)
        coverage = weights.sum(dim=(1, 2)).detach()
        normalized_weights = weights / coverage[:, None, None].clamp_min(1e-8)
        tangent = base_tangent.detach()

        edited_full = torch.cat((ego_current_state[:, None, :2], edited_future[..., :2]), dim=1)
        neutral_full = torch.cat((ego_current_state[:, None, :2], neutral_future[..., :2]), dim=1)
        edited_velocity = (edited_full[:, 1:] - edited_full[:, :-1]) / self.dt
        neutral_velocity = (neutral_full[:, 1:] - neutral_full[:, :-1]) / self.dt
        ego_long = (edited_velocity * tangent).sum(dim=-1)
        neutral_long = (neutral_velocity * tangent).sum(dim=-1)
        speed = torch.linalg.vector_norm(edited_velocity, dim=-1)

        neighbor_full = torch.cat(
            (neighbor_current_state[:, :, None, :2], neighbor_future[..., :2]), dim=2
        )
        neighbor_velocity = (neighbor_full[:, :, 1:] - neighbor_full[:, :, :-1]) / self.dt
        neighbor_long = (neighbor_velocity * tangent[:, None]).sum(dim=-1)
        relative = neighbor_future[..., :2] - edited_future[:, None, :, :2]
        longitudinal = (relative * tangent[:, None]).sum(dim=-1)
        if neighbor_current_state.shape[-1] > 7:
            lengths = neighbor_current_state[..., 7].abs()
            lengths = torch.where(lengths > 1e-3, lengths, torch.full_like(lengths, DEFAULT_NEIGHBOR_LENGTH_M))
        else:
            lengths = torch.full_like(longitudinal[:, :, 0], DEFAULT_NEIGHBOR_LENGTH_M)
        gap = (longitudinal - 0.5 * (self.ego_length_m + lengths[:, :, None])).clamp_min(0.0)
        thw = gap / ego_long[:, None].abs().clamp_min(self.min_speed_mps)
        headway_raw = (normalized_weights * thw).sum(dim=(1, 2))
        headway_valid = (coverage > 1e-6) & (confidence > 1e-6)

        neutral_closing = neutral_long[:, None] - neighbor_long
        ttc_valid_weights = weights * (neutral_closing >= self.closing_speed_min_mps).to(dtype=weights.dtype)
        ttc_coverage = ttc_valid_weights.sum(dim=(1, 2)).detach()
        closing = ego_long[:, None] - neighbor_long
        ttc = (gap / closing.clamp_min(1e-3)).clamp(max=self.ttc_cap_s)
        ttc_raw = (ttc_valid_weights / ttc_coverage[:, None, None].clamp_min(1e-8) * ttc).sum(dim=(1, 2))
        ttc_valid = (ttc_coverage > 1e-6) & (confidence > 1e-6)

        ratio = speed / speed_limit_mps[:, None].clamp_min(1e-3)
        speed_mask = speed_limit_valid[:, None].expand_as(ratio)
        speed_raw, speed_valid = _masked_mean(ratio, speed_mask)
        score, score_valid = _combine_scores(
            speed=self.calibration.canonical("speed_utilization", speed_raw),
            speed_valid=speed_valid,
            headway=self.calibration.canonical("headway_tightness_from_h", headway_raw),
            headway_valid=headway_valid,
            ttc=self.calibration.canonical("ttc_tightness", ttc_raw),
            ttc_valid=ttc_valid,
            free_mask=free_mask,
            car_mask=car_mask,
        )
        return AttentionBehaviorMeasurement(
            speed=self.calibration.canonical("speed_utilization", speed_raw),
            speed_valid=speed_valid,
            headway=self.calibration.canonical("headway_tightness_from_h", headway_raw),
            headway_valid=headway_valid,
            ttc=self.calibration.canonical("ttc_tightness", ttc_raw),
            ttc_valid=ttc_valid,
            score=score,
            score_valid=score_valid,
            raw_speed=speed_raw,
            raw_headway=headway_raw,
            raw_ttc=ttc_raw,
            attention_coverage=coverage,
            interaction_confidence=confidence,
        )

    def desired_score(
        self,
        neutral: BehaviorMeasurement,
        rho: torch.Tensor,
        *,
        free_mask: torch.Tensor,
        car_mask: torch.Tensor,
        positive_span: float,
        negative_span: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        speed = self.calibration.desired(neutral.speed, rho, positive_span=positive_span, negative_span=negative_span)
        headway = self.calibration.desired(neutral.headway, rho, positive_span=positive_span, negative_span=negative_span)
        ttc = self.calibration.desired(neutral.ttc, rho, positive_span=positive_span, negative_span=negative_span)
        return _combine_scores(
            speed=speed,
            speed_valid=neutral.speed_valid,
            headway=headway,
            headway_valid=neutral.headway_valid,
            ttc=ttc,
            ttc_valid=neutral.ttc_valid,
            free_mask=free_mask,
            car_mask=car_mask,
        )


def pathwise_order_loss(
    scores: torch.Tensor,
    valid: torch.Tensor,
    rho_grid: torch.Tensor,
    *,
    margin_per_rho: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adjacent-grid Preference Flow ordering, with no endpoint shortcut."""

    _require(scores.ndim == 2 and valid.ndim == 1, "scores must be [G,B] and valid [B]")
    _require(scores.shape[0] == rho_grid.numel() and scores.shape[1] == valid.numel(), "pathwise shape mismatch")
    deltas = scores[1:] - scores[:-1]
    required = float(margin_per_rho) * (rho_grid[1:] - rho_grid[:-1]).view(-1, 1)
    active = valid[None, :].expand_as(deltas)
    loss_values = torch.relu(required - deltas)
    denominator = active.to(dtype=loss_values.dtype).sum().clamp_min(1.0)
    return (loss_values * active.to(dtype=loss_values.dtype)).sum() / denominator, deltas


def feasible_pathwise_loss(
    scores: torch.Tensor,
    target_scores: torch.Tensor,
    valid: torch.Tensor,
    *,
    active_epsilon: float = 1e-5,
    active_fraction: float = 0.5,
) -> FeasiblePathwiseLoss:
    """Order a rho path without demanding motion beyond a clipped target edge.

    Targets are neutral-relative calibrated values.  Where an adjacent target
    interval is active, the score must move forward by a positive fraction of
    that target movement.  Where clipping produces a plateau, only reversal is
    penalized.
    """

    _require(scores.ndim == target_scores.ndim == 2 and valid.ndim == 1, "scores/targets must be [G,B] and valid [B]")
    _require(scores.shape == target_scores.shape and scores.shape[1] == valid.numel(), "feasible pathwise shape mismatch")
    _require(active_epsilon >= 0.0 and 0.0 < active_fraction <= 1.0, "invalid feasible pathwise settings")
    deltas = scores[1:] - scores[:-1]
    target_deltas = (target_scores[1:] - target_scores[:-1]).detach()
    interval_valid = valid[None, :].expand_as(deltas)
    active = interval_valid & (target_deltas > float(active_epsilon))
    saturated = interval_valid & ~active
    active_required = target_deltas * float(active_fraction)
    active_loss = torch.relu(active_required - deltas)
    saturated_loss = torch.relu(-deltas)
    losses = torch.where(active, active_loss, torch.where(saturated, saturated_loss, torch.zeros_like(deltas)))
    denominator = interval_valid.to(dtype=losses.dtype).sum().clamp_min(1.0)
    return FeasiblePathwiseLoss(
        loss=losses.sum() / denominator,
        deltas=deltas,
        target_deltas=target_deltas,
        active_mask=active,
        saturated_mask=saturated,
    )


def legacy_min_distance_proxy(
    ego_future: torch.Tensor, neighbor_future: torch.Tensor, neighbor_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Old proxy retained only for the Step-5-S1 sensitivity audit/report."""

    distance = torch.linalg.vector_norm(ego_future[:, None, :, :2] - neighbor_future[..., :2], dim=-1)
    distance = distance.masked_fill(neighbor_mask, float("inf"))
    flat = distance.reshape(distance.shape[0], -1)
    minimum, location = flat.min(dim=1)
    steps = neighbor_future.shape[2]
    valid = torch.isfinite(minimum)
    return -torch.where(valid, minimum, torch.zeros_like(minimum)), location // steps, location % steps, valid


__all__ = [
    "BehaviorAlignmentError",
    "AttentionBehaviorMeasurement",
    "BehaviorMeasurement",
    "DifferentiableBehaviorBridge",
    "ExistingFormalMetricResult",
    "FrozenAxisCalibration",
    "FrozenBehaviorCalibration",
    "FeasiblePathwiseLoss",
    "LeadReference",
    "NeutralAnchoredLeadReference",
    "legacy_min_distance_proxy",
    "existing_formal_metric_for_trajectory",
    "feasible_pathwise_loss",
    "longitudinal_perturbation",
    "pathwise_order_loss",
    "trajectory_tangents",
]

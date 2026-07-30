"""Pure data contracts for Step 5-S3A longitudinal transport targets.

The module deliberately has no model, optimizer, or trajectory-editing code.
It converts coherent V6 labels and an expert trajectory into an auditable
longitudinal progress target along a frozen neutral path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "preference_flow_step5_s3a_longitudinal_transport_v1"
DEFAULT_MAX_AXIS_SPREAD = 0.25
DEFAULT_MAX_PROJECTION_DISTANCE_M = 5.0
DEFAULT_NEUTRAL_RHO_BAND = 0.05
NUMERICAL_BACKTRACKING_LIMIT_M = 0.05
_EPS = 1e-8


class LongitudinalTransportError(ValueError):
    """Raised when a transport-contract input cannot be interpreted safely."""


@dataclass(frozen=True)
class AxisCoherence:
    active_axis_indices: tuple[int, ...]
    active_axes: tuple[float, ...]
    axis_spread: float | None
    rho_star: float | None
    target_rho_star: float | None
    one_dimensional_coherent: bool
    exclusion_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(frozen=True)
class ProjectionResult:
    neutral_progress: np.ndarray
    expert_projected_progress: np.ndarray
    projection_distances: np.ndarray
    projection_segment_indices: np.ndarray
    zero_length_segment_indices: np.ndarray
    total_neutral_length: float
    monotonic_violation: bool
    maximum_backward_progress: float
    valid: bool
    invalid_reasons: tuple[str, ...]

    @property
    def delta_s_target(self) -> np.ndarray | None:
        if self.neutral_progress.shape != self.expert_projected_progress.shape:
            return None
        return self.expert_projected_progress - self.neutral_progress

    def to_dict(self) -> dict[str, Any]:
        delta = self.delta_s_target
        return {
            "neutral_progress_m": json_ready(self.neutral_progress),
            "expert_projected_progress_m": json_ready(self.expert_projected_progress),
            "projection_distance_m": json_ready(self.projection_distances),
            "projection_segment_indices": json_ready(self.projection_segment_indices),
            "zero_length_segment_indices": json_ready(self.zero_length_segment_indices),
            "projection_distance_summary_m": scalar_summary(self.projection_distances),
            "total_neutral_length_m": float(self.total_neutral_length),
            "monotonic_violation": bool(self.monotonic_violation),
            "maximum_backward_progress_m": float(self.maximum_backward_progress),
            "backtracking_classification": classify_projection_backtracking(
                self.maximum_backward_progress
            ),
            "neutral_endpoint_progress_m": endpoint_or_none(self.neutral_progress),
            "expert_projected_endpoint_progress_m": endpoint_or_none(
                self.expert_projected_progress
            ),
            "endpoint_progress_delta_m": (
                None
                if not self.neutral_progress.size or not self.expert_projected_progress.size
                else float(self.expert_projected_progress[-1] - self.neutral_progress[-1])
            ),
            "delta_s_target_m": None if delta is None else json_ready(delta),
            "delta_s_target_summary_m": None if delta is None else scalar_summary(delta),
            "valid": bool(self.valid),
            "invalid_reasons": list(self.invalid_reasons),
        }


@dataclass(frozen=True)
class TransportTarget:
    preference_coordinate_r: float
    target_progress: np.ndarray
    target_progress_residual: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "preference_coordinate_r": float(self.preference_coordinate_r),
            "target_progress_m": json_ready(self.target_progress),
            "target_progress_residual_m": json_ready(self.target_progress_residual),
        }


def json_ready(value: Any) -> Any:
    """Return a JSON-native representation without rounding audit evidence."""
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def scalar_summary(values: Iterable[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "min": None, "mean": None, "std": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def endpoint_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(values[-1])


def classify_projection_backtracking(maximum_backward_progress_m: float) -> str:
    """Separate local numeric projection jitter from semantic reversal."""
    if not np.isfinite(maximum_backward_progress_m) or maximum_backward_progress_m < 0.0:
        raise LongitudinalTransportError("maximum_backward_progress_m must be finite and non-negative")
    if float(maximum_backward_progress_m) == 0.0:
        return "none"
    if float(maximum_backward_progress_m) <= NUMERICAL_BACKTRACKING_LIMIT_M:
        return "numerical_or_local_projection_backtracking"
    return "semantic_backtracking"


def axis_coherence_from_style(
    style_value_condition: Sequence[float] | np.ndarray,
    *,
    max_axis_spread: float = DEFAULT_MAX_AXIS_SPREAD,
) -> AxisCoherence:
    """Build the continuous label without quantising it to an old rho grid."""
    if not np.isfinite(max_axis_spread) or max_axis_spread < 0.0:
        raise LongitudinalTransportError("max_axis_spread must be finite and non-negative")
    try:
        style = np.asarray(style_value_condition, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        return AxisCoherence((), (), None, None, None, False, (f"invalid_style_vector:{error}",))
    if style.size != 12:
        return AxisCoherence((), (), None, None, None, False, ("style_vector_length_not_12",))
    targets, causal_mask = style[:3], style[3:6]
    if not np.isfinite(np.concatenate((targets, causal_mask))).all():
        return AxisCoherence((), (), None, None, None, False, ("nonfinite_style_axis",))
    indices = tuple(int(index) for index in np.flatnonzero(causal_mask > 0.5))
    if not indices:
        return AxisCoherence((), (), None, None, None, False, ("no_active_style_axis",))
    active = targets[list(indices)]
    if np.any((active < 0.0) | (active > 1.0)):
        return AxisCoherence(indices, tuple(float(value) for value in active), None, None, None, False, ("active_axis_outside_normalized_range",))
    spread = float(active.max() - active.min())
    rho_star = float(2.0 * active.mean() - 1.0)
    coherent = spread <= float(max_axis_spread)
    return AxisCoherence(
        active_axis_indices=indices,
        active_axes=tuple(float(value) for value in active),
        axis_spread=spread,
        rho_star=rho_star,
        target_rho_star=rho_star if coherent else None,
        one_dimensional_coherent=coherent,
        exclusion_reasons=() if coherent else ("axis_spread_exceeds_max_axis_spread",),
    )


def pairwise_axis_statistics(rows: Iterable[AxisCoherence]) -> dict[str, dict[str, float | int | None]]:
    """Report label agreement only; it does not create a learned style embedding."""
    materialized = list(rows)
    report: dict[str, dict[str, float | int | None]] = {}
    for first in range(3):
        for second in range(first + 1, 3):
            pairs: list[tuple[float, float]] = []
            for row in materialized:
                lookup = dict(zip(row.active_axis_indices, row.active_axes))
                if first in lookup and second in lookup:
                    pairs.append((float(lookup[first]), float(lookup[second])))
            key = f"axis_{first}_axis_{second}"
            if not pairs:
                report[key] = {
                    "pair_count": 0,
                    "pearson_correlation": None,
                    "same_direction_rate": None,
                    "opposite_direction_rate": None,
                    "neutral_tie_rate": None,
                }
                continue
            values = np.asarray(pairs, dtype=np.float64)
            centered = values - 0.5
            product = centered[:, 0] * centered[:, 1]
            neutral_tie = np.isclose(product, 0.0, atol=_EPS)
            opposite = product < -_EPS
            correlation = None
            if values.shape[0] >= 2 and values[:, 0].std() > _EPS and values[:, 1].std() > _EPS:
                correlation = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
            report[key] = {
                "pair_count": int(values.shape[0]),
                "pearson_correlation": correlation,
                "same_direction_rate": float(np.mean(~opposite)),
                "opposite_direction_rate": float(np.mean(opposite)),
                "neutral_tie_rate": float(np.mean(neutral_tie)),
            }
    return report


def rho_distribution(
    values: Iterable[float], *, neutral_band: float = DEFAULT_NEUTRAL_RHO_BAND
) -> dict[str, Any]:
    if not np.isfinite(neutral_band) or neutral_band < 0.0:
        raise LongitudinalTransportError("neutral_band must be finite and non-negative")
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    summary = scalar_summary(array)
    if array.size == 0:
        return {**summary, "quantiles": {}, "coverage": {"negative": 0, "exact_zero": 0, "positive": 0, "near_neutral": 0}, "near_neutral_abs_le": float(neutral_band)}
    return {
        **summary,
        "quantiles": {
            "q05": float(np.percentile(array, 5)),
            "q25": float(np.percentile(array, 25)),
            "q50": float(np.percentile(array, 50)),
            "q75": float(np.percentile(array, 75)),
            "q95": float(np.percentile(array, 95)),
        },
        "coverage": {
            "negative": int(np.sum(array < 0.0)),
            "exact_zero": int(np.sum(np.isclose(array, 0.0, atol=_EPS))),
            "positive": int(np.sum(array > 0.0)),
            "near_neutral": int(np.sum(np.abs(array) <= neutral_band)),
        },
        "near_neutral_abs_le": float(neutral_band),
    }


def _xy_points(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2 or array.shape[0] == 0:
        raise LongitudinalTransportError(f"{label} must have shape [T, >=2] with T > 0")
    array = array[:, :2]
    if not np.isfinite(array).all():
        raise LongitudinalTransportError(f"{label} contains NaN or Inf")
    return array


def _xy_point(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.isfinite(array[:2]).all():
        raise LongitudinalTransportError(f"{label} must contain two finite coordinates")
    return array[:2]


def trajectory_sanity_audit(
    ego_current_xy: Any,
    neutral_future_xy: Any,
    expert_future_xy: Any,
    *,
    dt_seconds: float = 0.1,
) -> dict[str, Any]:
    """Audit a physical neutral trajectory before any projection is attempted.

    The discontinuity flags are transparent geometric checks, not safety rules
    and never change the trajectory.
    """
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise LongitudinalTransportError("dt_seconds must be finite and positive")
    try:
        current = _xy_point(ego_current_xy, "ego_current_xy")
        neutral = _xy_points(neutral_future_xy, "neutral_future_xy")
        expert = _xy_points(expert_future_xy, "expert_future_xy")
    except LongitudinalTransportError as error:
        return {
            "all_finite": False,
            "neutral_generation_invalid": True,
            "invalid_reasons": [str(error)],
        }
    positions = np.concatenate((current[None, :], neutral), axis=0)
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = displacement / float(dt_seconds)
    median_displacement = float(np.median(displacement)) if displacement.size else 0.0
    discontinuity_limit = max(5.0, 5.0 * max(median_displacement, _EPS))
    first_displacement = float(displacement[0]) if displacement.size else 0.0
    reasons: list[str] = []
    if neutral.shape[0] != expert.shape[0]:
        reasons.append("neutral_expert_horizon_mismatch")
    if first_displacement > discontinuity_limit:
        reasons.append("first_future_point_discontinuity")
    if displacement.size and float(displacement.max()) > discontinuity_limit:
        reasons.append("trajectory_discontinuity")
    ade = None
    fde = None
    if neutral.shape == expert.shape:
        error = np.linalg.norm(neutral - expert, axis=1)
        ade = float(error.mean())
        fde = float(error[-1])
    return {
        "ego_current_xy": json_ready(current),
        "first_future_point_xy": json_ready(neutral[0]),
        "first_frame_displacement_m": first_displacement,
        "frame_displacement_m": scalar_summary(displacement),
        "speed_mps": scalar_summary(speed),
        "total_path_length_m": float(displacement.sum()),
        "direct_ade_m": ade,
        "direct_fde_m": fde,
        "all_finite": True,
        "trajectory_horizon": int(neutral.shape[0]),
        "expert_horizon": int(expert.shape[0]),
        "dt_seconds": float(dt_seconds),
        "discontinuity_limit_m": discontinuity_limit,
        "neutral_generation_invalid": bool(reasons),
        "invalid_reasons": reasons,
    }


def _invalid_projection(reason: str) -> ProjectionResult:
    empty_float = np.empty((0,), dtype=np.float64)
    return ProjectionResult(
        neutral_progress=empty_float,
        expert_projected_progress=empty_float,
        projection_distances=empty_float,
        projection_segment_indices=np.empty((0,), dtype=np.int64),
        zero_length_segment_indices=np.empty((0,), dtype=np.int64),
        total_neutral_length=0.0,
        monotonic_violation=False,
        maximum_backward_progress=0.0,
        valid=False,
        invalid_reasons=(reason,),
    )


def project_expert_onto_neutral_path(
    ego_current_xy: Any,
    neutral_future_xy: Any,
    expert_future_xy: Any,
    *,
    max_projection_distance_m: float = DEFAULT_MAX_PROJECTION_DISTANCE_M,
    monotonic_tolerance_m: float = 1e-5,
) -> ProjectionResult:
    """Project each expert point onto the physical neutral polyline.

    Zero-length neutral segments are skipped, never silently stretched.  The
    returned progress is the raw nearest-segment result; non-monotonic progress
    is flagged rather than repaired with ``cummax``.
    """
    if not np.isfinite(max_projection_distance_m) or max_projection_distance_m <= 0.0:
        raise LongitudinalTransportError("max_projection_distance_m must be finite and positive")
    if not np.isfinite(monotonic_tolerance_m) or monotonic_tolerance_m < 0.0:
        raise LongitudinalTransportError("monotonic_tolerance_m must be finite and non-negative")
    try:
        current = _xy_point(ego_current_xy, "ego_current_xy")
        neutral_future = _xy_points(neutral_future_xy, "neutral_future_xy")
        expert_future = _xy_points(expert_future_xy, "expert_future_xy")
    except LongitudinalTransportError as error:
        return _invalid_projection(str(error))

    polyline = np.concatenate((current[None, :], neutral_future), axis=0)
    segments = polyline[1:] - polyline[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    usable = segment_lengths > _EPS
    if not bool(np.any(usable)):
        return ProjectionResult(
            neutral_progress=cumulative[1:],
            expert_projected_progress=np.empty((0,), dtype=np.float64),
            projection_distances=np.empty((0,), dtype=np.float64),
            projection_segment_indices=np.empty((0,), dtype=np.int64),
            zero_length_segment_indices=np.flatnonzero(~usable).astype(np.int64),
            total_neutral_length=0.0,
            monotonic_violation=False,
            maximum_backward_progress=0.0,
            valid=False,
            invalid_reasons=("neutral_path_has_no_nonzero_segment",),
        )

    projected_progress = np.empty((expert_future.shape[0],), dtype=np.float64)
    distances = np.empty_like(projected_progress)
    indices = np.empty((expert_future.shape[0],), dtype=np.int64)
    usable_indices = np.flatnonzero(usable)
    for point_index, point in enumerate(expert_future):
        starts = polyline[usable_indices]
        vectors = segments[usable_indices]
        lengths_sq = np.square(segment_lengths[usable_indices])
        fractions = np.sum((point[None, :] - starts) * vectors, axis=1) / lengths_sq
        fractions = np.clip(fractions, 0.0, 1.0)
        candidates = starts + fractions[:, None] * vectors
        candidate_distances = np.linalg.norm(point[None, :] - candidates, axis=1)
        local_index = int(np.argmin(candidate_distances))
        segment_index = int(usable_indices[local_index])
        indices[point_index] = segment_index
        distances[point_index] = float(candidate_distances[local_index])
        projected_progress[point_index] = float(
            cumulative[segment_index] + fractions[local_index] * segment_lengths[segment_index]
        )

    neutral_progress = cumulative[1:]
    reasons: list[str] = []
    if neutral_future.shape[0] != expert_future.shape[0]:
        reasons.append("neutral_expert_horizon_mismatch")
    progress_delta = np.diff(projected_progress)
    maximum_backward = float(max(0.0, -float(progress_delta.min()))) if progress_delta.size else 0.0
    monotonic_violation = bool(np.any(progress_delta < -monotonic_tolerance_m))
    if monotonic_violation:
        reasons.append("expert_projected_progress_backtracks")
    if float(distances.max()) > float(max_projection_distance_m):
        reasons.append("projection_distance_exceeds_max_projection_distance_m")
    return ProjectionResult(
        neutral_progress=neutral_progress,
        expert_projected_progress=projected_progress,
        projection_distances=distances,
        projection_segment_indices=indices,
        zero_length_segment_indices=np.flatnonzero(~usable).astype(np.int64),
        total_neutral_length=float(cumulative[-1]),
        monotonic_violation=monotonic_violation,
        maximum_backward_progress=maximum_backward,
        valid=not reasons,
        invalid_reasons=tuple(reasons),
    )


def build_continuous_transport_target(
    neutral_progress: Any,
    expert_projected_progress: Any,
    rho_star: float,
    lambda_value: float,
) -> TransportTarget:
    """Interpolate the target progress from neutral to the projected expert."""
    neutral = np.asarray(neutral_progress, dtype=np.float64)
    expert = np.asarray(expert_projected_progress, dtype=np.float64)
    if neutral.ndim != 1 or expert.ndim != 1 or neutral.shape != expert.shape or neutral.size == 0:
        raise LongitudinalTransportError("neutral and expert progress must be equal non-empty 1-D arrays")
    if not np.isfinite(neutral).all() or not np.isfinite(expert).all():
        raise LongitudinalTransportError("progress arrays contain NaN or Inf")
    if not np.isfinite(rho_star) or not -1.0 - _EPS <= float(rho_star) <= 1.0 + _EPS:
        raise LongitudinalTransportError("rho_star must be finite and lie in [-1, 1]")
    if not np.isfinite(lambda_value) or not 0.0 <= float(lambda_value) <= 1.0:
        raise LongitudinalTransportError("lambda_value must be finite and lie in [0, 1]")
    if float(lambda_value) == 0.0:
        target = neutral.copy()
    elif float(lambda_value) == 1.0:
        target = expert.copy()
    else:
        target = neutral + float(lambda_value) * (expert - neutral)
    return TransportTarget(
        preference_coordinate_r=float(lambda_value) * float(rho_star),
        target_progress=target,
        target_progress_residual=target - neutral,
    )


def transport_coordinate_is_identifiable(
    neutral_progress: Any, expert_projected_progress: Any, rho_star: float
) -> bool:
    """Reject a non-neutral endpoint that would be assigned to r=0."""
    neutral = np.asarray(neutral_progress, dtype=np.float64)
    expert = np.asarray(expert_projected_progress, dtype=np.float64)
    if neutral.ndim != 1 or expert.ndim != 1 or neutral.shape != expert.shape:
        return False
    return float(rho_star) != 0.0 or bool(np.array_equal(neutral, expert))


def sample_neutral_polyline(
    ego_current_xy: Any, neutral_future_xy: Any, progress: Any
) -> np.ndarray:
    """Reconstruct XY only for raw feasibility diagnostics, never as a target."""
    current = _xy_point(ego_current_xy, "ego_current_xy")
    future = _xy_points(neutral_future_xy, "neutral_future_xy")
    requested = np.asarray(progress, dtype=np.float64).reshape(-1)
    if requested.size == 0 or not np.isfinite(requested).all():
        raise LongitudinalTransportError("progress must be a non-empty finite vector")
    polyline = np.concatenate((current[None, :], future), axis=0)
    vectors = polyline[1:] - polyline[:-1]
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if not bool(np.any(lengths > _EPS)):
        raise LongitudinalTransportError("neutral_path_has_no_nonzero_segment")
    if float(requested.min()) < -_EPS or float(requested.max()) > float(cumulative[-1]) + _EPS:
        raise LongitudinalTransportError("progress lies outside the neutral path")
    sampled = np.empty((requested.size, 2), dtype=np.float64)
    for row, value in enumerate(requested):
        candidates = np.flatnonzero((lengths > _EPS) & (value <= cumulative[1:] + _EPS) & (value >= cumulative[:-1] - _EPS))
        if candidates.size == 0:
            candidates = np.flatnonzero(lengths > _EPS)
            segment = int(candidates[-1])
        else:
            segment = int(candidates[0])
        fraction = (float(value) - cumulative[segment]) / lengths[segment]
        sampled[row] = polyline[segment] + np.clip(fraction, 0.0, 1.0) * vectors[segment]
    return sampled


def raw_physical_feasibility_audit(
    ego_current_xy: Any,
    neutral_future_xy: Any,
    projected_progress: Any,
    neighbor_future: Any | None,
    *,
    dt_seconds: float = 0.1,
) -> dict[str, Any]:
    """Report raw physics for the projected target without imposing a new rule."""
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise LongitudinalTransportError("dt_seconds must be finite and positive")
    try:
        current = _xy_point(ego_current_xy, "ego_current_xy")
        target_xy = sample_neutral_polyline(ego_current_xy, neutral_future_xy, projected_progress)
    except LongitudinalTransportError as error:
        return {
            "official_feasibility_available": False,
            "official_feasibility_unavailable": True,
            "official_collision": None,
            "raw_metrics_available": False,
            "failure": str(error),
        }
    positions = np.concatenate((current[None, :], target_xy), axis=0)
    step_distance = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = step_distance / float(dt_seconds)
    acceleration = np.diff(speed) / float(dt_seconds)
    jerk = np.diff(acceleration) / float(dt_seconds)
    progress = np.asarray(projected_progress, dtype=np.float64).reshape(-1)
    median_step = float(np.median(step_distance)) if step_distance.size else 0.0
    discontinuity = bool(
        step_distance.size
        and float(step_distance.max()) > max(1.0, 5.0 * max(median_step, _EPS))
    )
    min_neighbor_distance: float | None = None
    valid_neighbor_points = 0
    if neighbor_future is not None:
        neighbors = np.asarray(neighbor_future, dtype=np.float64)
        if neighbors.ndim == 3 and neighbors.shape[-1] >= 2:
            horizon = min(target_xy.shape[0], neighbors.shape[1])
            if horizon:
                neighbor_xy = neighbors[:, :horizon, :2]
                validity_dims = min(3, neighbors.shape[-1])
                valid = np.any(np.abs(neighbors[:, :horizon, :validity_dims]) > _EPS, axis=-1)
                valid &= np.isfinite(neighbor_xy).all(axis=-1)
                distance = np.linalg.norm(target_xy[None, :horizon, :] - neighbor_xy, axis=-1)
                valid_neighbor_points = int(valid.sum())
                if valid_neighbor_points:
                    min_neighbor_distance = float(distance[valid].min())
    return {
        "official_feasibility_available": False,
        "official_feasibility_unavailable": True,
        "official_collision": None,
        "official_tool_note": "No stable StylePlanner formal evaluator accepts a projected longitudinal-only cache target; raw physical diagnostics only.",
        "raw_metrics_available": True,
        "minimum_neighbor_distance_m": min_neighbor_distance,
        "valid_neighbor_points": valid_neighbor_points,
        "speed_mps": scalar_summary(speed),
        "acceleration_abs_mps2": scalar_summary(np.abs(acceleration)),
        "jerk_abs_mps3": scalar_summary(np.abs(jerk)),
        "maximum_frame_displacement_m": float(step_distance.max()) if step_distance.size else 0.0,
        "median_frame_displacement_m": median_step,
        "has_backward_progress": bool(np.any(np.diff(progress) < -1e-5)),
        "obvious_discontinuity": discontinuity,
        "discontinuity_rule": "raw flag: max frame displacement > max(1m, 5x median); not a safety criterion",
    }


def validate_explicit_audit_output_dir(output_dir: str | Path, repository_root: str | Path) -> Path:
    """S3A has no production default output; reject a baseline source location."""
    raw = str(output_dir).strip()
    if not raw:
        raise LongitudinalTransportError("an explicit --output-dir is required")
    output = Path(raw).expanduser().resolve()
    baseline_root = (Path(repository_root).expanduser().resolve() / "baseline")
    try:
        output.relative_to(baseline_root)
    except ValueError:
        return output
    raise LongitudinalTransportError("S3A audit output must not be written inside baseline/")


__all__ = [
    "AxisCoherence",
    "DEFAULT_MAX_AXIS_SPREAD",
    "DEFAULT_MAX_PROJECTION_DISTANCE_M",
    "DEFAULT_NEUTRAL_RHO_BAND",
    "LongitudinalTransportError",
    "NUMERICAL_BACKTRACKING_LIMIT_M",
    "ProjectionResult",
    "SCHEMA_VERSION",
    "TransportTarget",
    "axis_coherence_from_style",
    "build_continuous_transport_target",
    "classify_projection_backtracking",
    "json_ready",
    "pairwise_axis_statistics",
    "project_expert_onto_neutral_path",
    "raw_physical_feasibility_audit",
    "rho_distribution",
    "sample_neutral_polyline",
    "scalar_summary",
    "transport_coordinate_is_identifiable",
    "trajectory_sanity_audit",
    "validate_explicit_audit_output_dir",
]

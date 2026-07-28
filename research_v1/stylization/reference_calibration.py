"""V5 condition-calibrated behavior-axis data pipeline.

This module intentionally does *not* use ``split_valid``, ``style_label`` or
``subset_id`` as a label source.  The style-scene split is only an observed
scene/quality index; V5 measures behavior again from the planner cache and
constructs continuous labels inside the training split.

The active artifacts are deliberately staged so every scientific claim can be
audited:

``v5_candidates.jsonl`` -> ``v5_normalized.jsonl`` ->
``v5_conditional_rank.jsonl``.

V6 consumes the final conditional-percentile artifact directly.  The former
empirical single-latent-rho experiment is not part of the active CLI or data
contract.

The implementation only depends on NumPy.  If scikit-learn is available it is
used for scalable nearest-neighbour search; the NumPy fallback is suitable for
the intended small smoke run but should not be used for a full 100k+ split.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from research_v1.stylization.metrics import build_behavior_metric_bundle
from research_v1.stylization.schema import (
    CANONICAL_AXIS_BY_SCENE,
    RAW_METRIC_DIRECTION_BY_SCENE,
    SCENE_BUCKET_ORDER,
)
from research_v1.scene_data.index import (
    load_jsonl_records,
    normalize_split_index_record,
    write_jsonl_records,
)


V5_SCHEMA_VERSION = 5
V5_NAME = "continuous_style_v5_condition_calibrated"
PRIMARY_SCENES = tuple(SCENE_BUCKET_ORDER)
CONTROLLED_SCENES = ("straight_free_drive", "straight_car_follow")
INTERACTION_SCENES = ("straight_car_follow", "straight_lane_change")
# These axes are only observable when the relevant traffic opportunity occurs.
# They must be masked outside that opportunity, rather than judged by the same
# coverage target as always-observable execution axes.
OPPORTUNITY_AXIS_NAMES = frozenset({"small_gap_acceptance_from_m_gap"})
# A conditional CDF needs enough *valid labels* in its local reference set.
# Most axes are observed frequently, so a pool of 3k nearest contexts is ample.
# ``m_gap`` is deliberately an opportunity-masked axis: only roughly one in ten
# lane-change examples contains an observable target-lane front/rear conflict.
# It therefore receives a wider candidate pool, then still uses only the closest
# k valid labels.  This is not value imputation and it does not relax the
# effective-neighbour requirement.
DEFAULT_CONDITIONAL_CANDIDATE_POOL_MULTIPLIER = 3
DEFAULT_OPPORTUNITY_CANDIDATE_POOL_MULTIPLIER = 8


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to portable JSON values."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, target)


def _write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_records(str(target), (_json_safe(record) for record in records))


def _read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return [dict(record) for record in load_jsonl_records(str(path))]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bool_list(value: Any, length: int = 3) -> np.ndarray:
    values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else []
    return np.asarray([bool(values[index]) if index < len(values) else False for index in range(length)], dtype=bool)


def _float_vector(value: Any, length: int = 3, default: float = 0.0) -> np.ndarray:
    values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else []
    return np.asarray([_float(values[index], default) if index < len(values) else default for index in range(length)], dtype=np.float64)


def _stable_fraction(sample_id: str, seed: int) -> float:
    encoded = f"{int(seed)}:{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") / float(2**64)


def _stable_fold(sample_id: str, folds: int, seed: int) -> int:
    encoded = f"fold:{int(seed)}:{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % max(int(folds), 1)


def _scene_choice(scene_filter: str) -> Tuple[str, ...]:
    if scene_filter == "all":
        return PRIMARY_SCENES
    if scene_filter == "controlled":
        return CONTROLLED_SCENES
    if scene_filter == "interaction":
        return INTERACTION_SCENES
    if scene_filter not in PRIMARY_SCENES:
        raise ValueError(
            f"Unsupported scene_filter={scene_filter!r}; choices are all, controlled, interaction, or {PRIMARY_SCENES}"
        )
    return (scene_filter,)


def _cache_path(record: Mapping[str, Any], planner_cache_dir: str) -> str:
    """Resolve the cache path, preferring an explicit remote cache root.

    Sidecar indexes often contain the absolute path from the machine which
    built them.  ``--planner-cache-dir`` makes uploaded scripts portable.
    """

    if planner_cache_dir:
        filename = Path(str(record.get("filename", "") or "")).name
        if filename:
            return str(Path(planner_cache_dir) / filename)
    for key in ("planner_cache_path", "cache_path", "style_cache_path"):
        candidate = str(record.get(key, "") or "").strip()
        if candidate:
            return candidate
    return ""


def _route_curvature_from_cache(cache_path: str) -> float | None:
    """Robust route-polyline curvature summary for condition calibration."""

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            lanes = np.asarray(data["route_lanes"], dtype=np.float64)
            mask = np.asarray(data["route_lanes_mask"], dtype=bool)
    except (KeyError, OSError, ValueError):
        return None
    if lanes.ndim != 3 or lanes.shape[-1] < 2:
        return None
    values: List[Tuple[float, float]] = []
    for lane_index in range(min(lanes.shape[0], mask.shape[0] if mask.ndim >= 2 else 0)):
        valid = mask[lane_index] if mask.ndim == 2 else np.ones((lanes.shape[1],), dtype=bool)
        xy = lanes[lane_index, valid, :2]
        if xy.shape[0] < 3 or not np.all(np.isfinite(xy)):
            continue
        delta = np.diff(xy, axis=0)
        distance = np.linalg.norm(delta, axis=1)
        keep = distance > 1e-3
        if int(np.sum(keep)) < 2:
            continue
        heading = np.unwrap(np.arctan2(delta[keep, 1], delta[keep, 0]))
        segment = distance[keep]
        curvature = np.abs(np.diff(heading)) / np.maximum(0.5 * (segment[1:] + segment[:-1]), 1e-3)
        finite_curvature = curvature[np.isfinite(curvature)]
        if finite_curvature.size:
            # The ego origin is (0, 0) in every cached local frame.  Use the
            # nearest valid route lane rather than averaging unrelated route
            # branches downstream.
            origin_distance = float(np.min(np.sum(xy**2, axis=1)))
            values.append((origin_distance, float(np.median(finite_curvature))))
    if not values:
        return None
    nearest = min(values, key=lambda item: item[0])[1]
    return float(np.clip(nearest, 0.0, 0.5))


def _current_speed_from_cache(cache_path: str) -> float | None:
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            state = np.asarray(data["ego_current_state"], dtype=np.float64).reshape(-1)
    except (KeyError, OSError, ValueError):
        return None
    if state.shape[0] >= 6 and np.all(np.isfinite(state[4:6])):
        return float(np.linalg.norm(state[4:6]))
    return None


def _initial_local_route_speed_limit_from_cache(cache_path: str) -> float | None:
    """Return the speed limit of the route lane nearest the ego origin.

    Lane-change caches do not write a route-speed-limit value into their metric
    auxiliary fields.  Falling back to a global lane average would mix parallel
    or downstream branches, so this reads the causal map input and selects the
    closest valid speed-limited route lane at the current ego position.
    """

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            lanes = np.asarray(data["route_lanes"], dtype=np.float64)
            mask = np.asarray(data["route_lanes_mask"], dtype=bool)
            limits = np.asarray(data["route_lanes_speed_limit"], dtype=np.float64).reshape(-1)
            has_limit = np.asarray(data["route_lanes_has_speed_limit"], dtype=bool).reshape(-1)
    except (KeyError, OSError, ValueError):
        return None
    if lanes.ndim != 3 or lanes.shape[-1] < 2:
        return None
    best: Tuple[float, float] | None = None
    lane_count = min(lanes.shape[0], limits.shape[0], has_limit.shape[0])
    for lane_index in range(lane_count):
        if not bool(has_limit[lane_index]):
            continue
        limit = float(limits[lane_index])
        if not math.isfinite(limit) or limit <= 0.5:
            continue
        lane_mask = mask[lane_index] if mask.ndim == 2 and lane_index < mask.shape[0] else None
        xy = lanes[lane_index, :, :2]
        if lane_mask is not None and lane_mask.shape[0] == xy.shape[0]:
            xy = xy[lane_mask]
        else:
            xy = xy[np.linalg.norm(xy, axis=1) > 1e-4]
        if xy.size == 0 or not np.all(np.isfinite(xy)):
            continue
        distance_sq = float(np.min(np.sum(xy**2, axis=1)))
        candidate = (distance_sq, limit)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return None if best is None else float(best[1])


def _initial_lane_context_from_cache(cache_path: str) -> Tuple[float | None, float | None]:
    """Return causal lane-option and traffic-density context for lane changes.

    No target direction or future-reached lane is used here.  Both quantities
    are available at planning time and explain opportunity differences without
    leaking the lane-change outcome into conditional label calibration.
    """

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            route_mask = np.asarray(data["route_lanes_mask"], dtype=bool)
            agents = np.asarray(data["neighbor_agents_past"], dtype=np.float64)
            agent_mask = np.asarray(data["neighbor_agents_past_mask"], dtype=bool)
    except (KeyError, OSError, ValueError):
        return None, None
    route_lane_count = None
    if route_mask.ndim >= 2:
        route_lane_count = float(np.sum(np.any(route_mask, axis=-1)))
    if agents.ndim != 3 or agent_mask.ndim != 2 or agents.shape[0] == 0 or agent_mask.shape[1] == 0:
        return route_lane_count, None
    frame = min(agents.shape[1], agent_mask.shape[1]) - 1
    if frame < 0:
        return route_lane_count, None
    state = agents[:, frame]
    valid = agent_mask[: state.shape[0], frame].astype(bool)
    if state.shape[1] >= 9:
        valid &= state[:, 8] > 0.5
    finite_xy = np.all(np.isfinite(state[:, :2]), axis=1) if state.shape[1] >= 2 else np.zeros_like(valid)
    nearby = valid & finite_xy & (np.linalg.norm(state[:, :2], axis=1) <= 40.0)
    return route_lane_count, float(np.sum(nearby))


def _initial_follow_context_from_cache(
    cache_path: str,
) -> Tuple[float | None, float | None, float | None, float | None, float | None]:
    """Return current lead headway and closing speed from the *input* history.

    This deliberately reads ``neighbor_agents_past`` rather than a future
    outcome.  It makes the follow-scene conditional CDF usable even when the
    old sidecar does not contain a relative-speed field.
    """

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            ego = np.asarray(data["ego_current_state"], dtype=np.float64).reshape(-1)
            agents = np.asarray(data["neighbor_agents_past"], dtype=np.float64)
            mask = np.asarray(data["neighbor_agents_past_mask"], dtype=bool)
    except (KeyError, OSError, ValueError):
        return None, None, None, None, None
    if ego.shape[0] < 6 or agents.ndim != 3 or agents.shape[-1] < 6 or mask.ndim != 2:
        return None, None, None, None, None
    # Both ego and neighbour history were transformed into the anchor ego frame
    # during cache construction.  Longitudinal x velocity, rather than a speed
    # norm difference, is the physically relevant following closing speed.
    ego_speed = float(ego[4])
    if not math.isfinite(ego_speed):
        return None, None, None, None, None
    frame = min(agents.shape[1], mask.shape[1]) - 1
    if frame < 0:
        return None, None, None, None, None
    valid = mask[: agents.shape[0], frame]
    state = agents[: valid.shape[0], frame]
    if state.shape[0] == 0:
        return None, None, None, None, None
    longitudinal = state[:, 0]
    lateral = state[:, 1]
    candidate = valid & np.isfinite(longitudinal) & np.isfinite(lateral) & (longitudinal > 0.0) & (np.abs(lateral) < 1.9)
    if not np.any(candidate):
        return None, None, None, None, None
    indices = np.where(candidate)[0]
    lengths = state[indices, 7] if state.shape[1] > 7 else np.full((indices.shape[0],), 4.5)
    gap = np.maximum(longitudinal[indices] - 0.5 * (4.8 + np.maximum(lengths, 0.0)), 0.0)
    selected = int(indices[int(np.argmin(gap))])
    selected_gap = float(np.min(gap))
    lead_speed = float(state[selected, 4])
    lead_accel = None
    if frame >= 1 and mask[selected, frame - 1] and np.isfinite(agents[selected, frame - 1, 4]):
        lead_accel = float((lead_speed - float(agents[selected, frame - 1, 4])) / 0.1)
    headway = selected_gap / max(abs(ego_speed), 1.5)
    closing = max(ego_speed - lead_speed, 0.0)
    return headway, closing, ego_speed, lead_speed, lead_accel


def _condition_features(
    *, scene: str, cache_path: str, record: Mapping[str, Any], metric_aux: Mapping[str, Any]
) -> Tuple[List[str], List[float], List[bool]]:
    """Context covariates used only for conditional calibration, never labels."""

    current_speed = _current_speed_from_cache(cache_path)
    if current_speed is None:
        current_speed = _optional_float(metric_aux.get("ego_mean_speed_mps"))
    speed_limit = _optional_float(metric_aux.get("route_speed_limit_mps"))
    if speed_limit is None:
        speed_limit = _initial_local_route_speed_limit_from_cache(cache_path)
    curvature = _route_curvature_from_cache(cache_path)

    if scene == "straight_free_drive":
        speed_ratio = None
        if current_speed is not None and speed_limit is not None and speed_limit > 0.5:
            speed_ratio = float(np.clip(current_speed / speed_limit, 0.0, 1.5))
        names = ["initial_speed_over_limit", "route_speed_limit_mps", "route_curvature_1pm"]
        raw = [speed_ratio, speed_limit, curvature]
    elif scene == "straight_car_follow":
        initial_headway, initial_closing, ego_speed_long, lead_speed_long, lead_accel_long = _initial_follow_context_from_cache(cache_path)
        names = [
            "initial_headway_s",
            "initial_closing_speed_mps",
            "initial_ego_long_speed_mps",
            "initial_lead_long_speed_mps",
            "initial_lead_long_accel_mps2",
        ]
        raw = [
            initial_headway,
            initial_closing,
            ego_speed_long,
            lead_speed_long,
            lead_accel_long,
        ]
    else:
        route_lane_count, nearby_vehicle_count = _initial_lane_context_from_cache(cache_path)
        names = [
            "initial_speed_mps",
            "route_curvature_1pm",
            "route_speed_limit_mps",
            "route_lane_count",
            "nearby_vehicle_count_40m",
        ]
        raw = [current_speed, curvature, speed_limit, route_lane_count, nearby_vehicle_count]

    valid = [value is not None and math.isfinite(float(value)) for value in raw]
    # A deterministic neutral fill retains a rectangular matrix.  The validity
    # flags are preserved and reported, so missing context cannot be mistaken
    # for an observed zero-valued context.
    values = [float(value) if ok else 0.0 for value, ok in zip(raw, valid)]
    return names, values, valid


def build_v5_candidates(
    *,
    index_path: str,
    output_dir: str,
    planner_cache_dir: str = "",
    scene_filter: str = "all",
    sample_fraction: float = 1.0,
    sample_seed: int = 20260714,
    min_scene_confidence: float = 0.60,
    require_sample_quality: bool = True,
    min_ego_path_length_m: float = 12.0,
    min_ego_forward_progress_m: float = 8.0,
    max_selected_records: int = 0,
    log_interval: int = 1000,
) -> Dict[str, Any]:
    """Recompute raw behaviour axes from cache for scene-qualified candidates."""

    if not 0.0 < float(sample_fraction) <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1]")
    selected_scenes = set(_scene_choice(scene_filter))
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    candidates: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    rejection: Counter[str] = Counter()

    source = _read_jsonl(index_path)
    for position, raw_record in enumerate(source):
        if log_interval > 0 and (position + 1) % int(log_interval) == 0:
            top_rejections = ", ".join(
                f"{reason}={count}"
                for reason, count in rejection.most_common(4)
            ) or "none"
            print(
                f"[v5-candidates] scanned={position + 1}/{len(source)} "
                f"written={len(candidates)} scenes={dict(counts)} rejects={top_rejections}",
                flush=True,
            )
        record = normalize_split_index_record(raw_record)
        sample_id = str(record.get("sample_id", "") or record.get("filename", ""))
        if not sample_id:
            rejection["missing_sample_id"] += 1
            continue
        # Sampling precedes the scene filter.  A 10% free-driving smoke test is
        # thus a 10% train-set slice, not an arbitrary 10% of a prefiltered set.
        if _stable_fraction(sample_id, sample_seed) >= float(sample_fraction):
            rejection["hash_not_selected"] += 1
            continue
        scene = str(record.get("scene_bucket", "none"))
        if scene not in selected_scenes:
            rejection["scene_not_selected"] += 1
            continue
        if scene not in PRIMARY_SCENES:
            rejection["not_primary_scene"] += 1
            continue
        if _float(record.get("scene_confidence"), 0.0) < float(min_scene_confidence):
            rejection["scene_confidence"] += 1
            continue
        if require_sample_quality and not bool(record.get("sample_quality_valid", False)):
            rejection["sample_quality"] += 1
            continue
        cache_path = _cache_path(record, planner_cache_dir)
        if not cache_path or not Path(cache_path).is_file():
            rejection["cache_missing"] += 1
            continue

        metric_record = dict(record)
        metric_record["planner_cache_path"] = cache_path
        try:
            bundle = build_behavior_metric_bundle(metric_record)
        except (KeyError, OSError, ValueError, IndexError) as exc:
            rejection[f"metric_error:{type(exc).__name__}"] += 1
            continue
        if bundle.metric_source != "cache":
            rejection["metric_not_cache"] += 1
            continue
        path_length = _float(bundle.metric_aux.get("ego_path_length_m"), -1.0)
        forward_progress = _float(bundle.metric_aux.get("ego_forward_progress_m"), -1.0)
        if path_length < float(min_ego_path_length_m):
            rejection["ego_path_length"] += 1
            continue
        if forward_progress < float(min_ego_forward_progress_m):
            rejection["ego_forward_progress"] += 1
            continue

        condition_names, condition_values, condition_valid = _condition_features(
            scene=scene,
            cache_path=cache_path,
            record=record,
            metric_aux=bundle.metric_aux,
        )
        metric = bundle.to_json_dict()
        metric_mask = np.asarray(bundle.axis_valid_mask, dtype=bool)
        # Preserve only explicit upstream causal route metadata.  V5 never
        # fabricates these fields from the future trajectory; V6 later uses
        # them to decide whether lane-change axes may be conditioned online.
        # Keeping this small allow-list avoids accidentally treating arbitrary
        # split-sidecar annotations as causal runtime signals.
        causal_route_metadata = {
            key: record[key]
            for key in (
                "route_lane_change_intent",
                "route_lane_change_intent_available",
                "causal_target_lane_available",
                "route_target_lane_available",
                "runtime_target_lane_available",
                "causal_target_lane_interaction_observable",
                "route_target_lane_interaction_observable",
                "runtime_target_lane_interaction_observable",
            )
            if key in record
        }
        candidate = {
            "schema_version": V5_SCHEMA_VERSION,
            "artifact": "candidate",
            "sample_id": sample_id,
            "filename": str(record.get("filename", "")),
            "planner_cache_path": cache_path,
            "scene_bucket": scene,
            "scene_confidence": _float(record.get("scene_confidence"), 0.0),
            "sample_quality_valid": bool(record.get("sample_quality_valid", False)),
            "motion_quality_valid": True,
            "ego_path_length_m": path_length,
            "ego_forward_progress_m": forward_progress,
            "source_index_path": str(index_path),
            # Retained strictly for paired legacy-baseline analysis.  It is not
            # read by any V5 fitting function.
            "legacy_style_label_audit_only": str(record.get("style_label", "unknown")),
            **metric,
            "condition_names": condition_names,
            "condition_values": condition_values,
            "condition_valid_mask": condition_valid,
            "candidate_any_axis_valid": bool(np.any(metric_mask)),
            "candidate_complete_axis_valid": bool(np.all(metric_mask)),
            **causal_route_metadata,
        }
        candidates.append(candidate)
        counts[scene] += 1
        if max_selected_records > 0 and len(candidates) >= int(max_selected_records):
            break

    candidate_path = target_dir / "v5_candidates.jsonl"
    summary_path = target_dir / "v5_candidates_summary.json"
    _write_jsonl(candidate_path, candidates)
    print(
        f"[v5-candidates] complete scanned={len(source)} written={len(candidates)} "
        f"output={candidate_path}",
        flush=True,
    )
    summary = {
        "schema_version": V5_SCHEMA_VERSION,
        "artifact": "candidate_summary",
        "index_path": str(index_path),
        "planner_cache_dir_override": str(planner_cache_dir),
        "scene_filter": scene_filter,
        "sample_fraction": float(sample_fraction),
        "sample_seed": int(sample_seed),
        "min_scene_confidence": float(min_scene_confidence),
        "require_sample_quality": bool(require_sample_quality),
        "min_ego_path_length_m": float(min_ego_path_length_m),
        "min_ego_forward_progress_m": float(min_ego_forward_progress_m),
        "written": len(candidates),
        "scene_counts": dict(counts),
        "rejection_counts": dict(rejection),
        "candidate_path": str(candidate_path),
        "important": "V5 did not use split_valid, subset_id, memory_eligible, or legacy style_label as a fitting input.",
    }
    _write_json(summary_path, summary)
    return summary


def fit_v5_normalization(
    *, candidate_index_path: str, output_dir: str, scene_filter: str = "all", low_quantile: float = 0.05,
    high_quantile: float = 0.95, min_axis_samples: int = 40
) -> Dict[str, Any]:
    """Fit train-only robust axis normalization and emit canonical directions."""

    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("Need 0 <= low_quantile < high_quantile <= 1")
    selected_scenes = set(_scene_choice(scene_filter))
    records = [record for record in _read_jsonl(candidate_index_path) if record.get("scene_bucket") in selected_scenes]
    stats: Dict[str, Any] = {}
    for scene in selected_scenes:
        scene_records = [record for record in records if record.get("scene_bucket") == scene]
        axes: List[Dict[str, Any]] = []
        for axis in range(3):
            values = [
                _float_vector(record.get("behavior_metric_values"))[axis]
                for record in scene_records
                if _bool_list(record.get("behavior_metric_valid_mask"))[axis]
            ]
            if len(values) < int(min_axis_samples):
                raise RuntimeError(
                    f"{scene} axis {axis} has only {len(values)} valid values; need at least {min_axis_samples}. "
                    "Lower --min-axis-samples only for a smoke test."
                )
            lower, upper = np.quantile(np.asarray(values, dtype=np.float64), [low_quantile, high_quantile])
            if not math.isfinite(float(lower)) or not math.isfinite(float(upper)) or upper - lower <= 1e-6:
                raise RuntimeError(f"Degenerate normalization range for {scene} axis {axis}: [{lower}, {upper}]")
            axes.append({
                "raw_metric": str(record_or_default(scene_records, "behavior_metric_names", axis, axis)),
                "canonical_axis": CANONICAL_AXIS_BY_SCENE[scene][axis],
                "direction": RAW_METRIC_DIRECTION_BY_SCENE[scene][axis],
                "valid_count": len(values),
                "q_low": float(lower),
                "q_high": float(upper),
            })
        stats[scene] = {"axis_stats": axes, "record_count": len(scene_records)}

    normalized: List[Dict[str, Any]] = []
    for record in records:
        scene = str(record["scene_bucket"])
        raw_values = _float_vector(record.get("behavior_metric_values"))
        valid = _bool_list(record.get("behavior_metric_valid_mask"))
        canonical = np.full((3,), 0.5, dtype=np.float64)
        for axis, axis_stats in enumerate(stats[scene]["axis_stats"]):
            if not valid[axis]:
                continue
            lower = float(axis_stats["q_low"])
            upper = float(axis_stats["q_high"])
            value = float(np.clip((raw_values[axis] - lower) / (upper - lower), 0.0, 1.0))
            canonical[axis] = 1.0 - value if axis_stats["direction"] == "falling" else value
        updated = dict(record)
        updated.update({
            "artifact": "normalized",
            "normalization_schema": V5_NAME,
            "canonical_axis_names": list(CANONICAL_AXIS_BY_SCENE[scene]),
            "canonical_style_vec": canonical.tolist(),
            "canonical_axis_valid_mask": valid.tolist(),
        })
        normalized.append(updated)

    target_dir = Path(output_dir)
    normalized_path = target_dir / "v5_normalized.jsonl"
    normalization_path = target_dir / "v5_normalization.json"
    model = {
        "schema_version": V5_SCHEMA_VERSION,
        "artifact": "normalization_model",
        "name": V5_NAME,
        "fit_split": "train_only_input",
        "candidate_index_path": str(candidate_index_path),
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "min_axis_samples": int(min_axis_samples),
        "scenes": stats,
        "normalized_index_path": str(normalized_path),
    }
    _write_jsonl(normalized_path, normalized)
    _write_json(normalization_path, model)
    return model


def apply_v5_normalization(
    *,
    candidate_index_path: str,
    normalization_model_path: str,
    output_dir: str,
    scene_filter: str = "all",
) -> Dict[str, Any]:
    """Apply a frozen train-only V5 normalization model to val/test records.

    This function deliberately has no quantile or minimum-sample arguments:
    validation/test data are transformed with the exact train statistics and
    can never silently refit their own axis scales.
    """

    selected_scenes = set(_scene_choice(scene_filter))
    model = _read_json(normalization_model_path)
    if str(model.get("artifact", "")) != "normalization_model":
        raise ValueError(f"Not a V5 normalization model: {normalization_model_path}")
    model_scenes = dict(model.get("scenes", {}))
    missing_scenes = sorted(scene for scene in selected_scenes if scene not in model_scenes)
    if missing_scenes:
        raise ValueError(f"Frozen train normalization is missing scenes: {missing_scenes}")

    records = [
        record
        for record in _read_jsonl(candidate_index_path)
        if str(record.get("scene_bucket", "")) in selected_scenes
    ]
    if not records:
        raise ValueError("No candidate records match the requested scene filter")

    normalized: List[Dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    for record in records:
        scene = str(record["scene_bucket"])
        axis_stats = list(dict(model_scenes[scene]).get("axis_stats", []))
        if len(axis_stats) != 3:
            raise ValueError(f"Frozen train normalization for {scene} must contain exactly three axes")
        raw_values = _float_vector(record.get("behavior_metric_values"))
        valid = _bool_list(record.get("behavior_metric_valid_mask"))
        canonical = np.full((3,), 0.5, dtype=np.float64)
        for axis, stats in enumerate(axis_stats):
            if not valid[axis]:
                continue
            lower = _optional_float(dict(stats).get("q_low"))
            upper = _optional_float(dict(stats).get("q_high"))
            if lower is None or upper is None or upper - lower <= 1e-6:
                raise ValueError(f"Invalid frozen normalization range for {scene} axis {axis}")
            value = float(np.clip((raw_values[axis] - lower) / (upper - lower), 0.0, 1.0))
            canonical[axis] = 1.0 - value if str(dict(stats).get("direction", "")) == "falling" else value
        updated = dict(record)
        updated.update({
            "artifact": "normalized_frozen_train",
            "normalization_schema": V5_NAME,
            "normalization_fit_split": "frozen_train_only",
            "normalization_model_path": str(normalization_model_path),
            "canonical_axis_names": list(CANONICAL_AXIS_BY_SCENE[scene]),
            "canonical_style_vec": canonical.tolist(),
            "canonical_axis_valid_mask": valid.tolist(),
        })
        normalized.append(updated)
        scene_counts[scene] += 1

    target_dir = Path(output_dir)
    normalized_path = target_dir / "v5_normalized.jsonl"
    application_path = target_dir / "v5_normalization_application.json"
    _write_jsonl(normalized_path, normalized)
    application = {
        "schema_version": V5_SCHEMA_VERSION,
        "artifact": "normalization_application",
        "name": V5_NAME,
        "fit_split": "frozen_train_only",
        "candidate_index_path": str(candidate_index_path),
        "normalization_model_path": str(normalization_model_path),
        "scene_filter": scene_filter,
        "scene_counts": dict(scene_counts),
        "written": len(normalized),
        "normalized_index_path": str(normalized_path),
        "important": "No validation/test statistic was fitted; all axis ranges came from the train model.",
    }
    _write_json(application_path, application)
    return application


def record_or_default(records: Sequence[Mapping[str, Any]], key: str, axis: int, default: Any) -> Any:
    if not records:
        return default
    value = records[0].get(key, [])
    if isinstance(value, (list, tuple)) and axis < len(value):
        return value[axis]
    return default


def _robust_standardize(reference: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]]]:
    median = np.median(reference, axis=0)
    q25, q75 = np.quantile(reference, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-3)
    return (reference - median) / scale, (query - median) / scale, {
        "median": median.tolist(), "iqr": scale.tolist()
    }


def _nearest_neighbours(reference: np.ndarray, query: np.ndarray, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return distances/indices.  sklearn is optional but strongly preferred."""

    if reference.shape[0] == 0:
        raise RuntimeError("No conditional-reference samples available")
    count = min(max(int(n_neighbors), 1), reference.shape[0])
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        model = NearestNeighbors(n_neighbors=count, algorithm="auto", metric="euclidean")
        model.fit(reference)
        distances, indices = model.kneighbors(query, return_distance=True)
        return distances.astype(np.float64), indices.astype(np.int64), "sklearn"
    except ImportError:
        if reference.shape[0] > 30000:
            raise RuntimeError(
                "scikit-learn is required for V5 conditional ranks above 30k reference samples. "
                "Install scikit-learn in the server environment."
            )
        indices = np.empty((query.shape[0], count), dtype=np.int64)
        distances = np.empty((query.shape[0], count), dtype=np.float64)
        chunk = 256
        for start in range(0, query.shape[0], chunk):
            stop = min(start + chunk, query.shape[0])
            distance_sq = np.sum((query[start:stop, None, :] - reference[None, :, :]) ** 2, axis=2)
            local = np.argpartition(distance_sq, kth=count - 1, axis=1)[:, :count]
            local_distance = np.take_along_axis(distance_sq, local, axis=1)
            order = np.argsort(local_distance, axis=1)
            indices[start:stop] = np.take_along_axis(local, order, axis=1)
            distances[start:stop] = np.sqrt(np.take_along_axis(local_distance, order, axis=1))
        return distances, indices, "numpy_fallback"


def _rank_against_neighbours(
    *, query_value: float, reference_values: np.ndarray, reference_valid: np.ndarray,
    neighbour_indices: np.ndarray, neighbour_distances: np.ndarray, requested_k: int
) -> Tuple[float, float, bool]:
    valid_positions = [int(pos) for pos, index in enumerate(neighbour_indices) if reference_valid[int(index)]]
    valid_positions = valid_positions[:max(int(requested_k), 1)]
    if not valid_positions:
        return 0.5, 0.0, False
    positions = np.asarray(valid_positions, dtype=np.int64)
    distances = neighbour_distances[positions]
    local_values = reference_values[neighbour_indices[positions]]
    bandwidth = max(float(distances[-1]), 0.05)
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    effective_n = float(1.0 / max(float(np.sum(weights**2)), 1e-12))
    # Mid-distribution rank for atoms.  In free-drive, a real zero speed
    # response is common; ordinary ``F(x)=P(X<=x)`` would map every zero to the
    # *upper* edge of that atom (often above 0.5) and invert its conservative
    # meaning.  F(x-) + 0.5 P(X=x) keeps ties centred in their mass.
    query = float(query_value)
    tie = np.isclose(local_values, query, rtol=0.0, atol=1e-8)
    percentile = float(np.sum(weights * (local_values < query - 1e-8)) + 0.5 * np.sum(weights * tie))
    return float(np.clip(percentile, 0.0, 1.0)), effective_n, True


def fit_v5_conditional_ranks(
    *, normalized_index_path: str, output_dir: str, scene_filter: str = "all", folds: int = 5,
    neighbours: int = 64, min_effective_neighbours: float = 32.0,
    min_shared_condition_features: int = 3,
    opportunity_candidate_pool_multiplier: int = DEFAULT_OPPORTUNITY_CANDIDATE_POOL_MULTIPLIER,
    seed: int = 20260714,
) -> Dict[str, Any]:
    """Construct out-of-fold conditional percentiles ``u_j=F(z_j|c)``."""

    if folds < 2:
        raise ValueError("folds must be >= 2 for out-of-fold conditional ranks")
    if int(neighbours) < 1:
        raise ValueError("neighbours must be >= 1")
    if int(min_shared_condition_features) < 1:
        raise ValueError("min_shared_condition_features must be >= 1")
    if int(opportunity_candidate_pool_multiplier) < DEFAULT_CONDITIONAL_CANDIDATE_POOL_MULTIPLIER:
        raise ValueError(
            "opportunity_candidate_pool_multiplier must be at least "
            f"{DEFAULT_CONDITIONAL_CANDIDATE_POOL_MULTIPLIER}"
        )
    selected_scenes = set(_scene_choice(scene_filter))
    all_records = _read_jsonl(normalized_index_path)
    outputs: List[Dict[str, Any]] = []
    scene_models: Dict[str, Any] = {}
    backend_counter: Counter[str] = Counter()

    for scene in selected_scenes:
        records = [record for record in all_records if record.get("scene_bucket") == scene]
        if not records:
            continue
        condition_names = list(records[0].get("condition_names", []))
        if not condition_names:
            raise RuntimeError(f"{scene} has no condition features")
        condition = np.asarray([_float_vector(record.get("condition_values"), len(condition_names)) for record in records], dtype=np.float64)
        condition_valid = np.asarray([
            _bool_list(record.get("condition_valid_mask"), len(condition_names)) for record in records
        ], dtype=bool)
        # A missing map field must not become an artificial zero-valued context
        # cluster.  At the same time, requiring every field discards almost all
        # lane-change samples when an optional local speed limit is unavailable.
        # Queries are therefore compared only on their observed dimensions, and
        # every reference must observe those same dimensions.  This is a masked
        # Euclidean kNN distance, not median/zero imputation.
        required_condition_features = min(int(min_shared_condition_features), len(condition_names))
        context_feature_count = np.sum(condition_valid, axis=1)
        context_complete = context_feature_count == len(condition_names)
        context_usable = context_feature_count >= required_condition_features
        canonical = np.asarray([_float_vector(record.get("canonical_style_vec")) for record in records], dtype=np.float64)
        axis_valid = np.asarray([_bool_list(record.get("canonical_axis_valid_mask")) for record in records], dtype=bool)
        folds_for_record = np.asarray([
            _stable_fold(str(record.get("sample_id", index)), int(folds), int(seed)) for index, record in enumerate(records)
        ], dtype=np.int64)
        percentile = np.full_like(canonical, 0.5, dtype=np.float64)
        effective_n = np.zeros_like(canonical, dtype=np.float64)
        rank_valid = np.zeros_like(axis_valid, dtype=bool)

        for fold in range(int(folds)):
            query_mask = (folds_for_record == fold) & context_usable
            reference_fold_mask = folds_for_record != fold
            query_indices = np.where(query_mask)[0]
            if query_indices.size == 0:
                continue
            query_pattern = condition_valid[query_indices]
            unique_patterns = np.unique(query_pattern, axis=0)
            for pattern in unique_patterns:
                dimensions = np.where(pattern)[0]
                if dimensions.size < required_condition_features:
                    continue
                pattern_queries = query_indices[np.all(condition_valid[query_indices] == pattern[None, :], axis=1)]
                reference_mask = reference_fold_mask & np.all(condition_valid[:, dimensions], axis=1)
                reference_indices = np.where(reference_mask)[0]
                if pattern_queries.size == 0 or reference_indices.size == 0:
                    continue
                reference_x, query_x, _ = _robust_standardize(
                    condition[reference_indices][:, dimensions], condition[pattern_queries][:, dimensions]
                )
                base_candidate_count = min(
                    int(neighbours) * DEFAULT_CONDITIONAL_CANDIDATE_POOL_MULTIPLIER,
                    reference_indices.size,
                )
                distances, local_indices, backend = _nearest_neighbours(
                    reference_x, query_x, n_neighbors=base_candidate_count
                )
                backend_counter[backend] += 1
                for local_query, global_query in enumerate(pattern_queries.tolist()):
                    for axis in range(3):
                        if CANONICAL_AXIS_BY_SCENE[scene][axis] in OPPORTUNITY_AXIS_NAMES:
                            continue
                        if not axis_valid[global_query, axis]:
                            continue
                        result, n_eff, available = _rank_against_neighbours(
                            query_value=canonical[global_query, axis],
                            reference_values=canonical[reference_indices, axis],
                            reference_valid=axis_valid[reference_indices, axis],
                            neighbour_indices=local_indices[local_query],
                            neighbour_distances=distances[local_query],
                            requested_k=int(neighbours),
                        )
                        percentile[global_query, axis] = result
                        effective_n[global_query, axis] = n_eff
                        rank_valid[global_query, axis] = bool(available and n_eff >= float(min_effective_neighbours))

                # An opportunity axis must remain missing when the traffic
                # opportunity is absent.  For samples where it is genuinely
                # observed, retrieve a larger *context* candidate pool and
                # select the same nearest-k valid labels inside it.  Keeping
                # this retrieval axis-specific avoids changing the ordinary
                # axes, and avoids a large query-by-reference matrix for all
                # samples on full-scale runs.
                opportunity_axes = [
                    axis for axis, axis_name in enumerate(CANONICAL_AXIS_BY_SCENE[scene])
                    if axis_name in OPPORTUNITY_AXIS_NAMES
                ]
                if not opportunity_axes:
                    continue
                opportunity_query_locals = np.where(
                    np.any(axis_valid[pattern_queries][:, opportunity_axes], axis=1)
                )[0]
                if opportunity_query_locals.size == 0:
                    continue
                opportunity_candidate_count = min(
                    int(neighbours) * int(opportunity_candidate_pool_multiplier),
                    reference_indices.size,
                )
                if opportunity_candidate_count > base_candidate_count:
                    opportunity_distances, opportunity_indices, backend = _nearest_neighbours(
                        reference_x,
                        query_x[opportunity_query_locals],
                        n_neighbors=opportunity_candidate_count,
                    )
                    backend_counter[backend] += 1
                else:
                    opportunity_distances = distances[opportunity_query_locals]
                    opportunity_indices = local_indices[opportunity_query_locals]
                for local_position, local_query in enumerate(opportunity_query_locals.tolist()):
                    global_query = int(pattern_queries[local_query])
                    for axis in opportunity_axes:
                        if not axis_valid[global_query, axis]:
                            continue
                        result, n_eff, available = _rank_against_neighbours(
                            query_value=canonical[global_query, axis],
                            reference_values=canonical[reference_indices, axis],
                            reference_valid=axis_valid[reference_indices, axis],
                            neighbour_indices=opportunity_indices[local_position],
                            neighbour_distances=opportunity_distances[local_position],
                            requested_k=int(neighbours),
                        )
                        percentile[global_query, axis] = result
                        effective_n[global_query, axis] = n_eff
                        rank_valid[global_query, axis] = bool(available and n_eff >= float(min_effective_neighbours))

        # All-train reference is persisted for later val/test application.  It
        # is intentionally separate from the OOF outputs used for validation.
        reference_npz = Path(output_dir) / f"v5_conditional_reference_{scene}.npz"
        reference_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            reference_npz,
            condition_values=condition[context_usable].astype(np.float32),
            condition_valid_mask=condition_valid[context_usable],
            canonical_style_vec=canonical[context_usable].astype(np.float32),
            canonical_axis_valid_mask=axis_valid[context_usable],
            sample_ids=np.asarray([str(records[index].get("sample_id", "")) for index in np.where(context_usable)[0]], dtype="U"),
        )
        for index, record in enumerate(records):
            updated = dict(record)
            updated.update({
                "artifact": "conditional_rank_oof",
                "condition_calibration": {
                    "method": "oof_weighted_knn_empirical_cdf",
                    "fold": int(folds_for_record[index]),
                    "folds": int(folds),
                    "requested_neighbours": int(neighbours),
                    "candidate_pool_multiplier_by_axis": {
                        axis_name: int(opportunity_candidate_pool_multiplier)
                        if axis_name in OPPORTUNITY_AXIS_NAMES
                        else int(DEFAULT_CONDITIONAL_CANDIDATE_POOL_MULTIPLIER)
                        for axis_name in CANONICAL_AXIS_BY_SCENE[scene]
                    },
                    "min_effective_neighbours": float(min_effective_neighbours),
                    "min_shared_condition_features": int(required_condition_features),
                    "observed_condition_feature_count": int(context_feature_count[index]),
                    "context_complete": bool(context_complete[index]),
                    "context_usable": bool(context_usable[index]),
                },
                "conditional_percentile_vec": percentile[index].tolist(),
                "conditional_percentile_valid_mask": rank_valid[index].tolist(),
                "condition_effective_neighbours_vec": effective_n[index].tolist(),
            })
            outputs.append(updated)
        scene_models[scene] = {
            "record_count": len(records),
            "context_complete_count": int(np.sum(context_complete)),
            "context_usable_count": int(np.sum(context_usable)),
            "condition_valid_counts": [int(np.sum(condition_valid[:, axis])) for axis in range(len(condition_names))],
            "min_shared_condition_features": int(required_condition_features),
            "candidate_pool_multiplier_by_axis": {
                axis_name: int(opportunity_candidate_pool_multiplier)
                if axis_name in OPPORTUNITY_AXIS_NAMES
                else int(DEFAULT_CONDITIONAL_CANDIDATE_POOL_MULTIPLIER)
                for axis_name in CANONICAL_AXIS_BY_SCENE[scene]
            },
            "condition_names": condition_names,
            "reference_npz": str(reference_npz),
            "axis_rank_valid_counts": [int(np.sum(rank_valid[:, axis])) for axis in range(3)],
            "effective_neighbours_median": [
                float(np.median(effective_n[rank_valid[:, axis], axis])) if np.any(rank_valid[:, axis]) else 0.0
                for axis in range(3)
            ],
        }

    target_dir = Path(output_dir)
    rank_path = target_dir / "v5_conditional_rank.jsonl"
    model_path = target_dir / "v5_conditional_rank_model.json"
    _write_jsonl(rank_path, outputs)
    model = {
        "schema_version": V5_SCHEMA_VERSION,
        "artifact": "conditional_rank_model",
        "name": V5_NAME,
        "fit_split": "train_only_oof",
        "normalized_index_path": str(normalized_index_path),
        "rank_index_path": str(rank_path),
        "folds": int(folds),
        "neighbours": int(neighbours),
        "base_candidate_pool_multiplier": int(DEFAULT_CONDITIONAL_CANDIDATE_POOL_MULTIPLIER),
        "opportunity_candidate_pool_multiplier": int(opportunity_candidate_pool_multiplier),
        "min_effective_neighbours": float(min_effective_neighbours),
        "min_shared_condition_features": int(min_shared_condition_features),
        "seed": int(seed),
        "knn_backend_calls": dict(backend_counter),
        "scenes": scene_models,
    }
    _write_json(model_path, model)
    return model


def _resolve_frozen_reference_path(model_path: str, reference_path: str) -> Path:
    """Resolve a train reference after moving an artifact directory as a unit."""

    candidate = Path(reference_path)
    if candidate.is_file():
        return candidate
    sibling = Path(model_path).resolve().parent / candidate.name
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        f"Frozen conditional reference was not found at {candidate} or {sibling}"
    )


def apply_v5_conditional_ranks(
    *,
    normalized_index_path: str,
    conditional_rank_model_path: str,
    output_dir: str,
    scene_filter: str = "all",
) -> Dict[str, Any]:
    """Apply train-only conditional CDF references to val/test records.

    Unlike :func:`fit_v5_conditional_ranks`, this stage performs no fold split
    and fits no reference distribution.  Every query is ranked only against
    the all-train NPZ persisted by the training stage.
    """

    selected_scenes = set(_scene_choice(scene_filter))
    model = _read_json(conditional_rank_model_path)
    if str(model.get("artifact", "")) != "conditional_rank_model":
        raise ValueError(f"Not a V5 conditional-rank model: {conditional_rank_model_path}")
    scene_models = dict(model.get("scenes", {}))
    missing_scenes = sorted(scene for scene in selected_scenes if scene not in scene_models)
    if missing_scenes:
        raise ValueError(f"Frozen train conditional-rank model is missing scenes: {missing_scenes}")

    neighbours = int(model.get("neighbours", 64))
    min_effective_neighbours = float(model.get("min_effective_neighbours", 32.0))
    default_shared = int(model.get("min_shared_condition_features", 3))
    default_opportunity_multiplier = int(
        model.get("opportunity_candidate_pool_multiplier", DEFAULT_OPPORTUNITY_CANDIDATE_POOL_MULTIPLIER)
    )
    all_records = _read_jsonl(normalized_index_path)
    outputs: List[Dict[str, Any]] = []
    applied_scenes: Dict[str, Any] = {}
    backend_counter: Counter[str] = Counter()

    for scene in selected_scenes:
        records = [record for record in all_records if str(record.get("scene_bucket", "")) == scene]
        if not records:
            continue
        scene_model = dict(scene_models[scene])
        condition_names = list(scene_model.get("condition_names", []))
        if not condition_names:
            raise ValueError(f"Frozen train conditional-rank model for {scene} has no condition names")
        required_condition_features = int(scene_model.get("min_shared_condition_features", default_shared))
        candidate_multiplier_by_axis = dict(scene_model.get("candidate_pool_multiplier_by_axis", {}))
        reference_path = _resolve_frozen_reference_path(
            conditional_rank_model_path,
            str(scene_model.get("reference_npz", "")),
        )
        with np.load(reference_path, allow_pickle=False) as reference_data:
            reference_condition = np.asarray(reference_data["condition_values"], dtype=np.float64)
            reference_condition_valid = np.asarray(reference_data["condition_valid_mask"], dtype=bool)
            reference_canonical = np.asarray(reference_data["canonical_style_vec"], dtype=np.float64)
            reference_axis_valid = np.asarray(reference_data["canonical_axis_valid_mask"], dtype=bool)
        if reference_condition.ndim != 2 or reference_condition.shape[1] != len(condition_names):
            raise ValueError(f"Frozen train reference for {scene} has an incompatible condition dimension")
        if reference_canonical.ndim != 2 or reference_canonical.shape[1] != 3:
            raise ValueError(f"Frozen train reference for {scene} has an incompatible axis dimension")

        for record in records:
            if list(record.get("condition_names", [])) != condition_names:
                raise ValueError(f"Condition-name mismatch between val/test record and train reference for {scene}")
        query_condition = np.asarray(
            [_float_vector(record.get("condition_values"), len(condition_names)) for record in records],
            dtype=np.float64,
        )
        query_condition_valid = np.asarray(
            [_bool_list(record.get("condition_valid_mask"), len(condition_names)) for record in records],
            dtype=bool,
        )
        query_canonical = np.asarray(
            [_float_vector(record.get("canonical_style_vec")) for record in records], dtype=np.float64
        )
        query_axis_valid = np.asarray(
            [_bool_list(record.get("canonical_axis_valid_mask")) for record in records], dtype=bool
        )
        observed_feature_count = np.sum(query_condition_valid, axis=1)
        context_complete = observed_feature_count == len(condition_names)
        context_usable = observed_feature_count >= required_condition_features
        percentile = np.full_like(query_canonical, 0.5, dtype=np.float64)
        effective_n = np.zeros_like(query_canonical, dtype=np.float64)
        rank_valid = np.zeros_like(query_axis_valid, dtype=bool)

        query_indices = np.where(context_usable)[0]
        if query_indices.size:
            unique_patterns = np.unique(query_condition_valid[query_indices], axis=0)
            for pattern in unique_patterns:
                dimensions = np.where(pattern)[0]
                if dimensions.size < required_condition_features:
                    continue
                pattern_queries = query_indices[
                    np.all(query_condition_valid[query_indices] == pattern[None, :], axis=1)
                ]
                reference_mask = np.all(reference_condition_valid[:, dimensions], axis=1)
                reference_indices = np.where(reference_mask)[0]
                if pattern_queries.size == 0 or reference_indices.size == 0:
                    continue
                reference_x, query_x, _ = _robust_standardize(
                    reference_condition[reference_indices][:, dimensions],
                    query_condition[pattern_queries][:, dimensions],
                )
                base_candidate_count = min(
                    neighbours * DEFAULT_CONDITIONAL_CANDIDATE_POOL_MULTIPLIER,
                    reference_indices.size,
                )
                distances, local_indices, backend = _nearest_neighbours(
                    reference_x, query_x, n_neighbors=base_candidate_count
                )
                backend_counter[backend] += 1
                for local_query, global_query in enumerate(pattern_queries.tolist()):
                    for axis, axis_name in enumerate(CANONICAL_AXIS_BY_SCENE[scene]):
                        if axis_name in OPPORTUNITY_AXIS_NAMES or not query_axis_valid[global_query, axis]:
                            continue
                        result, n_eff, available = _rank_against_neighbours(
                            query_value=query_canonical[global_query, axis],
                            reference_values=reference_canonical[reference_indices, axis],
                            reference_valid=reference_axis_valid[reference_indices, axis],
                            neighbour_indices=local_indices[local_query],
                            neighbour_distances=distances[local_query],
                            requested_k=neighbours,
                        )
                        percentile[global_query, axis] = result
                        effective_n[global_query, axis] = n_eff
                        rank_valid[global_query, axis] = bool(
                            available and n_eff >= min_effective_neighbours
                        )

                opportunity_axes = [
                    axis
                    for axis, axis_name in enumerate(CANONICAL_AXIS_BY_SCENE[scene])
                    if axis_name in OPPORTUNITY_AXIS_NAMES
                ]
                if not opportunity_axes:
                    continue
                opportunity_query_locals = np.where(
                    np.any(query_axis_valid[pattern_queries][:, opportunity_axes], axis=1)
                )[0]
                if opportunity_query_locals.size == 0:
                    continue
                opportunity_multiplier = max(
                    [
                        int(candidate_multiplier_by_axis.get(
                            CANONICAL_AXIS_BY_SCENE[scene][axis], default_opportunity_multiplier
                        ))
                        for axis in opportunity_axes
                    ]
                )
                opportunity_candidate_count = min(
                    neighbours * opportunity_multiplier,
                    reference_indices.size,
                )
                if opportunity_candidate_count > base_candidate_count:
                    opportunity_distances, opportunity_indices, backend = _nearest_neighbours(
                        reference_x,
                        query_x[opportunity_query_locals],
                        n_neighbors=opportunity_candidate_count,
                    )
                    backend_counter[backend] += 1
                else:
                    opportunity_distances = distances[opportunity_query_locals]
                    opportunity_indices = local_indices[opportunity_query_locals]
                for local_position, local_query in enumerate(opportunity_query_locals.tolist()):
                    global_query = int(pattern_queries[local_query])
                    for axis in opportunity_axes:
                        if not query_axis_valid[global_query, axis]:
                            continue
                        result, n_eff, available = _rank_against_neighbours(
                            query_value=query_canonical[global_query, axis],
                            reference_values=reference_canonical[reference_indices, axis],
                            reference_valid=reference_axis_valid[reference_indices, axis],
                            neighbour_indices=opportunity_indices[local_position],
                            neighbour_distances=opportunity_distances[local_position],
                            requested_k=neighbours,
                        )
                        percentile[global_query, axis] = result
                        effective_n[global_query, axis] = n_eff
                        rank_valid[global_query, axis] = bool(
                            available and n_eff >= min_effective_neighbours
                        )

        for index, record in enumerate(records):
            updated = dict(record)
            updated.update({
                "artifact": "conditional_rank_frozen_train",
                "condition_calibration": {
                    "method": "frozen_train_weighted_knn_empirical_cdf",
                    "reference_split": "train_only",
                    "reference_model_path": str(conditional_rank_model_path),
                    "reference_npz": str(reference_path),
                    "requested_neighbours": neighbours,
                    "min_effective_neighbours": min_effective_neighbours,
                    "min_shared_condition_features": required_condition_features,
                    "observed_condition_feature_count": int(observed_feature_count[index]),
                    "context_complete": bool(context_complete[index]),
                    "context_usable": bool(context_usable[index]),
                },
                "conditional_percentile_vec": percentile[index].tolist(),
                "conditional_percentile_valid_mask": rank_valid[index].tolist(),
                "condition_effective_neighbours_vec": effective_n[index].tolist(),
            })
            outputs.append(updated)
        applied_scenes[scene] = {
            "record_count": len(records),
            "reference_count": int(reference_condition.shape[0]),
            "reference_npz": str(reference_path),
            "context_complete_count": int(np.sum(context_complete)),
            "context_usable_count": int(np.sum(context_usable)),
            "axis_rank_valid_counts": [int(np.sum(rank_valid[:, axis])) for axis in range(3)],
        }

    if not outputs:
        raise ValueError("No normalized records match the requested scene filter")
    target_dir = Path(output_dir)
    rank_path = target_dir / "v5_conditional_rank.jsonl"
    application_path = target_dir / "v5_conditional_rank_application.json"
    _write_jsonl(rank_path, outputs)
    application = {
        "schema_version": V5_SCHEMA_VERSION,
        "artifact": "conditional_rank_application",
        "name": V5_NAME,
        "fit_split": "frozen_train_only",
        "normalized_index_path": str(normalized_index_path),
        "conditional_rank_model_path": str(conditional_rank_model_path),
        "rank_index_path": str(rank_path),
        "scene_filter": scene_filter,
        "written": len(outputs),
        "knn_backend_calls": dict(backend_counter),
        "scenes": applied_scenes,
        "important": "No validation/test conditional distribution was fitted; all ranks use train references.",
    }
    _write_json(application_path, application)
    return application

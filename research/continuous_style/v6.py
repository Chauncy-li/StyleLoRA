"""V6 direct-axis conditioning for continuous personalized planning.

V5 deliberately answered a data question: do the three behavior axes in a
scene share one *empirical* latent coordinate?  V6 uses that result as a
diagnostic only.  The training condition is the observed three-axis vector
``u``; an external scalar ``rho`` is an inference-time human interface that
is mapped to a transparent command vector.

The module has no torch dependency.  It exports JSONL sidecars which can be
read by the existing preference-conditioned diffusion dataset through the
``style_value_condition`` field.

Important invariants:

* ``m_label`` says a future-derived label could be measured.  It is never an
  online feasibility signal.
* ``m_causal`` says a car-follow/free-drive axis is applicable from the current
  observable interaction state.  Lane-change is not an online router class.
* A V6 training condition uses ``m_train = m_label & m_causal & agreement``.
  A record with no active training axis gets an all-zero style condition so it
  still contributes to ordinary diffusion training, but not style training.
* At inference rho=0 is a real normal-anchor condition, not an all-zero CFG
  condition.  The all-zero vector remains reserved for classifier-free
  dropout / unconditional prediction.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from research.continuous_style.router import ContinuousSceneRoute, route_scene_from_record
from research.continuous_style.schema import CANONICAL_AXIS_BY_SCENE, SCENE_BUCKET_ORDER
from research.preference_execution.interaction_state.schema import AXIS_GATE_ORDER
from research.style_scene_split.index_utils import (
    load_jsonl_records,
    normalize_split_index_record,
    write_jsonl_records,
)


V6_SCHEMA_VERSION = 6
V6_NAME = "continuous_style_v6_direct_axes"
PRIMARY_SCENES: Tuple[str, ...] = tuple(SCENE_BUCKET_ORDER)
CONTROLLED_SCENES: Tuple[str, ...] = ("straight_free_drive", "straight_car_follow")
RHO_MIN = -1.0
RHO_MAX = 1.0
NORMAL_PERCENTILE = 0.5
DEFAULT_RHO_AMPLITUDE = 0.25

# The order is intentionally persisted in every model manifest.  A V6 model
# must not load a checkpoint with a differently ordered condition vector.
STYLE_CONDITION_LAYOUT: Tuple[str, ...] = (
    "axis_target_0",
    "axis_target_1",
    "axis_target_2",
    "causal_axis_mask_0",
    "causal_axis_mask_1",
    "causal_axis_mask_2",
    "scene_one_hot_free_drive",
    "scene_one_hot_car_follow",
    "scene_one_hot_lane_change",
    "scene_gate_free_drive",
    "scene_gate_car_follow",
    "scene_gate_lane_change",
)


def _json_safe(value: Any) -> Any:
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
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)


def _write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_records(str(target), (_json_safe(record) for record in records))


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return [dict(record) for record in load_jsonl_records(str(path))]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(parsed) if math.isfinite(parsed) else float(default)


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return float(parsed) if math.isfinite(parsed) else None


def _bool_vector(value: Any, length: int = 3) -> np.ndarray:
    values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else []
    return np.asarray([bool(values[index]) if index < len(values) else False for index in range(length)], dtype=bool)


def _float_vector(value: Any, length: int = 3, default: float = 0.0) -> np.ndarray:
    values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else []
    return np.asarray([
        _float(values[index], default) if index < len(values) else float(default)
        for index in range(length)
    ], dtype=np.float64)


def _scene_one_hot(scene_bucket: str) -> np.ndarray:
    output = np.zeros((len(PRIMARY_SCENES),), dtype=np.float64)
    if scene_bucket in PRIMARY_SCENES:
        output[PRIMARY_SCENES.index(scene_bucket)] = 1.0
    return output


def _scene_gate_vector(route: ContinuousSceneRoute | None) -> np.ndarray:
    output = np.zeros((len(PRIMARY_SCENES),), dtype=np.float64)
    if route is None:
        return output
    names = list(route.gate_bundle.scene_gate_names)
    values = np.asarray(route.gate_bundle.scene_gate_values, dtype=np.float64).reshape(-1)
    for index, scene in enumerate(PRIMARY_SCENES):
        if scene in names:
            source = names.index(scene)
            if source < values.shape[0] and math.isfinite(float(values[source])):
                output[index] = max(float(values[source]), 0.0)
    total = float(np.sum(output))
    return output / total if total > 1e-8 else output


def causal_scene_gate_vector(
    *, causal_scene_bucket: str, raw_scene_gate_values: Sequence[float], scene_selection_source: str
) -> np.ndarray:
    """Return two longitudinal applicability gates in the legacy 3-slot layout.

    Only free-drive and car-follow are style-control modes.  The lane-change
    slot is always zero and route intent never overrides these gates.  Their
    sum is intentionally allowed to be below one: the missing mass acts as a
    conservative applicability reduction under lateral/ambiguous interaction,
    not as an online lane-change classification decision.
    """

    del causal_scene_bucket, scene_selection_source
    raw = _float_vector(raw_scene_gate_values, length=len(PRIMARY_SCENES), default=0.0)
    raw = np.clip(raw, 0.0, 1.0)
    raw[PRIMARY_SCENES.index("straight_lane_change")] = 0.0
    return raw


def _validate_scene(scene_bucket: str) -> str:
    scene = str(scene_bucket)
    if scene not in PRIMARY_SCENES:
        raise ValueError(f"Unsupported V6 scene bucket {scene!r}; expected one of {PRIMARY_SCENES}")
    return scene


def _validate_rho(rho: float) -> float:
    value = _optional_float(rho)
    if value is None or value < RHO_MIN - 1e-8 or value > RHO_MAX + 1e-8:
        raise ValueError(f"rho must be finite and in [{RHO_MIN}, {RHO_MAX}], got {rho!r}")
    return float(np.clip(value, RHO_MIN, RHO_MAX))


def _validate_bounds(lower: Sequence[float] | None, upper: Sequence[float] | None) -> Tuple[np.ndarray, np.ndarray]:
    lower_array = _float_vector(lower, default=0.0) if lower is not None else np.zeros((3,), dtype=np.float64)
    upper_array = _float_vector(upper, default=1.0) if upper is not None else np.ones((3,), dtype=np.float64)
    if np.any(~np.isfinite(lower_array)) or np.any(~np.isfinite(upper_array)):
        raise ValueError("style command bounds must be finite")
    lower_array = np.clip(lower_array, 0.0, 1.0)
    upper_array = np.clip(upper_array, 0.0, 1.0)
    if np.any(lower_array > upper_array):
        raise ValueError("each style command lower bound must not exceed its upper bound")
    return lower_array, upper_array


def _validate_amplitude(amplitude: Sequence[float] | float | None) -> np.ndarray:
    if amplitude is None:
        values = np.full((3,), DEFAULT_RHO_AMPLITUDE, dtype=np.float64)
    elif isinstance(amplitude, (int, float, np.generic)):
        values = np.full((3,), _float(amplitude), dtype=np.float64)
    else:
        values = _float_vector(amplitude)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 0.5 + 1e-8):
        raise ValueError("rho amplitudes must be finite and in [0, 0.5]")
    return values


def build_style_condition_vector(
    *,
    axis_target: Sequence[float],
    axis_mask: Sequence[bool],
    scene_bucket: str,
    scene_gate_values: Sequence[float] | None = None,
    disabled_when_empty: bool = True,
) -> np.ndarray:
    """Encode the planner-facing V6 condition.

    The vector always exposes the three target coordinates, their hard masks,
    causal scene identity, and router gate distribution.  If no axis is active
    it is intentionally all zero: this is the ordinary diffusion path, not a
    fake rho=0 example.
    """

    scene = _validate_scene(scene_bucket)
    target = np.clip(_float_vector(axis_target, default=NORMAL_PERCENTILE), 0.0, 1.0)
    mask = _bool_vector(axis_mask)
    target = np.where(mask, target, NORMAL_PERCENTILE)
    if disabled_when_empty and not bool(np.any(mask)):
        return np.zeros((len(STYLE_CONDITION_LAYOUT),), dtype=np.float64)
    gate = np.clip(
        _float_vector(scene_gate_values, length=len(PRIMARY_SCENES), default=0.0),
        0.0,
        1.0,
    )
    return np.concatenate([target, mask.astype(np.float64), _scene_one_hot(scene), gate], axis=0)


@dataclass(frozen=True)
class StyleCommand:
    """Transparent rho command and its causally feasible execution form."""

    scene_bucket: str
    rho_requested: float
    axis_amplitude: np.ndarray
    causal_axis_mask: np.ndarray
    target_desired: np.ndarray
    target_executed: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    scene_gate_values: np.ndarray

    @property
    def enabled(self) -> bool:
        return bool(np.any(self.causal_axis_mask))

    @property
    def preference_reduction_l1(self) -> float:
        # Includes axes intentionally unavailable under the causal mask.  The
        # corresponding mask is emitted separately, so this cannot be mistaken
        # for a safety clipping event.
        return float(np.sum(np.abs(self.target_desired - self.target_executed)))

    @property
    def hard_clipped_axis_mask(self) -> np.ndarray:
        active = self.causal_axis_mask
        return active & (np.abs(self.target_desired - self.target_executed) > 1e-8)

    def style_value_condition(self) -> np.ndarray:
        return build_style_condition_vector(
            axis_target=self.target_executed,
            axis_mask=self.causal_axis_mask,
            scene_bucket=self.scene_bucket,
            scene_gate_values=self.scene_gate_values,
            disabled_when_empty=True,
        )

    def axis_token_inputs(self) -> np.ndarray:
        """Per-axis inputs reserved for the later lightweight attention module."""

        gate = float(np.max(self.scene_gate_values)) if self.scene_gate_values.size else 0.0
        return np.stack(
            [
                self.target_executed,
                self.causal_axis_mask.astype(np.float64),
                self.axis_amplitude,
                np.full((3,), gate, dtype=np.float64),
            ],
            axis=-1,
        )

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "rho_requested": float(self.rho_requested),
            "rho_semantics": "global conservative(-1) to aggressive(+1) user command",
            "axis_amplitude_vec": self.axis_amplitude.tolist(),
            "causal_axis_mask": self.causal_axis_mask.astype(bool).tolist(),
            "target_desired_percentile_vec": self.target_desired.tolist(),
            "target_executed_percentile_vec": self.target_executed.tolist(),
            "lower_bound_percentile_vec": self.lower_bound.tolist(),
            "upper_bound_percentile_vec": self.upper_bound.tolist(),
            "scene_gate_values": self.scene_gate_values.tolist(),
            "style_value_condition": self.style_value_condition().tolist(),
            "axis_token_inputs": self.axis_token_inputs().tolist(),
            "preference_reduction_l1": self.preference_reduction_l1,
            "hard_clipped_axis_mask": self.hard_clipped_axis_mask.astype(bool).tolist(),
            "conditioning_enabled": self.enabled,
        }


def build_rho_style_command(
    *,
    scene_bucket: str,
    rho: float,
    causal_axis_mask: Sequence[bool],
    scene_gate_values: Sequence[float] | None = None,
    amplitude: Sequence[float] | float | None = None,
    lower_bound: Sequence[float] | None = None,
    upper_bound: Sequence[float] | None = None,
) -> StyleCommand:
    """Map a human rho command into a three-axis percentile target.

    The default curve is ``0.5 + 0.25*rho`` on every active axis.  V6 keeps
    this deliberately simple: the data do not get to silently learn a new
    rho meaning.  Context is allowed to mask an axis or clamp it to explicit
    feasible bounds, never to invent an unlogged target mapping.
    """

    scene = _validate_scene(scene_bucket)
    rho_value = _validate_rho(rho)
    mask = _bool_vector(causal_axis_mask)
    amplitudes = _validate_amplitude(amplitude)
    lower, upper = _validate_bounds(lower_bound, upper_bound)
    gate = np.clip(
        _float_vector(scene_gate_values, length=len(PRIMARY_SCENES), default=0.0),
        0.0,
        1.0,
    )

    desired = np.clip(NORMAL_PERCENTILE + rho_value * amplitudes, 0.0, 1.0)
    executed = np.clip(desired, lower, upper)
    executed = np.where(mask, executed, NORMAL_PERCENTILE)
    return StyleCommand(
        scene_bucket=scene,
        rho_requested=rho_value,
        axis_amplitude=amplitudes,
        causal_axis_mask=mask,
        target_desired=desired,
        target_executed=executed,
        lower_bound=lower,
        upper_bound=upper,
        scene_gate_values=gate,
    )


def _route_feature_value(route: ContinuousSceneRoute, name: str, default: float = 0.0) -> float:
    names = list(route.feature_bundle.feature_names)
    values = np.asarray(route.feature_bundle.feature_values, dtype=np.float64).reshape(-1)
    if name not in names:
        return float(default)
    index = names.index(name)
    if index >= values.size or not math.isfinite(float(values[index])):
        return float(default)
    return float(values[index])


def derive_causal_style_state(
    *, route: ContinuousSceneRoute, record: Mapping[str, Any], min_router_confidence: float
) -> Dict[str, Any]:
    """Derive two online longitudinal style-applicability gates.

    The router never emits lane-change.  It chooses a style vocabulary only
    between free-drive and car-follow when one gate is sufficiently strong;
    lateral planning remains entirely with the original diffusion planner.
    No future metric, offline bucket, or route lane-change label is consulted.
    """

    if not 0.0 <= float(min_router_confidence) <= 1.0:
        raise ValueError("min_router_confidence must be in [0, 1]")
    del record
    raw_gate = _scene_gate_vector(route)
    style_gate = causal_scene_gate_vector(
        causal_scene_bucket="none",
        raw_scene_gate_values=raw_gate,
        scene_selection_source="two_longitudinal_applicability_gates",
    )
    controlled_indices = [PRIMARY_SCENES.index(scene) for scene in CONTROLLED_SCENES]
    local_winner = int(np.argmax(style_gate[controlled_indices]))
    scene = CONTROLLED_SCENES[local_winner]
    confidence = float(style_gate[PRIMARY_SCENES.index(scene)])
    router_confident = confidence >= float(min_router_confidence)
    scene_selection_source = "two_longitudinal_applicability_gates"
    mask = np.zeros((3,), dtype=bool)
    if router_confident and scene == "straight_free_drive":
        # The longitudinal axes are causally meaningful in a free-drive state.
        # Fine-grained speed-limit saturation is a later explicit bound, not a
        # hidden data-dependent redefinition of rho.
        mask[:] = True
    elif router_confident and scene == "straight_car_follow":
        lead_present = _route_feature_value(route, "lead_vehicle_present", 0.0) > 0.5
        mask[:] = bool(lead_present)

    return {
        "causal_scene_bucket": scene if router_confident else "none",
        "routed_scene_bucket_before_two_gate_projection": str(route.routed_scene_bucket),
        "scene_selection_source": scene_selection_source,
        "router_confidence": confidence,
        "router_confident": bool(router_confident),
        "causal_axis_mask": mask.tolist(),
        "longitudinal_style_gate_values": style_gate.tolist(),
        "longitudinal_style_gate_mass": float(np.sum(style_gate)),
        "online_gate_policy": "free_drive/car_follow applicability only; no lane_change router class",
        "route_lane_change_intent": bool(route.route_lane_change_intent),
        "route_lane_change_intent_available": bool(route.route_lane_change_intent_available),
        "target_lane_known": False,
        "target_lane_known_available": False,
        "target_lane_known_source": "unused_by_two_gate_policy",
        "target_lane_interaction_observable": False,
        "target_lane_interaction_available": False,
        "target_lane_interaction_source": "unused_by_two_gate_policy",
        "causal_target_lane_policy": "lane-change style disabled; diffusion route/map path remains unchanged",
    }


def _direct_axis_condition(
    *,
    scene_bucket: str,
    label_percentile: np.ndarray,
    training_axis_mask: np.ndarray,
    scene_gate_values: np.ndarray,
) -> np.ndarray:
    values = np.where(training_axis_mask, np.clip(label_percentile, 0.0, 1.0), NORMAL_PERCENTILE)
    return build_style_condition_vector(
        axis_target=values,
        axis_mask=training_axis_mask,
        scene_bucket=scene_bucket,
        scene_gate_values=scene_gate_values,
        disabled_when_empty=True,
    )


def build_v6_direct_axis_conditions(
    *,
    rank_index_path: str,
    output_dir: str,
    base_index_path: str = "",
    scene_filter: str = "all",
    min_router_confidence: float = 0.60,
) -> Dict[str, Any]:
    """Build direct-axis training conditions plus causal runtime command metadata.

    ``rank_index_path`` is the V5 conditional-rank artifact.  V6 intentionally
    does not read ``rho``, ``rho_oof``, or the old scene-constrained-target
    fields, so free-drive/lane-change OOF failure cannot contaminate training.
    """

    if scene_filter == "all":
        selected_scenes = set(PRIMARY_SCENES)
    elif scene_filter in PRIMARY_SCENES:
        selected_scenes = {scene_filter}
    else:
        raise ValueError(f"scene_filter must be all or one of {PRIMARY_SCENES}")
    if not 0.0 <= float(min_router_confidence) <= 1.0:
        raise ValueError("min_router_confidence must be in [0, 1]")

    rank_records = [
        record for record in _read_jsonl(rank_index_path)
        if str(record.get("scene_bucket", "")) in selected_scenes
    ]
    if not rank_records:
        raise ValueError("No conditional-rank records match the requested scene filter")

    if base_index_path:
        rank_by_id: Dict[str, Dict[str, Any]] = {}
        for record in rank_records:
            sample_id = str(record.get("sample_id", record.get("filename", "")) or "")
            if not sample_id or sample_id in rank_by_id:
                raise ValueError(f"Missing or duplicate sample_id in conditional-rank index: {sample_id!r}")
            rank_by_id[sample_id] = dict(record)
        source_records: List[Dict[str, Any]] = []
        seen_base_ids: set[str] = set()
        for raw_base in _read_jsonl(base_index_path):
            base = normalize_split_index_record(raw_base)
            if str(base.get("scene_bucket", "")) not in selected_scenes:
                continue
            sample_id = str(base.get("sample_id", base.get("filename", "")) or "")
            if not sample_id or sample_id in seen_base_ids:
                raise ValueError(f"Missing or duplicate sample_id in base split index: {sample_id!r}")
            seen_base_ids.add(sample_id)
            ranked = rank_by_id.get(sample_id)
            if ranked is None:
                record = dict(base)
                record["_v6_rank_label_available"] = False
            else:
                record = {**base, **ranked}
                record["_v6_rank_label_available"] = True
            source_records.append(record)
        orphan_rank_ids = sorted(set(rank_by_id) - seen_base_ids)
        if orphan_rank_ids:
            preview = orphan_rank_ids[:5]
            raise ValueError(
                f"{len(orphan_rank_ids)} conditional-rank records are absent from the base split index; examples={preview}"
            )
    else:
        source_records = []
        for rank_record in rank_records:
            record = dict(rank_record)
            record["_v6_rank_label_available"] = True
            source_records.append(record)
    if not source_records:
        raise ValueError("No base/rank records match the requested scene filter")

    outputs: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    rejection: Counter[str] = Counter()
    mask_patterns: Counter[str] = Counter()
    label_source_counts: Counter[str] = Counter()
    for raw_record in source_records:
        record = dict(raw_record)
        rank_label_available = bool(record.pop("_v6_rank_label_available", True))
        offline_scene = str(record.get("scene_bucket", "none"))
        sample_id = str(record.get("sample_id", record.get("filename", "")) or "")
        if not sample_id:
            rejection["missing_sample_id"] += 1
            continue
        label_source_counts["ranked" if rank_label_available else "base_passthrough"] += 1
        label_percentile = np.clip(_float_vector(record.get("conditional_percentile_vec")), 0.0, 1.0)
        label_mask = (
            _bool_vector(record.get("conditional_percentile_valid_mask"))
            if rank_label_available
            else np.zeros((3,), dtype=bool)
        )
        # Lane-change and metric-filtered passthrough rows intentionally avoid
        # another cache read: they train only the base diffusion objective.
        if offline_scene == "straight_lane_change" or not rank_label_available:
            causal = {
                "causal_scene_bucket": "none",
                "router_confidence": 0.0,
                "router_confident": False,
                "causal_axis_mask": [False, False, False],
                "scene_selection_source": "base_diffusion_passthrough",
                "online_gate_policy": "style_disabled_for_passthrough_sample",
                "route_lane_change_intent": False,
                "route_lane_change_intent_available": False,
                "target_lane_known": False,
                "target_lane_known_available": False,
                "target_lane_known_source": "unavailable",
                "target_lane_interaction_observable": False,
                "target_lane_interaction_available": False,
                "target_lane_interaction_source": "unavailable",
                "causal_target_lane_policy": "lane-change style disabled; diffusion route/map path remains unchanged",
            }
            route_payload = {
                "router_scene_bucket": "none",
                "router_selected_scene_score": 0.0,
                "router_scene_consistent": False,
                "router_source": "base_diffusion_passthrough",
                "router_note": "offline_lane_change" if offline_scene == "straight_lane_change" else "no_valid_rank_label",
                "scene_gate_values": [0.0, 0.0, 0.0],
            }
        else:
            try:
                route = route_scene_from_record(record)
                causal = derive_causal_style_state(
                    route=route,
                    record=record,
                    min_router_confidence=float(min_router_confidence),
                )
                route_payload = route.to_json_dict()
            except (KeyError, OSError, ValueError, IndexError, TypeError) as exc:
                rejection[f"router_error:{type(exc).__name__}"] += 1
                causal = {
                    "causal_scene_bucket": "none",
                    "router_confidence": 0.0,
                    "router_confident": False,
                    "causal_axis_mask": [False, False, False],
                    "scene_selection_source": "router_failed",
                    "online_gate_policy": "router_failed_style_disabled",
                    "route_lane_change_intent": False,
                    "route_lane_change_intent_available": False,
                    "target_lane_known": False,
                    "target_lane_known_available": False,
                    "target_lane_known_source": "unavailable",
                    "target_lane_interaction_observable": False,
                    "target_lane_interaction_available": False,
                    "target_lane_interaction_source": "unavailable",
                    "causal_target_lane_policy": "router_failed_style_disabled",
                }
                route_payload = {
                    "router_scene_bucket": "none",
                    "router_selected_scene_score": 0.0,
                    "router_scene_consistent": False,
                    "router_source": "v6_router_failed",
                    "router_note": type(exc).__name__,
                    "scene_gate_values": [0.0, 0.0, 0.0],
                }

        causal_scene = str(causal["causal_scene_bucket"])
        raw_scene_gate = _float_vector(route_payload.get("scene_gate_values"), length=len(PRIMARY_SCENES), default=0.0)
        scene_gate = causal_scene_gate_vector(
            causal_scene_bucket=causal_scene,
            raw_scene_gate_values=raw_scene_gate,
            scene_selection_source=str(causal.get("scene_selection_source", "")),
        )
        # Keep every batchable metadata vector at the legacy global gate width,
        # including router-failure records.  V6 global-only conditioning does
        # not consume this vector, but the generic Dataset still collates it.
        axis_gate = _float_vector(
            route_payload.get("axis_gate_values"),
            length=len(AXIS_GATE_ORDER),
            default=0.0,
        )
        causal_mask = _bool_vector(causal["causal_axis_mask"])
        # Offline lane-change samples are retained for ordinary diffusion
        # training but intentionally carry no style condition.  This protects
        # the already-working route/map-driven lateral planner while the style
        # module learns only the two validated longitudinal vocabularies.
        if offline_scene == "straight_lane_change":
            causal["pre_offline_lane_empty_causal_axis_mask"] = causal_mask.astype(bool).tolist()
            causal["causal_axis_mask"] = [False, False, False]
            causal["offline_lane_style_policy"] = "empty_style_condition_keep_base_diffusion_sample"
            causal_mask = np.zeros((3,), dtype=bool)
        router_consistent = bool(
            offline_scene in CONTROLLED_SCENES
            and causal_scene == offline_scene
            and bool(causal["router_confident"])
        )
        train_mask = label_mask & causal_mask & router_consistent
        train_values = np.where(train_mask, label_percentile, NORMAL_PERCENTILE)
        if causal_scene in PRIMARY_SCENES:
            style_value_condition = _direct_axis_condition(
                scene_bucket=causal_scene,
                label_percentile=train_values,
                training_axis_mask=train_mask,
                scene_gate_values=scene_gate,
            )
            normal_command = build_rho_style_command(
                scene_bucket=causal_scene,
                rho=0.0,
                causal_axis_mask=causal_mask,
                scene_gate_values=scene_gate,
            )
        else:
            style_value_condition = np.zeros((len(STYLE_CONDITION_LAYOUT),), dtype=np.float64)
            normal_command = None

        pattern = "".join("1" if value else "0" for value in causal_mask.tolist())
        mask_patterns[f"{causal_scene}:{pattern}"] += 1
        updated = dict(record)
        updated.update({
            "schema_version": V6_SCHEMA_VERSION,
            "artifact": "v6_direct_axis_condition",
            "v6_name": V6_NAME,
            "offline_scene_bucket": offline_scene,
            "causal_scene_bucket": causal_scene,
            "style_axis_names": list(CANONICAL_AXIS_BY_SCENE[offline_scene]),
            "axis_direction_contract": "all percentile axes increase toward more aggressive behavior",
            "direct_axis_label_percentile_vec": label_percentile.tolist(),
            "rank_label_source_available": bool(rank_label_available),
            "m_label": label_mask.astype(bool).tolist(),
            "m_causal": causal_mask.astype(bool).tolist(),
            "m_train": train_mask.astype(bool).tolist(),
            "router_training_agreement": bool(router_consistent),
            "style_direct_train_valid": bool(np.any(train_mask)),
            "style_direct_active_axis_count": int(np.sum(train_mask)),
            "style_value_condition": style_value_condition.tolist(),
            # These top-level fields keep the generic preference-conditioned
            # Dataset/loader well formed under the V6 global-only feature set.
            # They are metadata; m_train inside style_value_condition remains
            # the only direct-axis training mask.
            "scene_gate_values": scene_gate.tolist(),
            "raw_router_scene_gate_values": raw_scene_gate.tolist(),
            "axis_gate_values": axis_gate.tolist(),
            "local_axis_gate_values": causal_mask.astype(np.float64).tolist(),
            "style_condition_layout": list(STYLE_CONDITION_LAYOUT),
            "style_condition_source": "direct_axis_label_u_with_label_and_causal_masks",
            "style_axis_token_inputs": (
                np.stack(
                    [
                        train_values,
                        train_mask.astype(np.float64),
                        causal_mask.astype(np.float64),
                        np.full((3,), float(np.max(scene_gate)) if scene_gate.size else 0.0),
                    ],
                    axis=-1,
                ).tolist()
            ),
            "normal_anchor": normal_command.to_json_dict() if normal_command is not None else {
                "rho_requested": 0.0,
                "conditioning_enabled": False,
                "style_value_condition": np.zeros((len(STYLE_CONDITION_LAYOUT),), dtype=np.float64).tolist(),
            },
            "causal_style_state": causal,
            "v6_router": route_payload,
            "rho_training_policy": (
                "No empirical rho is used for V6 training. Train direct u with m_train; rho is inference-only."
            ),
        })
        outputs.append(updated)
        counts[offline_scene] += 1

    target_dir = Path(output_dir)
    target_path = target_dir / "v6_direct_axis_conditions.jsonl"
    model_path = target_dir / "v6_style_command_spec.json"
    _write_jsonl(target_path, outputs)
    model = {
        "schema_version": V6_SCHEMA_VERSION,
        "artifact": "v6_style_command_spec",
        "name": V6_NAME,
        "rank_index_path": str(rank_index_path),
        "base_index_path": str(base_index_path),
        "direct_axis_condition_index_path": str(target_path),
        "training_condition": "[u_label, m_train, causal_scene_one_hot, causal_scene_gate]",
        "inference_condition": "[p_exec(rho), m_causal, causal_scene_one_hot, causal_scene_gate]",
        "style_condition_layout": list(STYLE_CONDITION_LAYOUT),
        "style_condition_dim": len(STYLE_CONDITION_LAYOUT),
        "rho_range": [RHO_MIN, RHO_MAX],
        "rho_mapping": "p_des_j(rho)=clip(0.5 + 0.25*rho, 0, 1), then explicit causal mask/bounds produce p_exec",
        "default_axis_amplitude": [DEFAULT_RHO_AMPLITUDE] * 3,
        "normal_anchor": "rho=0 uses p_exec=0.5 on active causal axes; it is not the all-zero CFG condition",
        "mask_contract": {
            "m_label": "future-derived metric/rank label available",
            "m_causal": "online car-follow/free-drive applicability; forced empty for offline lane-change samples",
            "m_train": "m_label & m_causal & offline/causal router agreement",
        },
        "online_router_contract": {
            "controlled_modes": list(CONTROLLED_SCENES),
            "lane_change_is_a_router_class": False,
            "gate_layout": "[g_free, g_car_follow, 0] with sum allowed below one",
            "interpretation": "continuous longitudinal style applicability, not a three-class driving-task decision",
        },
        "lane_change_contract": {
            "style_condition": "all zero for offline lane-change training samples",
            "planning": "retain sample and let the original diffusion route/map path learn lateral behavior",
            "forbidden_signal": "offline m_gap, future reached lane, and route-intent labels are not online style-router inputs",
        },
        "scene_counts": dict(counts),
        "label_source_counts": dict(label_source_counts),
        "causal_mask_patterns": dict(mask_patterns),
        "router_rejections": dict(rejection),
        "axis_names": {scene: list(CANONICAL_AXIS_BY_SCENE[scene]) for scene in PRIMARY_SCENES},
    }
    _write_json(model_path, model)
    return {
        "schema_version": V6_SCHEMA_VERSION,
        "artifact": "v6_direct_axis_condition_summary",
        "written": len(outputs),
        "scene_counts": dict(counts),
        "label_source_counts": dict(label_source_counts),
        "causal_mask_patterns": dict(mask_patterns),
        "router_rejections": dict(rejection),
        "direct_axis_condition_path": str(target_path),
        "style_command_spec_path": str(model_path),
    }


def validate_v6_direct_axis_conditions(
    *,
    condition_index_path: str,
    output_dir: str,
    min_active_train_rate: float = 0.0,
) -> Dict[str, Any]:
    """Validate V6 field contracts and label/causal-mask separation."""

    if not 0.0 <= float(min_active_train_rate) <= 1.0:
        raise ValueError("min_active_train_rate must be in [0, 1]")
    records = _read_jsonl(condition_index_path)
    if not records:
        raise ValueError("No V6 direct-axis condition records were found")

    failures: Counter[str] = Counter()
    per_scene: Dict[str, Dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for scene in PRIMARY_SCENES:
        scene_records = [record for record in records if record.get("offline_scene_bucket") == scene]
        if not scene_records:
            continue
        active = 0
        causal_active = 0
        agreement = 0
        ranked_source = 0
        label_counts = np.zeros((3,), dtype=np.int64)
        causal_counts = np.zeros((3,), dtype=np.int64)
        train_counts = np.zeros((3,), dtype=np.int64)
        for record in scene_records:
            sample_id = str(record.get("sample_id", ""))
            if not sample_id or sample_id in seen_ids:
                failures["missing_or_duplicate_sample_id"] += 1
            seen_ids.add(sample_id)
            if list(record.get("style_condition_layout", [])) != list(STYLE_CONDITION_LAYOUT):
                failures["style_condition_layout"] += 1
            label = _bool_vector(record.get("m_label"))
            causal = _bool_vector(record.get("m_causal"))
            train = _bool_vector(record.get("m_train"))
            has_rank_source = bool(record.get("rank_label_source_available", False))
            ranked_source += int(has_rank_source)
            if not has_rank_source and (np.any(label) or np.any(causal) or np.any(train)):
                failures["base_passthrough_sample_has_active_style_mask"] += 1
            if np.any(train & ~label):
                failures["m_train_not_subset_m_label"] += 1
            if np.any(train & ~causal):
                failures["m_train_not_subset_m_causal"] += 1
            if bool(record.get("style_direct_train_valid", False)) != bool(np.any(train)):
                failures["train_valid_flag"] += 1
            vector = _float_vector(record.get("style_value_condition"), len(STYLE_CONDITION_LAYOUT), default=np.nan)
            if vector.shape[0] != len(STYLE_CONDITION_LAYOUT) or np.any(~np.isfinite(vector)):
                failures["style_value_condition_nonfinite_or_wrong_dim"] += 1
            scene_gate = _float_vector(record.get("scene_gate_values"), len(PRIMARY_SCENES), default=np.nan)
            axis_gate = _float_vector(record.get("axis_gate_values"), len(AXIS_GATE_ORDER), default=np.nan)
            local_gate = _float_vector(record.get("local_axis_gate_values"), 3, default=np.nan)
            if np.any(~np.isfinite(scene_gate)):
                failures["scene_gate_values_nonfinite_or_wrong_dim"] += 1
            if np.any(~np.isfinite(axis_gate)):
                failures["axis_gate_values_nonfinite_or_wrong_dim"] += 1
            if np.any(~np.isfinite(local_gate)):
                failures["local_axis_gate_values_nonfinite_or_wrong_dim"] += 1
            if np.any(train):
                if np.all(np.abs(vector) <= 1e-8):
                    failures["active_training_condition_is_zero"] += 1
                if not np.allclose(vector[3:6], train.astype(np.float64), atol=1e-6):
                    failures["training_mask_not_encoded"] += 1
            elif not np.all(np.abs(vector) <= 1e-8):
                failures["inactive_training_condition_not_zero"] += 1

            normal = dict(record.get("normal_anchor", {}))
            normal_vector = _float_vector(normal.get("style_value_condition"), len(STYLE_CONDITION_LAYOUT), default=np.nan)
            if np.any(causal):
                if np.all(np.abs(normal_vector) <= 1e-8):
                    failures["normal_anchor_collapsed_to_cfg_zero"] += 1
                if not np.allclose(normal_vector[:3][causal], NORMAL_PERCENTILE, atol=1e-6):
                    failures["normal_anchor_not_percentile_half"] += 1
                if not np.allclose(normal_vector[3:6], causal.astype(np.float64), atol=1e-6):
                    failures["normal_anchor_mask_not_causal"] += 1
            elif not np.all(np.abs(normal_vector) <= 1e-8):
                failures["disabled_normal_anchor_not_zero"] += 1

            causal_scene = str(record.get("causal_scene_bucket", "none"))
            if scene == "straight_lane_change":
                if np.any(causal):
                    failures["offline_lane_change_m_causal_not_empty"] += 1
                if np.any(train):
                    failures["offline_lane_change_m_train_not_empty"] += 1
                if np.any(np.abs(vector) > 1e-8):
                    failures["offline_lane_change_style_condition_not_zero"] += 1
            if scene_gate.shape[0] == len(PRIMARY_SCENES):
                lane_index = PRIMARY_SCENES.index("straight_lane_change")
                if abs(float(scene_gate[lane_index])) > 1e-8:
                    failures["lane_change_gate_slot_not_zero"] += 1
            if causal_scene not in (*PRIMARY_SCENES, "none"):
                failures["invalid_causal_scene"] += 1
            if causal_scene == "straight_lane_change":
                failures["online_router_emitted_lane_change"] += 1

            label_counts += label.astype(np.int64)
            causal_counts += causal.astype(np.int64)
            train_counts += train.astype(np.int64)
            active += int(np.any(train))
            causal_active += int(np.any(causal))
            agreement += int(bool(record.get("router_training_agreement", False)))
        per_scene[scene] = {
            "record_count": len(scene_records),
            "rank_label_source_rate": float(ranked_source / max(len(scene_records), 1)),
            "m_label_coverage": (label_counts / max(len(scene_records), 1)).tolist(),
            "m_causal_coverage": (causal_counts / max(len(scene_records), 1)).tolist(),
            "m_train_coverage": (train_counts / max(len(scene_records), 1)).tolist(),
            "direct_style_active_rate": float(active / max(len(scene_records), 1)),
            "causal_style_active_rate": float(causal_active / max(len(scene_records), 1)),
            "router_training_agreement_rate": float(agreement / max(len(scene_records), 1)),
            "required_min_active_train_rate": float(min_active_train_rate),
            "active_train_rate_pass": bool(active / max(len(scene_records), 1) >= float(min_active_train_rate)),
        }

    report = {
        "schema_version": V6_SCHEMA_VERSION,
        "artifact": "v6_direct_axis_condition_validation",
        "condition_index_path": str(condition_index_path),
        "style_condition_dim": len(STYLE_CONDITION_LAYOUT),
        "contract_pass": bool(not failures),
        "failure_counts": dict(failures),
        "scenes": per_scene,
        "claim": "validates data/interface contracts only; generated-trajectory controllability must be tested by a post-training rho sweep",
    }
    report_path = Path(output_dir) / "v6_direct_axis_condition_validation.json"
    _write_json(report_path, report)
    return report


def audit_v6_causal_router(*, condition_index_path: str, output_dir: str) -> Dict[str, Any]:
    """Quantify offline buckets versus two longitudinal applicability gates."""

    records = _read_jsonl(condition_index_path)
    if not records:
        raise ValueError("No V6 direct-axis condition records were found")
    confusion: Dict[str, Counter[str]] = {scene: Counter() for scene in PRIMARY_SCENES}
    router_confident: Counter[str] = Counter()
    lane_masks: Counter[str] = Counter()
    scene_selection_source: Counter[str] = Counter()
    agreement = 0
    for record in records:
        offline = str(record.get("offline_scene_bucket", "none"))
        causal = str(record.get("causal_scene_bucket", "none"))
        if offline in confusion:
            confusion[offline][causal] += 1
        causal_state = dict(record.get("causal_style_state", {}))
        scene_selection_source[str(causal_state.get("scene_selection_source", "unknown"))] += 1
        router_confident["yes" if bool(causal_state.get("router_confident", False)) else "no"] += 1
        agreement += int(bool(record.get("router_training_agreement", False)))
        if offline == "straight_lane_change":
            pattern = "".join("1" if item else "0" for item in _bool_vector(record.get("m_causal")).tolist())
            lane_masks[pattern] += 1

    report = {
        "schema_version": V6_SCHEMA_VERSION,
        "artifact": "v6_causal_router_audit",
        "condition_index_path": str(condition_index_path),
        "record_count": len(records),
        "offline_to_causal_confusion": {scene: dict(values) for scene, values in confusion.items()},
        "router_confidence_coverage": dict(router_confident),
        "causal_scene_selection_sources": dict(scene_selection_source),
        "offline_causal_training_agreement_rate": float(agreement / max(len(records), 1)),
        "online_router_contract": {
            "outputs": list(CONTROLLED_SCENES),
            "lane_change_output_enabled": False,
            "gate_layout": "[g_free, g_car_follow, 0]",
        },
        "lane_change": {
            "offline_record_count": int(sum(confusion.get("straight_lane_change", Counter()).values())),
            "causal_axis_mask_patterns": dict(lane_masks),
            "interpretation": (
                "Offline lane-change records stay in base diffusion training and must always carry an empty style condition."
            ),
        },
    }
    report_path = Path(output_dir) / "v6_causal_router_audit.json"
    _write_json(report_path, report)
    return report


def audit_v6_command_support(
    *,
    condition_index_path: str,
    output_dir: str,
    rho_values: Sequence[float] = (-0.8, -0.4, 0.0, 0.4, 0.8),
    min_references: int = 40,
    support_radius: float = 0.30,
    amplitude: float = DEFAULT_RHO_AMPLITUDE,
) -> Dict[str, Any]:
    """Audit whether transparent rho commands remain near observed direct-axis support.

    This is intentionally a data-space audit, not a projection operator.  V6
    first uses conservative interior commands; if this report exposes an
    unsupported endpoint, reduce the declared amplitude before introducing a
    more complex kNN/projector mechanism.
    """

    if int(min_references) < 1:
        raise ValueError("min_references must be >= 1")
    if float(support_radius) <= 0.0:
        raise ValueError("support_radius must be > 0")
    rho_grid = [_validate_rho(value) for value in rho_values]
    records = _read_jsonl(condition_index_path)
    if not records:
        raise ValueError("No V6 direct-axis condition records were found")

    groups: Dict[Tuple[str, Tuple[bool, bool, bool]], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        scene = str(record.get("causal_scene_bucket", "none"))
        mask = _bool_vector(record.get("m_train"))
        if scene not in PRIMARY_SCENES or not np.any(mask):
            continue
        groups[(scene, tuple(bool(value) for value in mask.tolist()))].append(record)

    group_reports: Dict[str, Any] = {}
    for (scene, pattern), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        active = np.asarray(pattern, dtype=bool)
        label_values = np.asarray([
            _float_vector(record.get("direct_axis_label_percentile_vec")) for record in group
        ], dtype=np.float64)[:, active]
        finite = np.all(np.isfinite(label_values), axis=1)
        label_values = label_values[finite]
        key = f"{scene}:" + "".join("1" if value else "0" for value in pattern)
        endpoints: List[Dict[str, Any]] = []
        for rho in rho_grid:
            command = build_rho_style_command(
                scene_bucket=scene,
                rho=rho,
                causal_axis_mask=active,
                amplitude=amplitude,
            )
            target = command.target_executed[active]
            if label_values.shape[0] == 0:
                nearest = None
                within_radius = None
            else:
                distance = np.linalg.norm(label_values - target[None, :], axis=1)
                nearest = float(np.min(distance))
                within_radius = bool(nearest <= float(support_radius))
            endpoints.append({
                "rho": float(rho),
                "target_executed_active_axes": target.tolist(),
                "nearest_training_label_l2": nearest,
                "within_support_radius": within_radius,
            })
        support_pass = label_values.shape[0] >= int(min_references) and all(
            bool(item["within_support_radius"]) for item in endpoints if item["within_support_radius"] is not None
        )
        group_reports[key] = {
            "scene_bucket": scene,
            "m_train_pattern": list(pattern),
            "active_axis_names": [
                axis_name for axis_name, enabled in zip(CANONICAL_AXIS_BY_SCENE[scene], pattern) if enabled
            ],
            "reference_count": int(label_values.shape[0]),
            "min_references": int(min_references),
            "support_radius": float(support_radius),
            "commands": endpoints,
            "support_pass": bool(support_pass),
        }

    report = {
        "schema_version": V6_SCHEMA_VERSION,
        "artifact": "v6_command_support_audit",
        "condition_index_path": str(condition_index_path),
        "rho_values": rho_grid,
        "amplitude": float(amplitude),
        "groups": group_reports,
        "interpretation": "Support is audited in direct conditional-percentile space. A failure asks for a smaller declared command range, not empirical rho fitting.",
    }
    report_path = Path(output_dir) / "v6_command_support_audit.json"
    _write_json(report_path, report)
    return report


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        stop = start + 1
        while stop < values.shape[0] and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 8:
        return None
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    if float(np.std(x_rank)) < 1e-9 or float(np.std(y_rank)) < 1e-9:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _rho_hat_from_axis_response(
    *, generated: np.ndarray, causal_mask: np.ndarray, amplitude: np.ndarray
) -> float | None:
    active = causal_mask & np.isfinite(generated) & (amplitude > 1e-8)
    if not np.any(active):
        return None
    direction = amplitude[active]
    numerator = float(np.dot(generated[active] - NORMAL_PERCENTILE, direction))
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-10:
        return None
    return float(np.clip(numerator / denominator, RHO_MIN, RHO_MAX))


def _rho_hat_from_normal_relative_response(
    *, response: np.ndarray, causal_mask: np.ndarray, amplitude: np.ndarray
) -> float | None:
    active = causal_mask & np.isfinite(response) & (amplitude > 1e-8)
    if not np.any(active):
        return None
    direction = amplitude[active]
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-10:
        return None
    numerator = float(np.dot(response[active], direction))
    return float(np.clip(numerator / denominator, RHO_MIN, RHO_MAX))


def _sweep_rho_from_record(record: Mapping[str, Any]) -> float:
    for key in ("rho_requested", "rho", "command_rho"):
        value = _optional_float(record.get(key))
        if value is not None:
            return _validate_rho(value)
    raise ValueError("generated sweep record needs rho_requested, rho, or command_rho")


def _generated_axis_vector(record: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    values = None
    for key in ("generated_axis_percentile_vec", "axis_percentile_vec", "generated_u_vec"):
        if key in record:
            values = record.get(key)
            break
    if values is None:
        raise ValueError("generated sweep record needs generated_axis_percentile_vec")
    vector = _float_vector(values, default=np.nan)
    valid = np.isfinite(vector) & (vector >= -1e-6) & (vector <= 1.0 + 1e-6)
    if "generated_axis_valid_mask" in record:
        valid &= _bool_vector(record.get("generated_axis_valid_mask"))
    return np.clip(np.where(np.isfinite(vector), vector, NORMAL_PERCENTILE), 0.0, 1.0), valid


def _generated_canonical_axis_vector(
    record: Mapping[str, Any],
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    values = record.get("generated_axis_canonical_vec")
    if values is None:
        return None, None
    vector = _float_vector(values, default=np.nan)
    valid = np.isfinite(vector)
    if "generated_axis_valid_mask" in record:
        valid &= _bool_vector(record.get("generated_axis_valid_mask"))
    return vector, valid


def evaluate_v6_rho_sweep(
    *,
    condition_index_path: str,
    generated_axis_path: str,
    output_dir: str,
    amplitude: float = DEFAULT_RHO_AMPLITUDE,
    normal_relative: bool = False,
) -> Dict[str, Any]:
    """Evaluate post-training continuous control from generated axis values.

    ``generated_axis_path`` is intentionally model-agnostic JSONL.  Every row
    must contain ``sample_id``, a rho field, and
    ``generated_axis_percentile_vec`` measured with the frozen train-only V5
    calibration reference.  Optional ``seed`` makes seed stability and paired
    rho-order tests available.  Trajectory-to-axis extraction remains outside
    this function because it depends on the simulator's future-agent rollout.
    """

    amplitudes = _validate_amplitude(amplitude)
    condition_records = _read_jsonl(condition_index_path)
    condition_by_id: Dict[str, Dict[str, Any]] = {}
    for record in condition_records:
        sample_id = str(record.get("sample_id", ""))
        if sample_id and sample_id not in condition_by_id:
            condition_by_id[sample_id] = record
    if not condition_by_id:
        raise ValueError("V6 condition index contains no sample_id records")

    generated_records = _read_jsonl(generated_axis_path)
    matched: List[Dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for generated in generated_records:
        sample_id = str(generated.get("sample_id", ""))
        condition = condition_by_id.get(sample_id)
        if condition is None:
            skipped["sample_not_in_condition_index"] += 1
            continue
        scene = str(condition.get("causal_scene_bucket", "none"))
        causal_mask = _bool_vector(condition.get("m_causal"))
        if scene not in PRIMARY_SCENES or not np.any(causal_mask):
            skipped["no_active_causal_style_axis"] += 1
            continue
        try:
            rho = _sweep_rho_from_record(generated)
            generated_axis, generated_valid = _generated_axis_vector(generated)
            canonical_axis, canonical_valid = _generated_canonical_axis_vector(
                generated
            )
        except ValueError as exc:
            skipped[f"invalid_generated_record:{type(exc).__name__}"] += 1
            continue
        command = build_rho_style_command(
            scene_bucket=scene,
            rho=rho,
            causal_axis_mask=causal_mask,
            scene_gate_values=condition.get("v6_router", {}).get("scene_gate_values", []),
            amplitude=amplitudes,
        )
        eval_mask = causal_mask & generated_valid
        if not normal_relative and not np.any(eval_mask):
            skipped["no_generated_axis_valid"] += 1
            continue
        rho_hat = (
            None
            if normal_relative
            else _rho_hat_from_axis_response(
                generated=generated_axis,
                causal_mask=eval_mask,
                amplitude=amplitudes,
            )
        )
        matched.append({
            "sample_id": sample_id,
            "seed": str(generated.get("seed", "single")),
            "scene": scene,
            "rho": rho,
            "rho_hat": rho_hat,
            "generated": generated_axis,
            "target": command.target_executed,
            "causal_mask": causal_mask,
            "generated_valid": generated_valid,
            "canonical": canonical_axis,
            "canonical_valid": canonical_valid,
            "eval_mask": eval_mask,
        })

    if normal_relative:
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in matched:
            grouped[(row["sample_id"], row["seed"])].append(row)
        relative_rows: List[Dict[str, Any]] = []
        for group in grouped.values():
            normal_rows = [row for row in group if abs(float(row["rho"])) <= 1e-8]
            if not normal_rows:
                skipped["missing_normal_reference"] += len(group)
                continue
            normal = normal_rows[0]
            fixed_mask = normal["causal_mask"] & normal["generated_valid"]
            if not np.any(fixed_mask):
                skipped["no_normal_axis_valid"] += len(group)
                continue
            for row in group:
                generated_response = row["generated"] - normal["generated"]
                target_response = row["target"] - normal["target"]
                row["eval_mask"] = fixed_mask.copy()
                row["generated_response"] = generated_response
                row["target_response"] = target_response
                if row["canonical"] is not None and normal["canonical"] is not None:
                    row["canonical_response"] = (
                        row["canonical"] - normal["canonical"]
                    )
                    row["canonical_eval_mask"] = (
                        fixed_mask
                        & normal["canonical_valid"]
                        & row["canonical_valid"]
                    )
                row["rho_hat"] = _rho_hat_from_normal_relative_response(
                    response=generated_response,
                    causal_mask=fixed_mask,
                    amplitude=amplitudes,
                )
                relative_rows.append(row)
        matched = relative_rows

    if not matched:
        raise ValueError("No generated sweep rows were eligible after V6 causal-mask matching")

    scene_reports: Dict[str, Any] = {}
    all_intensity_errors: List[float] = []
    for scene in PRIMARY_SCENES:
        rows = [row for row in matched if row["scene"] == scene]
        if not rows:
            continue
        axes: Dict[str, Any] = {}
        for axis, axis_name in enumerate(CANONICAL_AXIS_BY_SCENE[scene]):
            value_key = "generated_response" if normal_relative else "generated"
            target_key = "target_response" if normal_relative else "target"
            values = np.asarray([row[value_key][axis] for row in rows], dtype=np.float64)
            targets = np.asarray([row[target_key][axis] for row in rows], dtype=np.float64)
            rho_values = np.asarray([row["rho"] for row in rows], dtype=np.float64)
            valid = np.asarray([row["eval_mask"][axis] for row in rows], dtype=bool)
            if np.any(valid):
                mae = float(np.mean(np.abs(values[valid] - targets[valid])))
                correlation = _spearman(rho_values[valid], values[valid])
            else:
                mae = None
                correlation = None
            axis_report = {
                "evaluated_count": int(np.sum(valid)),
                "rho_axis_spearman": correlation,
            }
            axis_report[
                "axis_response_mae" if normal_relative else "axis_target_mae"
            ] = mae
            axes[axis_name] = axis_report

        canonical_rows = [
            row
            for row in rows
            if "canonical_response" in row
            and row.get("canonical_eval_mask") is not None
        ]
        canonical_report = None
        if canonical_rows:
            canonical_axes: Dict[str, Any] = {}
            canonical_order_correct = 0
            canonical_order_total = 0
            canonical_direction_cosines: List[float] = []
            for axis, axis_name in enumerate(CANONICAL_AXIS_BY_SCENE[scene]):
                axis_values = np.asarray(
                    [row["canonical_response"][axis] for row in canonical_rows],
                    dtype=np.float64,
                )
                axis_rho = np.asarray(
                    [row["rho"] for row in canonical_rows],
                    dtype=np.float64,
                )
                axis_valid = np.asarray(
                    [row["canonical_eval_mask"][axis] for row in canonical_rows],
                    dtype=bool,
                )
                mean_by_rho = {}
                for rho_value in sorted(set(axis_rho[axis_valid].tolist())):
                    selected = axis_valid & np.isclose(axis_rho, rho_value)
                    if np.any(selected):
                        mean_by_rho[f"{rho_value:g}"] = float(
                            np.mean(axis_values[selected])
                        )
                ordered_means = list(mean_by_rho.values())
                canonical_axes[axis_name] = {
                    "evaluated_count": int(np.sum(axis_valid)),
                    "rho_axis_spearman": (
                        _spearman(axis_rho[axis_valid], axis_values[axis_valid])
                        if np.any(axis_valid)
                        else None
                    ),
                    "mean_response_by_rho": mean_by_rho,
                    "response_span": (
                        float(ordered_means[-1] - ordered_means[0])
                        if len(ordered_means) >= 2
                        else None
                    ),
                }

            canonical_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
            for row in canonical_rows:
                canonical_groups[(row["sample_id"], row["seed"])].append(row)
            for group in canonical_groups.values():
                group = sorted(group, key=lambda row: row["rho"])
                for left in range(len(group)):
                    for right in range(left + 1, len(group)):
                        if group[right]["rho"] <= group[left]["rho"] + 1e-8:
                            continue
                        common = (
                            group[left]["canonical_eval_mask"]
                            & group[right]["canonical_eval_mask"]
                        )
                        if not np.any(common):
                            continue
                        difference = (
                            group[right]["canonical_response"][common]
                            - group[left]["canonical_response"][common]
                        )
                        canonical_order_correct += int(np.sum(difference > 0.0))
                        canonical_order_total += int(difference.size)
                for row in group:
                    if abs(float(row["rho"])) <= 1e-8:
                        continue
                    common = row["canonical_eval_mask"]
                    generated_direction = row["canonical_response"][common]
                    target_direction = row["target_response"][common]
                    denominator = float(
                        np.linalg.norm(generated_direction)
                        * np.linalg.norm(target_direction)
                    )
                    if denominator > 1e-8:
                        canonical_direction_cosines.append(
                            float(
                                np.dot(generated_direction, target_direction)
                                / denominator
                            )
                        )
            saturation_by_axis = {}
            for axis, axis_name in enumerate(CANONICAL_AXIS_BY_SCENE[scene]):
                valid_values = np.asarray(
                    [row["generated"][axis] for row in rows if row["eval_mask"][axis]],
                    dtype=np.float64,
                )
                saturation_by_axis[axis_name] = (
                    float(np.mean((valid_values <= 0.01) | (valid_values >= 0.99)))
                    if valid_values.size
                    else None
                )
            canonical_report = {
                "axis_space": "unclamped_train_normalized_raw_coordinate",
                "axes": canonical_axes,
                "paired_axis_order_accuracy": (
                    float(canonical_order_correct / canonical_order_total)
                    if canonical_order_total
                    else None
                ),
                "paired_axis_order_count": canonical_order_total,
                "direction_cosine_mean": (
                    float(np.mean(canonical_direction_cosines))
                    if canonical_direction_cosines
                    else None
                ),
                "direction_cosine_count": len(canonical_direction_cosines),
                "percentile_saturation_rate_by_axis": saturation_by_axis,
            }

        intensity_rows = [row for row in rows if row["rho_hat"] is not None]
        rho_values = np.asarray([row["rho"] for row in intensity_rows], dtype=np.float64)
        rho_hat_values = np.asarray([float(row["rho_hat"]) for row in intensity_rows], dtype=np.float64)
        intensity_error = np.abs(rho_hat_values - rho_values)
        all_intensity_errors.extend(intensity_error.tolist())
        if rho_values.size >= 2 and float(np.var(rho_values)) > 1e-10:
            slope, intercept = np.polyfit(rho_values, rho_hat_values, deg=1)
            calibration_slope = float(slope)
            calibration_intercept = float(intercept)
        else:
            calibration_slope = None
            calibration_intercept = None

        pair_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in intensity_rows:
            pair_groups[(row["sample_id"], row["seed"])].append(row)
        order_correct = 0
        order_total = 0
        direction_cosines: List[float] = []
        for group in pair_groups.values():
            group = sorted(group, key=lambda row: row["rho"])
            for left in range(len(group)):
                for right in range(left + 1, len(group)):
                    if group[right]["rho"] <= group[left]["rho"] + 1e-8:
                        continue
                    order_total += 1
                    order_correct += int(float(group[right]["rho_hat"]) > float(group[left]["rho_hat"]))
            normal_rows = [row for row in group if abs(float(row["rho"])) <= 1e-8]
            if not normal_rows:
                continue
            normal = normal_rows[0]
            for row in group:
                if abs(float(row["rho"])) <= 1e-8:
                    continue
                common = normal["eval_mask"] & row["eval_mask"]
                if not np.any(common):
                    continue
                delta_generated = row["generated"][common] - normal["generated"][common]
                delta_target = row["target"][common] - normal["target"][common]
                denominator = float(np.linalg.norm(delta_generated) * np.linalg.norm(delta_target))
                if denominator > 1e-8:
                    direction_cosines.append(float(np.dot(delta_generated, delta_target) / denominator))

        seed_groups: Dict[Tuple[str, float], List[float]] = defaultdict(list)
        for row in intensity_rows:
            seed_groups[(row["sample_id"], float(row["rho"]))].append(float(row["rho_hat"]))
        seed_std = [float(np.std(values)) for values in seed_groups.values() if len(values) >= 2]
        scene_reports[scene] = {
            "generated_record_count": len(rows),
            "intensity_eligible_count": len(intensity_rows),
            "axes": axes,
            "intensity": {
                "intensity_mae": float(np.mean(intensity_error)) if intensity_error.size else None,
                "intensity_rmse": float(np.sqrt(np.mean(intensity_error ** 2))) if intensity_error.size else None,
                "rho_hat_spearman": _spearman(rho_values, rho_hat_values),
                "calibration_slope": calibration_slope,
                "calibration_intercept": calibration_intercept,
                "paired_order_accuracy": float(order_correct / order_total) if order_total else None,
                "paired_order_pair_count": int(order_total),
                "direction_cosine_mean": float(np.mean(direction_cosines)) if direction_cosines else None,
                "direction_cosine_count": len(direction_cosines),
                "same_command_seed_rho_hat_std_mean": float(np.mean(seed_std)) if seed_std else None,
                "same_command_seed_group_count": len(seed_std),
            },
        }
        if canonical_report is not None:
            scene_reports[scene]["raw_canonical_response"] = canonical_report

    report = {
        "schema_version": V6_SCHEMA_VERSION,
        "artifact": "v6_post_training_rho_sweep_evaluation",
        "condition_index_path": str(condition_index_path),
        "generated_axis_path": str(generated_axis_path),
        "matched_generated_records": len(matched),
        "skipped_generated_records": dict(skipped),
        "default_amplitude": amplitudes.tolist(),
        "overall_intensity_mae": float(np.mean(all_intensity_errors)) if all_intensity_errors else None,
        "evaluation_origin": (
            "same_sample_rho_zero" if normal_relative else "absolute_percentile_0.5"
        ),
        "scenes": scene_reports,
        "claim": (
            (
                "Evaluates same-sample normal-referenced axis displacement, "
                "scalar intensity calibration, paired monotonicity, vector "
                "direction, and seed stability."
                if normal_relative
                else
                "Evaluates direct-axis target tracking, scalar intensity "
                "calibration, paired monotonicity, vector direction, and seed stability."
            )
            + " It does not require natural data axes to share an empirical rho."
        ),
        "generated_axis_input_contract": {
            "required": ["sample_id", "rho_requested|rho|command_rho", "generated_axis_percentile_vec"],
            "optional": [
                "generated_axis_valid_mask",
                "generated_axis_canonical_vec",
                "seed",
            ],
            "axis_space": "same frozen train-only conditional-percentile calibration as V5/V6",
        },
    }
    report_path = Path(output_dir) / "v6_rho_sweep_evaluation.json"
    _write_json(report_path, report)
    return report


def selftest_v6_style_command() -> Dict[str, Any]:
    """Dependency-light contract test for the command interface."""

    free = build_rho_style_command(
        scene_bucket="straight_free_drive",
        rho=1.0,
        causal_axis_mask=[True, True, True],
        scene_gate_values=[1.0, 0.0, 0.0],
    )
    normal = build_rho_style_command(
        scene_bucket="straight_car_follow",
        rho=0.0,
        causal_axis_mask=[True, True, True],
        scene_gate_values=[0.0, 1.0, 0.0],
    )
    two_gate = causal_scene_gate_vector(
        causal_scene_bucket="none",
        raw_scene_gate_values=[0.25, 0.35, 0.40],
        scene_selection_source="two_longitudinal_applicability_gates",
    )
    checks = {
        "rho_plus_one_maps_to_075": bool(np.allclose(free.target_executed, 0.75)),
        "normal_anchor_is_not_cfg_zero": bool(np.any(np.abs(normal.style_value_condition()) > 1e-8)),
        "condition_dim_matches_manifest": bool(free.style_value_condition().shape[0] == len(STYLE_CONDITION_LAYOUT)),
        "all_axes_are_monotone": bool(np.all(free.target_executed > NORMAL_PERCENTILE)),
        "lane_change_router_slot_is_zero": bool(np.isclose(two_gate[2], 0.0)),
        "two_gate_applicability_mass_is_preserved": bool(np.allclose(two_gate, [0.25, 0.35, 0.0])),
    }
    return {
        "schema_version": V6_SCHEMA_VERSION,
        "artifact": "v6_style_command_selftest",
        "pass": bool(all(checks.values())),
        "checks": checks,
        "style_condition_dim": len(STYLE_CONDITION_LAYOUT),
    }

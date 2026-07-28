"""Structured runtime preference trace export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from research_v1.execution.interaction.schema import AXIS_GATE_ORDER, SCENE_GATE_ORDER
from research_v1.scene_data.schema import style_axis_names_for_scene

CONTROLLED_RUNTIME_SCENES: tuple[str, ...] = (
    "straight_free_drive",
    "straight_car_follow",
)
AXIS_REFERENCE_SUPPORT_REASON_NAMES = {
    0: "enabled",
    1: "scene_or_axis_inactive",
    2: "missing_required_raw_input",
    3: "insufficient_shared_condition_features",
    4: "frozen_reference_unavailable",
    5: "active_traffic_control",
    6: "free_drive_not_clear",
    7: "no_valid_reference_axis",
}
AXIS_REFERENCE_SPEED_LIMIT_SOURCE_NAMES = {
    0: "none",
    1: "route_lane",
    2: "lane_fallback",
}

RUNTIME_CONTEXT_FIELD_ORDER: tuple[str, ...] = (
    "lead_vehicle_present",
    "following_min_gap",
    "following_min_thw",
    "merge_min_gap",
    "merge_lateral_closure",
    "ego_speed_ratio_to_limit",
    "ego_mean_speed",
    "event_speed_drop_ratio",
    "event_brake_peak",
    "ego_brake_peak",
    "ego_lateral_disp",
    "ego_lateral_speed_peak",
    "route_lane_count",
    "nearby_agent_count",
    "ego_progress",
    "ego_heading_change",
    "route_has_control",
)


def _float_list(values: Sequence[object] | None, expected_len: int) -> list[float]:
    if values is None:
        return [0.0] * expected_len
    output: list[float] = []
    for index in range(expected_len):
        try:
            output.append(float(values[index]))  # type: ignore[index]
        except (IndexError, TypeError, ValueError):
            output.append(0.0)
    return output


def _global_from_local(axis_names: Sequence[str], local_values: Sequence[object] | None) -> list[float]:
    global_values = [0.0] * len(AXIS_GATE_ORDER)
    local_values = _float_list(local_values, len(axis_names))
    for axis_name, value in zip(axis_names, local_values):
        if axis_name in AXIS_GATE_ORDER:
            global_values[AXIS_GATE_ORDER.index(axis_name)] = float(value)
    return global_values


def _bool_or_zero(value: object) -> int:
    return int(bool(value))


def _first_bool(value: object) -> bool:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value[0]) if len(value) > 0 else False
    return bool(value)


def _first_int(value: object, default: int = 0) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = value[0] if len(value) > 0 else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _context_dict(debug: Mapping[str, object]) -> dict[str, object]:
    payload = debug.get("runtime_context", {})
    if not isinstance(payload, Mapping):
        return {}
    return dict(payload)


def build_runtime_trace_row(
    *,
    step_index: int,
    iteration_index: int,
    time_us: int,
    debug: Mapping[str, object],
) -> Dict[str, object]:
    scene_bucket = str(debug.get("scene_bucket", "none"))
    scene_axis_names = list(debug.get("scene_axis_names", style_axis_names_for_scene(scene_bucket)))
    if len(scene_axis_names) != 3:
        scene_axis_names = list(style_axis_names_for_scene(scene_bucket))

    scene_gate_names = list(debug.get("scene_gate_names", SCENE_GATE_ORDER))
    if len(scene_gate_names) != len(SCENE_GATE_ORDER):
        scene_gate_names = list(SCENE_GATE_ORDER)

    axis_gate_names = list(debug.get("axis_gate_names", AXIS_GATE_ORDER))
    if len(axis_gate_names) != len(AXIS_GATE_ORDER):
        axis_gate_names = list(AXIS_GATE_ORDER)

    scene_gate_values = _float_list(debug.get("scene_gate_values"), len(scene_gate_names))
    axis_gate_values = _float_list(debug.get("axis_gate_values"), len(axis_gate_names))
    local_axis_gate_values = _float_list(debug.get("local_axis_gate_values"), len(scene_axis_names))

    target_scene = _float_list(debug.get("target_preference_scene_vec"), len(scene_axis_names))
    safe_scene = _float_list(debug.get("safe_preference_scene_vec"), len(scene_axis_names))
    effective_scene = _float_list(debug.get("effective_preference_scene_vec"), len(scene_axis_names))

    target_global = _float_list(debug.get("target_preference_global_vec"), len(axis_gate_names))
    if not any(abs(value) > 1e-8 for value in target_global):
        target_global = _global_from_local(scene_axis_names, target_scene)

    safe_global = _float_list(debug.get("safe_preference_global_vec"), len(axis_gate_names))
    if not any(abs(value) > 1e-8 for value in safe_global):
        safe_global = _global_from_local(scene_axis_names, safe_scene)

    effective_global = _float_list(debug.get("effective_preference_global_vec"), len(axis_gate_names))
    if not any(abs(value) > 1e-8 for value in effective_global):
        effective_global = [safe_global[index] * axis_gate_values[index] for index in range(len(axis_gate_names))]
    temporal_near_target = _float_list(debug.get("temporal_near_target_global_vec"), len(axis_gate_names))
    temporal_far_target = _float_list(debug.get("temporal_far_target_global_vec"), len(axis_gate_names))
    temporal_near_gate_target = _float_list(debug.get("temporal_near_gate_target_vec"), len(axis_gate_names))
    temporal_far_gate_target = _float_list(debug.get("temporal_far_gate_target_vec"), len(axis_gate_names))
    temporal_near_gate = _float_list(debug.get("temporal_near_gate"), len(axis_gate_names))
    temporal_far_gate = _float_list(debug.get("temporal_far_gate"), len(axis_gate_names))
    temporal_near_condition = _float_list(debug.get("temporal_near_condition"), len(axis_gate_names))
    temporal_far_condition = _float_list(debug.get("temporal_far_condition"), len(axis_gate_names))

    scene_axes = {
        axis_name: {
            "local_gate": float(local_axis_gate_values[index]),
            "p_target": float(target_scene[index]),
            "p_safe": float(safe_scene[index]),
            "p_eff": float(effective_scene[index]),
        }
        for index, axis_name in enumerate(scene_axis_names)
    }
    global_axes = {
        axis_name: {
            "axis_gate": float(axis_gate_values[index]),
            "p_target": float(target_global[index]),
            "p_safe": float(safe_global[index]),
            "p_eff": float(effective_global[index]),
        }
        for index, axis_name in enumerate(axis_gate_names)
    }
    scene_gates = {
        scene_name: float(scene_gate_values[index])
        for index, scene_name in enumerate(scene_gate_names)
    }
    context = _context_dict(debug)
    generated_axis_percentile = _float_list(
        debug.get("preference_generated_axis_percentile"),
        len(scene_axis_names),
    )
    generated_raw_axis = _float_list(
        debug.get("preference_generated_raw_axis"),
        len(scene_axis_names),
    )
    generated_axis_valid = [
        bool(value)
        for value in list(
            debug.get(
                "preference_generated_axis_valid_mask",
                [False] * len(scene_axis_names),
            )
        )[: len(scene_axis_names)]
    ]
    while len(generated_axis_valid) < len(scene_axis_names):
        generated_axis_valid.append(False)
    causal_axis_mask = [
        bool(value)
        for value in list(debug.get("causal_axis_mask", [False, False, False]))[
            :3
        ]
    ]
    while len(causal_axis_mask) < 3:
        causal_axis_mask.append(False)
    router_confident = bool(debug.get("router_confident", False))
    style_condition_enabled = bool(
        debug.get("style_condition_enabled", False)
    )
    controlled_router_active = bool(
        scene_bucket in CONTROLLED_RUNTIME_SCENES
        and router_confident
        and style_condition_enabled
        and any(causal_axis_mask)
    )
    axis_reference_support_reason_code = _first_int(
        debug.get("preference_axis_reference_support_reason_code", 1),
        default=1,
    )
    speed_limit_source_code = _first_int(
        debug.get("preference_axis_reference_speed_limit_source_code", 0),
    )
    axis_reference_valid = [
        bool(value)
        for value in list(
            debug.get(
                "preference_axis_reference_valid_axis_mask",
                [False] * len(scene_axis_names),
            )
        )[: len(scene_axis_names)]
    ]
    while len(axis_reference_valid) < len(scene_axis_names):
        axis_reference_valid.append(False)

    return {
        "step_index": int(step_index),
        "iteration_index": int(iteration_index),
        "time_us": int(time_us),
        "command": {
            "style_label": str(debug.get("style_label", "normal")),
            "style_intensity": float(debug.get("style_intensity", 0.0)),
            "condition_field": str(debug.get("condition_field", "")),
        },
        "scene": {
            "scene_bucket": scene_bucket,
            "scene_axis_names": scene_axis_names,
            "condition_density_level": str(debug.get("condition_density_level", "unknown")),
            "condition_speed_regime": str(debug.get("condition_speed_regime", "unknown")),
            "condition_curvature_level": str(debug.get("condition_curvature_level", "unknown")),
            "selected_bucket_level": str(debug.get("selected_bucket_level", "")),
            "selected_bucket_key": str(debug.get("selected_bucket_key", "")),
            "selected_bucket_count": int(debug.get("selected_bucket_count", 0)),
            "dominant_scene_gate_score": float(debug.get("dominant_scene_gate_score", 0.0)),
        },
        "context": context,
        "scene_gates": scene_gates,
        "scene_axes": scene_axes,
        "global_axes": global_axes,
        "temporal_execution": {
            "near_target": {
                axis_name: float(temporal_near_target[index]) for index, axis_name in enumerate(axis_gate_names)
            },
            "far_target": {
                axis_name: float(temporal_far_target[index]) for index, axis_name in enumerate(axis_gate_names)
            },
            "near_gate": {
                axis_name: float(temporal_near_gate[index]) for index, axis_name in enumerate(axis_gate_names)
            },
            "far_gate": {
                axis_name: float(temporal_far_gate[index]) for index, axis_name in enumerate(axis_gate_names)
            },
            "near_gate_target": {
                axis_name: float(temporal_near_gate_target[index]) for index, axis_name in enumerate(axis_gate_names)
            },
            "far_gate_target": {
                axis_name: float(temporal_far_gate_target[index]) for index, axis_name in enumerate(axis_gate_names)
            },
            "near_pref": {
                axis_name: float(temporal_near_condition[index]) for index, axis_name in enumerate(axis_gate_names)
            },
            "far_pref": {
                axis_name: float(temporal_far_condition[index]) for index, axis_name in enumerate(axis_gate_names)
            },
        },
        "v6_execution": {
            "rho_requested": float(debug.get("rho_requested", 0.0)),
            "router_confidence": float(debug.get("router_confidence", 0.0)),
            "router_confident": router_confident,
            "style_condition_enabled": style_condition_enabled,
            "controlled_router_active": controlled_router_active,
            "causal_axis_mask": causal_axis_mask,
            "normal_anchor_cfg_requested": bool(
                debug.get("normal_anchor_cfg_requested", False)
            ),
            "cfg_guidance_scale": float(
                debug.get("cfg_guidance_scale", 1.0)
            ),
            "normal_anchor_cfg_used": _first_bool(
                debug.get("normal_anchor_cfg_used", False)
            ),
            "empty_cfg_reference_used": _first_bool(
                debug.get("empty_cfg_reference_used", False)
            ),
            "generated_axis_percentile_vec": generated_axis_percentile,
            "generated_axis_valid_mask": generated_axis_valid,
            "generated_axis_percentile": {
                axis_name: float(generated_axis_percentile[index])
                for index, axis_name in enumerate(scene_axis_names)
            },
            "generated_raw_axis": {
                axis_name: float(generated_raw_axis[index])
                for index, axis_name in enumerate(scene_axis_names)
            },
            "generated_axis_valid": {
                axis_name: bool(generated_axis_valid[index])
                for index, axis_name in enumerate(scene_axis_names)
            },
            "axis_reference_support": {
                "reason_code": axis_reference_support_reason_code,
                "reason": AXIS_REFERENCE_SUPPORT_REASON_NAMES.get(
                    axis_reference_support_reason_code,
                    "unknown",
                ),
                "shared_condition_count": _first_int(
                    debug.get(
                        "preference_axis_reference_shared_condition_count",
                        0,
                    )
                ),
                "speed_limit_source_code": speed_limit_source_code,
                "speed_limit_source": AXIS_REFERENCE_SPEED_LIMIT_SOURCE_NAMES.get(
                    speed_limit_source_code,
                    "unknown",
                ),
                "speed_limit_valid": _first_bool(
                    debug.get(
                        "preference_axis_reference_speed_limit_valid",
                        False,
                    )
                ),
                "route_curvature_valid": _first_bool(
                    debug.get(
                        "preference_axis_reference_route_curvature_valid",
                        False,
                    )
                ),
                "free_drive_clear": _first_bool(
                    debug.get(
                        "preference_axis_reference_free_drive_clear",
                        False,
                    )
                ),
                "active_traffic_control": _first_bool(
                    debug.get(
                        "preference_axis_reference_active_traffic_control",
                        False,
                    )
                ),
                "reference_valid_axis_mask": axis_reference_valid,
            },
        },
    }


def runtime_trace_csv_fieldnames() -> list[str]:
    fields = [
        "step_index",
        "iteration_index",
        "time_us",
        "style_label",
        "style_intensity",
        "condition_field",
        "scene_bucket",
        "scene_axis_0",
        "scene_axis_1",
        "scene_axis_2",
        "condition_density_level",
        "condition_speed_regime",
        "condition_curvature_level",
        "selected_bucket_level",
        "selected_bucket_key",
        "selected_bucket_count",
        "dominant_scene_gate_score",
        "router_confidence",
        "router_confident",
        "style_condition_enabled",
        "controlled_router_active",
    ]
    fields.extend(RUNTIME_CONTEXT_FIELD_ORDER)
    fields.extend([f"scene_gate__{scene_name}" for scene_name in SCENE_GATE_ORDER])
    fields.extend([f"axis_gate__{axis_name}" for axis_name in AXIS_GATE_ORDER])
    fields.extend([f"p_target__{axis_name}" for axis_name in AXIS_GATE_ORDER])
    fields.extend([f"p_safe__{axis_name}" for axis_name in AXIS_GATE_ORDER])
    fields.extend([f"p_eff__{axis_name}" for axis_name in AXIS_GATE_ORDER])
    fields.extend([f"near_target__{axis_name}" for axis_name in AXIS_GATE_ORDER])
    fields.extend([f"far_target__{axis_name}" for axis_name in AXIS_GATE_ORDER])
    fields.extend([f"near_gate_target__{axis_name}" for axis_name in AXIS_GATE_ORDER])
    fields.extend([f"far_gate_target__{axis_name}" for axis_name in AXIS_GATE_ORDER])
    fields.extend([f"near_gate__{axis_name}" for axis_name in AXIS_GATE_ORDER])
    fields.extend([f"far_gate__{axis_name}" for axis_name in AXIS_GATE_ORDER])
    fields.extend([f"near_pref__{axis_name}" for axis_name in AXIS_GATE_ORDER])
    fields.extend([f"far_pref__{axis_name}" for axis_name in AXIS_GATE_ORDER])
    return fields


def flatten_runtime_trace_row(row: Mapping[str, object]) -> Dict[str, object]:
    command = dict(row.get("command", {})) if isinstance(row.get("command"), Mapping) else {}
    scene = dict(row.get("scene", {})) if isinstance(row.get("scene"), Mapping) else {}
    context = dict(row.get("context", {})) if isinstance(row.get("context"), Mapping) else {}
    scene_gates = dict(row.get("scene_gates", {})) if isinstance(row.get("scene_gates"), Mapping) else {}
    global_axes = dict(row.get("global_axes", {})) if isinstance(row.get("global_axes"), Mapping) else {}
    temporal_execution = (
        dict(row.get("temporal_execution", {})) if isinstance(row.get("temporal_execution"), Mapping) else {}
    )
    v6_execution = (
        dict(row.get("v6_execution", {}))
        if isinstance(row.get("v6_execution"), Mapping)
        else {}
    )
    near_target = dict(temporal_execution.get("near_target", {})) if isinstance(temporal_execution.get("near_target"), Mapping) else {}
    far_target = dict(temporal_execution.get("far_target", {})) if isinstance(temporal_execution.get("far_target"), Mapping) else {}
    near_gate_target = dict(temporal_execution.get("near_gate_target", {})) if isinstance(temporal_execution.get("near_gate_target"), Mapping) else {}
    far_gate_target = dict(temporal_execution.get("far_gate_target", {})) if isinstance(temporal_execution.get("far_gate_target"), Mapping) else {}
    near_gate = dict(temporal_execution.get("near_gate", {})) if isinstance(temporal_execution.get("near_gate"), Mapping) else {}
    far_gate = dict(temporal_execution.get("far_gate", {})) if isinstance(temporal_execution.get("far_gate"), Mapping) else {}
    near_pref = dict(temporal_execution.get("near_pref", {})) if isinstance(temporal_execution.get("near_pref"), Mapping) else {}
    far_pref = dict(temporal_execution.get("far_pref", {})) if isinstance(temporal_execution.get("far_pref"), Mapping) else {}
    scene_axis_names = list(scene.get("scene_axis_names", []))
    while len(scene_axis_names) < 3:
        scene_axis_names.append("")

    flat: Dict[str, object] = {
        "step_index": int(row.get("step_index", 0)),
        "iteration_index": int(row.get("iteration_index", 0)),
        "time_us": int(row.get("time_us", 0)),
        "style_label": str(command.get("style_label", "normal")),
        "style_intensity": float(command.get("style_intensity", 0.0)),
        "condition_field": str(command.get("condition_field", "")),
        "scene_bucket": str(scene.get("scene_bucket", "none")),
        "scene_axis_0": str(scene_axis_names[0]),
        "scene_axis_1": str(scene_axis_names[1]),
        "scene_axis_2": str(scene_axis_names[2]),
        "condition_density_level": str(scene.get("condition_density_level", "unknown")),
        "condition_speed_regime": str(scene.get("condition_speed_regime", "unknown")),
        "condition_curvature_level": str(scene.get("condition_curvature_level", "unknown")),
        "selected_bucket_level": str(scene.get("selected_bucket_level", "")),
        "selected_bucket_key": str(scene.get("selected_bucket_key", "")),
        "selected_bucket_count": int(scene.get("selected_bucket_count", 0)),
        "dominant_scene_gate_score": float(scene.get("dominant_scene_gate_score", 0.0)),
        "router_confidence": float(v6_execution.get("router_confidence", 0.0)),
        "router_confident": _bool_or_zero(
            v6_execution.get("router_confident", False)
        ),
        "style_condition_enabled": _bool_or_zero(
            v6_execution.get("style_condition_enabled", False)
        ),
        "controlled_router_active": _bool_or_zero(
            v6_execution.get("controlled_router_active", False)
        ),
    }
    for key in RUNTIME_CONTEXT_FIELD_ORDER:
        value = context.get(key, "")
        if key in {"lead_vehicle_present", "route_has_control"} and value != "":
            flat[key] = _bool_or_zero(value)
        else:
            flat[key] = value
    for scene_name in SCENE_GATE_ORDER:
        flat[f"scene_gate__{scene_name}"] = float(scene_gates.get(scene_name, 0.0))
    for axis_name in AXIS_GATE_ORDER:
        axis_payload = global_axes.get(axis_name, {})
        if not isinstance(axis_payload, Mapping):
            axis_payload = {}
        flat[f"axis_gate__{axis_name}"] = float(axis_payload.get("axis_gate", 0.0))
        flat[f"p_target__{axis_name}"] = float(axis_payload.get("p_target", 0.0))
        flat[f"p_safe__{axis_name}"] = float(axis_payload.get("p_safe", 0.0))
        flat[f"p_eff__{axis_name}"] = float(axis_payload.get("p_eff", 0.0))
        flat[f"near_target__{axis_name}"] = float(near_target.get(axis_name, 0.0))
        flat[f"far_target__{axis_name}"] = float(far_target.get(axis_name, 0.0))
        flat[f"near_gate_target__{axis_name}"] = float(near_gate_target.get(axis_name, 0.0))
        flat[f"far_gate_target__{axis_name}"] = float(far_gate_target.get(axis_name, 0.0))
        flat[f"near_gate__{axis_name}"] = float(near_gate.get(axis_name, 0.0))
        flat[f"far_gate__{axis_name}"] = float(far_gate.get(axis_name, 0.0))
        flat[f"near_pref__{axis_name}"] = float(near_pref.get(axis_name, 0.0))
        flat[f"far_pref__{axis_name}"] = float(far_pref.get(axis_name, 0.0))
    return flat


def append_runtime_trace_jsonl(path: str | Path, row: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def append_runtime_trace_csv(path: str | Path, row: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = runtime_trace_csv_fieldnames()
    flat_row = flatten_runtime_trace_row(row)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({name: flat_row.get(name, "") for name in fieldnames})

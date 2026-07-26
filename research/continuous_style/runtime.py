"""Runtime adapter for the V6 scalar-rho / direct-axis interface.

The adapter exposes only car-follow/free-drive style applicability.  It never
routes to lane-change; lateral behavior remains with the original map/route
conditioned diffusion planner.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import torch

from research.continuous_style.router import ContinuousSceneRoute
from research.continuous_style.schema import CANONICAL_AXIS_BY_SCENE
from research.continuous_style.v6 import (
    PRIMARY_SCENES,
    STYLE_CONDITION_LAYOUT,
    build_rho_style_command,
    causal_scene_gate_vector,
    derive_causal_style_state,
)
from research.preference_execution.interaction_state.features import build_interaction_state_features
from research.preference_execution.interaction_state.gating import compute_scene_gates
from research.preference_execution.interaction_state.schema import AXIS_GATE_ORDER
from research.preference_execution.runtime.online_preference import OnlinePreferenceConditioner


class ContinuousStyleRuntimeConditioner(OnlinePreferenceConditioner):
    """Convert runtime rho into the exact V6 planner-facing condition layout.

    The class reuses the existing, causal raw-input-to-interaction-state helper
    from ``OnlinePreferenceConditioner`` but does not load legacy prototype or
    projection statistics.  It intentionally supports only V6's global
    explicit layout; attention and normal-anchor CFG are separate later model
    variants rather than hidden behavior inside this adapter.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        self._rho = self._validate_rho(getattr(config, "runtime_rho", 0.0))
        self._normal_anchor_cfg_enabled = bool(
            getattr(config, "normal_anchor_cfg_enabled", False)
        )
        self._cfg_guidance_scale = float(
            getattr(config, "cfg_guidance_scale", 1.0)
        )
        self._preference_energy_guidance_scale = float(
            getattr(config, "preference_energy_guidance_scale", 0.0)
        )
        if not np.isfinite(self._cfg_guidance_scale) or self._cfg_guidance_scale < 0.0:
            raise ValueError("cfg_guidance_scale must be finite and non-negative")
        if (
            not np.isfinite(self._preference_energy_guidance_scale)
            or self._preference_energy_guidance_scale < 0.0
        ):
            raise ValueError(
                "preference_energy_guidance_scale must be finite and non-negative"
            )
        if self._normal_anchor_cfg_enabled and self._cfg_guidance_scale < 1.0:
            raise ValueError(
                "normal-anchor CFG requires cfg_guidance_scale >= 1.0; "
                "the selected Stage-B operating point is 1.1"
            )
        self._route_lane_change_intent = bool(getattr(config, "runtime_route_lane_change_intent", False))
        self._route_lane_change_intent_available = bool(
            getattr(config, "runtime_route_lane_change_intent_available", False)
        )
        self._target_lane_available = bool(getattr(config, "runtime_target_lane_available", False))
        self._target_lane_interaction_observable = bool(
            getattr(config, "runtime_target_lane_interaction_observable", False)
        )
        self._min_router_confidence = float(getattr(config, "runtime_v6_min_router_confidence", 0.60))
        if not 0.0 <= self._min_router_confidence <= 1.0:
            raise ValueError("runtime_v6_min_router_confidence must be in [0, 1]")
        expected_dim = int(getattr(config, "style_value_dim", len(STYLE_CONDITION_LAYOUT)))
        if expected_dim != len(STYLE_CONDITION_LAYOUT):
            raise ValueError(
                "continuous_v6 runtime requires style_value_dim="
                f"{len(STYLE_CONDITION_LAYOUT)}, got {expected_dim}."
            )

    @staticmethod
    def _validate_rho(value: object) -> float:
        try:
            rho = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"runtime rho must be numeric, got {value!r}") from exc
        if not np.isfinite(rho) or rho < -1.0 or rho > 1.0:
            raise ValueError(f"runtime rho must be finite and in [-1, 1], got {value!r}")
        return float(rho)

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def normal_anchor_cfg_enabled(self) -> bool:
        return self._normal_anchor_cfg_enabled

    @property
    def cfg_guidance_scale(self) -> float:
        return self._cfg_guidance_scale

    @property
    def preference_energy_guidance_scale(self) -> float:
        return self._preference_energy_guidance_scale

    def set_command(self, rho: float | None = None, **_: object) -> None:
        if rho is not None:
            self._rho = self._validate_rho(rho)

    def set_route_context(
        self,
        *,
        lane_change_intent: bool | None = None,
        intent_available: bool | None = None,
        target_lane_available: bool | None = None,
        target_lane_interaction_observable: bool | None = None,
    ) -> None:
        """Retain the legacy route-context setter for API compatibility.

        The two-gate V6 policy does not consume these values for style routing;
        the base planner continues to consume route/map inputs independently.
        """

        if lane_change_intent is not None:
            self._route_lane_change_intent = bool(lane_change_intent)
        if intent_available is not None:
            self._route_lane_change_intent_available = bool(intent_available)
        if target_lane_available is not None:
            self._target_lane_available = bool(target_lane_available)
        if target_lane_interaction_observable is not None:
            self._target_lane_interaction_observable = bool(target_lane_interaction_observable)

    def apply(
        self,
        raw_inputs: Mapping[str, Any],
        normalized_inputs: Mapping[str, Any],
        *,
        device: torch.device | str,
    ) -> tuple[dict[str, Any], dict[str, object]]:
        runtime_record = self._build_runtime_record(raw_inputs)
        runtime_record.update(
            {
                "route_lane_change_intent": self._route_lane_change_intent,
                "route_lane_change_intent_available": self._route_lane_change_intent_available,
                "runtime_target_lane_available": self._target_lane_available,
                "runtime_target_lane_interaction_observable": self._target_lane_interaction_observable,
            }
        )
        features = build_interaction_state_features(runtime_record)
        gates = compute_scene_gates(features)
        route = ContinuousSceneRoute(
            observed_scene_bucket="none",
            routed_scene_bucket=str(gates.dominant_scene_gate),
            routed_scene_score=float(gates.dominant_scene_gate_score),
            observed_scene_score=0.0,
            router_source="runtime_observation",
            router_note="raw_inputs_only",
            route_lane_change_intent=self._route_lane_change_intent,
            route_lane_change_intent_available=self._route_lane_change_intent_available,
            feature_bundle=features,
            gate_bundle=gates,
        )
        causal = derive_causal_style_state(
            route=route,
            record=runtime_record,
            min_router_confidence=self._min_router_confidence,
        )
        scene_bucket = str(causal["causal_scene_bucket"])
        raw_gate_values = np.asarray(gates.scene_gate_values, dtype=np.float32)
        gate_values = causal_scene_gate_vector(
            causal_scene_bucket=scene_bucket,
            raw_scene_gate_values=raw_gate_values,
            scene_selection_source=str(causal.get("scene_selection_source", "")),
        ).astype(np.float32)
        axis_gate_values = np.asarray(gates.axis_gate_values, dtype=np.float32)
        mask = np.asarray(causal["causal_axis_mask"], dtype=bool)
        if scene_bucket in PRIMARY_SCENES:
            command = build_rho_style_command(
                scene_bucket=scene_bucket,
                rho=self._rho,
                causal_axis_mask=mask,
                scene_gate_values=gate_values,
            )
            condition = command.style_value_condition()
            normal_anchor = build_rho_style_command(
                scene_bucket=scene_bucket,
                rho=0.0,
                causal_axis_mask=mask,
                scene_gate_values=gate_values,
            ).style_value_condition()
        else:
            command = None
            condition = np.zeros((len(STYLE_CONDITION_LAYOUT),), dtype=np.float32)
            normal_anchor = np.zeros_like(condition)

        model_inputs = dict(normalized_inputs)
        condition_tensor = torch.as_tensor(condition[None, :], dtype=torch.float32, device=device)
        normal_tensor = torch.as_tensor(normal_anchor[None, :], dtype=torch.float32, device=device)
        enabled = bool(np.any(mask)) and command is not None
        model_inputs["style_value_condition"] = condition_tensor
        model_inputs["normal_anchor_style_value_condition"] = normal_tensor
        model_inputs["style_feature_valid"] = torch.as_tensor([enabled], dtype=torch.float32, device=device)
        model_inputs["style_condition_used"] = torch.as_tensor([enabled], dtype=torch.float32, device=device)
        model_inputs["cfg_guidance_scale"] = self._cfg_guidance_scale
        model_inputs["scene_gate_values"] = torch.as_tensor(gate_values[None, :], dtype=torch.float32, device=device)
        model_inputs["axis_gate_values"] = torch.as_tensor(
            axis_gate_values[None, :], dtype=torch.float32, device=device
        )
        model_inputs["local_axis_gate_values"] = torch.as_tensor(
            mask.astype(np.float32)[None, :], dtype=torch.float32, device=device
        )
        model_inputs["causal_axis_mask"] = torch.as_tensor(mask[None, :], dtype=torch.float32, device=device)
        # Frozen-reference preference energy must use physical, unnormalized
        # observations. These keys are consumed only by StylePlanner's optional
        # energy module and never by the encoder or the base diffusion planner.
        raw_energy_keys = {
            "preference_ego_current_state_raw": "ego_current_state",
            "preference_ego_agent_past_raw": "ego_agent_past",
            "preference_neighbor_agents_past_raw": "neighbor_agents_past",
            "preference_neighbor_agents_past_mask_raw": "neighbor_agents_past_mask",
            "preference_lanes_raw": "lanes",
            "preference_lanes_mask_raw": "lanes_mask",
            "preference_lanes_speed_limit_raw": "lanes_speed_limit",
            "preference_lanes_has_speed_limit_raw": "lanes_has_speed_limit",
            "preference_route_lanes_raw": "route_lanes",
            "preference_route_lanes_mask_raw": "route_lanes_mask",
            "preference_route_lanes_speed_limit_raw": "route_lanes_speed_limit",
            "preference_route_lanes_has_speed_limit_raw": "route_lanes_has_speed_limit",
        }
        for output_key, raw_key in raw_energy_keys.items():
            value = raw_inputs.get(raw_key)
            if value is not None:
                model_inputs[output_key] = value.to(device)

        debug: Dict[str, object] = {
            "runtime_style_mode": "continuous_v6",
            "style_label": "continuous_rho",
            "style_intensity": float(self._rho),
            "condition_field": "style_value_condition",
            "rho_requested": float(self._rho),
            "normal_anchor_cfg_requested": self._normal_anchor_cfg_enabled,
            "cfg_guidance_scale": self._cfg_guidance_scale,
            "preference_energy_guidance_scale_requested": (
                self._preference_energy_guidance_scale
            ),
            "scene_bucket": scene_bucket,
            "scene_axis_names": list(CANONICAL_AXIS_BY_SCENE.get(scene_bucket, ("", "", ""))),
            "causal_scene_bucket": scene_bucket,
            "causal_axis_mask": mask.astype(bool).tolist(),
            "router_confidence": float(causal["router_confidence"]),
            "router_confident": bool(causal["router_confident"]),
            "dominant_scene_gate_score": float(gates.dominant_scene_gate_score),
            "scene_gate_values": [float(value) for value in gate_values.tolist()],
            "raw_router_scene_gate_values": [float(value) for value in raw_gate_values.tolist()],
            "axis_gate_names": list(AXIS_GATE_ORDER),
            "axis_gate_values": [float(value) for value in axis_gate_values.tolist()],
            "local_axis_gate_values": mask.astype(np.float32).tolist(),
            "style_condition_enabled": enabled,
            "normal_anchor_is_cfg_zero": bool(np.all(np.abs(normal_anchor) <= 1e-8)),
            "route_lane_change_intent": bool(causal["route_lane_change_intent"]),
            "route_lane_change_intent_available": bool(causal["route_lane_change_intent_available"]),
            "target_lane_known": bool(causal["target_lane_known"]),
            "target_lane_interaction_observable": bool(causal["target_lane_interaction_observable"]),
            "runtime_context": runtime_record,
        }
        if command is not None:
            debug.update(command.to_json_dict())
            debug["target_preference_scene_vec"] = command.target_desired.tolist()
            debug["safe_preference_scene_vec"] = command.target_executed.tolist()
            debug["effective_preference_scene_vec"] = command.target_executed.tolist()
        return model_inputs, debug

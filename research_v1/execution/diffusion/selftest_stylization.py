"""Structural self-test for the V6 StylePlanner model-side extensions."""

from __future__ import annotations

import argparse
import json

import numpy as np

from baseline.model.style_planner.layer.decoder import (
    selftest_signed_router_diffusion_gate,
)
from baseline.model.style_planner.guidance.preference_energy import (
    ConditionalPreferenceEnergy,
)
from baseline.model.style_planner.layer.preference_axis_router import (
    selftest_axis_temporal_kinematic_ego_signed_output_adapter,
    selftest_ego_signed_output_adapter,
    selftest_kinematic_ego_signed_output_adapter,
    selftest_preference_axis_router,
    selftest_scene_axis_temporal_kinematic_ego_signed_output_adapter,
    selftest_signed_preference_axis_router,
)
from research_v1.stylization.commands import build_rho_style_command
from research_v1.execution.diffusion.stylization_losses import (
    selftest_soft_worst_axis_aggregation,
    selftest_short_rollout_dpmpp,
)


class _IdentityStateNormalizer:
    def inverse(self, value):
        return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check V6 axis router, normal-anchor semantics, and frozen "
            "conditional-axis references."
        )
    )
    parser.add_argument("--normalization-path", required=True)
    parser.add_argument("--conditional-rank-model-path", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    legacy_router = selftest_preference_axis_router()
    signed_router = selftest_signed_preference_axis_router()
    ego_adapter = selftest_ego_signed_output_adapter()
    kinematic_ego_adapter = selftest_kinematic_ego_signed_output_adapter()
    axis_temporal_ego_adapter = (
        selftest_axis_temporal_kinematic_ego_signed_output_adapter()
    )
    scene_axis_temporal_ego_adapter = (
        selftest_scene_axis_temporal_kinematic_ego_signed_output_adapter()
    )
    terminal_diffusion_gate = selftest_signed_router_diffusion_gate()
    short_rollout = selftest_short_rollout_dpmpp()
    soft_worst_axis = selftest_soft_worst_axis_aggregation()
    gate = [0.0, 0.85, 0.0]
    mask = [True, True, True]
    conservative = build_rho_style_command(
        scene_bucket="straight_car_follow",
        rho=-0.8,
        causal_axis_mask=mask,
        scene_gate_values=gate,
    ).style_value_condition()
    normal = build_rho_style_command(
        scene_bucket="straight_car_follow",
        rho=0.0,
        causal_axis_mask=mask,
        scene_gate_values=gate,
    ).style_value_condition()
    aggressive = build_rho_style_command(
        scene_bucket="straight_car_follow",
        rho=0.8,
        causal_axis_mask=mask,
        scene_gate_values=gate,
    ).style_value_condition()
    empty = np.zeros_like(normal)

    axis_objective = ConditionalPreferenceEnergy(
        normalization_path=args.normalization_path,
        conditional_rank_model_path=args.conditional_rank_model_path,
        state_normalizer=_IdentityStateNormalizer(),
    )
    report = {
        "legacy_axis_router": legacy_router,
        "signed_axis_router": signed_router,
        "ego_signed_output_adapter": ego_adapter,
        "kinematic_ego_signed_output_adapter": kinematic_ego_adapter,
        "axis_temporal_kinematic_ego_signed_output_adapter": (
            axis_temporal_ego_adapter
        ),
        "scene_axis_temporal_kinematic_ego_signed_output_adapter": (
            scene_axis_temporal_ego_adapter
        ),
        "signed_router_terminal_diffusion_gate": terminal_diffusion_gate,
        "short_rollout_dpmpp": short_rollout,
        "soft_worst_axis_aggregation": soft_worst_axis,
        "condition_contract": {
            "empty_is_zero": bool(np.allclose(empty, 0.0)),
            "normal_is_not_empty": bool(not np.allclose(normal, empty)),
            "conservative_target": conservative[:3].tolist(),
            "normal_target": normal[:3].tolist(),
            "aggressive_target": aggressive[:3].tolist(),
            "ordered_targets": bool(
                np.all(conservative[:3] < normal[:3])
                and np.all(normal[:3] < aggressive[:3])
            ),
        },
        "axis_references": axis_objective.reference_summary(),
        "axis_reference_queries": axis_objective.reference_query_selftest(),
    }
    report["pass"] = bool(
        all(bool(value) for value in signed_router.values())
        and all(bool(value) for value in ego_adapter.values())
        and all(bool(value) for value in kinematic_ego_adapter.values())
        and all(bool(value) for value in axis_temporal_ego_adapter.values())
        and all(bool(value) for value in scene_axis_temporal_ego_adapter.values())
        and all(bool(value) for value in terminal_diffusion_gate.values())
        and all(bool(value) for value in short_rollout.values())
        and all(bool(value) for value in soft_worst_axis.values())
        and report["condition_contract"]["empty_is_zero"]
        and report["condition_contract"]["normal_is_not_empty"]
        and report["condition_contract"]["ordered_targets"]
        and all(
            int(item["reference_count"]) > 0
            for item in report["axis_references"].values()
        )
        and all(
            all(bool(value) for value in item["valid_axes"])
            and all(abs(float(value) - 1.0) < 1e-4 for value in item["weight_sums"])
            for item in report["axis_reference_queries"].values()
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

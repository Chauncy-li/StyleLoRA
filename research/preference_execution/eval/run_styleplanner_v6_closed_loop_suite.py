"""Launch reproducible NuPlan-DB closed-loop sweeps for V6 StylePlanner.

The launcher intentionally keeps DB simulation separate from planner-cache
open-loop controllability evaluation.  It validates every external path,
writes a command manifest, and only starts simulations when ``--execute`` is
provided.  This makes missing DB/map assets visible before a long multi-run
job is submitted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    # Direct execution sets sys.path[0] to this nested script directory. Keep
    # post-simulation repository imports available in the launcher process too.
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
CHECKPOINT_EPOCH_PATTERN = re.compile(r"(?:model|checkpoint)_epoch_(\d+)")
SUPPORTED_VARIANTS = ("router_only", "anchor_cfg")
CONTROLLED_RUNTIME_SCENES = (
    "straight_free_drive",
    "straight_car_follow",
)
REQUIRED_CLOSED_LOOP_CONFIG_KEYS = (
    "time_len",
    "future_len",
    "agent_num",
    "predicted_neighbor_num",
    "static_objects_num",
    "static_objects_state_dim",
    "lane_len",
    "lane_num",
    "route_num",
    "encoder_depth",
    "decoder_depth",
    "num_heads",
    "hidden_dim",
    "encoder_drop_path_rate",
    "decoder_drop_path_rate",
    "diffusion_model_type",
    "state_normalizer",
    "observation_normalizer",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a checkpoint/rho grid through NuPlan DB closed-loop simulation."
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--checkpoint-epochs", default="10,17,20,24")
    parser.add_argument("--checkpoint-paths", nargs="*", default=None)
    parser.add_argument("--rho-values", default="-0.8,0,0.8")
    parser.add_argument(
        "--variants",
        default="router_only,anchor_cfg",
        help="Comma-separated variants: router_only,anchor_cfg.",
    )
    parser.add_argument(
        "--normal-anchor-cfg-scale",
        type=float,
        default=1.1,
        help="Fixed Normal-Anchor CFG scale used by anchor_cfg.",
    )
    parser.add_argument(
        "--require-a3-8-terminal-executor",
        action="store_true",
        help="Fail preflight unless args.json is the A3.8 signed terminal executor contract.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--db-files",
        required=True,
        help="NuPlan .db file or directory containing the DB split to simulate.",
    )
    parser.add_argument("--maps-root", required=True)
    parser.add_argument("--nuplan-devkit-root", default=str(DEFAULT_DEVKIT_ROOT))
    parser.add_argument("--scenario-filter", default="val14")
    parser.add_argument("--scenario-builder", default="nuplan")
    parser.add_argument("--challenge", default="closed_loop_nonreactive_agents")
    parser.add_argument("--limit-total-scenarios", type=int, default=0)
    parser.add_argument(
        "--log-names-json",
        default="",
        help="Optional JSON list of DB log names. Use this for a fixed Boston-only cohort.",
    )
    parser.add_argument(
        "--scenario-tokens-json",
        default="",
        help=(
            "Optional JSON list of exact NuPlan scenario tokens. When supplied, "
            "the launcher overrides limit_total_scenarios to the token count."
        ),
    )
    parser.add_argument(
        "--scenario-types",
        default="",
        help=(
            "Optional comma-separated NuPlan scenario types used only as a "
            "coarse candidate filter; Router eligibility remains the final gate."
        ),
    )
    parser.add_argument(
        "--num-scenarios-per-type",
        type=int,
        default=0,
        help="Optional per-NuPlan-type candidate cap; 0 keeps the filter preset value.",
    )
    parser.add_argument(
        "--map-names",
        default="",
        help="Optional comma-separated NuPlan map filter, e.g. us-ma-boston.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--worker", default="sequential")
    parser.add_argument("--run-tag", default="")
    parser.add_argument(
        "--router-audit-window-steps",
        type=int,
        default=20,
        help="Number of initial closed-loop trace steps used to classify Router eligibility; 0 uses all steps.",
    )
    parser.add_argument(
        "--min-router-active-steps",
        type=int,
        default=10,
        help="Minimum controlled Router-active steps inside the audit window.",
    )
    parser.add_argument(
        "--min-router-active-ratio",
        type=float,
        default=0.50,
        help="Minimum controlled Router-active fraction inside the audit window.",
    )
    parser.add_argument(
        "--min-router-scene-purity",
        type=float,
        default=0.80,
        help="Minimum dominant controlled-scene fraction among active Router steps.",
    )
    parser.add_argument(
        "--require-router-eligible-scenarios",
        action="store_true",
        help="Fail a formal job unless every simulated scenario passes Router eligibility.",
    )
    parser.add_argument(
        "--required-controlled-scenes",
        default="",
        help="Optional comma-separated controlled scenes required in every job cohort.",
    )
    parser.add_argument(
        "--export-router-eligible-tokens-json",
        default="",
        help=(
            "Discovery mode output: export a balanced exact-token JSON list "
            "from one router_only/rho=0 job."
        ),
    )
    parser.add_argument(
        "--eligible-scenarios-per-controlled-scene",
        type=int,
        default=4,
        help="Balanced token count per controlled scene in discovery export.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--run-custom-metrics", action="store_true")
    return parser


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)


def _parse_int_list(raw: str) -> list[int]:
    values = [int(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("--checkpoint-epochs must contain positive integers")
    return values


def _parse_rhos(raw: str) -> list[float]:
    values = [float(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not values or any(value < -1.0 or value > 1.0 for value in values):
        raise ValueError("--rho-values must be in [-1, 1]")
    return values


def _parse_variants(raw: str) -> list[str]:
    variants = [value.strip() for value in str(raw).split(",") if value.strip()]
    if not variants:
        raise ValueError("--variants must contain at least one variant")
    unknown = [value for value in variants if value not in SUPPORTED_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unsupported --variants={unknown}; allowed values are {list(SUPPORTED_VARIANTS)}"
        )
    if len(set(variants)) != len(variants):
        raise ValueError("--variants must not contain duplicates")
    return variants


def _validate_guidance_controls(args: argparse.Namespace) -> None:
    scale = float(args.normal_anchor_cfg_scale)
    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError("--normal-anchor-cfg-scale must be finite and >= 1.0")
    if int(args.router_audit_window_steps) < 0:
        raise ValueError("--router-audit-window-steps must be >= 0")
    if int(args.min_router_active_steps) < 1:
        raise ValueError("--min-router-active-steps must be >= 1")
    for name in (
        "min_router_active_ratio",
        "min_router_scene_purity",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if int(args.eligible_scenarios_per_controlled_scene) < 1:
        raise ValueError(
            "--eligible-scenarios-per-controlled-scene must be >= 1"
        )
    if int(args.num_scenarios_per_type) < 0:
        raise ValueError("--num-scenarios-per-type must be >= 0")
    if (
        args.require_router_eligible_scenarios
        or str(args.export_router_eligible_tokens_json).strip()
    ) and str(args.worker) != "sequential":
        raise ValueError(
            "Router scenario identity auditing requires --worker sequential"
        )
    required_scenes = _parse_names(args.required_controlled_scenes)
    unknown_scenes = [
        scene
        for scene in required_scenes
        if scene not in CONTROLLED_RUNTIME_SCENES
    ]
    if unknown_scenes:
        raise ValueError(
            "--required-controlled-scenes contains unsupported values "
            f"{unknown_scenes}; allowed={list(CONTROLLED_RUNTIME_SCENES)}"
        )


def _variant_contract(
    variant: str,
    *,
    normal_anchor_cfg_scale: float,
) -> Dict[str, Any]:
    if variant == "router_only":
        normal_anchor_enabled = False
        cfg_scale = 1.0
    elif variant == "anchor_cfg":
        normal_anchor_enabled = True
        cfg_scale = float(normal_anchor_cfg_scale)
    else:
        raise ValueError(f"Unsupported guidance variant: {variant!r}")
    return {
        "variant": variant,
        "normal_anchor_cfg_enabled": normal_anchor_enabled,
        "cfg_guidance_scale": cfg_scale,
    }


def _variant_tag(contract: Dict[str, Any]) -> str:
    if not bool(contract["normal_anchor_cfg_enabled"]):
        return "router_only"
    scale = f"{float(contract['cfg_guidance_scale']):g}".replace("-", "m").replace(".", "p")
    return f"anchor_cfg_s{scale}"


def _trace_row_router_scene(row: Mapping[str, Any]) -> str:
    scene_payload = row.get("scene", {})
    if not isinstance(scene_payload, Mapping):
        scene_payload = {}
    scene_bucket = str(scene_payload.get("scene_bucket", "none"))
    if scene_bucket not in CONTROLLED_RUNTIME_SCENES:
        return "none"

    execution = row.get("v6_execution", {})
    if not isinstance(execution, Mapping):
        execution = {}
    explicit_active = execution.get("controlled_router_active")
    if explicit_active is not None:
        return scene_bucket if bool(explicit_active) else "none"

    # Compatibility with traces produced before controlled_router_active was
    # exported. The causal scene is already confidence-gated; an active local
    # axis mask is the remaining runtime condition-enablement requirement.
    scene_axes = row.get("scene_axes", {})
    if not isinstance(scene_axes, Mapping):
        scene_axes = {}
    local_gate_active = any(
        isinstance(payload, Mapping)
        and float(payload.get("local_gate", 0.0)) > 0.5
        for payload in scene_axes.values()
    )
    return scene_bucket if local_gate_active else "none"


def _segment_router_trace_rows(
    trace_rows_by_path: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]],
    *,
    audit_window_steps: int,
    min_active_steps: int,
    min_active_ratio: float,
    min_scene_purity: float,
) -> list[Dict[str, Any]]:
    raw_segments: list[tuple[Path, list[Mapping[str, Any]]]] = []
    for trace_path, rows in trace_rows_by_path:
        current: list[Mapping[str, Any]] = []
        previous_step: int | None = None
        for row in rows:
            step_index = int(row.get("step_index", 0))
            if current and previous_step is not None and step_index <= previous_step:
                raw_segments.append((trace_path, current))
                current = []
            current.append(row)
            previous_step = step_index
        if current:
            raw_segments.append((trace_path, current))

    segments: list[Dict[str, Any]] = []
    for segment_index, (trace_path, rows) in enumerate(raw_segments):
        window_rows = (
            list(rows[:audit_window_steps])
            if audit_window_steps > 0
            else list(rows)
        )
        active_scenes = [
            scene
            for row in window_rows
            if (scene := _trace_row_router_scene(row))
            in CONTROLLED_RUNTIME_SCENES
        ]
        scene_counts = Counter(active_scenes)
        active_steps = len(active_scenes)
        window_count = len(window_rows)
        active_ratio = active_steps / max(window_count, 1)
        if scene_counts:
            dominant_scene = max(
                CONTROLLED_RUNTIME_SCENES,
                key=lambda scene: (
                    int(scene_counts.get(scene, 0)),
                    -CONTROLLED_RUNTIME_SCENES.index(scene),
                ),
            )
            dominant_count = int(scene_counts[dominant_scene])
        else:
            dominant_scene = "none"
            dominant_count = 0
        scene_purity = dominant_count / max(active_steps, 1)

        reasons = []
        if active_steps < min_active_steps:
            reasons.append("insufficient_active_steps")
        if active_ratio < min_active_ratio:
            reasons.append("insufficient_active_ratio")
        if dominant_scene not in CONTROLLED_RUNTIME_SCENES:
            reasons.append("no_controlled_dominant_scene")
        if scene_purity < min_scene_purity:
            reasons.append("insufficient_scene_purity")

        confidence_values = []
        for row in window_rows:
            execution = row.get("v6_execution", {})
            if isinstance(execution, Mapping):
                try:
                    confidence_values.append(
                        float(execution.get("router_confidence", 0.0))
                    )
                except (TypeError, ValueError):
                    pass
        segments.append(
            {
                "segment_index": segment_index,
                "trace_path": str(trace_path),
                "trace_row_count": len(rows),
                "audit_window_row_count": window_count,
                "router_active_step_count": active_steps,
                "router_active_ratio": active_ratio,
                "active_scene_step_counts": {
                    scene: int(scene_counts.get(scene, 0))
                    for scene in CONTROLLED_RUNTIME_SCENES
                },
                "dominant_controlled_scene": dominant_scene,
                "dominant_scene_purity": scene_purity,
                "mean_router_confidence": (
                    sum(confidence_values) / len(confidence_values)
                    if confidence_values
                    else 0.0
                ),
                "eligible": not reasons,
                "ineligibility_reasons": reasons,
            }
        )
    return segments


def _attach_runner_report_metadata(
    *, run_dir: Path, segments: list[Dict[str, Any]]
) -> Dict[str, Any]:
    report_path = run_dir / "runner_report.parquet"
    if not report_path.is_file():
        return {
            "passed": False,
            "runner_report_path": str(report_path),
            "runner_report_row_count": 0,
            "trace_segment_count": len(segments),
            "reason": "runner_report_missing",
        }
    try:
        import pandas as pd

        report = pd.read_parquet(report_path)
    except Exception as exc:
        return {
            "passed": False,
            "runner_report_path": str(report_path),
            "runner_report_row_count": 0,
            "trace_segment_count": len(segments),
            "reason": f"runner_report_read_failed: {type(exc).__name__}: {exc}",
        }

    report_rows = report.to_dict(orient="records")
    passed = len(report_rows) == len(segments)
    if passed:
        # Discovery/formal Router auditing requires the sequential worker.
        # worker.map and trace append order then both follow scenario build order.
        for segment, metadata in zip(segments, report_rows):
            segment["scenario_name"] = str(metadata.get("scenario_name", ""))
            segment["scenario_token"] = str(metadata.get("scenario_name", ""))
            segment["log_name"] = str(metadata.get("log_name", ""))
            segment["planner_name"] = str(metadata.get("planner_name", ""))
            segment["simulation_succeeded"] = bool(
                metadata.get("succeeded", False)
            )
    return {
        "passed": passed,
        "runner_report_path": str(report_path),
        "runner_report_row_count": len(report_rows),
        "trace_segment_count": len(segments),
        "reason": "" if passed else "runner_report_trace_segment_count_mismatch",
    }


def _audit_runtime_contract(
    *,
    run_dir: Path,
    rho: float,
    contract: Dict[str, Any],
    audit_window_steps: int,
    min_router_active_steps: int,
    min_router_active_ratio: float,
    min_router_scene_purity: float,
    require_router_eligible_scenarios: bool,
    required_controlled_scenes: Sequence[str],
    expected_scenario_tokens: Sequence[str],
) -> Dict[str, Any]:
    trace_paths = sorted(run_dir.rglob("runtime_preference_trace.jsonl"))
    if not trace_paths:
        raise RuntimeError(
            f"No runtime_preference_trace.jsonl was produced under {run_dir}"
        )

    total_rows = 0
    normal_anchor_used_rows = 0
    empty_reference_used_rows = 0
    mismatches: list[Dict[str, Any]] = []
    trace_rows_by_path: list[tuple[Path, list[Mapping[str, Any]]]] = []
    expected_anchor = bool(contract["normal_anchor_cfg_enabled"])
    expected_cfg_scale = float(contract["cfg_guidance_scale"])
    for trace_path in trace_paths:
        trace_rows: list[Mapping[str, Any]] = []
        with open(trace_path, "r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    continue
                total_rows += 1
                row = json.loads(line)
                trace_rows.append(row)
                execution = row.get("v6_execution", {})
                if not isinstance(execution, dict):
                    execution = {}
                actual = {
                    "rho_requested": float(execution.get("rho_requested", math.nan)),
                    "normal_anchor_cfg_requested": bool(
                        execution.get("normal_anchor_cfg_requested", False)
                    ),
                    "cfg_guidance_scale": float(
                        execution.get("cfg_guidance_scale", math.nan)
                    ),
                }
                row_mismatches = {}
                if not math.isclose(
                    actual["rho_requested"], float(rho), abs_tol=1e-8
                ):
                    row_mismatches["rho_requested"] = actual["rho_requested"]
                if actual["normal_anchor_cfg_requested"] != expected_anchor:
                    row_mismatches["normal_anchor_cfg_requested"] = actual[
                        "normal_anchor_cfg_requested"
                    ]
                if not math.isclose(
                    actual["cfg_guidance_scale"],
                    expected_cfg_scale,
                    abs_tol=1e-8,
                ):
                    row_mismatches["cfg_guidance_scale"] = actual[
                        "cfg_guidance_scale"
                    ]
                if row_mismatches and len(mismatches) < 20:
                    mismatches.append(
                        {
                            "trace_path": str(trace_path),
                            "line_number": line_number,
                            "actual_mismatches": row_mismatches,
                        }
                    )
                normal_anchor_used_rows += int(
                    bool(execution.get("normal_anchor_cfg_used", False))
                )
                empty_reference_used_rows += int(
                    bool(execution.get("empty_cfg_reference_used", False))
                )
        trace_rows_by_path.append((trace_path, trace_rows))

    if total_rows <= 0:
        raise RuntimeError(
            f"Runtime trace files exist but contain no rows under {run_dir}"
        )
    if not expected_anchor and normal_anchor_used_rows > 0:
        mismatches.append(
            {
                "normal_anchor_cfg_used_rows": normal_anchor_used_rows,
                "expected": 0,
            }
        )
    router_segments = _segment_router_trace_rows(
        trace_rows_by_path,
        audit_window_steps=audit_window_steps,
        min_active_steps=min_router_active_steps,
        min_active_ratio=min_router_active_ratio,
        min_scene_purity=min_router_scene_purity,
    )
    router_active_audit_rows = sum(
        int(segment["router_active_step_count"])
        for segment in router_segments
    )
    if expected_anchor and router_active_audit_rows > 0:
        if normal_anchor_used_rows <= 0:
            mismatches.append(
                {
                    "normal_anchor_cfg_used_rows": normal_anchor_used_rows,
                    "expected": "> 0 when controlled Router is active",
                }
            )
        if empty_reference_used_rows > 0:
            mismatches.append(
                {
                    "empty_cfg_reference_used_rows": empty_reference_used_rows,
                    "expected": 0,
                }
            )
    identity_alignment = _attach_runner_report_metadata(
        run_dir=run_dir,
        segments=router_segments,
    )
    eligible_segments = [
        segment for segment in router_segments if bool(segment["eligible"])
    ]
    eligible_scene_counts = Counter(
        str(segment["dominant_controlled_scene"])
        for segment in eligible_segments
    )
    actual_scenario_tokens = [
        str(segment.get("scenario_token", ""))
        for segment in router_segments
        if str(segment.get("scenario_token", ""))
    ]
    if expected_scenario_tokens:
        expected_token_set = set(expected_scenario_tokens)
        actual_token_set = set(actual_scenario_tokens)
        if (
            not bool(identity_alignment["passed"])
            or actual_token_set != expected_token_set
            or len(actual_scenario_tokens) != len(expected_scenario_tokens)
        ):
            mismatches.append(
                {
                    "scenario_token_cohort_mismatch": {
                        "expected_count": len(expected_scenario_tokens),
                        "actual_count": len(actual_scenario_tokens),
                        "missing": sorted(
                            expected_token_set - actual_token_set
                        ),
                        "unexpected": sorted(
                            actual_token_set - expected_token_set
                        ),
                        "identity_alignment": identity_alignment,
                    }
                }
            )
    if require_router_eligible_scenarios:
        if not bool(identity_alignment["passed"]):
            mismatches.append(
                {
                    "router_scenario_identity_alignment": identity_alignment,
                    "expected": "passed",
                }
            )
        ineligible = [
            {
                "segment_index": int(segment["segment_index"]),
                "scenario_token": str(segment.get("scenario_token", "")),
                "log_name": str(segment.get("log_name", "")),
                "reasons": list(segment["ineligibility_reasons"]),
            }
            for segment in router_segments
            if not bool(segment["eligible"])
        ]
        if ineligible:
            mismatches.append(
                {
                    "router_ineligible_scenarios": ineligible[:20],
                    "ineligible_count": len(ineligible),
                    "expected": 0,
                }
            )
    missing_required_scenes = [
        scene
        for scene in required_controlled_scenes
        if int(eligible_scene_counts.get(scene, 0)) <= 0
    ]
    if missing_required_scenes:
        mismatches.append(
            {
                "missing_required_controlled_scenes": missing_required_scenes,
                "eligible_scene_counts": dict(eligible_scene_counts),
            }
        )
    audit = {
        "passed": not mismatches,
        "trace_file_count": len(trace_paths),
        "trace_row_count": total_rows,
        "normal_anchor_cfg_used_rows": normal_anchor_used_rows,
        "empty_cfg_reference_used_rows": empty_reference_used_rows,
        "router_eligibility_contract": {
            "audit_window_steps": audit_window_steps,
            "min_router_active_steps": min_router_active_steps,
            "min_router_active_ratio": min_router_active_ratio,
            "min_router_scene_purity": min_router_scene_purity,
            "require_every_scenario_eligible": (
                require_router_eligible_scenarios
            ),
            "required_controlled_scenes": list(required_controlled_scenes),
            "expected_scenario_token_count": len(
                expected_scenario_tokens
            ),
        },
        "router_scenario_identity_alignment": identity_alignment,
        "router_scenario_count": len(router_segments),
        "router_active_audit_window_rows": router_active_audit_rows,
        "router_eligible_scenario_count": len(eligible_segments),
        "router_eligible_scene_counts": {
            scene: int(eligible_scene_counts.get(scene, 0))
            for scene in CONTROLLED_RUNTIME_SCENES
        },
        "actual_scenario_tokens": actual_scenario_tokens,
        "router_scenarios": router_segments,
        "mismatches": mismatches,
        "note": (
            "CFG used-row counts may be below the total because style routing "
            "is inactive outside eligible car-follow/free-drive steps"
        ),
    }
    return audit


def _resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint_paths:
        checkpoints = [Path(value).expanduser().resolve() for value in args.checkpoint_paths]
    else:
        experiment_dir = Path(args.experiment_dir)
        checkpoints = []
        for epoch in _parse_int_list(args.checkpoint_epochs):
            candidates = sorted(experiment_dir.glob(f"model_epoch_{epoch}_trainloss_*.pth"))
            if not candidates:
                candidates = sorted(experiment_dir.glob(f"checkpoint_epoch_{epoch}.pth"))
            if not candidates:
                raise FileNotFoundError(f"No checkpoint found for epoch {epoch}")
            checkpoints.append(candidates[0].resolve())
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    return checkpoints


def _checkpoint_tag(path: Path) -> str:
    match = CHECKPOINT_EPOCH_PATTERN.search(path.name)
    return f"epoch_{int(match.group(1))}" if match else path.stem


def _rho_tag(rho: float) -> str:
    value = f"{abs(float(rho)):.2f}".replace(".", "p")
    prefix = "m" if rho < 0 else "p" if rho > 0 else "z"
    return f"rho_{prefix}{value}"


def _read_json_string_list(path: str, *, option_name: str) -> list[str]:
    if not str(path).strip():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(
            f"{option_name} must contain a JSON list of strings"
        )
    values = [item.strip() for item in payload if item.strip()]
    if len(set(values)) != len(values):
        raise ValueError(f"{option_name} must not contain duplicates")
    return values


def _read_log_names(path: str) -> list[str]:
    return _read_json_string_list(path, option_name="--log-names-json")


def _read_scenario_tokens(path: str) -> list[str]:
    return _read_json_string_list(
        path,
        option_name="--scenario-tokens-json",
    )


def _parse_names(raw: str) -> list[str]:
    return [value.strip() for value in str(raw).split(",") if value.strip()]


def _db_count(path: Path) -> int:
    if path.is_file():
        return int(path.suffix == ".db")
    return sum(1 for _ in path.rglob("*.db"))


def _preflight(
    args: argparse.Namespace,
    checkpoints: Sequence[Path],
) -> Dict[str, Any]:
    experiment_dir = Path(args.experiment_dir)
    args_file = experiment_dir / "args.json"
    db_path = Path(args.db_files)
    maps_root = Path(args.maps_root)
    devkit_root = Path(args.nuplan_devkit_root)
    required = {
        "experiment_dir": experiment_dir,
        "args_file": args_file,
        "db_files": db_path,
        "maps_root": maps_root,
        "nuplan_devkit_root": devkit_root,
        "local_hydra_config": REPO_ROOT / "baseline" / "config",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Closed-loop preflight failed; missing paths: "
            + ", ".join(f"{name}={required[name]}" for name in missing)
        )
    db_count = _db_count(db_path)
    if db_count <= 0:
        raise FileNotFoundError(f"No .db files found under --db-files={db_path}")

    train_args = json.loads(args_file.read_text(encoding="utf-8"))
    missing_runtime_keys = [
        key for key in REQUIRED_CLOSED_LOOP_CONFIG_KEYS if key not in train_args
    ]
    if missing_runtime_keys:
        raise ValueError(
            "args.json is missing required closed-loop model/input fields: "
            f"{missing_runtime_keys}"
        )
    effective_route_len = int(
        train_args.get("route_len", train_args.get("lane_len", 0))
    )
    if effective_route_len <= 0:
        raise ValueError(
            "args.json must provide a positive route_len, or a positive "
            "lane_len for the verified legacy route_len==lane_len fallback"
        )
    route_len_compatibility_fallback = "route_len" not in train_args
    required_artifact_keys = ("normalization_file_path",)
    missing_artifacts = []
    for key in required_artifact_keys:
        value = str(train_args.get(key, "")).strip()
        if value and not Path(value).exists():
            missing_artifacts.append(f"{key}={value}")
    if missing_artifacts:
        raise FileNotFoundError(
            "Required training-time normalization artifacts are missing on this server: "
            + ", ".join(missing_artifacts)
        )
    style_condition_encoder = str(train_args.get("style_condition_encoder", ""))
    if style_condition_encoder not in {"axis_router_v1", "axis_router_v2_signed"}:
        raise ValueError(
            "args.json is not a supported V6 Router experiment: "
            f"style_condition_encoder={style_condition_encoder!r}"
        )
    if int(train_args.get("style_value_dim", train_args.get("base_style_condition_dim", 0))) != 12:
        raise ValueError("args.json does not declare the fixed V6 12D style condition")
    if args.require_a3_8_terminal_executor:
        expected = {
            "style_condition_encoder": "axis_router_v2_signed",
            "signed_router_injection_mode": "ego_axis_temporal_residual",
            "signed_router_diffusion_gate_mode": "free_drive_terminal_only",
            "normal_anchor_cfg_enabled": False,
            "cfg_guidance_scale": 1.0,
        }
        mismatches = {
            key: {"expected": value, "actual": train_args.get(key)}
            for key, value in expected.items()
            if train_args.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "args.json does not satisfy the required A3.8 terminal-executor "
                f"contract: {mismatches}"
            )
        expected_geometry = {
            "time_len": 21,
            "future_len": 80,
            "agent_num": 32,
            "predicted_neighbor_num": 10,
            "static_objects_num": 5,
            "static_objects_state_dim": 10,
            "lane_len": 20,
            "lane_num": 70,
            "route_num": 25,
        }
        geometry_mismatches = {
            key: {"expected": value, "actual": train_args.get(key)}
            for key, value in expected_geometry.items()
            if train_args.get(key) != value
        }
        if effective_route_len != 20:
            geometry_mismatches["route_len"] = {
                "expected": 20,
                "actual": effective_route_len,
            }
        if geometry_mismatches:
            raise ValueError(
                "args.json does not satisfy the A3.8 closed-loop input "
                f"geometry contract: {geometry_mismatches}"
            )

    return {
        "db_file_count": db_count,
        "checkpoint_count": len(checkpoints),
        "args_file": str(args_file),
        "style_condition_encoder": train_args.get("style_condition_encoder"),
        "style_value_dim": train_args.get(
            "style_value_dim", train_args.get("base_style_condition_dim")
        ),
        "signed_router_injection_mode": train_args.get(
            "signed_router_injection_mode"
        ),
        "signed_router_diffusion_gate_mode": train_args.get(
            "signed_router_diffusion_gate_mode"
        ),
        "training_normal_anchor_cfg_enabled": bool(
            train_args.get("normal_anchor_cfg_enabled", False)
        ),
        "training_cfg_guidance_scale": float(
            train_args.get("cfg_guidance_scale", 1.0)
        ),
        "effective_route_len": effective_route_len,
        "route_len_compatibility_fallback": (
            route_len_compatibility_fallback
        ),
        "required_closed_loop_config_keys_checked": list(
            REQUIRED_CLOSED_LOOP_CONFIG_KEYS
        ),
        "required_a3_8_terminal_executor": bool(
            args.require_a3_8_terminal_executor
        ),
        "train_artifacts": {
            key: train_args.get(key, "") for key in required_artifact_keys
        },
    }


def _command(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    rho: float,
    variant_contract: Dict[str, Any],
    run_dir: Path,
    log_names: Sequence[str],
    map_names: Sequence[str],
    scenario_tokens: Sequence[str],
    scenario_types: Sequence[str],
) -> list[str]:
    config_root = REPO_ROOT / "baseline" / "config"
    search_path = (
        "[pkg://nuplan.planning.script.config.common,"
        "pkg://nuplan.planning.script.experiments,"
        f"file://{config_root}]"
    )
    runtime_trace_dir = run_dir / "runtime_traces"
    variant_tag = _variant_tag(variant_contract)
    experiment_uid = (
        f"{args.challenge}/styleplanner_v6_stage_b/"
        f"{_checkpoint_tag(checkpoint)}/{variant_tag}/{_rho_tag(rho)}"
    )
    # NuPlan's metric aggregator selects metric files by looking for the
    # challenge name in their full paths. Since this launcher overrides
    # output_dir per job, keep the stable job layout and make the metric
    # subdirectory carry the challenge identity explicitly.
    metric_dir = f"{args.challenge}/metrics"
    command = [
        sys.executable,
        "-m",
        "nuplan.planning.script.run_simulation",
        f"+simulation={args.challenge}",
        "planner=style_planner",
        f"planner.style_planner.config.args_file={Path(args.experiment_dir) / 'args.json'}",
        f"planner.style_planner.ckpt_path={checkpoint}",
        "planner.style_planner.device=cuda",
        "planner.style_planner.config.render_save_dir=null",
        "planner.style_planner.config.raw_data_save_dir=null",
        (
            "planner.style_planner.config.runtime_trace_save_dir="
            f"{runtime_trace_dir}"
        ),
        "planner.style_planner.config.runtime_preference_enabled=true",
        "planner.style_planner.config.runtime_style_mode=continuous_v6",
        f"planner.style_planner.config.runtime_rho={float(rho)}",
        (
            "planner.style_planner.config.normal_anchor_cfg_enabled="
            f"{str(bool(variant_contract['normal_anchor_cfg_enabled'])).lower()}"
        ),
        (
            "planner.style_planner.config.cfg_guidance_scale="
            f"{float(variant_contract['cfg_guidance_scale'])}"
        ),
        "planner.style_planner.config.runtime_trace_export_enabled=true",
        f"scenario_builder={args.scenario_builder}",
        f"scenario_filter={args.scenario_filter}",
        f"scenario_builder.db_files={args.db_files}",
        f"experiment_uid={experiment_uid}",
        f"output_dir={run_dir}",
        f"metric_dir={metric_dir}",
        f"hydra.run.dir={run_dir}",
        "verbose=true",
        f"worker={args.worker}",
        "enable_simulation_progress_bar=true",
        "number_of_gpus_allocated_per_simulation=1.0",
        f"seed={int(args.seed)}",
        f"hydra.searchpath={search_path}",
    ]
    if scenario_tokens:
        # Exact token cohorts must not be silently truncated by a scenario
        # filter preset's built-in debug limit.
        command.append(
            f"scenario_filter.limit_total_scenarios={len(scenario_tokens)}"
        )
    elif int(args.limit_total_scenarios) > 0:
        command.append(
            f"scenario_filter.limit_total_scenarios={int(args.limit_total_scenarios)}"
        )
    if int(args.num_scenarios_per_type) > 0:
        command.append(
            "scenario_filter.num_scenarios_per_type="
            f"{int(args.num_scenarios_per_type)}"
        )
    if log_names:
        compact = json.dumps(list(log_names), ensure_ascii=False, separators=(",", ":"))
        command.append(f"scenario_filter.log_names={compact}")
    if map_names:
        compact = json.dumps(list(map_names), ensure_ascii=False, separators=(",", ":"))
        command.append(f"scenario_filter.map_names={compact}")
    if scenario_tokens:
        compact = json.dumps(
            list(scenario_tokens),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        command.append(f"scenario_filter.scenario_tokens={compact}")
    if scenario_types:
        compact = json.dumps(
            list(scenario_types),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        command.append(f"scenario_filter.scenario_types={compact}")
    return command


def _export_balanced_router_tokens(
    *,
    output_path: str,
    audit: Mapping[str, Any],
    per_scene: int,
) -> Dict[str, Any]:
    selected: list[Dict[str, Any]] = []
    router_scenarios = audit.get("router_scenarios", [])
    if not isinstance(router_scenarios, list):
        router_scenarios = []
    for scene in CONTROLLED_RUNTIME_SCENES:
        candidates = [
            dict(segment)
            for segment in router_scenarios
            if isinstance(segment, Mapping)
            and bool(segment.get("eligible", False))
            and bool(segment.get("simulation_succeeded", False))
            and str(segment.get("dominant_controlled_scene", "")) == scene
            and str(segment.get("scenario_token", ""))
        ]
        if len(candidates) < per_scene:
            raise RuntimeError(
                "Router discovery did not find enough eligible scenarios for "
                f"{scene}: required={per_scene}, found={len(candidates)}"
            )
        selected.extend(candidates[:per_scene])

    tokens = [str(segment["scenario_token"]) for segment in selected]
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("Router discovery produced duplicate scenario tokens")
    target = Path(output_path)
    _write_json(target, tokens)
    metadata_path = target.with_name(
        f"{target.stem}_metadata{target.suffix or '.json'}"
    )
    metadata = {
        "artifact": "v6_closed_loop_router_eligible_cohort",
        "selection_source": "router_only_rho_zero_closed_loop",
        "selection_is_treatment_independent": True,
        "controlled_scenes": list(CONTROLLED_RUNTIME_SCENES),
        "per_scene": per_scene,
        "scenario_token_count": len(tokens),
        "scenario_tokens_json": str(target),
        "selected_scenarios": selected,
        "router_eligibility_contract": audit.get(
            "router_eligibility_contract",
            {},
        ),
    }
    _write_json(metadata_path, metadata)
    return {
        "scenario_tokens_json": str(target),
        "metadata_json": str(metadata_path),
        "scenario_token_count": len(tokens),
    }


def main() -> None:
    args = _parser().parse_args()
    checkpoints = _resolve_checkpoints(args)
    rho_values = _parse_rhos(args.rho_values)
    variants = _parse_variants(args.variants)
    _validate_guidance_controls(args)
    log_names = _read_log_names(args.log_names_json)
    map_names = _parse_names(args.map_names)
    scenario_tokens = _read_scenario_tokens(args.scenario_tokens_json)
    scenario_types = _parse_names(args.scenario_types)
    required_controlled_scenes = _parse_names(
        args.required_controlled_scenes
    )
    if scenario_tokens and (
        scenario_types or int(args.num_scenarios_per_type) > 0
    ):
        raise ValueError(
            "An exact --scenario-tokens-json cohort must not be combined with "
            "--scenario-types or --num-scenarios-per-type"
        )
    if str(args.export_router_eligible_tokens_json).strip():
        if (
            len(checkpoints) != 1
            or variants != ["router_only"]
            or len(rho_values) != 1
            or not math.isclose(rho_values[0], 0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                "--export-router-eligible-tokens-json requires exactly one "
                "checkpoint, --variants router_only, and --rho-values=0"
            )
        if scenario_tokens:
            raise ValueError(
                "Router discovery must scan candidates, not reuse "
                "--scenario-tokens-json"
            )
    preflight = _preflight(args, checkpoints)
    guidance_stage = "stage_b"
    run_tag = args.run_tag.strip() or dt.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    suite_root = Path(args.output_root) / run_tag
    suite_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for checkpoint in checkpoints:
        for variant in variants:
            variant_contract = _variant_contract(
                variant,
                normal_anchor_cfg_scale=args.normal_anchor_cfg_scale,
            )
            variant_tag = _variant_tag(variant_contract)
            for rho in rho_values:
                run_dir = (
                    suite_root
                    / _checkpoint_tag(checkpoint)
                    / variant_tag
                    / _rho_tag(rho)
                )
                command = _command(
                    args=args,
                    checkpoint=checkpoint,
                    rho=rho,
                    variant_contract=variant_contract,
                    run_dir=run_dir,
                    log_names=log_names,
                    map_names=map_names,
                    scenario_tokens=scenario_tokens,
                    scenario_types=scenario_types,
                )
                jobs.append(
                    {
                        "checkpoint": str(checkpoint),
                        "checkpoint_tag": _checkpoint_tag(checkpoint),
                        "rho": float(rho),
                        "variant": variant,
                        "variant_tag": variant_tag,
                        "runtime_contract": dict(variant_contract),
                        "run_dir": str(run_dir),
                        "command": command,
                        "status": "pending",
                    }
                )

    if args.execute:
        stale_run_artifacts = []
        for job in jobs:
            run_dir = Path(str(job["run_dir"]))
            if not run_dir.exists():
                continue
            trace_paths = sorted(
                run_dir.rglob("runtime_preference_trace.jsonl")
            )
            runner_report_path = run_dir / "runner_report.parquet"
            if trace_paths or runner_report_path.is_file():
                stale_run_artifacts.append(
                    {
                        "run_dir": str(run_dir),
                        "trace_file_count": len(trace_paths),
                        "runner_report_exists": runner_report_path.is_file(),
                    }
                )
        if stale_run_artifacts:
            preview = json.dumps(
                stale_run_artifacts[:6],
                ensure_ascii=False,
            )
            raise FileExistsError(
                "Refusing to append closed-loop traces into an existing run "
                "directory because that invalidates scenario/runner identity "
                "alignment. Choose a fresh --run-tag. Existing targets: "
                f"{preview}"
            )

    manifest_path = suite_root / "closed_loop_suite_manifest.json"
    guidance_runtime_contract = {
        "stage": guidance_stage,
        "variants": variants,
        "normal_anchor_cfg_scale": float(args.normal_anchor_cfg_scale),
        "rho_is_only_external_style_control": True,
        "same_checkpoint_scenarios_seed_across_variants": True,
        "raw_step_export_enabled": False,
        "runtime_trace_export_enabled": True,
    }
    manifest: Dict[str, Any] = {
        "artifact": "styleplanner_v6_closed_loop_suite",
        "execute": bool(args.execute),
        "preflight": preflight,
        "scenario_filter": args.scenario_filter,
        "challenge": args.challenge,
        "guidance_stage": guidance_stage,
        "guidance_runtime_contract": guidance_runtime_contract,
        f"{guidance_stage}_runtime_contract": guidance_runtime_contract,
        "limit_total_scenarios": int(args.limit_total_scenarios),
        "log_name_count": len(log_names),
        "map_names": map_names,
        "scenario_token_count": len(scenario_tokens),
        "scenario_tokens_json": str(args.scenario_tokens_json),
        "scenario_types": scenario_types,
        "num_scenarios_per_type": int(args.num_scenarios_per_type),
        "router_eligibility_contract": {
            "audit_window_steps": int(args.router_audit_window_steps),
            "min_router_active_steps": int(args.min_router_active_steps),
            "min_router_active_ratio": float(
                args.min_router_active_ratio
            ),
            "min_router_scene_purity": float(
                args.min_router_scene_purity
            ),
            "require_every_scenario_eligible": bool(
                args.require_router_eligible_scenarios
            ),
            "required_controlled_scenes": required_controlled_scenes,
            "selection_source": (
                "fixed_router_only_rho_zero_discovery"
                if scenario_tokens
                else "candidate_scan_or_unfixed"
            ),
        },
        "cuda_visible_devices": str(args.cuda_visible_devices),
        "jobs": jobs,
    }
    _write_json(manifest_path, manifest)
    print(f"[V6ClosedLoop] preflight={json.dumps(preflight, ensure_ascii=False)}")
    print(f"[V6ClosedLoop] jobs={len(jobs)} manifest={manifest_path}")
    if not args.execute:
        print("[V6ClosedLoop] dry-run only; add --execute after reviewing the manifest")
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env["NUPLAN_DEVKIT_ROOT"] = str(Path(args.nuplan_devkit_root))
    env["NUPLAN_DATA_ROOT"] = str(Path(args.db_files))
    env["NUPLAN_MAPS_ROOT"] = str(Path(args.maps_root))
    env["NUPLAN_EXP_ROOT"] = str(suite_root)
    env["HYDRA_FULL_ERROR"] = "1"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env["SWANLAB_MODE"] = "disabled"
    python_path = [str(REPO_ROOT), str(Path(args.nuplan_devkit_root))]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    for job in jobs:
        run_dir = Path(str(job["run_dir"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        job["status"] = "running"
        _write_json(manifest_path, manifest)
        print(
            f"[V6ClosedLoop] running {job['checkpoint_tag']} "
            f"variant={job['variant_tag']} rho={job['rho']} "
            f"output={run_dir}"
        )
        try:
            subprocess.run(job["command"], check=True, env=env, cwd=str(REPO_ROOT))
            job["runtime_contract_audit"] = _audit_runtime_contract(
                run_dir=run_dir,
                rho=float(job["rho"]),
                contract=dict(job["runtime_contract"]),
                audit_window_steps=int(args.router_audit_window_steps),
                min_router_active_steps=int(
                    args.min_router_active_steps
                ),
                min_router_active_ratio=float(
                    args.min_router_active_ratio
                ),
                min_router_scene_purity=float(
                    args.min_router_scene_purity
                ),
                require_router_eligible_scenarios=bool(
                    args.require_router_eligible_scenarios
                ),
                required_controlled_scenes=required_controlled_scenes,
                expected_scenario_tokens=scenario_tokens,
            )
            _write_json(
                run_dir / "v6_closed_loop_router_audit.json",
                job["runtime_contract_audit"],
            )
            if not bool(job["runtime_contract_audit"].get("passed", False)):
                raise RuntimeError(
                    f"Runtime {guidance_stage.replace('_', ' ').title()} "
                    "contract audit failed; inspect "
                    f"{run_dir / 'v6_closed_loop_router_audit.json'}"
                )
            if str(args.export_router_eligible_tokens_json).strip():
                manifest["router_discovery_export"] = (
                    _export_balanced_router_tokens(
                        output_path=str(
                            args.export_router_eligible_tokens_json
                        ),
                        audit=job["runtime_contract_audit"],
                        per_scene=int(
                            args.eligible_scenarios_per_controlled_scene
                        ),
                    )
                )
                print(
                    "[V6ClosedLoop] router discovery export="
                    f"{manifest['router_discovery_export']}"
                )
            job["status"] = "completed"
            if args.run_custom_metrics:
                job["custom_metrics_status"] = "running"
                _write_json(manifest_path, manifest)
                try:
                    from baseline.simulation.simulation_metrics import (
                        check_failure_cases,
                        load_aggregator_metric,
                        load_metrics_dataframe,
                        print_statistics_report,
                    )

                    aggregator_frame = load_aggregator_metric(str(run_dir))
                    scenario_frame = load_metrics_dataframe(str(run_dir))
                    if scenario_frame is None:
                        raise RuntimeError(
                            "No per-scenario metric frame was produced"
                        )
                    print_statistics_report(
                        scenario_frame,
                        aggregator_frame,
                    )
                    check_failure_cases(scenario_frame)
                    scenario_metrics_path = (
                        run_dir / "v6_closed_loop_scenario_metrics.csv"
                    )
                    scenario_frame.to_csv(
                        scenario_metrics_path,
                        index=False,
                    )
                    structured_metrics_path = (
                        run_dir / "v6_closed_loop_metrics.json"
                    )
                    _write_json(
                        structured_metrics_path,
                        {
                            "artifact": "v6_closed_loop_metrics",
                            "scenario_count": int(len(scenario_frame)),
                            "scenario_metrics": json.loads(
                                scenario_frame.to_json(orient="records")
                            ),
                            "aggregator_metrics": (
                                json.loads(
                                    aggregator_frame.to_json(
                                        orient="records"
                                    )
                                )
                                if aggregator_frame is not None
                                else []
                            ),
                        },
                    )
                    job["custom_metrics_artifacts"] = {
                        "structured_json": str(structured_metrics_path),
                        "scenario_csv": str(scenario_metrics_path),
                    }
                    job["custom_metrics_status"] = "completed"
                except Exception as metrics_exc:
                    # The NuPlan simulation and runtime-contract audit have
                    # already completed. Preserve that result while making the
                    # optional reporting failure explicit and recoverable.
                    job["custom_metrics_status"] = "failed"
                    job["custom_metrics_error"] = (
                        f"{type(metrics_exc).__name__}: {metrics_exc}"
                    )
                    print(
                        "[V6ClosedLoop] custom metrics warning: "
                        f"{job['custom_metrics_error']}"
                    )
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = f"{type(exc).__name__}: {exc}"
            _write_json(manifest_path, manifest)
            if not args.continue_on_error:
                raise
        _write_json(manifest_path, manifest)

    completed = sum(job["status"] == "completed" for job in jobs)
    failed = sum(job["status"] == "failed" for job in jobs)
    print(
        f"[V6ClosedLoop] finished completed={completed} failed={failed} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()

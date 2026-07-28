"""Phase-0 Stylization Transport Tensor diagnosis for V6 StylePlanner.

This evaluator never trains or modifies checkpoint weights.  It reuses the
existing V6 condition/data contract and changes exactly one target coordinate
at a time around the semantic-normal command.  A diagnostic-only sampler gate
then selects the DPM denoiser evaluations at which the signed style residual
is injected.

The raw output is one paired finite-difference measurement per
``(scenario, seed, solver setting, phase window, input axis)``.  Aggregation
first averages seed replicas within scenario and only then bootstraps scenarios,
which prevents seed/rho rows from being treated as independent samples.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
for _path in (REPO_ROOT, DEVKIT_ROOT):
    _value = str(_path)
    if _path.exists() and _value not in sys.path:
        sys.path.insert(0, _value)

from baseline.model.style_planner.library import dpm_solver_pytorch as dpm
from baseline.model.style_planner.library.sampling import (
    TransportInjectionProbe,
    selftest_transport_injection_probe,
)
from research.continuous_style.schema import CANONICAL_AXIS_BY_SCENE
from research.continuous_style.v6 import build_style_condition_vector
from research.preference_execution.diffusion.dataset import (
    PreferenceConditionedPlannerData,
)
from research.preference_execution.diffusion.style_condition import (
    style_condition_valid_mask,
)
from research.preference_execution.diffusion.training import (
    prepare_preference_conditioned_batch,
)
from research.preference_execution.eval.evaluate_styleplanner_v6_checkpoints import (
    CONTROLLED_SCENES,
    _batch_from_sample,
    _clone_inputs,
    _load_model,
    _model_args,
    _prediction,
    _select_indices,
    _set_seed,
    _trajectory_metrics,
    _write_condition_subset,
)


RAW_ROLLOUT_NAME = "phase0_transport_rollouts.jsonl.gz"
PAIR_NAME = "phase0_transport_pairs.jsonl"
TENSOR_CSV_NAME = "phase0_transport_tensor.csv"
LEAKAGE_CSV_NAME = "phase0_transport_leakage.csv"
CONTRACT_CSV_NAME = "phase0_legacy_equivalence.csv"
SUMMARY_NAME = "phase0_transport_summary.json"
MANIFEST_NAME = "phase0_run_manifest.json"
SELECTED_NAME = "phase0_selected_samples.json"
ELIGIBILITY_NAME = "phase0_eligibility.json"
SELECTED_CONDITIONS_NAME = "phase0_selected_conditions.jsonl"


@dataclass(frozen=True)
class ProbePhaseSpec:
    """One checkpoint or counterfactual residual-injection policy."""

    label: str
    policy: str
    active_call_indices: Optional[tuple[int, ...]]
    override_style_residual_gate: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase-0 same-noise single-axis Stylization Transport Tensor "
            "diagnosis for one A3.7/A3.8 V6 checkpoint."
        )
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--experiment-dir", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--split-root", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--condition-index-path", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--normalization-path", default="")
    parser.add_argument("--conditional-rank-model-path", default="")
    parser.add_argument(
        "--max-samples-per-controlled-scene",
        type=int,
        default=32,
        help="Number of causally eligible samples retained in each longitudinal scene.",
    )
    parser.add_argument(
        "--eligible-search-multiplier",
        type=int,
        default=4,
        help="Candidate oversampling factor before normal-axis eligibility filtering.",
    )
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument(
        "--solver-steps",
        default="10",
        help=(
            "Comma-separated DPM multistep evaluations before denoise-to-zero. "
            "Actual denoiser evaluations are solver_steps + 1."
        ),
    )
    parser.add_argument(
        "--denoiser-evaluations",
        default="",
        help=(
            "Optional comma-separated total model evaluations including the final "
            "denoise-to-zero call; mutually exclusive with --solver-steps."
        ),
    )
    parser.add_argument(
        "--phase-windows",
        default="legacy,all,early,middle,late,terminal",
        help=(
            "Comma-separated legacy,all,early,middle,late,terminal or call:<q>. "
            "Early/middle/late are normalized-logSNR windows; terminal is the "
            "unique denoise-to-zero evaluation."
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--legacy-equivalence-tolerance", type=float, default=1e-6)
    parser.add_argument("--require-legacy-equivalence", action="store_true")
    parser.add_argument("--prefer-ema", action="store_true", default=True)
    parser.add_argument("--disable-prefer-ema", action="store_true")
    # _model_args accepts this optional field; Phase-0 itself always forces 1.0.
    parser.add_argument("--cfg-guidance-scale", type=float, default=None)
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_json_safe(value), ensure_ascii=False)
                    if isinstance(value, (list, dict, tuple))
                    else _json_safe(value)
                    for key, value in row.items()
                }
            )


def _read_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object JSONL row at {path}:{line_number}")
            records.append(payload)
    return records


def _parse_positive_int_list(raw: str, *, name: str, minimum: int) -> list[int]:
    values = [int(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not values or any(value < minimum for value in values):
        raise ValueError(f"{name} must contain integers >= {minimum}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _resolve_solver_steps(args: argparse.Namespace) -> list[int]:
    if args.denoiser_evaluations and str(args.solver_steps).strip() not in {"", "10"}:
        raise ValueError(
            "--denoiser-evaluations is mutually exclusive with an explicit "
            "--solver-steps value"
        )
    if args.denoiser_evaluations:
        evaluations = _parse_positive_int_list(
            args.denoiser_evaluations,
            name="--denoiser-evaluations",
            minimum=3,
        )
        return [value - 1 for value in evaluations]
    return _parse_positive_int_list(args.solver_steps, name="--solver-steps", minimum=2)


def _expected_logsnr_trace(
    solver_steps: int,
    *,
    device: Optional[torch.device] = None,
) -> list[Dict[str, Any]]:
    """Match the project's DPM-Solver++/logSNR/denoise-to-zero call schedule."""

    noise_schedule = dpm.NoiseScheduleVP(schedule="linear")
    solver = dpm.DPM_Solver(
        lambda x, time: x,
        noise_schedule,
        algorithm_type="dpmsolver++",
    )
    t_end = 1.0 / float(noise_schedule.total_N)
    trace_device = torch.device("cpu") if device is None else torch.device(device)
    times = solver.get_time_steps(
        skip_type="logSNR",
        t_T=float(noise_schedule.T),
        t_0=t_end,
        N=int(solver_steps),
        device=trace_device,
    )
    # Multistep evaluates at times[0:solver_steps], then denoise-to-zero makes
    # exactly one final call at times[solver_steps].  Thus the evaluation-time
    # vector is the whole time grid, with its last entry marked terminal.
    half_logsnr = noise_schedule.marginal_lambda(times)
    denominator = float((half_logsnr[-1] - half_logsnr[0]).item())
    if abs(denominator) <= 1e-12:
        raise RuntimeError("logSNR schedule has zero phase span")
    output: list[Dict[str, Any]] = []
    for call_index, (time, value) in enumerate(zip(times, half_logsnr)):
        normalized = float((value - half_logsnr[0]).item() / denominator)
        if call_index == int(solver_steps):
            window = "terminal"
        elif normalized < 1.0 / 3.0:
            window = "early"
        elif normalized < 2.0 / 3.0:
            window = "middle"
        else:
            window = "late"
        output.append(
            {
                "call_index": int(call_index),
                "diffusion_time": float(time.item()),
                "half_logsnr": float(value.item()),
                "normalized_logsnr_phase": normalized,
                "window": window,
                "is_denoise_to_zero": call_index == int(solver_steps),
            }
        )
    if len(output) != int(solver_steps) + 1 or output[-1]["window"] != "terminal":
        raise RuntimeError("unexpected DPM denoiser-evaluation schedule")
    return output


def _parse_phase_specs(
    raw: str,
    *,
    solver_steps: int,
    device: Optional[torch.device] = None,
) -> list[ProbePhaseSpec]:
    trace = _expected_logsnr_trace(solver_steps, device=device)
    by_window: Dict[str, tuple[int, ...]] = {
        window: tuple(
            int(row["call_index"]) for row in trace if row["window"] == window
        )
        for window in ("early", "middle", "late", "terminal")
    }
    tokens = [value.strip().lower() for value in str(raw).split(",") if value.strip()]
    if not tokens:
        raise ValueError("--phase-windows must not be empty")
    specs: list[ProbePhaseSpec] = []
    seen: set[str] = set()
    for token in tokens:
        if token == "checkpoint":
            token = "legacy"
        if token == "legacy":
            spec = ProbePhaseSpec(
                label="legacy",
                policy="checkpoint",
                active_call_indices=None,
                override_style_residual_gate=False,
            )
        elif token == "all":
            spec = ProbePhaseSpec(
                label="all",
                policy="phase_override",
                active_call_indices=None,
                override_style_residual_gate=True,
            )
        elif token in by_window:
            indices = by_window[token]
            if not indices:
                raise ValueError(
                    f"{token} has no denoiser calls at solver_steps={solver_steps}"
                )
            spec = ProbePhaseSpec(
                label=token,
                policy="phase_override",
                active_call_indices=indices,
                override_style_residual_gate=True,
            )
        elif token.startswith("call:"):
            try:
                call_index = int(token.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"invalid explicit call selector {token!r}") from exc
            if call_index < 0 or call_index > int(solver_steps):
                raise ValueError(
                    f"{token!r} is outside [0, {solver_steps}] for solver_steps={solver_steps}"
                )
            spec = ProbePhaseSpec(
                label=f"call_{call_index}",
                policy="phase_override",
                active_call_indices=(call_index,),
                override_style_residual_gate=True,
            )
        else:
            raise ValueError(
                "--phase-windows entries must be legacy,all,early,middle,late,"
                "terminal, or call:<q>"
            )
        if spec.label in seen:
            raise ValueError(f"duplicate phase selector {spec.label!r}")
        seen.add(spec.label)
        specs.append(spec)
    return specs


def _trace_from_probe(
    probe: TransportInjectionProbe,
    *,
    expected_trace: Sequence[Mapping[str, Any]],
    phase_spec: ProbePhaseSpec,
    effective_residual_active_call_indices: Sequence[int],
) -> list[Dict[str, Any]]:
    diagnostics = probe.diagnostics()
    observed_times = list(diagnostics["model_evaluation_times"])
    active_mask = list(diagnostics["model_evaluation_active_mask"])
    if len(observed_times) != len(expected_trace):
        raise RuntimeError(
            "transport probe model-evaluation count mismatch: "
            f"expected {len(expected_trace)}, got {len(observed_times)}"
        )
    if len(active_mask) != len(expected_trace):
        raise RuntimeError("transport probe active-mask length mismatch")
    if phase_spec.active_call_indices is None:
        expected_active = [True] * len(expected_trace)
    else:
        active_set = set(phase_spec.active_call_indices)
        expected_active = [
            int(row["call_index"]) in active_set for row in expected_trace
        ]
    if [bool(value) for value in active_mask] != expected_active:
        raise RuntimeError("transport probe active-call trace does not match request")
    valid_call_indices = {int(row["call_index"]) for row in expected_trace}
    effective_active = {
        int(index) for index in effective_residual_active_call_indices
    }
    if not effective_active.issubset(valid_call_indices):
        raise RuntimeError("effective residual gate contains an invalid denoiser call")
    annotated: list[Dict[str, Any]] = []
    for expected, observed_time, active in zip(expected_trace, observed_times, active_mask):
        if abs(float(observed_time) - float(expected["diffusion_time"])) > 2e-5:
            raise RuntimeError(
                "transport probe diffusion-time trace differs from the expected "
                "logSNR DPM schedule: "
                f"call={expected['call_index']}, "
                f"expected={float(expected['diffusion_time']):.9g}, "
                f"observed={float(observed_time):.9g}"
            )
        row = dict(expected)
        row["observed_diffusion_time"] = float(observed_time)
        # ``probe_gate`` is the sampler observer/override request.  It is not
        # necessarily the deployed residual gate: an A3.8 legacy free-drive
        # rollout observes every call but actually injects only at terminal.
        row["probe_gate"] = bool(active)
        row["effective_residual_gate"] = int(expected["call_index"]) in effective_active
        row["injected"] = row["effective_residual_gate"]
        row["residual_gate_source"] = (
            "phase_override"
            if phase_spec.override_style_residual_gate
            else "checkpoint"
        )
        annotated.append(row)
    return annotated


def _as_float_vector(value: Any, *, length: int = 3) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value if value is not None else [], dtype=np.float64).reshape(-1)
    output = np.full((length,), np.nan, dtype=np.float64)
    output[: min(length, array.size)] = array[:length]
    return output


def _as_bool_vector(value: Any, *, length: int = 3) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value if value is not None else [], dtype=bool).reshape(-1)
    output = np.zeros((length,), dtype=bool)
    output[: min(length, array.size)] = array[:length]
    return output


def _optional_vector(value: np.ndarray) -> list[Optional[float]]:
    return [float(item) if math.isfinite(float(item)) else None for item in value]


def _axis_payload(diagnostics: Mapping[str, Any]) -> Dict[str, Any]:
    canonical = _as_float_vector(
        diagnostics.get("preference_generated_axis_canonical")
    )
    percentile = _as_float_vector(
        diagnostics.get("preference_generated_axis_percentile")
    )
    valid = _as_bool_vector(
        diagnostics.get("preference_generated_axis_valid_mask")
    )
    return {
        "canonical": canonical,
        "percentile": percentile,
        "valid": valid,
    }


def _sample_scene_gate(sample: Mapping[str, Any]) -> np.ndarray:
    values = _as_float_vector(sample.get("scene_gate_values"))
    return np.where(np.isfinite(values), values, 0.0)


def _sample_causal_mask(sample: Mapping[str, Any]) -> np.ndarray:
    # Match the existing V6 evaluator exactly: the stored gate is a numeric
    # causal availability signal, not Python truthiness.  This is important for
    # preserving a pre-generated sidecar's axis partition unchanged.
    values = _as_float_vector(sample.get("local_axis_gate_values"))
    return np.isfinite(values) & (values > 0.5)


def _normal_condition(
    *,
    scene: str,
    causal_mask: np.ndarray,
    scene_gate: np.ndarray,
) -> np.ndarray:
    condition = build_style_condition_vector(
        axis_target=(0.5, 0.5, 0.5),
        axis_mask=causal_mask.tolist(),
        scene_bucket=scene,
        scene_gate_values=scene_gate.tolist(),
        disabled_when_empty=True,
    ).astype(np.float32)
    if condition.shape != (12,):
        raise RuntimeError(f"V6 normal condition must have 12 entries, got {condition.shape}")
    return condition


def _single_axis_condition(
    normal_condition: np.ndarray,
    *,
    axis_index: int,
    sign: int,
    delta: float,
) -> np.ndarray:
    if axis_index < 0 or axis_index >= 3:
        raise ValueError("axis_index must be in [0, 2]")
    if sign not in {-1, 1}:
        raise ValueError("sign must be -1 or +1")
    normal = np.asarray(normal_condition, dtype=np.float32)
    condition = normal.copy()
    if condition.shape != (12,):
        raise ValueError("single-axis Phase-0 command requires a 12D V6 condition")
    condition[:3] = 0.5
    condition[axis_index] = float(0.5 + sign * delta)
    # Mask and scene identity are discrete and must match exactly.  Scene-gate
    # values are continuous; compare them after the deliberate float32 planner
    # cast rather than rejecting a harmless float64->float32 roundoff (e.g. 0.8).
    if not (
        np.array_equal(condition[3:9], normal[3:9])
        and np.allclose(condition[9:], normal[9:], atol=1e-7, rtol=0.0)
    ):
        raise RuntimeError("Phase-0 command changed a mask, scene slot, or gate")
    if not np.allclose(
        np.delete(condition[:3], axis_index),
        0.5,
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("Phase-0 command changed a non-target axis")
    return condition


def _build_style_inputs(
    base_inputs: Mapping[str, Any],
    *,
    condition: np.ndarray,
    normal_condition: np.ndarray,
    normal_prediction: Optional[np.ndarray],
    fixed_normal_neighbors: bool,
    normal_anchor_accel_support: bool,
) -> Dict[str, Any]:
    inputs = _clone_inputs(base_inputs)
    device = inputs["ego_current_state"].device
    dtype = inputs["ego_current_state"].dtype
    condition_tensor = torch.as_tensor(
        condition,
        device=device,
        dtype=torch.float32,
    ).reshape(1, -1)
    normal_tensor = torch.as_tensor(
        normal_condition,
        device=device,
        dtype=torch.float32,
    ).reshape(1, -1)
    inputs["style_value_condition"] = condition_tensor
    inputs["normal_anchor_style_value_condition"] = normal_tensor
    inputs["cfg_guidance_scale"] = 1.0
    valid = style_condition_valid_mask(condition_tensor)
    inputs["style_feature_valid"] = valid.float()
    inputs["style_condition_used"] = valid.float()
    if normal_prediction is not None:
        if fixed_normal_neighbors and normal_prediction.shape[0] > 1:
            inputs["preference_neighbor_reference_future"] = torch.as_tensor(
                normal_prediction[1:],
                device=device,
                dtype=dtype,
            ).unsqueeze(0)
        if normal_anchor_accel_support:
            inputs["preference_ego_reference_future"] = torch.as_tensor(
                normal_prediction[0],
                device=device,
                dtype=dtype,
            ).unsqueeze(0)
    return inputs


def _configure_solver_steps(model: torch.nn.Module, solver_steps: int) -> None:
    decoder = getattr(getattr(model, "decoder", None), "decoder", None)
    if decoder is None:
        raise TypeError("expected Diffusion_Planner.decoder.decoder")
    decoder._diffusion_steps = int(solver_steps)


def _validate_phase0_model(model: torch.nn.Module, model_args: argparse.Namespace) -> None:
    decoder = getattr(getattr(model, "decoder", None), "decoder", None)
    dit = getattr(decoder, "dit", None)
    if str(getattr(model_args, "style_condition_encoder", "")) != "axis_router_v2_signed":
        raise ValueError("Phase-0 transport requires style_condition_encoder=axis_router_v2_signed")
    if str(getattr(dit, "signed_router_injection_mode", "")) not in {
        "ego_axis_temporal_residual",
        "ego_scene_axis_temporal_residual",
    }:
        raise ValueError(
            "Phase-0 targets A3.7/A3.8/B3 and requires an axis-temporal "
            "signed_router_injection_mode"
        )
    # Always make this a router-only measurement.  The default current A3.8
    # production gate is retained by the trace-only `legacy` probe.
    decoder._normal_anchor_cfg_enabled = False


def _phase0_dit(model: torch.nn.Module) -> torch.nn.Module:
    decoder = getattr(getattr(model, "decoder", None), "decoder", None)
    dit = getattr(decoder, "dit", None)
    if dit is None:
        raise TypeError("expected Diffusion_Planner.decoder.decoder.dit")
    return dit


def _effective_residual_active_call_indices(
    scene: str,
    *,
    model: torch.nn.Module,
    expected_trace: Sequence[Mapping[str, Any]],
    phase_spec: ProbePhaseSpec,
) -> tuple[int, ...]:
    """Return calls where the signed residual really affects the trajectory.

    The probe observes every logical denoiser call for ``legacy`` so that its
    trace can validate the solver schedule.  Its effective residual calls must
    nevertheless follow the checkpoint's actual A3.7/A3.8 gate.
    """

    all_calls = tuple(int(row["call_index"]) for row in expected_trace)
    if phase_spec.override_style_residual_gate:
        if phase_spec.active_call_indices is None:
            return all_calls
        requested = tuple(int(index) for index in phase_spec.active_call_indices)
        if not set(requested).issubset(set(all_calls)):
            raise RuntimeError("phase override requested an invalid denoiser call")
        return requested

    dit = _phase0_dit(model)
    mode = str(getattr(dit, "signed_router_diffusion_gate_mode", "all_steps"))
    if mode == "all_steps":
        return all_calls
    if mode != "free_drive_terminal_only":
        raise ValueError(f"unsupported checkpoint residual gate mode: {mode!r}")
    if scene != "straight_free_drive":
        return all_calls

    terminal_t_max = float(getattr(dit, "signed_router_terminal_t_max", 0.0))
    active = tuple(
        int(row["call_index"])
        for row in expected_trace
        if float(row["diffusion_time"]) <= terminal_t_max + 1e-8
    )
    # A3.8's paper contract is a unique denoise-to-zero terminal call.  Fail
    # rather than silently relabel an altered sampler/warm-start schedule.
    if active != (all_calls[-1],):
        raise RuntimeError(
            "free-drive terminal gate is not unique under the observed DPM "
            f"schedule: threshold={terminal_t_max}, active_calls={active}"
        )
    return active


def _run_rollout(
    *,
    model: torch.nn.Module,
    base_inputs: Mapping[str, Any],
    condition: np.ndarray,
    normal_condition: np.ndarray,
    normal_prediction: Optional[np.ndarray],
    fixed_normal_neighbors: bool,
    normal_anchor_accel_support: bool,
    seed: int,
    probe: Optional[TransportInjectionProbe],
    expected_trace: Optional[Sequence[Mapping[str, Any]]],
    phase_spec: Optional[ProbePhaseSpec],
    effective_residual_active_call_indices: Optional[Sequence[int]],
) -> tuple[np.ndarray, Dict[str, Any], Optional[list[Dict[str, Any]]]]:
    inputs = _build_style_inputs(
        base_inputs,
        condition=condition,
        normal_condition=normal_condition,
        normal_prediction=normal_prediction,
        fixed_normal_neighbors=fixed_normal_neighbors,
        normal_anchor_accel_support=normal_anchor_accel_support,
    )
    if probe is not None:
        inputs["transport_injection_probe"] = probe
    _set_seed(int(seed))
    prediction, diagnostics = _prediction(model, inputs)
    trace = None
    if probe is not None:
        if (
            expected_trace is None
            or phase_spec is None
            or effective_residual_active_call_indices is None
        ):
            raise RuntimeError("transport probe requires an expected schedule and phase spec")
        trace = _trace_from_probe(
            probe,
            expected_trace=expected_trace,
            phase_spec=phase_spec,
            effective_residual_active_call_indices=effective_residual_active_call_indices,
        )
    return prediction, diagnostics, trace


def _checkpoint_equivalent_phase(scene: str, model: torch.nn.Module) -> Optional[str]:
    mode = str(
        getattr(_phase0_dit(model), "signed_router_diffusion_gate_mode", "all_steps")
    )
    if mode == "all_steps":
        return "all"
    if mode == "free_drive_terminal_only":
        return "terminal" if scene == "straight_free_drive" else "all"
    return None


def _trajectory_max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _ego_ade(left: np.ndarray, right: np.ndarray) -> float:
    if left.ndim < 3 or right.ndim < 3 or left.shape[0] == 0 or right.shape[0] == 0:
        return float("inf")
    count = min(left.shape[1], right.shape[1])
    if count <= 0:
        return 0.0
    return float(np.mean(np.linalg.norm(left[0, :count, :2] - right[0, :count, :2], axis=-1)))


def _build_pair_row(
    *,
    checkpoint_meta: Mapping[str, Any],
    scene: str,
    sample_id: str,
    dataset_index: int,
    seed: int,
    seed_index: int,
    solver_steps: int,
    phase_spec: ProbePhaseSpec,
    axis_index: int,
    axis_name: str,
    delta: float,
    normal_axis: Mapping[str, Any],
    plus_axis: Mapping[str, Any],
    minus_axis: Mapping[str, Any],
    normal_condition: np.ndarray,
    plus_condition: np.ndarray,
    minus_condition: np.ndarray,
    causal_mask: np.ndarray,
    fixed_input_axis_mask: np.ndarray,
    phase_trace: Sequence[Mapping[str, Any]],
    effective_residual_active_call_indices: Sequence[int],
    plus_metrics: Mapping[str, Any],
    minus_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    plus_canonical = np.asarray(plus_axis["canonical"], dtype=np.float64)
    minus_canonical = np.asarray(minus_axis["canonical"], dtype=np.float64)
    plus_percentile = np.asarray(plus_axis["percentile"], dtype=np.float64)
    minus_percentile = np.asarray(minus_axis["percentile"], dtype=np.float64)
    valid = (
        np.asarray(plus_axis["valid"], dtype=bool)
        & np.asarray(minus_axis["valid"], dtype=bool)
        & np.isfinite(plus_canonical)
        & np.isfinite(minus_canonical)
    )
    canonical_transport = np.full((3,), np.nan, dtype=np.float64)
    percentile_transport = np.full((3,), np.nan, dtype=np.float64)
    canonical_transport[valid] = (
        plus_canonical[valid] - minus_canonical[valid]
    ) / (2.0 * float(delta))
    percentile_valid = (
        np.asarray(plus_axis["valid"], dtype=bool)
        & np.asarray(minus_axis["valid"], dtype=bool)
        & np.isfinite(plus_percentile)
        & np.isfinite(minus_percentile)
    )
    percentile_transport[percentile_valid] = (
        plus_percentile[percentile_valid] - minus_percentile[percentile_valid]
    ) / (2.0 * float(delta))
    diagonal = canonical_transport[axis_index]
    all_axes_valid = bool(np.all(valid))
    leakage_ratio: Optional[float]
    if all_axes_valid and math.isfinite(float(diagonal)):
        leakage_ratio = float(
            np.sum(np.abs(np.delete(canonical_transport, axis_index)))
            / max(abs(float(diagonal)), 1e-8)
        )
    else:
        leakage_ratio = None
    return {
        "artifact": "phase0_stylization_transport_pair",
        "checkpoint": dict(checkpoint_meta),
        "scene_bucket": scene,
        "sample_id": sample_id,
        "dataset_index": int(dataset_index),
        "seed": int(seed),
        "seed_index": int(seed_index),
        "solver_steps": int(solver_steps),
        "denoiser_evaluations": int(solver_steps) + 1,
        "phase_label": phase_spec.label,
        "phase_policy": phase_spec.policy,
        # The effective field is the only one that should be interpreted as
        # actual residual injection.  For legacy A3.8 free-drive it is the
        # terminal call even though the probe observes every DPM call.
        "phase_active_call_indices": list(effective_residual_active_call_indices),
        "phase_probe_active_call_indices": (
            "all" if phase_spec.active_call_indices is None else list(phase_spec.active_call_indices)
        ),
        "phase_residual_gate_source": (
            "phase_override"
            if phase_spec.override_style_residual_gate
            else "checkpoint"
        ),
        "input_axis_index": int(axis_index),
        "input_axis_name": axis_name,
        "axis_order": list(CANONICAL_AXIS_BY_SCENE[scene]),
        "delta": float(delta),
        "normal_condition": normal_condition.tolist(),
        "plus_condition": plus_condition.tolist(),
        "minus_condition": minus_condition.tolist(),
        "causal_axis_mask": causal_mask.astype(bool).tolist(),
        "fixed_input_axis_mask": fixed_input_axis_mask.astype(bool).tolist(),
        "normal_axis_canonical": _optional_vector(np.asarray(normal_axis["canonical"])),
        "normal_axis_percentile": _optional_vector(np.asarray(normal_axis["percentile"])),
        "normal_axis_valid_mask": np.asarray(normal_axis["valid"], dtype=bool).tolist(),
        "plus_axis_canonical": _optional_vector(plus_canonical),
        "minus_axis_canonical": _optional_vector(minus_canonical),
        "plus_axis_percentile": _optional_vector(plus_percentile),
        "minus_axis_percentile": _optional_vector(minus_percentile),
        "plus_axis_valid_mask": np.asarray(plus_axis["valid"], dtype=bool).tolist(),
        "minus_axis_valid_mask": np.asarray(minus_axis["valid"], dtype=bool).tolist(),
        "transport_canonical_vec": _optional_vector(canonical_transport),
        "transport_percentile_vec": _optional_vector(percentile_transport),
        "transport_valid_mask": valid.astype(bool).tolist(),
        "diagonal_transport": float(diagonal) if math.isfinite(float(diagonal)) else None,
        "diagonal_wrong_sign": (
            bool(diagonal <= 0.0) if math.isfinite(float(diagonal)) else None
        ),
        "cross_axis_leakage_ratio": leakage_ratio,
        "phase_trace": list(phase_trace),
        "plus_trajectory_metrics": dict(plus_metrics),
        "minus_trajectory_metrics": dict(minus_metrics),
        "paired_noise_contract": "same rollout seed reset immediately before plus/minus",
    }


def _candidate_context(
    sample: Mapping[str, Any],
    *,
    expected_scene: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene = str(sample["scene_bucket"])
    if scene != expected_scene:
        raise RuntimeError(f"candidate scene mismatch: expected {expected_scene}, got {scene}")
    causal_mask = _sample_causal_mask(sample)
    scene_gate = _sample_scene_gate(sample)
    normal_condition = _normal_condition(
        scene=scene,
        causal_mask=causal_mask,
        scene_gate=scene_gate,
    )
    if not np.array_equal(normal_condition[3:6] > 0.5, causal_mask):
        raise RuntimeError("Phase-0 normal condition changed the V6 causal axis mask")
    return causal_mask, scene_gate, normal_condition


def _select_eligible_samples(
    *,
    model: torch.nn.Module,
    model_args: argparse.Namespace,
    dataset: PreferenceConditionedPlannerData,
    args: argparse.Namespace,
    eligibility_solver_steps: int,
    fixed_normal_neighbors: bool,
    normal_anchor_accel_support: bool,
) -> tuple[Dict[str, list[Dict[str, Any]]], Dict[str, Any]]:
    candidates, _ = _select_indices(
        dataset,
        per_controlled_scene=int(args.max_samples_per_controlled_scene),
        eligible_search_multiplier=int(args.eligible_search_multiplier),
        lane_count=0,
        seed=int(args.seed),
    )
    _configure_solver_steps(model, eligibility_solver_steps)
    selected: Dict[str, list[Dict[str, Any]]] = {scene: [] for scene in CONTROLLED_SCENES}
    eligibility: Dict[str, Any] = {
        "eligibility_solver_steps": int(eligibility_solver_steps),
        "eligibility_denoiser_evaluations": int(eligibility_solver_steps) + 1,
        "candidate_counts": {scene: len(indices) for scene, indices in candidates.items()},
        "selected_counts": {},
        "rejected_no_causal_axis": {scene: 0 for scene in CONTROLLED_SCENES},
        "rejected_no_measurable_axis": {scene: 0 for scene in CONTROLLED_SCENES},
    }
    for scene, indices in candidates.items():
        for dataset_index in tqdm(indices, desc=f"phase0-eligibility:{scene}", dynamic_ncols=True):
            if len(selected[scene]) >= int(args.max_samples_per_controlled_scene):
                break
            sample = dataset[dataset_index]
            causal_mask, scene_gate, normal_condition = _candidate_context(
                sample,
                expected_scene=scene,
            )
            if not bool(causal_mask.any()):
                eligibility["rejected_no_causal_axis"][scene] += 1
                continue
            batch = _batch_from_sample(sample)
            base_inputs, _, _, _, _ = prepare_preference_conditioned_batch(
                batch,
                model_args,
                train=False,
                aug=None,
            )
            rollout_seed = int(args.seed + dataset_index * 10007)
            normal_prediction, normal_diagnostics, _ = _run_rollout(
                model=model,
                base_inputs=base_inputs,
                condition=normal_condition,
                normal_condition=normal_condition,
                normal_prediction=None,
                fixed_normal_neighbors=fixed_normal_neighbors,
                normal_anchor_accel_support=normal_anchor_accel_support,
                seed=rollout_seed,
                probe=None,
                expected_trace=None,
                phase_spec=None,
                effective_residual_active_call_indices=None,
            )
            del normal_prediction
            normal_axis = _axis_payload(normal_diagnostics)
            fixed_input_mask = causal_mask & normal_axis["valid"]
            if not bool(fixed_input_mask.any()):
                eligibility["rejected_no_measurable_axis"][scene] += 1
                continue
            selected[scene].append(
                {
                    "dataset_index": int(dataset_index),
                    "sample_id": str(sample["sample_id"]),
                    "scene_bucket": scene,
                    "causal_axis_mask": causal_mask.astype(bool).tolist(),
                    "fixed_input_axis_mask": fixed_input_mask.astype(bool).tolist(),
                    "scene_gate_values": scene_gate.tolist(),
                    "normal_condition": normal_condition.tolist(),
                    "eligibility_normal_axis_canonical": _optional_vector(
                        np.asarray(normal_axis["canonical"])
                    ),
                    "eligibility_normal_axis_valid_mask": np.asarray(
                        normal_axis["valid"], dtype=bool
                    ).tolist(),
                }
            )
        eligibility["selected_counts"][scene] = len(selected[scene])
    missing = [scene for scene, values in selected.items() if not values]
    if missing:
        raise RuntimeError(
            "No Phase-0 eligible samples after normal-axis measurement for "
            f"{missing}; inspect {ELIGIBILITY_NAME}"
        )
    return selected, eligibility


def _stable_seed(*parts: object) -> int:
    source = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(source).digest()[:8], byteorder="big")


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None, None, None
    mean = float(np.mean(array))
    if array.size == 1 or int(replicates) <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(replicates), array.size))
    means = np.mean(array[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high)


def _scenario_values(
    records: Iterable[Mapping[str, Any]],
    *,
    value_getter,
) -> tuple[list[float], int]:
    by_sample: Dict[str, list[float]] = defaultdict(list)
    seed_pairs = 0
    for record in records:
        value = value_getter(record)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        by_sample[str(record["sample_id"])].append(numeric)
        seed_pairs += 1
    return [float(np.mean(values)) for values in by_sample.values() if values], seed_pairs


def _summarize_pairs(
    *,
    pair_path: Path,
    contract_path: Path,
    output_root: Path,
    bootstrap_replicates: int,
    seed: int,
) -> Dict[str, Any]:
    pairs = _read_jsonl(pair_path)
    contracts = _read_jsonl(contract_path) if contract_path.exists() else []
    tensor_groups: Dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    leakage_groups: Dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in pairs:
        base_key = (
            str(record["scene_bucket"]),
            int(record["solver_steps"]),
            int(record["denoiser_evaluations"]),
            str(record["phase_policy"]),
            str(record["phase_label"]),
            int(record["input_axis_index"]),
        )
        for output_axis in range(3):
            tensor_groups[base_key + (output_axis,)].append(record)
        leakage_groups[base_key].append(record)

    tensor_rows: list[Dict[str, Any]] = []
    for key in sorted(tensor_groups):
        scene, solver_steps, evaluation_count, policy, phase, input_axis, output_axis = key
        records = tensor_groups[key]
        values, seed_pair_count = _scenario_values(
            records,
            value_getter=lambda row, index=output_axis: (
                row.get("transport_canonical_vec", [None, None, None])[index]
            ),
        )
        mean, ci_low, ci_high = _bootstrap_mean_ci(
            values,
            replicates=bootstrap_replicates,
            seed=_stable_seed(seed, *key),
        )
        diagonal_wrong_sign_rate = None
        if input_axis == output_axis:
            wrong_values, _ = _scenario_values(
                records,
                value_getter=lambda row: 1.0
                if row.get("diagonal_wrong_sign") is True
                else (0.0 if row.get("diagonal_wrong_sign") is False else None),
            )
            diagonal_wrong_sign_rate = (
                float(np.mean(wrong_values)) if wrong_values else None
            )
        axis_order = CANONICAL_AXIS_BY_SCENE[scene]
        tensor_rows.append(
            {
                "scene_bucket": scene,
                "solver_steps": solver_steps,
                "denoiser_evaluations": evaluation_count,
                "phase_policy": policy,
                "phase_label": phase,
                "input_axis_index": input_axis,
                "input_axis_name": axis_order[input_axis],
                "output_axis_index": output_axis,
                "output_axis_name": axis_order[output_axis],
                "is_diagonal": input_axis == output_axis,
                "mean_transport": mean,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "scenario_count": len(values),
                "seed_pair_count": seed_pair_count,
                "diagonal_wrong_sign_rate": diagonal_wrong_sign_rate,
            }
        )

    leakage_rows: list[Dict[str, Any]] = []
    for key in sorted(leakage_groups):
        scene, solver_steps, evaluation_count, policy, phase, input_axis = key
        records = leakage_groups[key]
        values, seed_pair_count = _scenario_values(
            records,
            value_getter=lambda row: row.get("cross_axis_leakage_ratio"),
        )
        mean, ci_low, ci_high = _bootstrap_mean_ci(
            values,
            replicates=bootstrap_replicates,
            seed=_stable_seed(seed, "leakage", *key),
        )
        leakage_rows.append(
            {
                "scene_bucket": scene,
                "solver_steps": solver_steps,
                "denoiser_evaluations": evaluation_count,
                "phase_policy": policy,
                "phase_label": phase,
                "input_axis_index": input_axis,
                "input_axis_name": CANONICAL_AXIS_BY_SCENE[scene][input_axis],
                "mean_cross_axis_leakage_ratio": mean,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "scenario_count": len(values),
                "seed_pair_count": seed_pair_count,
            }
        )

    contract_rows: list[Dict[str, Any]] = []
    for record in contracts:
        contract_rows.append(dict(record))
    contract_counts_by_type: Dict[str, int] = defaultdict(int)
    contract_failures_by_type: Dict[str, int] = defaultdict(int)
    for row in contract_rows:
        contract_type = str(row.get("contract_type", "checkpoint_gate_vs_phase_override"))
        contract_counts_by_type[contract_type] += 1
        if row.get("passes", False) is False:
            contract_failures_by_type[contract_type] += 1
    _write_csv(output_root / TENSOR_CSV_NAME, tensor_rows)
    _write_csv(output_root / LEAKAGE_CSV_NAME, leakage_rows)
    _write_csv(output_root / CONTRACT_CSV_NAME, contract_rows)
    return {
        "pair_row_count": len(pairs),
        "tensor_rows": tensor_rows,
        "leakage_rows": leakage_rows,
        "legacy_equivalence_rows": contract_rows,
        "legacy_equivalence_failure_count": sum(
            int(row.get("passes", False) is False) for row in contract_rows
        ),
        "contract_row_counts_by_type": dict(contract_counts_by_type),
        "contract_failure_counts_by_type": dict(contract_failures_by_type),
        "artifacts": {
            "pairs": str(pair_path),
            "tensor_csv": str(output_root / TENSOR_CSV_NAME),
            "leakage_csv": str(output_root / LEAKAGE_CSV_NAME),
            "legacy_equivalence_csv": str(output_root / CONTRACT_CSV_NAME),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _selftest() -> Dict[str, Any]:
    # Mirror the float32 planner-facing condition contract so the self-test
    # does not confuse an intentional dtype cast of continuous scene gates
    # with a mutated 12D condition.
    normal = np.array(
        [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.8, 0.0, 0.0],
        dtype=np.float32,
    )
    plus = _single_axis_condition(normal, axis_index=1, sign=1, delta=0.2)
    minus = _single_axis_condition(normal, axis_index=1, sign=-1, delta=0.2)
    finite_difference = (np.array([0.54, 0.70, 0.45]) - np.array([0.46, 0.30, 0.35])) / 0.4
    trace = _expected_logsnr_trace(4)
    phases = _parse_phase_specs("legacy,all,early,middle,late,terminal,call:2", solver_steps=4)
    report = {
        "sampling_probe": selftest_transport_injection_probe(),
        "condition_contract": {
            "plus_minus_only_change_target_axis": bool(
                np.array_equal(plus[3:], normal[3:])
                and np.array_equal(minus[3:], normal[3:])
                and np.isclose(float(plus[1]), 0.7, atol=1e-6, rtol=0.0)
                and np.isclose(float(minus[1]), 0.3, atol=1e-6, rtol=0.0)
                and np.allclose(np.delete(plus[:3], 1), 0.5)
                and np.allclose(np.delete(minus[:3], 1), 0.5)
            ),
        },
        "finite_difference_contract": {
            "diagonal": float(finite_difference[1]),
            "expected_diagonal": 1.0,
            "cross_axis_leakage_ratio": float(
                (abs(finite_difference[0]) + abs(finite_difference[2]))
                / abs(finite_difference[1])
            ),
            "wrong_sign": bool(finite_difference[1] <= 0.0),
        },
        "schedule_contract": {
            "model_evaluation_count": len(trace),
            "terminal_is_last": trace[-1]["is_denoise_to_zero"] is True,
            "phase_labels": [spec.label for spec in phases],
            "terminal_selector": [
                spec.active_call_indices
                for spec in phases
                if spec.label == "terminal"
            ][0]
            == (4,),
        },
    }
    report["pass"] = bool(
        all(bool(value) for value in report["sampling_probe"].values())
        and report["condition_contract"]["plus_minus_only_change_target_axis"]
        and abs(report["finite_difference_contract"]["diagonal"] - 1.0) < 1e-8
        and not report["finite_difference_contract"]["wrong_sign"]
        and report["schedule_contract"]["terminal_is_last"]
        and report["schedule_contract"]["terminal_selector"]
    )
    return report


def _validate_args(args: argparse.Namespace) -> None:
    required = {
        "--experiment-dir": args.experiment_dir,
        "--checkpoint-path": args.checkpoint_path,
        "--split-root": args.split_root,
        "--cache-dir": args.cache_dir,
        "--condition-index-path": args.condition_index_path,
        "--output-root": args.output_root,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")
    if args.max_samples_per_controlled_scene <= 0:
        raise ValueError("--max-samples-per-controlled-scene must be positive")
    if args.eligible_search_multiplier <= 0:
        raise ValueError("--eligible-search-multiplier must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if not 0.0 < float(args.delta) <= 0.5:
        raise ValueError("--delta must be in (0, 0.5]")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative")
    if args.legacy_equivalence_tolerance < 0.0:
        raise ValueError("--legacy-equivalence-tolerance must be non-negative")


def main() -> None:
    args = _parser().parse_args()
    if args.self_test:
        print(json.dumps(_selftest(), ensure_ascii=False, indent=2, sort_keys=True))
        return
    _validate_args(args)
    if args.disable_prefer_ema:
        args.prefer_ema = False
    solver_steps_values = _resolve_solver_steps(args)
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    model_args = _model_args(args)
    # Phase-0 must use only one conditional forward per logical evaluation so
    # probe indices are model-evaluation indices, not CFG branch counts.
    model_args.cfg_guidance_scale = 1.0
    model, checkpoint_meta = _load_model(
        model_args,
        checkpoint_path,
        prefer_ema=bool(args.prefer_ema),
        flexible=False,
    )
    _validate_phase0_model(model, model_args)
    sampling_device = next(model.parameters()).device
    # Construct the expected logSNR trace on the same device used by DPM-Solver.
    # CPU and CUDA schedules are semantically identical but can differ slightly
    # in floating-point roundoff, which should never invalidate a trace audit.
    phase_specs_by_steps = {
        steps: _parse_phase_specs(
            args.phase_windows,
            solver_steps=steps,
            device=sampling_device,
        )
        for steps in solver_steps_values
    }
    if args.require_legacy_equivalence:
        expected_equivalents = {
            phase
            for scene in CONTROLLED_SCENES
            for phase in (_checkpoint_equivalent_phase(scene, model),)
            if phase is not None
        }
        if not expected_equivalents:
            raise ValueError(
                "--require-legacy-equivalence is only supported for the known "
                "all_steps or free_drive_terminal_only execution gates"
            )
        required_phase_labels = {"legacy", *expected_equivalents}
        for solver_steps, phase_specs in phase_specs_by_steps.items():
            configured = {spec.label for spec in phase_specs}
            missing = sorted(required_phase_labels - configured)
            if missing:
                raise ValueError(
                    "--require-legacy-equivalence requires phase selectors "
                    f"{sorted(required_phase_labels)} at solver_steps={solver_steps}; "
                    f"missing {missing}"
                )
    fixed_normal_neighbors = (
        str(getattr(model_args, "preference_axis_reference_mode", "self_generated"))
        == "normal_neighbor"
    )
    normal_anchor_accel_support = (
        str(getattr(model_args, "free_drive_accel_support_mode", "self_generated"))
        == "normal_anchor"
    )

    # Keep one dataset instance for both eligibility and rollout.  This makes
    # the recorded ``dataset_index`` identity unambiguous and avoids a second
    # sidecar/cache scan after the selected rows have been frozen.
    dataset = PreferenceConditionedPlannerData(
        cache_dir=str(args.cache_dir),
        split_root=str(args.split_root),
        condition_field="style_value_condition",
        conditioning_index_override=str(args.condition_index_path),
    )
    selected, eligibility = _select_eligible_samples(
        model=model,
        model_args=model_args,
        dataset=dataset,
        args=args,
        eligibility_solver_steps=solver_steps_values[0],
        fixed_normal_neighbors=fixed_normal_neighbors,
        normal_anchor_accel_support=normal_anchor_accel_support,
    )
    selected_payload = {
        "artifact": "phase0_transport_selected_samples",
        "selected": selected,
        "eligibility": eligibility,
    }
    _write_json(output_root / SELECTED_NAME, selected_payload)
    _write_json(output_root / ELIGIBILITY_NAME, eligibility)
    selected_sample_ids = {
        str(entry["sample_id"])
        for entries in selected.values()
        for entry in entries
    }
    selected_condition_path = output_root / SELECTED_CONDITIONS_NAME
    _write_condition_subset(
        source_path=args.condition_index_path,
        output_path=selected_condition_path,
        sample_ids=selected_sample_ids,
    )

    manifest: Dict[str, Any] = {
        "artifact": "phase0_stylization_transport_tensor",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": str(args.device),
        "checkpoint": {
            **checkpoint_meta,
            "sha256": _sha256(checkpoint_path),
        },
        "condition_index_path": str(args.condition_index_path),
        "selected_condition_index_path": str(selected_condition_path),
        "solver_steps": solver_steps_values,
        "denoiser_evaluations": [steps + 1 for steps in solver_steps_values],
        "phase_specs": {
            str(steps): [
                {
                    "label": spec.label,
                    "policy": spec.policy,
                    "probe_active_call_indices": (
                        "all" if spec.active_call_indices is None else list(spec.active_call_indices)
                    ),
                    "override_style_residual_gate": spec.override_style_residual_gate,
                }
                for spec in specs
            ]
            for steps, specs in phase_specs_by_steps.items()
        },
        "delta": float(args.delta),
        "num_seeds": int(args.num_seeds),
        "seed_base": int(args.seed),
        "cfg_guidance_scale": 1.0,
        "normal_anchor_cfg_enabled": False,
        "checkpoint_signed_residual_gate": {
            "mode": str(
                getattr(_phase0_dit(model), "signed_router_diffusion_gate_mode", "")
            ),
            "terminal_t_max": float(
                getattr(_phase0_dit(model), "signed_router_terminal_t_max", float("nan"))
            ),
        },
        "checkpoint_scene_axis_executor": {
            "style_condition_encoder": str(
                getattr(model_args, "style_condition_encoder", "")
            ),
            "signed_router_injection_mode": str(
                getattr(_phase0_dit(model), "signed_router_injection_mode", "")
            ),
            "car_follow_legacy_acceleration_fallback": bool(
                str(
                    getattr(_phase0_dit(model), "signed_router_injection_mode", "")
                )
                != "ego_scene_axis_temporal_residual"
            ),
        },
        "fixed_normal_neighbors": fixed_normal_neighbors,
        "normal_anchor_accel_support": normal_anchor_accel_support,
        "transport_measurement": {
            "axis_source": "ConditionalPreferenceEnergy generated_axis_canonical",
            "layout": "T[output_axis_k, input_axis_j, phase]",
            "finite_difference": "(u(+delta e_j)-u(-delta e_j))/(2*delta)",
            "statistical_unit": "scenario/sample_id after averaging seed replicas",
        },
    }
    _write_json(output_root / MANIFEST_NAME, manifest)

    rollout_path = output_root / RAW_ROLLOUT_NAME
    pair_path = output_root / PAIR_NAME
    contract_path = output_root / "phase0_legacy_equivalence.jsonl"
    contract_count = 0
    legacy_path_regression_count = 0
    with gzip.open(rollout_path, "wt", encoding="utf-8") as rollout_file, open(
        pair_path, "w", encoding="utf-8"
    ) as pair_file, open(contract_path, "w", encoding="utf-8") as contract_file:
        total_samples = sum(len(entries) for entries in selected.values())
        progress = tqdm(
            total=total_samples * len(solver_steps_values) * int(args.num_seeds),
            desc="phase0-transport",
            dynamic_ncols=True,
        )
        try:
            for scene, entries in selected.items():
                axis_order = CANONICAL_AXIS_BY_SCENE[scene]
                for entry_index, entry in enumerate(entries):
                    dataset_index = int(entry["dataset_index"])
                    sample = dataset[dataset_index]
                    causal_mask, scene_gate, normal_condition = _candidate_context(
                        sample,
                        expected_scene=scene,
                    )
                    stored_mask = _as_bool_vector(entry["fixed_input_axis_mask"])
                    if not np.array_equal(causal_mask, _as_bool_vector(entry["causal_axis_mask"])):
                        raise RuntimeError("selected sample causal mask changed after selection")
                    batch = _batch_from_sample(sample)
                    base_inputs, _, _, _, _ = prepare_preference_conditioned_batch(
                        batch,
                        model_args,
                        train=False,
                        aug=None,
                    )
                    for solver_steps in solver_steps_values:
                        _configure_solver_steps(model, solver_steps)
                        expected_trace = _expected_logsnr_trace(
                            solver_steps,
                            device=sampling_device,
                        )
                        phase_specs = phase_specs_by_steps[solver_steps]
                        for seed_index in range(int(args.num_seeds)):
                            rollout_seed = int(
                                args.seed + dataset_index * 10007 + seed_index
                            )
                            normal_prediction, normal_diagnostics, _ = _run_rollout(
                                model=model,
                                base_inputs=base_inputs,
                                condition=normal_condition,
                                normal_condition=normal_condition,
                                normal_prediction=None,
                                fixed_normal_neighbors=fixed_normal_neighbors,
                                normal_anchor_accel_support=normal_anchor_accel_support,
                                seed=rollout_seed,
                                probe=None,
                                expected_trace=None,
                                phase_spec=None,
                                effective_residual_active_call_indices=None,
                            )
                            normal_axis = _axis_payload(normal_diagnostics)
                            regression_axis_index = (
                                int(np.flatnonzero(stored_mask)[0])
                                if entry_index == 0 and seed_index == 0
                                else None
                            )
                            for axis_index, axis_name in enumerate(axis_order):
                                if not bool(stored_mask[axis_index]):
                                    continue
                                phase_predictions: Dict[tuple[str, int], np.ndarray] = {}
                                plus_condition = _single_axis_condition(
                                    normal_condition,
                                    axis_index=axis_index,
                                    sign=1,
                                    delta=float(args.delta),
                                )
                                minus_condition = _single_axis_condition(
                                    normal_condition,
                                    axis_index=axis_index,
                                    sign=-1,
                                    delta=float(args.delta),
                                )
                                for phase_spec in phase_specs:
                                    effective_residual_calls = (
                                        _effective_residual_active_call_indices(
                                            scene,
                                            model=model,
                                            expected_trace=expected_trace,
                                            phase_spec=phase_spec,
                                        )
                                    )
                                    branch: Dict[int, Dict[str, Any]] = {}
                                    for sign, condition in (
                                        (1, plus_condition),
                                        (-1, minus_condition),
                                    ):
                                        probe = TransportInjectionProbe(
                                            active_call_indices=phase_spec.active_call_indices,
                                            label=phase_spec.label,
                                            override_style_residual_gate=(
                                                phase_spec.override_style_residual_gate
                                            ),
                                        )
                                        prediction, diagnostics, phase_trace = _run_rollout(
                                            model=model,
                                            base_inputs=base_inputs,
                                            condition=condition,
                                            normal_condition=normal_condition,
                                            normal_prediction=normal_prediction,
                                            fixed_normal_neighbors=fixed_normal_neighbors,
                                            normal_anchor_accel_support=normal_anchor_accel_support,
                                            seed=rollout_seed,
                                            probe=probe,
                                            expected_trace=expected_trace,
                                            phase_spec=phase_spec,
                                            effective_residual_active_call_indices=(
                                                effective_residual_calls
                                            ),
                                        )
                                        if phase_trace is None:
                                            raise RuntimeError("transport rollout did not produce a probe trace")
                                        axis_payload = _axis_payload(diagnostics)
                                        metrics = _trajectory_metrics(
                                            prediction,
                                            sample,
                                            dt=float(args.dt),
                                        )
                                        branch[sign] = {
                                            "prediction": prediction,
                                            "diagnostics": diagnostics,
                                            "axis": axis_payload,
                                            "metrics": metrics,
                                            "trace": phase_trace,
                                            "condition": condition,
                                        }
                                        phase_predictions[(phase_spec.label, sign)] = prediction
                                        rollout_row = {
                                            "artifact": "phase0_stylization_transport_rollout",
                                            "checkpoint": checkpoint_meta,
                                            "scene_bucket": scene,
                                            "sample_id": str(sample["sample_id"]),
                                            "dataset_index": dataset_index,
                                            "seed": rollout_seed,
                                            "seed_index": seed_index,
                                            "solver_steps": solver_steps,
                                            "denoiser_evaluations": solver_steps + 1,
                                            "phase_label": phase_spec.label,
                                            "phase_policy": phase_spec.policy,
                                            "phase_active_call_indices": list(
                                                effective_residual_calls
                                            ),
                                            "phase_probe_active_call_indices": (
                                                "all"
                                                if phase_spec.active_call_indices is None
                                                else list(phase_spec.active_call_indices)
                                            ),
                                            "phase_residual_gate_source": (
                                                "phase_override"
                                                if phase_spec.override_style_residual_gate
                                                else "checkpoint"
                                            ),
                                            "input_axis_index": axis_index,
                                            "input_axis_name": axis_name,
                                            "command_sign": sign,
                                            "delta": float(args.delta),
                                            "style_condition": condition.tolist(),
                                            "normal_condition": normal_condition.tolist(),
                                            "causal_axis_mask": causal_mask.astype(bool).tolist(),
                                            "fixed_input_axis_mask": stored_mask.astype(bool).tolist(),
                                            "normal_axis_canonical": _optional_vector(
                                                np.asarray(normal_axis["canonical"])
                                            ),
                                            "normal_axis_valid_mask": np.asarray(
                                                normal_axis["valid"], dtype=bool
                                            ).tolist(),
                                            "generated_axis_canonical": _optional_vector(
                                                np.asarray(axis_payload["canonical"])
                                            ),
                                            "generated_axis_percentile": _optional_vector(
                                                np.asarray(axis_payload["percentile"])
                                            ),
                                            "generated_axis_valid_mask": np.asarray(
                                                axis_payload["valid"], dtype=bool
                                            ).tolist(),
                                            "trajectory_metrics": metrics,
                                            "phase_trace": phase_trace,
                                            "router_gate": diagnostics.get("axis_router_gate", []),
                                            "router_axis_residual_l2": diagnostics.get(
                                                "axis_router_axis_residual_l2", []
                                            ),
                                            "paired_noise_seed": rollout_seed,
                                        }
                                        rollout_file.write(
                                            json.dumps(_json_safe(rollout_row), ensure_ascii=False)
                                            + "\n"
                                        )
                                    pair_row = _build_pair_row(
                                        checkpoint_meta=checkpoint_meta,
                                        scene=scene,
                                        sample_id=str(sample["sample_id"]),
                                        dataset_index=dataset_index,
                                        seed=rollout_seed,
                                        seed_index=seed_index,
                                        solver_steps=solver_steps,
                                        phase_spec=phase_spec,
                                        axis_index=axis_index,
                                        axis_name=axis_name,
                                        delta=float(args.delta),
                                        normal_axis=normal_axis,
                                        plus_axis=branch[1]["axis"],
                                        minus_axis=branch[-1]["axis"],
                                        normal_condition=normal_condition,
                                        plus_condition=plus_condition,
                                        minus_condition=minus_condition,
                                        causal_mask=causal_mask,
                                        fixed_input_axis_mask=stored_mask,
                                        phase_trace=branch[1]["trace"],
                                        effective_residual_active_call_indices=(
                                            effective_residual_calls
                                        ),
                                        plus_metrics=branch[1]["metrics"],
                                        minus_metrics=branch[-1]["metrics"],
                                    )
                                    pair_file.write(
                                        json.dumps(_json_safe(pair_row), ensure_ascii=False)
                                        + "\n"
                                    )

                                    # Directly prove that adding the observer
                                    # wrapper for legacy tracing does not alter
                                    # the pre-Phase-0, no-probe inference path.
                                    # One eligible axis and both command signs
                                    # per scene/NFE are sufficient as a cheap
                                    # regression contract; the full transport
                                    # grid remains the actual measurement.
                                    if (
                                        phase_spec.label == "legacy"
                                        and regression_axis_index == axis_index
                                    ):
                                        for sign, condition in (
                                            (-1, minus_condition),
                                            (1, plus_condition),
                                        ):
                                            no_probe_prediction, _, _ = _run_rollout(
                                                model=model,
                                                base_inputs=base_inputs,
                                                condition=condition,
                                                normal_condition=normal_condition,
                                                normal_prediction=normal_prediction,
                                                fixed_normal_neighbors=fixed_normal_neighbors,
                                                normal_anchor_accel_support=(
                                                    normal_anchor_accel_support
                                                ),
                                                seed=rollout_seed,
                                                probe=None,
                                                expected_trace=None,
                                                phase_spec=None,
                                                effective_residual_active_call_indices=None,
                                            )
                                            max_abs = _trajectory_max_abs(
                                                no_probe_prediction,
                                                branch[sign]["prediction"],
                                            )
                                            contract_file.write(
                                                json.dumps(
                                                    _json_safe(
                                                        {
                                                            "contract_type": (
                                                                "no_probe_vs_legacy_observer_probe"
                                                            ),
                                                            "scene_bucket": scene,
                                                            "sample_id": str(
                                                                sample["sample_id"]
                                                            ),
                                                            "dataset_index": dataset_index,
                                                            "seed": rollout_seed,
                                                            "seed_index": seed_index,
                                                            "solver_steps": solver_steps,
                                                            "denoiser_evaluations": (
                                                                solver_steps + 1
                                                            ),
                                                            "input_axis_index": axis_index,
                                                            "input_axis_name": axis_name,
                                                            "command_sign": sign,
                                                            "expected_equivalent_phase": (
                                                                "no_probe"
                                                            ),
                                                            "max_abs_trajectory_difference": max_abs,
                                                            "ego_ade_difference": _ego_ade(
                                                                no_probe_prediction,
                                                                branch[sign]["prediction"],
                                                            ),
                                                            "tolerance": float(
                                                                args.legacy_equivalence_tolerance
                                                            ),
                                                            "passes": max_abs
                                                            <= float(
                                                                args.legacy_equivalence_tolerance
                                                            ),
                                                        }
                                                    ),
                                                    ensure_ascii=False,
                                                )
                                                + "\n"
                                            )
                                            contract_count += 1
                                            legacy_path_regression_count += 1

                                expected_phase = _checkpoint_equivalent_phase(
                                    scene,
                                    model,
                                )
                                if expected_phase is not None and "legacy" in {
                                    spec.label for spec in phase_specs
                                }:
                                    for sign in (-1, 1):
                                        legacy_prediction = phase_predictions.get(("legacy", sign))
                                        equivalent_prediction = phase_predictions.get(
                                            (expected_phase, sign)
                                        )
                                        if legacy_prediction is None or equivalent_prediction is None:
                                            contract_row = {
                                                "contract_type": "checkpoint_gate_vs_phase_override",
                                                "scene_bucket": scene,
                                                "sample_id": str(sample["sample_id"]),
                                                "dataset_index": dataset_index,
                                                "seed": rollout_seed,
                                                "seed_index": seed_index,
                                                "solver_steps": solver_steps,
                                                "denoiser_evaluations": solver_steps + 1,
                                                "input_axis_index": axis_index,
                                                "input_axis_name": axis_name,
                                                "command_sign": sign,
                                                "expected_equivalent_phase": expected_phase,
                                                "passes": False,
                                                "reason": "missing legacy or equivalent phase rollout",
                                            }
                                        else:
                                            max_abs = _trajectory_max_abs(
                                                legacy_prediction,
                                                equivalent_prediction,
                                            )
                                            contract_row = {
                                                "contract_type": "checkpoint_gate_vs_phase_override",
                                                "scene_bucket": scene,
                                                "sample_id": str(sample["sample_id"]),
                                                "dataset_index": dataset_index,
                                                "seed": rollout_seed,
                                                "seed_index": seed_index,
                                                "solver_steps": solver_steps,
                                                "denoiser_evaluations": solver_steps + 1,
                                                "input_axis_index": axis_index,
                                                "input_axis_name": axis_name,
                                                "command_sign": sign,
                                                "expected_equivalent_phase": expected_phase,
                                                "max_abs_trajectory_difference": max_abs,
                                                "ego_ade_difference": _ego_ade(
                                                    legacy_prediction,
                                                    equivalent_prediction,
                                                ),
                                                "tolerance": float(args.legacy_equivalence_tolerance),
                                                "passes": max_abs
                                                <= float(args.legacy_equivalence_tolerance),
                                            }
                                        contract_file.write(
                                            json.dumps(_json_safe(contract_row), ensure_ascii=False)
                                            + "\n"
                                        )
                                        contract_count += 1
                            progress.update(1)
        finally:
            progress.close()

    summary = _summarize_pairs(
        pair_path=pair_path,
        contract_path=contract_path,
        output_root=output_root,
        bootstrap_replicates=int(args.bootstrap_replicates),
        seed=int(args.seed),
    )
    summary.update(
        {
            "artifact": "phase0_stylization_transport_tensor_summary",
            "checkpoint": manifest["checkpoint"],
            "manifest_path": str(output_root / MANIFEST_NAME),
            "raw_rollouts": str(rollout_path),
            "selected_samples": str(output_root / SELECTED_NAME),
            "eligibility": eligibility,
            "legacy_equivalence_row_count": contract_count,
            "legacy_path_regression_row_count": legacy_path_regression_count,
        }
    )
    _write_json(output_root / SUMMARY_NAME, summary)
    print(f"[Phase0] summary={output_root / SUMMARY_NAME}")
    print(f"[Phase0] tensor_csv={output_root / TENSOR_CSV_NAME}")
    print(f"[Phase0] raw_rollouts={rollout_path}")
    if args.require_legacy_equivalence and contract_count == 0:
        raise RuntimeError(
            "legacy-equivalence was requested but no eligible paired contract "
            "rows were produced"
        )
    expected_path_regressions = 2 * len(CONTROLLED_SCENES) * len(solver_steps_values)
    if (
        args.require_legacy_equivalence
        and legacy_path_regression_count != expected_path_regressions
    ):
        raise RuntimeError(
            "legacy observer-path regression contract was incomplete: "
            f"expected {expected_path_regressions}, got {legacy_path_regression_count}"
        )
    if args.require_legacy_equivalence and summary["legacy_equivalence_failure_count"]:
        raise RuntimeError(
            "legacy-equivalence contract failed; inspect "
            f"{output_root / CONTRACT_CSV_NAME}"
        )


if __name__ == "__main__":
    main()

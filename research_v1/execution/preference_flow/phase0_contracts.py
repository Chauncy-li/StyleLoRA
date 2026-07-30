"""Read-only structural contracts for the approved preference-flow Phase 0.

Phase 0 locks the existing StylePlanner baseline without changing its planner
outputs.  In particular, this module must not introduce a preference editor,
dual DPM stream, trajectory warp, content curve, or training loss.
"""

from __future__ import annotations

import ast
import importlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PHASE0_SCHEMA_VERSION = "preference_flow_phase0_v1"

SERVER_REPO_ROOT_DEFAULT = Path("/home/lisw/programs/Nuplan-Diffusion-Baseline")
SERVER_RECORD_ROOT_DEFAULT = Path("/mnt/mydata/lishangwen/Nuplan-Baseline-Record")
PREFERENCE_FLOW_ROOT_NAME = "preference_flow_v1"
A3_8_CHECKPOINT_ROOT_DEFAULT = (
    SERVER_RECORD_ROOT_DEFAULT
    / PREFERENCE_FLOW_ROOT_NAME
    / "checkpoints"
    / "a3_8"
)

CONTROLLED_SCENES = ("straight_free_drive", "straight_car_follow")
EXPECTED_CANONICAL_AXES = {
    "straight_free_drive": (
        "speed_utilization",
        "accel_willingness",
        "speed_response_intensity",
    ),
    "straight_car_follow": (
        "headway_tightness_from_h",
        "ttc_tightness",
        "closing_tolerance",
    ),
}
DEFAULT_RHO_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)


class Phase0ContractError(RuntimeError):
    """Raised when a Phase-0 structural invariant is violated."""


@dataclass(frozen=True)
class Phase0Paths:
    """Resolved server-facing paths recorded by the Phase-0 manifest."""

    repo_root: Path
    record_root: Path
    preference_flow_root: Path
    a3_8_checkpoint_root: Path
    dataset_split_path: Path
    default_config_path: Path
    smoke_cohort_path: Path
    default_manifest_path: Path

    def to_json_dict(self) -> dict[str, str]:
        return {
            "repo_root": self.repo_root.as_posix(),
            "record_root": self.record_root.as_posix(),
            "preference_flow_root": self.preference_flow_root.as_posix(),
            "a3_8_checkpoint_root": self.a3_8_checkpoint_root.as_posix(),
            "dataset_split_path": self.dataset_split_path.as_posix(),
            "default_config_path": self.default_config_path.as_posix(),
            "smoke_cohort_path": self.smoke_cohort_path.as_posix(),
            "default_manifest_path": self.default_manifest_path.as_posix(),
        }


def _path_from_value(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default
    text = str(value).strip()
    return Path(text) if text else default


def resolve_phase0_paths(
    *,
    repo_root: str | Path | None = None,
    record_root: str | Path | None = None,
    preference_flow_root: str | Path | None = None,
    a3_8_checkpoint_root: str | Path | None = None,
    dataset_split_path: str | Path | None = None,
    config_path: str | Path | None = None,
    smoke_cohort_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Phase0Paths:
    """Resolve Phase-0 paths without probing data or creating directories.

    Environment overrides are intentionally limited to the existing server-root
    variables.  No Windows data path is a default or fallback in this module.
    """

    resolved_repo_root = _path_from_value(
        repo_root,
        Path(os.environ.get("NUPLAN_SERVER_REPO_ROOT", str(SERVER_REPO_ROOT_DEFAULT))),
    )
    resolved_record_root = _path_from_value(
        record_root,
        Path(os.environ.get("NUPLAN_RECORD_ROOT", str(SERVER_RECORD_ROOT_DEFAULT))),
    )
    resolved_preference_flow_root = _path_from_value(
        preference_flow_root,
        Path(
            os.environ.get(
                "NUPLAN_PREFERENCE_FLOW_ROOT",
                str(resolved_record_root / PREFERENCE_FLOW_ROOT_NAME),
            )
        ),
    )
    resolved_a3_8_root = _path_from_value(
        a3_8_checkpoint_root,
        Path(
            os.environ.get(
                "NUPLAN_A3_8_CHECKPOINT_ROOT",
                str(resolved_preference_flow_root / "checkpoints" / "a3_8"),
            )
        ),
    )
    default_split = (
        resolved_record_root
        / "CACHE"
        / "style_scene_split_straight_train_v2"
        / "split_index.jsonl"
    )
    default_config = resolved_repo_root / "baseline" / "config" / "planner" / "style_planner.yaml"
    default_cohort = (
        resolved_preference_flow_root
        / "cohorts"
        / "smoke_cohort_tokens.json"
    )
    default_manifest = (
        resolved_preference_flow_root
        / "manifests"
        / "phase0_manifest.json"
    )
    return Phase0Paths(
        repo_root=resolved_repo_root,
        record_root=resolved_record_root,
        preference_flow_root=resolved_preference_flow_root,
        a3_8_checkpoint_root=resolved_a3_8_root,
        dataset_split_path=_path_from_value(dataset_split_path, default_split),
        default_config_path=_path_from_value(config_path, default_config),
        smoke_cohort_path=_path_from_value(smoke_cohort_path, default_cohort),
        default_manifest_path=_path_from_value(manifest_path, default_manifest),
    )


def parse_rho_grid(value: str | Sequence[float] | None) -> tuple[float, ...]:
    """Parse and validate a reproducible preference-coordinate grid.

    The Phase-0 grid must include the neutral coordinate ``0`` and every value
    must lie in the current user-command domain ``[-1, 1]``.
    """

    if value is None:
        raw_values: Iterable[object] = DEFAULT_RHO_GRID
    elif isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw_values = value
    parsed: list[float] = []
    for raw_value in raw_values:
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise Phase0ContractError(f"rho grid contains a non-numeric value: {raw_value!r}") from exc
        if not math.isfinite(number) or number < -1.0 or number > 1.0:
            raise Phase0ContractError(f"rho grid value must be finite and in [-1, 1], got {raw_value!r}")
        parsed.append(float(number))
    if not parsed:
        raise Phase0ContractError("rho grid must contain at least one value")
    if not any(abs(number) <= 1e-12 for number in parsed):
        raise Phase0ContractError("rho grid must include the neutral coordinate 0.0")
    return tuple(parsed)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _literal_assignment(path: Path, name: str) -> Any:
    """Read a literal module assignment without importing runtime dependencies."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value = node.value
        if target_name == name and value is not None:
            return ast.literal_eval(value)
    raise Phase0ContractError(f"Unable to find literal assignment {name!r} in {path}")


def _source_defines_symbol(path: Path, symbol_name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol_name:
            return True
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == symbol_name for target in node.targets):
                return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == symbol_name:
            return True
    return False


def _check_axis_schema() -> dict[str, Any]:
    schema_path = _repository_root() / "research_v1" / "stylization" / "schema.py"
    commands_path = _repository_root() / "research_v1" / "stylization" / "commands.py"
    canonical_axes = dict(_literal_assignment(schema_path, "CANONICAL_AXIS_BY_SCENE"))
    controlled_scenes = tuple(_literal_assignment(commands_path, "CONTROLLED_SCENES"))
    actual = {scene: tuple(canonical_axes.get(scene, ())) for scene in CONTROLLED_SCENES}
    passed = actual == EXPECTED_CANONICAL_AXES and controlled_scenes == CONTROLLED_SCENES
    return {
        "passed": passed,
        "expected": {scene: list(values) for scene, values in EXPECTED_CANONICAL_AXES.items()},
        "actual": {scene: list(values) for scene, values in actual.items()},
        "axis_count": sum(len(values) for values in actual.values()),
        "controlled_scenes": list(controlled_scenes),
    }


def _check_command_activation() -> dict[str, Any]:
    commands_path = _repository_root() / "research_v1" / "stylization" / "commands.py"
    layout = tuple(_literal_assignment(commands_path, "STYLE_CONDITION_LAYOUT"))
    source = commands_path.read_text(encoding="utf-8")
    expected_layout = (
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
    source_markers = {
        "causal_mask_is_read": "mask = _bool_vector(causal_axis_mask)" in source,
        "mask_is_serialized": "mask.astype(np.float64)" in source,
        "three_axis_condition_layout": layout == expected_layout,
    }
    reports = {
        scene: {
            "passed": len(EXPECTED_CANONICAL_AXES[scene]) == 3 and all(source_markers.values()),
            "active_axis_count": len(EXPECTED_CANONICAL_AXES[scene]),
            "causal_mask_slots": list(layout[3:6]),
            "condition_dim": len(layout),
        }
        for scene in CONTROLLED_SCENES
    }
    return {
        "passed": all(item["passed"] for item in reports.values()),
        "source_markers": source_markers,
        "scenes": reports,
    }


def _raw_input_keys_from_source(source: str) -> set[str]:
    bracket_keys = re.findall(r"raw_inputs\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", source)
    get_keys = re.findall(r"raw_inputs\.get\(\s*['\"]([^'\"]+)['\"]", source)
    mapping_values = re.findall(r"['\"][^'\"]+['\"]\s*:\s*['\"]([^'\"]+)['\"]", source)
    return {str(key) for key in (*bracket_keys, *get_keys, *mapping_values)}


def _check_runtime_router_causality() -> dict[str, Any]:
    root = _repository_root()
    sources = (
        (root / "research_v1" / "execution" / "runtime" / "online_preference.py").read_text(
            encoding="utf-8"
        ),
        (root / "research_v1" / "stylization" / "runtime.py").read_text(encoding="utf-8"),
    )
    input_keys: set[str] = set()
    for source in sources:
        input_keys.update(_raw_input_keys_from_source(source))
    forbidden = sorted(key for key in input_keys if "future" in key.lower())
    required_past_keys = {
        "ego_current_state",
        "ego_agent_past",
        "neighbor_agents_past",
        "neighbor_agents_past_mask",
    }
    missing_past = sorted(required_past_keys - input_keys)
    return {
        "passed": not forbidden and not missing_past,
        "raw_input_keys": sorted(input_keys),
        "forbidden_future_input_keys": forbidden,
        "missing_required_past_input_keys": missing_past,
    }


def _check_frozen_b3_imports(*, require_runtime_import: bool) -> dict[str, Any]:
    root = _repository_root()
    symbols = (
        (
            "baseline.model.style_planner.diffusion_planner",
            root / "baseline" / "model" / "style_planner" / "diffusion_planner.py",
            "Diffusion_Planner",
        ),
        (
            "baseline.model.style_planner.layer.decoder",
            root / "baseline" / "model" / "style_planner" / "layer" / "decoder.py",
            "Decoder",
        ),
        (
            "baseline.model.style_planner.layer.preference_axis_router",
            root / "baseline" / "model" / "style_planner" / "layer" / "preference_axis_router.py",
            "SceneAxisTemporalKinematicEgoSignedOutputAdapter",
        ),
        (
            "research_v1.execution.diffusion.train_stylized_diffusion",
            root / "research_v1" / "execution" / "diffusion" / "train_stylized_diffusion.py",
            "EXPERIMENT_PRESET_V6_SIGNED_ROUTER_NCQT_STAGE_B3_SCENE_AXIS_EXECUTOR",
        ),
    )
    static_symbols: dict[str, bool] = {}
    for module_name, source_path, symbol_name in symbols:
        static_symbols[f"{module_name}:{symbol_name}"] = _source_defines_symbol(source_path, symbol_name)

    imported: dict[str, bool] = {}
    errors: dict[str, str] = {}
    if require_runtime_import:
        for module_name, _source_path, symbol_name in symbols:
            key = f"{module_name}:{symbol_name}"
            try:
                module = importlib.import_module(module_name)
                imported[key] = hasattr(module, symbol_name)
            except Exception as exc:  # pragma: no cover - environment-specific diagnostic
                imported[key] = False
                errors[key] = f"{type(exc).__name__}: {exc}"
    runtime_passed = all(imported.values()) if require_runtime_import else True
    return {
        "passed": all(static_symbols.values()) and runtime_passed,
        "static_symbol_contract": static_symbols,
        "runtime_import_required": bool(require_runtime_import),
        "runtime_import_status": "passed" if require_runtime_import and runtime_passed else (
            "failed" if require_runtime_import else "not_requested"
        ),
        "runtime_imports": imported,
        "errors": errors,
    }


def _check_phase0_scope() -> dict[str, Any]:
    """Ensure this package has not become a hidden planner-output editor."""

    package_dir = Path(__file__).resolve().parent
    # Build the sentinel strings at runtime so this checker can audit its own
    # source without matching the list of sentinels below.
    forbidden_fragments = (
        "compute_" + "planner_trajectory",
        "outputs_" + "to_trajectory",
        "outputs" + "[\"prediction\"]",
        "Preference" + "VectorField",
        "Dual" + "DPM",
    )
    violations: dict[str, list[str]] = {}
    files = sorted(package_dir.glob("*.py"))
    for path in files:
        source = path.read_text(encoding="utf-8")
        found = [fragment for fragment in forbidden_fragments if fragment in source]
        if found:
            violations[str(path.name)] = found
    return {
        "passed": not violations,
        "checked_files": [path.name for path in files],
        "forbidden_output_edit_fragments": violations,
    }


def run_phase0_contract_checks(*, require_runtime_import: bool = False) -> dict[str, Any]:
    """Run meaningful read-only checks before any preference-flow model work.

    The result is JSON-serializable so it can be embedded in the manifest and
    asserted by :mod:`selftest_phase0` without accessing server data.
    """

    checks = {
        "axis_schema": _check_axis_schema(),
        "three_axis_activation": _check_command_activation(),
        "rho_domain": {
            "passed": parse_rho_grid(DEFAULT_RHO_GRID) == DEFAULT_RHO_GRID,
            "default_grid": list(DEFAULT_RHO_GRID),
        },
        "runtime_router_causality": _check_runtime_router_causality(),
        "frozen_b3_imports": _check_frozen_b3_imports(
            require_runtime_import=require_runtime_import
        ),
        "phase0_scope": _check_phase0_scope(),
    }
    failed = [name for name, report in checks.items() if not bool(report.get("passed", False))]
    return {
        "schema_version": PHASE0_SCHEMA_VERSION,
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def assert_phase0_contracts() -> dict[str, Any]:
    """Return the contract report or raise a concise actionable error."""

    report = run_phase0_contract_checks()
    if not report["passed"]:
        failed = ", ".join(report["failed_checks"])
        raise Phase0ContractError(f"Phase-0 contract check failed: {failed}")
    return report

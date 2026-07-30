"""Build a reproducibility manifest for the approved preference-flow Phase 0.

This tool is intentionally read-only with respect to planner code and research
artifacts.  A normal invocation writes only its requested JSON manifest; a
``--dry-run`` performs no artifact hashing and writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from research_v1.execution.preference_flow.phase0_contracts import (
    PHASE0_SCHEMA_VERSION,
    Phase0Paths,
    parse_rho_grid,
    resolve_phase0_paths,
    run_phase0_contract_checks,
)


HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ArtifactSpec:
    """One manifest artifact and its required type."""

    name: str
    path: str | Path | None
    required: bool
    expected_kind: str = "file"


@dataclass(frozen=True)
class ManifestRequest:
    """Fully resolved input to :func:`build_manifest`."""

    paths: Phase0Paths
    base_checkpoint_path: str
    style_checkpoint_path: str
    calibration_paths: tuple[str, ...]
    reference_paths: tuple[str, ...]
    config_paths: tuple[str, ...]
    smoke_cohort_path: str
    seed: int
    rho_grid: tuple[float, ...]
    dry_run: bool
    strict: bool
    actual_command: str


def _clean_paths(paths: Iterable[str | Path]) -> tuple[str, ...]:
    return tuple(_display_path(path) for path in paths if _display_path(path))


def _display_path(value: str | Path) -> str:
    """Preserve POSIX server defaults when this CLI is dry-run on Windows."""

    return value.as_posix() if isinstance(value, Path) else str(value).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(HASH_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    """Hash a directory deterministically without relying on mtimes.

    Checkpoints should normally be supplied as files.  Directory hashing is
    retained for a small immutable artifact bundle only and deliberately hashes
    every contained file rather than manufacturing a pseudo hash from metadata.
    """

    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _describe_artifact(spec: ArtifactSpec, *, dry_run: bool) -> dict[str, Any]:
    path_text = _display_path(spec.path) if spec.path is not None else ""
    record: dict[str, Any] = {
        "name": spec.name,
        "path": path_text or None,
        "required": bool(spec.required),
        "expected_kind": spec.expected_kind,
        "status": "unspecified",
        "sha256": None,
        "hash_status": "not_requested",
    }
    if not path_text:
        return record

    path = Path(path_text)
    if not path.exists():
        record.update({"status": "missing", "hash_status": "missing"})
        return record

    actual_kind = "directory" if path.is_dir() else "file" if path.is_file() else "other"
    record["actual_kind"] = actual_kind
    if actual_kind != spec.expected_kind:
        record.update({"status": "wrong_type", "hash_status": "not_hashed"})
        return record

    record["status"] = "present"
    if dry_run:
        # ``stat`` is metadata-only and does not read a large split/checkpoint.
        record.update(
            {
                "size_bytes": int(path.stat().st_size) if path.is_file() else None,
                "hash_status": "skipped_dry_run",
            }
        )
        return record

    record["sha256"] = _sha256_file(path) if path.is_file() else _sha256_directory(path)
    record["hash_status"] = "computed"
    record["size_bytes"] = int(path.stat().st_size) if path.is_file() else None
    return record


def _environment_manifest() -> dict[str, Any]:
    torch_info: dict[str, Any]
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_info = {
            "status": "available",
            "version": str(torch.__version__),
            "cuda_build_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        }
    except Exception as exc:  # pragma: no cover - environment-specific diagnostic
        torch_info = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "pytorch": torch_info,
    }


def _cohort_summary(path: str, *, dry_run: bool) -> dict[str, Any]:
    """Validate that a fixed cohort file is present without inventing tokens.

    The cohort contents remain user-curated.  Phase 0 only locks the provided
    file by path and hash; dry-run deliberately does not parse it.
    """

    if not path:
        return {"status": "unspecified", "token_count": None}
    target = Path(path)
    if not target.is_file():
        return {"status": "missing", "token_count": None}
    if dry_run:
        return {"status": "not_read_dry_run", "token_count": None}
    try:
        text = target.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return {"status": "not_utf8", "token_count": None}
    if not text:
        return {"status": "empty", "token_count": 0}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        token_count = len([line for line in text.splitlines() if line.strip()])
        return {"status": "text_or_jsonl", "token_count": token_count}
    if isinstance(payload, list):
        return {"status": "json_list", "token_count": len(payload)}
    if isinstance(payload, dict):
        for key in ("tokens", "sample_tokens", "sample_ids", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                return {"status": f"json_object:{key}", "token_count": len(value)}
        return {"status": "json_object_without_token_list", "token_count": None}
    return {"status": "unsupported_json", "token_count": None}


def _required_failures(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for name, record in manifest["artifacts"].items():
        if record.get("required") and record.get("status") != "present":
            failures.append(f"{name}: {record.get('status')}")
    cohort = manifest["smoke_cohort"]
    if cohort["artifact"]["status"] == "present" and cohort["validation"]["status"] in {
        "empty",
        "not_utf8",
        "json_object_without_token_list",
        "unsupported_json",
    }:
        failures.append(f"smoke_cohort_validation: {cohort['validation']['status']}")
    contracts = manifest["phase0_contracts"]
    if not bool(contracts.get("passed", False)):
        failures.append("phase0_contracts: failed")
    return failures


def build_manifest(request: ManifestRequest) -> dict[str, Any]:
    """Collect a manifest; never writes artifacts or changes the repository."""

    artifact_specs = [
        ArtifactSpec("base_checkpoint", request.base_checkpoint_path, required=True),
        ArtifactSpec("style_checkpoint", request.style_checkpoint_path, required=True),
        ArtifactSpec("dataset_split", request.paths.dataset_split_path, required=True),
        ArtifactSpec("config_0", request.config_paths[0] if request.config_paths else None, required=True),
    ]
    artifact_specs.extend(
        ArtifactSpec(f"calibration_{index}", path, required=True)
        for index, path in enumerate(request.calibration_paths)
    )
    artifact_specs.extend(
        ArtifactSpec(f"reference_{index}", path, required=True)
        for index, path in enumerate(request.reference_paths)
    )
    if not request.calibration_paths:
        artifact_specs.append(ArtifactSpec("calibration_0", None, required=True))
    if not request.reference_paths:
        artifact_specs.append(ArtifactSpec("reference_0", None, required=True))
    if len(request.config_paths) > 1:
        artifact_specs.extend(
            ArtifactSpec(f"config_{index}", path, required=True)
            for index, path in enumerate(request.config_paths[1:], start=1)
        )
    artifacts = {
        spec.name: _describe_artifact(spec, dry_run=request.dry_run) for spec in artifact_specs
    }
    cohort_artifact = _describe_artifact(
        ArtifactSpec("smoke_cohort", request.smoke_cohort_path, required=True),
        dry_run=request.dry_run,
    )
    manifest: dict[str, Any] = {
        "schema_version": PHASE0_SCHEMA_VERSION,
        "phase": 0,
        "purpose": "reproducible baseline and structural acceptance only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(request.dry_run),
        "strict": bool(request.strict),
        "server_paths": request.paths.to_json_dict(),
        "artifacts": artifacts,
        "a3_8_checkpoint_root": request.paths.a3_8_checkpoint_root.as_posix(),
        "smoke_cohort": {
            "artifact": cohort_artifact,
            "validation": _cohort_summary(request.smoke_cohort_path, dry_run=request.dry_run),
        },
        "seed": int(request.seed),
        "rho_grid": [float(value) for value in request.rho_grid],
        "environment": _environment_manifest(),
        "execution_command": request.actual_command,
        "phase0_contracts": run_phase0_contract_checks(
            require_runtime_import=bool(request.strict)
        ),
        "scope": {
            "baseline_modified_by_phase0": False,
            "allowed_write": "requested manifest JSON only when --dry-run is absent",
            "prohibited": [
                "baseline modifications",
                "preference flow implementation",
                "dual DPM stream",
                "trajectory post-processing",
                "content curve",
                "route continuation",
                "automatic source-control mutation",
            ],
        },
    }
    manifest["strict_failures"] = _required_failures(manifest)
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Atomically write the requested manifest and nothing else."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)


def _request_from_args(args: argparse.Namespace, argv: Sequence[str] | None = None) -> ManifestRequest:
    paths = resolve_phase0_paths(
        repo_root=args.repo_root,
        record_root=args.record_root,
        preference_flow_root=args.preference_flow_root,
        a3_8_checkpoint_root=args.a3_8_checkpoint_root,
        dataset_split_path=args.dataset_split_path or None,
        config_path=None,
        smoke_cohort_path=args.smoke_cohort_path or None,
        manifest_path=args.output_path or None,
    )
    config_paths = _clean_paths(args.config_path)
    if not config_paths:
        config_paths = (paths.default_config_path.as_posix(),)
    actual_command = str(args.actual_command).strip()
    if not actual_command:
        effective_argv = list(sys.argv if argv is None else argv)
        actual_command = shlex.join(effective_argv)
    return ManifestRequest(
        paths=paths,
        base_checkpoint_path=str(args.base_checkpoint_path).strip(),
        style_checkpoint_path=str(args.style_checkpoint_path).strip(),
        calibration_paths=_clean_paths(args.calibration_path),
        reference_paths=_clean_paths(args.reference_path),
        config_paths=config_paths,
        smoke_cohort_path=paths.smoke_cohort_path.as_posix(),
        seed=int(args.seed),
        rho_grid=parse_rho_grid(args.rho_grid),
        dry_run=bool(args.dry_run),
        strict=bool(args.strict),
        actual_command=actual_command,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only Phase-0 reproducibility manifest for preference flow."
    )
    parser.add_argument("--repo-root", default="", help="Server repository root.")
    parser.add_argument("--record-root", default="", help="Server record/cache root.")
    parser.add_argument(
        "--preference-flow-root",
        default="",
        help="Root for newly created preference-flow artifacts under the record root.",
    )
    parser.add_argument(
        "--a3-8-checkpoint-root",
        default="",
        help="Run directory for the rebuilt A3.8 checkpoint lineage.",
    )
    parser.add_argument("--base-checkpoint-path", default="", help="Frozen base checkpoint file.")
    parser.add_argument("--style-checkpoint-path", default="", help="StylePlanner checkpoint file.")
    parser.add_argument("--dataset-split-path", default="", help="Fixed split-index file.")
    parser.add_argument(
        "--calibration-path",
        action="append",
        default=[],
        help="Required frozen calibration artifact; repeat for multiple files.",
    )
    parser.add_argument(
        "--reference-path",
        action="append",
        default=[],
        help="Required frozen reference artifact; repeat for multiple files.",
    )
    parser.add_argument(
        "--config-path",
        action="append",
        default=[],
        help="Config file to lock; defaults to baseline/config/planner/style_planner.yaml.",
    )
    parser.add_argument("--smoke-cohort-path", default="", help="Curated fixed smoke-cohort token file.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--rho-grid",
        default="-1.0,-0.5,0.0,0.5,1.0",
        help="Comma-separated grid in [-1, 1] that includes 0; use --rho-grid=<values>.",
    )
    parser.add_argument("--actual-command", default="", help="Optional server command recorded verbatim.")
    parser.add_argument("--output-path", default="", help="Manifest JSON path for non-dry runs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and report artifacts only; do not hash data/checkpoints or write a manifest.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return failure when a required artifact or contract is unavailable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    request = _request_from_args(args, argv=argv)
    manifest = build_manifest(request)

    # Always print the complete report first, including a local strict-mode
    # failure.  This makes missing server-mounted artifacts actionable instead
    # of silently selecting another checkpoint or split.
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if request.strict and manifest["strict_failures"]:
        return 2
    if request.dry_run:
        return 0
    write_manifest(request.paths.default_manifest_path, manifest)
    print(f"Wrote Phase-0 manifest: {request.paths.default_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

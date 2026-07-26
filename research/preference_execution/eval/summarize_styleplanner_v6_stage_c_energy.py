"""Summarize the isolated closed-loop Stage-C Energy comparison.

The input is the manifest written by ``run_styleplanner_v6_closed_loop_suite``.
This script does not launch simulation or mutate checkpoints.  It verifies the
runtime Energy contract and compares structured per-scenario metrics between
``anchor_cfg`` and ``full_energy`` at each rho.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


SUMMARY_NAME = "v6_stage_c_energy_summary.json"
VARIANT_ORDER = ("anchor_cfg", "full_energy")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a completed Stage-C closed-loop Energy suite."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--output",
        default="",
        help=f"Optional output JSON; default is beside the manifest as {SUMMARY_NAME}.",
    )
    return parser


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)


def _load_job_audit(job: Mapping[str, Any]) -> Dict[str, Any]:
    embedded = job.get("runtime_contract_audit")
    if isinstance(embedded, Mapping):
        return dict(embedded)
    path = Path(str(job["run_dir"])) / "v6_closed_loop_router_audit.json"
    return dict(_read_json(path)) if path.is_file() else {}


def _load_job_metrics(job: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(str(job["run_dir"])) / "v6_closed_loop_metrics.json"
    return dict(_read_json(path)) if path.is_file() else {}


def _record_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("scenario_name", "")),
        str(record.get("log_name", "")),
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _compare_scenario_metrics(
    baseline_metrics: Mapping[str, Any],
    energy_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline_rows = baseline_metrics.get("scenario_metrics", [])
    energy_rows = energy_metrics.get("scenario_metrics", [])
    if not isinstance(baseline_rows, list):
        baseline_rows = []
    if not isinstance(energy_rows, list):
        energy_rows = []
    baseline_by_key = {
        _record_key(row): row
        for row in baseline_rows
        if isinstance(row, Mapping)
    }
    energy_by_key = {
        _record_key(row): row
        for row in energy_rows
        if isinstance(row, Mapping)
    }
    common_keys = sorted(set(baseline_by_key) & set(energy_by_key))
    missing_in_energy = sorted(set(baseline_by_key) - set(energy_by_key))
    unexpected_in_energy = sorted(set(energy_by_key) - set(baseline_by_key))

    metric_names = sorted(
        {
            name
            for key in common_keys
            for row in (baseline_by_key[key], energy_by_key[key])
            for name, value in row.items()
            if _finite_number(value) is not None
        }
    )
    metric_comparison: Dict[str, Dict[str, Any]] = {}
    absolute_deltas = []
    for name in metric_names:
        paired_values = []
        for key in common_keys:
            baseline_value = _finite_number(baseline_by_key[key].get(name))
            energy_value = _finite_number(energy_by_key[key].get(name))
            if baseline_value is None or energy_value is None:
                continue
            delta = energy_value - baseline_value
            paired_values.append((baseline_value, energy_value, delta))
            absolute_deltas.append(abs(delta))
        if not paired_values:
            continue
        metric_comparison[name] = {
            "paired_count": len(paired_values),
            "anchor_cfg_mean": sum(row[0] for row in paired_values)
            / len(paired_values),
            "full_energy_mean": sum(row[1] for row in paired_values)
            / len(paired_values),
            "full_minus_anchor_mean": sum(row[2] for row in paired_values)
            / len(paired_values),
            "full_minus_anchor_min": min(row[2] for row in paired_values),
            "full_minus_anchor_max": max(row[2] for row in paired_values),
        }
    return {
        "anchor_cfg_scenario_count": len(baseline_by_key),
        "full_energy_scenario_count": len(energy_by_key),
        "paired_scenario_count": len(common_keys),
        "missing_in_full_energy": [list(key) for key in missing_in_energy],
        "unexpected_in_full_energy": [
            list(key) for key in unexpected_in_energy
        ],
        "max_abs_numeric_metric_delta": (
            max(absolute_deltas) if absolute_deltas else None
        ),
        "all_numeric_metrics_exact": bool(absolute_deltas)
        and max(absolute_deltas) <= 1e-12,
        "metrics": metric_comparison,
    }


def _job_runtime_summary(
    job: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "status": str(job.get("status", "")),
        "custom_metrics_status": str(job.get("custom_metrics_status", "")),
        "audit_passed": bool(audit.get("passed", False)),
        "trace_row_count": int(audit.get("trace_row_count", 0)),
        "router_eligible_scene_counts": audit.get(
            "router_eligible_scene_counts",
            {},
        ),
        "energy_expected": bool(
            audit.get("preference_energy_expected", False)
        ),
        "energy_guidance_used_rows": int(
            audit.get("preference_energy_guidance_used_rows", 0)
        ),
        "energy_used_scene_counts": audit.get(
            "energy_used_scene_counts",
            {},
        ),
        "energy_diagnostic_scene_counts": audit.get(
            "energy_diagnostic_scene_counts",
            {},
        ),
        "energy_nonfinite_diagnostic_rows": int(
            audit.get("energy_nonfinite_diagnostic_rows", 0)
        ),
        "energy_used_without_controlled_router_rows": int(
            audit.get("energy_used_without_controlled_router_rows", 0)
        ),
        "mismatches": audit.get("mismatches", []),
    }


def _pair_contract_passed(
    *,
    rho: float,
    baseline: Mapping[str, Any],
    energy: Mapping[str, Any],
    metric_comparison: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    reasons = []
    for name, runtime in (
        ("anchor_cfg", baseline),
        ("full_energy", energy),
    ):
        if runtime.get("status") != "completed":
            reasons.append(f"{name}_job_not_completed")
        if runtime.get("custom_metrics_status") != "completed":
            reasons.append(f"{name}_custom_metrics_not_completed")
        if not bool(runtime.get("audit_passed", False)):
            reasons.append(f"{name}_runtime_audit_failed")
        if int(runtime.get("energy_nonfinite_diagnostic_rows", 0)) > 0:
            reasons.append(f"{name}_nonfinite_energy_diagnostics")
        if int(runtime.get("energy_used_without_controlled_router_rows", 0)) > 0:
            reasons.append(f"{name}_energy_used_outside_router")
    if int(baseline.get("energy_guidance_used_rows", 0)) != 0:
        reasons.append("anchor_cfg_energy_guidance_was_used")
    energy_used_rows = int(energy.get("energy_guidance_used_rows", 0))
    if abs(float(rho)) <= 1e-12:
        if energy_used_rows != 0:
            reasons.append("rho_zero_energy_gradient_was_used")
        if not bool(metric_comparison.get("all_numeric_metrics_exact", False)):
            reasons.append("rho_zero_metrics_not_exact")
    elif energy_used_rows <= 0:
        reasons.append("nonzero_rho_energy_guidance_not_used")
    if int(metric_comparison.get("paired_scenario_count", 0)) <= 0:
        reasons.append("no_paired_scenario_metrics")
    if metric_comparison.get("missing_in_full_energy"):
        reasons.append("scenario_metrics_missing_in_full_energy")
    if metric_comparison.get("unexpected_in_full_energy"):
        reasons.append("unexpected_full_energy_scenario_metrics")
    return not reasons, reasons


def summarize(manifest_path: Path) -> Dict[str, Any]:
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("Manifest must contain one JSON object")
    if str(manifest.get("guidance_stage", "")) != "stage_c":
        raise ValueError("Manifest is not a Stage-C guidance suite")
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("Manifest jobs must be a list")

    grouped: Dict[tuple[str, float], Dict[str, Mapping[str, Any]]] = {}
    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        variant = str(job.get("variant", ""))
        if variant not in VARIANT_ORDER:
            continue
        key = (
            str(job.get("checkpoint_tag", "")),
            float(job.get("rho", math.nan)),
        )
        grouped.setdefault(key, {})[variant] = job

    pairs = []
    all_passed = True
    for (checkpoint_tag, rho), pair_jobs in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        missing_variants = [
            variant for variant in VARIANT_ORDER if variant not in pair_jobs
        ]
        if missing_variants:
            pairs.append(
                {
                    "checkpoint_tag": checkpoint_tag,
                    "rho": rho,
                    "contract_passed": False,
                    "failure_reasons": [
                        f"missing_variant:{variant}"
                        for variant in missing_variants
                    ],
                }
            )
            all_passed = False
            continue
        baseline_job = pair_jobs["anchor_cfg"]
        energy_job = pair_jobs["full_energy"]
        baseline_audit = _load_job_audit(baseline_job)
        energy_audit = _load_job_audit(energy_job)
        baseline_runtime = _job_runtime_summary(
            baseline_job,
            baseline_audit,
        )
        energy_runtime = _job_runtime_summary(
            energy_job,
            energy_audit,
        )
        metric_comparison = _compare_scenario_metrics(
            _load_job_metrics(baseline_job),
            _load_job_metrics(energy_job),
        )
        pair_passed, failure_reasons = _pair_contract_passed(
            rho=rho,
            baseline=baseline_runtime,
            energy=energy_runtime,
            metric_comparison=metric_comparison,
        )
        all_passed = all_passed and pair_passed
        pairs.append(
            {
                "checkpoint_tag": checkpoint_tag,
                "rho": rho,
                "contract_passed": pair_passed,
                "failure_reasons": failure_reasons,
                "anchor_cfg": baseline_runtime,
                "full_energy": energy_runtime,
                "scenario_metric_comparison": metric_comparison,
            }
        )

    expected_pair_count = len(
        {
            (
                str(job.get("checkpoint_tag", "")),
                float(job.get("rho", math.nan)),
            )
            for job in jobs
            if isinstance(job, Mapping)
        }
    )
    if len(pairs) != expected_pair_count or not pairs:
        all_passed = False
    return {
        "artifact": "styleplanner_v6_stage_c_energy_summary",
        "source_manifest": str(manifest_path),
        "stage_c_contract_passed": all_passed,
        "expected_pair_count": expected_pair_count,
        "summarized_pair_count": len(pairs),
        "energy_artifacts": manifest.get("preflight", {}).get(
            "energy_artifacts",
            {},
        ),
        "guidance_runtime_contract": manifest.get(
            "guidance_runtime_contract",
            {},
        ),
        "pairs": pairs,
        "interpretation_note": (
            "contract_passed validates deterministic rho=0 preservation, "
            "runtime activation, finite diagnostics, Router scope, and paired "
            "metric availability. It does not assert that nonzero-rho Energy "
            "improves controllability or safety; inspect per-metric deltas."
        ),
    }


def main() -> None:
    args = _parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    output_path = (
        Path(args.output).expanduser().resolve()
        if str(args.output).strip()
        else manifest_path.parent / SUMMARY_NAME
    )
    report = summarize(manifest_path)
    _write_json(output_path, report)
    print(
        "[V6StageC] contract_passed="
        f"{report['stage_c_contract_passed']} "
        f"pairs={report['summarized_pair_count']}/"
        f"{report['expected_pair_count']} output={output_path}"
    )
    for pair in report["pairs"]:
        energy_runtime = pair.get("full_energy", {})
        comparison = pair.get("scenario_metric_comparison", {})
        print(
            "[V6StageC] "
            f"{pair.get('checkpoint_tag')} rho={pair.get('rho'):+.2f} "
            f"passed={pair.get('contract_passed')} "
            f"energy_rows={energy_runtime.get('energy_guidance_used_rows', 0)} "
            f"paired_scenarios={comparison.get('paired_scenario_count', 0)} "
            f"max_metric_delta={comparison.get('max_abs_numeric_metric_delta')} "
            f"reasons={pair.get('failure_reasons', [])}"
        )


if __name__ == "__main__":
    main()

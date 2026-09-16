"""从一次性闭环采集中构造所有方法、所有 rho 的共同成功场景报告。"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from stylelora.lora.evaluation.closed_loop_style import analyze_closed_loop_style_runs


SAFETY_METRICS = (
    "no_ego_at_fault_collisions",
    "time_to_collision_within_bound",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "speed_limit_compliance",
    "ego_is_comfortable",
)


def _parse_named_reports(values: list[str]) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--report 必须使用 NAME=PATH：{raw}")
        name, path = (part.strip() for part in raw.split("=", 1))
        if not name or not path or name in reports:
            raise ValueError(f"报告名称或路径无效/重复：{raw}")
        reports[name] = Path(path).expanduser().resolve()
    if len(reports) < 2:
        raise ValueError("至少需要两份闭环报告才能构造共同成功场景")
    return reports


def _slug(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not value:
        raise ValueError(f"报告名称无法生成文件名：{name}")
    return value


def _candidate_tokens(report: dict) -> list[str]:
    tokens = report.get("settings", {}).get("scenario_tokens", [])
    if not isinstance(tokens, list) or not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("闭环报告缺少无重复的 settings.scenario_tokens")
    return [str(token) for token in tokens]


def _rho_key(value: float) -> str:
    return f"rho_{value:+.2f}"


def _record_rows(record: dict) -> dict[str, dict]:
    official = record.get("official_aggregator", {})
    if not official.get("available", False):
        raise ValueError(f"rho={record.get('rho')} 缺少可用的官方 aggregator")
    rows = official.get("per_scenario", [])
    mapping = {str(row["token"]): row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"rho={record.get('rho')} 官方 aggregator 包含重复 token")
    return mapping


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("共同场景集合为空，无法计算均值")
    return float(sum(values) / len(values))


def _filter_official(official: dict, common_tokens: list[str]) -> dict:
    rows = {str(row["token"]): row for row in official["per_scenario"]}
    selected = [copy.deepcopy(rows[token]) for token in common_tokens]
    return {
        **{key: copy.deepcopy(value) for key, value in official.items()
           if key not in {"per_scenario", "challenge_score", "safety_metric_means"}},
        "available": True,
        "partial": False,
        "scenario_count": len(selected),
        "candidate_scenario_count": len(selected),
        "missing_tokens": [],
        "unexpected_tokens": [],
        "duplicate_tokens": [],
        "challenge_score": _mean([float(row["challenge_score"]) for row in selected]),
        "safety_metric_means": {
            metric: _mean([float(row["safety_metrics"][metric]) for row in selected])
            for metric in SAFETY_METRICS
        },
        "per_scenario": selected,
    }


def _rho_zero_baseline(records: list[dict]) -> dict:
    zero = next(record for record in records if abs(float(record["rho"])) < 1e-12)
    official = zero["official_aggregator"]
    return {
        "rho": 0.0,
        "scenario_count": official["scenario_count"],
        "challenge_score": official["challenge_score"],
        "safety_metric_means": official["safety_metric_means"],
        "per_scenario": official["per_scenario"],
    }


def _paired_differences(records: list[dict]) -> dict[str, dict]:
    baseline = _rho_zero_baseline(records)
    baseline_rows = {row["token"]: row for row in baseline["per_scenario"]}
    output: dict[str, dict] = {}
    for record in records:
        rho = float(record["rho"])
        if abs(rho) < 1e-12:
            continue
        current_rows = {
            row["token"]: row for row in record["official_aggregator"]["per_scenario"]
        }
        differences = []
        for token in baseline_rows:
            base, current = baseline_rows[token], current_rows[token]
            differences.append({
                "token": token,
                "log_name": current.get("log_name", ""),
                "challenge_score_delta": (
                    float(current["challenge_score"]) - float(base["challenge_score"])
                ),
                "safety_metric_deltas": {
                    metric: (
                        float(current["safety_metrics"][metric])
                        - float(base["safety_metrics"][metric])
                    )
                    for metric in SAFETY_METRICS
                },
            })
        output[_rho_key(rho)] = {
            "paired_scenario_count": len(differences),
            "mean_challenge_score_delta": _mean([
                row["challenge_score_delta"] for row in differences
            ]),
            "mean_safety_metric_deltas": {
                metric: _mean([
                    row["safety_metric_deltas"][metric] for row in differences
                ])
                for metric in SAFETY_METRICS
            },
            "per_scenario": differences,
        }
    return output


def _raw_roots(records: list[dict]) -> dict[float, Path]:
    return {
        float(record["rho"]): Path(record["result_dir"]).resolve().parent / "raw_step_data"
        for record in records
    }


def _filtered_report(
    report: dict,
    common_tokens: list[str],
    common_tokens_file: Path,
) -> dict:
    filtered = copy.deepcopy(report)
    filtered.pop("failure", None)
    filtered.pop("partial_collection", None)
    filtered.pop("metric_deltas_vs_rho_zero", None)
    filtered["settings"]["scenario_tokens"] = common_tokens
    filtered["settings"]["scenario_token_count"] = len(common_tokens)
    filtered["settings"]["expected_scenario_count"] = len(common_tokens)
    filtered["settings"]["scenario_tokens_file"] = str(common_tokens_file)
    filtered["settings"]["allow_partial_scenarios"] = False
    for record in filtered["records"]:
        record["raw_collection_runner"] = record.pop("runner", {})
        raw_metric_summary = {
            key: record.pop(key)
            for key in (
                "closed_loop_pass", "metric_means", "required_metric_values",
                "missing_required_metrics", "below_threshold", "failure",
            )
            if key in record
        }
        record["raw_collection_metric_summary"] = raw_metric_summary
        record["runner"] = {
            "all_succeeded": True,
            "runner_count": len(common_tokens),
            "successful_count": len(common_tokens),
            "failed_count": 0,
            "scenario_tokens": common_tokens,
            "successful_tokens": common_tokens,
            "failed_tokens": [],
            "token_set_matches": True,
        }
        record["official_aggregator"] = _filter_official(
            record["official_aggregator"], common_tokens
        )
        record["scenario_count"] = len(common_tokens)
    filtered["closed_loop_pass"] = all(
        record["official_aggregator"].get("available", False)
        for record in filtered["records"]
    )
    filtered["rho_zero_baseline"] = _rho_zero_baseline(filtered["records"])
    filtered["paired_official_differences_vs_rho_zero"] = _paired_differences(
        filtered["records"]
    )
    filtered["style_proxy"] = analyze_closed_loop_style_runs(
        _raw_roots(filtered["records"]), allowed_tokens=set(common_tokens)
    )
    filtered["common_subset"] = {
        "scenario_count": len(common_tokens),
        "scenario_tokens_file": str(common_tokens_file),
        "scope": "intersection across every method and every rho",
    }
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从四组一次性300候选场景采集中构造共同成功 token 和正式配对报告"
    )
    parser.add_argument("--report", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    report_paths = _parse_named_reports(args.report)
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in report_paths.items()
    }
    first_name = next(iter(payloads))
    candidates = _candidate_tokens(payloads[first_name])
    expected_rhos = [float(row["rho"]) for row in payloads[first_name].get("records", [])]
    if not expected_rhos or 0.0 not in expected_rhos:
        raise ValueError("闭环报告的 rho 网格无效或缺少 rho=0")

    valid_sets: list[set[str]] = []
    audit_methods: dict[str, dict] = {}
    candidate_set = set(candidates)
    for name, report in payloads.items():
        if _candidate_tokens(report) != candidates:
            raise ValueError(f"{name} 使用的候选 token 或顺序与其他方法不一致")
        records = report.get("records", [])
        rhos = [float(row["rho"]) for row in records]
        if rhos != expected_rhos:
            raise ValueError(f"{name} 的 rho 网格或顺序与其他方法不一致")
        per_rho = {}
        for record in records:
            rows = _record_rows(record)
            if not set(rows).issubset(candidate_set):
                raise ValueError(f"{name} rho={record['rho']} 出现候选名单外 token")
            valid = set(rows)
            valid_sets.append(valid)
            per_rho[_rho_key(float(record["rho"]))] = {
                "valid_count": len(valid),
                "failed_count": len(candidates) - len(valid),
                "failed_tokens": [token for token in candidates if token not in valid],
                "runner_failed_scenarios": record.get("runner", {}).get(
                    "failed_scenarios", []
                ),
                "invalid_metric_tokens": record.get("official_aggregator", {}).get(
                    "invalid_metric_tokens", []
                ),
            }
        audit_methods[name] = {"source_report": str(report_paths[name]), "per_rho": per_rho}

    common_set = set.intersection(*valid_sets)
    common_tokens = [token for token in candidates if token in common_set]
    if not common_tokens:
        raise RuntimeError("所有方法、所有 rho 没有共同成功场景")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    common_tokens_file = output_root / "common_valid_tokens.json"
    common_tokens_file.write_text(
        json.dumps(common_tokens, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    filtered_paths = {}
    for index, (name, report) in enumerate(payloads.items(), start=1):
        print(f"[filter {index}/{len(payloads)}] {name}: {len(common_tokens)} 个共同场景", flush=True)
        filtered = _filtered_report(report, common_tokens, common_tokens_file)
        destination = output_root / f"{_slug(name)}.filtered.json"
        destination.write_text(
            json.dumps(filtered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        filtered_paths[name] = str(destination)

    audit = {
        "candidate_count": len(candidates),
        "common_valid_count": len(common_tokens),
        "removed_count": len(candidates) - len(common_tokens),
        "common_valid_tokens_file": str(common_tokens_file),
        "removed_tokens": [token for token in candidates if token not in common_set],
        "rho_grid": expected_rhos,
        "methods": audit_methods,
        "filtered_reports": filtered_paths,
    }
    audit_path = output_root / "common_valid_scenarios_report.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "candidate_count": audit["candidate_count"],
        "common_valid_count": audit["common_valid_count"],
        "removed_count": audit["removed_count"],
        "common_valid_tokens_file": audit["common_valid_tokens_file"],
        "filtered_reports": audit["filtered_reports"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

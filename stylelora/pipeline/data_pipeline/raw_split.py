"""Split raw nuPlan DB logs before planner-cache generation.

The split happens at the raw log/session level before ``DataProcessor`` creates
``.npz`` samples. The module writes JSON/YAML manifests only, keeping held-out
closed-loop simulation logs out of the training cache by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from stylelora.pipeline.paths import (
    DEFAULT_DATA_SPLITS_ROOT,
    DEFAULT_NUPLAN_LOG_NAMES_PATH,
    DEFAULT_SCENARIO_FILTER_ROOT,
    REPO_ROOT,
)

DEFAULT_LOG_NAMES_PATH = Path(DEFAULT_NUPLAN_LOG_NAMES_PATH) if DEFAULT_NUPLAN_LOG_NAMES_PATH else Path("")
DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_SPLITS_ROOT)
DEFAULT_CONFIG_OUTPUT_DIR = Path(DEFAULT_SCENARIO_FILTER_ROOT)


def load_json_list(path: os.PathLike[str] | str) -> List[str]:
    """Load a JSON list of strings."""

    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list at {path}, got {type(payload).__name__}")
    return [str(item) for item in payload]


def discover_db_log_names(data_path: os.PathLike[str] | str) -> List[str]:
    """Discover nuPlan log names from a directory of .db files.

    This mirrors the DB-list part of the root-level data_process_json_construction.py
    while keeping the retrieval-guidance split workflow self-contained.
    """

    data_path = Path(data_path)
    if data_path.is_file():
        paths = [data_path]
    elif data_path.is_dir():
        paths = sorted(path for path in data_path.rglob("*.db") if path.is_file())
    else:
        raise FileNotFoundError(f"Raw DB path does not exist: {data_path}")
    log_names = sorted({path.stem for path in paths})
    if not log_names:
        raise ValueError(f"No .db files found under {data_path}")
    return log_names


def load_or_discover_log_names(
    log_names_path: os.PathLike[str] | str,
    data_path: os.PathLike[str] | str = "",
) -> List[str]:
    """Load a DB-name JSON list, or discover it from raw DB files."""

    if str(log_names_path):
        path = Path(log_names_path)
        if path.exists():
            return load_json_list(path)
        if not str(data_path):
            raise FileNotFoundError(f"log_names_path does not exist: {path}")
    if not str(data_path):
        raise ValueError("Either --log_names_path or --data_path must be provided")
    return discover_db_log_names(data_path)


def write_json(path: os.PathLike[str] | str, payload: object) -> None:
    """Write JSON with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    os.replace(tmp_path, path)


def infer_session_id(log_name: str) -> str:
    """Return the drive-session prefix from a nuPlan log segment name."""

    parts = str(log_name).split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return str(log_name)


def group_log_names(log_names: Sequence[str], group_by: str) -> "OrderedDict[str, List[str]]":
    """Group log names by log segment or drive session."""

    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    for log_name in log_names:
        if group_by == "log":
            group_id = str(log_name)
        elif group_by == "session":
            group_id = infer_session_id(str(log_name))
        else:
            raise ValueError(f"Unsupported group_by={group_by!r}; expected 'session' or 'log'")
        groups.setdefault(group_id, []).append(str(log_name))
    return groups


def _stable_score(seed: int, value: str) -> str:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return digest


def choose_test_groups(
    groups: Mapping[str, Sequence[str]],
    *,
    test_ratio: float,
    seed: int,
    test_log_limit: int = 0,
    test_group_limit: int = 0,
) -> List[str]:
    """Choose held-out groups with deterministic hashing."""

    if not groups:
        raise ValueError("Cannot split an empty log list")
    if test_ratio <= 0.0 and test_log_limit <= 0 and test_group_limit <= 0:
        raise ValueError("At least one of test_ratio/test_log_limit/test_group_limit must select logs")
    if test_ratio < 0.0 or test_ratio >= 1.0:
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")

    group_ids = sorted(groups.keys(), key=lambda item: (_stable_score(seed, item), item))
    if test_group_limit > 0:
        return group_ids[: min(test_group_limit, len(group_ids))]

    total_logs = sum(len(logs) for logs in groups.values())
    target_logs = int(round(total_logs * test_ratio)) if test_log_limit <= 0 else int(test_log_limit)
    target_logs = max(1, min(target_logs, total_logs - 1))

    selected: List[str] = []
    selected_log_count = 0
    for group_id in group_ids:
        if selected_log_count >= target_logs:
            break
        selected.append(group_id)
        selected_log_count += len(groups[group_id])
    return selected


def split_log_names(
    log_names: Sequence[str],
    *,
    test_ratio: float = 0.10,
    seed: int = 3407,
    group_by: str = "session",
    test_log_limit: int = 0,
    test_group_limit: int = 0,
) -> Dict[str, object]:
    """Split raw log names into cache and simulation-test partitions."""

    unique_log_names = list(OrderedDict((str(item), None) for item in log_names).keys())
    groups = group_log_names(unique_log_names, group_by=group_by)
    test_groups = choose_test_groups(
        groups,
        test_ratio=test_ratio,
        seed=seed,
        test_log_limit=test_log_limit,
        test_group_limit=test_group_limit,
    )
    test_group_set = set(test_groups)

    cache_log_names: List[str] = []
    test_log_names: List[str] = []
    for log_name in unique_log_names:
        group_id = log_name if group_by == "log" else infer_session_id(log_name)
        if group_id in test_group_set:
            test_log_names.append(log_name)
        else:
            cache_log_names.append(log_name)

    overlap = sorted(set(cache_log_names).intersection(test_log_names))
    if overlap:
        raise RuntimeError(f"Split produced overlapping logs: {overlap[:5]}")
    if not cache_log_names or not test_log_names:
        raise RuntimeError(
            f"Split must produce non-empty partitions, got cache={len(cache_log_names)} test={len(test_log_names)}"
        )

    return {
        "cache_log_names": cache_log_names,
        "test_log_names": test_log_names,
        "test_group_ids": test_groups,
        "summary": {
            "schema_version": 1,
            "total_log_count": len(unique_log_names),
            "total_group_count": len(groups),
            "cache_log_count": len(cache_log_names),
            "test_log_count": len(test_log_names),
            "cache_group_count": len(groups) - len(test_groups),
            "test_group_count": len(test_groups),
            "test_ratio_requested": float(test_ratio),
            "test_ratio_actual": float(len(test_log_names) / len(unique_log_names)),
            "seed": int(seed),
            "group_by": group_by,
            "test_log_limit": int(test_log_limit),
            "test_group_limit": int(test_group_limit),
        },
    }


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _yaml_string_list(name: str, values: Sequence[str] | None) -> List[str]:
    if values is None:
        return [f"{name}: null"]
    lines = [f"{name}:"]
    for value in values:
        escaped = str(value).replace('"', '\\"')
        lines.append(f'  - "{escaped}"')
    return lines


def scenario_filter_yaml(
    *,
    log_names: Sequence[str] | None,
    scenario_tokens: Sequence[str] | None = None,
    scenario_types: Sequence[str] | None = None,
    limit_total_scenarios: int | None = None,
    timestamp_threshold_s: float | None = None,
    expand_scenarios: bool = False,
    remove_invalid_goals: bool = True,
    shuffle: bool = False,
    header: str = "",
) -> str:
    """Build a nuPlan ScenarioFilter YAML string without requiring PyYAML."""

    lines: List[str] = []
    if header:
        lines.extend([f"# {line}" if line else "#" for line in header.splitlines()])
    lines.extend(
        [
            "_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter",
            '_convert_: "all"',
            "",
        ]
    )
    lines.extend(_yaml_string_list("scenario_types", scenario_types))
    lines.append("")
    lines.extend(_yaml_string_list("scenario_tokens", scenario_tokens))
    lines.append("")
    lines.extend(_yaml_string_list("log_names", log_names))
    lines.extend(
        [
            "map_names: null",
            "",
            "num_scenarios_per_type: null",
            f"limit_total_scenarios: {_yaml_scalar(limit_total_scenarios)}",
            f"timestamp_threshold_s: {_yaml_scalar(timestamp_threshold_s)}",
            "ego_displacement_minimum_m: null",
            "ego_start_speed_threshold: null",
            "ego_stop_speed_threshold: null",
            "speed_noise_tolerance: null",
            "",
            f"expand_scenarios: {_yaml_scalar(expand_scenarios)}",
            f"remove_invalid_goals: {_yaml_scalar(remove_invalid_goals)}",
            f"shuffle: {_yaml_scalar(shuffle)}",
            "",
        ]
    )
    return "\n".join(lines)


def write_scenario_filter(path: os.PathLike[str] | str, yaml_text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(yaml_text, encoding="utf-8")
    os.replace(tmp_path, path)


def materialize_split(
    *,
    log_names_path: os.PathLike[str] | str,
    data_path: os.PathLike[str] | str = "",
    output_dir: os.PathLike[str] | str,
    config_output_dir: os.PathLike[str] | str,
    prefix: str,
    test_ratio: float,
    seed: int,
    group_by: str,
    test_log_limit: int,
    test_group_limit: int,
    timestamp_threshold_s: float,
    write_filters: bool,
) -> Dict[str, object]:
    """Create JSON split artifacts and optional Hydra scenario filters."""

    log_names = load_or_discover_log_names(log_names_path, data_path=data_path)
    split = split_log_names(
        log_names,
        test_ratio=test_ratio,
        seed=seed,
        group_by=group_by,
        test_log_limit=test_log_limit,
        test_group_limit=test_group_limit,
    )
    output_dir = Path(output_dir)
    config_output_dir = Path(config_output_dir)

    cache_log_names = list(split["cache_log_names"])
    test_log_names = list(split["test_log_names"])
    test_group_ids = list(split["test_group_ids"])
    summary: MutableMapping[str, object] = dict(split["summary"])  # type: ignore[arg-type]
    summary["log_names_path"] = str(Path(log_names_path).resolve())
    summary["data_path"] = str(Path(data_path).resolve()) if str(data_path) else ""
    summary["output_dir"] = str(output_dir.resolve())
    summary["config_output_dir"] = str(config_output_dir.resolve()) if write_filters else ""

    cache_path = output_dir / "cache_log_names.json"
    test_path = output_dir / "test_simu_log_names.json"
    groups_path = output_dir / "test_simu_group_ids.json"
    summary_path = output_dir / "split_summary.json"
    write_json(cache_path, cache_log_names)
    write_json(test_path, test_log_names)
    write_json(groups_path, test_group_ids)

    filter_paths: Dict[str, str] = {}
    if write_filters:
        test_filter_path = config_output_dir / f"{prefix}_test_simu.yaml"
        cache_filter_path = config_output_dir / f"{prefix}_cache_raw.yaml"
        common_header = (
            "Generated by research_v1.data_pipeline.raw_split.\n"
            f"Source log list: {Path(log_names_path).name}\n"
            f"Split is disjoint at {group_by} level with seed={seed}."
        )
        write_scenario_filter(
            test_filter_path,
            scenario_filter_yaml(
                log_names=test_log_names,
                timestamp_threshold_s=timestamp_threshold_s,
                expand_scenarios=False,
                remove_invalid_goals=True,
                shuffle=False,
                header=f"{common_header}\nClosed-loop simulation held-out partition.",
            ),
        )
        write_scenario_filter(
            cache_filter_path,
            scenario_filter_yaml(
                log_names=cache_log_names,
                timestamp_threshold_s=None,
                expand_scenarios=True,
                remove_invalid_goals=False,
                shuffle=False,
                header=f"{common_header}\nRaw cache-generation partition.",
            ),
        )
        filter_paths = {
            "test_simu_filter_path": str(test_filter_path.resolve()),
            "cache_raw_filter_path": str(cache_filter_path.resolve()),
        }
        summary.update(filter_paths)

    summary["cache_log_names_path"] = str(cache_path.resolve())
    summary["test_simu_log_names_path"] = str(test_path.resolve())
    summary["test_simu_group_ids_path"] = str(groups_path.resolve())
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path.resolve())
    return dict(summary)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw Boston DB logs before cache generation")
    parser.add_argument("--log_names_path", type=str, default=str(DEFAULT_LOG_NAMES_PATH))
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Optional raw DB directory/file. Used when --log_names_path is empty or missing.",
    )
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config_output_dir", type=str, default=str(DEFAULT_CONFIG_OUTPUT_DIR))
    parser.add_argument("--prefix", type=str, default="boston")
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--group_by", type=str, choices=["session", "log"], default="session")
    parser.add_argument("--test_log_limit", type=int, default=0)
    parser.add_argument("--test_group_limit", type=int, default=0)
    parser.add_argument("--timestamp_threshold_s", type=float, default=15.0)
    parser.add_argument("--write_filters", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    summary = materialize_split(
        log_names_path=args.log_names_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        config_output_dir=args.config_output_dir,
        prefix=args.prefix,
        test_ratio=args.test_ratio,
        seed=args.seed,
        group_by=args.group_by,
        test_log_limit=args.test_log_limit,
        test_group_limit=args.test_group_limit,
        timestamp_threshold_s=args.timestamp_threshold_s,
        write_filters=bool(args.write_filters),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

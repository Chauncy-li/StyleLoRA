"""
训练/仿真测试前置划分脚本。

功能说明：
1. 从给定 log 名单（JSON）或 .db 目录读取原始 log 列表；
2. 一次性划分并冻结 test 日志集（默认复用已存在 test 文件，不重复抽样）；
3. 生成以下 JSON（不带 v1 后缀）：
   - {prefix}_test_logs.json
   - {prefix}_cache_logs.json
   - {prefix}_split_summary.json

设计原则：
- 保证 test 与 cache 互斥；
- 划分可复现（固定 seed + 稳定哈希排序）；
- 默认“test 冻结”，避免每次运行都改变测试集。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_LOG_JSON = SCRIPT_DIR / "resources" / "mini" / "nuplan_scenarios_mini.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "resources" / "mini" / "splits"


def _read_json_list(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON, got {type(payload).__name__}: {path}")
    return [str(item) for item in payload]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def _stable_unique(items: Sequence[str]) -> List[str]:
    return list(OrderedDict((str(item), None) for item in items).keys())


def _discover_log_names_from_db(db_path: Path) -> List[str]:
    if db_path.is_file():
        db_files = [db_path]
    elif db_path.is_dir():
        db_files = sorted(path for path in db_path.rglob("*.db") if path.is_file())
    else:
        raise FileNotFoundError(f"DB path does not exist: {db_path}")
    log_names = sorted({path.stem for path in db_files})
    if not log_names:
        raise ValueError(f"No .db files found under: {db_path}")
    return log_names


def _infer_session_id(log_name: str) -> str:
    parts = str(log_name).split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return str(log_name)


def _group_log_names(log_names: Sequence[str], group_by: str) -> "OrderedDict[str, List[str]]":
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    for log_name in log_names:
        if group_by == "log":
            group_id = str(log_name)
        elif group_by == "session":
            group_id = _infer_session_id(str(log_name))
        else:
            raise ValueError(f"Unsupported group_by={group_by!r}; choose 'log' or 'session'")
        groups.setdefault(group_id, []).append(str(log_name))
    return groups


def _stable_score(seed: int, value: str) -> str:
    return hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _choose_heldout_groups(
    groups: Mapping[str, Sequence[str]],
    *,
    ratio: float,
    seed: int,
    log_limit: int,
    group_limit: int,
) -> List[str]:
    if not groups:
        raise ValueError("Cannot split an empty log list.")
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")
    if ratio <= 0.0 and log_limit <= 0 and group_limit <= 0:
        raise ValueError("At least one of ratio/log_limit/group_limit must select held-out logs.")

    sorted_group_ids = sorted(groups.keys(), key=lambda item: (_stable_score(seed, item), item))
    if group_limit > 0:
        return sorted_group_ids[: min(group_limit, len(sorted_group_ids))]

    total_logs = sum(len(values) for values in groups.values())
    target_logs = int(round(total_logs * ratio)) if log_limit <= 0 else int(log_limit)
    target_logs = max(1, min(target_logs, total_logs - 1))

    selected: List[str] = []
    selected_count = 0
    for group_id in sorted_group_ids:
        if selected_count >= target_logs:
            break
        selected.append(group_id)
        selected_count += len(groups[group_id])
    return selected


def _split_logs(
    log_names: Sequence[str],
    *,
    ratio: float,
    seed: int,
    group_by: str,
    log_limit: int,
    group_limit: int,
) -> Dict[str, object]:
    unique_log_names = _stable_unique(log_names)
    groups = _group_log_names(unique_log_names, group_by=group_by)
    heldout_groups = _choose_heldout_groups(
        groups,
        ratio=ratio,
        seed=seed,
        log_limit=log_limit,
        group_limit=group_limit,
    )
    heldout_group_set = set(heldout_groups)

    keep_logs: List[str] = []
    heldout_logs: List[str] = []
    for log_name in unique_log_names:
        group_id = log_name if group_by == "log" else _infer_session_id(log_name)
        if group_id in heldout_group_set:
            heldout_logs.append(log_name)
        else:
            keep_logs.append(log_name)

    overlap = sorted(set(keep_logs).intersection(heldout_logs))
    if overlap:
        raise RuntimeError(f"Split overlap detected: {overlap[:5]}")
    if not keep_logs or not heldout_logs:
        raise RuntimeError(
            "Split failed: one side is empty "
            f"(keep={len(keep_logs)}, heldout={len(heldout_logs)}). "
            "Try smaller ratio/limit."
        )

    return {
        "keep_logs": keep_logs,
        "heldout_logs": heldout_logs,
        "heldout_group_ids": heldout_groups,
        "stats": {
            "total_log_count": len(unique_log_names),
            "total_group_count": len(groups),
            "keep_log_count": len(keep_logs),
            "heldout_log_count": len(heldout_logs),
            "keep_group_count": len(groups) - len(heldout_groups),
            "heldout_group_count": len(heldout_groups),
            "ratio_requested": float(ratio),
            "ratio_actual": float(len(heldout_logs) / len(unique_log_names)),
            "seed": int(seed),
            "group_by": group_by,
            "log_limit": int(log_limit),
            "group_limit": int(group_limit),
        },
    }


def _paths(output_dir: Path, prefix: str) -> Dict[str, Path]:
    return {
        "test": output_dir / f"{prefix}_test_logs.json",
        "cache": output_dir / f"{prefix}_cache_logs.json",
        "summary": output_dir / f"{prefix}_split_summary.json",
    }


def _load_source_logs(source_json: str, db_path: str) -> List[str]:
    if source_json:
        source_path = Path(source_json)
        if source_path.exists():
            return _read_json_list(source_path)
        if not db_path:
            raise FileNotFoundError(f"source_log_json does not exist: {source_path}")
    if not db_path:
        raise ValueError("Either --source_log_json or --db_path must be provided.")
    return _discover_log_names_from_db(Path(db_path))


def build_splits(args: argparse.Namespace) -> Dict[str, object]:
    source_logs = _load_source_logs(args.source_log_json, args.db_path)
    source_logs = _stable_unique(source_logs)
    if len(source_logs) < 2:
        raise ValueError(f"Need at least 2 logs to split, got {len(source_logs)}.")

    output_dir = Path(args.output_dir)
    path_dict = _paths(output_dir=output_dir, prefix=args.prefix)

    test_path = path_dict["test"]
    if args.reuse_existing_test and test_path.exists():
        test_logs = _stable_unique(_read_json_list(test_path))
        missing = sorted(set(test_logs) - set(source_logs))
        if missing:
            raise RuntimeError(
                f"Existing test file contains logs not in source list: {missing[:5]} "
                f"(total missing={len(missing)})"
            )
        cache_logs = [log_name for log_name in source_logs if log_name not in set(test_logs)]
        if not cache_logs:
            raise RuntimeError("Existing test set consumes all logs; no cache logs left.")
        test_stats = {
            "mode": "reuse_existing_test",
            "heldout_log_count": len(test_logs),
            "keep_log_count": len(cache_logs),
            "ratio_actual": float(len(test_logs) / len(source_logs)),
            "seed": None,
            "group_by": args.group_by,
        }
    else:
        test_split = _split_logs(
            source_logs,
            ratio=args.test_ratio,
            seed=args.test_seed,
            group_by=args.group_by,
            log_limit=args.test_log_limit,
            group_limit=args.test_group_limit,
        )
        test_logs = list(test_split["heldout_logs"])
        cache_logs = list(test_split["keep_logs"])
        test_stats = dict(test_split["stats"])  # type: ignore[arg-type]
        test_stats["mode"] = "fresh_sample"

    # 互斥性保护
    if set(cache_logs) & set(test_logs):
        raise RuntimeError("cache/test overlap detected.")

    _write_json(path_dict["test"], test_logs)
    _write_json(path_dict["cache"], cache_logs)

    summary: Dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prefix": args.prefix,
        "source_log_json": str(args.source_log_json),
        "db_path": str(args.db_path),
        "output_dir": str(output_dir.resolve()),
        "group_by": args.group_by,
        "reuse_existing_test": bool(args.reuse_existing_test),
        "test_split": test_stats,
        "files": {key: str(path.resolve()) for key, path in path_dict.items()},
        "counts": {
            "source_logs": len(source_logs),
            "test_logs": len(test_logs),
            "cache_logs": len(cache_logs),
        },
    }
    _write_json(path_dict["summary"], summary)
    return summary


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed test/cache log split JSONs.")
    parser.add_argument("--source_log_json", type=str, default=str(DEFAULT_SOURCE_LOG_JSON))
    parser.add_argument("--db_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", type=str, default="mini")
    parser.add_argument("--group_by", choices=["session", "log"], default="session")

    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--test_seed", type=int, default=3407)
    parser.add_argument("--test_log_limit", type=int, default=0)
    parser.add_argument("--test_group_limit", type=int, default=0)

    parser.add_argument(
        "--reuse_existing_test",
        type=int,
        default=1,
        help="1=如果 test 文件已存在则直接复用；0=每次按参数重采样 test。",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    summary = build_splits(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""从学长 raw_split / cache_split 的划分产物生成 comparison/splits.json。

原则：**不自行切分**，直接复用学长既有的 train/val/test 日志划分，保证四个模型与
学长管线训练/验证/留出集口径一致（train/val 来自 cache_split，test 来自 raw_split
的留出闭环日志），从根上消除训练集泄漏。

输入（三份 JSON 日志名列表，学长划分管线已产出，路径按本机调整）：
  - cache_train_log_names.json   -> train
  - cache_val_log_names.json     -> val
  - test_simu_log_names.json     -> test（留出闭环日志，仅用于 AD-MLP 开环验证）

用法：
  python comparison/make_splits.py \
      --data-root <boston_db_dir> --maps-root <maps_dir> \
      --train-logs <cache_train_log_names.json> \
      --val-logs   <cache_val_log_names.json> \
      --test-logs  <test_simu_log_names.json> \
      [--map-version nuplan-maps-v1.0] [--output comparison/splits.json]

不传 --train-logs/--val-logs/--test-logs 时，回退读 paths.local.json 的
cache_train_log_names_path / cache_val_log_names_path / test_simu_log_names_path。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (REPO_ROOT / "nuplan-devkit", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

from comparison._io import config_value


def _scenario_filter(log_names):
    # 与 process_data.py / extract.py 同款参数顺序（expand=True, remove_invalid_goals=False）。
    return ScenarioFilter(None, None, log_names, None, None, None, None, None,
                          True, False, False, None, None, None, None)


def _load_log_list(path, label):
    if not path:
        raise FileNotFoundError(
            f"[make_splits] 未提供 {label}（用 --train-logs/--val-logs/--test-logs 或 paths.local.json 对应键指定）"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[make_splits] {label} 不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"[make_splits] {label} 应为 JSON 字符串列表: {path}")
    return sorted({str(x) for x in payload})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="boston db 目录（splits/train_boston）")
    ap.add_argument("--maps-root", required=True, help="maps 目录")
    ap.add_argument("--map-version", default="nuplan-maps-v1.0", help="与 baseline/process_data.py 保持一致")
    ap.add_argument("--train-logs", default=None, help="学长 cache_train_log_names.json")
    ap.add_argument("--val-logs", default=None, help="学长 cache_val_log_names.json")
    ap.add_argument("--test-logs", default=None, help="学长 test_simu_log_names.json")
    ap.add_argument("--limit", type=int, default=None, help="debug：只加载前 N 个 scenario")
    ap.add_argument("--output", default=str(SCRIPT_DIR / "splits.json"))
    args = ap.parse_args()

    train_logs = _load_log_list(args.train_logs or config_value("cache_train_log_names_path"), "train-logs")
    val_logs = _load_log_list(args.val_logs or config_value("cache_val_log_names_path"), "val-logs")
    test_logs = _load_log_list(args.test_logs or config_value("test_simu_log_names_path"), "test-logs")

    # 三份名单必须两两不相交（学长划分管线已保证，这里兜底校验）。
    overlap = (set(train_logs) & set(val_logs)) | (set(train_logs) & set(test_logs)) | (set(val_logs) & set(test_logs))
    if overlap:
        raise RuntimeError(f"[make_splits] train/val/test 日志名单存在重叠: {sorted(overlap)[:5]}")

    union_logs = sorted(set(train_logs) | set(val_logs) | set(test_logs))
    builder = NuPlanScenarioBuilder(args.data_root, args.maps_root, None, None, args.map_version)
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = list(builder.get_scenarios(_scenario_filter(union_logs), worker))
    if args.limit:
        scenarios = scenarios[: args.limit]
    print(f"[make_splits] loaded {len(scenarios)} scenarios from {len(union_logs)} logs")

    train_set, val_set, test_set = set(train_logs), set(val_logs), set(test_logs)
    buckets = {"train_tokens": [], "val_tokens": [], "test_tokens": []}
    unmatched = 0
    for s in scenarios:
        log = str(s.log_name)
        if log in test_set:
            buckets["test_tokens"].append(str(s.token))
        elif log in val_set:
            buckets["val_tokens"].append(str(s.token))
        elif log in train_set:
            buckets["train_tokens"].append(str(s.token))
        else:
            unmatched += 1
    if unmatched:
        print(f"[make_splits] ⚠️ {unmatched} 个 scenario 不属于三份名单（已丢弃）")

    out = {
        "train_logs": train_logs,
        "val_logs": val_logs,
        "test_logs": test_logs,
        "train_tokens": buckets["train_tokens"],
        "val_tokens": buckets["val_tokens"],
        "test_tokens": buckets["test_tokens"],
        "counts": {
            "train_logs": len(train_logs),
            "val_logs": len(val_logs),
            "test_logs": len(test_logs),
            "train_tokens": len(buckets["train_tokens"]),
            "val_tokens": len(buckets["val_tokens"]),
            "test_tokens": len(buckets["test_tokens"]),
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[make_splits] saved -> {args.output}")
    print(f"[make_splits] counts: {out['counts']}")


if __name__ == "__main__":
    main()

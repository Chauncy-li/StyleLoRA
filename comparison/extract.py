"""统一抽取：一次读 scenario，产出三模型共用的统一 .npz。

按 split 读取 splits.json 里的 log 名单 -> 加载 scenario -> UnifiedExtractor 落盘到
<output_root>/<split>/<token>.npz。产物字段见 comparison/README.md 第 3.2 节。

用法：
  python comparison/extract.py --splits comparison/splits.json \
      --data-root <boston_db_dir> --maps-root <maps_dir> --output-root <cache_root> \
      --split train [--num-workers 8]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (REPO_ROOT / "nuplan-devkit", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

from baseline.data_process.unified_extractor import UnifiedExtractor


def _scenario_filter(log_names):
    return ScenarioFilter(None, None, log_names, None, None, None, None, None,
                          True, False, False, None, None, None, None)


def _build_config(args):
    return SimpleNamespace(
        save_path=None,
        agent_num=args.agent_num,
        static_objects_num=args.static_objects_num,
        lane_len=args.lane_len,
        lane_num=args.lane_num,
        route_len=args.route_len,
        route_num=args.route_num,
    )


def process_in_chunks(scenarios_chunk, args, process_id):
    out_dir = os.path.join(args.output_root, args.split)
    os.makedirs(out_dir, exist_ok=True)
    extractor = UnifiedExtractor(_build_config(args))
    n = 0
    for s in scenarios_chunk:
        out_path = os.path.join(out_dir, f"{s.token}.npz")
        if os.path.exists(out_path):
            continue
        try:
            data = extractor.extract_scenario(s)
            extractor.save(out_dir, data)
            n += 1
        except Exception as exc:
            print(f"[Process {process_id}] skip {s.token}: {exc}")
    print(f"[Process {process_id}] extracted {n} -> {out_dir}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(SCRIPT_DIR / "splits.json"))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--maps-root", required=True)
    ap.add_argument("--map-version", default="nuplan-maps-v1.0")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="debug：每个 split 只抽取前 N 个")
    # 特征维度（与 train.yaml / comparison/README.md 保持一致）
    ap.add_argument("--agent-num", type=int, default=32)
    ap.add_argument("--static-objects-num", type=int, default=5)
    ap.add_argument("--lane-len", type=int, default=20)
    ap.add_argument("--lane-num", type=int, default=70)
    ap.add_argument("--route-len", type=int, default=20)
    ap.add_argument("--route-num", type=int, default=25)
    args = ap.parse_args()

    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)
    log_names = splits.get(f"{args.split}_logs")
    if not log_names:
        print(f"[extract] splits.json 里没有 {args.split}_logs，退出")
        return
    if args.limit:
        log_names = log_names[: args.limit]

    builder = NuPlanScenarioBuilder(args.data_root, args.maps_root, None, None, args.map_version)
    worker = SingleMachineParallelExecutor(use_process_pool=False)
    scenarios = list(builder.get_scenarios(_scenario_filter(log_names), worker))
    print(f"[extract] {args.split}: {len(scenarios)} scenarios from {len(log_names)} logs")

    n_proc = max(1, args.num_workers)
    if n_proc > 1:
        chunks = [scenarios[i::n_proc] for i in range(n_proc)]
        pool_args = [(c, args, i) for i, c in enumerate(chunks)]
        with multiprocessing.Pool(n_proc) as pool:
            pool.starmap(process_in_chunks, pool_args)
    else:
        process_in_chunks(scenarios, args, 0)


if __name__ == "__main__":
    main()

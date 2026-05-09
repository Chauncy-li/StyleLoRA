"""
NuPlan 原始场景转训练缓存脚本。

功能说明：
1. 读取 NuPlan 场景并按过滤条件采样；
2. 调用 `DataProcessor` 将每个场景转为 `.npz` 训练样本；
3. 处理结束后导出 `.npz` 文件列表 JSON。

使用建议：
- 数据规模大时用 `--num_workers` 并行；
- 场景列表 JSON 可通过 `--scenario_log_json` 显式指定。
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESOURCES_DIR = SCRIPT_DIR / "resources"

NUPLAN_DEVKIT_PATH = REPO_ROOT / "nuplan-devkit"
if not NUPLAN_DEVKIT_PATH.is_dir():
    raise FileNotFoundError(f"nuplan-devkit not found at expected path: {NUPLAN_DEVKIT_PATH}")
if str(NUPLAN_DEVKIT_PATH) not in sys.path:
    sys.path.insert(0, str(NUPLAN_DEVKIT_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from baseline.data_process.data_processor import DataProcessor


from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor


def get_filter_parameters(
    num_scenarios_per_type=None,
    limit_total_scenarios=None,
    shuffle=True,
    scenario_tokens=None,
    log_names=None,
):
    """构造 NuPlan `ScenarioFilter` 参数元组。"""
    scenario_types = None
    map_names = None
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None
    expand_scenarios = True
    remove_invalid_goals = False
    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return (
        scenario_types,
        scenario_tokens,
        log_names,
        map_names,
        num_scenarios_per_type,
        limit_total_scenarios,
        timestamp_threshold_s,
        ego_displacement_minimum_m,
        expand_scenarios,
        remove_invalid_goals,
        shuffle,
        ego_start_speed_threshold,
        ego_stop_speed_threshold,
        speed_noise_tolerance,
    )


def process_in_chunks(scenarios_chunk, args, process_id):
    """子进程执行函数：每个进程单独实例化 DataProcessor。"""
    try:
        processor = DataProcessor(args)
        print(f"[Process {process_id}] Start processing {len(scenarios_chunk)} scenarios...")
        processor.work(scenarios_chunk)
        print(f"[Process {process_id}] Finished.")
        return True
    except Exception as exc:
        print(f"[Process {process_id}] Error: {exc}")
        return False


def _first_existing(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0] if candidates else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="NuPlan data processing")
    parser.add_argument(
        "--data_path",
        default="/media/lsw/Work/ubuntu_system/DATASET/nuplan-v1.1/splits/mini",
        type=str,
        help="Path to raw nuPlan DB data",
    )
    parser.add_argument(
        "--map_path",
        default="/media/lsw/Work/ubuntu_system/DATASET/maps",
        type=str,
        help="Path to map data",
    )
    parser.add_argument(
        "--save_path",
        default="/media/lsw/Other/Ubuntu_copy/CACHE/minicache",
        type=str,
        help="Path to save processed .npz data",
    )

    parser.add_argument("--scenarios_per_type", type=int, default=None, help="Number of scenarios per type")
    parser.add_argument("--total_scenarios", type=int, default=443218, help="Limit total number of scenarios")  # 一共是 443218
    parser.add_argument("--shuffle_scenarios", type=bool, default=True, help="Whether to shuffle scenarios")

    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--static_objects_num", type=int, default=5)
    parser.add_argument("--lane_len", type=int, default=20)
    parser.add_argument("--lane_num", type=int, default=70)
    parser.add_argument("--route_len", type=int, default=20)
    parser.add_argument("--route_num", type=int, default=25)

    parser.add_argument(
        "--scenario_log_json",
        type=str,
        default=_first_existing(
            [
                "/media/lsw/Work/ubuntu_system/DATASET/nuplan-v1.1/splits/nuplan_scenarios_mini.json",
                str(RESOURCES_DIR / "nuplan_scenarios_mini.json"),
                str(REPO_ROOT / "nuplan_scenarios_mini.json"),
            ]
        ),
        help="JSON file containing selected scenario log names",
    )
    parser.add_argument(
        "--npz_list_output_json",
        type=str,
        default="/media/lsw/Work/ubuntu_system/CACHE/minicache_list.json",
        help="Output JSON file storing generated .npz filename list",
    )

    default_workers = max(1, multiprocessing.cpu_count() // 2)
    parser.add_argument("--num_workers", type=int, default=default_workers, help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    with open(args.scenario_log_json, "r", encoding="utf-8") as file_obj:
        log_names = json.load(file_obj)

    sensor_root = None
    db_files = None
    map_version = "nuplan-maps-v1.0"

    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            args.scenarios_per_type,
            args.total_scenarios,
            args.shuffle_scenarios,
            log_names=log_names,
        )
    )

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    del worker, builder, scenario_filter

    if args.num_workers > 1:
        print(f"Starting parallel processing with {args.num_workers} workers...")
        total_items = len(scenarios)
        chunk_size = math.ceil(total_items / args.num_workers)
        chunks = [scenarios[i : i + chunk_size] for i in range(0, total_items, chunk_size)]
        print(f"Split data into {len(chunks)} chunks, approx {chunk_size} scenarios per worker.")

        pool_args = [(chunk, args, i) for i, chunk in enumerate(chunks)]
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            pool.starmap(process_in_chunks, pool_args)
    else:
        print("Using single process execution...")
        processor = DataProcessor(args)
        processor.work(scenarios)

    print("All processing finished. Generating file list...")
    npz_files = [file_name for file_name in os.listdir(args.save_path) if file_name.endswith(".npz")]

    output_dir = os.path.dirname(args.npz_list_output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.npz_list_output_json, "w", encoding="utf-8") as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"Saved {len(npz_files)} .npz file names")


if __name__ == "__main__":
    main()

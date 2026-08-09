"""Build planner NPZ cache from the cache partition of a raw DB split.

This is a research-local replacement for running the root
data_process.py directly on all Boston logs.  It consumes cache_log_names.json
from stylelora.pipeline.data_pipeline.raw_split, so held-out simulation logs never
enter the training cache.

When --train_log_names_path and --val_log_names_path are provided, the output is
still one cache directory plus one JSON list.  The list is written in
train-first, val-second order so existing train/val slicing configs continue to
work unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from stylelora.pipeline.paths import (
    DEFAULT_DATA_SPLITS_ROOT,
    DEFAULT_NUPLAN_DATA_PATH,
    DEFAULT_NUPLAN_MAP_PATH,
    DEFAULT_NUM_WORKERS,
    DEFAULT_RECORD_ROOT,
    ensure_repo_on_path,
)
from stylelora.pipeline.data_pipeline.raw_split import load_json_list

ensure_repo_on_path()


# ==============================================================================
# User configuration
# ==============================================================================
# Edit this block when running on a new machine.  Every value can still be
# overridden from CLI, but normal use should only require changing these paths.
DATA_PATH = DEFAULT_NUPLAN_DATA_PATH
MAP_PATH = DEFAULT_NUPLAN_MAP_PATH

DATA_ROOT_PATH = Path(DEFAULT_RECORD_ROOT)

SPLIT_ROOT = Path(DEFAULT_DATA_SPLITS_ROOT)
LOG_NAMES_PATH = ""
TRAIN_LOG_NAMES_PATH = SPLIT_ROOT / "cache_train_log_names.json"
VAL_LOG_NAMES_PATH = SPLIT_ROOT / "cache_val_log_names.json"
SCENARIO_TOKENS_PATH = ""

SAVE_PATH = DATA_ROOT_PATH / "CACHE" / "boston_cache_train_val"
OUTPUT_LIST_PATH = DATA_ROOT_PATH / "CACHE" / "boston_cache_train_val_list.json"
MANIFEST_PATH = DATA_ROOT_PATH / "CACHE" / "boston_cache_train_val_manifest.json"

MAP_VERSION = "nuplan-maps-v1.0"
SCENARIOS_PER_TYPE = None
TOTAL_SCENARIOS = -1  # -1 means use all scenarios in the selected log split.
SHUFFLE_SCENARIOS = False
DRY_RUN = False

AGENT_NUM = 32
STATIC_OBJECTS_NUM = 5
LANE_LEN = 20
LANE_NUM = 70
ROUTE_LEN = 20
ROUTE_NUM = 25
NUM_WORKERS = DEFAULT_NUM_WORKERS


def str_to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def load_optional_json_list(path: str) -> Optional[List[str]]:
    if not path:
        return None
    return load_json_list(path)


def get_filter_parameters(
    *,
    num_scenarios_per_type: Optional[int],
    limit_total_scenarios: Optional[int],
    shuffle: bool,
    scenario_tokens: Optional[Sequence[str]],
    log_names: Optional[Sequence[str]],
):
    """Mirror data_process.py defaults while allowing a raw-log split."""

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


def sort_scenarios(scenarios: Sequence[object]) -> List[object]:
    """Make processing order deterministic across worker implementations."""

    return sorted(
        scenarios,
        key=lambda item: (
            str(getattr(item, "log_name", "")),
            int(getattr(item, "_initial_lidar_timestamp", 0) or 0),
            str(getattr(item, "token", "")),
        ),
    )


def build_scenarios_for_log_names(args: argparse.Namespace, log_names: Optional[Sequence[str]]) -> List[object]:
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

    scenario_tokens = load_optional_json_list(args.scenario_tokens_path)
    limit_total_scenarios = None if args.total_scenarios < 0 else args.total_scenarios

    builder = NuPlanScenarioBuilder(
        args.data_path,
        args.map_path,
        sensor_root=None,
        db_files=None,
        map_version=args.map_version,
        scenario_mapping=ScenarioMapping({}, None),
    )
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            num_scenarios_per_type=args.scenarios_per_type,
            limit_total_scenarios=limit_total_scenarios,
            shuffle=args.shuffle_scenarios,
            scenario_tokens=scenario_tokens,
            log_names=log_names,
        )
    )
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    del worker, builder, scenario_filter
    return sort_scenarios(list(scenarios))


def build_scenarios(args: argparse.Namespace) -> Tuple[List[object], int, int]:
    """Build scenarios, optionally preserving train-first/val-second layout."""

    if args.train_log_names_path or args.val_log_names_path:
        if not args.train_log_names_path or not args.val_log_names_path:
            raise ValueError("--train_log_names_path and --val_log_names_path must be provided together")
        train_logs = load_json_list(args.train_log_names_path)
        val_logs = load_json_list(args.val_log_names_path)
        overlap = sorted(set(train_logs).intersection(val_logs))
        if overlap:
            raise ValueError(f"train/val log lists overlap, e.g. {overlap[:5]}")
        train_scenarios = build_scenarios_for_log_names(args, train_logs)
        val_scenarios = build_scenarios_for_log_names(args, val_logs)
        return train_scenarios + val_scenarios, len(train_scenarios), len(val_scenarios)

    log_names = load_optional_json_list(args.log_names_path)
    scenarios = build_scenarios_for_log_names(args, log_names)
    return scenarios, len(scenarios), 0


def process_in_chunks(scenarios_chunk: Sequence[object], args: argparse.Namespace, process_id: int) -> bool:
    try:
        from baseline.data_process.data_processor import DataProcessor

        processor = DataProcessor(args)
        print(f"[CacheSplitProcess {process_id}] start scenarios={len(scenarios_chunk)}")
        processor.work(scenarios_chunk)
        print(f"[CacheSplitProcess {process_id}] finished")
        return True
    except Exception as exc:
        print(f"[CacheSplitProcess {process_id}] error: {exc}")
        return False


def scenario_cache_filename(scenario: object) -> str:
    return f"{getattr(scenario, '_map_name')}_{scenario.token}.npz"


def write_cache_list(
    save_path: str,
    output_list_path: str,
    ordered_scenarios: Optional[Sequence[object]] = None,
) -> List[str]:
    if ordered_scenarios is None:
        npz_files = sorted(name for name in os.listdir(save_path) if name.endswith(".npz"))
    else:
        npz_files = [scenario_cache_filename(scenario) for scenario in ordered_scenarios]
        if len(npz_files) != len(set(npz_files)):
            raise RuntimeError("Duplicate cache filenames detected while writing ordered cache list")
        missing = [name for name in npz_files if not os.path.exists(os.path.join(save_path, name))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} expected cache files are missing under {save_path}; first missing: {missing[:5]}"
            )
    output_path = Path(output_list_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(npz_files, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    os.replace(tmp_path, output_path)
    return npz_files


def write_manifest(
    args: argparse.Namespace,
    scenario_count: int,
    cache_count: int,
    train_num_samples: int,
    val_num_samples: int,
) -> None:
    if not args.manifest_path:
        return
    manifest = {
        "schema_version": 1,
        "data_path": args.data_path,
        "map_path": args.map_path,
        "log_names_path": args.log_names_path,
        "train_log_names_path": args.train_log_names_path,
        "val_log_names_path": args.val_log_names_path,
        "scenario_tokens_path": args.scenario_tokens_path,
        "save_path": args.save_path,
        "output_list_path": args.output_list_path,
        "scenario_count": int(scenario_count),
        "cache_file_count": int(cache_count),
        "train_start_index": 0,
        "train_num_samples": int(train_num_samples),
        "val_start_index": int(train_num_samples),
        "val_num_samples": int(val_num_samples),
        "map_version": args.map_version,
        "shuffle_scenarios": bool(args.shuffle_scenarios),
        "total_scenarios": int(args.total_scenarios),
        "scenarios_per_type": args.scenarios_per_type,
    }
    path = Path(args.manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    os.replace(tmp_path, path)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process only the cache side of a raw DB split")
    parser.add_argument("--data_path", type=str, default=str(DATA_PATH))
    parser.add_argument("--map_path", type=str, default=str(MAP_PATH))
    parser.add_argument("--log_names_path", type=str, default=str(LOG_NAMES_PATH))
    parser.add_argument("--train_log_names_path", type=str, default=str(TRAIN_LOG_NAMES_PATH))
    parser.add_argument("--val_log_names_path", type=str, default=str(VAL_LOG_NAMES_PATH))
    parser.add_argument("--scenario_tokens_path", type=str, default=str(SCENARIO_TOKENS_PATH))
    parser.add_argument("--save_path", type=str, default=str(SAVE_PATH))
    parser.add_argument("--output_list_path", type=str, default=str(OUTPUT_LIST_PATH))
    parser.add_argument("--manifest_path", type=str, default=str(MANIFEST_PATH))
    parser.add_argument("--map_version", type=str, default=str(MAP_VERSION))

    parser.add_argument("--scenarios_per_type", type=int, default=SCENARIOS_PER_TYPE)
    parser.add_argument(
        "--total_scenarios",
        type=int,
        default=TOTAL_SCENARIOS,
        help="-1 means use all scenarios in the cache split.",
    )
    parser.add_argument("--shuffle_scenarios", type=str_to_bool, default=SHUFFLE_SCENARIOS)
    parser.add_argument("--dry_run", type=str_to_bool, default=DRY_RUN)

    parser.add_argument("--agent_num", type=int, default=AGENT_NUM)
    parser.add_argument("--static_objects_num", type=int, default=STATIC_OBJECTS_NUM)
    parser.add_argument("--lane_len", type=int, default=LANE_LEN)
    parser.add_argument("--lane_num", type=int, default=LANE_NUM)
    parser.add_argument("--route_len", type=int, default=ROUTE_LEN)
    parser.add_argument("--route_num", type=int, default=ROUTE_NUM)

    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)
    if not args.manifest_path:
        args.manifest_path = str(Path(args.output_list_path).with_name("boston_cache_manifest.json"))

    scenarios, train_num_samples, val_num_samples = build_scenarios(args)
    print(f"[CacheSplitProcess] total_scenarios={len(scenarios)}")
    print(f"[CacheSplitProcess] log_names_path={args.log_names_path}")
    if args.train_log_names_path or args.val_log_names_path:
        print(f"[CacheSplitProcess] train_log_names_path={args.train_log_names_path}")
        print(f"[CacheSplitProcess] val_log_names_path={args.val_log_names_path}")
        print(
            "[CacheSplitProcess] planned_slice="
            f"train_start=0 train_num={train_num_samples} "
            f"val_start={train_num_samples} val_num={val_num_samples}"
        )
    print(f"[CacheSplitProcess] save_path={args.save_path}")

    if args.dry_run:
        print("[CacheSplitProcess] dry_run=true, skip DataProcessor.work")
        write_manifest(
            args,
            scenario_count=len(scenarios),
            cache_count=0,
            train_num_samples=train_num_samples,
            val_num_samples=val_num_samples,
        )
        return

    if args.num_workers > 1:
        chunk_size = math.ceil(len(scenarios) / args.num_workers)
        chunks = [scenarios[idx : idx + chunk_size] for idx in range(0, len(scenarios), chunk_size)]
        pool_args: List[Tuple[Sequence[object], argparse.Namespace, int]] = [
            (chunk, args, idx) for idx, chunk in enumerate(chunks)
        ]
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            results = pool.starmap(process_in_chunks, pool_args)
        if not all(results):
            raise RuntimeError("At least one cache-processing worker failed")
    else:
        process_in_chunks(scenarios, args, 0)

    npz_files = write_cache_list(args.save_path, args.output_list_path, ordered_scenarios=scenarios)
    write_manifest(
        args,
        scenario_count=len(scenarios),
        cache_count=len(npz_files),
        train_num_samples=train_num_samples,
        val_num_samples=val_num_samples,
    )
    print(f"[CacheSplitProcess] wrote_cache_list={args.output_list_path}")
    print(f"[CacheSplitProcess] cache_file_count={len(npz_files)}")


if __name__ == "__main__":
    main()



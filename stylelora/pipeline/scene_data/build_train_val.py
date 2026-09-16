"""Build v2 straight-scene splits from train/val or held-out test caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _repo_root_str = str(_REPO_ROOT)
    if _repo_root_str not in sys.path:
        sys.path.insert(0, _repo_root_str)

from stylelora.pipeline.paths import (
    DEFAULT_CACHE_TRAIN_VAL_DIR,
    DEFAULT_CACHE_TRAIN_VAL_LIST_PATH,
    DEFAULT_CACHE_TRAIN_VAL_MANIFEST_PATH,
    DEFAULT_NUM_WORKERS,
)
from stylelora.pipeline.scene_data.paths import (
    DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR,
    DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR,
    PRIMARY_SCENE_BUCKETS,
)
from stylelora.pipeline.scene_data.builder import StyleSceneSplitBuilderV2
from stylelora.pipeline.scene_data.index import load_split_index

SCENE_BUCKETS: Sequence[str] = PRIMARY_SCENE_BUCKETS

DEFAULT_TRAIN_OUTPUT_DIR = Path(DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR)
DEFAULT_VAL_OUTPUT_DIR = Path(DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR)
DEFAULT_TEST_OUTPUT_DIR = DEFAULT_VAL_OUTPUT_DIR.parent / "style_scene_split_straight_test_simu_v2"


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json(path: os.PathLike[str] | str, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    os.replace(tmp_path, path)


def _materialize_scene_lists(output_dir: str) -> Dict[str, object]:
    index_path = Path(output_dir) / "split_index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"split_index.jsonl not found under {output_dir}")

    records = load_split_index(str(index_path), normalize=True)
    scene_dir = Path(output_dir) / "scene_lists"
    memory_scene_dir = Path(output_dir) / "memory_scene_lists"
    scene_dir.mkdir(parents=True, exist_ok=True)

    valid_by_scene: Dict[str, List[str]] = {scene: [] for scene in SCENE_BUCKETS}
    memory_by_scene: Dict[str, List[str]] = {scene: [] for scene in SCENE_BUCKETS}

    for record in records:
        scene_bucket = str(record.get("scene_bucket", "none"))
        if scene_bucket not in valid_by_scene:
            continue
        filename = str(record.get("filename", ""))
        if not filename:
            continue
        if bool(record.get("split_valid", False)):
            valid_by_scene[scene_bucket].append(filename)
        if bool(record.get("memory_eligible", False)):
            memory_by_scene[scene_bucket].append(filename)

    for scene_bucket, filenames in valid_by_scene.items():
        _write_json(scene_dir / f"{scene_bucket}.json", sorted(filenames))

    has_memory_lists = any(filenames for filenames in memory_by_scene.values())
    if has_memory_lists:
        memory_scene_dir.mkdir(parents=True, exist_ok=True)
        for scene_bucket, filenames in memory_by_scene.items():
            _write_json(memory_scene_dir / f"{scene_bucket}.json", sorted(filenames))

    summary = {
        "index_path": str(index_path),
        "scene_list_dir": str(scene_dir),
        "memory_scene_list_dir": str(memory_scene_dir) if has_memory_lists else "",
        "valid_scene_counts": {scene: len(filenames) for scene, filenames in valid_by_scene.items()},
        "memory_scene_counts": {scene: len(filenames) for scene, filenames in memory_by_scene.items()},
    }
    _write_json(Path(output_dir) / "scene_list_summary.json", summary)
    return summary


def _run_partition(
    *,
    partition_name: str,
    planner_cache_dir: str,
    data_list_path: str,
    output_dir: str,
    start_index: int,
    end_index: int,
    time_delta: float,
    skip_existing: bool,
    log_interval: int,
    num_workers: int,
) -> Dict[str, object]:
    builder = StyleSceneSplitBuilderV2(
        planner_cache_dir=planner_cache_dir,
        output_dir=output_dir,
        time_delta=time_delta,
        skip_existing=skip_existing,
        log_interval=log_interval,
        num_workers=num_workers,
    )
    summary = builder.process_from_json_list(
        data_list_path=data_list_path,
        start_index=start_index,
        end_index=end_index,
    )
    scene_summary = _materialize_scene_lists(output_dir)
    merged_summary = dict(summary)
    merged_summary["partition_name"] = partition_name
    merged_summary["partition_start_index"] = int(start_index)
    merged_summary["partition_end_index"] = int(end_index)
    merged_summary["scene_list_summary"] = scene_summary
    _write_json(Path(output_dir) / "partition_summary.json", merged_summary)
    return merged_summary


def _partition_ranges(manifest: Mapping[str, object]) -> Dict[str, tuple[int, int]]:
    train_start = int(manifest.get("train_start_index", 0))
    train_num = int(manifest.get("train_num_samples", 0))
    val_start = int(manifest.get("val_start_index", train_start + train_num))
    val_num = int(manifest.get("val_num_samples", 0))
    # test_simu cache generation is a single-partition job.  Its historical
    # manifest stores the held-out test population in train_* fields and sets
    # val_num_samples=0; accept explicit test_* fields as well if introduced.
    test_start = int(manifest.get("test_start_index", train_start))
    test_num = int(manifest.get("test_num_samples", manifest.get("scenario_count", train_num)))
    return {
        "train": (train_start, train_start + train_num),
        "val": (val_start, val_start + val_num),
        "test": (test_start, test_start + test_num),
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build straight-scene splits from train/val or held-out test planner caches"
    )
    parser.add_argument("--planner_cache_dir", type=str, default=str(DEFAULT_CACHE_TRAIN_VAL_DIR))
    parser.add_argument("--data_list_path", type=str, default=str(DEFAULT_CACHE_TRAIN_VAL_LIST_PATH))
    parser.add_argument("--manifest_path", type=str, default=str(DEFAULT_CACHE_TRAIN_VAL_MANIFEST_PATH))
    parser.add_argument("--split_name", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--train_output_dir", type=str, default=str(DEFAULT_TRAIN_OUTPUT_DIR))
    parser.add_argument("--val_output_dir", type=str, default=str(DEFAULT_VAL_OUTPUT_DIR))
    parser.add_argument("--test_output_dir", type=str, default=str(DEFAULT_TEST_OUTPUT_DIR))
    parser.add_argument("--time_delta", type=float, default=0.1)
    parser.add_argument("--skip_existing", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    manifest = _load_json(args.manifest_path)
    partition_ranges = _partition_ranges(manifest)

    run_plan = []
    if args.split_name in {"train", "all"}:
        run_plan.append(("train", args.train_output_dir))
    if args.split_name in {"val", "all"}:
        run_plan.append(("val", args.val_output_dir))
    if args.split_name == "test":
        run_plan.append(("test", args.test_output_dir))

    summaries: Dict[str, object] = {
        "planner_cache_dir": args.planner_cache_dir,
        "data_list_path": args.data_list_path,
        "manifest_path": args.manifest_path,
        "splits": {},
    }

    for split_name, output_dir in run_plan:
        start_index, end_index = partition_ranges[split_name]
        print(
            f"[TrainValStyleSceneSplitV2] split={split_name} "
            f"start_index={start_index} end_index={end_index} output_dir={output_dir} "
            f"num_workers={args.num_workers}"
        )
        summaries["splits"][split_name] = _run_partition(
            partition_name=split_name,
            planner_cache_dir=args.planner_cache_dir,
            data_list_path=args.data_list_path,
            output_dir=output_dir,
            start_index=start_index,
            end_index=end_index,
            time_delta=args.time_delta,
            skip_existing=bool(args.skip_existing),
            log_interval=args.log_interval,
            num_workers=args.num_workers,
        )

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

